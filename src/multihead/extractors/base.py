"""Base interface for LLM-powered extractors."""

from __future__ import annotations

import abc
import asyncio
import json
import logging
import re
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from multihead.adapters.base import HeadAdapter
from multihead.chunker import Chunk

logger = logging.getLogger(__name__)

# A generate function: (prompt) -> dict with "text", "tokens_in", etc.
GenerateFn = Callable[[str], Coroutine[Any, Any, dict[str, Any]]]


class BatchPending(Exception):
    """Raised when a batch was submitted but results aren't ready yet.

    Not an error — signals the pipeline to exit cleanly and retry next run.
    The batch will be recovered via the stalled-batch path.
    """

    def __init__(self, message: str, *, batch_id: str = "", stage_name: str = ""):
        super().__init__(message)
        self.batch_id = batch_id
        self.stage_name = stage_name


class BatchPartialError(RuntimeError):
    """Raised when a batch completes partially but results can't be downloaded.

    Carries completed_count so callers can skip already-processed prompts
    in sequential fallback.
    """

    def __init__(self, message: str, *, completed_count: int = 0, total_count: int = 0):
        super().__init__(message)
        self.completed_count = completed_count
        self.total_count = total_count

# Module-level shutdown flag — set by signal handlers in pipeline.py
shutdown_requested = False


class ExtractorResult(BaseModel):
    """Result from an extraction pass."""
    items: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class BaseExtractor(abc.ABC):
    """Interface for LLM-powered extractors used by Night Shift."""

    @abc.abstractmethod
    async def extract(
        self, chunks: list[Chunk], adapter: HeadAdapter | GenerateFn, **kwargs: Any
    ) -> ExtractorResult:
        """Run extraction on chunks using the given LLM adapter or generate function."""

    @staticmethod
    async def call_generate(adapter: HeadAdapter | GenerateFn, prompt: str) -> dict[str, Any]:
        """Call generate on either a HeadAdapter or a bare generate function."""
        if callable(adapter) and not isinstance(adapter, HeadAdapter):
            return await adapter(prompt)
        return await adapter.generate(prompt)

    @staticmethod
    async def map_generate(
        adapter: HeadAdapter | GenerateFn,
        prompts: list[str],
        concurrency: int = 1,
        checkpoint_dir: Path | None = None,
        stage_name: str = "",
        batch_mode: bool = False,
        no_wait: bool = False,
        on_chunk_progress: Callable[[int, int], None] | None = None,
    ) -> list[dict[str, Any] | Exception]:
        """Call generate for each prompt, optionally in parallel or batch.

        Returns a list matching the input order. Each element is either
        a response dict or an Exception if that call failed.

        Modes:
        - concurrency=1: Sequential with per-call checkpointing (default).
        - concurrency>1: Parallel with semaphore.
        - batch_mode=True: Submit all prompts as a single batch via
          adapter.generate_batch() (Anthropic Batch API — 50% cheaper).
          Falls back to parallel if adapter doesn't support batch.

        If checkpoint_dir and stage_name are provided, saves progress after
        each call so the stage can resume from where it left off.
        """
        from multihead.night_shift.checkpoint import (
            StageCheckpoint, clear_checkpoint, load_checkpoint, save_checkpoint,
        )

        # Load existing checkpoint (resume support)
        skip_count = 0
        prior_results: list[dict[str, Any] | None] = []
        if checkpoint_dir and stage_name:
            ckpt = load_checkpoint(checkpoint_dir, stage_name)
            if ckpt and ckpt.total_count == len(prompts) and ckpt.processed_count > 0:
                skip_count = ckpt.processed_count
                prior_results = ckpt.partial_results[:skip_count]
                logger.info(
                    "Resuming %s from checkpoint: %d/%d already done",
                    stage_name, skip_count, len(prompts),
                )

        # Batch mode: submit all remaining prompts as one batch
        if batch_mode and hasattr(adapter, "generate_batch"):
            remaining_prompts = prompts[skip_count:]
            if remaining_prompts:
                logger.info(
                    "Batch mode: submitting %d prompts to %s (skipped %d from checkpoint)",
                    len(remaining_prompts), stage_name or "batch", skip_count,
                )

                def _on_progress(completed: int, total: int) -> None:
                    logger.info("Batch %s: %d/%d complete", stage_name or "batch", completed, total)

                try:
                    batch_kwargs: dict[str, Any] = {"on_progress": _on_progress}
                    if stage_name:
                        batch_kwargs["stage_name"] = stage_name
                    if no_wait:
                        batch_kwargs["no_wait"] = True
                    batch_results = await adapter.generate_batch(
                        remaining_prompts, **batch_kwargs,
                    )
                except BatchPending:
                    raise  # Let pipeline handle — exit cleanly, retry next run
                except BatchPartialError as e:
                    # Batch completed on OpenAI but output file not ready.
                    # Don't fall back to sequential (wastes $$$).
                    # Exit cleanly — next run will recover the batch.
                    logger.warning(
                        "Batch partial: %s (%d/%d completed, no output file) — "
                        "exiting cleanly, will recover on next run",
                        e, e.completed_count, e.total_count,
                    )
                    raise BatchPending(
                        f"Batch completed but output not ready ({e.completed_count}/{e.total_count})",
                        batch_id="",
                        stage_name=stage_name,
                    )
                except Exception as e:
                    logger.error("Batch failed: %s — falling back to sequential", e)
                    batch_results = None

                if batch_results is not None:
                    # Check for missing/error entries and re-run them sequentially
                    missing_indices = [
                        i for i, r in enumerate(batch_results)
                        if isinstance(r, dict) and r.get("error") == "missing"
                    ]
                    if missing_indices:
                        logger.info(
                            "Batch %s: %d/%d results missing — filling gaps sequentially",
                            stage_name or "batch", len(missing_indices), len(batch_results),
                        )
                        for idx in missing_indices:
                            try:
                                result = await BaseExtractor.call_generate(
                                    adapter, remaining_prompts[idx],
                                )
                                batch_results[idx] = result
                            except Exception as fill_err:
                                batch_results[idx] = {"text": "", "error": str(fill_err)}

                    all_results = list(prior_results) + batch_results  # type: ignore[arg-type]
                    if checkpoint_dir and stage_name:
                        clear_checkpoint(checkpoint_dir, stage_name)
                    return all_results

            else:
                # All prompts were already in checkpoint
                if checkpoint_dir and stage_name:
                    clear_checkpoint(checkpoint_dir, stage_name)
                return list(prior_results)  # type: ignore[return-value]

        if concurrency <= 1:
            # Sequential with checkpoint support
            results: list[dict[str, Any] | Exception] = list(prior_results)  # type: ignore[arg-type]
            ckpt_obj = StageCheckpoint(
                stage_name=stage_name,
                total_count=len(prompts),
                processed_count=skip_count,
                partial_results=list(prior_results),
            ) if checkpoint_dir and stage_name else None

            for idx, prompt in enumerate(prompts):
                if idx < skip_count:
                    continue
                if shutdown_requested:
                    logger.info("Shutdown requested at %d/%d in %s — saving checkpoint",
                                idx, len(prompts), stage_name or "map_generate")
                    if ckpt_obj and checkpoint_dir:
                        save_checkpoint(checkpoint_dir, ckpt_obj)
                    raise InterruptedError(
                        f"Shutdown at {idx}/{len(prompts)} in {stage_name or 'map_generate'}"
                    )
                if on_chunk_progress:
                    on_chunk_progress(idx, len(prompts))
                try:
                    result = await BaseExtractor.call_generate(adapter, prompt)
                    results.append(result)
                except Exception as e:
                    results.append(e)
                    result = None  # type: ignore[assignment]

                # Save checkpoint after each call
                if ckpt_obj and checkpoint_dir:
                    ckpt_obj.processed_count = idx + 1
                    ckpt_obj.partial_results.append(
                        result if not isinstance(result, Exception) else None
                    )
                    save_checkpoint(checkpoint_dir, ckpt_obj)

            # Stage completed — clear checkpoint
            if checkpoint_dir and stage_name:
                clear_checkpoint(checkpoint_dir, stage_name)
            return results

        # Parallel with semaphore (no per-call checkpointing — batch only)
        sem = asyncio.Semaphore(concurrency)

        async def _call(prompt: str, sem_: asyncio.Semaphore) -> dict[str, Any]:
            async with sem_:
                return await BaseExtractor.call_generate(adapter, prompt)

        async def _safe_call(prompt: str) -> dict[str, Any] | Exception:
            try:
                return await _call(prompt, sem)
            except Exception as e:
                return e

        remaining_prompts = prompts[skip_count:]
        new_results = await asyncio.gather(*[_safe_call(p) for p in remaining_prompts])
        all_results = list(prior_results) + list(new_results)  # type: ignore[arg-type]

        if checkpoint_dir and stage_name:
            clear_checkpoint(checkpoint_dir, stage_name)
        return all_results

    @staticmethod
    def _filter_dicts(items: list) -> list[dict[str, Any]]:
        """Keep only dict items from a parsed JSON list.

        LLMs sometimes return string arrays instead of object arrays
        which causes 'str' object does not support item assignment
        when callers try to add fields.
        """
        return [item for item in items if isinstance(item, dict)]

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Strip <think>...</think> blocks from Qwen3-style reasoning output."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    @staticmethod
    def _repair_json(text: str) -> str:
        """Attempt to fix common JSON issues from small LLMs."""
        # Remove trailing commas before ] or }
        text = re.sub(r",\s*([}\]])", r"\1", text)
        # Try to close truncated arrays/objects
        open_brackets = text.count("[") - text.count("]")
        open_braces = text.count("{") - text.count("}")
        # If truncated mid-string, try to close it
        if open_braces > 0 or open_brackets > 0:
            # Remove last incomplete element (likely truncated)
            # Find last complete object/value boundary
            last_complete = max(text.rfind("},"), text.rfind("}]"))
            if last_complete > 0:
                text = text[: last_complete + 1]
                # Recount after trim
                open_brackets = text.count("[") - text.count("]")
                open_braces = text.count("{") - text.count("}")
            text += "}" * max(open_braces, 0)
            text += "]" * max(open_brackets, 0)
        return text

    @staticmethod
    def parse_json_response(text: str) -> list[dict[str, Any]]:
        """Robust JSON extraction from LLM output."""
        # Strip thinking blocks (Qwen3 /think mode)
        text = BaseExtractor._strip_thinking(text)

        # Try direct parse
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return BaseExtractor._filter_dicts(data)
            if isinstance(data, dict):
                return [data]
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code blocks
        code_blocks = re.findall(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        for block in code_blocks:
            try:
                data = json.loads(block.strip())
                if isinstance(data, list):
                    return BaseExtractor._filter_dicts(data)
                if isinstance(data, dict):
                    return [data]
            except json.JSONDecodeError:
                continue

        # Try finding JSON arrays in the text (greedy to capture full arrays)
        array_matches = re.findall(r"\[[\s\S]*\]", text)
        for match in array_matches:
            try:
                data = json.loads(match)
                if isinstance(data, list):
                    return BaseExtractor._filter_dicts(data)
            except json.JSONDecodeError:
                continue

        # Try finding JSON objects (greedy)
        obj_matches = re.findall(r"\{[\s\S]*\}", text)
        for match in obj_matches:
            try:
                data = json.loads(match)
                if isinstance(data, dict):
                    return [data]
            except json.JSONDecodeError:
                continue

        # Last resort: repair truncated/malformed JSON
        repaired = BaseExtractor._repair_json(text)
        try:
            data = json.loads(repaired)
            if isinstance(data, list):
                return BaseExtractor._filter_dicts(data)
            if isinstance(data, dict):
                return [data]
        except json.JSONDecodeError:
            pass

        # Try repair on code blocks and array matches too
        for candidate in code_blocks + array_matches:
            repaired = BaseExtractor._repair_json(candidate.strip() if isinstance(candidate, str) else candidate)
            try:
                data = json.loads(repaired)
                if isinstance(data, list):
                    return BaseExtractor._filter_dicts(data)
                if isinstance(data, dict):
                    return [data]
            except json.JSONDecodeError:
                continue

        return []
