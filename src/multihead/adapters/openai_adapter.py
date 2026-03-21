"""OpenAI-compatible API adapter for remote LLM inference.

Supports any OpenAI-compatible API (OpenAI, LM Studio, Together, etc.)
by configuring the endpoint URL. Uses httpx for async HTTP calls.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..extractors.base import BatchPartialError, BatchPending
from ..models import HeadManifest
from .base import HeadAdapter

logger = logging.getLogger(__name__)

# Timeouts
_GENERATE_TIMEOUT = 120.0  # Long responses
_HEALTHCHECK_TIMEOUT = 10.0


class OpenAIAdapter(HeadAdapter):
    """Adapter for OpenAI-compatible chat completion APIs.

    No GPU resources needed — pure HTTP client.
    API key resolved from manifest.extra["api_key"] or OPENAI_API_KEY env var.
    """

    def __init__(self, manifest: HeadManifest) -> None:
        super().__init__(manifest)
        self.model_name = manifest.model
        self.base_url = (manifest.endpoint or "https://api.openai.com/v1").rstrip("/")
        self._api_key: str | None = manifest.extra.get("api_key") or os.environ.get("OPENAI_API_KEY")
        self._loaded = False

    async def load(self) -> None:
        """Validate API key exists. No GPU resources to acquire."""
        if not self._api_key:
            raise RuntimeError(
                "OpenAI API key not configured. "
                "Set 'api_key' in head manifest extra or OPENAI_API_KEY env var."
            )
        # Resolve env var references like ${OPENAI_API_KEY}
        if self._api_key.startswith("${") and self._api_key.endswith("}"):
            env_name = self._api_key[2:-1]
            self._api_key = os.environ.get(env_name)
            if not self._api_key:
                raise RuntimeError(f"Environment variable {env_name} not set")

        logger.info("OpenAI adapter ready: %s via %s", self.model_name, self.base_url)
        self._loaded = True

    async def unload(self) -> None:
        """No-op — stateless HTTP client."""
        self._loaded = False
        logger.info("OpenAI adapter unloaded: %s", self.model_name)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def generate(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Generate via /chat/completions with a single user message."""
        if not self._loaded:
            raise RuntimeError(f"OpenAI adapter {self.model_name} not loaded")

        messages = [{"role": "user", "content": prompt}]
        return await self._chat_completions(messages, **kwargs)

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        """Chat-style inference with native OpenAI message format."""
        if not self._loaded:
            raise RuntimeError(f"OpenAI adapter {self.model_name} not loaded")

        return await self._chat_completions(messages, **kwargs)

    async def _chat_completions(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> dict[str, Any]:
        """Core method: POST /chat/completions."""
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
        }

        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]

        async with httpx.AsyncClient(timeout=_GENERATE_TIMEOUT) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )

            if resp.status_code == 401:
                raise RuntimeError("OpenAI API key invalid or expired (401 Unauthorized)")
            if resp.status_code == 429:
                raise RuntimeError("OpenAI rate limit exceeded (429). Retry later.")
            resp.raise_for_status()

            data = resp.json()

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = data.get("usage", {})

        return {
            "text": message.get("content", ""),
            "tokens_in": usage.get("prompt_tokens", 0),
            "tokens_out": usage.get("completion_tokens", 0),
            "model": data.get("model", self.model_name),
            "finish_reason": choice.get("finish_reason", ""),
        }

    async def generate_stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Stream tokens via SSE from /chat/completions with stream=True."""
        if not self._loaded:
            raise RuntimeError(f"OpenAI adapter {self.model_name} not loaded")

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }

        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]

        async with httpx.AsyncClient(timeout=_GENERATE_TIMEOUT) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]  # Strip "data: " prefix
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    async def generate_batch(
        self,
        prompts: list[str],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        system: str | None = None,
        poll_interval: int = 30,
        on_progress: Any = None,
        stage_name: str = "",
        save_dir: str = "",
        stall_timeout: int = 300,
        no_wait: bool = False,
    ) -> list[dict[str, Any]]:
        """Submit prompts as an OpenAI batch and wait for results.

        Uses OpenAI Batch API: upload JSONL → create batch → poll → download.
        Returns list of result dicts in same order as prompts.

        If no_wait=True, submits the batch and raises BatchPending instead of
        polling. The pipeline should exit cleanly and pick up results on the
        next run via the stalled-batch recovery path.

        Features:
        - Saves results to disk before returning (crash-safe)
        - Batch tracking manifest (batch_id → stage mapping)
        - Stall timeout: if >95% done and no progress for stall_timeout seconds, accepts partial
        - Cost tracking per batch
        """
        import asyncio
        import time
        from pathlib import Path

        if not self._loaded or not self._api_key:
            raise RuntimeError(f"OpenAI adapter {self.model_name} not loaded")

        # Resolve save directory
        if not save_dir:
            save_dir = os.environ.get("MULTIHEAD_DATA_DIR", str(Path.home() / ".multihead"))
        batches_dir = Path(save_dir) / "nightshift_output" / "batches"
        batches_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = batches_dir / "manifest.json"

        # Check for existing saved results (resume support)
        if stage_name:
            for existing in batches_dir.glob("*.jsonl"):
                meta_path = existing.with_suffix(".meta.json")
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text())
                        if (meta.get("stage_name") == stage_name
                                and meta.get("prompt_count") == len(prompts)
                                and meta.get("status") == "completed"):
                            logger.info("OpenAI batch: resuming from saved results %s for %s",
                                        existing.name, stage_name)
                            return self._parse_saved_results(existing, len(prompts))
                    except (json.JSONDecodeError, OSError):
                        continue

            # Check for stalled batches that may have completed on OpenAI's side
            for meta_file in batches_dir.glob("*.meta.json"):
                try:
                    meta = json.loads(meta_file.read_text())
                    if (meta.get("stage_name") == stage_name
                            and meta.get("prompt_count") == len(prompts)
                            and meta.get("status") in ("stalled", "submitted")):
                        batch_id_stalled = meta.get("batch_id")
                        if batch_id_stalled:
                            logger.info(
                                "Found stalled batch %s for %s — checking if OpenAI completed it",
                                batch_id_stalled, stage_name,
                            )
                            results = await self._try_recover_stalled_batch(
                                batch_id_stalled, len(prompts), batches_dir, meta_file,
                            )
                            if results is not None:
                                return results
                except (json.JSONDecodeError, OSError):
                    continue

        # 1. Build JSONL request file
        lines = []
        for i, prompt in enumerate(prompts):
            messages = [{"role": "user", "content": prompt}]
            if system:
                messages.insert(0, {"role": "system", "content": system})
            request = {
                "custom_id": f"req_{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            }
            lines.append(json.dumps(request))

        jsonl_content = "\n".join(lines)
        logger.info("OpenAI batch: %d requests for %s, uploading...", len(prompts), stage_name or "batch")

        async with httpx.AsyncClient(timeout=300.0) as client:
            # 2. Upload the JSONL file
            resp = await client.post(
                f"{self.base_url}/files",
                headers={"Authorization": f"Bearer {self._api_key}"},
                files={"file": ("batch_input.jsonl", jsonl_content.encode(), "application/jsonl")},
                data={"purpose": "batch"},
            )
            resp.raise_for_status()
            file_id = resp.json()["id"]

            # 3. Create batch
            resp = await client.post(
                f"{self.base_url}/batches",
                headers=self._headers(),
                json={
                    "input_file_id": file_id,
                    "endpoint": "/v1/chat/completions",
                    "completion_window": "24h",
                },
            )
            resp.raise_for_status()
            batch_id = resp.json()["id"]
            logger.info("OpenAI batch: %s created for stage=%s (%d prompts)",
                        batch_id, stage_name or "?", len(prompts))

            # Save batch metadata for tracking
            meta = {
                "batch_id": batch_id,
                "stage_name": stage_name,
                "model": self.model_name,
                "prompt_count": len(prompts),
                "submitted_at": time.time(),
                "status": "submitted",
            }
            meta_path = batches_dir / f"{batch_id}.meta.json"
            meta_path.write_text(json.dumps(meta, indent=2))

            # No-wait mode: submit and exit — results picked up next run
            if no_wait:
                logger.info(
                    "OpenAI batch %s submitted (no_wait mode) — will recover on next run",
                    batch_id,
                )
                raise BatchPending(
                    f"Batch {batch_id} submitted, results pending",
                    batch_id=batch_id,
                    stage_name=stage_name or "",
                )

            # 4. Poll for completion with stall detection
            t0 = time.time()
            last_completed = 0
            last_progress_time = time.time()

            while True:
                await asyncio.sleep(poll_interval)
                resp = await client.get(
                    f"{self.base_url}/batches/{batch_id}",
                    headers=self._headers(),
                )
                resp.raise_for_status()
                batch_data = resp.json()
                status = batch_data.get("status", "")
                counts = batch_data.get("request_counts", {})
                completed = counts.get("completed", 0)
                failed = counts.get("failed", 0)
                total = counts.get("total", len(prompts))

                if on_progress and completed > 0:
                    on_progress(completed, total)

                elapsed = time.time() - t0
                logger.info(
                    "OpenAI batch %s [%s]: %s (%d/%d, %d failed, %.0fs)",
                    batch_id, stage_name or "?", status,
                    completed, total, failed, elapsed,
                )

                # Track progress for stall detection
                if completed > last_completed:
                    last_completed = completed
                    last_progress_time = time.time()

                if status == "completed":
                    break
                elif status in ("failed", "expired", "cancelled"):
                    # Try to get partial results
                    output_file_id = batch_data.get("output_file_id")
                    if output_file_id and completed > 0:
                        logger.warning("Batch %s %s with %d/%d — downloading partial results",
                                       batch_id, status, completed, total)
                        break
                    raise RuntimeError(f"OpenAI batch {batch_id} {status}")

                # Stall detection: >95% done and no progress for stall_timeout
                if total > 0 and completed / total > 0.95:
                    stall_duration = time.time() - last_progress_time
                    if stall_duration > stall_timeout:
                        logger.warning(
                            "Batch %s stalled at %d/%d for %.0fs — accepting partial results",
                            batch_id, completed, total, stall_duration,
                        )
                        break

            # 5. Download results (retry if output file not ready yet)
            output_file_id = batch_data.get("output_file_id")
            if not output_file_id:
                # Stalled batches may need extra time for output file generation
                for retry in range(6):
                    await asyncio.sleep(10)
                    resp = await client.get(
                        f"{self.base_url}/batches/{batch_id}",
                        headers=self._headers(),
                    )
                    resp.raise_for_status()
                    batch_data = resp.json()
                    output_file_id = batch_data.get("output_file_id")
                    if output_file_id:
                        logger.info("Batch %s output file appeared after %d retries", batch_id, retry + 1)
                        break
                if not output_file_id:
                    # Save stalled status so next run can try to recover
                    meta["status"] = "stalled"
                    meta["stalled_at"] = time.time()
                    meta["completed_count"] = completed
                    meta_path.write_text(json.dumps(meta, indent=2))
                    raise BatchPartialError(
                        f"No output file for batch {batch_id}",
                        completed_count=completed,
                        total_count=total,
                    )

            resp = await client.get(
                f"{self.base_url}/files/{output_file_id}/content",
                headers=self._headers(),
            )
            resp.raise_for_status()

        # 6. Save raw results to disk BEFORE parsing
        results_path = batches_dir / f"{batch_id}.jsonl"
        results_path.write_text(resp.text)

        # Update metadata
        total_tokens_in = 0
        total_tokens_out = 0
        meta["status"] = "completed"
        meta["completed_at"] = time.time()
        meta["duration_seconds"] = round(time.time() - t0, 1)
        meta["results_file"] = str(results_path)

        # 7. Parse results and reorder
        results_by_id: dict[str, dict] = {}
        for line in resp.text.strip().split("\n"):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                custom_id = item.get("custom_id", "")
                response = item.get("response", {})
                body = response.get("body", {})
                if response.get("status_code") == 200:
                    choice = body.get("choices", [{}])[0]
                    usage = body.get("usage", {})
                    tokens_in = usage.get("prompt_tokens", 0)
                    tokens_out = usage.get("completion_tokens", 0)
                    total_tokens_in += tokens_in
                    total_tokens_out += tokens_out
                    results_by_id[custom_id] = {
                        "text": choice.get("message", {}).get("content", ""),
                        "tokens_in": tokens_in,
                        "tokens_out": tokens_out,
                        "model": body.get("model", self.model_name),
                    }
                else:
                    results_by_id[custom_id] = {
                        "text": "",
                        "error": f"status {response.get('status_code')}",
                    }
            except json.JSONDecodeError:
                continue

        # Reorder to match input
        ordered = []
        for i in range(len(prompts)):
            key = f"req_{i}"
            ordered.append(results_by_id.get(key, {"text": "", "error": "missing"}))

        # Save final metadata with cost tracking
        meta["total_tokens_in"] = total_tokens_in
        meta["total_tokens_out"] = total_tokens_out
        meta["results_count"] = len(results_by_id)
        meta_path.write_text(json.dumps(meta, indent=2))

        logger.info(
            "OpenAI batch %s [%s] complete: %d results, %d tokens in, %d tokens out, %.0fs",
            batch_id, stage_name or "?", len(results_by_id),
            total_tokens_in, total_tokens_out, time.time() - t0,
        )
        return ordered

    def _parse_saved_results(self, path, prompt_count: int) -> list[dict[str, Any]]:
        """Parse previously saved batch results from disk."""
        results_by_id: dict[str, dict] = {}
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    custom_id = item.get("custom_id", "")
                    response = item.get("response", {})
                    body = response.get("body", {})
                    if response.get("status_code") == 200:
                        choice = body.get("choices", [{}])[0]
                        usage = body.get("usage", {})
                        results_by_id[custom_id] = {
                            "text": choice.get("message", {}).get("content", ""),
                            "tokens_in": usage.get("prompt_tokens", 0),
                            "tokens_out": usage.get("completion_tokens", 0),
                            "model": body.get("model", self.model_name),
                        }
                    else:
                        results_by_id[custom_id] = {"text": "", "error": "failed"}
                except json.JSONDecodeError:
                    continue

        ordered = []
        for i in range(prompt_count):
            ordered.append(results_by_id.get(f"req_{i}", {"text": "", "error": "missing"}))
        logger.info("Loaded %d results from saved file %s", len(results_by_id), path.name)
        return ordered

    async def _try_recover_stalled_batch(
        self,
        batch_id: str,
        prompt_count: int,
        batches_dir: Path,
        meta_path: Path,
    ) -> list[dict[str, Any]] | None:
        """Try to download results from a previously stalled batch.

        OpenAI keeps batch output files for 30 days, so a batch that stalled
        during polling may have since completed.
        """
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.get(
                    f"{self.base_url}/batches/{batch_id}",
                    headers=self._headers(),
                )
                resp.raise_for_status()
                batch_data = resp.json()
                status = batch_data.get("status", "")
                output_file_id = batch_data.get("output_file_id")

                if not output_file_id:
                    logger.info("Stalled batch %s still has no output file (status=%s)", batch_id, status)
                    return None

                logger.info("Recovering stalled batch %s (status=%s) — downloading output", batch_id, status)
                resp = await client.get(
                    f"{self.base_url}/files/{output_file_id}/content",
                    headers=self._headers(),
                )
                resp.raise_for_status()

            # Save results to disk
            results_path = batches_dir / f"{batch_id}.jsonl"
            results_path.write_text(resp.text)

            # Update metadata
            meta = json.loads(meta_path.read_text())
            meta["status"] = "completed"
            meta["recovered_at"] = __import__("time").time()
            meta["results_file"] = str(results_path)
            meta_path.write_text(json.dumps(meta, indent=2))

            return self._parse_saved_results(results_path, prompt_count)
        except Exception as e:
            logger.warning("Failed to recover stalled batch %s: %s", batch_id, e)
            return None

    async def healthcheck(self) -> bool:
        """Check API connectivity by listing models."""
        if not self._api_key:
            return False
        try:
            async with httpx.AsyncClient(timeout=_HEALTHCHECK_TIMEOUT) as client:
                resp = await client.get(
                    f"{self.base_url}/models",
                    headers=self._headers(),
                )
                return resp.status_code == 200
        except Exception:
            return False

    async def sleep(self, level: int = 1) -> None:
        """No-op — no GPU state to manage."""
        pass

    async def wake(self) -> None:
        """No-op — stateless."""
        await self.load()
