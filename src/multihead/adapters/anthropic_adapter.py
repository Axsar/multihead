"""Anthropic Messages API adapter with batch support.

Supports two modes:
- Real-time: single generate() calls via Messages API
- Batch: bulk generate_batch() via Message Batches API (50% cheaper)

Uses the official anthropic SDK (>= 0.75.0).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any

from ..models import HeadManifest
from .base import HeadAdapter

logger = logging.getLogger(__name__)

_GENERATE_TIMEOUT = 120.0
_BATCH_POLL_INTERVAL = 30  # seconds between batch status checks
_BATCH_MAX_WAIT = 86400  # 24 hours max wait
_BATCH_CHUNK_SIZE = 2000  # max requests per batch submission


class AnthropicAdapter(HeadAdapter):
    """Adapter for Anthropic Messages API (real-time + batch).

    API key resolved from manifest.extra["api_key"] or ANTHROPIC_API_KEY env var.
    Batch mode available via generate_batch() for bulk workloads at 50% cost.
    """

    def __init__(self, manifest: HeadManifest) -> None:
        super().__init__(manifest)
        self.model_name = manifest.model
        self._api_key: str | None = (
            (manifest.extra or {}).get("api_key")
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        self._client = None
        self._async_client = None
        self._loaded = False

    async def load(self) -> None:
        """Initialize the Anthropic client."""
        import anthropic

        if not self._api_key:
            raise RuntimeError(
                "Anthropic API key not configured. "
                "Set 'api_key' in head manifest extra or ANTHROPIC_API_KEY env var."
            )
        # Resolve ${ENV_VAR} references
        if self._api_key.startswith("${") and self._api_key.endswith("}"):
            env_name = self._api_key[2:-1]
            self._api_key = os.environ.get(env_name)
            if not self._api_key:
                raise RuntimeError(f"Environment variable {env_name} not set")

        self._client = anthropic.Anthropic(api_key=self._api_key)
        self._async_client = anthropic.AsyncAnthropic(api_key=self._api_key)
        self._loaded = True
        logger.info("Anthropic adapter ready: %s", self.model_name)

    async def unload(self) -> None:
        """Close clients."""
        self._client = None
        self._async_client = None
        self._loaded = False
        logger.info("Anthropic adapter unloaded: %s", self.model_name)

    async def generate(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Real-time generation via Messages API."""
        if not self._loaded or not self._async_client:
            raise RuntimeError(f"Anthropic adapter {self.model_name} not loaded")

        params: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "messages": [{"role": "user", "content": prompt}],
        }

        if "temperature" in kwargs:
            params["temperature"] = kwargs["temperature"]
        if "system" in kwargs:
            params["system"] = kwargs["system"]

        message = await self._async_client.messages.create(**params)

        text = ""
        for block in message.content:
            if hasattr(block, "text"):
                text += block.text

        return {
            "text": text,
            "tokens_in": message.usage.input_tokens,
            "tokens_out": message.usage.output_tokens,
            "model": message.model,
            "finish_reason": message.stop_reason or "",
        }

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        """Chat-style inference with native Anthropic message format."""
        if not self._loaded or not self._async_client:
            raise RuntimeError(f"Anthropic adapter {self.model_name} not loaded")

        # Separate system message if present
        system = None
        chat_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system = msg["content"]
            else:
                chat_messages.append(msg)

        params: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "messages": chat_messages,
        }
        if system:
            params["system"] = system
        if "temperature" in kwargs:
            params["temperature"] = kwargs["temperature"]

        message = await self._async_client.messages.create(**params)

        text = ""
        for block in message.content:
            if hasattr(block, "text"):
                text += block.text

        return {
            "text": text,
            "tokens_in": message.usage.input_tokens,
            "tokens_out": message.usage.output_tokens,
            "model": message.model,
            "finish_reason": message.stop_reason or "",
        }

    async def generate_stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Stream tokens via Messages API."""
        if not self._loaded or not self._async_client:
            raise RuntimeError(f"Anthropic adapter {self.model_name} not loaded")

        async with self._async_client.messages.stream(
            model=self.model_name,
            max_tokens=kwargs.get("max_tokens", 2048),
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield text

    # ------------------------------------------------------------------
    # Batch API
    # ------------------------------------------------------------------

    async def generate_batch(
        self,
        prompts: list[str],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        system: str | None = None,
        poll_interval: int = _BATCH_POLL_INTERVAL,
        on_progress: Any = None,
    ) -> list[dict[str, Any]]:
        """Submit prompts as a batch and wait for results.

        Returns a list of result dicts in the same order as prompts.
        Each dict has 'text', 'tokens_in', 'tokens_out', or 'error'.

        Args:
            prompts: List of prompt strings.
            max_tokens: Max output tokens per request.
            temperature: Sampling temperature.
            system: Optional system prompt (shared across all requests).
            poll_interval: Seconds between status polls.
            on_progress: Optional callback(completed, total) for progress.
        """
        if not self._loaded or not self._client:
            raise RuntimeError(f"Anthropic adapter {self.model_name} not loaded")

        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        # Build batch requests
        requests = []
        for i, prompt in enumerate(prompts):
            params: dict[str, Any] = {
                "model": self.model_name,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if temperature is not None:
                params["temperature"] = temperature
            if system:
                params["system"] = system

            requests.append(
                Request(
                    custom_id=f"req_{i:06d}",
                    params=MessageCreateParamsNonStreaming(**params),
                )
            )

        logger.info(
            "Submitting batch: %d requests to %s", len(requests), self.model_name
        )

        # Split into sub-batches to avoid payload size limits
        all_results: dict[str, dict[str, Any]] = {}
        chunks = [
            requests[i : i + _BATCH_CHUNK_SIZE]
            for i in range(0, len(requests), _BATCH_CHUNK_SIZE)
        ]
        logger.info(
            "Split into %d sub-batch(es) of up to %d requests",
            len(chunks), _BATCH_CHUNK_SIZE,
        )

        for chunk_idx, chunk_requests in enumerate(chunks):
            try:
                batch = self._client.messages.batches.create(requests=chunk_requests)
            except Exception as e:
                logger.error("Batch create failed for sub-batch %d (%d requests): %s", chunk_idx + 1, len(chunk_requests), e)
                raise
            batch_id = batch.id
            logger.info(
                "Sub-batch %d/%d created: %s (%d requests, status: %s)",
                chunk_idx + 1, len(chunks), batch_id,
                len(chunk_requests), batch.processing_status,
            )

            # Poll until done
            t0 = time.time()
            while True:
                batch = self._client.messages.batches.retrieve(batch_id)
                status = batch.processing_status

                if on_progress and hasattr(batch, "request_counts"):
                    counts = batch.request_counts
                    completed_so_far = len(all_results) + getattr(counts, "succeeded", 0) + getattr(counts, "errored", 0)
                    on_progress(completed_so_far, len(prompts))

                if status == "ended":
                    logger.info("Sub-batch %s completed in %.1fs", batch_id, time.time() - t0)
                    break

                if status == "canceled":
                    raise RuntimeError(f"Batch {batch_id} was canceled")

                elapsed = time.time() - t0
                if elapsed > _BATCH_MAX_WAIT:
                    raise TimeoutError(
                        f"Batch {batch_id} did not complete within {_BATCH_MAX_WAIT}s"
                    )

                logger.debug(
                    "Batch %s: %s (%.0fs elapsed)", batch_id, status, elapsed
                )
                await asyncio.sleep(poll_interval)

            # Collect results from this sub-batch
            for result in self._client.messages.batches.results(batch_id):
                custom_id = result.custom_id
                if result.result.type == "succeeded":
                    message = result.result.message
                    text = ""
                    for block in message.content:
                        if hasattr(block, "text"):
                            text += block.text
                    all_results[custom_id] = {
                        "text": text,
                        "tokens_in": message.usage.input_tokens,
                        "tokens_out": message.usage.output_tokens,
                        "model": message.model,
                        "finish_reason": message.stop_reason or "",
                    }
                elif result.result.type == "errored":
                    all_results[custom_id] = {
                        "text": "",
                        "error": str(result.result.error),
                        "tokens_in": 0,
                        "tokens_out": 0,
                    }
                else:
                    all_results[custom_id] = {
                        "text": "",
                        "error": f"Request {result.result.type}",
                        "tokens_in": 0,
                        "tokens_out": 0,
                    }

        # Return in original order
        ordered = []
        for i in range(len(prompts)):
            key = f"req_{i:06d}"
            ordered.append(all_results.get(key, {"text": "", "error": "missing"}))

        succeeded = sum(1 for r in ordered if "error" not in r)
        logger.info(
            "Batch complete: %d/%d succeeded across %d sub-batch(es)",
            succeeded, len(prompts), len(chunks),
        )

        return ordered

    async def healthcheck(self) -> bool:
        """Check API connectivity."""
        if not self._loaded or not self._async_client:
            return False
        try:
            # Small test call
            message = await self._async_client.messages.create(
                model=self.model_name,
                max_tokens=5,
                messages=[{"role": "user", "content": "hi"}],
            )
            return bool(message.content)
        except Exception:
            return False

    async def sleep(self, level: int = 1) -> None:
        """No-op — stateless HTTP client."""
        pass

    async def wake(self) -> None:
        """No-op — stateless."""
        await self.load()
