"""vLLM adapter with sleep/wake support for fast head swapping."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..models import HeadManifest
from .base import HeadAdapter

logger = logging.getLogger(__name__)


class VLLMAdapter(HeadAdapter):
    """Adapter for vLLM inference server with sleep mode support.

    vLLM sleep mode releases GPU memory without stopping the server:
    - Level 1: offload weights to CPU RAM (fast wake, uses CPU RAM)
    - Level 2: discard weights entirely (slower wake, minimal CPU RAM)

    Server must be started with --enable-sleep-mode and VLLM_SERVER_DEV_MODE=1.
    """

    def __init__(self, manifest: HeadManifest) -> None:
        super().__init__(manifest)
        self.base_url = manifest.endpoint or "http://localhost:8000"
        self.model_name = manifest.model

    async def load(self) -> None:
        """Wake the vLLM server model from sleep."""
        await self.wake()

    async def unload(self) -> None:
        """Put model to sleep at level 2 (discard weights)."""
        await self.sleep(level=2)

    async def generate(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Generate via OpenAI-compatible /v1/completions."""
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 0.7),
        }

        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(f"{self.base_url}/v1/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()

        choice = data.get("choices", [{}])[0]
        usage = data.get("usage", {})
        return {
            "text": choice.get("text", ""),
            "tokens_in": usage.get("prompt_tokens", 0),
            "tokens_out": usage.get("completion_tokens", 0),
        }

    async def generate_stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Stream tokens via OpenAI-compatible /v1/completions with stream=True."""
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream("POST", f"{self.base_url}/v1/completions", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]  # Strip "data: " prefix
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        choice = data.get("choices", [{}])[0]
                        token = choice.get("text", "")
                        if token:
                            yield token
                    except json.JSONDecodeError:
                        continue

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        """Chat via OpenAI-compatible /v1/chat/completions."""
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 0.7),
        }

        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(f"{self.base_url}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()

        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        usage = data.get("usage", {})
        return {
            "text": msg.get("content", ""),
            "tokens_in": usage.get("prompt_tokens", 0),
            "tokens_out": usage.get("completion_tokens", 0),
        }

    async def sleep(self, level: int = 1) -> None:
        """Put model into sleep mode. Level 1 = CPU offload, Level 2 = discard."""
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(f"{self.base_url}/sleep", params={"level": level})
                resp.raise_for_status()
            logger.info("vLLM: model sleeping at level %d", level)
        except httpx.HTTPError as e:
            logger.warning("vLLM sleep failed (server may not support sleep mode): %s", e)

    async def wake(self) -> None:
        """Wake model from sleep."""
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(f"{self.base_url}/wake_up")
                resp.raise_for_status()
            logger.info("vLLM: model awake")
        except httpx.HTTPError as e:
            logger.warning("vLLM wake failed: %s", e)

    async def healthcheck(self) -> bool:
        """Check if vLLM server is responding."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.base_url}/v1/models")
                resp.raise_for_status()
                return True
        except Exception:
            return False
