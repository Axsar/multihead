"""Mock adapter for testing pipelines without real models."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from ..models import HeadManifest
from .base import HeadAdapter


class MockAdapter(HeadAdapter):
    """Returns canned responses. Useful for testing orchestration logic."""

    def __init__(self, manifest: HeadManifest) -> None:
        super().__init__(manifest)
        self._loaded = False
        self._call_count = 0

    async def load(self) -> None:
        await asyncio.sleep(0.1)  # Simulate load time
        self._loaded = True

    async def unload(self) -> None:
        await asyncio.sleep(0.05)
        self._loaded = False

    async def generate(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        if not self._loaded:
            raise RuntimeError(f"Mock head {self.manifest.head_id} not loaded")
        self._call_count += 1
        await asyncio.sleep(0.05)  # Simulate inference

        # Return different mock responses based on head kind
        if self.manifest.kind == "vlm":
            return {
                "text": json.dumps({
                    "caption": f"Mock VLM caption for call #{self._call_count}",
                    "entities": ["object_a", "object_b"],
                    "confidence": 0.95,
                }),
                "tokens_in": 100,
                "tokens_out": 50,
            }
        else:
            return {
                "text": f"Mock LLM response for call #{self._call_count}: processed prompt of {len(prompt)} chars",
                "tokens_in": len(prompt) // 4,
                "tokens_out": 50,
            }

    async def generate_stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Yield response word by word to simulate token streaming."""
        result = await self.generate(prompt, **kwargs)
        text = result.get("text", "")
        words = text.split()
        for i, word in enumerate(words):
            await asyncio.sleep(0.01)  # Simulate per-token latency
            yield word if i == 0 else f" {word}"

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return await self.generate(prompt, **kwargs)

    async def healthcheck(self) -> bool:
        return self._loaded
