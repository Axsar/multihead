"""Abstract base class for head adapters."""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from typing import Any

from ..models import HeadManifest


class HeadAdapter(abc.ABC):
    """Interface that all head adapters must implement."""

    def __init__(self, manifest: HeadManifest) -> None:
        self.manifest = manifest

    @abc.abstractmethod
    async def load(self) -> None:
        """Load the model into memory (GPU/CPU)."""

    @abc.abstractmethod
    async def unload(self) -> None:
        """Fully unload the model from memory."""

    @abc.abstractmethod
    async def generate(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Run inference. Returns dict with at least 'text' key."""

    async def generate_stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Yield text chunks as they are generated.

        Default: calls generate() and yields the full result as one chunk.
        Subclasses can override for true token-level streaming.
        """
        result = await self.generate(prompt, **kwargs)
        yield result.get("text", "")

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        """Chat-style inference. Default: format messages into a prompt and call generate."""
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return await self.generate(prompt, **kwargs)

    async def healthcheck(self) -> bool:
        """Check if the model is loaded and responsive."""
        return True

    async def sleep(self, level: int = 1) -> None:
        """Put model into low-power state. Not all adapters support this."""
        await self.unload()

    async def wake(self) -> None:
        """Wake model from sleep. Default: just load."""
        await self.load()
