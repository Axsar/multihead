"""Adapter for embedding models (e.g., via Ollama /api/embeddings)."""

from __future__ import annotations

from typing import Any

import httpx

from multihead.adapters.base import HeadAdapter
from multihead.models import HeadManifest


class EmbeddingAdapter(HeadAdapter):
    """Adapter for embedding models. Uses Ollama /api/embeddings by default."""

    def __init__(self, manifest: HeadManifest) -> None:
        super().__init__(manifest)
        self.endpoint = manifest.endpoint or "http://localhost:11434"
        self._loaded = False

    async def load(self) -> None:
        self._loaded = True

    async def unload(self) -> None:
        self._loaded = False

    async def generate(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """For embeddings, generate returns the embedding vector."""
        return await self.embed(prompt)

    async def embed(self, text: str) -> dict[str, Any]:
        """Return embedding vector for a single text."""
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self.endpoint}/api/embeddings",
                json={"model": self.manifest.model, "prompt": text},
            )
            resp.raise_for_status()
            data = resp.json()
        return {
            "embedding": data.get("embedding", []),
            "tokens_in": len(text) // 4,
        }

    async def embed_batch(self, texts: list[str]) -> dict[str, Any]:
        """Batch embedding for efficiency."""
        embeddings = []
        total_tokens = 0
        for text in texts:
            result = await self.embed(text)
            embeddings.append(result["embedding"])
            total_tokens += result.get("tokens_in", 0)
        return {
            "embeddings": embeddings,
            "tokens_in": total_tokens,
            "count": len(texts),
        }

    async def healthcheck(self) -> bool:
        return self._loaded
