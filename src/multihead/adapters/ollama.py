"""Ollama HTTP adapter for model lifecycle and inference."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..models import HeadManifest
from .base import HeadAdapter

logger = logging.getLogger(__name__)


class OllamaAdapter(HeadAdapter):
    """Adapter for Ollama local model server.

    Controls model lifecycle via keep_alive parameter:
    - load: empty prompt with keep_alive=-1 (keep loaded indefinitely)
    - unload: keep_alive=0 (unload immediately)
    - generate: POST /api/chat or /api/generate
    - healthcheck: GET /api/ps (check model in running list)
    """

    def __init__(self, manifest: HeadManifest) -> None:
        super().__init__(manifest)
        self.base_url = manifest.endpoint or "http://localhost:11434"
        self.model_name = manifest.model

    async def load(self) -> None:
        """Preload model by sending empty prompt with keep_alive=-1."""
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": "",
                    "keep_alive": -1,
                },
            )
            resp.raise_for_status()
        logger.info("Ollama: loaded %s", self.model_name)

    async def unload(self) -> None:
        """Unload model by sending keep_alive=0."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": "",
                    "keep_alive": 0,
                },
            )
            resp.raise_for_status()
        logger.info("Ollama: unloaded %s", self.model_name)

    async def generate(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Generate text via Ollama API."""
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "keep_alive": -1,
        }
        if "temperature" in kwargs:
            payload["options"] = {"temperature": kwargs["temperature"]}
        if "max_tokens" in kwargs:
            payload.setdefault("options", {})["num_predict"] = kwargs["max_tokens"]

        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()

        return {
            "text": data.get("response", ""),
            "tokens_in": data.get("prompt_eval_count", 0),
            "tokens_out": data.get("eval_count", 0),
            "total_duration_ns": data.get("total_duration", 0),
        }

    async def generate_stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Stream tokens from Ollama /api/generate with stream=True."""
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "keep_alive": -1,
        }
        if "temperature" in kwargs:
            payload["options"] = {"temperature": kwargs["temperature"]}
        if "max_tokens" in kwargs:
            payload.setdefault("options", {})["num_predict"] = kwargs["max_tokens"]

        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield token
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        """Chat-style inference via Ollama /api/chat."""
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "keep_alive": -1,
        }
        if "temperature" in kwargs:
            payload["options"] = {"temperature": kwargs["temperature"]}

        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

        msg = data.get("message", {})
        return {
            "text": msg.get("content", ""),
            "tokens_in": data.get("prompt_eval_count", 0),
            "tokens_out": data.get("eval_count", 0),
            "total_duration_ns": data.get("total_duration", 0),
        }

    async def healthcheck(self) -> bool:
        """Check if model is loaded via /api/ps."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.base_url}/api/ps")
                resp.raise_for_status()
                data = resp.json()
            models = data.get("models", [])
            return any(m.get("name", "").startswith(self.model_name) for m in models)
        except Exception:
            return False
