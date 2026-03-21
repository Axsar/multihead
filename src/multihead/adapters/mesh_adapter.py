"""Adapter for remote mesh peer heads — routes generate() to a peer node via MeshClient."""

from __future__ import annotations

import logging
import time
from typing import Any

from .base import HeadAdapter
from ..models import HeadManifest

logger = logging.getLogger(__name__)


class MeshHeadAdapter(HeadAdapter):
    """Routes inference requests to a remote peer node's REST API.

    The peer_url and auth_token are stored in manifest.extra:
        {"peer_url": "http://192.168.1.10:7337", "peer_node_id": "desktop-01"}
    """

    def __init__(self, manifest: HeadManifest) -> None:
        super().__init__(manifest)
        extra = manifest.extra or {}
        peer_url = extra.get("peer_url", "")
        auth_token = extra.get("auth_token", "")
        if not peer_url:
            raise ValueError(f"MeshHeadAdapter requires 'peer_url' in manifest.extra for {manifest.head_id}")

        # Lazy import to avoid circular dependency
        from ..mesh.client import MeshClient
        self._client = MeshClient(
            base_url=peer_url,
            auth_token=auth_token,
            timeout=extra.get("timeout", 60.0),
        )
        self._peer_url = peer_url
        self._peer_node_id = extra.get("peer_node_id", "unknown")

    async def load(self) -> None:
        """Mesh peers are always 'loaded' (remote service)."""
        logger.debug("MeshHeadAdapter %s: load() (no-op for remote peer %s)", self.manifest.head_id, self._peer_node_id)

    async def unload(self) -> None:
        """Mesh peers don't need unloading."""
        logger.debug("MeshHeadAdapter %s: unload() (no-op for remote peer %s)", self.manifest.head_id, self._peer_node_id)

    async def generate(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Submit task to remote peer and return result."""
        t0 = time.perf_counter()

        result = await self._client.submit_task(
            capability_kind=self.manifest.kind,
            prompt=prompt,
            model=self.manifest.model,
            params=kwargs,
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        text = result.get("result", "")
        if isinstance(text, dict):
            text = text.get("text", str(text))

        return {
            "text": str(text),
            "tokens_in": len(prompt) // 4,
            "tokens_out": len(str(text)) // 4,
            "latency_ms": latency_ms,
            "peer_node_id": self._peer_node_id,
            "remote_head_id": result.get("head_id", ""),
        }

    async def healthcheck(self) -> bool:
        """Check if the peer is reachable."""
        try:
            health = await self._client.health()
            return health.get("status") == "ok"
        except Exception:
            return False
