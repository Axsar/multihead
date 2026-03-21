"""Aggregates peer capabilities from discovery into a unified registry of remote heads."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from ..models import AdapterKind, HeadManifest

if TYPE_CHECKING:
    from .discovery import MeshDiscovery
    from .security import MeshSecurity

logger = logging.getLogger(__name__)


@dataclass
class PeerHead:
    """A remote head available on a peer node."""
    head_id: str
    node_id: str
    peer_url: str
    capability_kind: str  # llm | vlm | embed
    model: str = ""
    name: str = ""
    status: str = "available"
    latency_ms: float = 0.0
    last_refreshed: float = field(default_factory=time.monotonic)


class PeerRegistry:
    """Maintains a live view of all remote heads across the mesh.

    Polls discovered peers for their capabilities and registers them
    as HeadManifest entries that the Router can score and select.
    """

    def __init__(
        self,
        mesh_discovery: MeshDiscovery,
        auth_token: str = "",
        refresh_interval: float = 30.0,
    ) -> None:
        self._discovery = mesh_discovery
        self._auth_token = auth_token
        self._refresh_interval = refresh_interval
        self._peer_heads: dict[str, PeerHead] = {}
        self._refresh_task: asyncio.Task[None] | None = None

    @property
    def peer_heads(self) -> dict[str, PeerHead]:
        """Return current remote heads."""
        return dict(self._peer_heads)

    def get_peer_for_head(self, head_id: str) -> PeerHead | None:
        """Look up a remote head by its mesh head_id."""
        return self._peer_heads.get(head_id)

    async def refresh(self) -> None:
        """Poll all discovered peers for capabilities and update registry."""
        from .client import MeshClient

        nodes = self._discovery.get_discovered_nodes()
        seen_head_ids: set[str] = set()

        for node_id, info in nodes.items():
            host = info.get("host", "")
            port = info.get("port", 7337)
            if not host:
                continue

            peer_url = f"http://{host}:{port}"
            client = MeshClient(
                base_url=peer_url,
                auth_token=self._auth_token,
                timeout=10.0,
                max_retries=0,
            )

            try:
                t0 = time.perf_counter()
                caps = await client.capabilities()
                latency_ms = (time.perf_counter() - t0) * 1000

                for cap in caps:
                    cap_kind = cap.get("kind", "llm")
                    cap_name = cap.get("name", "")
                    cap_model = cap.get("model", "")
                    cap_status = cap.get("status", "available")

                    head_id = f"mesh-{node_id}-{cap_kind}"
                    if cap_model:
                        # Make head_id more specific if model is known
                        safe_model = cap_model.replace("/", "-").replace(":", "-")
                        head_id = f"mesh-{node_id}-{safe_model}"

                    peer_head = PeerHead(
                        head_id=head_id,
                        node_id=node_id,
                        peer_url=peer_url,
                        capability_kind=cap_kind,
                        model=cap_model,
                        name=cap_name or f"{node_id}/{cap_kind}",
                        status=cap_status,
                        latency_ms=latency_ms,
                    )
                    self._peer_heads[head_id] = peer_head
                    seen_head_ids.add(head_id)

            except Exception as e:
                logger.warning("Failed to refresh peer %s (%s): %s", node_id, peer_url, e)
                # Mark existing heads from this peer as offline
                for hid, ph in self._peer_heads.items():
                    if ph.node_id == node_id:
                        ph.status = "offline"
                        seen_head_ids.add(hid)

        # Remove heads from peers that are no longer discovered
        stale = [hid for hid in self._peer_heads if hid not in seen_head_ids]
        for hid in stale:
            del self._peer_heads[hid]
            logger.debug("Removed stale peer head: %s", hid)

    def to_manifests(self) -> dict[str, HeadManifest]:
        """Convert peer heads to HeadManifest objects for router consumption."""
        manifests: dict[str, HeadManifest] = {}
        for head_id, ph in self._peer_heads.items():
            if ph.status == "offline":
                continue
            manifests[head_id] = HeadManifest(
                head_id=head_id,
                name=ph.name,
                adapter=AdapterKind.MESH,
                model=ph.model,
                kind=ph.capability_kind,
                gpu_required=False,
                is_local=False,
                privacy_level="encrypted",
                extra={
                    "peer_url": ph.peer_url,
                    "peer_node_id": ph.node_id,
                    "auth_token": self._auth_token,
                },
            )
        return manifests

    async def start_periodic_refresh(self) -> None:
        """Start background refresh loop."""
        if self._refresh_task is not None:
            return
        self._refresh_task = asyncio.create_task(self._refresh_loop())
        logger.info("PeerRegistry periodic refresh started (interval=%.0fs)", self._refresh_interval)

    async def stop(self) -> None:
        """Stop background refresh loop."""
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
            self._refresh_task = None

    async def _refresh_loop(self) -> None:
        """Periodically refresh peer capabilities."""
        while True:
            try:
                await self.refresh()
            except Exception as e:
                logger.warning("PeerRegistry refresh failed: %s", e)
            await asyncio.sleep(self._refresh_interval)
