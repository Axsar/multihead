"""Capability registry for mesh nodes."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from multihead.models import HeadManifest, new_id


class Capability(BaseModel):
    """A capability advertised by a mesh node."""
    capability_id: str = ""
    node_id: str = ""
    name: str
    kind: str  # llm | vlm | embed | tool
    model: str = ""
    gpu_required: bool = False
    vram_hint_mb: int = 0
    status: str = "available"  # available | busy | offline
    registered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def model_post_init(self, __context: Any) -> None:
        if not self.capability_id:
            self.capability_id = new_id("cap_")


class CapabilityRegistry:
    """Registry of capabilities across mesh nodes."""

    def __init__(self) -> None:
        self._capabilities: dict[str, Capability] = {}

    def register(self, cap: Capability) -> None:
        self._capabilities[cap.capability_id] = cap

    def unregister(self, capability_id: str) -> None:
        self._capabilities.pop(capability_id, None)

    def list_capabilities(self, kind: str | None = None) -> list[Capability]:
        caps = list(self._capabilities.values())
        if kind:
            caps = [c for c in caps if c.kind == kind]
        return caps

    def find_by_model(self, model: str) -> list[Capability]:
        return [c for c in self._capabilities.values() if c.model == model]

    def find_available(self, kind: str | None = None) -> list[Capability]:
        caps = self.list_capabilities(kind)
        return [c for c in caps if c.status == "available"]

    def update_status(self, capability_id: str, status: str) -> None:
        if capability_id in self._capabilities:
            self._capabilities[capability_id].status = status


def auto_register_from_heads(
    registry: CapabilityRegistry,
    manifests: dict[str, HeadManifest],
    node_id: str,
) -> list[Capability]:
    """Auto-register capabilities from head manifests."""
    registered = []
    for head_id, manifest in manifests.items():
        cap = Capability(
            node_id=node_id,
            name=manifest.name,
            kind=manifest.kind,
            model=manifest.model,
            gpu_required=manifest.gpu_required,
            vram_hint_mb=manifest.vram_hint_mb,
        )
        registry.register(cap)
        registered.append(cap)
    return registered
