"""VRAM management policy for core/worker head switching."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from multihead.head_manager import HeadManager
from multihead.models import HeadState

logger = logging.getLogger(__name__)


@dataclass
class VRAMPolicy:
    """Policy for managing VRAM between core LLM and worker heads."""
    core_mode: Literal["keep_loaded", "cpu_fallback", "unload_during_batch"] = "keep_loaded"
    worker_load_policy: Literal["per_stage", "keep_warm"] = "per_stage"


class VRAMManager:
    """Manages VRAM allocation between core and worker heads."""

    def __init__(self, head_manager: HeadManager, policy: VRAMPolicy, core_head_id: str = "core-llm") -> None:
        self.heads = head_manager
        self.policy = policy
        self.core_head_id = core_head_id
        self._core_was_active = False

    async def prepare_for_batch(self, worker_head_id: str) -> None:
        """Prepare VRAM for a worker batch (potentially unloading core)."""
        if self.policy.core_mode == "keep_loaded":
            # Core stays loaded, worker must share or wait
            pass
        elif self.policy.core_mode == "unload_during_batch":
            # Unload core to free VRAM for worker
            state = self.heads.get_state(self.core_head_id)
            self._core_was_active = state == HeadState.ACTIVE
            if self._core_was_active:
                logger.info("Unloading core for worker batch: %s", worker_head_id)
                await self.heads.unload_head(self.core_head_id)
        elif self.policy.core_mode == "cpu_fallback":
            # Core can fall back to CPU (implementation depends on adapter)
            logger.info("Core falling back to CPU for worker batch: %s", worker_head_id)

        # Load worker
        await self.heads.ensure_active(worker_head_id)

    async def restore_after_batch(self) -> None:
        """Restore core after a worker batch completes."""
        if self.policy.core_mode == "unload_during_batch" and self._core_was_active:
            logger.info("Restoring core after batch")
            await self.heads.ensure_active(self.core_head_id)
            self._core_was_active = False

    async def ensure_core_available(self) -> None:
        """Ensure the core LLM is loaded and ready."""
        await self.heads.ensure_active(self.core_head_id)

    def get_vram_status(self) -> dict[str, Any]:
        """Get current VRAM allocation status."""
        states = self.heads.get_states()
        return {
            "core_head_id": self.core_head_id,
            "core_state": states.get(self.core_head_id, {}).get("state", "unknown"),
            "policy": {
                "core_mode": self.policy.core_mode,
                "worker_load_policy": self.policy.worker_load_policy,
            },
            "active_head": self.heads.active_head,
            "heads": {hid: info["state"] for hid, info in states.items()},
        }
