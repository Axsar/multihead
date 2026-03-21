"""Mesh-aware failover policy for dynamic fallback routing."""

from __future__ import annotations

import logging
import time
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..head_manager import HeadManager
    from .peer_registry import PeerRegistry

logger = logging.getLogger(__name__)


class MeshFailoverPolicy:
    """Builds dynamic fallback lists from local heads + mesh peers.

    When a step fails, the policy provides alternative heads ranked by:
    1. Local heads first (lower latency, no network)
    2. Remote peers by latency (fastest first)
    3. Excludes heads with open circuit breakers
    4. Excludes explicitly excluded head_ids
    """

    def __init__(
        self,
        head_manager: HeadManager,
        peer_registry: PeerRegistry | None = None,
    ) -> None:
        self._head_manager = head_manager
        self._peer_registry = peer_registry
        # Track per-head success/failure counts for ranking
        self._success_counts: dict[str, int] = {}
        self._failure_counts: dict[str, int] = {}
        self._last_latency: dict[str, float] = {}

    def get_fallbacks(
        self,
        head_id: str,
        required_kind: str = "",
        exclude: set[str] | None = None,
    ) -> list[str]:
        """Return ranked list of alternative heads for a failed head.

        Args:
            head_id: The head that failed.
            required_kind: Only return heads matching this kind (llm/vlm/embed).
            exclude: Additional head_ids to exclude.

        Returns:
            List of head_ids, best first. Does NOT include the failed head.
        """
        exclude = (exclude or set()) | {head_id}
        candidates: list[tuple[float, str]] = []

        # 1) Local heads
        for hid, manifest in self._head_manager._manifests.items():
            if hid in exclude:
                continue
            if required_kind and manifest.kind != required_kind:
                continue

            # Skip circuit-broken heads
            breaker = self._head_manager.get_breaker(hid)
            if breaker and breaker.state == "open":
                continue

            # Score: local heads get base 100, adjusted by error rate
            score = 100.0
            failures = self._failure_counts.get(hid, 0)
            successes = self._success_counts.get(hid, 0)
            total = failures + successes
            if total > 0:
                error_rate = failures / total
                score -= error_rate * 50  # Penalize up to -50 for 100% error rate
            candidates.append((score, hid))

        # 2) Mesh peers
        if self._peer_registry:
            for ph_id, ph in self._peer_registry.peer_heads.items():
                if ph_id in exclude:
                    continue
                if ph.status == "offline":
                    continue
                if required_kind and ph.capability_kind != required_kind:
                    continue

                # Score: remote peers get base 50, adjusted by latency
                score = 50.0
                if ph.latency_ms > 0:
                    latency_penalty = min(ph.latency_ms / 100.0, 20.0)  # Cap at -20
                    score -= latency_penalty
                failures = self._failure_counts.get(ph_id, 0)
                successes = self._success_counts.get(ph_id, 0)
                total = failures + successes
                if total > 0:
                    error_rate = failures / total
                    score -= error_rate * 30
                candidates.append((score, ph_id))

        # Sort by score descending
        candidates.sort(key=lambda x: -x[0])
        return [hid for _, hid in candidates]

    def report_failure(self, head_id: str, error: str = "") -> None:
        """Record a failure for a head to demote in future rankings."""
        self._failure_counts[head_id] = self._failure_counts.get(head_id, 0) + 1
        logger.debug("Failover: recorded failure for %s (%s)", head_id, error)

    def report_success(self, head_id: str, latency_ms: float = 0.0) -> None:
        """Record success to boost ranking."""
        self._success_counts[head_id] = self._success_counts.get(head_id, 0) + 1
        if latency_ms > 0:
            self._last_latency[head_id] = latency_ms
