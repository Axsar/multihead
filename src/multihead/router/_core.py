"""Head router: select the best head for a step based on requirements and system state."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..head_manager import HeadManager
    from ..observability import MetricsCollector
    from ..resilience import ResourceMonitor
    from ..registry.solver_registry import SolverRegistry
    from ..capability_discovery import CapabilityDiscovery

from ._scoring import ScoringMixin
from ._filtering import FilteringMixin
from ._knowledge import KnowledgeMixin
from ._discovery import DiscoveryMixin
from ._marketplace import MarketplaceMixin

logger = logging.getLogger(__name__)


class Router(
    ScoringMixin,
    FilteringMixin,
    KnowledgeMixin,
    DiscoveryMixin,
    MarketplaceMixin,
):
    """Select the best head_id for a step given system state.

    Scoring:
    1. Kind match — hard filter (must match required_kind)
    2. Already active — high weight (avoid GPU swap cost)
    3. Circuit breaker healthy — high weight (closed > half_open >> open)
    4. VRAM fits available — medium weight
    5. Lower error rate — medium weight
    6. Lower latency — low weight
    """

    def __init__(
        self,
        head_manager: HeadManager,
        metrics: MetricsCollector | None = None,
        resource_monitor: ResourceMonitor | None = None,
        registry: SolverRegistry | None = None,  # Phase 5: meta-reasoning preferences
        discovery: CapabilityDiscovery | None = None,  # Gap #4: dynamic capability discovery
        knowledge_store: Any | None = None,  # Feedback loop: read learned head preferences
        peer_registry: Any | None = None,  # v1.0: PeerRegistry for mesh routing
    ) -> None:
        self.heads = head_manager
        self.metrics = metrics
        self.resource_monitor = resource_monitor
        self.registry = registry
        self.discovery = discovery
        self.knowledge_store = knowledge_store
        self.peer_registry = peer_registry
        self._knowledge_cache: dict[str, float] = {}  # head_id -> boost (cached per session)
        self._knowledge_cache_built = False
        # Phase 3: Provider discovery cache (capability -> (providers, timestamp))
        self._provider_cache: dict[str, tuple[list[dict[str, Any]], float]] = {}
        self._provider_cache_ttl_s: float = 300.0  # 5 minutes

    def route(self, required_kind: str, *, exclude: set[str] | None = None) -> str | None:
        """Return the best head_id for the given kind, or None if no candidate.

        Args:
            required_kind: The kind of head needed ("llm", "vlm", etc.)
            exclude: Head IDs to skip (e.g., already-failed heads)
        """
        exclude = exclude or set()
        candidates = self._filter_candidates(required_kind, exclude)
        if not candidates:
            return None

        scored = [(hid, self._score(hid)) for hid in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        winner_id, winner_score = scored[0]
        logger.info(
            "Router selected %s (score=%.1f) for kind=%s from %d candidates",
            winner_id, winner_score, required_kind, len(candidates),
        )
        if logger.isEnabledFor(logging.DEBUG):
            for hid, score in scored:
                logger.debug("  %s: %.1f", hid, score)

        return winner_id

    def rank(self, required_kind: str) -> list[tuple[str, float]]:
        """Return all candidates ranked by score (for fallback ordering)."""
        candidates = self._filter_candidates(required_kind, set())
        scored = [(hid, self._score(hid)) for hid in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def rank_by_task(
        self,
        task_types: list[str],
        privacy: Any = None,
        exclude: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return all capable candidates ranked by score (for fallback ordering).

        Args:
            task_types: Required task types
            privacy: Privacy constraints
            exclude: Head IDs to skip

        Returns:
            List of (head_id, score) tuples sorted by score descending
        """
        exclude = exclude or set()
        candidates = self._filter_by_capability(task_types, privacy, exclude)
        scored = [(hid, self._score_with_capability(hid, task_types)) for hid in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def route_by_task(
        self,
        task_types: list[str],
        privacy: Any = None,  # PrivacyConstraint
        exclude: set[str] | None = None
    ) -> str | None:
        """Route based on task_types and privacy constraints (Phase 1).

        Args:
            task_types: List of task types needed (e.g., ["coordinate_transform"])
            privacy: Privacy constraints (DataSensitivity level, etc.)
            exclude: Head IDs to skip

        Returns:
            Best head_id or None if no capable solver found
        """
        exclude = exclude or set()
        candidates = self._filter_by_capability(task_types, privacy, exclude)
        if not candidates:
            return None

        scored = [(hid, self._score_with_capability(hid, task_types)) for hid in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        winner_id, winner_score = scored[0]
        logger.info(
            "Router selected %s (score=%.1f) for tasks=%s from %d candidates",
            winner_id, winner_score, task_types, len(candidates),
        )
        if logger.isEnabledFor(logging.DEBUG):
            for hid, score in scored:
                logger.debug("  %s: %.1f", hid, score)

        return winner_id

    def route_mesh(
        self,
        required_kind: str,
        *,
        task_types: list[str] | None = None,
        privacy: Any = None,
        exclude: set[str] | None = None,
    ) -> str | None:
        """Route across local heads AND remote mesh peers.

        Tries local routing first, then falls back to mesh peers.
        CONFIDENTIAL tasks never route to remote peers.

        Returns:
            Best head_id (local or mesh-*) or None if no candidate found.
        """
        exclude = exclude or set()

        # 1) Try local routing first
        if task_types:
            local = self.route_by_task(task_types, privacy=privacy, exclude=exclude)
        else:
            local = self.route(required_kind, exclude=exclude)
        if local:
            return local

        # 2) Privacy gate: CONFIDENTIAL stays local
        if privacy:
            from ..models import DataSensitivity
            if hasattr(privacy, "data_sensitivity") and privacy.data_sensitivity == DataSensitivity.CONFIDENTIAL:
                logger.info("Mesh routing blocked: CONFIDENTIAL data cannot leave local node")
                return None

        # 3) Check mesh peers
        if not self.peer_registry:
            return None

        peer_heads = self.peer_registry.peer_heads
        if not peer_heads:
            return None

        # Score remote peers: match kind, prefer lower latency
        candidates: list[tuple[str, float]] = []
        for head_id, ph in peer_heads.items():
            if head_id in exclude:
                continue
            if ph.status != "available":
                continue
            if ph.capability_kind != required_kind:
                continue

            score = 50.0  # base score for mesh peer
            # Latency penalty: lower is better (0-10 penalty)
            if ph.latency_ms > 0:
                score -= min(10.0, ph.latency_ms / 100.0)
            candidates.append((head_id, score))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1], reverse=True)
        winner_id, winner_score = candidates[0]
        logger.info(
            "Router mesh-selected %s (score=%.1f) for kind=%s from %d remote peers",
            winner_id, winner_score, required_kind, len(candidates),
        )
        return winner_id
