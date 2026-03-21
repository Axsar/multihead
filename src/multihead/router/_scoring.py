"""Scoring weights and scoring mixins for the Router."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Scoring weights (higher = more important)
_W_CAPABILITY = 40   # Capability match (Phase 1)
_W_ACTIVE = 40       # Already loaded — avoid GPU swap cost
_W_BREAKER = 30      # Circuit breaker health
_W_ACCURACY = 20     # Higher accuracy score (Phase 1)
_W_VRAM_FIT = 15     # VRAM fits available memory
_W_ERROR_RATE = 10   # Lower recent error rate
_W_COST = 10         # Lower cost per call (Phase 1)
_W_LATENCY = 5       # Lower recent latency
_W_PREFERENCE = 5    # Phase 5: Learned preference boost


class ScoringMixin:
    """Scoring methods for the Router.

    Requires: self.heads, self.metrics, self.resource_monitor, self.registry
    """

    def _score(self, head_id: str) -> float:
        """Score a candidate head (higher is better)."""
        score = 0.0
        manifest = self.heads.get_manifest(head_id)
        states = self.heads.get_states()
        info = states.get(head_id, {})

        # Factor 1: Already active (avoid swap cost)
        if info.get("is_active"):
            score += _W_ACTIVE

        # Factor 2: Circuit breaker health
        breaker = self.heads.get_breaker(head_id)
        if breaker:
            match breaker.state:
                case "closed":
                    score += _W_BREAKER
                case "half_open":
                    score += _W_BREAKER * 0.3
                # "open" already filtered out in _filter_candidates

        # Factor 3: VRAM fit
        if manifest and self.resource_monitor:
            try:
                resources = self.resource_monitor.check()
                if not manifest.gpu_required:
                    score += _W_VRAM_FIT
                elif manifest.vram_hint_mb > 0 and resources.gpu_vram_free_mb > 0:
                    if resources.gpu_vram_free_mb >= manifest.vram_hint_mb:
                        score += _W_VRAM_FIT
                    else:
                        ratio = resources.gpu_vram_free_mb / manifest.vram_hint_mb
                        score += _W_VRAM_FIT * ratio * 0.5
            except Exception:
                pass  # Resource check failed, skip this factor

        # Factor 4: Error rate (lower is better)
        if self.metrics:
            gen_total = self.metrics.counter(
                "head_generate_total", labels={"head_id": head_id},
            )
            err_total = self.metrics.counter(
                "head_generate_errors_total", labels={"head_id": head_id},
            )
            if gen_total > 0:
                error_rate = err_total / gen_total
                score += _W_ERROR_RATE * (1.0 - error_rate)
            else:
                score += _W_ERROR_RATE * 0.5  # No history, benefit of the doubt

        # Factor 5: Latency (lower is better)
        if self.metrics:
            hist = self.metrics.histogram(
                "head_generate_seconds", labels={"head_id": head_id},
            )
            if hist["count"] > 0:
                avg_latency = hist["avg"]
                # sub-1s = perfect, scales down linearly to 30s
                latency_score = max(0.0, 1.0 - avg_latency / 30.0)
                score += _W_LATENCY * latency_score
            else:
                score += _W_LATENCY * 0.5

        # Factor 6: Knowledge-based feedback (5 points)
        knowledge_boost = self._get_knowledge_boost(head_id)
        if knowledge_boost != 0.0:
            score += knowledge_boost

        return score

    def _score_with_capability(self, head_id: str, task_types: list[str]) -> float:
        """Score head with capability matching (Phase 1).

        Args:
            head_id: Head to score
            task_types: Required task types

        Returns:
            Score (0.0-100.0, higher is better)
        """
        score = 0.0
        manifest = self.heads.get_manifest(head_id)
        states = self.heads.get_states()
        info = states.get(head_id, {})

        if not manifest:
            return 0.0

        # Factor 1: Capability match (40 points)
        if manifest.capabilities:
            caps = manifest.capabilities
            # Count how many required tasks this solver can do
            matches = sum(1 for t in task_types if t in caps.task_types)
            total = len(task_types)
            capability_match = matches / total if total > 0 else 0.0
            score += _W_CAPABILITY * capability_match
        else:
            # No capability info = partial credit
            score += _W_CAPABILITY * 0.5

        # Factor 2: Accuracy (20 points)
        if manifest.capabilities and manifest.capabilities.accuracy_score is not None:
            score += _W_ACCURACY * manifest.capabilities.accuracy_score

        # Factor 3: Already active (40 points)
        if info.get("is_active"):
            score += _W_ACTIVE

        # Factor 4: Circuit breaker health (30 points)
        breaker = self.heads.get_breaker(head_id)
        if breaker:
            match breaker.state:
                case "closed":
                    score += _W_BREAKER
                case "half_open":
                    score += _W_BREAKER * 0.3

        # Factor 5: Cost efficiency (10 points)
        if manifest.capabilities and manifest.capabilities.cost_per_call is not None:
            # Lower cost = higher score
            # Free (0.0) = full points, $1+ = zero points
            cost = manifest.capabilities.cost_per_call
            cost_score = max(0.0, 1.0 - cost)  # $0 = 1.0, $1+ = 0.0
            score += _W_COST * cost_score
        else:
            # No cost info = assume moderate cost
            score += _W_COST * 0.5

        # Factor 6: VRAM fit (15 points)
        if manifest and self.resource_monitor:
            try:
                resources = self.resource_monitor.check()
                if not manifest.gpu_required:
                    score += _W_VRAM_FIT
                elif manifest.vram_hint_mb > 0 and resources.gpu_vram_free_mb > 0:
                    if resources.gpu_vram_free_mb >= manifest.vram_hint_mb:
                        score += _W_VRAM_FIT
                    else:
                        ratio = resources.gpu_vram_free_mb / manifest.vram_hint_mb
                        score += _W_VRAM_FIT * ratio * 0.5
            except Exception:
                pass

        # Factor 7: Error rate (10 points)
        if self.metrics:
            gen_total = self.metrics.counter(
                "head_generate_total", labels={"head_id": head_id},
            )
            err_total = self.metrics.counter(
                "head_generate_errors_total", labels={"head_id": head_id},
            )
            if gen_total > 0:
                error_rate = err_total / gen_total
                score += _W_ERROR_RATE * (1.0 - error_rate)
            else:
                score += _W_ERROR_RATE * 0.5

        # Factor 8: Latency (5 points)
        if manifest.capabilities and manifest.capabilities.latency_p50_ms is not None:
            latency_ms = manifest.capabilities.latency_p50_ms
            # <1ms = perfect, scales to 30s (30000ms)
            latency_score = max(0.0, 1.0 - latency_ms / 30000.0)
            score += _W_LATENCY * latency_score
        elif self.metrics:
            hist = self.metrics.histogram(
                "head_generate_seconds", labels={"head_id": head_id},
            )
            if hist["count"] > 0:
                avg_latency = hist["avg"]
                latency_score = max(0.0, 1.0 - avg_latency / 30.0)
                score += _W_LATENCY * latency_score
            else:
                score += _W_LATENCY * 0.5

        # Factor 9: Learned preference (Phase 5 - 5 points)
        # If meta-reasoning selected this solver for these task types, boost score
        if self.registry and task_types:
            for task_type in task_types:
                pref = self.registry.get_preference(task_type)
                if pref and pref["preferred_solver_id"] == head_id:
                    # Preference boost weighted by confidence
                    confidence = pref["confidence_score"]
                    score += _W_PREFERENCE * confidence
                    logger.debug(
                        "Preference boost for %s on %s (+%.1f from confidence=%.2f)",
                        head_id, task_type, _W_PREFERENCE * confidence, confidence
                    )
                    break  # Only apply boost once per head

        # Factor 10: Knowledge-based feedback (5 points)
        # Read success/failure claims from knowledge.db to boost/penalize heads
        knowledge_boost = self._get_knowledge_boost(head_id)
        if knowledge_boost != 0.0:
            score += knowledge_boost
            logger.debug(
                "Knowledge boost for %s: %+.1f",
                head_id, knowledge_boost,
            )

        return score
