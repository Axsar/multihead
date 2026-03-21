"""Dynamic capability discovery mixin for the Router (Gap #4)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class DiscoveryMixin:
    """Capability discovery methods for the Router.

    Requires: self.discovery, self.heads, self.route_by_task, self._score,
              self._passes_privacy_check
    """

    def route_with_discovery(
        self,
        capability_query: str,
        *,
        task_types: list[str] | None = None,
        privacy: Any = None,
        exclude: set[str] | None = None,
    ) -> str | None:
        """Route using dynamic capability discovery (Gap #4 fix).

        Queries CapabilityDiscovery for matching capabilities from:
        1. Local registry (acp_state.json)
        2. Knowledge base (model performance data)

        Then maps to local head_ids and scores using Router's existing logic.

        Args:
            capability_query: Capability query (ID prefix, semantic query, or kind)
                             e.g., "com.multihead.llm", "YOLO detection", "vlm"
            task_types: Optional task types for additional filtering
            privacy: Privacy constraints
            exclude: Head IDs to skip

        Returns:
            Best head_id or None if no match found

        Example:
            # Query by capability ID
            head_id = router.route_with_discovery("com.multihead.llm")

            # Semantic query
            head_id = router.route_with_discovery("YOLO detection")

            # Query by kind with task types
            head_id = router.route_with_discovery(
                "vlm",
                task_types=["image_understanding"],
                privacy=PrivacyConstraint(data_sensitivity="confidential")
            )
        """
        if not self.discovery:
            # No discovery available, fall back to traditional routing
            if task_types:
                return self.route_by_task(task_types, privacy, exclude)
            else:
                logger.warning("No discovery available and no task_types provided for fallback")
                return None

        exclude = exclude or set()

        # Query discovery for matching capabilities
        matches = self.discovery.query(capability_query, limit=50)

        # Filter matches to only local heads that exist in HeadManager
        local_head_ids = set(self.heads.get_states().keys())
        candidates: dict[str, Any] = {}  # head_id -> capability_match

        for match in matches:
            # Only consider local heads (not external agents like claude-session-agent)
            if match.source != "local":
                # Knowledge base matches might reference models we have locally
                # Try to map model name to head_id
                head_id = self._map_model_to_head(match.model)
                if head_id and head_id in local_head_ids:
                    candidates[head_id] = match
                continue

            # Extract head_id from capability_id: com.multihead.{kind}.{head_id}
            parts = match.capability_id.split(".")
            if len(parts) >= 4 and parts[0] == "com" and parts[1] == "multihead":
                head_id = parts[3]
                if head_id in local_head_ids and head_id not in exclude:
                    candidates[head_id] = match

        if not candidates:
            logger.info(
                "Discovery found %d matches but none map to local heads for query: %s",
                len(matches), capability_query
            )
            # Fall back to traditional routing if we have task_types
            if task_types:
                return self.route_by_task(task_types, privacy, exclude)
            return None

        # Apply privacy filtering if specified
        if privacy:
            filtered = {}
            for head_id, match in candidates.items():
                manifest = self.heads.get_manifest(head_id)
                if manifest and self._passes_privacy_check(manifest, privacy):
                    filtered[head_id] = match
            candidates = filtered

        if not candidates:
            logger.info("No candidates passed privacy filtering")
            return None

        # Score candidates using existing Router logic + discovery metadata
        scored = []
        for head_id, match in candidates.items():
            base_score = self._score(head_id)

            # Boost score based on discovery match quality
            discovery_boost = match.match_score * 10  # 0-10 points

            # Add performance data boost if available
            if match.performance:
                # Higher performance = higher score (max +10 points)
                perf_values = [v for v in match.performance.values() if isinstance(v, (int, float))]
                if perf_values:
                    avg_perf = sum(perf_values) / len(perf_values)
                    discovery_boost += avg_perf * 10

            total_score = base_score + discovery_boost
            scored.append((head_id, total_score, match))

        scored.sort(key=lambda x: x[1], reverse=True)

        winner_id, winner_score, winner_match = scored[0]
        logger.info(
            "Router selected %s (score=%.1f, discovery_match=%.2f) for capability=%s from %d candidates",
            winner_id, winner_score, winner_match.match_score, capability_query, len(candidates),
        )
        if logger.isEnabledFor(logging.DEBUG):
            for head_id, score, match in scored[:5]:
                logger.debug(
                    "  %s: %.1f (match=%.2f, kind=%s, model=%s)",
                    head_id, score, match.match_score, match.kind, match.model
                )

        return winner_id

    def _map_model_to_head(self, model_name: str) -> str | None:
        """Map model name from knowledge base to local head_id.

        Args:
            model_name: Model name (e.g., "YOLOv8m", "Qwen/Qwen3-8B")

        Returns:
            head_id if found, None otherwise
        """
        # Check if any local head uses this model
        for head_id, info in self.heads.get_states().items():
            manifest = self.heads.get_manifest(head_id)
            if manifest and manifest.model == model_name:
                return head_id

        # Fuzzy matching for common cases
        model_lower = model_name.lower()
        if "qwen3-8b" in model_lower or "qwen/qwen3-8b" in model_lower:
            return "qwen-llm" if "qwen-llm" in self.heads.get_states() else None
        if "qwen3-vl" in model_lower:
            if "32b" in model_lower or "thinking" in model_lower:
                return "qwen-vlm" if "qwen-vlm" in self.heads.get_states() else None
            if "8b" in model_lower:
                return "qwen-vlm-8b" if "qwen-vlm-8b" in self.heads.get_states() else None

        return None
