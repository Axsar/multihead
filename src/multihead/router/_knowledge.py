"""Knowledge-based feedback mixin for the Router."""

from __future__ import annotations

import logging

from ._scoring import _W_PREFERENCE

logger = logging.getLogger(__name__)


class KnowledgeMixin:
    """Knowledge feedback methods for the Router (Step 10 feedback loop).

    Requires: self.knowledge_store, self._knowledge_cache, self._knowledge_cache_built
    """

    def _get_knowledge_boost(self, head_id: str) -> float:
        """Look up success/failure claims for a head and return a score adjustment.

        Queries knowledge.db for claims matching ``head.{head_id}.*`` patterns.
        Success claims with confidence >= 0.8 give +10, failure claims give -10.
        Results are cached for the lifetime of the Router instance.

        Returns:
            Score adjustment (-10 to +10), 0 if no knowledge store or no claims found.
        """
        if not self.knowledge_store:
            return 0.0

        # Build cache on first call
        if not self._knowledge_cache_built:
            self._build_knowledge_cache()

        return self._knowledge_cache.get(head_id, 0.0)

    def _build_knowledge_cache(self) -> None:
        """One-time scan of knowledge.db for head performance claims.

        Looks for claims with keys like ``head.{head_id}.*`` and adjusts
        the cache based on success/failure patterns.
        """
        self._knowledge_cache_built = True
        if not self.knowledge_store:
            return

        try:
            # Query for accepted claims about head performance
            claims = self.knowledge_store.list_claims(
                status="accepted",
                claim_type="fact",
                limit=500,
            )

            for claim in claims:
                key = claim.claim_key
                # Match patterns like head.{head_id}.success or head.{head_id}.failure
                if not key.startswith("head."):
                    continue

                parts = key.split(".")
                if len(parts) < 3:
                    continue

                hid = parts[1]
                suffix = parts[2]

                if suffix in ("success", "compatible_with") and claim.confidence >= 0.8:
                    self._knowledge_cache[hid] = self._knowledge_cache.get(hid, 0.0) + _W_PREFERENCE
                elif suffix in ("failure", "error", "incompatible_with"):
                    self._knowledge_cache[hid] = self._knowledge_cache.get(hid, 0.0) - _W_PREFERENCE

            # Clamp values to [-10, +10]
            for hid in self._knowledge_cache:
                self._knowledge_cache[hid] = max(-10.0, min(10.0, self._knowledge_cache[hid]))

            if self._knowledge_cache:
                logger.info(
                    "Knowledge cache built: %d heads with feedback",
                    len(self._knowledge_cache),
                )

        except Exception as e:
            logger.warning("Failed to build knowledge cache: %s", e)
