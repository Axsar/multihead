"""Candidate filtering mixins for the Router."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models import HeadManifest

logger = logging.getLogger(__name__)


class FilteringMixin:
    """Filtering methods for the Router.

    Requires: self.heads
    """

    def _filter_candidates(self, required_kind: str, exclude: set[str]) -> list[str]:
        """Return head_ids matching kind and not excluded/broken."""
        candidates = []
        for head_id, info in self.heads.get_states().items():
            if head_id in exclude:
                continue
            if info["kind"] != required_kind:
                continue
            # Hard filter: skip heads with open circuit breaker
            breaker = self.heads.get_breaker(head_id)
            if breaker and breaker.state == "open":
                continue
            candidates.append(head_id)
        return candidates

    def _filter_by_capability(
        self,
        task_types: list[str],
        privacy: Any,  # PrivacyConstraint | None
        exclude: set[str]
    ) -> list[str]:
        """Filter heads by capability match and privacy constraints (Phase 1).

        Args:
            task_types: Required task types
            privacy: Privacy constraints (or None)
            exclude: Head IDs to skip

        Returns:
            List of capable head_ids
        """
        from ..models import DataSensitivity

        candidates = []

        for head_id, info in self.heads.get_states().items():
            if head_id in exclude:
                continue

            # Hard filter: circuit breaker
            breaker = self.heads.get_breaker(head_id)
            if breaker and breaker.state == "open":
                continue

            manifest = self.heads.get_manifest(head_id)
            if not manifest:
                continue

            # Phase 1: Check capability match
            if manifest.capabilities:
                caps = manifest.capabilities
                # Check if solver can handle ANY of the task types
                if not any(t in caps.task_types for t in task_types):
                    continue  # Can't do any of the required tasks
            else:
                # Backward compatibility: no capabilities defined
                # Fall back to kind matching if no task_types provided
                continue

            # Privacy filter
            if privacy:
                # Confidential data: local only
                if privacy.data_sensitivity == DataSensitivity.CONFIDENTIAL:
                    if not manifest.is_local:
                        continue

                # Internal data: local or encrypted
                if privacy.data_sensitivity == DataSensitivity.INTERNAL:
                    if not manifest.is_local and manifest.privacy_level != "encrypted":
                        continue

                # Whitelist/blacklist
                if privacy.allowed_providers:
                    if head_id not in privacy.allowed_providers:
                        continue
                if privacy.blocked_providers:
                    if head_id in privacy.blocked_providers:
                        continue

            candidates.append(head_id)

        return candidates

    def _passes_privacy_check(self, manifest: HeadManifest, privacy: Any) -> bool:
        """Check if manifest satisfies privacy constraints.

        Args:
            manifest: Head manifest to check
            privacy: PrivacyConstraint

        Returns:
            True if passes, False otherwise
        """
        from ..models import DataSensitivity

        if not privacy:
            return True

        # Confidential: local only
        if privacy.data_sensitivity == DataSensitivity.CONFIDENTIAL:
            if not manifest.is_local:
                return False

        # Internal: local or encrypted
        if privacy.data_sensitivity == DataSensitivity.INTERNAL:
            if not manifest.is_local and manifest.privacy_level != "encrypted":
                return False

        # Whitelist/blacklist (would need head_id, skip for now in this helper)
        # Those are checked in the main filtering loop

        return True
