"""Priority-ordered evidence fusion for code narrative.

Combines multiple evidence sources into a coherent narrative unit,
respecting source priority and handling conflicts non-destructively.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from multihead.knowledge_models import (
    Claim,
    ClaimStatus,
    KnowledgeEvent,
    Record,
)
from multihead.narrative.confidence import ConfidenceCalibrator, SourcePriority
from multihead.narrative.verification import NarrativeVerifier, VerificationResult

logger = logging.getLogger(__name__)


@dataclass
class FusionBundle:
    """All evidence for one narrative unit (session, PR, time window).

    This is the input to fusion — everything the pipeline has extracted
    from all source types, ready to be combined.
    """

    unit_id: str  # Unique ID for this narrative unit
    unit_type: str  # "session", "pr", "sprint", "day"
    time_range: tuple[datetime, datetime] | None = None

    # Evidence from all extractors, grouped by source
    records: list[Record] = field(default_factory=list)
    events: list[KnowledgeEvent] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)

    # Source attribution for each claim (claim_id → SourcePriority)
    claim_sources: dict[str, SourcePriority] = field(default_factory=dict)

    # Context from previous narrative units (for continuity)
    prior_context: list[str] = field(default_factory=list)


@dataclass
class Conflict:
    """A detected disagreement between sources."""

    conflict_id: str
    claim_a_id: str
    claim_b_id: str
    source_a: SourcePriority
    source_b: SourcePriority
    description: str
    resolution: str  # "a_wins", "b_wins", "forked", "unresolved"


@dataclass
class FusedNarrative:
    """Output of the fusion process."""

    unit_id: str
    unit_type: str
    accepted_claims: list[Claim] = field(default_factory=list)
    contested_claims: list[Claim] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    verification: VerificationResult | None = None
    summary: str = ""

    @property
    def total_claims(self) -> int:
        return len(self.accepted_claims) + len(self.contested_claims)


class NarrativeFusion:
    """Fuse evidence from multiple sources into a coherent narrative.

    The fusion process:
    1. Group claims by claim_key (what they're about)
    2. For each group, resolve conflicts using priority ordering
    3. Verify the fused result
    4. Output accepted claims + detected conflicts
    """

    def __init__(self):
        self.calibrator = ConfidenceCalibrator()
        self.verifier = NarrativeVerifier()

    def fuse(self, bundle: FusionBundle) -> FusedNarrative:
        """Fuse a bundle of evidence into a coherent narrative.

        Args:
            bundle: FusionBundle containing all evidence for one unit.

        Returns:
            FusedNarrative with accepted/contested claims and conflicts.
        """
        result = FusedNarrative(
            unit_id=bundle.unit_id,
            unit_type=bundle.unit_type,
        )

        # Step 1: Group claims by claim_key
        groups = self._group_by_key(bundle.claims, bundle.claim_sources)

        # Step 2: Resolve each group
        for claim_key, group in groups.items():
            resolved = self._resolve_group(
                claim_key, group, bundle.claim_sources, result,
            )
            result.accepted_claims.extend(resolved["accepted"])
            result.contested_claims.extend(resolved["contested"])

        # Step 3: Add claims without conflicts (single-source claims)
        singleton_keys = {
            c.canonical.claim_key
            for c in bundle.claims
            if c.canonical.claim_key not in groups
        }
        for claim in bundle.claims:
            key = claim.canonical.claim_key
            if key in singleton_keys:
                result.accepted_claims.append(claim)
                singleton_keys.discard(key)

        # Step 4: Apply corroboration bonuses
        self._apply_corroboration(result, bundle.claim_sources)

        # Step 5: Verify
        result.verification = self.verifier.verify_batch(
            result.accepted_claims + result.contested_claims,
            bundle.events,
            bundle.records,
        )

        # Step 6: Generate summary
        result.summary = self._generate_summary(result)

        return result

    def _group_by_key(
        self,
        claims: list[Claim],
        sources: dict[str, SourcePriority],
    ) -> dict[str, list[Claim]]:
        """Group claims by claim_key, only groups with >1 claim."""
        groups: dict[str, list[Claim]] = {}
        for claim in claims:
            key = claim.canonical.claim_key
            groups.setdefault(key, []).append(claim)

        # Only return groups with conflicts (>1 claim)
        return {k: v for k, v in groups.items() if len(v) > 1}

    def _resolve_group(
        self,
        claim_key: str,
        claims: list[Claim],
        sources: dict[str, SourcePriority],
        result: FusedNarrative,
    ) -> dict[str, list[Claim]]:
        """Resolve conflicts within a group of claims on the same key."""
        resolved: dict[str, list[Claim]] = {"accepted": [], "contested": []}

        if len(claims) == 1:
            resolved["accepted"].append(claims[0])
            return resolved

        # Check if claims agree (same value)
        values = {str(c.canonical.object.value) for c in claims}
        if len(values) == 1:
            # All agree — accept highest-priority, mark others as corroboration
            by_priority = sorted(
                claims,
                key=lambda c: sources.get(c.claim_id, SourcePriority.AGENT_INTERPRETATION),
            )
            resolved["accepted"].append(by_priority[0])
            return resolved

        # Claims disagree — resolve by priority
        by_priority = sorted(
            claims,
            key=lambda c: (
                sources.get(c.claim_id, SourcePriority.AGENT_INTERPRETATION),
                -c.confidence,
            ),
        )

        winner = by_priority[0]
        losers = by_priority[1:]

        winner_source = sources.get(winner.claim_id, SourcePriority.AGENT_INTERPRETATION)

        for loser in losers:
            loser_source = sources.get(loser.claim_id, SourcePriority.AGENT_INTERPRETATION)

            resolution = self.calibrator.resolve_conflict(
                winner_source, winner.confidence,
                loser_source, loser.confidence,
            )

            conflict = Conflict(
                conflict_id=f"cnf_{winner.claim_id[:8]}_{loser.claim_id[:8]}",
                claim_a_id=winner.claim_id,
                claim_b_id=loser.claim_id,
                source_a=winner_source,
                source_b=loser_source,
                description=(
                    f"Conflict on '{claim_key}': "
                    f"{winner_source.name} says '{str(winner.canonical.object.value)[:60]}' vs "
                    f"{loser_source.name} says '{str(loser.canonical.object.value)[:60]}'"
                ),
                resolution=f"{'a' if resolution == 'a' else 'b' if resolution == 'b' else resolution}_wins",
            )
            result.conflicts.append(conflict)

            if resolution == "fork":
                # Too close to call — keep both as contested
                resolved["contested"].append(loser)
                result.open_questions.append(
                    f"Unresolved conflict on '{claim_key}': "
                    f"need more evidence to decide"
                )
            else:
                loser.claim_status = ClaimStatus.CONTESTED
                loser.contested_reason = (
                    f"Overridden by {winner_source.name} "
                    f"(priority {winner_source.value} vs {loser_source.value})"
                )
                resolved["contested"].append(loser)

        resolved["accepted"].append(winner)
        return resolved

    def _apply_corroboration(
        self,
        result: FusedNarrative,
        sources: dict[str, SourcePriority],
    ) -> None:
        """Boost confidence for claims corroborated by multiple sources."""
        # Group accepted claims by statement similarity
        statement_groups: dict[str, list[Claim]] = {}
        for claim in result.accepted_claims:
            key = claim.statement.lower().strip()[:100]
            statement_groups.setdefault(key, []).append(claim)

        for key, group in statement_groups.items():
            if len(group) > 1:
                corroborating = [
                    sources.get(c.claim_id, SourcePriority.AGENT_INTERPRETATION)
                    for c in group[1:]
                ]
                for claim in group:
                    source = sources.get(
                        claim.claim_id, SourcePriority.AGENT_INTERPRETATION,
                    )
                    cal = self.calibrator.calibrate(
                        claim.confidence, source, corroborating,
                    )
                    claim.confidence = cal.calibrated

    def _generate_summary(self, result: FusedNarrative) -> str:
        """Generate a brief summary of the fusion result."""
        parts = [
            f"Narrative unit '{result.unit_id}' ({result.unit_type}):",
            f"  {len(result.accepted_claims)} accepted claims",
            f"  {len(result.contested_claims)} contested claims",
            f"  {len(result.conflicts)} conflicts detected",
        ]
        if result.open_questions:
            parts.append(f"  {len(result.open_questions)} open questions")
        if result.verification:
            parts.append(f"  Verification: {result.verification.summary}")
        return "\n".join(parts)
