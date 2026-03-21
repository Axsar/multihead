"""Deterministic verification checks for narrative claims.

Five mandatory checks run on every claim batch before storage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from multihead.knowledge_models import Claim, ClaimStatus, KnowledgeEvent, Record
from multihead.narrative.confidence import (
    EPISTEMIC_CEILING,
    UNKNOWN_THRESHOLD,
    ConfidenceCalibrator,
)

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Result of a single verification check."""

    check_name: str
    passed: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    corrections_applied: int = 0


@dataclass
class VerificationResult:
    """Result of full verification pass."""

    passed: bool = True
    checks: list[CheckResult] = field(default_factory=list)
    total_claims_checked: int = 0
    claims_corrected: int = 0
    claims_rejected: int = 0

    @property
    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        failed = [c.check_name for c in self.checks if not c.passed]
        return (
            f"[{status}] {self.total_claims_checked} claims checked, "
            f"{self.claims_corrected} corrected, {self.claims_rejected} rejected"
            + (f" | Failed: {', '.join(failed)}" if failed else "")
        )


class NarrativeVerifier:
    """Deterministic verification of narrative claims.

    Runs 5 deterministic checks:
    1. Citation Integrity — all evidence_refs resolve to real records
    2. Confidence Clamping — enforce [0, EPISTEMIC_CEILING] bounds
    3. Consistency — no self-contradicting claims in batch
    4. Completeness — required fields present
    5. Deduplication — detect near-duplicate claims
    """

    def __init__(self):
        self.calibrator = ConfidenceCalibrator()

    def verify_batch(
        self,
        claims: list[Claim],
        events: list[KnowledgeEvent],
        records: list[Record],
    ) -> VerificationResult:
        """Run all verification checks on a batch of artifacts.

        Args:
            claims: Claims to verify.
            events: Events associated with claims.
            records: Records that evidence pointers should reference.

        Returns:
            VerificationResult with per-check details.
        """
        record_ids = {r.record_id for r in records}
        result = VerificationResult(total_claims_checked=len(claims))

        # Check 1: Citation Integrity
        result.checks.append(
            self._check_citation_integrity(claims, events, record_ids)
        )

        # Check 2: Confidence Clamping
        check2 = self._check_confidence_bounds(claims)
        result.checks.append(check2)
        result.claims_corrected += check2.corrections_applied

        # Check 3: Consistency
        result.checks.append(self._check_consistency(claims))

        # Check 4: Completeness
        check4 = self._check_completeness(claims)
        result.checks.append(check4)
        result.claims_rejected += len(check4.errors)

        # Check 5: Deduplication
        result.checks.append(self._check_deduplication(claims))

        result.passed = all(c.passed for c in result.checks)
        return result

    def _check_citation_integrity(
        self,
        claims: list[Claim],
        events: list[KnowledgeEvent],
        record_ids: set[str],
    ) -> CheckResult:
        """Check 1: All evidence pointers reference existing records."""
        check = CheckResult(check_name="citation_integrity", passed=True)

        for claim in claims:
            for ep in claim.evidence_supports + claim.evidence_opposes:
                if ep.record_id not in record_ids:
                    check.warnings.append(
                        f"Claim {claim.claim_id}: evidence {ep.evidence_id} "
                        f"references unknown record {ep.record_id}"
                    )

        for event in events:
            for ep in event.evidence_supports:
                if ep.record_id not in record_ids:
                    check.warnings.append(
                        f"Event {event.event_id}: evidence {ep.evidence_id} "
                        f"references unknown record {ep.record_id}"
                    )

        if len(check.warnings) > 0:
            check.passed = False

        return check

    def _check_confidence_bounds(self, claims: list[Claim]) -> CheckResult:
        """Check 2: Enforce confidence bounds and epistemic ceiling."""
        check = CheckResult(check_name="confidence_bounds", passed=True)

        for claim in claims:
            original = claim.confidence

            # Clamp to [0, 1]
            if claim.confidence < 0.0:
                claim.confidence = 0.0
                check.corrections_applied += 1
                check.warnings.append(
                    f"Claim {claim.claim_id}: confidence {original} clamped to 0.0"
                )
            elif claim.confidence > 1.0:
                claim.confidence = 1.0
                check.corrections_applied += 1
                check.warnings.append(
                    f"Claim {claim.claim_id}: confidence {original} clamped to 1.0"
                )

            # Epistemic ceiling
            if claim.confidence > EPISTEMIC_CEILING:
                claim.confidence = EPISTEMIC_CEILING
                check.corrections_applied += 1
                check.warnings.append(
                    f"Claim {claim.claim_id}: confidence {original} "
                    f"capped at epistemic ceiling {EPISTEMIC_CEILING}"
                )

            # Mark low-confidence claims
            if claim.confidence < UNKNOWN_THRESHOLD:
                check.warnings.append(
                    f"Claim {claim.claim_id}: confidence {claim.confidence:.2f} "
                    f"below unknown threshold ({UNKNOWN_THRESHOLD})"
                )

        return check

    def _check_consistency(self, claims: list[Claim]) -> CheckResult:
        """Check 3: Detect conflicting claims within the same batch."""
        check = CheckResult(check_name="consistency", passed=True)

        # Group claims by claim_key
        by_key: dict[str, list[Claim]] = {}
        for claim in claims:
            key = claim.canonical.claim_key
            by_key.setdefault(key, []).append(claim)

        # Flag duplicates on same key with different values
        for key, group in by_key.items():
            if len(group) > 1:
                values = set()
                for c in group:
                    val = str(c.canonical.object.value)
                    if val in values:
                        check.warnings.append(
                            f"Duplicate claim_key '{key}' with same value"
                        )
                    values.add(val)

                if len(values) > 1:
                    check.warnings.append(
                        f"Conflicting claims on key '{key}': "
                        f"{len(values)} different values from {len(group)} claims"
                    )
                    # Link conflicts
                    for i, c1 in enumerate(group):
                        for c2 in group[i + 1:]:
                            if str(c1.canonical.object.value) != str(c2.canonical.object.value):
                                c1.conflicts_with_claim_ids.append(c2.claim_id)
                                c2.conflicts_with_claim_ids.append(c1.claim_id)

        if any("Conflicting" in w for w in check.warnings):
            check.passed = False

        return check

    def _check_completeness(self, claims: list[Claim]) -> CheckResult:
        """Check 4: Required fields are present and valid."""
        check = CheckResult(check_name="completeness", passed=True)

        for claim in claims:
            if not claim.statement:
                check.errors.append(
                    f"Claim {claim.claim_id}: missing statement"
                )
            if not claim.canonical.claim_key:
                check.errors.append(
                    f"Claim {claim.claim_id}: missing claim_key"
                )
            if not claim.evidence_supports:
                check.warnings.append(
                    f"Claim {claim.claim_id}: no supporting evidence"
                )

        if check.errors:
            check.passed = False

        return check

    def _check_deduplication(self, claims: list[Claim]) -> CheckResult:
        """Check 5: Detect near-duplicate claims."""
        check = CheckResult(check_name="deduplication", passed=True)

        seen_statements: dict[str, str] = {}  # normalized statement → claim_id
        for claim in claims:
            normalized = claim.statement.lower().strip()[:200]
            if normalized in seen_statements:
                check.warnings.append(
                    f"Claim {claim.claim_id} appears to duplicate "
                    f"{seen_statements[normalized]}: '{normalized[:60]}...'"
                )
            else:
                seen_statements[normalized] = claim.claim_id

        return check
