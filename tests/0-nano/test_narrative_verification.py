"""Tests for narrative verification module.

Tests the five mandatory verification checks:
1. Citation Integrity
2. Confidence Clamping
3. Consistency
4. Completeness
5. Deduplication
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multihead.knowledge_models import (
    Claim,
    ClaimCanonical,
    ClaimScope,
    ClaimStatus,
    ClaimType,
    EntityRef,
    EvidencePointer,
    EventType,
    KnowledgeEvent,
    Provenance,
    Record,
    ScopeType,
    TimeBlock,
    ValueObject,
)
from multihead.narrative.confidence import EPISTEMIC_CEILING, UNKNOWN_THRESHOLD
from multihead.narrative.verification import NarrativeVerifier


def make_claim(
    claim_key: str,
    value: str,
    confidence: float = 0.8,
    claim_id: str = "",
    statement: str = "",
) -> Claim:
    """Helper to create a minimal Claim for testing."""
    if not claim_id:
        claim_id = f"clm_{claim_key.replace('.', '_')}"
    if not statement:
        statement = f"Test claim: {claim_key} = {value}"

    return Claim(
        claim_id=claim_id,
        claim_type=ClaimType.FACT,
        claim_status=ClaimStatus.PROPOSED,
        statement=statement,
        confidence=confidence,
        canonical=ClaimCanonical(
            claim_key=claim_key,
            subject=EntityRef(entity_type="module", entity_id="test_module", label="TestModule"),
            predicate="has_value",
            object=ValueObject(value_type="string", value=value),
        ),
        scope=ClaimScope(scope_type=ScopeType.PROJECT, scope_id="test_project"),
        provenance=Provenance(produced_by={"kind": "test", "id": "pytest"}),
    )


class TestNarrativeVerifier:
    """Test the NarrativeVerifier class."""

    def test_verify_empty_batch(self):
        """Test verifying an empty batch passes."""
        verifier = NarrativeVerifier()
        result = verifier.verify_batch(claims=[], events=[], records=[])

        assert result.passed is True
        assert result.total_claims_checked == 0
        assert result.claims_corrected == 0
        assert result.claims_rejected == 0
        assert len(result.checks) == 5  # All 5 checks run

    def test_citation_integrity_pass(self):
        """Test citation integrity passes when all evidence refs are valid."""
        verifier = NarrativeVerifier()
        record = Record(
            record_id="rec_001",
            uri="test://record",
            sha256="abc123",
        )
        claim = make_claim("module.property", "value1", claim_id="clm_001")
        claim.evidence_supports.append(
            EvidencePointer(
                evidence_id="evp_001",
                record_id=record.record_id,
                uri="test://evidence",
            )
        )

        result = verifier.verify_batch(claims=[claim], events=[], records=[record])

        citation_check = next(c for c in result.checks if c.check_name == "citation_integrity")
        assert citation_check.passed is True
        assert len(citation_check.warnings) == 0

    def test_citation_integrity_fail(self):
        """Test citation integrity fails when evidence references missing record."""
        verifier = NarrativeVerifier()
        claim = make_claim("module.property", "value1", claim_id="clm_001")
        # Reference a record that doesn't exist
        claim.evidence_supports.append(
            EvidencePointer(
                evidence_id="evp_001",
                record_id="rec_missing",
                uri="test://evidence",
            )
        )

        result = verifier.verify_batch(claims=[claim], events=[], records=[])

        citation_check = next(c for c in result.checks if c.check_name == "citation_integrity")
        assert citation_check.passed is False
        assert len(citation_check.warnings) > 0
        assert "rec_missing" in citation_check.warnings[0]

    def test_confidence_clamping_negative(self):
        """Test negative confidence is clamped to 0."""
        verifier = NarrativeVerifier()
        claim = make_claim("module.property", "value1", confidence=-0.5, claim_id="clm_001")

        result = verifier.verify_batch(claims=[claim], events=[], records=[])

        confidence_check = next(c for c in result.checks if c.check_name == "confidence_bounds")
        assert confidence_check.corrections_applied > 0
        assert claim.confidence == 0.0
        assert result.claims_corrected > 0
        assert any("-0.5" in w for w in confidence_check.warnings)

    def test_confidence_clamping_over_one(self):
        """Test confidence >1.0 is clamped to 1.0."""
        verifier = NarrativeVerifier()
        claim = make_claim("module.property", "value1", confidence=1.5, claim_id="clm_001")

        result = verifier.verify_batch(claims=[claim], events=[], records=[])

        confidence_check = next(c for c in result.checks if c.check_name == "confidence_bounds")
        assert confidence_check.corrections_applied > 0
        assert claim.confidence <= 1.0
        assert result.claims_corrected > 0
        assert any("1.5" in w for w in confidence_check.warnings)

    def test_confidence_epistemic_ceiling(self):
        """Test confidence >EPISTEMIC_CEILING is capped."""
        verifier = NarrativeVerifier()
        claim = make_claim("module.property", "value1", confidence=0.99, claim_id="clm_001")

        result = verifier.verify_batch(claims=[claim], events=[], records=[])

        confidence_check = next(c for c in result.checks if c.check_name == "confidence_bounds")
        assert claim.confidence <= EPISTEMIC_CEILING
        if claim.confidence == EPISTEMIC_CEILING:
            assert confidence_check.corrections_applied > 0
            assert any("epistemic ceiling" in w for w in confidence_check.warnings)

    def test_consistency_conflicting_values(self):
        """Test same key with different values is flagged as inconsistent."""
        verifier = NarrativeVerifier()
        claim1 = make_claim("module.property", "value1", claim_id="clm_001")
        claim2 = make_claim("module.property", "value2", claim_id="clm_002")

        result = verifier.verify_batch(claims=[claim1, claim2], events=[], records=[])

        consistency_check = next(c for c in result.checks if c.check_name == "consistency")
        assert consistency_check.passed is False
        assert any("Conflicting" in w for w in consistency_check.warnings)
        assert any("module.property" in w for w in consistency_check.warnings)

        # Claims should be linked as conflicts
        assert claim2.claim_id in claim1.conflicts_with_claim_ids
        assert claim1.claim_id in claim2.conflicts_with_claim_ids

    def test_completeness_missing_statement(self):
        """Test claim with empty statement is rejected."""
        verifier = NarrativeVerifier()
        claim = make_claim("module.property", "value1", statement="", claim_id="clm_001")
        claim.statement = ""  # Force empty statement

        result = verifier.verify_batch(claims=[claim], events=[], records=[])

        completeness_check = next(c for c in result.checks if c.check_name == "completeness")
        assert completeness_check.passed is False
        assert len(completeness_check.errors) > 0
        assert any("missing statement" in e for e in completeness_check.errors)
        assert result.claims_rejected > 0

    def test_completeness_missing_claim_key(self):
        """Test claim with empty claim_key is rejected."""
        verifier = NarrativeVerifier()
        claim = make_claim("module.property", "value1", claim_id="clm_001")
        claim.canonical.claim_key = ""  # Force empty key

        result = verifier.verify_batch(claims=[claim], events=[], records=[])

        completeness_check = next(c for c in result.checks if c.check_name == "completeness")
        assert completeness_check.passed is False
        assert len(completeness_check.errors) > 0
        assert any("missing claim_key" in e for e in completeness_check.errors)
        assert result.claims_rejected > 0

    def test_deduplication_duplicate_statements(self):
        """Test near-duplicate claims are flagged."""
        verifier = NarrativeVerifier()
        statement = "This is a test statement about module property"
        claim1 = make_claim("module.prop1", "value1", claim_id="clm_001", statement=statement)
        claim2 = make_claim("module.prop2", "value2", claim_id="clm_002", statement=statement)

        result = verifier.verify_batch(claims=[claim1, claim2], events=[], records=[])

        dedup_check = next(c for c in result.checks if c.check_name == "deduplication")
        assert len(dedup_check.warnings) > 0
        assert any("duplicate" in w.lower() for w in dedup_check.warnings)

    def test_all_checks_run(self):
        """Test that all 5 checks are always executed."""
        verifier = NarrativeVerifier()
        claim = make_claim("module.property", "value1")

        result = verifier.verify_batch(claims=[claim], events=[], records=[])

        assert len(result.checks) == 5
        check_names = {c.check_name for c in result.checks}
        assert check_names == {
            "citation_integrity",
            "confidence_bounds",
            "consistency",
            "completeness",
            "deduplication",
        }

    def test_verification_summary(self):
        """Test verification result summary format."""
        verifier = NarrativeVerifier()
        claim = make_claim("module.property", "value1")

        result = verifier.verify_batch(claims=[claim], events=[], records=[])

        assert "PASS" in result.summary or "FAIL" in result.summary
        assert "claims checked" in result.summary
        assert str(result.total_claims_checked) in result.summary

    def test_failed_check_in_summary(self):
        """Test failed checks appear in summary."""
        verifier = NarrativeVerifier()
        claim = make_claim("module.property", "value1", statement="")  # Will fail completeness
        claim.statement = ""

        result = verifier.verify_batch(claims=[claim], events=[], records=[])

        assert result.passed is False
        assert "FAIL" in result.summary
        assert "Failed:" in result.summary
        assert "completeness" in result.summary

    def test_event_citation_integrity(self):
        """Test citation integrity also checks events."""
        verifier = NarrativeVerifier()
        event = KnowledgeEvent(
            event_id="evt_001",
            event_type=EventType.COMMIT,
            title="Test event",
            time=TimeBlock(happened_at=datetime.now(timezone.utc)),
            provenance=Provenance(produced_by={"kind": "test", "id": "pytest"}),
        )
        # Add evidence pointing to missing record
        event.evidence_supports.append(
            EvidencePointer(
                evidence_id="evp_001",
                record_id="rec_missing",
                uri="test://evidence",
            )
        )

        result = verifier.verify_batch(claims=[], events=[event], records=[])

        citation_check = next(c for c in result.checks if c.check_name == "citation_integrity")
        assert citation_check.passed is False
        assert len(citation_check.warnings) > 0
        assert any("rec_missing" in w for w in citation_check.warnings)

    def test_low_confidence_warning(self):
        """Test claims below unknown threshold generate warnings."""
        verifier = NarrativeVerifier()
        claim = make_claim("module.property", "value1", confidence=UNKNOWN_THRESHOLD - 0.1)

        result = verifier.verify_batch(claims=[claim], events=[], records=[])

        confidence_check = next(c for c in result.checks if c.check_name == "confidence_bounds")
        # Should have warning about low confidence
        assert any("unknown threshold" in w.lower() for w in confidence_check.warnings)
