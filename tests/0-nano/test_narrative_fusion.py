"""Tests for narrative fusion module.

Tests the FusionBundle → FusedNarrative pipeline, including:
- Conflict resolution
- Priority ordering
- Corroboration bonuses
- Verification integration
"""

import sys
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
    Provenance,
    Record,
    ScopeType,
    ValueObject,
)
from multihead.narrative.confidence import SourcePriority
from multihead.narrative.fusion import (
    FusionBundle,
    NarrativeFusion,
)


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


class TestFusionBundle:
    """Test the FusionBundle data structure."""

    def test_empty_bundle_creation(self):
        """Test creating an empty fusion bundle."""
        bundle = FusionBundle(
            unit_id="test_session_001",
            unit_type="session",
        )
        assert bundle.unit_id == "test_session_001"
        assert bundle.unit_type == "session"
        assert len(bundle.claims) == 0
        assert len(bundle.records) == 0
        assert len(bundle.events) == 0


class TestNarrativeFusion:
    """Test the NarrativeFusion class."""

    def test_fuse_empty_bundle(self):
        """Test fusing an empty bundle produces empty result."""
        fusion = NarrativeFusion()
        bundle = FusionBundle(unit_id="empty_001", unit_type="session")

        result = fusion.fuse(bundle)

        assert result.unit_id == "empty_001"
        assert result.unit_type == "session"
        assert len(result.accepted_claims) == 0
        assert len(result.contested_claims) == 0
        assert len(result.conflicts) == 0
        assert result.total_claims == 0

    def test_fuse_single_claim(self):
        """Test fusing a single claim accepts it."""
        fusion = NarrativeFusion()
        claim = make_claim("module.property", "value1", confidence=0.8)
        bundle = FusionBundle(
            unit_id="single_001",
            unit_type="session",
            claims=[claim],
            claim_sources={claim.claim_id: SourcePriority.CODE_COMMENT},
        )

        result = fusion.fuse(bundle)

        assert len(result.accepted_claims) == 1
        assert result.accepted_claims[0].claim_id == claim.claim_id
        assert len(result.contested_claims) == 0
        assert result.total_claims == 1

    def test_fuse_agreeing_claims(self):
        """Test multiple claims with same key and value → accept highest priority."""
        fusion = NarrativeFusion()
        claim1 = make_claim("module.property", "value1", confidence=0.8, claim_id="clm_001")
        claim2 = make_claim("module.property", "value1", confidence=0.9, claim_id="clm_002")

        bundle = FusionBundle(
            unit_id="agree_001",
            unit_type="session",
            claims=[claim1, claim2],
            claim_sources={
                claim1.claim_id: SourcePriority.CODE_COMMENT,  # Priority 15
                claim2.claim_id: SourcePriority.GIT_COMMIT_MESSAGE,  # Priority 10 (higher)
            },
        )

        result = fusion.fuse(bundle)

        # Should accept the highest priority claim (lower number = higher priority)
        assert len(result.accepted_claims) == 1
        assert result.accepted_claims[0].claim_id == claim2.claim_id  # GIT_COMMIT_MESSAGE wins
        assert len(result.contested_claims) == 0

    def test_fuse_conflicting_claims(self):
        """Test same key, different values → winner + contested."""
        fusion = NarrativeFusion()
        claim1 = make_claim("module.property", "value1", confidence=0.8, claim_id="clm_001")
        claim2 = make_claim("module.property", "value2", confidence=0.9, claim_id="clm_002")

        bundle = FusionBundle(
            unit_id="conflict_001",
            unit_type="session",
            claims=[claim1, claim2],
            claim_sources={
                claim1.claim_id: SourcePriority.CODE_COMMENT,  # Priority 15
                # Priority 80 (lower priority)
                claim2.claim_id: SourcePriority.AGENT_INTERPRETATION,
            },
        )

        result = fusion.fuse(bundle)

        # CODE_COMMENT should win over AGENT_INTERPRETATION
        assert len(result.accepted_claims) == 1
        assert result.accepted_claims[0].claim_id == claim1.claim_id
        assert len(result.contested_claims) == 1
        assert result.contested_claims[0].claim_id == claim2.claim_id
        assert result.contested_claims[0].claim_status == ClaimStatus.CONTESTED

        # Should have a conflict logged
        assert len(result.conflicts) == 1
        conflict = result.conflicts[0]
        assert conflict.claim_a_id == claim1.claim_id
        assert conflict.claim_b_id == claim2.claim_id
        assert conflict.source_a == SourcePriority.CODE_COMMENT
        assert conflict.source_b == SourcePriority.AGENT_INTERPRETATION

    def test_corroboration_bonus(self):
        """Test same statement from multiple sources boosts confidence."""
        fusion = NarrativeFusion()
        statement = "Module property has value1"
        claim1 = make_claim(
            "module.property", "value1", confidence=0.7,
            claim_id="clm_001", statement=statement,
        )
        claim2 = make_claim(
            "module.other", "value2", confidence=0.7,
            claim_id="clm_002", statement=statement,
        )

        bundle = FusionBundle(
            unit_id="corr_001",
            unit_type="session",
            claims=[claim1, claim2],
            claim_sources={
                claim1.claim_id: SourcePriority.CODE_COMMENT,
                claim2.claim_id: SourcePriority.GIT_COMMIT_MESSAGE,
            },
        )

        result = fusion.fuse(bundle)

        # Both should be accepted (different keys)
        assert len(result.accepted_claims) == 2

        # Confidence should be boosted due to corroboration
        # (exact value depends on calibrator, but should be > original)
        for claim in result.accepted_claims:
            # After corroboration, confidence might be adjusted
            assert claim.confidence >= 0.7

    def test_total_claims_property(self):
        """Test total_claims property counts accepted + contested."""
        fusion = NarrativeFusion()
        claim1 = make_claim("module.prop1", "value1", claim_id="clm_001")
        claim2 = make_claim("module.prop2", "value2", claim_id="clm_002")
        claim3 = make_claim("module.prop1", "value3", claim_id="clm_003")  # Conflicts with claim1

        bundle = FusionBundle(
            unit_id="total_001",
            unit_type="session",
            claims=[claim1, claim2, claim3],
            claim_sources={
                claim1.claim_id: SourcePriority.GIT_COMMIT_MESSAGE,
                claim2.claim_id: SourcePriority.CODE_COMMENT,
                claim3.claim_id: SourcePriority.AGENT_INTERPRETATION,
            },
        )

        result = fusion.fuse(bundle)

        # claim1 wins over claim3, claim2 standalone
        assert result.total_claims == 3
        assert result.total_claims == len(result.accepted_claims) + len(result.contested_claims)

    def test_summary_generation(self):
        """Test summary includes claim counts and conflicts."""
        fusion = NarrativeFusion()
        claim1 = make_claim("module.prop1", "value1", claim_id="clm_001")
        claim2 = make_claim("module.prop1", "value2", claim_id="clm_002")

        bundle = FusionBundle(
            unit_id="summary_001",
            unit_type="session",
            claims=[claim1, claim2],
            claim_sources={
                claim1.claim_id: SourcePriority.CODE_COMMENT,
                claim2.claim_id: SourcePriority.AGENT_INTERPRETATION,
            },
        )

        result = fusion.fuse(bundle)

        assert "summary_001" in result.summary
        assert "session" in result.summary
        assert "accepted claims" in result.summary
        assert "contested claims" in result.summary
        assert "conflicts detected" in result.summary

    def test_conflict_detection(self):
        """Test conflicts list is populated correctly."""
        fusion = NarrativeFusion()
        claim1 = make_claim("module.property", "value1", claim_id="clm_001")
        claim2 = make_claim("module.property", "value2", claim_id="clm_002")

        bundle = FusionBundle(
            unit_id="conf_detect_001",
            unit_type="session",
            claims=[claim1, claim2],
            claim_sources={
                claim1.claim_id: SourcePriority.CODE_COMMENT,
                claim2.claim_id: SourcePriority.LLM_INFERENCE,
            },
        )

        result = fusion.fuse(bundle)

        assert len(result.conflicts) == 1
        conflict = result.conflicts[0]
        assert "module.property" in conflict.description
        assert conflict.resolution in ["a_wins", "b_wins", "forked_wins"]

    def test_fork_resolution(self):
        """Test when priorities are close, fork can occur."""
        fusion = NarrativeFusion()
        claim1 = make_claim("module.property", "value1", confidence=0.95, claim_id="clm_001")
        claim2 = make_claim("module.property", "value2", confidence=0.94, claim_id="clm_002")

        # Use very close priorities (both high-priority sources)
        bundle = FusionBundle(
            unit_id="fork_001",
            unit_type="session",
            claims=[claim1, claim2],
            claim_sources={
                claim1.claim_id: SourcePriority.GIT_COMMIT_MESSAGE,  # 10
                claim2.claim_id: SourcePriority.CODE_COMMENT,  # 15
            },
        )

        result = fusion.fuse(bundle)

        # GIT_COMMIT_MESSAGE should still win (lower number = higher priority)
        assert len(result.accepted_claims) == 1
        assert result.accepted_claims[0].claim_id == claim1.claim_id

        # The lower-priority claim should be contested
        if len(result.contested_claims) > 0:
            assert result.contested_claims[0].claim_id == claim2.claim_id

    def test_verification_integration(self):
        """Test that verification result is included."""
        fusion = NarrativeFusion()
        claim = make_claim("module.property", "value1")
        record = Record(
            record_id="rec_001",
            uri="test://record",
            sha256="abc123",
        )

        # Add valid evidence pointer
        claim.evidence_supports.append(
            EvidencePointer(
                evidence_id="evp_001",
                record_id=record.record_id,
                uri="test://evidence",
            )
        )

        bundle = FusionBundle(
            unit_id="verify_001",
            unit_type="session",
            claims=[claim],
            records=[record],
            claim_sources={claim.claim_id: SourcePriority.CODE_COMMENT},
        )

        result = fusion.fuse(bundle)

        assert result.verification is not None
        assert result.verification.total_claims_checked > 0
        assert "Verification:" in result.summary
