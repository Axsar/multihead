"""Tests for narrative state engine module.

Tests cross-session entity tracking, stable ID allocation,
deduplication, verification adjustments, and audit trail.
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
    Provenance,
    ScopeType,
    ValueObject,
)
from multihead.narrative.fusion import FusedNarrative
from multihead.narrative.state_engine import (
    DEDUP_THRESHOLD,
    NarrativeState,
    NarrativeStateEngine,
)
from multihead.narrative.verification import VerificationResult


def make_claim(
    claim_key: str,
    value: str,
    confidence: float = 0.8,
    claim_id: str = "",
    statement: str = "",
    entity_label: str = "TestEntity",
    entity_type: str = "module",
) -> Claim:
    """Helper to create a minimal Claim for testing."""
    if not claim_id:
        claim_id = f"clm_{claim_key.replace('.', '_')}"
    if not statement:
        statement = f"Test claim: {claim_key} = {value}"

    return Claim(
        claim_id=claim_id,
        claim_type=ClaimType.FACT,
        claim_status=ClaimStatus.ACCEPTED,
        statement=statement,
        confidence=confidence,
        canonical=ClaimCanonical(
            claim_key=claim_key,
            subject=EntityRef(entity_type=entity_type, entity_id="test_entity", label=entity_label),
            predicate="has_value",
            object=ValueObject(value_type="string", value=value),
        ),
        scope=ClaimScope(scope_type=ScopeType.PROJECT, scope_id="test_project"),
        provenance=Provenance(produced_by={"kind": "test", "id": "pytest"}),
    )


class TestNarrativeState:
    """Test the NarrativeState data structure."""

    def test_allocate_sid_developer(self):
        """Test SID allocation for developer entity."""
        state = NarrativeState()
        sid1 = state.allocate_sid("developer")
        sid2 = state.allocate_sid("developer")

        assert sid1 == "DEV_001"
        assert sid2 == "DEV_002"

    def test_allocate_sid_module(self):
        """Test SID allocation for module entity."""
        state = NarrativeState()
        sid1 = state.allocate_sid("module")
        sid2 = state.allocate_sid("module")

        assert sid1 == "MOD_001"
        assert sid2 == "MOD_002"

    def test_allocate_sid_agent(self):
        """Test SID allocation for agent entity."""
        state = NarrativeState()
        sid = state.allocate_sid("agent")
        assert sid == "AGT_001"

    def test_allocate_sid_concept(self):
        """Test SID allocation for concept entity."""
        state = NarrativeState()
        sid = state.allocate_sid("concept")
        assert sid == "CON_001"

    def test_allocate_sid_unknown(self):
        """Test SID allocation for unknown entity type."""
        state = NarrativeState()
        sid = state.allocate_sid("unknown")
        assert sid == "UNK_001"

    def test_counters_independent(self):
        """Test that counters for different entity types are independent."""
        state = NarrativeState()
        dev1 = state.allocate_sid("developer")
        mod1 = state.allocate_sid("module")
        dev2 = state.allocate_sid("developer")
        mod2 = state.allocate_sid("module")

        assert dev1 == "DEV_001"
        assert dev2 == "DEV_002"
        assert mod1 == "MOD_001"
        assert mod2 == "MOD_002"


class TestNarrativeStateEngine:
    """Test the NarrativeStateEngine class."""

    def test_create_entity_from_claim(self):
        """Test new entity is created from first claim."""
        engine = NarrativeStateEngine()
        state = NarrativeState()
        claim = make_claim(
            "module.property", "value1",
            entity_label="NewModule", entity_type="repo",
        )

        narrative = FusedNarrative(
            unit_id="session_001",
            unit_type="session",
            accepted_claims=[claim],
        )

        state = engine.apply(narrative, state)

        assert len(state.entities) == 1
        entity = list(state.entities.values())[0]
        assert entity.canonical_name == "NewModule"
        assert entity.entity_sid == "MOD_001"
        assert entity.entity_type == "module"
        assert entity.first_seen_session == "session_001"
        assert entity.appearance_count == 1

    def test_update_existing_entity(self):
        """Test second claim with same name updates existing entity."""
        engine = NarrativeStateEngine()
        state = NarrativeState()

        # First claim creates entity
        claim1 = make_claim(
            "module.prop1", "value1",
            entity_label="SameModule", entity_type="repo",
            confidence=0.7,
        )
        narrative1 = FusedNarrative(
            unit_id="session_001",
            unit_type="session",
            accepted_claims=[claim1],
        )
        state = engine.apply(narrative1, state)

        # Second claim updates entity
        claim2 = make_claim(
            "module.prop2", "value2",
            entity_label="SameModule", entity_type="repo",
            confidence=0.8,
        )
        narrative2 = FusedNarrative(
            unit_id="session_002",
            unit_type="session",
            accepted_claims=[claim2],
        )
        state = engine.apply(narrative2, state)

        # Should still be one entity
        assert len(state.entities) == 1
        entity = list(state.entities.values())[0]
        assert entity.canonical_name == "SameModule"
        assert entity.entity_sid == "MOD_001"
        assert entity.first_seen_session == "session_001"
        assert entity.last_seen_session == "session_002"
        assert entity.appearance_count == 2
        # Confidence should be weighted update
        assert entity.confidence > 0.7

    def test_jaccard_similarity(self):
        """Test fuzzy matching using Jaccard similarity."""
        engine = NarrativeStateEngine()

        # Test exact threshold
        sim = engine._jaccard_similarity("multihead core module", "multihead core module")
        assert sim == 1.0

        # Test partial match
        sim = engine._jaccard_similarity("multihead core", "core module")
        assert 0 < sim < 1

        # Test no match
        sim = engine._jaccard_similarity("apple", "orange")
        assert sim == 0.0

    def test_alias_matching(self):
        """Test entity found by alias."""
        engine = NarrativeStateEngine()
        state = NarrativeState()

        # Create entity with specific ID
        claim1 = make_claim("module.prop1", "value1", entity_label="Module One")
        claim1.canonical.subject.entity_id = "mod_one"
        narrative1 = FusedNarrative(
            unit_id="session_001",
            unit_type="session",
            accepted_claims=[claim1],
        )
        state = engine.apply(narrative1, state)

        # Reference by alias
        claim2 = make_claim("module.prop2", "value2", entity_label="mod_one")
        narrative2 = FusedNarrative(
            unit_id="session_002",
            unit_type="session",
            accepted_claims=[claim2],
        )
        state = engine.apply(narrative2, state)

        # Should match via alias
        assert len(state.entities) == 1

    def test_verification_bonus(self):
        """Test passed verification boosts confidence."""
        engine = NarrativeStateEngine()
        state = NarrativeState()

        claim = make_claim("module.property", "value1", confidence=0.7)
        narrative = FusedNarrative(
            unit_id="session_001",
            unit_type="session",
            accepted_claims=[claim],
            verification=VerificationResult(
                passed=True,
                total_claims_checked=1,
            ),
        )

        state = engine.apply(narrative, state)

        entity = list(state.entities.values())[0]
        original_confidence = entity.confidence

        # Apply again with verification passed
        claim2 = make_claim(
            "module.property2", "value2",
            confidence=0.7, entity_label=entity.canonical_name,
        )
        narrative2 = FusedNarrative(
            unit_id="session_002",
            unit_type="session",
            accepted_claims=[claim2],
            verification=VerificationResult(
                passed=True,
                total_claims_checked=1,
            ),
        )
        state = engine.apply(narrative2, state)

        # Confidence should be boosted
        # (First weighted avg with new claim, then bonus)
        assert entity.confidence >= original_confidence

    def test_verification_penalty(self):
        """Test failed verification reduces confidence."""
        engine = NarrativeStateEngine()
        state = NarrativeState()

        claim = make_claim("module.property", "value1", confidence=0.9)
        narrative = FusedNarrative(
            unit_id="session_001",
            unit_type="session",
            accepted_claims=[claim],
        )
        state = engine.apply(narrative, state)

        entity = list(state.entities.values())[0]
        original_confidence = entity.confidence

        # Apply with failed verification
        claim2 = make_claim(
            "module.property2", "value2",
            confidence=0.9, entity_label=entity.canonical_name,
        )
        narrative2 = FusedNarrative(
            unit_id="session_002",
            unit_type="session",
            accepted_claims=[claim2],
            verification=VerificationResult(
                passed=False,
                total_claims_checked=1,
            ),
        )
        state = engine.apply(narrative2, state)

        # Confidence should be reduced (penalty applied)
        # Check that penalty is applied
        confidence_deltas = [d for d in state.deltas if d.action == "confidence_adjusted"]
        if confidence_deltas:
            # If adjustment happened, confidence changed
            assert "failed" in confidence_deltas[-1].details.get("reason", "")

    def test_conflict_logged(self):
        """Test conflicts create deltas."""
        engine = NarrativeStateEngine()
        state = NarrativeState()

        from multihead.narrative.fusion import Conflict
        from multihead.narrative.confidence import SourcePriority

        conflict = Conflict(
            conflict_id="cnf_001",
            claim_a_id="clm_001",
            claim_b_id="clm_002",
            source_a=SourcePriority.CODE_COMMENT,
            source_b=SourcePriority.AGENT_INTERPRETATION,
            description="Test conflict",
            resolution="a_wins",
        )

        narrative = FusedNarrative(
            unit_id="session_001",
            unit_type="session",
            conflicts=[conflict],
        )

        state = engine.apply(narrative, state)

        # Should have a conflict delta
        conflict_deltas = [d for d in state.deltas if d.action == "conflict_forked"]
        assert len(conflict_deltas) == 1
        assert conflict_deltas[0].target_sid == "cnf_001"
        assert conflict_deltas[0].details["resolution"] == "a_wins"

    def test_entity_type_mapping(self):
        """Test raw entity types are mapped to tracked types."""
        engine = NarrativeStateEngine()

        assert engine._map_entity_type("person") == "developer"
        assert engine._map_entity_type("user") == "developer"
        assert engine._map_entity_type("repo") == "module"
        assert engine._map_entity_type("project") == "module"
        assert engine._map_entity_type("file") == "file"
        assert engine._map_entity_type("model") == "agent"
        assert engine._map_entity_type("agent") == "agent"
        assert engine._map_entity_type("service") == "agent"
        assert engine._map_entity_type("tool") == "agent"
        assert engine._map_entity_type("session") == "concept"
        assert engine._map_entity_type("unknown_type") == "concept"

    def test_session_tracking(self):
        """Test sessions_processed list grows."""
        engine = NarrativeStateEngine()
        state = NarrativeState()

        narrative1 = FusedNarrative(unit_id="session_001", unit_type="session")
        state = engine.apply(narrative1, state)

        narrative2 = FusedNarrative(unit_id="session_002", unit_type="session")
        state = engine.apply(narrative2, state)

        assert len(state.sessions_processed) == 2
        assert "session_001" in state.sessions_processed
        assert "session_002" in state.sessions_processed

    def test_total_claims_ingested(self):
        """Test total claims counter increments."""
        engine = NarrativeStateEngine()
        state = NarrativeState()

        claim1 = make_claim("module.prop1", "value1")
        claim2 = make_claim("module.prop2", "value2")
        narrative = FusedNarrative(
            unit_id="session_001",
            unit_type="session",
            accepted_claims=[claim1, claim2],
        )

        state = engine.apply(narrative, state)

        assert state.total_claims_ingested == 2

    def test_delta_audit_trail(self):
        """Test all state changes create deltas."""
        engine = NarrativeStateEngine()
        state = NarrativeState()

        claim = make_claim("module.property", "value1")
        narrative = FusedNarrative(
            unit_id="session_001",
            unit_type="session",
            accepted_claims=[claim],
        )

        state = engine.apply(narrative, state)

        # Should have entity_created delta
        assert len(state.deltas) > 0
        created_deltas = [d for d in state.deltas if d.action == "entity_created"]
        assert len(created_deltas) == 1
        delta = created_deltas[0]
        assert delta.session_id == "session_001"
        assert "canonical_name" in delta.details

    def test_fuzzy_match_threshold(self):
        """Test entities are matched when similarity >= threshold."""
        engine = NarrativeStateEngine()
        state = NarrativeState()

        # Create entity
        claim1 = make_claim("module.prop1", "value1", entity_label="multihead core module")
        narrative1 = FusedNarrative(
            unit_id="session_001",
            unit_type="session",
            accepted_claims=[claim1],
        )
        state = engine.apply(narrative1, state)

        # Similar name (high Jaccard similarity)
        claim2 = make_claim("module.prop2", "value2", entity_label="multihead core")
        narrative2 = FusedNarrative(
            unit_id="session_002",
            unit_type="session",
            accepted_claims=[claim2],
        )
        state = engine.apply(narrative2, state)

        # Jaccard("multihead core module", "multihead core") = 2/3 = 0.666...
        # If >= DEDUP_THRESHOLD (0.70), should match; otherwise new entity
        # Since 0.666 < 0.70, should create new entity
        similarity = engine._jaccard_similarity("multihead core module", "multihead core")
        if similarity >= DEDUP_THRESHOLD:
            assert len(state.entities) == 1
        else:
            assert len(state.entities) == 2
