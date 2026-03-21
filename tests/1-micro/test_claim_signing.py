"""Tests for mesh claim HMAC-SHA256 signing."""

import pytest
from datetime import datetime, timezone

from multihead.knowledge_store import KnowledgeStore
from multihead.knowledge_models import (
    Claim,
    ClaimStatus,
    ClaimType,
    ClaimScope,
    ScopeType,
    ClaimCanonical,
    EntityRef,
    ValueObject,
    Provenance,
)
from multihead.mesh.security import MeshSecurity


@pytest.fixture
def mesh_security():
    """Create MeshSecurity with a test secret."""
    return MeshSecurity("test-secret-at-least-16-chars")


@pytest.fixture
def signed_store(tmp_path, mesh_security):
    """Create a KnowledgeStore with mesh signing enabled."""
    return KnowledgeStore(
        tmp_path / "signed.db",
        mesh_security=mesh_security,
        agent_id="test-agent",
    )


@pytest.fixture
def unsigned_store(tmp_path):
    """Create a KnowledgeStore without mesh signing."""
    return KnowledgeStore(tmp_path / "unsigned.db")


@pytest.fixture
def sample_claim():
    """Create a sample claim for testing."""
    return Claim(
        claim_status=ClaimStatus.ACCEPTED,
        claim_type=ClaimType.FACT,
        scope=ClaimScope(
            scope_type=ScopeType.PROJECT,
            scope_id="test",
            visibility="project",
            valid_from=datetime.now(timezone.utc),
        ),
        canonical=ClaimCanonical(
            claim_key="test.signing",
            subject=EntityRef(entity_type="module", entity_id="auth", label="Auth"),
            predicate="status",
            object=ValueObject(value_type="string", value="refactored"),
        ),
        statement="Auth module has been refactored",
        confidence=0.9,
        provenance=Provenance(produced_by={"id": "test-agent", "method": "manual"}),
    )


class TestClaimSigning:
    """Test HMAC-SHA256 claim signing on insert."""

    def test_auto_signs_on_insert(self, signed_store, sample_claim):
        """Claims should be auto-signed when mesh_security is configured."""
        assert sample_claim.signature == ""
        result = signed_store.insert_claim(sample_claim)
        assert result.signature != ""
        assert result.signed_by == "test-agent"

    def test_signature_is_hex_digest(self, signed_store, sample_claim):
        """Signature should be a valid hex string (64 chars for SHA-256)."""
        result = signed_store.insert_claim(sample_claim)
        assert len(result.signature) == 64
        assert all(c in "0123456789abcdef" for c in result.signature)

    def test_no_signature_without_security(self, unsigned_store, sample_claim):
        """Claims should remain unsigned when no mesh_security is set."""
        result = unsigned_store.insert_claim(sample_claim)
        assert result.signature == ""
        assert result.signed_by == ""

    def test_preserves_pre_signed_claims(self, signed_store, sample_claim):
        """If claim is already signed, don't re-sign."""
        sample_claim.signature = "pre-existing-signature"
        sample_claim.signed_by = "other-agent"
        result = signed_store.insert_claim(sample_claim)
        assert result.signature == "pre-existing-signature"
        assert result.signed_by == "other-agent"

    def test_signature_persisted_in_db(self, signed_store, sample_claim):
        """Signature should survive a round-trip through the database."""
        signed_store.insert_claim(sample_claim)
        loaded = signed_store.get_claim(sample_claim.claim_id)
        assert loaded is not None
        assert loaded.signature == sample_claim.signature
        assert loaded.signed_by == "test-agent"

    def test_different_claims_different_signatures(self, signed_store):
        """Two different claims should produce different signatures."""
        claim1 = Claim(
            claim_type=ClaimType.FACT,
            scope=ClaimScope(
                scope_type=ScopeType.PROJECT, scope_id="test",
                visibility="project", valid_from=datetime.now(timezone.utc),
            ),
            canonical=ClaimCanonical(
                claim_key="test.a", subject=EntityRef(entity_type="x", entity_id="1"),
                predicate="is", object=ValueObject(value_type="string", value="A"),
            ),
            statement="Claim A",
            provenance=Provenance(produced_by={"id": "agent"}),
        )
        claim2 = Claim(
            claim_type=ClaimType.FACT,
            scope=ClaimScope(
                scope_type=ScopeType.PROJECT, scope_id="test",
                visibility="project", valid_from=datetime.now(timezone.utc),
            ),
            canonical=ClaimCanonical(
                claim_key="test.b", subject=EntityRef(entity_type="x", entity_id="2"),
                predicate="is", object=ValueObject(value_type="string", value="B"),
            ),
            statement="Claim B",
            provenance=Provenance(produced_by={"id": "agent"}),
        )
        signed_store.insert_claim(claim1)
        signed_store.insert_claim(claim2)
        assert claim1.signature != claim2.signature


class TestClaimVerification:
    """Test claim signature verification."""

    def test_verify_valid_claim(self, signed_store, sample_claim):
        """Valid signed claim should verify True."""
        signed_store.insert_claim(sample_claim)
        loaded = signed_store.get_claim(sample_claim.claim_id)
        assert signed_store.verify_claim(loaded) is True

    def test_verify_tampered_statement(self, signed_store, sample_claim):
        """Claim with modified statement should verify False."""
        signed_store.insert_claim(sample_claim)
        loaded = signed_store.get_claim(sample_claim.claim_id)
        loaded.statement = "TAMPERED: this was changed"
        assert signed_store.verify_claim(loaded) is False

    def test_verify_tampered_claim_key(self, signed_store, sample_claim):
        """Claim with modified claim_key should verify False."""
        signed_store.insert_claim(sample_claim)
        loaded = signed_store.get_claim(sample_claim.claim_id)
        loaded.canonical.claim_key = "tampered.key"
        assert signed_store.verify_claim(loaded) is False

    def test_verify_unsigned_returns_none(self, signed_store, sample_claim):
        """Unsigned claim should return None (not verifiable)."""
        sample_claim.signature = ""
        assert signed_store.verify_claim(sample_claim) is None

    def test_verify_without_security_returns_none(self, unsigned_store, sample_claim):
        """Store without security should return None."""
        assert unsigned_store.verify_claim(sample_claim) is None

    def test_verify_wrong_secret_fails(self, signed_store, sample_claim, tmp_path):
        """Claim signed with one secret should fail verification with another."""
        signed_store.insert_claim(sample_claim)
        loaded = signed_store.get_claim(sample_claim.claim_id)

        other_security = MeshSecurity("different-secret-at-least-16")
        other_store = KnowledgeStore(
            tmp_path / "other.db",
            mesh_security=other_security,
            agent_id="other-agent",
        )
        assert other_store.verify_claim(loaded) is False


class TestCanonicalJsonForSigning:
    """Test deterministic JSON serialization for signing."""

    def test_canonical_json_is_deterministic(self, sample_claim):
        """Same claim should always produce same canonical JSON."""
        json1 = sample_claim.canonical_json_for_signing()
        json2 = sample_claim.canonical_json_for_signing()
        assert json1 == json2

    def test_canonical_json_is_sorted(self, sample_claim):
        """Keys should be sorted for deterministic output."""
        cj = sample_claim.canonical_json_for_signing()
        assert '"claim_id"' in cj
        assert '"claim_key"' in cj
        assert '"statement"' in cj
        # claim_id comes before claim_key alphabetically
        assert cj.index('"claim_id"') < cj.index('"claim_key"')

    def test_canonical_json_no_whitespace(self, sample_claim):
        """Compact JSON with no extra whitespace."""
        cj = sample_claim.canonical_json_for_signing()
        assert " : " not in cj
        assert ", " not in cj


class TestSchemaMigration:
    """Test that existing databases get signature columns added."""

    def test_migration_adds_columns(self, tmp_path):
        """Opening a pre-existing DB should add signature columns."""
        import sqlite3
        db_path = tmp_path / "old.db"
        # Create a DB with the old schema (no signature columns)
        conn = sqlite3.connect(db_path)
        conn.execute("""CREATE TABLE IF NOT EXISTS claims (
            claim_id TEXT PRIMARY KEY,
            claim_status TEXT NOT NULL,
            claim_type TEXT NOT NULL,
            scope_type TEXT NOT NULL,
            scope_id TEXT NOT NULL,
            visibility TEXT NOT NULL DEFAULT 'private',
            valid_from TEXT NOT NULL,
            valid_to TEXT,
            claim_key TEXT NOT NULL,
            predicate TEXT NOT NULL,
            subject_json TEXT NOT NULL,
            object_json TEXT NOT NULL,
            statement TEXT NOT NULL,
            rationale TEXT,
            confidence REAL,
            stability TEXT,
            importance REAL,
            superseded_by_claim_id TEXT,
            rejection_reason TEXT,
            contested_reason TEXT,
            derived_from_json TEXT NOT NULL DEFAULT '[]',
            related_json TEXT NOT NULL DEFAULT '[]',
            conflicts_json TEXT NOT NULL DEFAULT '[]',
            provenance_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""")
        conn.commit()
        cols_before = {r[1] for r in conn.execute("PRAGMA table_info(claims)").fetchall()}
        assert "signature" not in cols_before
        conn.close()

        # Opening with KnowledgeStore should migrate
        store = KnowledgeStore(db_path)
        conn2 = sqlite3.connect(db_path)
        cols_after = {r[1] for r in conn2.execute("PRAGMA table_info(claims)").fetchall()}
        conn2.close()
        assert "signature" in cols_after
        assert "signed_by" in cols_after
