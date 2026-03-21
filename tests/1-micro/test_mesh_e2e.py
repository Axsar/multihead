"""End-to-end integration tests for the mesh protocol."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from multihead.adapters.mesh_adapter import MeshHeadAdapter
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
from multihead.knowledge_store import KnowledgeStore
from multihead.mesh.failover import MeshFailoverPolicy
from multihead.mesh.peer_registry import PeerHead, PeerRegistry
from multihead.mesh.security import MeshSecurity, NodeIdentity, TrustStore
from multihead.models import AdapterKind, HeadManifest
from multihead.resilience import CircuitBreaker
from multihead.router import Router


def _make_claim(
    claim_id: str,
    claim_key: str,
    statement: str,
    visibility: str = "shared",
    scope_id: str = "mesh-test",
) -> Claim:
    return Claim(
        claim_id=claim_id,
        claim_status=ClaimStatus.PROPOSED,
        claim_type=ClaimType.FACT,
        scope=ClaimScope(
            scope_type=ScopeType.PROJECT,
            scope_id=scope_id,
            visibility=visibility,
        ),
        canonical=ClaimCanonical(
            claim_key=claim_key,
            subject=EntityRef(entity_type="test", entity_id="e2e"),
            predicate="is",
            object=ValueObject(value_type="string", value="true"),
        ),
        statement=statement,
        confidence=0.9,
        provenance=Provenance(produced_by={"kind": "agent", "id": "e2e-test"}),
    )


# -----------------------------------------------------------------------
# Two-node simulation
# -----------------------------------------------------------------------


class TestTwoNodeSync:
    """Simulate two nodes with separate knowledge stores syncing claims."""

    @pytest.fixture
    def node_a(self, tmp_path):
        """Node A with its own knowledge store."""
        return KnowledgeStore(db_path=tmp_path / "node_a.db")

    @pytest.fixture
    def node_b(self, tmp_path):
        """Node B with its own knowledge store."""
        return KnowledgeStore(db_path=tmp_path / "node_b.db")

    def test_claim_inserted_on_one_node_available_on_other(self, node_a, node_b):
        """Claims from node A should be importable to node B."""
        claim = _make_claim("clm_e2e_1", "e2e.fact", "Node A knows something")
        node_a.insert_claim(claim)

        # Simulate "peer fetch" — node B reads A's shared claims
        shared = node_a.get_shared_claims_since()
        assert len(shared) == 1

        # Import to node B
        for c in shared:
            if not node_b.get_claim(c.claim_id):
                node_b.insert_claim(c)

        # Verify on node B
        imported = node_b.get_claim("clm_e2e_1")
        assert imported is not None
        assert imported.statement == "Node A knows something"

    def test_private_claims_stay_local(self, node_a, node_b):
        """Private claims should not be returned by get_shared_claims_since."""
        private = _make_claim(
            "clm_e2e_priv", "e2e.private", "Secret!", visibility="private"
        )
        node_a.insert_claim(private)

        shared = node_a.get_shared_claims_since()
        assert len(shared) == 0

    def test_signed_claims_verify_on_peer(self, tmp_path):
        """Signed claims should verify on peer nodes with same shared secret."""
        secret = "mesh-shared-secret-abc"
        store_a = KnowledgeStore(
            db_path=tmp_path / "signed_a.db",
            mesh_security=MeshSecurity(shared_secret=secret),
            agent_id="node-a",
        )
        store_b = KnowledgeStore(
            db_path=tmp_path / "signed_b.db",
            mesh_security=MeshSecurity(shared_secret=secret),
            agent_id="node-b",
        )

        claim = _make_claim("clm_e2e_signed", "e2e.signed", "Signed fact")
        store_a.insert_claim(claim)
        assert claim.signature != ""

        # Verify on peer B
        result = store_b.verify_claim(claim)
        assert result is True


# -----------------------------------------------------------------------
# Router mesh integration
# -----------------------------------------------------------------------


class TestRouterMeshIntegration:
    """Test full route_mesh flow with real objects."""

    def test_route_local_then_mesh(self):
        """Should prefer local head, fall back to mesh peer."""
        manager = Mock()
        manager._manifests = {
            "local-vlm": HeadManifest(
                head_id="local-vlm", name="Local VLM",
                adapter=AdapterKind.MOCK, model="test", kind="vlm",
            ),
        }
        manager._states = {"local-vlm": "active"}
        manager._breakers = {"local-vlm": CircuitBreaker()}
        manager._adapters = {}
        manager.get_states.return_value = {}
        manager.get_manifest.return_value = None
        manager.get_breaker.return_value = None

        peer_reg = Mock(spec=PeerRegistry)
        peer_reg.peer_heads = {
            "mesh-peer-llm": PeerHead(
                head_id="mesh-peer-llm", node_id="peer1",
                peer_url="http://peer:7337", capability_kind="llm",
                status="available", latency_ms=30.0,
            ),
        }

        router = Router(head_manager=manager, peer_registry=peer_reg)

        # Request LLM — no local LLM, should get mesh peer
        with patch.object(router, "route", return_value=None):
            result = router.route_mesh("llm")
        assert result == "mesh-peer-llm"

    def test_privacy_gate_blocks_mesh(self):
        """Should block CONFIDENTIAL tasks from mesh routing."""
        from multihead.models import DataSensitivity

        manager = Mock()
        manager._manifests = {}
        manager._states = {}
        manager._breakers = {}
        manager._adapters = {}
        manager.get_states.return_value = {}
        manager.get_manifest.return_value = None
        manager.get_breaker.return_value = None

        peer_reg = Mock(spec=PeerRegistry)
        peer_reg.peer_heads = {
            "mesh-peer-llm": PeerHead(
                head_id="mesh-peer-llm", node_id="peer1",
                peer_url="http://peer:7337", capability_kind="llm",
                status="available",
            ),
        }

        router = Router(head_manager=manager, peer_registry=peer_reg)
        privacy = Mock()
        privacy.data_sensitivity = DataSensitivity.CONFIDENTIAL

        with patch.object(router, "route", return_value=None):
            result = router.route_mesh("llm", privacy=privacy)
        assert result is None


# -----------------------------------------------------------------------
# Failover E2E
# -----------------------------------------------------------------------


class TestFailoverE2E:
    """Test failover with real CircuitBreaker objects."""

    def test_fallback_after_circuit_breaker_trips(self):
        """Should exclude circuit-broken heads and route to mesh peer."""
        manager = Mock()
        breaker = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            breaker.record_failure()
        assert breaker.state == "open"

        manager._manifests = {
            "local-llm": HeadManifest(
                head_id="local-llm", name="LLM",
                adapter=AdapterKind.MOCK, model="test", kind="llm",
            ),
        }
        manager.get_breaker = lambda hid: breaker if hid == "local-llm" else None

        peer_reg = Mock(spec=PeerRegistry)
        peer_reg.peer_heads = {
            "mesh-backup-llm": PeerHead(
                head_id="mesh-backup-llm", node_id="backup",
                peer_url="http://backup:7337", capability_kind="llm",
                status="available", latency_ms=50.0,
            ),
        }

        policy = MeshFailoverPolicy(
            head_manager=manager,
            peer_registry=peer_reg,
        )

        fallbacks = policy.get_fallbacks("failed-head", required_kind="llm")
        # local-llm should be excluded (circuit broken)
        assert "local-llm" not in fallbacks
        # mesh backup should be available
        assert "mesh-backup-llm" in fallbacks


# -----------------------------------------------------------------------
# Security E2E
# -----------------------------------------------------------------------


class TestSecurityE2E:
    """Test Ed25519 identity + trust store flow."""

    def test_full_identity_verification_flow(self, tmp_path):
        """Node A signs request, Node B verifies using trust store."""
        # Node A generates key
        key_a = tmp_path / "node_a_key.pem"
        identity_a = NodeIdentity(private_key_path=key_a, node_id="node-a")

        # Node B's trust store has Node A's public key
        trust_b = TrustStore()
        trust_b.add_peer("node-a", public_key=identity_a.public_key_b64, trusted=True)

        security_b = MeshSecurity(
            shared_secret="shared",
            trust_store=trust_b,
        )

        # Node A creates identity headers
        security_a = MeshSecurity(
            shared_secret="shared",
            node_identity=identity_a,
        )
        headers = security_a.sign_node_identity(timestamp="1234567890")

        # Node B verifies
        result = security_b.verify_node_identity(
            node_id=headers["X-Node-ID"],
            timestamp=headers["X-Node-Timestamp"],
            signature=headers["X-Node-Signature"],
        )
        assert result is True

    def test_untrusted_node_rejected(self, tmp_path):
        """Should reject identity from untrusted node."""
        key_path = tmp_path / "untrusted_key.pem"
        identity = NodeIdentity(private_key_path=key_path, node_id="untrusted")

        trust = TrustStore()  # Empty — no trusted peers
        security = MeshSecurity(shared_secret="shared", trust_store=trust)

        ts = "123"
        message = f"untrusted:{ts}"
        sig = identity.sign(message)

        result = security.verify_node_identity("untrusted", ts, sig)
        assert result is False

    def test_trust_store_persistence(self, tmp_path):
        """Trust decisions should persist across restarts."""
        yaml_path = tmp_path / "peers.yaml"

        # Session 1: add trust
        store1 = TrustStore(path=yaml_path)
        store1.add_peer("node-x", public_key="key-abc", trusted=True)
        store1.save()

        # Session 2: reload
        store2 = TrustStore(path=yaml_path)
        assert store2.is_trusted("node-x") is True
        assert store2.get_public_key("node-x") == "key-abc"


# -----------------------------------------------------------------------
# MeshHeadAdapter E2E
# -----------------------------------------------------------------------


class TestMeshAdapterE2E:
    """Test MeshHeadAdapter end-to-end flow."""

    async def test_generate_through_adapter(self):
        """Should route generate call to remote peer and return result."""
        manifest = HeadManifest(
            head_id="mesh-test-llm",
            name="Test LLM",
            adapter=AdapterKind.MESH,
            model="qwen3:8b",
            kind="llm",
            gpu_required=False,
            is_local=False,
            extra={
                "peer_url": "http://10.0.0.1:7337",
                "peer_node_id": "test-desktop",
                "auth_token": "token-abc",
            },
        )
        adapter = MeshHeadAdapter(manifest)

        mock_result = {
            "task_id": "t-1",
            "status": "completed",
            "result": "Hello from remote peer!",
            "head_id": "core-llm",
        }
        with patch.object(
            adapter._client, "submit_task",
            new_callable=AsyncMock, return_value=mock_result,
        ):
            result = await adapter.generate("Say hello")

        assert result["text"] == "Hello from remote peer!"
        assert result["peer_node_id"] == "test-desktop"
