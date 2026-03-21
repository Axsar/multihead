"""Tests for KnowledgeSync and mesh claim replication."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest

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
from multihead.mesh.knowledge_sync import KnowledgeSync
from multihead.mesh.peer_registry import PeerHead, PeerRegistry


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _make_shared_claim(
    claim_id: str = "",
    claim_key: str = "test.key",
    statement: str = "Test claim",
    confidence: float = 0.8,
    visibility: str = "shared",
    scope_id: str = "test-project",
) -> Claim:
    """Create a minimal shared claim for testing."""
    return Claim(
        claim_id=claim_id or f"clm_test_{claim_key}",
        claim_status=ClaimStatus.ACCEPTED,
        claim_type=ClaimType.FACT,
        scope=ClaimScope(
            scope_type=ScopeType.PROJECT,
            scope_id=scope_id,
            visibility=visibility,
        ),
        canonical=ClaimCanonical(
            claim_key=claim_key,
            subject=EntityRef(entity_type="test", entity_id="s1"),
            predicate="is",
            object=ValueObject(value_type="string", value="true"),
        ),
        statement=statement,
        confidence=confidence,
        provenance=Provenance(produced_by={"kind": "agent", "id": "test-agent"}),
    )


@pytest.fixture
def knowledge_store(tmp_path):
    """Create a temporary knowledge store."""
    return KnowledgeStore(db_path=tmp_path / "test_knowledge.db")


@pytest.fixture
def mock_peer_registry():
    """Create a mock PeerRegistry with one available peer."""
    reg = Mock(spec=PeerRegistry)
    reg.peer_heads = {
        "mesh-peer1-llm": PeerHead(
            head_id="mesh-peer1-llm",
            node_id="peer1",
            peer_url="http://192.168.1.10:7337",
            capability_kind="llm",
            status="available",
        ),
    }
    return reg


@pytest.fixture
def sync(knowledge_store, mock_peer_registry):
    """Create a KnowledgeSync instance."""
    return KnowledgeSync(
        knowledge_store=knowledge_store,
        peer_registry=mock_peer_registry,
        sync_interval=10.0,
    )


# -----------------------------------------------------------------------
# KnowledgeStore.get_shared_claims_since tests
# -----------------------------------------------------------------------


class TestGetSharedClaimsSince:
    """Test the replication query method on KnowledgeStore."""

    def test_returns_only_shared_claims(self, knowledge_store):
        """Should only return claims with visibility='shared'."""
        shared = _make_shared_claim(
            claim_id="clm_shared1", claim_key="shared.key",
            visibility="shared",
        )
        private = _make_shared_claim(
            claim_id="clm_private1", claim_key="private.key",
            visibility="private",
        )
        knowledge_store.insert_claim(shared)
        knowledge_store.insert_claim(private)

        result = knowledge_store.get_shared_claims_since()
        assert len(result) == 1
        assert result[0].claim_id == "clm_shared1"

    def test_filters_by_since(self, knowledge_store):
        """Should only return claims updated after 'since' timestamp."""
        claim = _make_shared_claim(claim_id="clm_new1")
        knowledge_store.insert_claim(claim)

        # Query with future timestamp should return nothing
        future = datetime(2099, 1, 1, tzinfo=timezone.utc)
        result = knowledge_store.get_shared_claims_since(since=future)
        assert len(result) == 0

        # Query with past timestamp should return the claim
        past = datetime(2020, 1, 1, tzinfo=timezone.utc)
        result = knowledge_store.get_shared_claims_since(since=past)
        assert len(result) == 1

    def test_filters_by_scope_id(self, knowledge_store):
        """Should filter claims by scope_id."""
        c1 = _make_shared_claim(claim_id="clm_s1", scope_id="project-a", claim_key="k1")
        c2 = _make_shared_claim(claim_id="clm_s2", scope_id="project-b", claim_key="k2")
        knowledge_store.insert_claim(c1)
        knowledge_store.insert_claim(c2)

        result = knowledge_store.get_shared_claims_since(scope_id="project-a")
        assert len(result) == 1
        assert result[0].scope.scope_id == "project-a"

    def test_respects_limit(self, knowledge_store):
        """Should respect the limit parameter."""
        for i in range(5):
            c = _make_shared_claim(claim_id=f"clm_lim{i}", claim_key=f"key.{i}")
            knowledge_store.insert_claim(c)

        result = knowledge_store.get_shared_claims_since(limit=2)
        assert len(result) == 2


# -----------------------------------------------------------------------
# KnowledgeSync.sync_from_peer tests
# -----------------------------------------------------------------------


class TestSyncFromPeer:
    """Test pulling claims from a single peer."""

    async def test_imports_valid_claims(self, sync, knowledge_store):
        """Should import shared claims from peer."""
        remote_claim = _make_shared_claim(
            claim_id="clm_remote1",
            claim_key="remote.fact",
            statement="Remote fact",
        )
        remote_data = [remote_claim.model_dump(mode="json")]

        with patch("multihead.mesh.client.MeshClient") as MockClient:
            instance = AsyncMock()
            instance.fetch_claims = AsyncMock(return_value=remote_data)
            MockClient.return_value = instance

            count = await sync.sync_from_peer(
                peer_url="http://peer1:7337",
                node_id="peer1",
            )

        assert count == 1
        imported = knowledge_store.get_claim("clm_remote1")
        assert imported is not None
        assert imported.statement == "Remote fact"
        # Should be imported as proposed, not accepted
        assert imported.claim_status == ClaimStatus.PROPOSED

    async def test_skips_duplicate_claims(self, sync, knowledge_store):
        """Should not re-import claims that already exist locally."""
        existing = _make_shared_claim(claim_id="clm_dup1", claim_key="dup.key")
        knowledge_store.insert_claim(existing)

        remote_data = [existing.model_dump(mode="json")]

        with patch("multihead.mesh.client.MeshClient") as MockClient:
            instance = AsyncMock()
            instance.fetch_claims = AsyncMock(return_value=remote_data)
            MockClient.return_value = instance

            count = await sync.sync_from_peer(
                peer_url="http://peer1:7337",
                node_id="peer1",
            )

        assert count == 0

    async def test_skips_private_claims(self, sync, knowledge_store):
        """Should not import private claims."""
        private = _make_shared_claim(
            claim_id="clm_priv1", claim_key="priv.key", visibility="private"
        )
        remote_data = [private.model_dump(mode="json")]

        with patch("multihead.mesh.client.MeshClient") as MockClient:
            instance = AsyncMock()
            instance.fetch_claims = AsyncMock(return_value=remote_data)
            MockClient.return_value = instance

            count = await sync.sync_from_peer(
                peer_url="http://peer1:7337",
                node_id="peer1",
            )

        assert count == 0
        assert knowledge_store.get_claim("clm_priv1") is None

    async def test_rejects_tampered_claims(self, sync, knowledge_store, tmp_path):
        """Should reject claims with invalid signatures."""
        from multihead.mesh.security import MeshSecurity

        security = MeshSecurity(shared_secret="test-secret")
        store_with_sec = KnowledgeStore(
            db_path=tmp_path / "sec_knowledge.db",
            mesh_security=security,
            agent_id="test-node",
        )
        sync_with_sec = KnowledgeSync(
            knowledge_store=store_with_sec,
            peer_registry=Mock(spec=PeerRegistry),
        )

        # Create a claim with valid signature, then tamper it
        claim = _make_shared_claim(claim_id="clm_tampered1", claim_key="tampered.key")
        claim.signature = security.sign_message(claim.canonical_json_for_signing())
        claim.signed_by = "remote-agent"
        # Tamper the statement after signing
        claim_data = claim.model_dump(mode="json")
        claim_data["statement"] = "TAMPERED statement"

        with patch("multihead.mesh.client.MeshClient") as MockClient:
            instance = AsyncMock()
            instance.fetch_claims = AsyncMock(return_value=[claim_data])
            MockClient.return_value = instance

            count = await sync_with_sec.sync_from_peer(
                peer_url="http://peer1:7337",
                node_id="peer1",
            )

        assert count == 0
        assert store_with_sec.get_claim("clm_tampered1") is None

    async def test_handles_connection_failure(self, sync):
        """Should return 0 on connection failure."""
        with patch("multihead.mesh.client.MeshClient") as MockClient:
            instance = AsyncMock()
            instance.fetch_claims = AsyncMock(side_effect=Exception("connection refused"))
            MockClient.return_value = instance

            count = await sync.sync_from_peer(
                peer_url="http://dead:7337",
                node_id="dead",
            )

        assert count == 0

    async def test_updates_watermark(self, sync):
        """Should update sync watermark after successful sync."""
        with patch("multihead.mesh.client.MeshClient") as MockClient:
            instance = AsyncMock()
            instance.fetch_claims = AsyncMock(return_value=[])
            MockClient.return_value = instance

            await sync.sync_from_peer(
                peer_url="http://peer1:7337",
                node_id="peer1",
            )

        assert "peer1" in sync._sync_watermarks

    async def test_passes_since_watermark(self, sync):
        """Should pass 'since' parameter from watermark to client."""
        sync._sync_watermarks["peer1"] = "2026-01-01T00:00:00+00:00"

        with patch("multihead.mesh.client.MeshClient") as MockClient:
            instance = AsyncMock()
            instance.fetch_claims = AsyncMock(return_value=[])
            MockClient.return_value = instance

            await sync.sync_from_peer(
                peer_url="http://peer1:7337",
                node_id="peer1",
            )

            instance.fetch_claims.assert_called_once_with(
                since="2026-01-01T00:00:00+00:00"
            )


# -----------------------------------------------------------------------
# Conflict handling tests
# -----------------------------------------------------------------------


class TestConflictResolution:
    """Test claim conflict detection during import."""

    async def test_records_conflict_on_different_statement(self, sync, knowledge_store):
        """Should detect and record conflicts for same key with different content."""
        # Insert a local accepted claim (needs evidence for accept)
        local = _make_shared_claim(
            claim_id="clm_local1",
            claim_key="shared.fact",
            statement="Local version",
        )
        knowledge_store.insert_claim(local)
        # Insert evidence and accept
        from multihead.knowledge_models import EvidencePointer, Record
        rec = Record(uri="test://evidence")
        knowledge_store.insert_record(rec)
        evp = EvidencePointer(record_id=rec.record_id)
        knowledge_store.insert_evidence(evp)
        knowledge_store.link_claim_evidence(local.claim_id, evp.evidence_id)
        knowledge_store.accept_claim(local.claim_id)

        # Remote claim with same key but different statement
        remote = _make_shared_claim(
            claim_id="clm_remote_conflict1",
            claim_key="shared.fact",
            statement="Remote version (different)",
            confidence=0.95,
        )
        remote_data = [remote.model_dump(mode="json")]

        with patch("multihead.mesh.client.MeshClient") as MockClient:
            instance = AsyncMock()
            instance.fetch_claims = AsyncMock(return_value=remote_data)
            MockClient.return_value = instance

            count = await sync.sync_from_peer(
                peer_url="http://peer1:7337",
                node_id="peer1",
            )

        # Both should exist
        assert count == 1
        assert knowledge_store.get_claim("clm_remote_conflict1") is not None


# -----------------------------------------------------------------------
# sync_all_peers tests
# -----------------------------------------------------------------------


class TestSyncAllPeers:
    """Test syncing from all discovered peers."""

    async def test_syncs_from_all_available_peers(self, sync, mock_peer_registry):
        """Should sync from each unique available peer."""
        # Add a second peer
        mock_peer_registry.peer_heads["mesh-peer2-vlm"] = PeerHead(
            head_id="mesh-peer2-vlm",
            node_id="peer2",
            peer_url="http://192.168.1.20:7337",
            capability_kind="vlm",
            status="available",
        )

        with patch("multihead.mesh.client.MeshClient") as MockClient:
            instance = AsyncMock()
            instance.fetch_claims = AsyncMock(return_value=[])
            MockClient.return_value = instance

            results = await sync.sync_all_peers()

        assert "peer1" in results
        assert "peer2" in results

    async def test_skips_offline_peers(self, sync, mock_peer_registry):
        """Should not sync from offline peers."""
        mock_peer_registry.peer_heads["mesh-peer1-llm"].status = "offline"

        with patch("multihead.mesh.client.MeshClient") as MockClient:
            instance = AsyncMock()
            instance.fetch_claims = AsyncMock(return_value=[])
            MockClient.return_value = instance

            results = await sync.sync_all_peers()

        assert len(results) == 0

    async def test_deduplicates_peers_with_multiple_heads(self, sync, mock_peer_registry):
        """Should only sync once per node even if it has multiple heads."""
        # Add a second head for the same peer
        mock_peer_registry.peer_heads["mesh-peer1-vlm"] = PeerHead(
            head_id="mesh-peer1-vlm",
            node_id="peer1",
            peer_url="http://192.168.1.10:7337",
            capability_kind="vlm",
            status="available",
        )

        with patch("multihead.mesh.client.MeshClient") as MockClient:
            instance = AsyncMock()
            instance.fetch_claims = AsyncMock(return_value=[])
            MockClient.return_value = instance

            results = await sync.sync_all_peers()

        # Only one sync for peer1, despite two heads
        assert len(results) == 1
        assert "peer1" in results


# -----------------------------------------------------------------------
# Lifecycle tests
# -----------------------------------------------------------------------


class TestSyncLifecycle:
    """Test start/stop of periodic sync."""

    async def test_start_stop(self, sync):
        """Should start and stop periodic sync without errors."""
        await sync.start_periodic_sync()
        assert sync._sync_task is not None

        await sync.stop()
        assert sync._sync_task is None

    async def test_start_idempotent(self, sync):
        """Should not create duplicate tasks on multiple start calls."""
        await sync.start_periodic_sync()
        task1 = sync._sync_task

        await sync.start_periodic_sync()
        task2 = sync._sync_task

        assert task1 is task2
        await sync.stop()

    async def test_stop_when_not_started(self, sync):
        """Should not raise when stopping an un-started sync."""
        await sync.stop()  # Should not raise


# -----------------------------------------------------------------------
# MeshClient claim methods tests
# -----------------------------------------------------------------------


class TestMeshClientClaimMethods:
    """Test the fetch_claims and push_claims methods on MeshClient."""

    async def test_fetch_claims_basic(self):
        """fetch_claims() should GET /v1/claims."""
        from multihead.mesh.client import MeshClient

        client = MeshClient(base_url="http://peer:7337", max_retries=0)
        import httpx
        resp = httpx.Response(200, json=[{"claim_id": "clm_1", "statement": "test"}])
        resp._request = httpx.Request("GET", "http://peer:7337/v1/claims")

        with patch("httpx.AsyncClient.request", new_callable=AsyncMock, return_value=resp):
            result = await client.fetch_claims()

        assert len(result) == 1
        assert result[0]["claim_id"] == "clm_1"

    async def test_fetch_claims_with_params(self):
        """fetch_claims() should pass since and scope_id params."""
        from multihead.mesh.client import MeshClient

        client = MeshClient(base_url="http://peer:7337", max_retries=0)
        import httpx
        resp = httpx.Response(200, json=[])
        resp._request = httpx.Request("GET", "http://peer:7337/v1/claims")

        with patch(
            "httpx.AsyncClient.request",
            new_callable=AsyncMock, return_value=resp,
        ) as mock_req:
            await client.fetch_claims(since="2026-01-01T00:00:00", scope_id="h2v")

        call_kwargs = mock_req.call_args
        params = call_kwargs.kwargs.get("params", {})
        assert params["since"] == "2026-01-01T00:00:00"
        assert params["scope_id"] == "h2v"

    async def test_push_claims(self):
        """push_claims() should POST /v1/claims/import."""
        from multihead.mesh.client import MeshClient

        client = MeshClient(base_url="http://peer:7337", max_retries=0)
        import httpx
        resp = httpx.Response(
            200, json={"imported": 2, "skipped": 0, "errors": []},
        )
        resp._request = httpx.Request("POST", "http://peer:7337/v1/claims/import")

        with patch(
            "httpx.AsyncClient.request",
            new_callable=AsyncMock, return_value=resp,
        ) as mock_req:
            result = await client.push_claims([{"claim_id": "clm_1"}, {"claim_id": "clm_2"}])

        assert result["imported"] == 2
        call_kwargs = mock_req.call_args
        body = call_kwargs.kwargs.get("json", {})
        assert len(body["claims"]) == 2
