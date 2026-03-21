"""Integration tests: multi-module flows (fast tests only)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from multihead.head_manager import HeadManager
from multihead.knowledge_models import (
    Claim,
    ClaimCanonical,
    ClaimScope,
    ClaimType,
    EntityRef,
    EvidencePointer,
    Provenance,
    Record,
    ScopeType,
    SpanRef,
    ValueObject,
)
from multihead.knowledge_store import KnowledgeStore
from multihead.mesh.capability import CapabilityRegistry, auto_register_from_heads
from multihead.mesh.security import MeshSecurity
from multihead.models import AdapterKind, HeadManifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prov():
    return Provenance(produced_by={"kind": "test", "id": "unit"})


def _make_record(uri="file:///test.txt"):
    return Record(uri=uri)


def _make_evidence(record_id, uri="file:///test.txt"):
    return EvidencePointer(record_id=record_id, uri=uri, span=SpanRef(start=0, end=100))


def _make_claim(claim_key="test.key", statement="Test claim."):
    return Claim(
        claim_type=ClaimType.DECISION,
        scope=ClaimScope(scope_type=ScopeType.PROJECT, scope_id="multihead"),
        canonical=ClaimCanonical(
            claim_key=claim_key,
            subject=EntityRef(entity_type="component", entity_id="core"),
            predicate="is",
            object=ValueObject(value_type="string", value="true"),
        ),
        statement=statement,
        provenance=_prov(),
    )


# ---------------------------------------------------------------------------
# 1. Knowledge Pipeline: Record -> Evidence -> Claim linking (fast)
# ---------------------------------------------------------------------------


class TestKnowledgePipeline:
    def test_record_to_evidence_chain(self, tmp_path):
        """Verify record -> evidence -> claim linking works end-to-end."""
        ks = KnowledgeStore(tmp_path / "knowledge.db")

        # Insert record
        rec = _make_record(uri="file:///design.md")
        rec = ks.insert_record(rec)

        # Insert evidence pointing to record
        ev = _make_evidence(rec.record_id, uri="file:///design.md")
        ev = ks.insert_evidence(ev)

        # Insert claim with evidence
        claim = _make_claim(claim_key="arch.decision", statement="Use event sourcing")
        claim = ks.insert_claim(claim)
        ks.link_claim_evidence(claim.claim_id, ev.evidence_id)

        # Verify the chain
        claims = ks.list_claims()
        assert len(claims) == 1
        assert claims[0].statement == "Use event sourcing"


# ---------------------------------------------------------------------------
# 4. Mesh: Capability Registration + Auth
# ---------------------------------------------------------------------------


class TestMeshIntegration:
    def test_capability_auto_registration(self):
        """Heads auto-register as capabilities in the mesh registry."""
        manifests = {
            "llm-1": HeadManifest(
                head_id="llm-1", name="LLM Head", adapter=AdapterKind.MOCK,
                model="phi-3", kind="llm", gpu_required=True, vram_hint_mb=4096,
            ),
            "embed-1": HeadManifest(
                head_id="embed-1", name="Embed Head", adapter=AdapterKind.MOCK,
                model="bge-small", kind="embed", gpu_required=False,
            ),
        }
        reg = CapabilityRegistry()
        caps = auto_register_from_heads(reg, manifests, "test-node")

        assert len(caps) == 2
        llm_caps = reg.find_available("llm")
        embed_caps = reg.find_available("embed")
        assert len(llm_caps) == 1
        assert len(embed_caps) == 1
        assert llm_caps[0].gpu_required is True
        assert llm_caps[0].vram_hint_mb == 4096

    def test_mesh_api_with_auth(self, tmp_path):
        """Full mesh API test: register capabilities, auth on routes."""
        from fastapi import FastAPI
        from multihead.mesh.mesh_routes import router

        app = FastAPI()
        app.include_router(router, prefix="/v1")

        manifests = {
            "mock-llm": HeadManifest(
                head_id="mock-llm", name="Mock", adapter=AdapterKind.MOCK,
                model="mock-v1", kind="llm", gpu_required=False,
            ),
        }
        reg = CapabilityRegistry()
        auto_register_from_heads(reg, manifests, "test-node")

        app.state.capability_registry = reg
        app.state.node_id = "test-node"
        app.state.head_manager = HeadManager(manifests)

        # Configure auth
        secret = "integration-test-secret"
        app.state.mesh_security = MeshSecurity(secret)

        client = TestClient(app)

        # Unauthenticated -> 401
        resp = client.get("/v1/capabilities")
        assert resp.status_code == 401

        # Authenticated -> 200
        headers = {"Authorization": f"Bearer {secret}"}
        resp = client.get("/v1/capabilities", headers=headers)
        assert resp.status_code == 200
        caps = resp.json()
        assert len(caps) == 1
        assert caps[0]["kind"] == "llm"

        # Node info with auth
        resp = client.get("/v1/node", headers=headers)
        assert resp.status_code == 200
        assert resp.json()["node_id"] == "test-node"

        # Health always open
        resp = client.get("/v1/health")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 5. API End-to-End (fast tests only — chat test moved to slow)
# ---------------------------------------------------------------------------


class TestAPIEndToEnd:
    def test_full_run_lifecycle(self, api_client):
        """Create run -> get status -> list runs."""
        # Create run
        resp = api_client.post("/runs", json={
            "goal": "integration test",
            "steps": [
                {"name": "step1", "head_id": "mock-llm", "prompt_template": "test"}
            ],
        })
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]

        # Get run state
        resp = api_client.get(f"/runs/{run_id}")
        assert resp.status_code == 200
        assert resp.json()["run_id"] == run_id

        # List runs
        resp = api_client.get("/runs")
        assert resp.status_code == 200
        runs = resp.json()
        assert any(r["run_id"] == run_id for r in runs)

    def test_health_includes_heads(self, api_client):
        """Health endpoint shows loaded heads."""
        resp = api_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "mock-llm" in data["heads"]

    def test_knowledge_endpoints(self, api_client):
        """Knowledge endpoints return empty initially."""
        for endpoint in ["/knowledge/events", "/knowledge/claims", "/knowledge/records"]:
            resp = api_client.get(endpoint)
            assert resp.status_code == 200
            assert resp.json() == []

    def test_metrics_endpoint(self, api_client):
        """Metrics endpoint returns valid JSON."""
        resp = api_client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "counters" in data
        assert "gauges" in data
        assert "histograms" in data

    def test_dashboard_returns_html(self, api_client):
        """Dashboard endpoint returns HTML."""
        resp = api_client.get("/dashboard")
        assert resp.status_code == 200
        assert "MultiHead" in resp.text
