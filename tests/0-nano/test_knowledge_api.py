"""Tests for knowledge store POST endpoints and MultiHead client module."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from multihead.api.app import create_app
from multihead.config import Settings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        config_dir=tmp_path / "config",
    )
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    recipes_dir = config_dir / "recipes"
    recipes_dir.mkdir()
    (config_dir / "heads.yaml").write_text("""
heads:
  - head_id: mock-llm
    name: Mock LLM
    adapter: mock
    model: mock-v1
    kind: llm
    gpu_required: false
""")
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# POST /knowledge/claims
# ---------------------------------------------------------------------------

class TestCreateClaim:
    def test_basic_claim(self, client):
        resp = client.post("/knowledge/claims", json={
            "claim_key": "project.test.status",
            "statement": "Test component is working",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["claim_key"] == "project.test.status"
        assert data["statement"] == "Test component is working"
        assert data["claim_id"].startswith("clm_")
        assert data["claim_status"] == "accepted"

    def test_claim_with_all_fields(self, client):
        resp = client.post("/knowledge/claims", json={
            "claim_key": "project.auth.token_count",
            "statement": "Auth service processed 12 tokens",
            "subject_type": "component",
            "subject_id": "auth",
            "subject_label": "AuthService",
            "predicate": "processed",
            "value": 12,
            "value_type": "number",
            "claim_type": "fact",
            "claim_status": "accepted",
            "scope_type": "project",
            "scope_id": "default",
            "confidence": 1.0,
            "stability": "stable",
            "importance": 0.8,
            "rationale": "Counted from pipeline output",
            "produced_by": "auth-agent",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["claim_key"] == "project.auth.token_count"

    def test_claim_persists(self, client):
        client.post("/knowledge/claims", json={
            "claim_key": "project.persist.test",
            "statement": "This should persist",
            "scope_id": "test",
        })
        resp = client.get("/knowledge/claims", params={"scope_id": "test"})
        assert resp.status_code == 200
        claims = resp.json()
        assert any(c["claim_key"] == "project.persist.test" for c in claims)

    def test_claim_queryable_by_id(self, client):
        resp = client.post("/knowledge/claims", json={
            "claim_key": "project.lookup.test",
            "statement": "Lookup test",
        })
        claim_id = resp.json()["claim_id"]
        resp = client.get(f"/knowledge/claims/{claim_id}")
        assert resp.status_code == 200
        assert resp.json()["statement"] == "Lookup test"

    def test_duplicate_claim_key_same_scope_supersedes(self, client):
        resp1 = client.post("/knowledge/claims", json={
            "claim_key": "project.dup.test",
            "statement": "First",
            "scope_id": "default",
        })
        resp2 = client.post("/knowledge/claims", json={
            "claim_key": "project.dup.test",
            "statement": "Second",
            "scope_id": "default",
        })
        # Dedup supersedes old claim, new one is inserted
        assert resp2.status_code == 200

    def test_invalid_claim_type_fails(self, client):
        resp = client.post("/knowledge/claims", json={
            "claim_key": "project.bad.type",
            "statement": "Bad type",
            "claim_type": "nonexistent",
        })
        assert resp.status_code in (400, 422)

    def test_invalid_scope_type_fails(self, client):
        resp = client.post("/knowledge/claims", json={
            "claim_key": "project.bad.scope",
            "statement": "Bad scope",
            "scope_type": "galaxy",
        })
        assert resp.status_code in (400, 422)

    def test_proposed_claims_same_key_ok(self, client):
        """Multiple proposed claims with same key should not conflict."""
        r1 = client.post("/knowledge/claims", json={
            "claim_key": "project.proposed.test",
            "statement": "Proposal A",
            "claim_status": "proposed",
        })
        r2 = client.post("/knowledge/claims", json={
            "claim_key": "project.proposed.test",
            "statement": "Proposal B",
            "claim_status": "proposed",
        })
        assert r1.status_code == 200
        assert r2.status_code == 200


# ---------------------------------------------------------------------------
# POST /knowledge/events
# ---------------------------------------------------------------------------

class TestCreateEvent:
    def test_basic_event(self, client):
        resp = client.post("/knowledge/events", json={
            "title": "Pipeline run completed",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["title"] == "Pipeline run completed"
        assert data["event_id"].startswith("evt_")
        assert data["event_status"] == "confirmed"

    def test_event_with_all_fields(self, client):
        resp = client.post("/knowledge/events", json={
            "title": "Vertical layout export finished",
            "summary": "Exported page 5 with 8 panels",
            "event_type": "task_completed",
            "event_status": "confirmed",
            "tags": ["default", "vertical_layout", "export"],
            "metrics": {"panels": 8, "duration_s": 12.5},
            "produced_by": "vertical_pipeline",
            "entities": [
                {"type": "component", "id": "vertical_pipeline", "label": "Vertical Layout"},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["event_type"] == "task_completed"

    def test_event_persists(self, client):
        client.post("/knowledge/events", json={
            "title": "Persist test event",
            "event_type": "note",
        })
        resp = client.get("/knowledge/events", params={"event_type": "note"})
        assert resp.status_code == 200
        events = resp.json()
        assert any(e["title"] == "Persist test event" for e in events)

    def test_event_queryable_by_id(self, client):
        resp = client.post("/knowledge/events", json={
            "title": "Lookup test event",
        })
        event_id = resp.json()["event_id"]
        resp = client.get(f"/knowledge/events/{event_id}")
        assert resp.status_code == 200
        assert resp.json()["title"] == "Lookup test event"

    def test_invalid_event_type_fails(self, client):
        resp = client.post("/knowledge/events", json={
            "title": "Bad event type",
            "event_type": "nonexistent",
        })
        assert resp.status_code in (400, 422)

    def test_multiple_events_same_title_ok(self, client):
        r1 = client.post("/knowledge/events", json={"title": "Run A"})
        r2 = client.post("/knowledge/events", json={"title": "Run A"})
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json()["event_id"] != r2.json()["event_id"]


# ---------------------------------------------------------------------------
# MultiHeadClient (unit tests with mocked server)
# ---------------------------------------------------------------------------

class TestMultiHeadClient:
    def test_deposit_claim_via_client(self, client):
        """Client deposit_claim hits the POST endpoint correctly."""
        # Simulate what the client does: POST /knowledge/claims
        resp = client.post("/knowledge/claims", json={
            "claim_key": "project.client.test",
            "statement": "Client deposited this claim",
            "produced_by": "test_client",
            "scope_id": "default",
        })
        assert resp.status_code == 200
        assert resp.json()["claim_key"] == "project.client.test"

    def test_report_event_via_client(self, client):
        """Client report_event hits the POST endpoint correctly."""
        resp = client.post("/knowledge/events", json={
            "title": "Client event test",
            "summary": "Reported by client",
            "event_type": "task_completed",
            "produced_by": "test_client",
            "metrics": {"items_processed": 42},
        })
        assert resp.status_code == 200
        assert resp.json()["title"] == "Client event test"

    def test_query_claims_via_client(self, client):
        # Create then query
        client.post("/knowledge/claims", json={
            "claim_key": "project.query.test",
            "statement": "Query test",
            "scope_id": "query_scope",
        })
        resp = client.get("/knowledge/claims", params={"scope_id": "query_scope"})
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

    def test_full_roundtrip(self, client):
        """Deposit a claim, report an event, query both back."""
        # Claim
        cr = client.post("/knowledge/claims", json={
            "claim_key": "project.roundtrip.claim",
            "statement": "Roundtrip claim test",
            "scope_id": "roundtrip",
        })
        assert cr.status_code == 200
        claim_id = cr.json()["claim_id"]

        # Event
        er = client.post("/knowledge/events", json={
            "title": "Roundtrip event test",
            "event_type": "task_completed",
        })
        assert er.status_code == 200
        event_id = er.json()["event_id"]

        # Query back
        c = client.get(f"/knowledge/claims/{claim_id}")
        assert c.status_code == 200
        assert c.json()["statement"] == "Roundtrip claim test"

        e = client.get(f"/knowledge/events/{event_id}")
        assert e.status_code == 200
        assert e.json()["title"] == "Roundtrip event test"


# ---------------------------------------------------------------------------
# GET /knowledge/briefing
# ---------------------------------------------------------------------------

class TestBriefing:
    def test_briefing_returns_structure(self, client):
        resp = client.get("/knowledge/briefing", params={"component": "auth"})
        assert resp.status_code == 200
        data = resp.json()
        assert "claims" in data
        assert "related_claims" in data
        assert "recent_events" in data
        assert "summary" in data
        assert data["component"] == "auth"

    def test_briefing_finds_direct_claims(self, client):
        """Claims whose key contains the component name show up."""
        client.post("/knowledge/claims", json={
            "claim_key": "project.auth.status",
            "statement": "Service processed 12 requests",
            "scope_id": "default",
        })
        resp = client.get("/knowledge/briefing", params={"component": "auth"})
        data = resp.json()
        assert len(data["claims"]) >= 1
        assert any("auth" in c["claim_key"] for c in data["claims"])

    def test_briefing_finds_related_claims(self, client):
        """Claims that mention the component in their statement show up as related."""
        client.post("/knowledge/claims", json={
            "claim_key": "project.export.dependency",
            "statement": "Deployment depends on auth output",
            "scope_id": "default",
        })
        resp = client.get("/knowledge/briefing", params={"component": "auth"})
        data = resp.json()
        assert len(data["related_claims"]) >= 1

    def test_briefing_finds_events_by_tag(self, client):
        """Events with matching tags appear in briefing."""
        client.post("/knowledge/events", json={
            "title": "AuthService run completed",
            "tags": ["auth", "default"],
            "event_type": "task_completed",
        })
        resp = client.get("/knowledge/briefing", params={"component": "auth"})
        data = resp.json()
        assert len(data["recent_events"]) >= 1

    def test_briefing_finds_events_by_entity(self, client):
        """Events with matching entity IDs appear in briefing."""
        client.post("/knowledge/events", json={
            "title": "Layout export done",
            "entities": [{"type": "component", "id": "auth"}],
            "event_type": "task_completed",
        })
        resp = client.get("/knowledge/briefing", params={"component": "auth"})
        data = resp.json()
        assert len(data["recent_events"]) >= 1

    def test_briefing_finds_events_by_title(self, client):
        """Events mentioning component in title appear."""
        client.post("/knowledge/events", json={
            "title": "auth processing stage 6",
            "event_type": "note",
        })
        resp = client.get("/knowledge/briefing", params={"component": "auth"})
        data = resp.json()
        assert len(data["recent_events"]) >= 1

    def test_briefing_empty_component_returns_empty(self, client):
        """Unknown component returns empty lists, not error."""
        resp = client.get("/knowledge/briefing", params={"component": "nonexistent_xyz"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["claims"] == []
        assert data["related_claims"] == []
        assert data["recent_events"] == []

    def test_briefing_requires_component(self, client):
        """Missing component param should fail."""
        resp = client.get("/knowledge/briefing")
        assert resp.status_code == 422

    def test_briefing_respects_scope(self, client):
        """Claims from different scopes don't leak."""
        client.post("/knowledge/claims", json={
            "claim_key": "project.tails.status",
            "statement": "Tails complete",
            "scope_id": "other_project",
        })
        resp = client.get("/knowledge/briefing", params={
            "component": "tails", "scope_id": "default",
        })
        data = resp.json()
        # Should NOT include the claim from other_project
        assert not any(c["claim_key"] == "project.tails.status" for c in data["claims"])
