"""Tests for v0.2 API endpoints: knowledge, packs, nightshift."""

import pytest
from fastapi.testclient import TestClient

from multihead.api.app import create_app
from multihead.config import Settings


@pytest.fixture
def client(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        config_dir=tmp_path / "config",
    )
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "recipes").mkdir()
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


# -------------------------------------------------------------------
# Health
# -------------------------------------------------------------------

def test_health_v02(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["version"]  # Just check it exists


# -------------------------------------------------------------------
# Knowledge endpoints
# -------------------------------------------------------------------

def test_list_events_empty(client):
    resp = client.get("/knowledge/events")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_claims_empty(client):
    resp = client.get("/knowledge/claims")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_records_empty(client):
    resp = client.get("/knowledge/records")
    assert resp.status_code == 200
    assert resp.json() == []


def test_event_not_found(client):
    resp = client.get("/knowledge/events/nonexistent")
    assert resp.status_code == 404


def test_claim_not_found(client):
    resp = client.get("/knowledge/claims/nonexistent")
    assert resp.status_code == 404


# -------------------------------------------------------------------
# Pack endpoints
# -------------------------------------------------------------------

def test_list_packs_empty(client):
    resp = client.get("/packs")
    assert resp.status_code == 200
    assert resp.json() == []


def test_build_pack(client):
    resp = client.post("/packs/build", json={"purpose": "Test Pack"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["purpose"] == "Test Pack"
    assert "pack_id" in data


def test_build_standard_packs(client):
    resp = client.post("/packs/build-standard")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 5


def test_get_pack(client):
    # Build first
    resp = client.post("/packs/build", json={"purpose": "Fetch Me"})
    pack_id = resp.json()["pack_id"]

    resp = client.get(f"/packs/{pack_id}")
    assert resp.status_code == 200
    assert resp.json()["purpose"] == "Fetch Me"


def test_get_pack_not_found(client):
    resp = client.get("/packs/nonexistent")
    assert resp.status_code == 404


# -------------------------------------------------------------------
# Night Shift endpoints (status/report only — trigger is slow)
# -------------------------------------------------------------------

def test_nightshift_status(client):
    resp = client.get("/nightshift/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["running"] is False


def test_nightshift_no_report(client):
    resp = client.get("/nightshift/report")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "no_report"
