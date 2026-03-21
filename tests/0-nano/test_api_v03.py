"""Tests for v0.3 API endpoints: chat sessions (fast tests only)."""

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


def test_health_v03(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["version"]  # version present


def test_list_sessions_empty(client):
    resp = client.get("/chat/sessions")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_get_session_not_found(client):
    resp = client.get("/chat/sessions/nonexistent")
    assert resp.status_code == 404
