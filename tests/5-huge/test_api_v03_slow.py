"""Slow tests for v0.3 API endpoints — chat tests run through agentic core."""

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


def test_chat_new_session(client):
    resp = client.post("/chat", json={"message": "Hello!"})
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert "response" in data
    assert len(data["response"]) > 0


def test_chat_existing_session(client):
    # Create a session
    resp1 = client.post("/chat", json={"message": "Hello!"})
    sid = resp1.json()["session_id"]

    # Continue the conversation
    resp2 = client.post("/chat", json={"message": "How are you?", "session_id": sid})
    assert resp2.status_code == 200
    assert resp2.json()["session_id"] == sid


def test_chat_missing_session(client):
    resp = client.post("/chat", json={"message": "Hi", "session_id": "nonexistent"})
    assert resp.status_code == 404


def test_list_sessions_after_chat(client):
    client.post("/chat", json={"message": "Hello!"})
    resp = client.get("/chat/sessions")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


def test_get_session_details(client):
    resp1 = client.post("/chat", json={"message": "Hello!"})
    sid = resp1.json()["session_id"]

    resp = client.get(f"/chat/sessions/{sid}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == sid
    assert len(data["messages"]) >= 2  # user + assistant
