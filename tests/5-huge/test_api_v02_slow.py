"""Slow tests for v0.2 API endpoints — nightshift trigger runs full pipeline."""

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


def test_nightshift_trigger(client):
    resp = client.post("/nightshift/trigger")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "triggered"
