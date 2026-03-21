"""Tests for the FastAPI endpoints."""

import time

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
    # Create minimal config
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    recipes_dir = config_dir / "recipes"
    recipes_dir.mkdir()

    # Write heads.yaml with mock heads
    (config_dir / "heads.yaml").write_text("""
heads:
  - head_id: mock-llm
    name: Mock LLM
    adapter: mock
    model: mock-v1
    kind: llm
    gpu_required: false
  - head_id: mock-vlm
    name: Mock VLM
    adapter: mock
    model: mock-v1
    kind: vlm
    gpu_required: false
""")

    # Write a test recipe
    (recipes_dir / "test-pipeline.yaml").write_text("""
goal: "Test pipeline"
steps:
  - name: plan
    head_id: mock-llm
    prompt_template: "Create a plan for testing"
  - name: extract
    head_id: mock-vlm
    prompt_template: "Extract data"
""")

    app = create_app(settings)
    # Use context manager to ensure lifespan runs
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "mock-llm" in data["heads"]


def test_list_heads(client):
    resp = client.get("/heads")
    assert resp.status_code == 200
    data = resp.json()
    assert "mock-llm" in data
    assert "mock-vlm" in data


def test_create_and_get_run(client):
    # Create a run
    resp = client.post("/runs", json={
        "recipe": "test-pipeline",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "run_id" in data
    run_id = data["run_id"]

    # Wait for background task
    time.sleep(2)

    # Get run status
    resp = client.get(f"/runs/{run_id}")
    assert resp.status_code == 200


def test_list_runs(client):
    # Create a run first
    client.post("/runs", json={"recipe": "test-pipeline"})
    time.sleep(1)

    resp = client.get("/runs")
    assert resp.status_code == 200
    runs = resp.json()
    assert len(runs) >= 1


def test_run_not_found(client):
    resp = client.get("/runs/nonexistent")
    assert resp.status_code == 404
