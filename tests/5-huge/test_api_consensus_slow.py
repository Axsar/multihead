"""Slow tests for consensus API routes — tests with time.sleep."""

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
  - head_id: mock-vlm
    name: Mock VLM
    adapter: mock
    model: mock-v1
    kind: vlm
    gpu_required: false
  - head_id: mock-ocr
    name: Mock OCR
    adapter: mock
    model: mock-v1
    kind: llm
    gpu_required: false
""")

    app = create_app(settings)
    with TestClient(app) as c:
        yield c


class TestRunResults:
    def test_get_results_after_run(self, client):
        """GET /runs/{run_id}/results after a completed run."""
        # Create and wait for run
        resp = client.post("/runs", json={
            "goal": "Test results endpoint",
            "work_order": {
                "goal": "Test results endpoint",
                "steps": [
                    {
                        "name": "step1",
                        "head_id": "mock-llm",
                        "prompt_template": "Hello",
                    },
                ],
            },
        })
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]

        time.sleep(2)

        resp = client.get(f"/runs/{run_id}/results")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == run_id
        assert "results" in data
        assert data["completed_steps"] >= 1

    def test_results_not_found(self, client):
        """GET /runs/{run_id}/results for nonexistent run."""
        resp = client.get("/runs/nonexistent/results")
        assert resp.status_code == 404
