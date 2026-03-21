"""Tests for consensus API routes (fast tests only)."""

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


class TestConsensusExecute:
    def test_basic_majority(self, client):
        """POST /consensus/execute with majority strategy."""
        resp = client.post("/consensus/execute", json={
            "prompt": "What is 2+2?",
            "heads": [
                {"head_id": "mock-llm"},
                {"head_id": "mock-vlm"},
            ],
            "strategy": "majority",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "consensus_outputs" in data
        assert "agreement_score" in data
        assert data["strategy_used"] == "majority"
        assert len(data["all_votes"]) == 2

    def test_three_heads(self, client):
        """Consensus with 3 heads."""
        resp = client.post("/consensus/execute", json={
            "prompt": "Count objects",
            "heads": [
                {"head_id": "mock-llm"},
                {"head_id": "mock-vlm"},
                {"head_id": "mock-ocr"},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["all_votes"]) == 3
        assert data["agreement_score"] > 0

    def test_weighted_strategy(self, client):
        """POST /consensus/execute with weighted strategy."""
        resp = client.post("/consensus/execute", json={
            "prompt": "Analyze this",
            "heads": [
                {"head_id": "mock-llm", "weight": 2.0},
                {"head_id": "mock-vlm", "weight": 1.0},
            ],
            "strategy": "weighted",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["strategy_used"] == "weighted"

    def test_unanimous_strategy(self, client):
        """POST /consensus/execute with unanimous strategy."""
        resp = client.post("/consensus/execute", json={
            "prompt": "Test",
            "heads": [
                {"head_id": "mock-llm"},
                {"head_id": "mock-vlm"},
            ],
            "strategy": "unanimous",
        })
        assert resp.status_code == 200

    def test_threshold_strategy(self, client):
        """POST /consensus/execute with threshold strategy."""
        resp = client.post("/consensus/execute", json={
            "prompt": "Test",
            "heads": [
                {"head_id": "mock-llm"},
                {"head_id": "mock-vlm"},
            ],
            "strategy": "threshold",
            "threshold": 0.6,
        })
        assert resp.status_code == 200

    def test_invalid_strategy(self, client):
        """Invalid strategy returns 400."""
        resp = client.post("/consensus/execute", json={
            "prompt": "Test",
            "heads": [{"head_id": "mock-llm"}],
            "strategy": "invalid-strat",
        })
        assert resp.status_code == 400
        assert "Invalid strategy" in resp.json()["detail"]

    def test_unknown_head(self, client):
        """Unknown head_id returns 404."""
        resp = client.post("/consensus/execute", json={
            "prompt": "Test",
            "heads": [{"head_id": "nonexistent-head"}],
        })
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]

    def test_cross_modal(self, client):
        """Cross-modal execution with custom prompts."""
        resp = client.post("/consensus/execute", json={
            "prompt": "Base prompt",
            "heads": [
                {
                    "head_id": "mock-llm",
                    "prompt_template": "Detect objects",
                    "extract_fields": ["count"],
                },
                {
                    "head_id": "mock-vlm",
                    "prompt_template": "Describe scene",
                    "extract_fields": ["count"],
                },
            ],
            "cross_modal": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "all_votes" in data

    def test_metrics_populated(self, client):
        """Metrics should be populated in response."""
        resp = client.post("/consensus/execute", json={
            "prompt": "Test",
            "heads": [
                {"head_id": "mock-llm"},
                {"head_id": "mock-vlm"},
            ],
        })
        data = resp.json()
        assert "total_heads" in data["metrics"]
        assert data["metrics"]["total_heads"] == 2


class TestConsensusStrategies:
    def test_list_strategies(self, client):
        """GET /consensus/strategies."""
        resp = client.get("/consensus/strategies")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 5
        names = {s["name"] for s in data}
        assert names == {"majority", "weighted", "unanimous", "threshold", "first_to_ahead"}
