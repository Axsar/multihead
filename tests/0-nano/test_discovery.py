"""Tests for discovery system."""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from multihead.discovery.base import SolverCandidate
from multihead.discovery.huggingface import HuggingFaceDiscovery
from multihead.discovery.ollama import OllamaDiscovery
from multihead.discovery.botvibes import BotVibesDiscovery


class TestSolverCandidate:
    """Test SolverCandidate dataclass."""

    def test_create_basic_candidate(self):
        """Test creating a basic solver candidate."""
        candidate = SolverCandidate(
            solver_id="test-solver",
            name="Test Solver",
            source="test",
            solver_type="llm",
            task_types=["text_generation"],
        )

        assert candidate.solver_id == "test-solver"
        assert candidate.source == "test"
        assert candidate.solver_type == "llm"

    def test_candidate_to_dict(self):
        """Test converting candidate to dictionary."""
        candidate = SolverCandidate(
            solver_id="test-solver",
            name="Test Solver",
            source="test",
            solver_type="llm",
            task_types=["text_generation"],
            benchmark_scores={"mmlu": 0.72},
        )

        data = candidate.to_dict()
        assert data["solver_id"] == "test-solver"
        assert data["benchmark_scores"]["mmlu"] == 0.72
        assert "discovered_at" in data

    def test_candidate_validates_solver_id(self):
        """Test that solver_id is required."""
        with pytest.raises(ValueError, match="solver_id is required"):
            SolverCandidate(
                solver_id="",
                name="Test",
                source="test",
                solver_type="llm",
            )


class TestHuggingFaceDiscovery:
    """Test HuggingFace discovery agent."""

    @patch("httpx.AsyncClient")
    async def test_discover_new_solvers(self, mock_client_class):
        """Test discovering new solvers from HuggingFace."""
        # Mock HuggingFace API response
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value=[
            {
                "id": "meta-llama/Llama-2-7b",
                "pipeline_tag": "text-generation",
                "downloads": 50000,
                "likes": 500,
                "tags": ["llama", "text-generation"],
                "cardData": {
                    "license": "llama2",
                    "description": "Llama 2 7B model"
                },
            }
        ])

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_client_class.return_value = mock_client

        discovery = HuggingFaceDiscovery()
        candidates = await discovery.discover_new_solvers(limit=10)

        assert len(candidates) == 1
        assert candidates[0].solver_id == "hf-meta-llama-Llama-2-7b"
        assert candidates[0].solver_type == "llm"
        assert candidates[0].source == "huggingface"
        assert "text_generation" in candidates[0].task_types

    @patch("httpx.AsyncClient")
    async def test_get_solver_details(self, mock_client_class):
        """Test getting details for a specific model."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value={
            "id": "meta-llama/Llama-2-7b",
            "pipeline_tag": "text-generation",
            "downloads": 50000,
        })

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_client_class.return_value = mock_client

        discovery = HuggingFaceDiscovery()
        candidate = await discovery.get_solver_details("meta-llama/Llama-2-7b")

        assert candidate is not None
        assert candidate.model_id == "meta-llama/Llama-2-7b"


class TestOllamaDiscovery:
    """Test Ollama discovery agent."""

    @patch("httpx.AsyncClient")
    async def test_discover_from_local(self, mock_client_class):
        """Test discovering models from local Ollama server."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value={
            "models": [
                {
                    "name": "llama3.1:8b",
                    "size": 4661211808,
                    "modified_at": "2024-01-01T00:00:00Z",
                }
            ]
        })

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_client_class.return_value = mock_client

        discovery = OllamaDiscovery()
        candidates = await discovery.discover_new_solvers(limit=10)

        # Should discover from popular models list (fallback)
        assert len(candidates) > 0
        assert all(c.source == "ollama" for c in candidates)
        assert all(c.estimated_cost == 0.0 for c in candidates)  # Local = free

    def test_model_to_candidate_llm(self):
        """Test converting Ollama model to candidate."""
        discovery = OllamaDiscovery()

        model = {
            "name": "llama3.1",
            "size": "8b",
            "tag": "latest",
        }

        candidate = discovery._model_to_candidate(model)

        assert candidate is not None
        assert candidate.solver_id == "ollama-llama3.1"
        assert candidate.solver_type == "llm"
        assert "text_generation" in candidate.task_types

    def test_model_to_candidate_vlm(self):
        """Test converting Ollama VLM to candidate."""
        discovery = OllamaDiscovery()

        model = {
            "name": "llava",
            "size": "7b",
            "tag": "latest",
        }

        candidate = discovery._model_to_candidate(model)

        assert candidate is not None
        assert candidate.solver_type == "vlm"
        assert "image" in candidate.modalities


class TestBotVibesDiscovery:
    """Test BotVibes discovery agent."""

    @patch("httpx.AsyncClient")
    async def test_discover_new_solvers(self, mock_client_class):
        """Test discovering providers from BotVibes marketplace."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value={
            "results": [
                {
                    "listing": {
                        "agent_id": "vision-expert-123",
                        "name": "Vision Expert",
                        "unit_price": 0.08,
                        "sla_p95_ms": 5000,
                        "capability_id": "visual_reasoning",
                    },
                    "stats": {
                        "quality_score": 0.94,
                        "ewma_latency_ms": 4500,
                    },
                    "scoring": {
                        "total_score": 92.5,
                    },
                }
            ]
        })

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_client_class.return_value = mock_client

        discovery = BotVibesDiscovery(
            acp_url="http://localhost:8000",
            acp_token="test-token"
        )
        candidates = await discovery.discover_new_solvers(
            solver_types=["vlm"],
            limit=10
        )

        assert len(candidates) > 0
        assert candidates[0].solver_id == "botvibes-vision-expert-123"
        assert candidates[0].solver_type == "vlm"
        assert candidates[0].source == "botvibes"
        assert candidates[0].estimated_cost == 0.08

    @patch("httpx.AsyncClient")
    async def test_get_solver_details(self, mock_client_class):
        """Test getting details for a specific provider."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value={
            "results": [
                {
                    "listing": {
                        "agent_id": "vision-expert-123",
                        "name": "Vision Expert",
                        "unit_price": 0.08,
                        "capability_id": "visual_reasoning",
                    },
                    "stats": {
                        "quality_score": 0.94,
                        "ewma_latency_ms": 4500,
                    },
                }
            ]
        })

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_client_class.return_value = mock_client

        discovery = BotVibesDiscovery(
            acp_url="http://localhost:8000",
            acp_token="test-token"
        )
        candidate = await discovery.get_solver_details("botvibes-vision-expert-123")

        assert candidate is not None
        assert candidate.model_id == "vision-expert-123"

    def test_capability_to_solver_type(self):
        """Test capability to solver type mapping."""
        discovery = BotVibesDiscovery("http://test", "token")

        assert discovery._capability_to_solver_type("visual_reasoning") == "vlm"
        assert discovery._capability_to_solver_type("object_detection") == "object_detection"
        assert discovery._capability_to_solver_type("text_generation") == "llm"
        assert discovery._capability_to_solver_type("unknown") == "external_service"
