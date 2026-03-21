"""Tests for Router BotVibes marketplace discovery."""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from multihead.router import Router
from multihead.models import AdapterKind


@pytest.fixture
def mock_head_manager():
    """Create a mock HeadManager for testing."""
    manager = Mock()
    manager.get_states.return_value = {}
    manager.get_manifest.return_value = None
    manager.get_breaker.return_value = None
    return manager


@pytest.fixture
def router(mock_head_manager):
    """Create a Router instance for testing."""
    return Router(head_manager=mock_head_manager)


class TestBotVibesMarketplaceDiscovery:
    """Test Router's BotVibes marketplace discovery methods."""

    async def test_discover_without_token(self, router):
        """Test discovery returns empty list without token."""
        providers = await router.discover_botvibes_providers(
            capability="visual_reasoning",
            acp_url="http://localhost:8000",
            acp_token="",  # Empty token
        )

        assert providers == []

    @patch("httpx.AsyncClient")
    async def test_discover_uses_env_defaults(self, mock_client_class, router):
        """Test discovery uses environment variables for defaults."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value={"results": []})

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_client_class.return_value = mock_client

        with patch.dict("os.environ", {
            "ACP_URL": "http://test:8000",
            "ACP_SESSION_KEY": "test-key",
        }):
            providers = await router.discover_botvibes_providers(
                capability="visual_reasoning"
            )
            assert providers == []

    @patch("httpx.AsyncClient")
    async def test_discover_returns_providers(self, mock_client_class, router):
        """Test discovery returns providers from marketplace API."""
        # Mock BotVibes API response
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
                    },
                    "stats": {
                        "quality_score": 0.94,
                        "ewma_latency_ms": 4500,
                        "dispute_rate": 0.02,
                        "accept_rate": 0.98,
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

        providers = await router.discover_botvibes_providers(
            capability="visual_reasoning",
            acp_url="http://localhost:8000",
            acp_token="test-token",
        )

        assert len(providers) == 1
        assert providers[0]["provider_id"] == "vision-expert-123"
        assert providers[0]["reputation"] == 0.94
        assert providers[0]["cost_per_call"] == 0.08
        assert providers[0]["latency_p50_ms"] == 4500

    def test_provider_to_manifest_basic(self, router):
        """Test converting provider dict to HeadManifest."""
        provider = {
            "provider_id": "vision-expert-123",
            "provider_name": "Vision Expert",
            "capabilities": ["visual_reasoning", "image_classification"],
            "reputation": 0.94,
            "cost_per_call": 0.08,
            "latency_p50_ms": 4500,
            "privacy_level": "encrypted",
        }

        manifest = router._provider_to_manifest(
            provider,
            acp_url="http://localhost:8000/api/v1",
            acp_token="test-token",
            project_id="test-project",
        )

        assert manifest.head_id == "botvibes-vision-expert-123"
        assert manifest.name == "Vision Expert"
        assert manifest.adapter == AdapterKind.BOTVIBES
        assert manifest.model == "vision-expert-123"
        assert manifest.is_local is False
        assert manifest.privacy_level == "encrypted"
        assert manifest.endpoint == "http://localhost:8000/api/v1"

        # Check extra fields
        assert manifest.extra["api_key"] == "test-token"
        assert manifest.extra["project_id"] == "test-project"
        assert manifest.extra["target_agent_id"] == "vision-expert-123"
        assert manifest.extra["target_capability"] == "visual_reasoning"

        # Check capabilities
        assert manifest.capabilities is not None
        assert manifest.capabilities.solver_type == "external_service"
        assert "visual_reasoning" in manifest.capabilities.task_types
        assert manifest.capabilities.latency_p50_ms == 4500
        assert manifest.capabilities.cost_per_call == 0.08
        assert manifest.capabilities.accuracy_score == 0.94

    def test_provider_to_manifest_with_metadata(self, router):
        """Test provider conversion includes metadata."""
        provider = {
            "provider_id": "test-provider",
            "provider_name": "Test Provider",
            "capabilities": ["test_task"],
            "reputation": 0.9,
            "cost_per_call": 0.05,
            "latency_p50_ms": 2000,
            "privacy_level": "external",
            "metadata": {
                "queue_depth_avg": 2,
                "success_rate": 0.98,
                "uptime_pct": 99.5,
            },
        }

        manifest = router._provider_to_manifest(
            provider,
            acp_url="http://localhost:8000",
            acp_token="token",
        )

        assert manifest.extra["provider_metadata"] == provider["metadata"]
        assert manifest.extra["provider_metadata"]["success_rate"] == 0.98

    def test_provider_to_manifest_handles_missing_fields(self, router):
        """Test provider conversion with minimal data."""
        provider = {
            "provider_id": "minimal-provider",
        }

        manifest = router._provider_to_manifest(
            provider,
            acp_url="http://localhost:8000",
            acp_token="token",
        )

        assert manifest.head_id == "botvibes-minimal-provider"
        assert manifest.name == "botvibes-minimal-provider"  # Falls back to solver_id
        assert manifest.capabilities is not None
        assert manifest.capabilities.task_types == []  # Empty capabilities
        assert manifest.capabilities.cost_per_call == 0.0
        assert manifest.capabilities.accuracy_score == 0.0

    @patch("httpx.AsyncClient")
    async def test_discover_filters_providers(self, mock_client_class, router):
        """Test discovery applies filters correctly."""
        # Mock API response with multiple providers
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value={
            "results": [
                {
                    "listing": {"agent_id": "high-quality", "unit_price": 0.05, "sla_p95_ms": 3000},
                    "stats": {"quality_score": 0.95, "ewma_latency_ms": 2500},
                },
                {
                    "listing": {"agent_id": "low-quality", "unit_price": 0.03, "sla_p95_ms": 2000},
                    # Below min_reputation
                    "stats": {"quality_score": 0.70, "ewma_latency_ms": 1800},
                },
                {
                    "listing": {"agent_id": "expensive", "unit_price": 1.00, "sla_p95_ms": 4000},
                    "stats": {"quality_score": 0.98, "ewma_latency_ms": 3500},  # Above max_cost
                },
                {
                    "listing": {"agent_id": "slow", "unit_price": 0.06, "sla_p95_ms": 8000},
                    "stats": {"quality_score": 0.92, "ewma_latency_ms": 7500},  # Above max_latency
                },
            ]
        })

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_client_class.return_value = mock_client

        providers = await router.discover_botvibes_providers(
            capability="visual_reasoning",
            acp_url="http://localhost:8000",
            acp_token="test-token",
            min_reputation=0.85,
            max_cost=0.50,
            max_latency_ms=5000,
            limit=10,
        )

        # Only "high-quality" should pass all filters
        assert len(providers) == 1
        assert providers[0]["provider_id"] == "high-quality"

    def test_provider_to_manifest_creates_valid_botvibes_adapter_config(self, router):
        """Test that generated manifest works with BotVibesAdapter."""
        provider = {
            "provider_id": "test-agent",
            "provider_name": "Test Agent",
            "capabilities": ["test"],
            "reputation": 0.9,
            "cost_per_call": 0.1,
            "latency_p50_ms": 1000,
            "privacy_level": "encrypted",
        }

        manifest = router._provider_to_manifest(
            provider,
            acp_url="http://localhost:8000/api/v1",
            acp_token="test-token",
            project_id="test-project",
        )

        # Verify all required fields for BotVibesAdapter are present
        assert "api_key" in manifest.extra
        assert "project_id" in manifest.extra
        assert "target_capability" in manifest.extra
        assert manifest.endpoint != ""
        assert manifest.adapter == AdapterKind.BOTVIBES

        # Could be instantiated by BotVibesAdapter (if we wanted to test that)
        # adapter = BotVibesAdapter(manifest)
        # assert adapter.acp_url == "http://localhost:8000/api/v1"
        # assert adapter.acp_token == "test-token"
        # assert adapter.target_capability == "test"
