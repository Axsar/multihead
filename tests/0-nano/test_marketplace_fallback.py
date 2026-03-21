"""Tests for Router marketplace fallback discovery."""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from multihead.router import Router
from multihead.models import AdapterKind


@pytest.fixture
def mock_head_manager():
    """Create a mock HeadManager."""
    manager = Mock()
    manager.get_states.return_value = {}
    manager.get_manifest.return_value = None
    manager.get_breaker.return_value = None
    manager._manifests = {}
    manager._adapters = {}
    manager._states = {}
    manager._breakers = {}
    return manager


@pytest.fixture
def router(mock_head_manager):
    """Create a Router instance."""
    return Router(head_manager=mock_head_manager)


class TestRouteWithMarketplaceFallback:
    """Test the route_with_marketplace_fallback method."""

    async def test_returns_local_head_when_available(self, router):
        """Should prefer local head over marketplace."""
        with patch.object(router, "route_with_discovery", return_value="core-llm"):
            result = await router.route_with_marketplace_fallback("text_generation")
        assert result == "core-llm"

    async def test_discovers_marketplace_when_no_local(self, router, mock_head_manager):
        """Should discover marketplace providers when no local head matches."""
        provider = {
            "provider_id": "vision-expert",
            "provider_name": "Vision Expert",
            "capabilities": ["visual_reasoning"],
            "reputation": 0.94,
            "cost_per_call": 0.08,
            "latency_p50_ms": 4500,
            "privacy_level": "encrypted",
        }

        with (
            patch.object(router, "route_with_discovery", return_value=None),
            patch.object(
                router, "discover_botvibes_providers",
                new_callable=AsyncMock, return_value=[provider],
            ),
            patch(
                "multihead.router._marketplace.os.environ",
                {"ACP_URL": "http://test", "ACP_SESSION_KEY": "key"},
            ),
            patch("multihead.router.BotVibesAdapter", create=True),
        ):
            result = await router.route_with_marketplace_fallback(
                "visual_reasoning",
                acp_url="http://test",
                acp_token="key",
            )

        assert result == "botvibes-vision-expert"
        assert "botvibes-vision-expert" in mock_head_manager._manifests

    async def test_returns_none_when_no_providers(self, router):
        """Should return None when no providers found."""
        with (
            patch.object(router, "route_with_discovery", return_value=None),
            patch.object(
                router, "discover_botvibes_providers",
                new_callable=AsyncMock, return_value=[],
            ),
        ):
            result = await router.route_with_marketplace_fallback(
                "unknown_capability",
            )
        assert result is None

    async def test_blocks_confidential_from_marketplace(self, router):
        """Should not discover marketplace for CONFIDENTIAL data."""
        from multihead.models import DataSensitivity

        privacy = Mock()
        privacy.data_sensitivity = DataSensitivity.CONFIDENTIAL

        with (
            patch.object(router, "route_with_discovery", return_value=None),
            patch.object(
                router, "discover_botvibes_providers",
                new_callable=AsyncMock,
            ) as mock_discover,
        ):
            result = await router.route_with_marketplace_fallback(
                "text_generation", privacy=privacy,
            )

        assert result is None
        mock_discover.assert_not_called()

    async def test_passes_filters_to_discovery(self, router):
        """Should pass cost/latency/reputation filters."""
        with (
            patch.object(router, "route_with_discovery", return_value=None),
            patch.object(
                router, "discover_botvibes_providers",
                new_callable=AsyncMock, return_value=[],
            ) as mock_discover,
        ):
            await router.route_with_marketplace_fallback(
                "visual_reasoning",
                max_cost=0.25,
                max_latency_ms=5000,
                min_reputation=0.90,
            )

        mock_discover.assert_called_once()
        call_kwargs = mock_discover.call_args
        assert call_kwargs.kwargs["max_cost"] == 0.25
        assert call_kwargs.kwargs["max_latency_ms"] == 5000
        assert call_kwargs.kwargs["min_reputation"] == 0.90

    async def test_reuses_existing_registered_head(self, router, mock_head_manager):
        """Should not re-register if botvibes head already exists."""
        mock_head_manager.get_states.return_value = {"botvibes-existing": {"state": "off"}}
        mock_head_manager._manifests["botvibes-existing"] = Mock()

        provider = {
            "provider_id": "existing",
            "provider_name": "Existing",
            "capabilities": ["test"],
            "reputation": 0.9,
            "cost_per_call": 0.05,
            "latency_p50_ms": 2000,
        }

        with (
            patch.object(router, "route_with_discovery", return_value=None),
            patch.object(
                router, "discover_botvibes_providers",
                new_callable=AsyncMock, return_value=[provider],
            ),
            patch(
                "multihead.router._marketplace.os.environ",
                {"ACP_URL": "http://test", "ACP_SESSION_KEY": "key"},
            ),
        ):
            result = await router.route_with_marketplace_fallback(
                "test",
                acp_url="http://test",
                acp_token="key",
            )

        assert result == "botvibes-existing"


class TestMarketplaceProviderRegistration:
    """Test that discovered providers are properly registered."""

    async def test_registers_adapter_and_breaker(self, router, mock_head_manager):
        """Should register manifest, adapter, state, and breaker."""
        provider = {
            "provider_id": "new-provider",
            "provider_name": "New Provider",
            "capabilities": ["text_generation"],
            "reputation": 0.92,
            "cost_per_call": 0.10,
            "latency_p50_ms": 3000,
            "privacy_level": "external",
        }

        with (
            patch.object(router, "route_with_discovery", return_value=None),
            patch.object(
                router, "discover_botvibes_providers",
                new_callable=AsyncMock, return_value=[provider],
            ),
            patch(
                "multihead.router._marketplace.os.environ",
                {"ACP_URL": "http://test:8000", "ACP_SESSION_KEY": "token123"},
            ),
        ):
            result = await router.route_with_marketplace_fallback(
                "text_generation",
                acp_url="http://test:8000",
                acp_token="token123",
            )

        head_id = "botvibes-new-provider"
        assert result == head_id
        assert head_id in mock_head_manager._manifests
        assert head_id in mock_head_manager._adapters
        assert head_id in mock_head_manager._states
        assert head_id in mock_head_manager._breakers

        manifest = mock_head_manager._manifests[head_id]
        assert manifest.adapter == AdapterKind.BOTVIBES
        assert manifest.is_local is False
