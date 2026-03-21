"""Tests for PeerRegistry and Router mesh routing."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, Mock, patch

from multihead.mesh.peer_registry import PeerRegistry, PeerHead
from multihead.models import AdapterKind
from multihead.router import Router


# -----------------------------------------------------------------------
# PeerRegistry tests
# -----------------------------------------------------------------------


@pytest.fixture
def mock_discovery():
    """Create a mock MeshDiscovery."""
    discovery = Mock()
    discovery.get_discovered_nodes.return_value = {
        "desktop-01": {"host": "192.168.1.10", "port": 7337, "source": "mdns"},
        "laptop-02": {"host": "192.168.1.20", "port": 7337, "source": "db"},
    }
    return discovery


@pytest.fixture
def registry(mock_discovery):
    return PeerRegistry(mesh_discovery=mock_discovery, auth_token="secret")


class TestPeerRegistryRefresh:
    """Test PeerRegistry.refresh() polling."""

    async def test_refresh_discovers_remote_heads(self, registry):
        """Should create PeerHead entries from peer capabilities."""
        caps = [
            {"name": "core-llm", "kind": "llm", "model": "qwen3:8b", "status": "available"},
            {"name": "vision-vlm", "kind": "vlm", "model": "qwen3-vl:32b", "status": "available"},
        ]
        with patch("multihead.mesh.client.MeshClient") as MockClient:
            instance = AsyncMock()
            instance.capabilities = AsyncMock(return_value=caps)
            MockClient.return_value = instance

            await registry.refresh()

        heads = registry.peer_heads
        assert len(heads) >= 2  # At least 2 caps from 2 peers (may have duplicates)
        # Check that heads from desktop-01 exist
        desktop_heads = [h for h in heads.values() if h.node_id == "desktop-01"]
        assert len(desktop_heads) == 2

    async def test_refresh_marks_unreachable_peers_offline(self, registry):
        """Should mark heads as offline when peer is unreachable."""
        # First, populate with a head
        registry._peer_heads["mesh-desktop-01-llm"] = PeerHead(
            head_id="mesh-desktop-01-llm",
            node_id="desktop-01",
            peer_url="http://192.168.1.10:7337",
            capability_kind="llm",
            status="available",
        )

        with patch("multihead.mesh.client.MeshClient") as MockClient:
            instance = AsyncMock()
            instance.capabilities = AsyncMock(side_effect=Exception("connection refused"))
            MockClient.return_value = instance

            await registry.refresh()

        head = registry._peer_heads.get("mesh-desktop-01-llm")
        assert head is not None
        assert head.status == "offline"

    async def test_refresh_removes_stale_peers(self, registry):
        """Should remove heads from peers no longer in discovery."""
        registry._peer_heads["mesh-old-node-llm"] = PeerHead(
            head_id="mesh-old-node-llm",
            node_id="old-node",
            peer_url="http://10.0.0.1:7337",
            capability_kind="llm",
        )

        with patch("multihead.mesh.client.MeshClient") as MockClient:
            instance = AsyncMock()
            instance.capabilities = AsyncMock(return_value=[{"kind": "llm", "status": "available"}])
            MockClient.return_value = instance

            await registry.refresh()

        assert "mesh-old-node-llm" not in registry._peer_heads

    async def test_empty_discovery_clears_all(self, mock_discovery):
        """Should clear all heads when no peers discovered."""
        mock_discovery.get_discovered_nodes.return_value = {}
        reg = PeerRegistry(mesh_discovery=mock_discovery)
        reg._peer_heads["mesh-some-llm"] = PeerHead(
            head_id="mesh-some-llm", node_id="some", peer_url="http://x:7337",
            capability_kind="llm",
        )

        await reg.refresh()
        assert len(reg._peer_heads) == 0


class TestPeerRegistryManifests:
    """Test HeadManifest generation."""

    def test_to_manifests_converts_peer_heads(self):
        """Should convert PeerHead to HeadManifest with AdapterKind.MESH."""
        reg = PeerRegistry(mesh_discovery=Mock(), auth_token="token123")
        reg._peer_heads["mesh-node1-llm"] = PeerHead(
            head_id="mesh-node1-llm",
            node_id="node1",
            peer_url="http://10.0.0.1:7337",
            capability_kind="llm",
            model="qwen3:8b",
            name="Node1 LLM",
        )

        manifests = reg.to_manifests()
        assert "mesh-node1-llm" in manifests
        m = manifests["mesh-node1-llm"]
        assert m.adapter == AdapterKind.MESH
        assert m.kind == "llm"
        assert m.is_local is False
        assert m.gpu_required is False
        assert m.extra["peer_url"] == "http://10.0.0.1:7337"
        assert m.extra["auth_token"] == "token123"

    def test_to_manifests_skips_offline(self):
        """Should not generate manifests for offline peers."""
        reg = PeerRegistry(mesh_discovery=Mock())
        reg._peer_heads["mesh-dead-llm"] = PeerHead(
            head_id="mesh-dead-llm",
            node_id="dead",
            peer_url="http://x:7337",
            capability_kind="llm",
            status="offline",
        )

        manifests = reg.to_manifests()
        assert len(manifests) == 0

    def test_get_peer_for_head(self):
        """Should look up peer by head_id."""
        reg = PeerRegistry(mesh_discovery=Mock())
        ph = PeerHead(
            head_id="mesh-abc-llm", node_id="abc",
            peer_url="http://x:7337", capability_kind="llm",
        )
        reg._peer_heads["mesh-abc-llm"] = ph
        assert reg.get_peer_for_head("mesh-abc-llm") is ph
        assert reg.get_peer_for_head("nonexistent") is None


# -----------------------------------------------------------------------
# Router mesh routing tests
# -----------------------------------------------------------------------


@pytest.fixture
def mock_head_manager():
    manager = Mock()
    manager.get_states.return_value = {}
    manager.get_manifest.return_value = None
    manager.get_breaker.return_value = None
    manager._manifests = {}
    manager._adapters = {}
    manager._states = {}
    manager._breakers = {}
    return manager


class TestRouterMesh:
    """Test Router.route_mesh() method."""

    def test_prefers_local_over_mesh(self, mock_head_manager):
        """Should prefer local head when available."""
        router = Router(head_manager=mock_head_manager)
        with patch.object(router, "route", return_value="local-llm"):
            result = router.route_mesh("llm")
        assert result == "local-llm"

    def test_falls_back_to_mesh_peers(self, mock_head_manager):
        """Should use mesh peer when no local head available."""
        peer_reg = Mock()
        peer_reg.peer_heads = {
            "mesh-peer1-llm": PeerHead(
                head_id="mesh-peer1-llm", node_id="peer1",
                peer_url="http://peer1:7337", capability_kind="llm",
                status="available", latency_ms=50.0,
            ),
        }
        router = Router(head_manager=mock_head_manager, peer_registry=peer_reg)
        with patch.object(router, "route", return_value=None):
            result = router.route_mesh("llm")
        assert result == "mesh-peer1-llm"

    def test_blocks_confidential_from_mesh(self, mock_head_manager):
        """Should not route CONFIDENTIAL data to remote peers."""
        from multihead.models import DataSensitivity

        peer_reg = Mock()
        peer_reg.peer_heads = {
            "mesh-peer1-llm": PeerHead(
                head_id="mesh-peer1-llm", node_id="peer1",
                peer_url="http://peer1:7337", capability_kind="llm",
                status="available",
            ),
        }
        privacy = Mock()
        privacy.data_sensitivity = DataSensitivity.CONFIDENTIAL

        router = Router(head_manager=mock_head_manager, peer_registry=peer_reg)
        with patch.object(router, "route", return_value=None):
            result = router.route_mesh("llm", privacy=privacy)
        assert result is None

    def test_returns_none_without_peer_registry(self, mock_head_manager):
        """Should return None when no peer_registry configured."""
        router = Router(head_manager=mock_head_manager)
        with patch.object(router, "route", return_value=None):
            result = router.route_mesh("llm")
        assert result is None

    def test_skips_offline_peers(self, mock_head_manager):
        """Should not route to offline peers."""
        peer_reg = Mock()
        peer_reg.peer_heads = {
            "mesh-dead-llm": PeerHead(
                head_id="mesh-dead-llm", node_id="dead",
                peer_url="http://dead:7337", capability_kind="llm",
                status="offline",
            ),
        }
        router = Router(head_manager=mock_head_manager, peer_registry=peer_reg)
        with patch.object(router, "route", return_value=None):
            result = router.route_mesh("llm")
        assert result is None

    def test_skips_wrong_kind(self, mock_head_manager):
        """Should only match peers with correct kind."""
        peer_reg = Mock()
        peer_reg.peer_heads = {
            "mesh-peer1-vlm": PeerHead(
                head_id="mesh-peer1-vlm", node_id="peer1",
                peer_url="http://peer1:7337", capability_kind="vlm",
                status="available",
            ),
        }
        router = Router(head_manager=mock_head_manager, peer_registry=peer_reg)
        with patch.object(router, "route", return_value=None):
            result = router.route_mesh("llm")
        assert result is None

    def test_excludes_specified_heads(self, mock_head_manager):
        """Should respect exclude set for mesh peers."""
        peer_reg = Mock()
        peer_reg.peer_heads = {
            "mesh-peer1-llm": PeerHead(
                head_id="mesh-peer1-llm", node_id="peer1",
                peer_url="http://peer1:7337", capability_kind="llm",
                status="available",
            ),
        }
        router = Router(head_manager=mock_head_manager, peer_registry=peer_reg)
        with patch.object(router, "route", return_value=None):
            result = router.route_mesh("llm", exclude={"mesh-peer1-llm"})
        assert result is None

    def test_prefers_lower_latency_peer(self, mock_head_manager):
        """Should prefer the peer with lower latency."""
        peer_reg = Mock()
        peer_reg.peer_heads = {
            "mesh-slow-llm": PeerHead(
                head_id="mesh-slow-llm", node_id="slow",
                peer_url="http://slow:7337", capability_kind="llm",
                status="available", latency_ms=500.0,
            ),
            "mesh-fast-llm": PeerHead(
                head_id="mesh-fast-llm", node_id="fast",
                peer_url="http://fast:7337", capability_kind="llm",
                status="available", latency_ms=20.0,
            ),
        }
        router = Router(head_manager=mock_head_manager, peer_registry=peer_reg)
        with patch.object(router, "route", return_value=None):
            result = router.route_mesh("llm")
        assert result == "mesh-fast-llm"
