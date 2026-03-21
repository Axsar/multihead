"""Tests for zero-config mesh discovery (v1.0).

Covers:
- get_lan_ip() returns a non-0.0.0.0 address
- MeshDiscovery node_id normalises colons to dashes
- MeshDiscovery stores knowledge_store ref and DB scan runs in heartbeat
- PresenceMonitor lifecycle (start/stop)
- AgenticCore._detect_peers() DB fallback
- solve.py discover_sessions_mdns() returns peers from running MeshDiscovery
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from multihead.mesh.discovery import MeshDiscovery, get_lan_ip
from multihead.mesh.presence import PresenceMonitor
from multihead.knowledge_store import KnowledgeStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_knowledge_store(tmp_path: Path) -> KnowledgeStore:
    return KnowledgeStore(tmp_path / "mesh_test.db")


def _make_presence_monitor(
    tmp_path: Path,
    node_id: str = "node-test-7337",
    scope_id: str = "multihead",
) -> tuple[PresenceMonitor, KnowledgeStore]:
    ks = _make_knowledge_store(tmp_path)
    pm = PresenceMonitor(node_id=node_id, store=ks, interval=30.0, port=7337, scope_id=scope_id)
    return pm, ks


# ---------------------------------------------------------------------------
# 1. get_lan_ip
# ---------------------------------------------------------------------------


class TestGetLanIP:
    def test_returns_string(self):
        ip = get_lan_ip()
        assert isinstance(ip, str)

    def test_not_zero_addr(self):
        """get_lan_ip should NOT return 0.0.0.0."""
        ip = get_lan_ip()
        assert ip != "0.0.0.0"

    def test_valid_ipv4_format(self):
        ip = get_lan_ip()
        parts = ip.split(".")
        assert len(parts) == 4
        for part in parts:
            assert 0 <= int(part) <= 255

    def test_fallback_when_udp_fails(self):
        """When the UDP trick fails, falls back to gethostbyname or 127.0.0.1."""
        with patch("socket.socket") as mock_sock:
            mock_sock.side_effect = OSError("mocked")
            ip = get_lan_ip()
            # Should still return *something* valid
            assert isinstance(ip, str)
            assert len(ip.split(".")) == 4


# ---------------------------------------------------------------------------
# 2. MeshDiscovery node_id normalisation
# ---------------------------------------------------------------------------


class TestNodeIdNormalisation:
    def test_colons_replaced_with_dashes(self):
        disc = MeshDiscovery(node_id="node-host:8000")
        assert disc.node_id == "node-host-8000"

    def test_already_dashed_unchanged(self):
        disc = MeshDiscovery(node_id="node-host-8000")
        assert disc.node_id == "node-host-8000"

    def test_multiple_colons(self):
        disc = MeshDiscovery(node_id="a:b:c:d")
        assert disc.node_id == "a-b-c-d"


# ---------------------------------------------------------------------------
# 3. MeshDiscovery stores knowledge_store and DB scan in heartbeat
# ---------------------------------------------------------------------------


class TestMeshDiscoveryDBScan:
    def test_knowledge_store_stored(self, tmp_path):
        ks = _make_knowledge_store(tmp_path)
        disc = MeshDiscovery(node_id="n1", knowledge_store=ks)
        assert disc.knowledge_store is ks

    def test_knowledge_store_defaults_none(self):
        disc = MeshDiscovery(node_id="n1")
        assert disc.knowledge_store is None

    def test_scan_db_for_peers_called_in_heartbeat(self, tmp_path):
        """The heartbeat loop calls scan_db_for_peers when knowledge_store is set."""
        ks = _make_knowledge_store(tmp_path)
        disc = MeshDiscovery(node_id="n1", knowledge_store=ks)

        original_scan = disc.scan_db_for_peers
        call_count = 0

        def counting_scan(store):
            nonlocal call_count
            call_count += 1
            return original_scan(store)

        disc.scan_db_for_peers = counting_scan

        # Simulate one heartbeat iteration directly
        disc._stop_event.clear()
        # Override HEARTBEAT_INTERVAL to 0 so the loop fires immediately
        disc.HEARTBEAT_INTERVAL = 0

        # Run heartbeat in thread, let it fire once, then stop
        import threading

        disc._stop_event = threading.Event()

        def run_once():
            disc._heartbeat_loop()

        t = threading.Thread(target=run_once, daemon=True)
        t.start()
        time.sleep(0.3)
        disc._stop_event.set()
        t.join(timeout=2)

        assert call_count >= 1

    def test_scan_db_picks_up_presence_peer(self, tmp_path):
        """scan_db_for_peers adds a peer discovered via PresenceMonitor claims."""
        ks = _make_knowledge_store(tmp_path)

        # Emit a presence claim for a fake peer
        pm = PresenceMonitor(node_id="peer-node-9999", store=ks, port=9999, scope_id="multihead")
        pm._emit_claim(status="online")

        disc = MeshDiscovery(node_id="local-node-7337", knowledge_store=ks)
        added = disc.scan_db_for_peers(ks)
        assert added >= 1

        nodes = disc.get_discovered_nodes()
        assert any("peer-node" in nid for nid in nodes)


# ---------------------------------------------------------------------------
# 4. PresenceMonitor lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPresenceMonitorLifecycle:
    async def test_start_creates_task(self, tmp_path):
        pm, ks = _make_presence_monitor(tmp_path)
        await pm.start()
        assert pm._task is not None
        assert not pm._task.done()
        await pm.stop()

    async def test_stop_emits_offline(self, tmp_path):
        pm, ks = _make_presence_monitor(tmp_path)
        await pm.start()
        await pm.stop()

        claims = ks.list_claims(scope_id="multihead", limit=100)
        offline = [
            c for c in claims
            if c.canonical and "node-test-7337" in c.canonical.claim_key
            and isinstance(c.canonical.object.value, dict)
            and c.canonical.object.value.get("status") == "offline"
        ]
        assert len(offline) >= 1

    async def test_start_is_idempotent(self, tmp_path):
        pm, ks = _make_presence_monitor(tmp_path)
        await pm.start()
        task1 = pm._task
        await pm.start()  # should not create a new task
        assert pm._task is task1
        await pm.stop()


# ---------------------------------------------------------------------------
# 5. AgenticCore._detect_peers DB fallback
# ---------------------------------------------------------------------------


class TestDetectPeersDBFallback:
    def test_returns_peer_from_presence_claims(self, tmp_path):
        ks = _make_knowledge_store(tmp_path)

        # Write an online presence claim
        pm = PresenceMonitor(node_id="remote-node-A", store=ks, port=8001, scope_id="multihead")
        pm._emit_claim(status="online")

        # Create a minimal AgenticCore-like object
        from multihead.agentic_core import AgenticCore

        core = AgenticCore.__new__(AgenticCore)
        core.mesh_discovery = None
        core.knowledge_store = ks
        core.project_id = "multihead"

        peers = core._detect_peers()
        assert "remote-node-A" in peers

    def test_excludes_offline_peers(self, tmp_path):
        ks = _make_knowledge_store(tmp_path)

        pm = PresenceMonitor(node_id="dead-node", store=ks, port=8002, scope_id="multihead")
        pm._emit_claim(status="offline")

        from multihead.agentic_core import AgenticCore

        core = AgenticCore.__new__(AgenticCore)
        core.mesh_discovery = None
        core.knowledge_store = ks
        core.project_id = "multihead"

        peers = core._detect_peers()
        assert "dead-node" not in peers

    def test_prefers_mdns_over_db(self, tmp_path):
        """If mesh_discovery has nodes, returns those without hitting DB."""
        ks = _make_knowledge_store(tmp_path)

        mock_disc = Mock()
        mock_disc.get_discovered_nodes.return_value = {"mdns-peer": {"node_id": "mdns-peer"}}

        from multihead.agentic_core import AgenticCore

        core = AgenticCore.__new__(AgenticCore)
        core.mesh_discovery = mock_disc
        core.knowledge_store = ks
        core.project_id = "multihead"

        peers = core._detect_peers()
        assert peers == ["mdns-peer"]

    def test_returns_empty_when_no_store(self):
        from multihead.agentic_core import AgenticCore

        core = AgenticCore.__new__(AgenticCore)
        core.mesh_discovery = None
        core.knowledge_store = None
        core.project_id = "multihead"

        assert core._detect_peers() == []


# ---------------------------------------------------------------------------
# 6. solve.py discover_sessions_mdns
# ---------------------------------------------------------------------------


class TestDiscoverSessionsMdns:
    def test_with_existing_discovery(self):
        """When passed a MeshDiscovery with known nodes, returns them."""
        from multihead.solve import discover_sessions_mdns

        mock_disc = Mock()
        mock_disc.get_discovered_nodes.return_value = {
            "peer-A": {"node_id": "peer-A", "host": "10.0.0.2", "port": 7337, "source": "mdns"},
        }

        result = discover_sessions_mdns(discovery=mock_disc)
        assert len(result) == 1
        assert result[0]["node_id"] == "peer-A"

    def test_without_discovery_no_zeroconf(self):
        """When zeroconf is not installed, returns empty list gracefully."""
        from multihead.solve import discover_sessions_mdns

        with patch("multihead.mesh.discovery.MeshDiscovery.start", return_value=False):
            result = discover_sessions_mdns(discovery=None, timeout=0.1)
            assert result == []

    def test_returns_empty_on_exception(self):
        from multihead.solve import discover_sessions_mdns

        with patch("multihead.mesh.discovery.MeshDiscovery.start", side_effect=Exception("boom")):
            result = discover_sessions_mdns(discovery=None, timeout=0.1)
            assert result == []
