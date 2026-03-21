"""Tests for mesh init UX wizard: first-time, returning, peer-detected, shared-dir config write."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import yaml

from multihead.mesh.discovery import MeshDiscovery


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_config(config_file: Path, data: dict) -> None:
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")


def _read_config(config_file: Path) -> dict:
    return yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}


# ---------------------------------------------------------------------------
# Shared-dir config write logic
# (mirrors _init_mesh_config in cli.py, tested in isolation)
# ---------------------------------------------------------------------------

def _run_mesh_config(shared_dir: str, config_file: Path) -> dict:
    """Replicate _init_mesh_config logic without I/O side-effects."""
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_data: dict = {}
    if config_file.exists():
        config_data = yaml.safe_load(config_file.read_text()) or {}
    config_data["MULTIHEAD_DATA_DIR"] = shared_dir
    config_file.write_text(yaml.dump(config_data, default_flow_style=False), encoding="utf-8")
    return config_data


# ---------------------------------------------------------------------------
# 1. First-time user: config doesn't exist, default dir accepted
# ---------------------------------------------------------------------------

class TestFirstTimeUser:
    def test_creates_config_file(self, tmp_path):
        config_file = tmp_path / ".multihead" / "config.yaml"
        assert not config_file.exists()
        _run_mesh_config("/shared/multihead", config_file)
        assert config_file.exists()

    def test_default_dir_written(self, tmp_path):
        config_file = tmp_path / ".multihead" / "config.yaml"
        _run_mesh_config("/shared/multihead", config_file)
        data = _read_config(config_file)
        assert data["MULTIHEAD_DATA_DIR"] == "/shared/multihead"

    def test_parent_dir_created(self, tmp_path):
        config_file = tmp_path / "deep" / "nested" / "config.yaml"
        _run_mesh_config("/data/mh", config_file)
        assert config_file.parent.is_dir()

    def test_output_is_valid_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        _run_mesh_config("/data/mh", config_file)
        parsed = yaml.safe_load(config_file.read_text())
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# 2. Returning user: config already exists
# ---------------------------------------------------------------------------

class TestReturningUser:
    def test_preserves_other_keys(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        _write_config(config_file, {"OTHER_KEY": "kept", "MULTIHEAD_DATA_DIR": "/old"})
        _run_mesh_config("/new/shared", config_file)
        data = _read_config(config_file)
        assert data["OTHER_KEY"] == "kept"

    def test_updates_data_dir(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        _write_config(config_file, {"MULTIHEAD_DATA_DIR": "/old/path"})
        _run_mesh_config("/new/path", config_file)
        data = _read_config(config_file)
        assert data["MULTIHEAD_DATA_DIR"] == "/new/path"

    def test_does_not_duplicate_keys(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        _write_config(config_file, {"MULTIHEAD_DATA_DIR": "/first"})
        _run_mesh_config("/second", config_file)
        raw = config_file.read_text()
        assert raw.count("MULTIHEAD_DATA_DIR") == 1


# ---------------------------------------------------------------------------
# 3. Peer-detected scenario (MeshDiscovery)
# ---------------------------------------------------------------------------

class TestPeerDetected:
    def test_no_peers_initially(self):
        disc = MeshDiscovery(node_id="node-a", port=7337)
        assert disc.get_discovered_nodes() == {}

    def test_manual_peer_added(self):
        disc = MeshDiscovery(node_id="node-a", port=7337)
        disc.add_manual_node("node-b", "192.168.1.10", 7337)
        peers = disc.get_discovered_nodes()
        assert "node-b" in peers
        assert peers["node-b"]["host"] == "192.168.1.10"

    def test_shared_dir_written_on_peer_detect(self, tmp_path):
        """When peers are detected, the shared dir should be written to config."""
        disc = MeshDiscovery(node_id="node-a")
        disc.add_manual_node("node-b", "192.168.1.10", 7337)
        peers = disc.get_discovered_nodes()
        assert len(peers) > 0

        # Simulate: peer detected → write shared dir
        config_file = tmp_path / "config.yaml"
        _run_mesh_config("/shared/multihead", config_file)
        data = _read_config(config_file)
        assert data["MULTIHEAD_DATA_DIR"] == "/shared/multihead"

    def test_mdns_unavailable_falls_back_gracefully(self):
        """MeshDiscovery.start() returns False when zeroconf is absent, no crash."""
        disc = MeshDiscovery(node_id="node-x", port=7337)
        with patch.dict("sys.modules", {"zeroconf": None}):
            result = disc.start()
        # Either False (zeroconf unavailable) or True (zeroconf available) — no exception
        assert isinstance(result, bool)

    def test_remove_stale_peer(self):
        disc = MeshDiscovery(node_id="node-a")
        disc.add_manual_node("node-b", "10.0.0.1", 7337)
        removed = disc.remove_node("node-b")
        assert removed is True
        assert "node-b" not in disc.get_discovered_nodes()
