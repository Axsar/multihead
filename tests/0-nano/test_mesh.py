"""Tests for mesh protocol components."""

from __future__ import annotations

from unittest.mock import MagicMock

from starlette.testclient import TestClient

from multihead.mesh.capability import (
    Capability,
    CapabilityRegistry,
    auto_register_from_heads,
)
from multihead.mesh.security import MeshTokenAuth, MeshSecurity
from multihead.mesh.discovery import MeshDiscovery, _ServiceListener
from multihead.models import HeadManifest


# ---------------------------------------------------------------------------
# Capability model
# ---------------------------------------------------------------------------


class TestCapability:
    def test_auto_id(self):
        cap = Capability(name="test-llm", kind="llm")
        assert cap.capability_id.startswith("cap_")

    def test_explicit_id(self):
        cap = Capability(capability_id="cap_custom", name="x", kind="llm")
        assert cap.capability_id == "cap_custom"

    def test_default_status(self):
        cap = Capability(name="x", kind="llm")
        assert cap.status == "available"

    def test_serialization_roundtrip(self):
        cap = Capability(
            name="embed-model", kind="embed", model="bge-small",
            gpu_required=True, vram_hint_mb=512,
        )
        data = cap.model_dump(mode="json")
        restored = Capability.model_validate(data)
        assert restored.name == "embed-model"
        assert restored.gpu_required is True
        assert restored.vram_hint_mb == 512


# ---------------------------------------------------------------------------
# CapabilityRegistry
# ---------------------------------------------------------------------------


class TestCapabilityRegistry:
    def test_register_and_list(self):
        reg = CapabilityRegistry()
        cap = Capability(name="a", kind="llm")
        reg.register(cap)
        assert len(reg.list_capabilities()) == 1

    def test_unregister(self):
        reg = CapabilityRegistry()
        cap = Capability(name="a", kind="llm")
        reg.register(cap)
        reg.unregister(cap.capability_id)
        assert len(reg.list_capabilities()) == 0

    def test_unregister_missing(self):
        reg = CapabilityRegistry()
        reg.unregister("nonexistent")  # should not raise

    def test_list_by_kind(self):
        reg = CapabilityRegistry()
        reg.register(Capability(name="a", kind="llm"))
        reg.register(Capability(name="b", kind="embed"))
        reg.register(Capability(name="c", kind="llm"))
        assert len(reg.list_capabilities("llm")) == 2
        assert len(reg.list_capabilities("embed")) == 1
        assert len(reg.list_capabilities("vlm")) == 0

    def test_find_by_model(self):
        reg = CapabilityRegistry()
        reg.register(Capability(name="a", kind="llm", model="phi-3"))
        reg.register(Capability(name="b", kind="llm", model="mistral"))
        assert len(reg.find_by_model("phi-3")) == 1
        assert len(reg.find_by_model("nonexistent")) == 0

    def test_find_available(self):
        reg = CapabilityRegistry()
        cap1 = Capability(name="a", kind="llm")
        cap2 = Capability(name="b", kind="llm")
        reg.register(cap1)
        reg.register(cap2)
        reg.update_status(cap1.capability_id, "busy")
        available = reg.find_available("llm")
        assert len(available) == 1
        assert available[0].capability_id == cap2.capability_id

    def test_update_status(self):
        reg = CapabilityRegistry()
        cap = Capability(name="a", kind="llm")
        reg.register(cap)
        reg.update_status(cap.capability_id, "offline")
        assert reg.list_capabilities()[0].status == "offline"

    def test_update_status_missing(self):
        reg = CapabilityRegistry()
        reg.update_status("nonexistent", "busy")  # should not raise


# ---------------------------------------------------------------------------
# auto_register_from_heads
# ---------------------------------------------------------------------------


class TestAutoRegister:
    def test_registers_from_manifests(self):
        reg = CapabilityRegistry()
        manifests = {
            "head-1": HeadManifest(
                head_id="head-1",
                name="llm-head",
                kind="llm",
                model="phi-3",
                adapter="mock",
                gpu_required=True,
                vram_hint_mb=4096,
            ),
            "head-2": HeadManifest(
                head_id="head-2",
                name="embed-head",
                kind="embed",
                model="bge-small",
                adapter="mock",
                gpu_required=False,
            ),
        }
        registered = auto_register_from_heads(reg, manifests, "node-local")
        assert len(registered) == 2
        assert len(reg.list_capabilities()) == 2
        # Verify node_id propagation
        for cap in registered:
            assert cap.node_id == "node-local"

    def test_empty_manifests(self):
        reg = CapabilityRegistry()
        registered = auto_register_from_heads(reg, {}, "node-x")
        assert len(registered) == 0


# ---------------------------------------------------------------------------
# MeshTokenAuth
# ---------------------------------------------------------------------------


class TestMeshTokenAuth:
    def test_auto_token(self):
        auth = MeshTokenAuth()
        assert len(auth.token) > 10

    def test_explicit_token(self):
        auth = MeshTokenAuth("my-secret")
        assert auth.token == "my-secret"

    def test_validate_correct(self):
        auth = MeshTokenAuth("secret123")
        assert auth.validate("secret123") is True

    def test_validate_wrong(self):
        auth = MeshTokenAuth("secret123")
        assert auth.validate("wrong") is False

    def test_validate_no_token_required(self):
        auth = MeshTokenAuth()
        auth.token = None
        assert auth.validate("anything") is True

    def test_generate_token(self):
        auth = MeshTokenAuth("old")
        new = auth.generate_token()
        assert new != "old"
        assert auth.token == new

    def test_auth_header(self):
        auth = MeshTokenAuth("tok_abc")
        header = auth.get_auth_header()
        assert header == {"Authorization": "Bearer tok_abc"}


# ---------------------------------------------------------------------------
# MeshSecurity
# ---------------------------------------------------------------------------


class TestMeshSecurity:
    def test_trust_and_check(self):
        sec = MeshSecurity("shared-secret")
        sec.trust_node("node-a")
        assert sec.is_trusted("node-a") is True
        assert sec.is_trusted("node-b") is False

    def test_untrust(self):
        sec = MeshSecurity("shared-secret")
        sec.trust_node("node-a")
        sec.untrust_node("node-a")
        assert sec.is_trusted("node-a") is False

    def test_untrust_missing(self):
        sec = MeshSecurity("shared-secret")
        sec.untrust_node("nonexistent")  # should not raise

    def test_sign_and_verify(self):
        sec = MeshSecurity("my-secret-key")
        msg = "hello world"
        sig = sec.sign_message(msg)
        assert sec.verify_signature(msg, sig) is True

    def test_verify_wrong_signature(self):
        sec = MeshSecurity("my-secret-key")
        assert sec.verify_signature("hello", "badsig") is False

    def test_different_messages_different_sigs(self):
        sec = MeshSecurity("key")
        sig1 = sec.sign_message("msg1")
        sig2 = sec.sign_message("msg2")
        assert sig1 != sig2


# ---------------------------------------------------------------------------
# MeshDiscovery
# ---------------------------------------------------------------------------


class TestMeshDiscovery:
    def test_manual_node(self):
        disc = MeshDiscovery("node-1", port=7337)
        disc.add_manual_node("node-2", "192.168.1.100", 7337)
        nodes = disc.get_discovered_nodes()
        assert "node-2" in nodes
        assert nodes["node-2"]["host"] == "192.168.1.100"
        assert nodes["node-2"]["source"] == "manual"

    def test_empty_discovered(self):
        disc = MeshDiscovery("node-1")
        assert disc.get_discovered_nodes() == {}

    def test_start_without_zeroconf(self):
        disc = MeshDiscovery("node-1")
        # Should return False gracefully when zeroconf not installed
        result = disc.start()
        # Result depends on whether zeroconf is installed
        assert isinstance(result, bool)

    def test_stop_without_start(self):
        disc = MeshDiscovery("node-1")
        disc.stop()  # should not raise

    def test_remove_node(self):
        disc = MeshDiscovery("node-1")
        disc.add_manual_node("node-2", "10.0.0.1", 7337)
        assert disc.remove_node("node-2") is True
        assert disc.get_discovered_nodes() == {}

    def test_remove_nonexistent_node(self):
        disc = MeshDiscovery("node-1")
        assert disc.remove_node("ghost") is False


# ---------------------------------------------------------------------------
# Service Listener (mDNS)
# ---------------------------------------------------------------------------


class TestServiceListener:
    def _make_mock_info(self, node_id: str, host: str = "192.168.1.50", port: int = 7337):
        """Create a mock ServiceInfo object."""
        info = MagicMock()
        info.properties = {b"node_id": node_id.encode()}
        info.parsed_addresses.return_value = [host]
        info.port = port
        return info

    def test_add_service_discovers_peer(self):
        disc = MeshDiscovery("node-1")
        listener = _ServiceListener(disc)

        mock_zc = MagicMock()
        mock_zc.get_service_info.return_value = self._make_mock_info("node-2")

        listener.add_service(mock_zc, "_multihead._tcp.local.", "node-2._multihead._tcp.local.")

        nodes = disc.get_discovered_nodes()
        assert "node-2" in nodes
        assert nodes["node-2"]["host"] == "192.168.1.50"
        assert nodes["node-2"]["port"] == 7337
        assert nodes["node-2"]["source"] == "mdns"

    def test_add_service_ignores_self(self):
        disc = MeshDiscovery("node-1")
        listener = _ServiceListener(disc)

        mock_zc = MagicMock()
        mock_zc.get_service_info.return_value = self._make_mock_info("node-1")

        listener.add_service(mock_zc, "_multihead._tcp.local.", "node-1._multihead._tcp.local.")
        assert disc.get_discovered_nodes() == {}

    def test_add_service_handles_none_info(self):
        disc = MeshDiscovery("node-1")
        listener = _ServiceListener(disc)

        mock_zc = MagicMock()
        mock_zc.get_service_info.return_value = None

        listener.add_service(mock_zc, "_multihead._tcp.local.", "unknown._multihead._tcp.local.")
        assert disc.get_discovered_nodes() == {}

    def test_remove_service(self):
        disc = MeshDiscovery("node-1")
        disc._discovered["node-2"] = {
            "node_id": "node-2", "host": "10.0.0.1",
            "port": 7337, "source": "mdns",
        }
        listener = _ServiceListener(disc)

        mock_zc = MagicMock()
        listener.remove_service(mock_zc, "_multihead._tcp.local.", "node-2._multihead._tcp.local.")
        assert "node-2" not in disc.get_discovered_nodes()

    def test_remove_service_unknown_node(self):
        disc = MeshDiscovery("node-1")
        listener = _ServiceListener(disc)

        mock_zc = MagicMock()
        listener.remove_service(mock_zc, "_multihead._tcp.local.", "ghost._multihead._tcp.local.")
        # Should not raise

    def test_update_service_refreshes_info(self):
        disc = MeshDiscovery("node-1")
        listener = _ServiceListener(disc)

        mock_zc = MagicMock()
        mock_zc.get_service_info.return_value = self._make_mock_info("node-2", "10.0.0.99", 8000)

        listener.update_service(mock_zc, "_multihead._tcp.local.", "node-2._multihead._tcp.local.")
        nodes = disc.get_discovered_nodes()
        assert nodes["node-2"]["host"] == "10.0.0.99"
        assert nodes["node-2"]["port"] == 8000

    def test_multiple_peers_discovered(self):
        disc = MeshDiscovery("node-1")
        listener = _ServiceListener(disc)

        mock_zc = MagicMock()
        for i in range(2, 5):
            mock_zc.get_service_info.return_value = self._make_mock_info(f"node-{i}", f"10.0.0.{i}")
            listener.add_service(
                mock_zc, "_multihead._tcp.local.",
                f"node-{i}._multihead._tcp.local.",
            )

        nodes = disc.get_discovered_nodes()
        assert len(nodes) == 3
        assert "node-2" in nodes
        assert "node-3" in nodes
        assert "node-4" in nodes


# ---------------------------------------------------------------------------
# Mesh Route Auth Middleware
# ---------------------------------------------------------------------------


def _make_mesh_app(secret: str | None = None):
    """Create a minimal FastAPI app with mesh routes and optional auth."""
    from fastapi import FastAPI
    from multihead.mesh.mesh_routes import router

    app = FastAPI()
    app.include_router(router, prefix="/v1")

    reg = CapabilityRegistry()
    reg.register(Capability(name="test-llm", kind="llm", model="phi-3"))
    app.state.capability_registry = reg
    app.state.node_id = "test-node"

    if secret:
        app.state.mesh_security = MeshSecurity(secret)
    else:
        app.state.mesh_security = None

    return app


class TestMeshRouteAuth:
    def test_no_auth_configured_allows_all(self):
        app = _make_mesh_app(secret=None)
        client = TestClient(app)
        resp = client.get("/v1/capabilities")
        assert resp.status_code == 200

    def test_auth_rejects_missing_token(self):
        app = _make_mesh_app(secret="my-secret")
        client = TestClient(app)
        resp = client.get("/v1/capabilities")
        assert resp.status_code == 401

    def test_auth_rejects_wrong_token(self):
        app = _make_mesh_app(secret="my-secret")
        client = TestClient(app)
        resp = client.get("/v1/capabilities", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 403

    def test_auth_accepts_correct_token(self):
        app = _make_mesh_app(secret="my-secret")
        client = TestClient(app)
        resp = client.get("/v1/capabilities", headers={"Authorization": "Bearer my-secret"})
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_health_open_without_auth(self):
        app = _make_mesh_app(secret="my-secret")
        client = TestClient(app)
        resp = client.get("/v1/health")
        assert resp.status_code == 200

    def test_node_requires_auth(self):
        app = _make_mesh_app(secret="my-secret")
        client = TestClient(app)
        resp = client.get("/v1/node")
        assert resp.status_code == 401

    def test_node_with_auth(self):
        app = _make_mesh_app(secret="my-secret")
        client = TestClient(app)
        resp = client.get("/v1/node", headers={"Authorization": "Bearer my-secret"})
        assert resp.status_code == 200
        assert resp.json()["node_id"] == "test-node"

    def test_tasks_requires_auth(self):
        app = _make_mesh_app(secret="my-secret")
        client = TestClient(app)
        resp = client.post("/v1/tasks", json={"capability_kind": "llm", "prompt": "hi"})
        assert resp.status_code == 401
