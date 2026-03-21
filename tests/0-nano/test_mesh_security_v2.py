"""Tests for enhanced mesh security: Ed25519 identity, TrustStore, audit."""

from __future__ import annotations

from multihead.mesh.security import (
    MeshSecurity,
    NodeIdentity,
    TrustStore,
)


# -----------------------------------------------------------------------
# NodeIdentity (Ed25519) tests
# -----------------------------------------------------------------------


class TestNodeIdentity:
    """Test Ed25519 key pair management."""

    def test_generate_key_pair(self, tmp_path):
        """Should generate and save an Ed25519 key pair."""
        key_path = tmp_path / "mesh_key.pem"
        identity = NodeIdentity(private_key_path=key_path, node_id="test-node")

        assert key_path.exists()
        assert identity.public_key_b64 != ""
        assert identity.node_id == "test-node"

    def test_load_existing_key(self, tmp_path):
        """Should load an existing key pair from disk."""
        key_path = tmp_path / "mesh_key.pem"
        # Generate first
        id1 = NodeIdentity(private_key_path=key_path, node_id="node1")
        pub1 = id1.public_key_b64

        # Load
        id2 = NodeIdentity(private_key_path=key_path, node_id="node1")
        assert id2.public_key_b64 == pub1

    def test_sign_and_verify(self, tmp_path):
        """Should sign messages that can be verified with public key."""
        key_path = tmp_path / "mesh_key.pem"
        identity = NodeIdentity(private_key_path=key_path, node_id="signer")

        message = "test-node:1234567890"
        signature = identity.sign(message)
        assert signature != ""

        # Verify with public key
        assert NodeIdentity.verify(message, signature, identity.public_key_b64) is True

    def test_verify_rejects_tampered_message(self, tmp_path):
        """Should reject verification when message is tampered."""
        key_path = tmp_path / "mesh_key.pem"
        identity = NodeIdentity(private_key_path=key_path, node_id="signer")

        signature = identity.sign("original-message")
        assert NodeIdentity.verify("tampered-message", signature, identity.public_key_b64) is False

    def test_verify_rejects_wrong_key(self, tmp_path):
        """Should reject verification with wrong public key."""
        key1 = tmp_path / "key1.pem"
        key2 = tmp_path / "key2.pem"
        id1 = NodeIdentity(private_key_path=key1, node_id="node1")
        id2 = NodeIdentity(private_key_path=key2, node_id="node2")

        signature = id1.sign("test-message")
        # Verify with wrong key
        assert NodeIdentity.verify("test-message", signature, id2.public_key_b64) is False

    def test_no_key_path_returns_empty(self):
        """Should return empty strings when no key path provided."""
        identity = NodeIdentity()
        assert identity.public_key_b64 == ""
        assert identity.sign("test") == ""

    def test_verify_rejects_invalid_base64(self):
        """Should return False for invalid base64 inputs."""
        assert NodeIdentity.verify("msg", "not-valid-b64!!!", "also-invalid!!!") is False


# -----------------------------------------------------------------------
# TrustStore tests
# -----------------------------------------------------------------------


class TestTrustStore:
    """Test persistent trust store for peer nodes."""

    def test_add_and_check_peer(self):
        """Should add peers and check trust status."""
        store = TrustStore()
        store.add_peer("node-a", public_key="abc123", trusted=True)
        assert store.is_trusted("node-a") is True
        assert store.is_trusted("unknown") is False

    def test_remove_peer(self):
        """Should remove peers."""
        store = TrustStore()
        store.add_peer("node-a", trusted=True)
        store.remove_peer("node-a")
        assert store.is_trusted("node-a") is False

    def test_get_public_key(self):
        """Should return stored public key."""
        store = TrustStore()
        store.add_peer("node-a", public_key="pub-key-abc")
        assert store.get_public_key("node-a") == "pub-key-abc"
        assert store.get_public_key("unknown") == ""

    def test_list_peers(self):
        """Should list all peer entries."""
        store = TrustStore()
        store.add_peer("node-a", trusted=True)
        store.add_peer("node-b", trusted=False)
        peers = store.list_peers()
        assert len(peers) == 2

    def test_save_and_load(self, tmp_path):
        """Should persist to YAML and reload."""
        yaml_path = tmp_path / "mesh_peers.yaml"
        store1 = TrustStore(path=yaml_path)
        store1.add_peer("node-x", public_key="key-x", trusted=True)
        store1.add_peer("node-y", public_key="key-y", trusted=False)
        store1.save()

        assert yaml_path.exists()

        # Reload from file
        store2 = TrustStore(path=yaml_path)
        assert store2.is_trusted("node-x") is True
        assert store2.is_trusted("node-y") is False
        assert store2.get_public_key("node-x") == "key-x"

    def test_load_missing_file(self, tmp_path):
        """Should handle missing file gracefully."""
        store = TrustStore(path=tmp_path / "nonexistent.yaml")
        assert store.list_peers() == []

    def test_untrusted_peer(self):
        """Should not trust peers marked as untrusted."""
        store = TrustStore()
        store.add_peer("bad-node", trusted=False)
        assert store.is_trusted("bad-node") is False


# -----------------------------------------------------------------------
# MeshSecurity enhanced tests
# -----------------------------------------------------------------------


class TestMeshSecurityEnhanced:
    """Test MeshSecurity with Ed25519 and TrustStore."""

    def test_is_trusted_checks_trust_store(self):
        """Should check both in-memory and trust store."""
        trust = TrustStore()
        trust.add_peer("store-node", trusted=True)

        security = MeshSecurity(shared_secret="secret", trust_store=trust)
        # In trust store
        assert security.is_trusted("store-node") is True
        # In memory
        security.trust_node("memory-node")
        assert security.is_trusted("memory-node") is True
        # Neither
        assert security.is_trusted("unknown") is False

    def test_sign_node_identity_headers(self, tmp_path):
        """Should create identity headers with Ed25519 signature."""
        key_path = tmp_path / "mesh_key.pem"
        identity = NodeIdentity(private_key_path=key_path, node_id="my-node")

        security = MeshSecurity(
            shared_secret="secret",
            node_identity=identity,
        )

        headers = security.sign_node_identity(timestamp="1234567890")
        assert headers["X-Node-ID"] == "my-node"
        assert headers["X-Node-Timestamp"] == "1234567890"
        assert headers["X-Node-Signature"] != ""

    def test_sign_node_identity_empty_without_identity(self):
        """Should return empty dict when no node identity configured."""
        security = MeshSecurity(shared_secret="secret")
        headers = security.sign_node_identity()
        assert headers == {}

    def test_verify_node_identity(self, tmp_path):
        """Should verify identity using trust store public key."""
        key_path = tmp_path / "mesh_key.pem"
        identity = NodeIdentity(private_key_path=key_path, node_id="remote-node")

        trust = TrustStore()
        trust.add_peer("remote-node", public_key=identity.public_key_b64, trusted=True)

        security = MeshSecurity(shared_secret="secret", trust_store=trust)

        # Sign
        ts = "1234567890"
        message = f"remote-node:{ts}"
        sig = identity.sign(message)

        # Verify
        assert security.verify_node_identity("remote-node", ts, sig) is True

    def test_verify_node_identity_rejects_unknown(self):
        """Should reject identity for unknown nodes."""
        security = MeshSecurity(shared_secret="secret")
        assert security.verify_node_identity("unknown", "123", "sig") is False

    def test_backward_compat_hmac_still_works(self):
        """HMAC signing/verification should still work."""
        security = MeshSecurity(shared_secret="my-shared-secret")
        sig = security.sign_message("hello world")
        assert security.verify_signature("hello world", sig) is True
        assert security.verify_signature("tampered", sig) is False
