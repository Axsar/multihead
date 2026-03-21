"""Security helpers for mesh communication."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MeshTokenAuth:
    """Simple bearer token authentication for mesh nodes."""

    def __init__(self, token: str | None = None) -> None:
        self.token = token or secrets.token_urlsafe(32)

    def generate_token(self) -> str:
        """Generate a new random token."""
        self.token = secrets.token_urlsafe(32)
        return self.token

    def validate(self, provided: str) -> bool:
        """Validate a provided token."""
        if not self.token:
            return True  # No auth required
        return hmac.compare_digest(self.token, provided)

    def get_auth_header(self) -> dict[str, str]:
        """Get the Authorization header for requests."""
        return {"Authorization": f"Bearer {self.token}"}


class NodeIdentity:
    """Ed25519 key pair for per-node identity.

    Each node generates a key pair on init. The public key is shared
    with peers for identity verification.
    """

    def __init__(
        self,
        private_key_path: Path | None = None,
        node_id: str = "",
    ) -> None:
        self.node_id = node_id
        self._private_key: Any = None
        self._public_key: Any = None
        self._private_key_path = private_key_path

        if private_key_path and private_key_path.exists():
            self._load_key(private_key_path)
        elif private_key_path:
            self._generate_and_save(private_key_path)

    def _load_key(self, path: Path) -> None:
        """Load Ed25519 private key from PEM file."""
        from cryptography.hazmat.primitives.serialization import load_pem_private_key
        pem_data = path.read_bytes()
        self._private_key = load_pem_private_key(pem_data, password=None)
        self._public_key = self._private_key.public_key()

    def _generate_and_save(self, path: Path) -> None:
        """Generate a new Ed25519 key pair and save to PEM."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import (
            Encoding, NoEncryption, PrivateFormat, PublicFormat,
        )

        self._private_key = Ed25519PrivateKey.generate()
        self._public_key = self._private_key.public_key()

        path.parent.mkdir(parents=True, exist_ok=True)
        pem = self._private_key.private_bytes(
            Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()
        )
        path.write_bytes(pem)
        logger.info("Generated Ed25519 key pair at %s", path)

    @property
    def public_key_b64(self) -> str:
        """Return base64-encoded public key for sharing."""
        if not self._public_key:
            return ""
        from cryptography.hazmat.primitives.serialization import (
            Encoding, PublicFormat,
        )
        raw = self._public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
        return base64.b64encode(raw).decode()

    def sign(self, message: str) -> str:
        """Sign a message with Ed25519 private key. Returns base64 signature."""
        if not self._private_key:
            return ""
        sig = self._private_key.sign(message.encode())
        return base64.b64encode(sig).decode()

    @staticmethod
    def verify(message: str, signature_b64: str, public_key_b64: str) -> bool:
        """Verify an Ed25519 signature given base64 public key."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        try:
            pub_bytes = base64.b64decode(public_key_b64)
            pub_key = Ed25519PublicKey.from_public_bytes(pub_bytes)
            sig_bytes = base64.b64decode(signature_b64)
            pub_key.verify(sig_bytes, message.encode())
            return True
        except Exception:
            return False


class TrustStore:
    """Persistent store for trusted peer nodes.

    Loads/saves from a YAML file (config/mesh_peers.yaml).
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._peers: dict[str, dict[str, Any]] = {}
        if path and path.exists():
            self._load()

    def _load(self) -> None:
        """Load trust store from YAML file."""
        try:
            import yaml
            data = yaml.safe_load(self._path.read_text()) or {}
            for peer in data.get("peers", []):
                node_id = peer.get("node_id", "")
                if node_id:
                    self._peers[node_id] = peer
        except Exception as e:
            logger.warning("Failed to load trust store: %s", e)

    def save(self) -> None:
        """Save trust store to YAML file."""
        if not self._path:
            return
        try:
            import yaml
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "peers": list(self._peers.values()),
            }
            self._path.write_text(yaml.dump(data, default_flow_style=False))
        except Exception as e:
            logger.warning("Failed to save trust store: %s", e)

    def add_peer(self, node_id: str, public_key: str = "", trusted: bool = True) -> None:
        """Add or update a peer in the trust store."""
        self._peers[node_id] = {
            "node_id": node_id,
            "public_key": public_key,
            "trusted": trusted,
        }

    def remove_peer(self, node_id: str) -> None:
        """Remove a peer from the trust store."""
        self._peers.pop(node_id, None)

    def is_trusted(self, node_id: str) -> bool:
        """Check if a node is trusted."""
        peer = self._peers.get(node_id)
        return peer is not None and peer.get("trusted", False)

    def get_public_key(self, node_id: str) -> str:
        """Get the public key for a peer node."""
        peer = self._peers.get(node_id)
        return peer.get("public_key", "") if peer else ""

    def list_peers(self) -> list[dict[str, Any]]:
        """Return all peer entries."""
        return list(self._peers.values())


class MeshSecurity:
    """Security manager for mesh communications."""

    def __init__(
        self,
        shared_secret: str | None = None,
        node_identity: NodeIdentity | None = None,
        trust_store: TrustStore | None = None,
    ) -> None:
        self.auth = MeshTokenAuth(shared_secret)
        self._trusted_nodes: set[str] = set()
        self.node_identity = node_identity
        self.trust_store = trust_store or TrustStore()

    def trust_node(self, node_id: str) -> None:
        """Mark a node as trusted."""
        self._trusted_nodes.add(node_id)

    def untrust_node(self, node_id: str) -> None:
        """Remove trust for a node."""
        self._trusted_nodes.discard(node_id)

    def is_trusted(self, node_id: str) -> bool:
        """Check if a node is trusted (in-memory or trust store)."""
        if node_id in self._trusted_nodes:
            return True
        return self.trust_store.is_trusted(node_id)

    def sign_message(self, message: str) -> str:
        """Sign a message with the shared secret (HMAC-SHA256)."""
        return hmac.new(
            self.auth.token.encode(), message.encode(), hashlib.sha256
        ).hexdigest()

    def verify_signature(self, message: str, signature: str) -> bool:
        """Verify a message signature (HMAC-SHA256)."""
        expected = self.sign_message(message)
        return hmac.compare_digest(expected, signature)

    def sign_node_identity(self, timestamp: str = "") -> dict[str, str]:
        """Create identity headers for a request.

        Returns headers: X-Node-ID, X-Node-Timestamp, X-Node-Signature
        """
        if not self.node_identity:
            return {}
        ts = timestamp or str(int(time.time()))
        message = f"{self.node_identity.node_id}:{ts}"
        sig = self.node_identity.sign(message)
        return {
            "X-Node-ID": self.node_identity.node_id,
            "X-Node-Timestamp": ts,
            "X-Node-Signature": sig,
        }

    def verify_node_identity(
        self, node_id: str, timestamp: str, signature: str,
    ) -> bool:
        """Verify a node's identity signature using its stored public key."""
        pub_key = self.trust_store.get_public_key(node_id)
        if not pub_key:
            return False
        message = f"{node_id}:{timestamp}"
        return NodeIdentity.verify(message, signature, pub_key)
