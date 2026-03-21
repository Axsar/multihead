"""Optional mDNS discovery for mesh nodes."""

from __future__ import annotations

import logging
import socket
import threading
import time
import urllib.request
import urllib.error
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from multihead.knowledge_store import KnowledgeStore

logger = logging.getLogger(__name__)


def get_lan_ip() -> str:
    """Return the best-guess LAN IP address for this machine.

    Uses a UDP connect trick (no actual traffic) to find the interface
    that routes to the LAN.  Falls back to gethostbyname, then 127.0.0.1.
    """
    # Approach 1: UDP connect trick — works on Linux, macOS, WSL
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # doesn't send anything
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                return ip
    except Exception:
        pass

    # Approach 2: gethostbyname
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass

    return "127.0.0.1"


class _ServiceListener:
    """Zeroconf listener that populates discovery dict when peers appear/disappear."""

    def __init__(self, discovery: MeshDiscovery) -> None:
        self._discovery = discovery

    def add_service(self, zc: Any, type_: str, name: str) -> None:
        """Called when a new service is found on the network."""
        try:
            info = zc.get_service_info(type_, name)
            if info is None:
                return
            props = {k.decode(): v.decode() if isinstance(v, bytes) else v
                     for k, v in info.properties.items()}
            node_id = props.get("node_id", name.split(".")[0])

            # Don't add ourselves
            if node_id == self._discovery.node_id:
                return

            addresses = info.parsed_addresses()
            host = addresses[0] if addresses else "unknown"

            self._discovery._discovered[node_id] = {
                "node_id": node_id,
                "host": host,
                "port": info.port,
                "source": "mdns",
                "properties": props,
                "last_seen": time.monotonic(),
            }
            logger.info("Discovered node: %s at %s:%d", node_id, host, info.port)
        except Exception as e:
            logger.warning("Error processing discovered service %s: %s", name, e)

    def remove_service(self, zc: Any, type_: str, name: str) -> None:
        """Called when a service is removed from the network."""
        node_id = name.split(".")[0]
        if node_id in self._discovery._discovered:
            del self._discovery._discovered[node_id]
            logger.info("Node removed: %s", node_id)

    def update_service(self, zc: Any, type_: str, name: str) -> None:
        """Called when a service is updated. Re-add with latest info."""
        self.add_service(zc, type_, name)


class MeshDiscovery:
    """Discover other MultiHead nodes on the local network.

    Uses zeroconf (optional dependency) for mDNS service discovery.
    Falls back gracefully if zeroconf is not installed.
    A background heartbeat thread pings each discovered peer every
    HEARTBEAT_INTERVAL seconds and removes any peer that has been
    unreachable for longer than PEER_TIMEOUT seconds.
    """

    SERVICE_TYPE = "_multihead._tcp.local."
    HEARTBEAT_INTERVAL = 30  # seconds between liveness checks
    PEER_TIMEOUT = 90        # seconds of silence before a peer is removed
    PING_TIMEOUT = 2         # HTTP connect/read timeout per ping attempt

    def __init__(
        self,
        node_id: str,
        port: int = 7337,
        knowledge_store: "KnowledgeStore | None" = None,
    ) -> None:
        # mDNS service names cannot contain colons — normalise to dashes
        self.node_id = node_id.replace(":", "-")
        self.port = port
        self.knowledge_store = knowledge_store
        self._zeroconf = None
        self._browser = None
        self._service_info = None
        self._discovered: dict[str, dict[str, Any]] = {}
        self._stop_event = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

    def _ping_peer(self, host: str, port: int) -> bool:
        """Return True if the peer's /health endpoint responds within PING_TIMEOUT."""
        url = f"http://{host}:{port}/health"
        try:
            with urllib.request.urlopen(url, timeout=self.PING_TIMEOUT) as resp:
                return resp.status < 500
        except Exception:
            return False

    def _heartbeat_loop(self) -> None:
        """Background thread: ping every peer and evict those silent >PEER_TIMEOUT.

        Also calls scan_db_for_peers() each cycle so that peers discovered
        via PresenceMonitor claims are picked up automatically.
        """
        while not self._stop_event.wait(self.HEARTBEAT_INTERVAL):
            # DB scan: pick up peers from shared knowledge.db
            if self.knowledge_store is not None:
                try:
                    added = self.scan_db_for_peers(self.knowledge_store)
                    if added:
                        logger.info("DB scan added %d new peer(s)", added)
                except Exception as exc:
                    logger.debug("DB peer scan error in heartbeat: %s", exc)

            now = time.monotonic()
            stale: list[str] = []
            # Snapshot keys to avoid mutation during iteration
            for node_id, info in list(self._discovered.items()):
                reachable = self._ping_peer(info["host"], info["port"])
                if reachable:
                    self._discovered[node_id]["last_seen"] = now
                    logger.debug("Heartbeat OK: %s", node_id)
                elif now - info.get("last_seen", 0) > self.PEER_TIMEOUT:
                    stale.append(node_id)
                    logger.info(
                        "Peer %s unreachable for >%ds — removing",
                        node_id,
                        self.PEER_TIMEOUT,
                    )
            for node_id in stale:
                self.remove_node(node_id)

    def start(self) -> bool:
        """Start mDNS discovery: register our service and browse for peers.

        Returns False if zeroconf not available.
        """
        try:
            from zeroconf import ServiceBrowser, Zeroconf, ServiceInfo

            self._zeroconf = Zeroconf()

            # Register our service with the actual LAN IP
            lan_ip = get_lan_ip()
            self._service_info = ServiceInfo(
                self.SERVICE_TYPE,
                f"{self.node_id}.{self.SERVICE_TYPE}",
                addresses=[socket.inet_aton(lan_ip)],
                port=self.port,
                properties={"node_id": self.node_id},
            )
            logger.info("Registering mDNS with LAN IP %s", lan_ip)
            self._zeroconf.register_service(self._service_info)
            logger.info("mDNS service registered: %s", self.node_id)

            # Browse for peer services
            listener = _ServiceListener(self)
            self._browser = ServiceBrowser(self._zeroconf, self.SERVICE_TYPE, listener)
            logger.info("mDNS browser started for %s", self.SERVICE_TYPE)

            # Start heartbeat thread to evict stale peers
            self._stop_event.clear()
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                name="mesh-heartbeat",
                daemon=True,
            )
            self._heartbeat_thread.start()
            logger.info("Heartbeat loop started (interval=%ds, timeout=%ds)",
                        self.HEARTBEAT_INTERVAL, self.PEER_TIMEOUT)

            return True

        except ImportError:
            logger.info("zeroconf not installed, mDNS discovery disabled")
            return False
        except Exception as e:
            logger.warning("mDNS start failed: %s", e)
            return False

    def stop(self) -> None:
        """Stop mDNS discovery, heartbeat thread, and unregister service."""
        # Signal and join the heartbeat thread first
        self._stop_event.set()
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=self.PING_TIMEOUT + 1)
        self._heartbeat_thread = None

        if self._zeroconf:
            try:
                if self._service_info:
                    self._zeroconf.unregister_service(self._service_info)
                if self._browser:
                    self._browser.cancel()
                self._zeroconf.close()
            except Exception as e:
                logger.warning("Error during mDNS shutdown: %s", e)
            self._zeroconf = None
            self._browser = None
            self._service_info = None

    def get_discovered_nodes(self) -> dict[str, dict[str, Any]]:
        """Return discovered nodes."""
        return dict(self._discovered)

    def add_manual_node(self, node_id: str, host: str, port: int) -> None:
        """Manually add a known node (for non-mDNS setups)."""
        self._discovered[node_id] = {
            "node_id": node_id,
            "host": host,
            "port": port,
            "source": "manual",
            "last_seen": time.monotonic(),
        }

    def remove_node(self, node_id: str) -> bool:
        """Remove a discovered node. Returns True if found and removed."""
        if node_id in self._discovered:
            del self._discovered[node_id]
            return True
        return False

    def scan_db_for_peers(self, store: "KnowledgeStore") -> int:
        """Discover peers via accepted presence claims in a shared knowledge.db.

        Intended as a fallback when mDNS/zeroconf is unavailable (e.g. across
        subnets or when zeroconf is not installed).  Peers already known via
        mDNS are left untouched; new entries are added with source='db'.

        Returns the number of new peers added to ``_discovered``.
        """
        try:
            peers = store.get_presence_peers(
                exclude_node_id=self.node_id,
                stale_after_secs=self.PEER_TIMEOUT,
            )
        except Exception as exc:
            logger.warning("DB peer scan failed: %s", exc)
            return 0

        now = time.monotonic()
        added = 0
        for peer in peers:
            node_id = peer["node_id"]
            if node_id in self._discovered:
                # Already known via mDNS or manual; refresh last_seen only
                self._discovered[node_id]["last_seen"] = now
                continue
            self._discovered[node_id] = {
                "node_id": node_id,
                "host": peer["host"],
                "port": peer["port"],
                "source": "db",
                "last_seen": now,
                "properties": {
                    "hostname": peer.get("hostname", ""),
                    "last_seen": peer.get("last_seen", ""),
                },
            }
            logger.info(
                "DB-discovered node: %s at %s:%d (last_seen=%s)",
                node_id, peer["host"], peer["port"], peer.get("last_seen", "?"),
            )
            added += 1
        return added
