"""Session presence monitor: emits heartbeat claims to knowledge.db."""

from __future__ import annotations

import asyncio
import logging
import socket
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from multihead.knowledge_models import (
    Claim,
    ClaimCanonical,
    ClaimScope,
    ClaimStatus,
    ClaimType,
    EntityRef,
    EvidencePointer,
    Provenance,
    Record,
    ScopeType,
    Stability,
    ValueObject,
)

if TYPE_CHECKING:
    from multihead.knowledge_store import KnowledgeStore

logger = logging.getLogger(__name__)

_PRESENCE_KEY_PREFIX = "mesh.presence"
_STALE_THRESHOLD_SECS = 90.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PresenceMonitor:
    """Periodically emits a 'node is online' presence claim to knowledge.db.

    Every *interval* seconds the monitor inserts a new volatile presence claim
    and calls ``accept_claim`` so it supersedes the previous heartbeat.  On
    ``stop()`` a final 'offline' claim is written before the loop exits.

    Usage::

        monitor = PresenceMonitor(node_id="my-node", store=knowledge_store)
        await monitor.start()
        # … later …
        await monitor.stop()
    """

    def __init__(
        self,
        node_id: str,
        store: KnowledgeStore,
        interval: float = 30.0,
        port: int = 7337,
        scope_id: str = "multihead",
    ) -> None:
        self.node_id = node_id
        self._store = store
        self.interval = interval
        self.port = port
        self.scope_id = scope_id
        self._task: asyncio.Task | None = None
        self._hostname = socket.gethostname()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the heartbeat loop (idempotent)."""
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._heartbeat_loop(), name=f"presence-{self.node_id}")
        logger.info("PresenceMonitor started for node %s (interval=%ss)", self.node_id, self.interval)

    async def stop(self) -> None:
        """Stop the heartbeat loop and emit a final offline claim."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        try:
            self._emit_claim(status="offline")
        except Exception as exc:
            logger.warning("Failed to emit offline claim for %s: %s", self.node_id, exc)

        logger.info("PresenceMonitor stopped for node %s", self.node_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        while True:
            try:
                self._emit_claim(status="online")
            except Exception as exc:
                logger.warning("Presence heartbeat failed for %s: %s", self.node_id, exc)
            try:
                self._scan_stale()
            except Exception as exc:
                logger.warning("Stale-session scan failed: %s", exc)
            await asyncio.sleep(self.interval)

    def _scan_stale(self) -> None:
        """Mark peer sessions as 'absent' when their last_seen exceeds the stale threshold."""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=_STALE_THRESHOLD_SECS)
        claims = self._store.list_claims(scope_id=self.scope_id, limit=500)
        for claim in claims:
            key = claim.canonical.claim_key if claim.canonical else None
            if not key or not key.startswith(_PRESENCE_KEY_PREFIX + "."):
                continue
            peer_node_id = key[len(_PRESENCE_KEY_PREFIX) + 1 :]
            if peer_node_id == self.node_id:
                continue  # never mark ourselves stale
            try:
                value = claim.canonical.object.value if claim.canonical and claim.canonical.object else {}
                current_status = value.get("status", "")
                if current_status == "absent":
                    continue  # already marked absent
                last_seen_raw = value.get("last_seen")
                if not last_seen_raw:
                    continue
                last_seen = datetime.fromisoformat(last_seen_raw)
                if last_seen < cutoff:
                    self._emit_absent_for(
                        node_id=peer_node_id,
                        hostname=value.get("hostname", peer_node_id),
                        port=value.get("port", 0),
                    )
                    logger.info(
                        "Marked stale peer absent: node=%s last_seen=%s",
                        peer_node_id,
                        last_seen_raw,
                    )
            except Exception as exc:
                logger.debug("Skipping presence claim %s during stale scan: %s", key, exc)

    def _emit_absent_for(self, node_id: str, hostname: str, port: int) -> None:
        """Emit an 'absent' presence claim on behalf of a stale peer node."""
        now = datetime.now(timezone.utc)
        claim = Claim(
            claim_type=ClaimType.FACT,
            claim_status=ClaimStatus.PROPOSED,
            scope=ClaimScope(
                scope_type=ScopeType.AGENT,
                scope_id=self.scope_id,
                visibility="shared",
                valid_from=now,
            ),
            canonical=ClaimCanonical(
                claim_key=f"{_PRESENCE_KEY_PREFIX}.{node_id}",
                subject=EntityRef(entity_type="node", entity_id=node_id),
                predicate="presence_status",
                object=ValueObject(
                    value_type="json",
                    value={
                        "status": "absent",
                        "port": port,
                        "hostname": hostname,
                        "last_seen": now.isoformat(),
                    },
                ),
            ),
            statement=f"Node {node_id!r} on {hostname} is absent (stale heartbeat).",
            confidence=0.9,
            stability=Stability.VOLATILE,
            provenance=Provenance(
                produced_by={"kind": "agent", "id": "presence_monitor"},
            ),
        )
        self._store.insert_claim(claim)
        self._add_self_evidence(claim.claim_id)
        self._store.accept_claim(claim.claim_id)

    def _add_self_evidence(self, claim_id: str) -> None:
        """Add a self-referencing evidence pointer so accept_claim succeeds."""
        record = Record(uri=f"presence://{self.node_id}", mime="application/json")
        self._store.insert_record(record)
        ev = EvidencePointer(record_id=record.record_id, uri=record.uri)
        self._store.insert_evidence(ev)
        self._store.link_claim_evidence(claim_id, ev.evidence_id, stance="supports")

    def _emit_claim(self, status: str) -> None:
        """Insert a presence claim and accept it.

        Purges old superseded heartbeats to prevent DB bloat — presence
        claims are ephemeral and have no audit value.
        """
        # Purge old superseded presence claims for this node (keep last 5)
        key = f"{_PRESENCE_KEY_PREFIX}.{self.node_id}"
        try:
            with self._store._connect() as conn:
                conn.execute(
                    "DELETE FROM claims WHERE claim_key = ? AND claim_status = 'superseded' "
                    "AND claim_id NOT IN ("
                    "  SELECT claim_id FROM claims WHERE claim_key = ? "
                    "  AND claim_status = 'superseded' ORDER BY created_at DESC LIMIT 5"
                    ")",
                    (key, key),
                )
        except Exception:
            pass  # Non-critical cleanup

        now = datetime.now(timezone.utc)
        claim = Claim(
            claim_type=ClaimType.FACT,
            claim_status=ClaimStatus.PROPOSED,
            scope=ClaimScope(
                scope_type=ScopeType.AGENT,
                scope_id=self.scope_id,
                visibility="shared",
                valid_from=now,
            ),
            canonical=ClaimCanonical(
                claim_key=f"{_PRESENCE_KEY_PREFIX}.{self.node_id}",
                subject=EntityRef(entity_type="node", entity_id=self.node_id),
                predicate="presence_status",
                object=ValueObject(
                    value_type="json",
                    value={
                        "status": status,
                        "port": self.port,
                        "hostname": self._hostname,
                        "last_seen": now.isoformat(),
                    },
                ),
            ),
            statement=(
                f"Node {self.node_id!r} on {self._hostname} is {status} at port {self.port}."
            ),
            confidence=1.0,
            stability=Stability.VOLATILE,
            provenance=Provenance(
                produced_by={"kind": "agent", "id": "presence_monitor"},
            ),
        )
        self._store.insert_claim(claim)
        self._add_self_evidence(claim.claim_id)
        self._store.accept_claim(claim.claim_id)
        logger.debug("Presence claim emitted: node=%s status=%s", self.node_id, status)
