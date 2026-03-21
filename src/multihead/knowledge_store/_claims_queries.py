"""Claim query operations: presence peers, mesh replication, and pack queries."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from multihead.knowledge_models import Claim

from ._helpers import logger


class ClaimsQueryMixin:
    """Mixin providing presence, mesh replication, and pack query operations.

    Expects self._connect() and self._row_to_claim() from the main class.
    """

    # -------------------------------------------------------------------
    # Presence queries
    # -------------------------------------------------------------------

    # Presence-claim key prefix used by mesh.presence.PresenceMonitor
    _PRESENCE_KEY_PREFIX = "mesh.presence"

    def get_presence_peers(
        self,
        exclude_node_id: str | None = None,
        stale_after_secs: float = 90.0,
    ) -> list[dict[str, Any]]:
        """Return online peers discovered via accepted presence claims in the DB.

        Used as a shared-DB fallback when mDNS/zeroconf is unavailable.
        Each returned dict has keys: node_id, host, port, hostname, last_seen.
        """
        prefix = self._PRESENCE_KEY_PREFIX + "."
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT claim_key, object_json FROM claims"
                " WHERE claim_status = 'accepted' AND claim_key LIKE ?"
                " ORDER BY updated_at DESC",
                (prefix + "%",),
            ).fetchall()

        now = datetime.now(timezone.utc)
        peers: list[dict[str, Any]] = []
        for row in rows:
            try:
                obj = json.loads(row["object_json"])
                presence = obj.get("value", {})
                if not isinstance(presence, dict):
                    continue
                if presence.get("status") != "online":
                    continue

                node_id = row["claim_key"][len(prefix):]
                if not node_id:
                    continue
                if exclude_node_id and node_id == exclude_node_id:
                    continue

                # Drop stale entries
                last_seen_str = presence.get("last_seen", "")
                if last_seen_str:
                    try:
                        last_seen = datetime.fromisoformat(last_seen_str)
                        age = (now - last_seen).total_seconds()
                        if age > stale_after_secs:
                            continue
                    except (ValueError, TypeError):
                        pass  # unknown freshness -- include anyway

                peers.append({
                    "node_id": node_id,
                    "host": presence.get("hostname", "unknown"),
                    "port": presence.get("port", 7337),
                    "hostname": presence.get("hostname", "unknown"),
                    "last_seen": last_seen_str,
                })
            except Exception as exc:
                logger.warning(
                    "Skipping malformed presence claim %s: %s",
                    row["claim_key"], exc,
                )
        return peers

    # -------------------------------------------------------------------
    # Mesh replication queries
    # -------------------------------------------------------------------

    def get_shared_claims_since(
        self,
        since: datetime | None = None,
        scope_id: str | None = None,
        limit: int = 100,
    ) -> list[Claim]:
        """Return shared claims for mesh replication.

        Only returns claims with visibility='shared', ordered by updated_at.
        """
        clauses = ["visibility = 'shared'"]
        params: list[Any] = []
        if since:
            clauses.append("updated_at >= ?")
            params.append(since.isoformat())
        if scope_id:
            clauses.append("scope_id = ?")
            params.append(scope_id)
        where = " AND ".join(clauses)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM claims WHERE {where} ORDER BY updated_at ASC LIMIT ?",
                params + [limit],
            ).fetchall()
        return [self._row_to_claim(r) for r in rows]

    # -------------------------------------------------------------------
    # Pack queries (claims)
    # -------------------------------------------------------------------

    def get_accepted_claims_for_pack(
        self, scope_id: str | None = None, since: datetime | None = None,
        limit: int = 5000,
    ) -> list[Claim]:
        clauses = ["claim_status = 'accepted'"]
        params: list[Any] = []
        if scope_id:
            clauses.append("scope_id = ?")
            params.append(scope_id)
        if since:
            clauses.append("updated_at >= ?")
            params.append(since.isoformat())
        where = " AND ".join(clauses)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM claims WHERE {where} ORDER BY importance DESC LIMIT ?",
                params + [limit],
            ).fetchall()
        return [self._row_to_claim(r) for r in rows]
