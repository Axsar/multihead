"""Inbox, interaction tracking, and response query operations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from multihead.knowledge_models import Claim

from ._helpers import _now_iso
from ._retry import _sqlite_retry


class InboxMixin:
    """Mixin providing inbox, interaction, and response query operations."""

    # These methods expect self._connect() and self._row_to_claim() to be
    # available from the main class (via ClaimsMixin).

    # -------------------------------------------------------------------
    # Participant registry
    # -------------------------------------------------------------------

    def register_participant(
        self,
        name: str,
        context_hash: str,
        agent_type: str = "shell",
        metadata: dict[str, str] | None = None,
    ) -> "Participant":
        """Register or update a participant. Returns existing if context_hash matches."""
        from multihead.knowledge_models import Participant

        now = _now_iso()
        meta_json = json.dumps(metadata or {})

        with self._connect() as conn:
            # Look up by context_hash first (same environment = same participant)
            row = conn.execute(
                "SELECT participant_id, name, agent_type, context_hash, "
                "created_at, last_seen_at, metadata_json "
                "FROM participants WHERE context_hash = ?",
                (context_hash,),
            ).fetchone()

            if row:
                # Update last_seen and metadata
                conn.execute(
                    "UPDATE participants SET last_seen_at = ?, metadata_json = ? "
                    "WHERE participant_id = ?",
                    (now, meta_json, row[0]),
                )
                return Participant(
                    participant_id=row[0],
                    name=row[1],
                    agent_type=row[2],
                    context_hash=row[3],
                    metadata=metadata or json.loads(row[6]),
                )

            # New participant
            p = Participant(
                name=name,
                agent_type=agent_type,
                context_hash=context_hash,
                metadata=metadata or {},
            )
            conn.execute(
                "INSERT INTO participants "
                "(participant_id, name, agent_type, context_hash, "
                "created_at, last_seen_at, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (p.participant_id, p.name, p.agent_type, p.context_hash,
                 now, now, meta_json),
            )
            return p

    def get_participant(self, participant_id: str) -> "Participant | None":
        """Look up a participant by ID."""
        from multihead.knowledge_models import Participant

        with self._connect() as conn:
            row = conn.execute(
                "SELECT participant_id, name, agent_type, context_hash, "
                "created_at, last_seen_at, metadata_json "
                "FROM participants WHERE participant_id = ?",
                (participant_id,),
            ).fetchone()
            if not row:
                return None
            return Participant(
                participant_id=row[0],
                name=row[1],
                agent_type=row[2],
                context_hash=row[3],
                metadata=json.loads(row[6]) if row[6] else {},
            )

    def list_participants(self) -> list["Participant"]:
        """List all registered participants."""
        from multihead.knowledge_models import Participant

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT participant_id, name, agent_type, context_hash, "
                "created_at, last_seen_at, metadata_json "
                "FROM participants ORDER BY last_seen_at DESC",
            ).fetchall()
            return [
                Participant(
                    participant_id=r[0],
                    name=r[1],
                    agent_type=r[2],
                    context_hash=r[3],
                    metadata=json.loads(r[6]) if r[6] else {},
                )
                for r in rows
            ]

    # -------------------------------------------------------------------
    # Inbox / Response Queries
    # -------------------------------------------------------------------

    def get_responses_to_claim(self, claim_id: str, limit: int = 50) -> list[Claim]:
        """Get claims that are responses to a specific claim.

        Searches related_json for references to the claim_id.

        Args:
            claim_id: The claim to find responses for
            limit: Maximum number of responses to return

        Returns:
            List of claims that reference the given claim_id
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM claims WHERE related_json LIKE ? "
                "ORDER BY created_at DESC LIMIT ?",
                (f'%{claim_id}%', limit),
            ).fetchall()
        return [self._row_to_claim(r) for r in rows]

    def get_pending_messages(
        self,
        session_id: str,
        scope_id: str | None = None,
        max_age_hours: int = 24,
        limit: int = 50,
    ) -> list[Claim]:
        """Get pending question/request claims for a session.

        Finds question/request type claims that haven't been answered yet,
        optionally filtered by scope and age.

        Args:
            session_id: The session to check messages for
            scope_id: Optional scope filter (e.g., 'myproject', 'multihead')
            max_age_hours: Ignore claims older than this (default 24h)
            limit: Maximum number of messages to return

        Returns:
            List of pending question/request claims
        """
        from datetime import datetime, timedelta, timezone

        cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).isoformat()

        with self._connect() as conn:
            if scope_id:
                rows = conn.execute(
                    "SELECT * FROM claims "
                    "WHERE claim_type IN ('question', 'request') "
                    "AND claim_status IN ('proposed', 'accepted') "
                    "AND scope_id = ? "
                    "AND created_at > ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (scope_id, cutoff, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM claims "
                    "WHERE claim_type IN ('question', 'request') "
                    "AND claim_status IN ('proposed', 'accepted') "
                    "AND created_at > ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (cutoff, limit),
                ).fetchall()

        # Filter out questions from this session (don't show own questions)
        claims = [self._row_to_claim(r) for r in rows]
        return [
            c for c in claims
            if c.provenance.produced_by.get("id") != session_id
        ]

    def get_claims_by_relation(
        self,
        claim_id: str,
        relation_type: str | None = None,
        limit: int = 50,
    ) -> list[Claim]:
        """Get claims related to a specific claim via related_json.

        Args:
            claim_id: The claim to find relations for
            relation_type: Optional filter by relation type (e.g., 'response_to', 'supports')
            limit: Maximum number of claims to return

        Returns:
            List of related claims
        """
        with self._connect() as conn:
            if relation_type:
                # Search for both claim_id and relation_type in related_json
                rows = conn.execute(
                    "SELECT * FROM claims "
                    "WHERE related_json LIKE ? AND related_json LIKE ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (f'%{claim_id}%', f'%{relation_type}%', limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM claims WHERE related_json LIKE ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (f'%{claim_id}%', limit),
                ).fetchall()
        return [self._row_to_claim(r) for r in rows]

    # ---- Claim Interaction Tracking ------------------------------------

    @_sqlite_retry
    def record_interaction(
        self,
        claim_id: str,
        agent_id: str,
        action: str,
        response_claim_id: str | None = None,
        context: str | None = None,
    ) -> bool:
        """Record that an agent performed an action on a claim.

        Actions: 'read', 'responded', 'dismissed', 'acknowledged'.
        Uses INSERT OR IGNORE so duplicate (claim_id, agent_id, action) is a no-op.

        Returns:
            True if a new interaction was recorded, False if already existed.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO claim_interactions "
                "(claim_id, agent_id, action, response_claim_id, context, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (claim_id, agent_id, action, response_claim_id, context, _now_iso()),
            )
            return cursor.rowcount > 0

    def has_interacted(
        self,
        claim_id: str,
        agent_id: str,
        action: str | None = None,
    ) -> bool:
        """Check if an agent has already interacted with a claim.

        Args:
            claim_id: The claim to check
            agent_id: The agent to check
            action: Optional specific action to check (e.g., 'responded').
                    If None, checks for any interaction.
        """
        with self._connect() as conn:
            if action:
                row = conn.execute(
                    "SELECT 1 FROM claim_interactions "
                    "WHERE claim_id = ? AND agent_id = ? AND action = ?",
                    (claim_id, agent_id, action),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT 1 FROM claim_interactions "
                    "WHERE claim_id = ? AND agent_id = ?",
                    (claim_id, agent_id),
                ).fetchone()
            return row is not None

    def get_unhandled_claims(
        self,
        agent_id: str,
        claim_types: list[str] | None = None,
        scope_id: str | None = None,
        max_age_hours: int = 48,
        limit: int = 20,
        key_prefixes: list[str] | None = None,
    ) -> list[Claim]:
        """Get claims that an agent has NOT yet interacted with.

        This is the core inbox query -- returns claims needing attention,
        excluding anything this agent has already read/responded/dismissed.

        Args:
            agent_id: The agent checking their inbox
            claim_types: Filter by claim type (e.g., ['question', 'request']).
                When key_prefixes is also set, results match EITHER filter.
            scope_id: Optional scope filter. None = all scopes.
            max_age_hours: Ignore claims older than this
            limit: Max results
            key_prefixes: Optional claim_key prefix filter (e.g.,
                ['action.', 'solve.consensus.']). Matches via LIKE 'prefix%'.

        Returns:
            List of claims not yet handled by this agent.
        """
        from datetime import timedelta

        cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).isoformat()

        # Build WHERE conditions
        conditions = [
            "c.created_at > ?",
            "c.claim_status IN ('proposed', 'accepted')",
        ]
        where_params: list[Any] = [cutoff]

        # claim_types and key_prefixes are OR'd together so either match surfaces items
        type_and_prefix_parts: list[str] = []
        if claim_types:
            placeholders = ",".join("?" for _ in claim_types)
            type_and_prefix_parts.append(f"c.claim_type IN ({placeholders})")
            where_params.extend(claim_types)
        if key_prefixes:
            prefix_clauses = []
            for prefix in key_prefixes:
                prefix_clauses.append("c.claim_key LIKE ?")
                where_params.append(f"{prefix}%")
            type_and_prefix_parts.append(f"({' OR '.join(prefix_clauses)})")
        if type_and_prefix_parts:
            conditions.append(f"({' OR '.join(type_and_prefix_parts)})")

        if scope_id:
            conditions.append("c.scope_id = ?")
            where_params.append(scope_id)

        # Exclude claims produced by this agent
        conditions.append(
            "json_extract(c.provenance_json, '$.produced_by.id') != ?"
        )
        where_params.append(agent_id)

        where = " AND ".join(conditions)

        query = f"""
            SELECT c.* FROM claims c
            LEFT JOIN claim_interactions ci
                ON c.claim_id = ci.claim_id AND ci.agent_id = ?
            WHERE {where}
              AND ci.id IS NULL
            ORDER BY c.created_at DESC
            LIMIT ?
        """

        # Parameter order: JOIN agent_id first, then WHERE params, then LIMIT
        params: list[Any] = [agent_id] + where_params + [limit]

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_claim(r) for r in rows]

    def get_interactions_for_claim(
        self,
        claim_id: str,
    ) -> list[dict[str, Any]]:
        """Get all interactions recorded for a claim.

        Useful for seeing which agents have seen/responded to a claim.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM claim_interactions WHERE claim_id = ? "
                "ORDER BY created_at",
                (claim_id,),
            ).fetchall()
        return [
            {
                "id": r[0],
                "claim_id": r[1],
                "agent_id": r[2],
                "action": r[3],
                "response_claim_id": r[4],
                "context": r[5],
                "created_at": r[6],
            }
            for r in rows
        ]
