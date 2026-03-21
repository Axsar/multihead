"""Claim CRUD, row conversion, and evidence/conflict operations."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from multihead.knowledge_models import (
    Claim,
    ClaimCanonical,
    ClaimScope,
    ClaimStatus,
    ClaimType,
    EntityRef,
    Provenance,
    ScopeType,
    Stability,
    ValueObject,
)

from ._helpers import _coerce_float, _get_producer_id, _now_iso, _safe_json_loads
from ._retry import _sqlite_retry
from ._claims_search import ClaimsSearchMixin
from ._claims_queries import ClaimsQueryMixin


class ClaimsMixin(ClaimsSearchMixin, ClaimsQueryMixin):
    """Mixin providing claim operations.

    Inherits search (FTS/LIKE) from ClaimsSearchMixin and
    presence/mesh/pack queries from ClaimsQueryMixin.

    These methods expect self._connect(), self._mesh_security, self._agent_id
    to be available from the main class.
    """

    # -------------------------------------------------------------------
    # Pre-insert dedup
    # -------------------------------------------------------------------

    @staticmethod
    def _word_overlap(a: str, b: str) -> float:
        """Compute word overlap ratio between two statements (Jaccard)."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    def _check_duplicate(self, claim: Claim) -> str | None:
        """Check for duplicate claims: exact key match, then semantic similarity.

        Returns:
            "skip" — duplicate exists (exact or semantic), caller should skip insert
            "supersede" — older claim found with same key, marked as superseded
            None — no duplicate, proceed with insert
        """
        key = claim.canonical.claim_key
        if not key:
            return None

        # 1. Exact key match
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT claim_id, statement, claim_status FROM claims "
                "WHERE claim_key = ? AND claim_status NOT IN ('superseded', 'rejected') "
                "ORDER BY created_at DESC LIMIT 1",
                (key,),
            ).fetchone()

        if existing:
            existing_id = existing["claim_id"]
            existing_stmt = existing["statement"] or ""

            # Exact duplicate — skip
            if existing_stmt.strip() == (claim.statement or "").strip():
                return "skip"

            # Same key, different statement — supersede old claim
            now = _now_iso()
            with self._connect() as conn:
                conn.execute(
                    "UPDATE claims SET claim_status = 'superseded', "
                    "superseded_by_claim_id = ?, updated_at = ? WHERE claim_id = ?",
                    (claim.claim_id, now, existing_id),
                )
            return "supersede"

        # 2. Semantic dedup via FTS + word overlap
        stmt = claim.statement or ""
        if len(stmt) < 20:
            return None  # Too short for meaningful similarity check

        try:
            # Extract key words for FTS query
            words = [w for w in stmt.split() if len(w) > 3][:6]
            if len(words) < 2:
                return None
            fts_query = " OR ".join(words)

            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT c.claim_id, c.statement FROM claims_fts f "
                    "JOIN claims c ON c.rowid = f.rowid "
                    "WHERE claims_fts MATCH ? "
                    "AND c.claim_status NOT IN ('superseded', 'rejected') "
                    "AND c.claim_id != ? "
                    "ORDER BY rank LIMIT 5",
                    (fts_query, claim.claim_id),
                ).fetchall()

            for row in rows:
                overlap = self._word_overlap(stmt, row["statement"] or "")
                if overlap > 0.80:
                    return "skip"  # Near-duplicate found
        except Exception:
            pass  # FTS might not be available

        return None

    # -------------------------------------------------------------------
    # Claims CRUD
    # -------------------------------------------------------------------

    @_sqlite_retry
    def insert_claim(self, claim: Claim, dedup: bool = True) -> Claim:
        # Pre-insert dedup: check for existing claim with same claim_key
        if dedup and claim.canonical.claim_key:
            existing = self._check_duplicate(claim)
            if existing == "skip":
                return claim  # Silently skip exact duplicate
            elif existing == "supersede":
                # New claim supersedes old — handled inside _check_duplicate
                pass

        # Auto-sign if mesh_security is configured and claim is unsigned
        if self._mesh_security and not claim.signature:
            claim.signature = self._mesh_security.sign_message(
                claim.canonical_json_for_signing()
            )
            claim.signed_by = self._agent_id

        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO claims "
                "(claim_id, claim_status, claim_type, scope_type, scope_id, visibility, "
                "valid_from, valid_to, claim_key, predicate, subject_json, object_json, "
                "statement, rationale, confidence, stability, importance, "
                "superseded_by_claim_id, rejection_reason, contested_reason, "
                "derived_from_json, related_json, conflicts_json, "
                "provenance_json, signature, signed_by, created_at, updated_at, observation_method, producer) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (claim.claim_id, claim.claim_status.value, claim.claim_type.value,
                 claim.scope.scope_type.value, claim.scope.scope_id,
                 claim.scope.visibility, claim.scope.valid_from.isoformat(),
                 claim.scope.valid_to.isoformat() if claim.scope.valid_to else None,
                 claim.canonical.claim_key, claim.canonical.predicate,
                 json.dumps(claim.canonical.subject.model_dump(mode="json")),
                 json.dumps(claim.canonical.object.model_dump(mode="json")),
                 claim.statement, claim.rationale or None,
                 claim.confidence, claim.stability.value if claim.stability else None,
                 claim.importance,
                 claim.superseded_by_claim_id, claim.rejection_reason, claim.contested_reason,
                 json.dumps(claim.derived_from_event_ids),
                 json.dumps(claim.related_claim_ids),
                 json.dumps(claim.conflicts_with_claim_ids),
                 json.dumps(claim.provenance.model_dump(mode="json")),
                 claim.signature, claim.signed_by,
                 now, now,
                 claim.provenance.observation_method or "",
                 _get_producer_id(claim.provenance.model_dump(mode="json"))),
            )
            # Get the rowid for indexing
            rowid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

            # Update FTS5 index
            try:
                conn.execute(
                    "INSERT INTO claims_fts(rowid, claim_key, statement) "
                    "VALUES (?, ?, ?)",
                    (rowid, claim.canonical.claim_key, claim.statement),
                )
            except Exception:
                pass  # FTS5 might not be available

            # Update vector embedding index
            try:
                self._insert_vec_embedding(conn, rowid, claim.statement)
            except Exception:
                pass  # sqlite-vec might not be available

        return claim

    def _insert_vec_embedding(self, conn, rowid: int, statement: str) -> None:
        """Insert vector embedding for a claim into claims_vec."""
        if not statement or len(statement) < 10:
            return
        # Check if claims_vec table exists
        table_check = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='claims_vec'"
        ).fetchone()
        if not table_check:
            return

        model = self._get_embedding_model_cached()
        if model is None:
            return

        import numpy as np
        emb = model.encode(statement, normalize_embeddings=True)
        conn.execute(
            "INSERT OR REPLACE INTO claims_vec(rowid, embedding) VALUES (?, ?)",
            [rowid, emb.astype(np.float32).tobytes()],
        )

    def _get_embedding_model_cached(self):
        """Lazy-load embedding model (singleton)."""
        if not hasattr(self, "_emb_model"):
            self._emb_model = None
            try:
                from sentence_transformers import SentenceTransformer
                self._emb_model = SentenceTransformer(
                    "all-MiniLM-L6-v2",
                )
            except ImportError:
                pass
        return self._emb_model

    def get_claim(self, claim_id: str) -> Claim | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM claims WHERE claim_id = ?", (claim_id,)).fetchone()
        if not row:
            return None
        return self._row_to_claim(row)

    def verify_claim(self, claim: Claim) -> bool | None:
        """Verify a claim's HMAC signature.

        Returns True if valid, False if tampered, None if no security configured
        or claim is unsigned.
        """
        if not self._mesh_security:
            return None
        if not claim.signature:
            return None
        return self._mesh_security.verify_signature(
            claim.canonical_json_for_signing(), claim.signature
        )

    @_sqlite_retry
    def update_claim_status(
        self, claim_id: str, status: ClaimStatus,
        superseded_by: str | None = None,
        reason: str | None = None,
    ) -> None:
        now = _now_iso()
        with self._connect() as conn:
            if superseded_by:
                conn.execute(
                    "UPDATE claims SET claim_status = ?, superseded_by_claim_id = ?, updated_at = ? "
                    "WHERE claim_id = ?",
                    (status.value, superseded_by, now, claim_id),
                )
            elif reason and status == ClaimStatus.CONTESTED:
                conn.execute(
                    "UPDATE claims SET claim_status = ?, contested_reason = ?, updated_at = ? "
                    "WHERE claim_id = ?",
                    (status.value, reason, now, claim_id),
                )
            else:
                conn.execute(
                    "UPDATE claims SET claim_status = ?, updated_at = ? WHERE claim_id = ?",
                    (status.value, now, claim_id),
                )

            # Remove from FTS/vec indexes when claim is no longer searchable
            if status in (ClaimStatus.SUPERSEDED, ClaimStatus.REJECTED):
                row = conn.execute(
                    "SELECT rowid FROM claims WHERE claim_id = ?", (claim_id,)
                ).fetchone()
                if row:
                    try:
                        conn.execute(
                            "INSERT INTO claims_fts(claims_fts, rowid, claim_key, statement) "
                            "VALUES('delete', ?, '', '')",
                            (row[0],),
                        )
                    except Exception:
                        pass
                    try:
                        conn.execute(
                            "DELETE FROM claims_vec WHERE rowid = ?", (row[0],)
                        )
                    except Exception:
                        pass

    def resolve_claim(self, claim_id: str) -> bool:
        """Mark a claim as resolved (handled/done). Returns True if claim existed."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE claims SET claim_status = ?, updated_at = ? WHERE claim_id = ?",
                (ClaimStatus.RESOLVED.value, _now_iso(), claim_id),
            )
            return cur.rowcount > 0

    @_sqlite_retry
    def accept_claim(self, claim_id: str) -> None:
        """Atomic: supersede old accepted claim for same key+scope, then accept new one."""
        with self._connect() as conn:
            # Get the claim to accept
            row = conn.execute("SELECT * FROM claims WHERE claim_id = ?", (claim_id,)).fetchone()
            if not row:
                raise ValueError(f"Claim {claim_id} not found")

            # Check evidence exists
            ev_count = conn.execute(
                "SELECT COUNT(*) FROM claim_evidence WHERE claim_id = ? AND stance = 'supports'",
                (claim_id,),
            ).fetchone()[0]
            if ev_count < 1:
                raise ValueError("Cannot accept claim without supporting evidence")

            scope_type = row["scope_type"]
            scope_id = row["scope_id"]
            claim_key = row["claim_key"]

            # Find existing accepted claim for same key+scope
            old = conn.execute(
                "SELECT claim_id FROM claims "
                "WHERE scope_type = ? AND scope_id = ? AND claim_key = ? AND claim_status = 'accepted'",
                (scope_type, scope_id, claim_key),
            ).fetchone()

            now = _now_iso()
            if old and old["claim_id"] != claim_id:
                # Supersede old claim first (to avoid unique index violation)
                conn.execute(
                    "UPDATE claims SET claim_status = 'superseded', superseded_by_claim_id = ?, updated_at = ? "
                    "WHERE claim_id = ?",
                    (claim_id, now, old["claim_id"]),
                )

            # Accept new claim (bypass trigger by directly setting status)
            conn.execute(
                "UPDATE claims SET claim_status = 'accepted', updated_at = ? WHERE claim_id = ?",
                (now, claim_id),
            )

    def get_accepted_claim(self, scope_type: str, scope_id: str, claim_key: str) -> Claim | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM claims "
                "WHERE scope_type = ? AND scope_id = ? AND claim_key = ? AND claim_status = 'accepted'",
                (scope_type, scope_id, claim_key),
            ).fetchone()
        if not row:
            return None
        return self._row_to_claim(row)

    def list_claims(
        self, status: str | None = None, claim_type: str | None = None,
        scope_id: str | None = None, limit: int = 100,
    ) -> list[Claim]:
        clauses = []
        params: list[Any] = []
        if status:
            clauses.append("claim_status = ?")
            params.append(status)
        if claim_type:
            clauses.append("claim_type = ?")
            params.append(claim_type)
        if scope_id:
            clauses.append("scope_id = ?")
            params.append(scope_id)
        where = " AND ".join(clauses) if clauses else "1=1"
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM claims WHERE {where} ORDER BY updated_at DESC LIMIT ?",
                params + [limit],
            ).fetchall()
        return [self._row_to_claim(r) for r in rows]

    def count_claims(self, status: str | None = "accepted") -> int:
        """Count claims, optionally filtered by status."""
        if status:
            sql = "SELECT COUNT(*) FROM claims WHERE claim_status = ?"
            params: list[Any] = [status]
        else:
            sql = "SELECT COUNT(*) FROM claims"
            params = []
        with self._connect() as conn:
            return conn.execute(sql, params).fetchone()[0]

    def list_claims_paginated(
        self, offset: int = 0, limit: int = 500,
        status: str | None = "accepted",
        exclude_key_prefixes: list[str] | None = None,
    ) -> list[Claim]:
        """Paginated claim retrieval with cursor-based offset."""
        clauses = []
        params: list[Any] = []
        if status:
            clauses.append("claim_status = ?")
            params.append(status)
        if exclude_key_prefixes:
            for prefix in exclude_key_prefixes:
                clauses.append("claim_key NOT LIKE ?")
                params.append(prefix + "%")
        where = " AND ".join(clauses) if clauses else "1=1"
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM claims WHERE {where} ORDER BY rowid ASC LIMIT ? OFFSET ?",
                params + [limit, offset],
            ).fetchall()
        return [self._row_to_claim(r) for r in rows]

    def get_claim_by_id(self, claim_id: str) -> Claim | None:
        """Fetch a single claim by ID (direct query, O(1))."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM claims WHERE claim_id = ?", (claim_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_claim(row)

    @_sqlite_retry
    def link_claim_evidence(self, claim_id: str, evidence_id: str, stance: str = "supports") -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO claim_evidence (claim_id, evidence_id, stance) VALUES (?, ?, ?)",
                (claim_id, evidence_id, stance),
            )

    @_sqlite_retry
    def add_claim_conflict(self, claim_id_a: str, claim_id_b: str, reason: str = "") -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO claim_conflicts (claim_id_a, claim_id_b, reason) VALUES (?, ?, ?)",
                (claim_id_a, claim_id_b, reason or None),
            )

    # -------------------------------------------------------------------
    # Row conversion helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _coerce_entity_ref(data: dict | str) -> EntityRef:
        """Parse EntityRef, coercing non-standard LLM output to valid form."""
        if isinstance(data, str):
            return EntityRef(entity_type="concept", entity_id=data, label=data)
        if "entity_type" in data and "entity_id" in data:
            return EntityRef.model_validate(data)
        if len(data) == 1:
            k, v = next(iter(data.items()))
            return EntityRef(entity_type=k, entity_id=str(v))
        return EntityRef(
            entity_type=data.get("type", "unknown"),
            entity_id=data.get("id", data.get("name", str(data))),
            label=data.get("label", ""),
        )

    @staticmethod
    def _coerce_value_object(data: dict | str | list) -> ValueObject:
        """Parse ValueObject, coercing non-standard LLM output to valid form."""
        if isinstance(data, str):
            return ValueObject(value_type="string", value=data)
        if isinstance(data, list):
            return ValueObject(value_type="list", value=str(data))
        if "value_type" in data:
            return ValueObject.model_validate(data)
        val = data.get("value", str(data))
        return ValueObject(value_type="string", value=str(val))

    def _row_to_claim(self, row: sqlite3.Row) -> Claim:
        try:
            claim_type = ClaimType(row["claim_type"])
        except ValueError:
            claim_type = ClaimType.FACT
        try:
            scope_type = ScopeType(row["scope_type"])
        except ValueError:
            scope_type = ScopeType.PROJECT

        try:
            stability = Stability(row["stability"]) if row["stability"] else Stability.MEDIUM
        except ValueError:
            stability = Stability.MEDIUM

        return Claim(
            claim_id=row["claim_id"],
            claim_status=ClaimStatus(row["claim_status"]),
            claim_type=claim_type,
            scope=ClaimScope(
                scope_type=scope_type,
                scope_id=row["scope_id"],
                visibility=row["visibility"],
                valid_from=datetime.fromisoformat(row["valid_from"]) if isinstance(row["valid_from"], str) else (row["valid_from"] or datetime.now(timezone.utc)),
                valid_to=datetime.fromisoformat(row["valid_to"]) if isinstance(row["valid_to"], str) and row["valid_to"] else None,
            ),
            canonical=ClaimCanonical(
                claim_key=row["claim_key"],
                subject=self._coerce_entity_ref(
                    _safe_json_loads(row["subject_json"], {}, "claim.subject")
                ),
                predicate=row["predicate"],
                object=self._coerce_value_object(
                    _safe_json_loads(row["object_json"], {}, "claim.object")
                ),
            ),
            statement=row["statement"],
            rationale=row["rationale"] or "",
            confidence=_coerce_float(row["confidence"], 0.0),
            stability=stability,
            importance=_coerce_float(row["importance"], 0.0),
            superseded_by_claim_id=row["superseded_by_claim_id"],
            rejection_reason=row["rejection_reason"],
            contested_reason=row["contested_reason"],
            derived_from_event_ids=_safe_json_loads(row["derived_from_json"], [], "claim.derived_from"),
            related_claim_ids=self._normalize_claim_id_list(
                _safe_json_loads(row["related_json"], [], "claim.related")
            ),
            conflicts_with_claim_ids=_safe_json_loads(row["conflicts_json"], [], "claim.conflicts"),
            provenance=self._coerce_provenance(
                _safe_json_loads(row["provenance_json"], {}, "claim.provenance")
            ),
            signature=row["signature"] or "" if "signature" in row.keys() else "",
            signed_by=row["signed_by"] or "" if "signed_by" in row.keys() else "",
        )

    @staticmethod
    def _normalize_claim_id_list(data: list) -> list[str]:
        """Normalize related_json to list of claim_id strings.

        Handles two formats:
        1. ["clm_123", "clm_456"] - simple list of strings
        2. [{"claim_id": "clm_123", "rel": "review_of"}] - rich format with relations
        """
        if not data:
            return []
        result = []
        for item in data:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict) and "claim_id" in item:
                result.append(item["claim_id"])
        return result

    @staticmethod
    def _coerce_provenance(data: dict) -> Provenance:
        """Parse Provenance, coercing non-standard LLM output."""
        if "produced_by" in data:
            try:
                return Provenance.model_validate(data)
            except Exception:
                pass
        source = data.get("source", data.get("agent", data.get("kind", "unknown")))
        return Provenance(produced_by={"kind": "llm", "id": str(source)})
