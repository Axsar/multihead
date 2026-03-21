"""KnowledgeStore class — the main entry point composing all mixins."""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

from ._claims import ClaimsMixin
from ._events import EventsMixin
from ._inbox import InboxMixin
from ._links import LinksMixin
from ._records import RecordsMixin
from ._retry import _BUSY_TIMEOUT_MS, _WAL_CHECKPOINT_INTERVAL
from ._schema import _SCHEMA, _TRIGGERS

logger = logging.getLogger(__name__)


class KnowledgeStore(
    RecordsMixin,
    EventsMixin,
    ClaimsMixin,
    LinksMixin,
    InboxMixin,
):
    """SQLite-backed store for records, events, claims, links, and evidence."""

    def __init__(
        self,
        db_path: Path,
        mesh_security: Any | None = None,
        agent_id: str = "",
    ) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_count = 0
        self._write_lock = threading.Lock()
        self._mesh_security = mesh_security
        self._agent_id = agent_id
        self._init_db()

    def _maybe_checkpoint(self) -> None:
        """Increment write counter and run a passive WAL checkpoint every 1000 writes.

        PASSIVE mode does not block readers/writers -- it only checkpoints frames
        that are not currently read by any connection, so it is safe to call after
        every successful write without any performance penalty.
        """
        with self._write_lock:
            self._write_count += 1
            if self._write_count % _WAL_CHECKPOINT_INTERVAL == 0:
                try:
                    with self._connect() as conn:
                        conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                    logger.debug(
                        "WAL checkpoint triggered after %d writes on %s",
                        self._write_count,
                        self.db_path,
                    )
                except sqlite3.OperationalError as exc:
                    logger.warning("WAL checkpoint failed (non-fatal): %s", exc)

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            conn.executescript(_SCHEMA)
            conn.executescript(_TRIGGERS)
            # Migration: add signature columns to existing DBs
            cols = {r[1] for r in conn.execute("PRAGMA table_info(claims)").fetchall()}
            if "signature" not in cols:
                conn.execute("ALTER TABLE claims ADD COLUMN signature TEXT DEFAULT ''")
            if "signed_by" not in cols:
                conn.execute("ALTER TABLE claims ADD COLUMN signed_by TEXT DEFAULT ''")
            if "observation_method" not in cols:
                conn.execute("ALTER TABLE claims ADD COLUMN observation_method TEXT DEFAULT ''")
            if "producer" not in cols:
                conn.execute("ALTER TABLE claims ADD COLUMN producer TEXT DEFAULT ''")
                # Backfill from provenance_json — handles both dict and string formats
                conn.execute("""
                    UPDATE claims SET producer = COALESCE(
                        CASE
                            WHEN json_valid(provenance_json) AND json_type(provenance_json, '$.produced_by') = 'object'
                                THEN json_extract(provenance_json, '$.produced_by.id')
                            WHEN json_valid(provenance_json) AND json_type(provenance_json, '$.produced_by') = 'text'
                                THEN json_extract(provenance_json, '$.produced_by')
                            ELSE ''
                        END,
                        ''
                    )
                    WHERE producer IS NULL OR producer = ''
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_claims_producer ON claims(producer)")
                logger.info("Migration: added producer column and backfilled from provenance_json")
            # Migration: add claim_interactions table for existing DBs
            existing_tables = {
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "claim_interactions" not in existing_tables:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS claim_interactions (
                        id            INTEGER PRIMARY KEY AUTOINCREMENT,
                        claim_id      TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
                        agent_id      TEXT NOT NULL,
                        action        TEXT NOT NULL CHECK (action IN ('read','responded','dismissed','acknowledged','resolved')),
                        response_claim_id TEXT,
                        context       TEXT,
                        created_at    TEXT NOT NULL,
                        UNIQUE(claim_id, agent_id, action)
                    );
                    CREATE INDEX IF NOT EXISTS idx_ci_claim ON claim_interactions(claim_id);
                    CREATE INDEX IF NOT EXISTS idx_ci_agent ON claim_interactions(agent_id);
                    CREATE INDEX IF NOT EXISTS idx_ci_agent_action ON claim_interactions(agent_id, action);
                """)
            # Migration: add participants table for existing DBs
            if "participants" not in existing_tables:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS participants (
                        participant_id   TEXT PRIMARY KEY,
                        name             TEXT NOT NULL UNIQUE,
                        agent_type       TEXT NOT NULL DEFAULT 'shell',
                        context_hash     TEXT NOT NULL,
                        created_at       TEXT NOT NULL,
                        last_seen_at     TEXT NOT NULL,
                        metadata_json    TEXT NOT NULL DEFAULT '{}'
                    );
                    CREATE INDEX IF NOT EXISTS idx_participants_context
                        ON participants(context_hash);
                """)
            # Migration: add claim_type indexes for event_watcher performance
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_claims_type "
                "ON claims(claim_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_claims_type_status "
                "ON claims(claim_type, claim_status, created_at)"
            )
            # FTS5 full-text search index for fast keyword queries
            self._ensure_fts(conn)

    @staticmethod
    def _ensure_fts(conn: sqlite3.Connection) -> None:
        """Create FTS5 virtual table for fast full-text search on claims.

        Uses content-sync mode (content=claims) so the FTS index
        automatically stays in sync with the claims table.
        """
        try:
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS claims_fts
                USING fts5(
                    claim_key, statement,
                    content=claims, content_rowid=rowid
                )
            """)
            # Populate FTS if empty (first time or after schema change)
            count = conn.execute("SELECT COUNT(*) FROM claims_fts").fetchone()[0]
            if count == 0:
                claims_count = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
                if claims_count > 0:
                    conn.execute("""
                        INSERT INTO claims_fts(rowid, claim_key, statement)
                        SELECT rowid, claim_key, statement FROM claims
                    """)
                    logger.info("FTS5 index populated with %d claims", claims_count)
        except Exception as e:
            # FTS5 might not be compiled into SQLite -- non-fatal
            logger.debug("FTS5 setup skipped: %s", e)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn
