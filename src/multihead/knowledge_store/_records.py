"""Record and evidence pointer CRUD operations."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING

from multihead.knowledge_models import EvidencePointer, Record, SpanRef

from ._retry import _sqlite_retry

if TYPE_CHECKING:
    pass


class RecordsMixin:
    """Mixin providing record and evidence pointer operations."""

    # These methods expect self._connect() to be available from the main class.

    # -------------------------------------------------------------------
    # Records
    # -------------------------------------------------------------------

    @_sqlite_retry
    def insert_record(self, record: Record) -> Record:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO records (record_id, uri, sha256, mime, captured_at, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (record.record_id, record.uri, record.sha256 or None,
                 record.mime or None, record.captured_at.isoformat(),
                 record.created_at.isoformat()),
            )
        return record

    def get_record(self, record_id: str) -> Record | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM records WHERE record_id = ?", (record_id,)).fetchone()
        if not row:
            return None
        return Record(
            record_id=row["record_id"], uri=row["uri"],
            sha256=row["sha256"] or "", mime=row["mime"] or "",
            captured_at=datetime.fromisoformat(row["captured_at"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def list_records(self, since: datetime | None = None, limit: int = 100) -> list[Record]:
        with self._connect() as conn:
            if since:
                rows = conn.execute(
                    "SELECT * FROM records WHERE captured_at >= ? ORDER BY captured_at DESC LIMIT ?",
                    (since.isoformat(), limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM records ORDER BY captured_at DESC LIMIT ?", (limit,)
                ).fetchall()
        return [
            Record(record_id=r["record_id"], uri=r["uri"], sha256=r["sha256"] or "",
                   mime=r["mime"] or "", captured_at=datetime.fromisoformat(r["captured_at"]),
                   created_at=datetime.fromisoformat(r["created_at"]))
            for r in rows
        ]

    def count_records_since(self, since: datetime) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM records WHERE captured_at >= ?", (since.isoformat(),)
            ).fetchone()
        return row[0]

    # -------------------------------------------------------------------
    # Evidence pointers
    # -------------------------------------------------------------------

    @_sqlite_retry
    def insert_evidence(self, pointer: EvidencePointer) -> EvidencePointer:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO evidence_pointers "
                "(evidence_id, record_id, uri, sha256, span_start, span_end, span_unit, "
                "page, line_start, line_end, json_path, quote, captured_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (pointer.evidence_id, pointer.record_id,
                 pointer.uri or None, pointer.sha256 or None,
                 pointer.span.start if pointer.span else None,
                 pointer.span.end if pointer.span else None,
                 pointer.span.unit if pointer.span else None,
                 pointer.locator.page if pointer.locator else None,
                 pointer.locator.line_start if pointer.locator else None,
                 pointer.locator.line_end if pointer.locator else None,
                 pointer.locator.json_path if pointer.locator else None,
                 pointer.quote or None,
                 pointer.captured_at.isoformat()),
            )
        return pointer

    def get_evidence(self, evidence_id: str) -> EvidencePointer | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM evidence_pointers WHERE evidence_id = ?", (evidence_id,)
            ).fetchone()
        if not row:
            return None
        return self._row_to_evidence(row)

    def get_evidence_for_record(self, record_id: str) -> list[EvidencePointer]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM evidence_pointers WHERE record_id = ?", (record_id,)
            ).fetchall()
        return [self._row_to_evidence(r) for r in rows]

    def _row_to_evidence(self, row: sqlite3.Row) -> EvidencePointer:
        span = None
        if row["span_start"] is not None:
            span = SpanRef(start=row["span_start"], end=row["span_end"], unit=row["span_unit"] or "chars")
        return EvidencePointer(
            evidence_id=row["evidence_id"], record_id=row["record_id"],
            uri=row["uri"] or "", sha256=row["sha256"] or "",
            span=span, quote=row["quote"] or "",
            captured_at=datetime.fromisoformat(row["captured_at"]),
        )
