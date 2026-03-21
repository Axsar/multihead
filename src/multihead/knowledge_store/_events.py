"""Knowledge event CRUD operations."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from multihead.knowledge_models import (
    ActorRef,
    EntityRef,
    EventStatus,
    EventType,
    KnowledgeEvent,
    Provenance,
    TimeBlock,
    TimePrecision,
)

from ._helpers import _now_iso, _safe_json_loads
from ._retry import _sqlite_retry


class EventsMixin:
    """Mixin providing knowledge event operations."""

    # These methods expect self._connect() to be available from the main class.

    # -------------------------------------------------------------------
    # Knowledge events
    # -------------------------------------------------------------------

    @_sqlite_retry
    def insert_event(self, event: KnowledgeEvent) -> KnowledgeEvent:
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO knowledge_events "
                "(event_id, event_status, event_type, title, summary, happened_at, ended_at, "
                "tz, time_precision, actors_json, entities_json, tags_json, topic_ids_json, "
                "metrics_json, caused_by_json, supersedes_json, duplicates_json, "
                "provenance_json, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (event.event_id, event.event_status.value, event.event_type.value,
                 event.title, event.summary or None,
                 event.time.happened_at.isoformat(),
                 event.time.ended_at.isoformat() if event.time.ended_at else None,
                 event.time.timezone, event.time.time_precision.value,
                 json.dumps([a.model_dump(mode="json") for a in event.actors]),
                 json.dumps([e.model_dump(mode="json") for e in event.entities]),
                 json.dumps(event.tags),
                 json.dumps(event.topic_ids),
                 json.dumps(event.metrics),
                 json.dumps(event.caused_by_event_ids),
                 json.dumps(event.supersedes_event_ids),
                 json.dumps(event.duplicates_event_ids),
                 json.dumps(event.provenance.model_dump(mode="json")),
                 now, now),
            )
        return event

    def get_event(self, event_id: str) -> KnowledgeEvent | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM knowledge_events WHERE event_id = ?", (event_id,)
            ).fetchone()
        if not row:
            return None
        return self._row_to_event(row)

    @_sqlite_retry
    def update_event_status(self, event_id: str, status: EventStatus) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE knowledge_events SET event_status = ?, updated_at = ? WHERE event_id = ?",
                (status.value, _now_iso(), event_id),
            )

    def list_events(
        self, since: datetime | None = None, status: str | None = None,
        event_type: str | None = None, limit: int = 100,
    ) -> list[KnowledgeEvent]:
        clauses = []
        params: list[Any] = []
        if since:
            clauses.append("happened_at >= ?")
            params.append(since.isoformat())
        if status:
            clauses.append("event_status = ?")
            params.append(status)
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        where = " AND ".join(clauses) if clauses else "1=1"
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM knowledge_events WHERE {where} ORDER BY happened_at DESC LIMIT ?",
                params + [limit],
            ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def get_event_by_id(self, event_id: str) -> KnowledgeEvent | None:
        """Fetch a single event by ID (direct query, O(1))."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM knowledge_events WHERE event_id = ?", (event_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_event(row)

    @_sqlite_retry
    def link_event_evidence(self, event_id: str, evidence_id: str, role: str = "supports") -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO event_evidence (event_id, evidence_id, role) VALUES (?, ?, ?)",
                (event_id, evidence_id, role),
            )

    @_sqlite_retry
    def link_events(self, from_id: str, to_id: str, link_type: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO event_links (from_event_id, to_event_id, link_type) VALUES (?, ?, ?)",
                (from_id, to_id, link_type),
            )

    def _row_to_event(self, row) -> KnowledgeEvent:
        ended_at = datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None
        return KnowledgeEvent(
            event_id=row["event_id"],
            event_status=EventStatus(row["event_status"]),
            event_type=EventType(row["event_type"]),
            title=row["title"],
            summary=row["summary"] or "",
            time=TimeBlock(
                happened_at=datetime.fromisoformat(row["happened_at"]),
                ended_at=ended_at,
                timezone=row["tz"],
                time_precision=TimePrecision(row["time_precision"]),
            ),
            actors=[ActorRef(**a) for a in _safe_json_loads(row["actors_json"], [], "event.actors")],
            entities=[EntityRef(**e) for e in _safe_json_loads(row["entities_json"], [], "event.entities")],
            tags=_safe_json_loads(row["tags_json"], [], "event.tags"),
            topic_ids=_safe_json_loads(row["topic_ids_json"], [], "event.topic_ids"),
            metrics=_safe_json_loads(row["metrics_json"], {}, "event.metrics"),
            caused_by_event_ids=_safe_json_loads(row["caused_by_json"], [], "event.caused_by"),
            supersedes_event_ids=_safe_json_loads(row["supersedes_json"], [], "event.supersedes"),
            duplicates_event_ids=_safe_json_loads(row["duplicates_json"], [], "event.duplicates"),
            provenance=Provenance.model_validate(
                _safe_json_loads(row["provenance_json"], {}, "event.provenance")
            ),
        )

    # -------------------------------------------------------------------
    # Pack queries (events)
    # -------------------------------------------------------------------

    def get_confirmed_events_for_pack(
        self, since: datetime | None = None, limit: int = 5000,
    ) -> list[KnowledgeEvent]:
        clauses = ["event_status = 'confirmed'"]
        params: list[Any] = []
        if since:
            clauses.append("happened_at >= ?")
            params.append(since.isoformat())
        where = " AND ".join(clauses)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM knowledge_events WHERE {where} ORDER BY happened_at DESC LIMIT ?",
                params + [limit],
            ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def get_open_loops(self, limit: int = 1000) -> list[KnowledgeEvent]:
        """Events of type question/task_created that have no corresponding task_completed."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM knowledge_events "
                "WHERE event_type IN ('question', 'task_created') "
                "AND event_status NOT IN ('retracted') "
                "ORDER BY happened_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_event(r) for r in rows]
