"""Cross-topic link CRUD operations."""

from __future__ import annotations

import json

from multihead.knowledge_models import EntityRef, Link, Provenance

from ._helpers import _now_iso, _safe_json_loads
from ._retry import _sqlite_retry


class LinksMixin:
    """Mixin providing cross-topic link operations."""

    # These methods expect self._connect() to be available from the main class.

    # -------------------------------------------------------------------
    # Links
    # -------------------------------------------------------------------

    @_sqlite_retry
    def insert_link(self, link: Link) -> Link:
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO cross_links "
                "(link_id, from_type, from_id, from_label, to_type, to_id, to_label, "
                "reason_type, reason, score, link_status, provenance_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (link.link_id,
                 link.from_entity.entity_type, link.from_entity.entity_id,
                 link.from_entity.label or None,
                 link.to_entity.entity_type, link.to_entity.entity_id,
                 link.to_entity.label or None,
                 link.reason_type, link.reason or None,
                 link.score, link.link_status,
                 json.dumps(link.provenance.model_dump(mode="json")),
                 now),
            )
        return link

    def list_links(self, entity_id: str | None = None, limit: int = 100) -> list[Link]:
        with self._connect() as conn:
            if entity_id:
                rows = conn.execute(
                    "SELECT * FROM cross_links WHERE from_id = ? OR to_id = ? ORDER BY score DESC LIMIT ?",
                    (entity_id, entity_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM cross_links ORDER BY score DESC LIMIT ?", (limit,)
                ).fetchall()
        return [self._row_to_link(r) for r in rows]

    def _row_to_link(self, row) -> Link:
        return Link(
            link_id=row["link_id"],
            from_entity=EntityRef(entity_type=row["from_type"], entity_id=row["from_id"],
                                  label=row["from_label"] or ""),
            to_entity=EntityRef(entity_type=row["to_type"], entity_id=row["to_id"],
                                label=row["to_label"] or ""),
            reason_type=row["reason_type"],
            reason=row["reason"] or "",
            score=row["score"],
            link_status=row["link_status"],
            provenance=Provenance.model_validate(
                _safe_json_loads(row["provenance_json"], {}, "link.provenance")
            ),
        )
