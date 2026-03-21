"""Solver CRUD operations for the solver registry."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multihead.discovery.base import SolverCandidate

logger = logging.getLogger(__name__)


class SolversMixin:
    """Mixin providing solver CRUD operations."""

    db_path: Path

    def add_solver(self, candidate: SolverCandidate, *, adoption_status: str = "candidate") -> None:
        """Add or update a solver in the registry.

        Args:
            candidate: Solver candidate to add
            adoption_status: Status (candidate, adopted, rejected)
        """
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO solvers (
                solver_id, name, source, solver_type, task_types, modalities,
                benchmark_scores, estimated_latency_ms, estimated_cost,
                model_id, version, license, description, url, tags,
                discovered_at, discovery_metadata, registered_at, adoption_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            candidate.solver_id,
            candidate.name,
            candidate.source,
            candidate.solver_type,
            json.dumps(candidate.task_types),
            json.dumps(candidate.modalities),
            json.dumps(candidate.benchmark_scores),
            candidate.estimated_latency_ms,
            candidate.estimated_cost,
            candidate.model_id,
            candidate.version,
            candidate.license,
            candidate.description,
            candidate.url,
            json.dumps(candidate.tags),
            candidate.discovered_at.isoformat(),
            json.dumps(candidate.discovery_metadata),
            datetime.now(timezone.utc).isoformat(),
            adoption_status,
        ))

        conn.commit()
        conn.close()
        logger.info("Added solver %s to registry (status=%s)", candidate.solver_id, adoption_status)

    def get_solver(self, solver_id: str) -> dict[str, Any] | None:
        """Get solver by ID.

        Args:
            solver_id: Solver identifier

        Returns:
            Solver dict or None if not found
        """
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM solvers WHERE solver_id = ?", (solver_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return self._row_to_solver(row)

    def list_solvers(
        self,
        *,
        solver_type: str | None = None,
        source: str | None = None,
        adoption_status: str | None = None,
        min_aggregate_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """List solvers with optional filtering.

        Args:
            solver_type: Filter by solver type
            source: Filter by source
            adoption_status: Filter by status
            min_aggregate_score: Minimum aggregate benchmark score

        Returns:
            List of solver dicts
        """
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM solvers WHERE 1=1"
        params = []

        if solver_type:
            query += " AND solver_type = ?"
            params.append(solver_type)

        if source:
            query += " AND source = ?"
            params.append(source)

        if adoption_status:
            query += " AND adoption_status = ?"
            params.append(adoption_status)

        query += " ORDER BY registered_at DESC"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        solvers = [self._row_to_solver(row) for row in rows]

        # Post-filter by aggregate score if needed
        if min_aggregate_score is not None:
            solvers = [
                s for s in solvers
                if self._get_aggregate_score(s["solver_id"]) >= min_aggregate_score
            ]

        return solvers

    def update_adoption_status(
        self,
        solver_id: str,
        status: str,
        *,
        rule_id: str | None = None,
        notes: str | None = None,
    ) -> None:
        """Update solver adoption status.

        Args:
            solver_id: Solver identifier
            status: New status (candidate, adopted, rejected)
            rule_id: Rule that triggered adoption
            notes: Optional notes
        """
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE solvers
            SET adoption_status = ?,
                adoption_rule_id = ?,
                notes = ?
            WHERE solver_id = ?
        """, (status, rule_id, notes, solver_id))

        conn.commit()
        conn.close()
        logger.info("Updated %s adoption status to: %s", solver_id, status)

    def _row_to_solver(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert DB row to solver dict.

        Args:
            row: SQLite row

        Returns:
            Solver dict
        """
        return {
            "solver_id": row["solver_id"],
            "name": row["name"],
            "source": row["source"],
            "solver_type": row["solver_type"],
            "task_types": json.loads(row["task_types"]) if row["task_types"] else [],
            "modalities": json.loads(row["modalities"]) if row["modalities"] else [],
            "benchmark_scores": json.loads(row["benchmark_scores"]) if row["benchmark_scores"] else {},
            "estimated_latency_ms": row["estimated_latency_ms"],
            "estimated_cost": row["estimated_cost"],
            "model_id": row["model_id"],
            "version": row["version"],
            "license": row["license"],
            "description": row["description"],
            "url": row["url"],
            "tags": json.loads(row["tags"]) if row["tags"] else [],
            "discovered_at": row["discovered_at"],
            "discovery_metadata": json.loads(row["discovery_metadata"]) if row["discovery_metadata"] else {},
            "registered_at": row["registered_at"],
            "last_benchmarked_at": row["last_benchmarked_at"],
            "adoption_status": row["adoption_status"],
            "adoption_rule_id": row["adoption_rule_id"],
            "notes": row["notes"],
        }
