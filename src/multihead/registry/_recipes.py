"""Recipe version tracking operations for the solver registry."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RecipesMixin:
    """Mixin providing recipe version tracking (Phase 6: Recipe Learning)."""

    db_path: Path

    def add_recipe_version(
        self,
        recipe_id: str,
        task_type: str,
        goal: str,
        source: str,
        recipe_yaml: str,
        *,
        performance_score: float | None = None,
        success_rate: float | None = None,
        avg_latency_ms: float | None = None,
        avg_cost_usd: float | None = None,
        test_cases_count: int = 0,
        adoption_status: str = "candidate",
        notes: str | None = None,
    ) -> int:
        """Add a new recipe version.

        Auto-increments version number for the given recipe_id.

        Returns:
            The new version number.
        """
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        cursor = conn.cursor()

        # Get next version
        cursor.execute(
            "SELECT MAX(version) FROM recipe_versions WHERE recipe_id = ?",
            (recipe_id,),
        )
        row = cursor.fetchone()
        version = (row[0] or 0) + 1

        now = datetime.now(timezone.utc).isoformat()
        cursor.execute("""
            INSERT INTO recipe_versions (
                recipe_id, version, task_type, goal, source, recipe_yaml,
                performance_score, success_rate, avg_latency_ms, avg_cost_usd,
                test_cases_count, adoption_status, created_at, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            recipe_id, version, task_type, goal, source, recipe_yaml,
            performance_score, success_rate, avg_latency_ms, avg_cost_usd,
            test_cases_count, adoption_status, now, notes,
        ))

        conn.commit()
        conn.close()
        return version

    def adopt_recipe_version(self, recipe_id: str, version: int) -> None:
        """Mark a recipe version as adopted."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        cursor = conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        cursor.execute("""
            UPDATE recipe_versions
            SET adoption_status = 'adopted', adopted_at = ?
            WHERE recipe_id = ? AND version = ?
        """, (now, recipe_id, version))
        conn.commit()
        conn.close()

    def get_best_recipe(self, task_type: str) -> dict[str, Any] | None:
        """Get the best adopted recipe for a task type.

        Returns the most recently adopted recipe, or the highest-scoring candidate.
        """
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Prefer adopted recipes
        cursor.execute("""
            SELECT * FROM recipe_versions
            WHERE task_type = ? AND adoption_status = 'adopted'
            ORDER BY adopted_at DESC
            LIMIT 1
        """, (task_type,))
        row = cursor.fetchone()

        if not row:
            # Fall back to best candidate by performance
            cursor.execute("""
                SELECT * FROM recipe_versions
                WHERE task_type = ? AND adoption_status = 'candidate'
                ORDER BY performance_score DESC NULLS LAST
                LIMIT 1
            """, (task_type,))
            row = cursor.fetchone()

        conn.close()
        return dict(row) if row else None

    def list_recipe_versions(
        self, recipe_id: str | None = None, task_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """List recipe versions with optional filters."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        conditions = []
        params: list[Any] = []
        if recipe_id:
            conditions.append("recipe_id = ?")
            params.append(recipe_id)
        if task_type:
            conditions.append("task_type = ?")
            params.append(task_type)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        cursor.execute(f"""
            SELECT * FROM recipe_versions {where}
            ORDER BY recipe_id, version DESC
        """, params)

        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def add_recipe_evaluation(
        self,
        recipe_id: str,
        version: int,
        head_id: str,
        vote: str,
        confidence: float,
        rationale: str,
    ) -> None:
        """Record a head's evaluation vote on a recipe version."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        cursor = conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        cursor.execute("""
            INSERT INTO recipe_evaluations (
                recipe_id, version, head_id, vote, confidence, rationale, evaluated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (recipe_id, version, head_id, vote, confidence, rationale, now))
        conn.commit()
        conn.close()

    def get_recipe_evaluations(
        self, recipe_id: str, version: int,
    ) -> list[dict[str, Any]]:
        """Get all evaluation votes for a recipe version."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM recipe_evaluations
            WHERE recipe_id = ? AND version = ?
            ORDER BY evaluated_at
        """, (recipe_id, version))
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]
