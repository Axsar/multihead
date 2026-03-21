"""Solver preference / meta-reasoning operations for the solver registry."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PreferencesMixin:
    """Mixin providing solver preference operations (Phase 5: Meta-Reasoning)."""

    db_path: Path

    def record_selection(
        self,
        task_type: str,
        preferred_solver_id: str,
        reasoning: str,
        confidence_score: float,
        consensus_votes: dict[str, str] | None = None,
        benchmark_results: dict[str, float] | None = None,
    ) -> None:
        """Record a meta-reasoning solver selection (Phase 5).

        Args:
            task_type: Task type this selection applies to
            preferred_solver_id: Selected solver ID
            reasoning: Explanation of why this solver was chosen
            confidence_score: Confidence in selection (0.0-1.0)
            consensus_votes: {head_id: vote} mapping
            benchmark_results: {benchmark_name: score} mapping
        """
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO solver_preferences (
                task_type, preferred_solver_id, reasoning, confidence_score,
                consensus_votes, benchmark_results, selected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            task_type,
            preferred_solver_id,
            reasoning,
            confidence_score,
            json.dumps(consensus_votes or {}),
            json.dumps(benchmark_results or {}),
            datetime.now(timezone.utc).isoformat(),
        ))

        conn.commit()
        conn.close()
        logger.info(
            "Recorded preference: %s -> %s (confidence=%.2f)",
            task_type, preferred_solver_id, confidence_score
        )

    def get_preference(self, task_type: str) -> dict[str, Any] | None:
        """Get the most recent preference for a task type (Phase 5).

        Args:
            task_type: Task type to query

        Returns:
            Preference dict or None if no preference exists
        """
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM solver_preferences
            WHERE task_type = ?
            ORDER BY selected_at DESC
            LIMIT 1
        """, (task_type,))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return {
            "id": row["id"],
            "task_type": row["task_type"],
            "preferred_solver_id": row["preferred_solver_id"],
            "reasoning": row["reasoning"],
            "confidence_score": row["confidence_score"],
            "consensus_votes": json.loads(row["consensus_votes"]) if row["consensus_votes"] else {},
            "benchmark_results": json.loads(row["benchmark_results"]) if row["benchmark_results"] else {},
            "selected_at": row["selected_at"],
        }

    def list_preferences(self, task_type: str | None = None) -> list[dict[str, Any]]:
        """List all preferences, optionally filtered by task type (Phase 5).

        Args:
            task_type: Optional task type filter

        Returns:
            List of preference dicts
        """
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if task_type:
            cursor.execute("""
                SELECT * FROM solver_preferences
                WHERE task_type = ?
                ORDER BY selected_at DESC
            """, (task_type,))
        else:
            cursor.execute("""
                SELECT * FROM solver_preferences
                ORDER BY selected_at DESC
            """)

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "id": row["id"],
                "task_type": row["task_type"],
                "preferred_solver_id": row["preferred_solver_id"],
                "reasoning": row["reasoning"],
                "confidence_score": row["confidence_score"],
                "consensus_votes": json.loads(row["consensus_votes"]) if row["consensus_votes"] else {},
                "benchmark_results": json.loads(row["benchmark_results"]) if row["benchmark_results"] else {},
                "selected_at": row["selected_at"],
            }
            for row in rows
        ]
