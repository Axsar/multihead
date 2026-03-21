"""Benchmark operations for the solver registry."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multihead.benchmarking.base import BenchmarkResult

logger = logging.getLogger(__name__)


class BenchmarksMixin:
    """Mixin providing benchmark-related operations."""

    db_path: Path

    def add_benchmark_result(self, result: BenchmarkResult) -> None:
        """Store a benchmark result.

        Args:
            result: Benchmark result to store
        """
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO benchmark_results (
                solver_id, benchmark_name, solver_type, score, metrics,
                runtime_seconds, sample_count, error_count, error_message,
                started_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.solver_id,
            result.benchmark_name,
            result.solver_type,
            result.score,
            json.dumps(result.metrics),
            result.runtime_seconds,
            result.sample_count,
            result.error_count,
            result.error_message,
            result.started_at.isoformat(),
            result.completed_at.isoformat() if result.completed_at else None,
        ))

        # Update solver's last_benchmarked_at
        cursor.execute("""
            UPDATE solvers
            SET last_benchmarked_at = ?
            WHERE solver_id = ?
        """, (datetime.now(timezone.utc).isoformat(), result.solver_id))

        conn.commit()
        conn.close()
        logger.debug("Stored benchmark result for %s: %s=%.2f", result.solver_id, result.benchmark_name, result.score)

    def get_benchmark_results(
        self,
        solver_id: str,
        *,
        benchmark_name: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get benchmark results for a solver.

        Args:
            solver_id: Solver identifier
            benchmark_name: Optional filter by benchmark name
            limit: Maximum results to return (most recent first)

        Returns:
            List of benchmark result dicts
        """
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM benchmark_results WHERE solver_id = ?"
        params = [solver_id]

        if benchmark_name:
            query += " AND benchmark_name = ?"
            params.append(benchmark_name)

        query += " ORDER BY started_at DESC"

        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_benchmark_result(row) for row in rows]

    def _get_aggregate_score(self, solver_id: str) -> float:
        """Calculate aggregate score from latest benchmark results.

        Args:
            solver_id: Solver identifier

        Returns:
            Aggregate score (0.0-1.0)
        """
        # Get latest result for each benchmark
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT benchmark_name, score
            FROM benchmark_results
            WHERE solver_id = ?
            AND (benchmark_name, started_at) IN (
                SELECT benchmark_name, MAX(started_at)
                FROM benchmark_results
                WHERE solver_id = ?
                GROUP BY benchmark_name
            )
        """, (solver_id, solver_id))

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return 0.0

        # Simple average
        return sum(score for _, score in rows) / len(rows)

    def compare_solvers(
        self,
        solver_id_a: str,
        solver_id_b: str,
    ) -> dict[str, Any]:
        """Compare two solvers based on their latest benchmarks.

        Args:
            solver_id_a: First solver ID
            solver_id_b: Second solver ID

        Returns:
            Comparison dict with scores and winner
        """
        score_a = self._get_aggregate_score(solver_id_a)
        score_b = self._get_aggregate_score(solver_id_b)

        # Get per-benchmark comparison
        results_a = self.get_benchmark_results(solver_id_a)
        results_b = self.get_benchmark_results(solver_id_b)

        # Build latest scores per benchmark
        latest_a = {}
        for r in results_a:
            bench = r["benchmark_name"]
            if bench not in latest_a:
                latest_a[bench] = r["score"]

        latest_b = {}
        for r in results_b:
            bench = r["benchmark_name"]
            if bench not in latest_b:
                latest_b[bench] = r["score"]

        common_benchmarks = set(latest_a.keys()) & set(latest_b.keys())
        per_benchmark = {}

        for bench in common_benchmarks:
            diff = latest_a[bench] - latest_b[bench]
            per_benchmark[bench] = {
                "score_a": latest_a[bench],
                "score_b": latest_b[bench],
                "difference": diff,
            }

        return {
            "solver_a": solver_id_a,
            "solver_b": solver_id_b,
            "aggregate_score_a": score_a,
            "aggregate_score_b": score_b,
            "difference": score_a - score_b,
            "winner": "a" if score_a > score_b else "b" if score_b > score_a else "tie",
            "per_benchmark": per_benchmark,
        }

    def _row_to_benchmark_result(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert DB row to benchmark result dict.

        Args:
            row: SQLite row

        Returns:
            Benchmark result dict
        """
        return {
            "id": row["id"],
            "solver_id": row["solver_id"],
            "benchmark_name": row["benchmark_name"],
            "solver_type": row["solver_type"],
            "score": row["score"],
            "metrics": json.loads(row["metrics"]) if row["metrics"] else {},
            "runtime_seconds": row["runtime_seconds"],
            "sample_count": row["sample_count"],
            "error_count": row["error_count"],
            "error_message": row["error_message"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
        }
