"""Adoption rule operations for the solver registry."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from multihead.registry._models import AdoptionRule

logger = logging.getLogger(__name__)


class AdoptionMixin:
    """Mixin providing adoption rule operations."""

    db_path: Path

    def add_adoption_rule(self, rule: AdoptionRule) -> None:
        """Add an adoption rule.

        Args:
            rule: Adoption rule to add
        """
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO adoption_rules (
                rule_id, name, solver_type, min_aggregate_score,
                required_benchmarks, min_benchmark_scores,
                max_cost_per_call, max_latency_ms, required_license,
                excluded_sources, auto_register, notify_user,
                created_at, enabled
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rule.rule_id,
            rule.name,
            rule.solver_type,
            rule.min_aggregate_score,
            json.dumps(rule.required_benchmarks),
            json.dumps(rule.min_benchmark_scores),
            rule.max_cost_per_call,
            rule.max_latency_ms,
            json.dumps(rule.required_license) if rule.required_license else None,
            json.dumps(rule.excluded_sources),
            1 if rule.auto_register else 0,
            1 if rule.notify_user else 0,
            rule.created_at.isoformat(),
            1 if rule.enabled else 0,
        ))

        conn.commit()
        conn.close()
        logger.info("Added adoption rule: %s", rule.rule_id)

    def check_adoption_rules(self, solver_id: str) -> list[str]:
        """Check if a solver meets any adoption rules.

        Args:
            solver_id: Solver to check

        Returns:
            List of matching rule IDs
        """
        solver = self.get_solver(solver_id)
        if not solver:
            return []

        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM adoption_rules
            WHERE enabled = 1
            AND solver_type = ?
        """, (solver["solver_type"],))

        rows = cursor.fetchall()
        conn.close()

        matching_rules = []

        for row in rows:
            if self._meets_adoption_criteria(solver, row):
                matching_rules.append(row["rule_id"])

        return matching_rules

    def _meets_adoption_criteria(self, solver: dict[str, Any], rule_row: sqlite3.Row) -> bool:
        """Check if solver meets adoption rule criteria.

        Args:
            solver: Solver dict
            rule_row: Rule row from database

        Returns:
            True if solver meets all criteria
        """
        # Check aggregate score
        aggregate = self._get_aggregate_score(solver["solver_id"])
        if aggregate < rule_row["min_aggregate_score"]:
            return False

        # Check required benchmarks
        required_benchmarks = json.loads(rule_row["required_benchmarks"]) if rule_row["required_benchmarks"] else []
        if required_benchmarks:
            results = self.get_benchmark_results(solver["solver_id"])
            completed_benchmarks = {r["benchmark_name"] for r in results}
            if not set(required_benchmarks).issubset(completed_benchmarks):
                return False

        # Check min benchmark scores
        min_scores = json.loads(rule_row["min_benchmark_scores"]) if rule_row["min_benchmark_scores"] else {}
        if min_scores:
            results = self.get_benchmark_results(solver["solver_id"])
            latest_scores = {}
            for r in results:
                bench = r["benchmark_name"]
                if bench not in latest_scores:
                    latest_scores[bench] = r["score"]

            for bench, min_score in min_scores.items():
                if latest_scores.get(bench, 0.0) < min_score:
                    return False

        # Check cost constraint
        if rule_row["max_cost_per_call"] is not None:
            if solver.get("estimated_cost") is None:
                return False
            if solver["estimated_cost"] > rule_row["max_cost_per_call"]:
                return False

        # Check latency constraint
        if rule_row["max_latency_ms"] is not None:
            if solver.get("estimated_latency_ms") is None:
                return False
            if solver["estimated_latency_ms"] > rule_row["max_latency_ms"]:
                return False

        # Check license requirement
        if rule_row["required_license"]:
            required_licenses = json.loads(rule_row["required_license"])
            if solver.get("license") not in required_licenses:
                return False

        # Check excluded sources
        excluded_sources = json.loads(rule_row["excluded_sources"]) if rule_row["excluded_sources"] else []
        if solver["source"] in excluded_sources:
            return False

        return True
