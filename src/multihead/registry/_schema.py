"""Database schema initialization for the solver registry."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class SchemaMixin:
    """Mixin providing database schema initialization."""

    db_path: Path

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        cursor = conn.cursor()

        # Solvers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS solvers (
                solver_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                source TEXT NOT NULL,
                solver_type TEXT NOT NULL,
                task_types TEXT,  -- JSON list
                modalities TEXT,  -- JSON list
                benchmark_scores TEXT,  -- JSON dict
                estimated_latency_ms INTEGER,
                estimated_cost REAL,
                model_id TEXT,
                version TEXT,
                license TEXT,
                description TEXT,
                url TEXT,
                tags TEXT,  -- JSON list
                discovered_at TEXT NOT NULL,
                discovery_metadata TEXT,  -- JSON dict
                registered_at TEXT,  -- When added to registry
                last_benchmarked_at TEXT,
                adoption_status TEXT DEFAULT 'candidate',  -- candidate, adopted, rejected
                adoption_rule_id TEXT,
                notes TEXT
            )
        """)

        # Benchmark results table (time series)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                solver_id TEXT NOT NULL,
                benchmark_name TEXT NOT NULL,
                solver_type TEXT NOT NULL,
                score REAL NOT NULL,
                metrics TEXT,  -- JSON dict
                runtime_seconds REAL,
                sample_count INTEGER,
                error_count INTEGER,
                error_message TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                FOREIGN KEY (solver_id) REFERENCES solvers(solver_id)
            )
        """)

        # Adoption rules table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS adoption_rules (
                rule_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                solver_type TEXT NOT NULL,
                min_aggregate_score REAL DEFAULT 0.0,
                required_benchmarks TEXT,  -- JSON list
                min_benchmark_scores TEXT,  -- JSON dict
                max_cost_per_call REAL,
                max_latency_ms INTEGER,
                required_license TEXT,  -- JSON list
                excluded_sources TEXT,  -- JSON list
                auto_register INTEGER DEFAULT 1,
                notify_user INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                enabled INTEGER DEFAULT 1
            )
        """)

        # Solver preferences table (Phase 5: Meta-Reasoning)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS solver_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT NOT NULL,
                preferred_solver_id TEXT NOT NULL,
                reasoning TEXT,  -- Why this solver was selected
                confidence_score REAL,  -- 0.0-1.0
                consensus_votes TEXT,  -- JSON: {head_id: vote}
                benchmark_results TEXT,  -- JSON: {benchmark: score}
                selected_at TEXT NOT NULL,
                FOREIGN KEY (preferred_solver_id) REFERENCES solvers(solver_id)
            )
        """)

        # Recipe versions table (Phase 6: Recipe Learning)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recipe_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recipe_id TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                task_type TEXT NOT NULL,
                goal TEXT NOT NULL,
                source TEXT NOT NULL,
                recipe_yaml TEXT NOT NULL,
                performance_score REAL,
                success_rate REAL,
                avg_latency_ms REAL,
                avg_cost_usd REAL,
                test_cases_count INTEGER DEFAULT 0,
                adoption_status TEXT DEFAULT 'candidate',
                created_at TEXT NOT NULL,
                adopted_at TEXT,
                notes TEXT,
                UNIQUE(recipe_id, version)
            )
        """)

        # Recipe evaluations table (Phase 6: consensus votes on recipes)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recipe_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recipe_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                head_id TEXT NOT NULL,
                vote TEXT NOT NULL,
                confidence REAL,
                rationale TEXT,
                evaluated_at TEXT NOT NULL
            )
        """)

        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_solvers_type ON solvers(solver_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_solvers_source ON solvers(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_solvers_status ON solvers(adoption_status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_benchmark_solver ON benchmark_results(solver_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_benchmark_name ON benchmark_results(benchmark_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_preferences_task ON solver_preferences(task_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_preferences_solver ON solver_preferences(preferred_solver_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_recipe_versions_id ON recipe_versions(recipe_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_recipe_versions_task ON recipe_versions(task_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_recipe_evals_id ON recipe_evaluations(recipe_id)")

        conn.commit()
        conn.close()
        logger.info("Initialized solver registry at %s", self.db_path)
