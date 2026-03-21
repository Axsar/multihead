"""Solver registry for tracking discovered models and their performance.

This module composes the SolverRegistry class from domain-specific mixins.
All public API remains identical — external imports are unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path

from multihead.registry._adoption import AdoptionMixin
from multihead.registry._benchmarks import BenchmarksMixin
from multihead.registry._models import AdoptionRule
from multihead.registry._preferences import PreferencesMixin
from multihead.registry._recipes import RecipesMixin
from multihead.registry._schema import SchemaMixin
from multihead.registry._solvers import SolversMixin

logger = logging.getLogger(__name__)

# Re-export so `from multihead.registry.solver_registry import AdoptionRule` still works
__all__ = ["SolverRegistry", "AdoptionRule"]


class SolverRegistry(
    SchemaMixin,
    SolversMixin,
    BenchmarksMixin,
    AdoptionMixin,
    PreferencesMixin,
    RecipesMixin,
):
    """Registry for tracking discovered solvers and their performance.

    This is the persistent store for the discovery system. It tracks:
    - Discovered solver candidates (from HuggingFace, Ollama, BotVibes)
    - Benchmark results over time
    - Version history
    - Adoption rules for automatic head registration
    """

    def __init__(self, db_path: Path | str):
        """Initialize solver registry.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
