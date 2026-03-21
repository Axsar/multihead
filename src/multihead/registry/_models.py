"""Data models for the solver registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class AdoptionRule:
    """Rule for automatically adopting discovered solvers.

    When a solver meets these criteria, it can be auto-registered
    as a head in the system.
    """

    rule_id: str
    name: str
    solver_type: str  # "llm", "vlm", etc.

    # Requirements
    min_aggregate_score: float = 0.0  # Minimum benchmark aggregate (0.0-1.0)
    required_benchmarks: list[str] = field(default_factory=list)  # Must pass these
    min_benchmark_scores: dict[str, float] = field(default_factory=dict)  # Per-benchmark mins

    # Constraints
    max_cost_per_call: float | None = None  # USD
    max_latency_ms: int | None = None
    required_license: list[str] | None = None  # e.g., ["apache-2.0", "mit"]
    excluded_sources: list[str] = field(default_factory=list)  # e.g., ["botvibes"]

    # Actions
    auto_register: bool = True  # Automatically add to heads.yaml
    notify_user: bool = True

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    enabled: bool = True
