"""Base classes for benchmarking."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from running a benchmark on a solver.

    This stores the performance metrics for a specific solver on a
    specific benchmark.
    """

    # Identity
    benchmark_name: str
    solver_id: str
    solver_type: str

    # Performance metrics
    score: float  # Primary metric (0.0-1.0, higher is better)
    metrics: dict[str, Any] = field(default_factory=dict)  # Additional metrics

    # Execution metadata
    runtime_seconds: float = 0.0
    sample_count: int = 0
    error_count: int = 0
    error_message: str | None = None

    # Timestamps
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    def __post_init__(self):
        """Validate fields."""
        if not 0.0 <= self.score <= 1.0:
            logger.warning("Score %.2f outside [0, 1] range for %s", self.score, self.benchmark_name)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "benchmark_name": self.benchmark_name,
            "solver_id": self.solver_id,
            "solver_type": self.solver_type,
            "score": self.score,
            "metrics": self.metrics,
            "runtime_seconds": self.runtime_seconds,
            "sample_count": self.sample_count,
            "error_count": self.error_count,
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class Benchmark(ABC):
    """Base class for benchmarks.

    Each benchmark knows how to evaluate a specific capability
    (e.g., reasoning, vision, math) and returns a standardized score.
    """

    def __init__(self, name: str, solver_types: list[str]):
        """Initialize benchmark.

        Args:
            name: Benchmark name (e.g., "mmlu", "gsm8k")
            solver_types: Solver types this benchmark applies to
        """
        self.name = name
        self.solver_types = solver_types

    @abstractmethod
    async def run(
        self,
        solver_id: str,
        generate_func: Any,  # Callable that takes prompt and returns text
        *,
        sample_limit: int | None = None,
        timeout_seconds: float = 300.0,
    ) -> BenchmarkResult:
        """Run benchmark on a solver.

        Args:
            solver_id: Solver identifier
            generate_func: Function to generate text from prompts
            sample_limit: Maximum samples to test (for faster runs)
            timeout_seconds: Maximum time for benchmark

        Returns:
            BenchmarkResult with scores and metrics
        """
        pass

    def applies_to(self, solver_type: str) -> bool:
        """Check if benchmark applies to solver type.

        Args:
            solver_type: Solver type to check

        Returns:
            True if benchmark is applicable
        """
        return solver_type in self.solver_types


class BenchmarkRunner:
    """Runs multiple benchmarks on solvers and aggregates results."""

    def __init__(self):
        """Initialize benchmark runner."""
        self.benchmarks: dict[str, Benchmark] = {}

    def register_benchmark(self, benchmark: Benchmark) -> None:
        """Register a benchmark.

        Args:
            benchmark: Benchmark to register
        """
        self.benchmarks[benchmark.name] = benchmark
        logger.info("Registered benchmark: %s (applies to %s)", benchmark.name, benchmark.solver_types)

    async def run_all_benchmarks(
        self,
        solver_id: str,
        solver_type: str,
        generate_func: Any,
        *,
        sample_limit: int | None = None,
        timeout_per_benchmark: float = 300.0,
        parallel: bool = False,
    ) -> list[BenchmarkResult]:
        """Run all applicable benchmarks on a solver.

        Args:
            solver_id: Solver identifier
            solver_type: Solver type
            generate_func: Function to generate text from prompts
            sample_limit: Maximum samples per benchmark
            timeout_per_benchmark: Timeout for each benchmark
            parallel: Run benchmarks in parallel (faster but higher resource usage)

        Returns:
            List of benchmark results
        """
        # Filter applicable benchmarks
        applicable = [
            bench for bench in self.benchmarks.values()
            if bench.applies_to(solver_type)
        ]

        if not applicable:
            logger.warning("No benchmarks applicable for solver type %s", solver_type)
            return []

        logger.info(
            "Running %d benchmarks on %s (type=%s)",
            len(applicable), solver_id, solver_type
        )

        results = []

        if parallel:
            # Run benchmarks in parallel
            tasks = [
                bench.run(
                    solver_id,
                    generate_func,
                    sample_limit=sample_limit,
                    timeout_seconds=timeout_per_benchmark,
                )
                for bench in applicable
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions
            results = [
                r for r in results
                if isinstance(r, BenchmarkResult)
            ]
        else:
            # Run benchmarks sequentially
            for bench in applicable:
                try:
                    result = await bench.run(
                        solver_id,
                        generate_func,
                        sample_limit=sample_limit,
                        timeout_seconds=timeout_per_benchmark,
                    )
                    results.append(result)
                except Exception as e:
                    logger.error("Benchmark %s failed for %s: %s", bench.name, solver_id, e)
                    # Create error result
                    results.append(BenchmarkResult(
                        benchmark_name=bench.name,
                        solver_id=solver_id,
                        solver_type=solver_type,
                        score=0.0,
                        error_count=1,
                        error_message=str(e),
                    ))

        logger.info("Completed %d benchmarks for %s", len(results), solver_id)
        return results

    def get_aggregate_score(self, results: list[BenchmarkResult]) -> float:
        """Calculate aggregate score from benchmark results.

        Args:
            results: List of benchmark results

        Returns:
            Weighted average score (0.0-1.0)
        """
        if not results:
            return 0.0

        # Simple average (can be made weighted in the future)
        total_score = sum(r.score for r in results)
        return total_score / len(results)

    def compare_solvers(
        self,
        results_a: list[BenchmarkResult],
        results_b: list[BenchmarkResult],
    ) -> dict[str, Any]:
        """Compare two solvers based on benchmark results.

        Args:
            results_a: Results for solver A
            results_b: Results for solver B

        Returns:
            Comparison dict with winner and score differences
        """
        score_a = self.get_aggregate_score(results_a)
        score_b = self.get_aggregate_score(results_b)

        # Per-benchmark comparison
        benchmarks_a = {r.benchmark_name: r for r in results_a}
        benchmarks_b = {r.benchmark_name: r for r in results_b}

        common_benchmarks = set(benchmarks_a.keys()) & set(benchmarks_b.keys())
        per_benchmark = {}

        for bench_name in common_benchmarks:
            diff = benchmarks_a[bench_name].score - benchmarks_b[bench_name].score
            per_benchmark[bench_name] = {
                "score_a": benchmarks_a[bench_name].score,
                "score_b": benchmarks_b[bench_name].score,
                "difference": diff,
            }

        return {
            "aggregate_score_a": score_a,
            "aggregate_score_b": score_b,
            "difference": score_a - score_b,
            "winner": "a" if score_a > score_b else "b" if score_b > score_a else "tie",
            "per_benchmark": per_benchmark,
        }
