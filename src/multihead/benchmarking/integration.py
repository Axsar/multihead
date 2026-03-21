"""Benchmarking integration - wires HeadManager into discovery for auto-benchmarking.

This module enables continuous benchmarking of discovered solvers by:
1. Loading models via HeadManager
2. Creating generate functions from loaded models
3. Running benchmarks and storing results in SolverRegistry
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from multihead.benchmarking.base import BenchmarkRunner, BenchmarkResult
from multihead.head_manager import HeadManager
from multihead.models import AdapterKind, HeadManifest
from multihead.registry.solver_registry import SolverRegistry

logger = logging.getLogger(__name__)


class BenchmarkingIntegration:
    """Integrates HeadManager with benchmarking system.

    Enables automatic benchmarking of discovered solvers by:
    - Converting solver manifests to HeadManifest format
    - Loading models via HeadManager
    - Creating generate functions
    - Running benchmarks via BenchmarkRunner
    """

    def __init__(
        self,
        head_manager: HeadManager,
        benchmark_runner: BenchmarkRunner,
        registry: SolverRegistry,
    ):
        """Initialize benchmarking integration.

        Args:
            head_manager: HeadManager for loading models
            benchmark_runner: BenchmarkRunner with registered benchmarks
            registry: SolverRegistry for storing results
        """
        self.head_manager = head_manager
        self.benchmarks = benchmark_runner
        self.registry = registry

    def solver_to_head_manifest(self, solver: dict[str, Any]) -> HeadManifest | None:
        """Convert solver dict to HeadManifest for loading.

        Args:
            solver: Solver dict from registry

        Returns:
            HeadManifest if convertible, None if not loadable locally
        """
        # Check if solver is local (not BotVibes)
        source = solver.get("source", "")
        if source == "botvibes":
            logger.debug("Skipping BotVibes solver %s (external)", solver["solver_id"])
            return None

        # Extract adapter type
        model_id = solver.get("model_id", "")
        if not model_id:
            logger.warning("Solver %s has no model_id", solver["solver_id"])
            return None

        # Infer adapter from source
        adapter_map = {
            "huggingface": AdapterKind.TRANSFORMERS,
            "ollama": AdapterKind.OLLAMA,
        }
        adapter = adapter_map.get(source)
        if not adapter:
            logger.debug("Unknown source %s for solver %s", source, solver["solver_id"])
            return None

        # Build HeadManifest
        try:
            manifest = HeadManifest(
                head_id=solver["solver_id"],
                name=solver.get("name", solver["solver_id"]),
                adapter=adapter,
                model=model_id,
                kind=solver.get("solver_type", "llm"),
                gpu_required=True,  # Assume GPU for now
                vram_hint_mb=solver.get("vram_mb", 8000),
            )
            return manifest
        except Exception as e:
            logger.error("Failed to create HeadManifest for %s: %s", solver["solver_id"], e)
            return None

    async def create_generate_function(
        self, solver_id: str
    ) -> Callable[[str], Any] | None:
        """Create a generate function for a solver.

        Loads the model via HeadManager and wraps it in a callable.

        Args:
            solver_id: Solver identifier

        Returns:
            Async generate function, or None if loading fails
        """
        solver = self.registry.get_solver(solver_id)
        if not solver:
            logger.warning("Solver %s not found in registry", solver_id)
            return None

        # Convert to HeadManifest
        manifest = self.solver_to_head_manifest(solver)
        if not manifest:
            return None

        # Add to HeadManager temporarily
        original_manifests = dict(self.head_manager._manifests)
        self.head_manager._manifests[solver_id] = manifest

        try:
            # Load head
            logger.info("Loading solver %s for benchmarking", solver_id)
            await self.head_manager.ensure_active(solver_id)

            # Create generate function
            async def generate(prompt: str) -> Any:
                """Generate function for benchmarking."""
                result = await self.head_manager.generate(
                    head_id=solver_id,
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=512,
                )
                return result

            return generate

        except Exception as e:
            logger.error("Failed to load solver %s: %s", solver_id, e)
            return None
        finally:
            # Restore original manifests (don't permanently add test solvers)
            self.head_manager._manifests = original_manifests

    async def benchmark_solver(
        self,
        solver_id: str,
        *,
        sample_limit: int = 10,
        timeout_seconds: float = 600.0,
    ) -> list[BenchmarkResult]:
        """Benchmark a solver and store results.

        Args:
            solver_id: Solver to benchmark
            sample_limit: Max samples per benchmark
            timeout_seconds: Total timeout for all benchmarks

        Returns:
            List of benchmark results
        """
        solver = self.registry.get_solver(solver_id)
        if not solver:
            logger.warning("Solver %s not found", solver_id)
            return []

        solver_type = solver.get("solver_type", "llm")

        # Create generate function
        generate_func = await self.create_generate_function(solver_id)
        if not generate_func:
            logger.warning("Could not create generate function for %s", solver_id)
            return []

        # Run benchmarks
        logger.info("Running benchmarks for %s (type=%s)", solver_id, solver_type)
        try:
            results = await self.benchmarks.run_all_benchmarks(
                solver_id=solver_id,
                solver_type=solver_type,
                generate_func=generate_func,
                sample_limit=sample_limit,
                timeout_per_benchmark=timeout_seconds,
            )

            # Store results in registry
            for result in results:
                self.registry.add_benchmark_result(result)
                logger.info(
                    "Benchmark %s for %s: score=%.2f",
                    result.benchmark_name,
                    solver_id,
                    result.score,
                )

            return results

        except Exception as e:
            logger.error("Benchmarking failed for %s: %s", solver_id, e)
            return []

    async def update_solver_capabilities(self, solver_id: str) -> None:
        """Update solver capabilities with benchmark results.

        Populates quality metrics (accuracy_score, latency_p50_ms, etc.)
        based on benchmark results.

        Args:
            solver_id: Solver to update
        """
        # Get all benchmark results for this solver
        # Note: SolverRegistry stores benchmarks in database
        # Query directly from database
        import sqlite3
        import json

        benchmarks = []
        try:
            conn = sqlite3.connect(self.registry.db_path, timeout=10.0)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM benchmarks WHERE solver_id = ?",
                (solver_id,)
            )
            rows = cursor.fetchall()
            conn.close()

            if rows:
                for row in rows:
                    benchmarks.append({
                        "benchmark_name": row[1],
                        "solver_id": row[2],
                        "score": row[4],
                        "metrics": json.loads(row[5] or "{}"),
                    })
        except sqlite3.OperationalError as e:
            # Table might not exist yet
            logger.debug("Could not query benchmarks: %s", e)
            benchmarks = []
        if not benchmarks:
            logger.debug("No benchmarks for solver %s", solver_id)
            return

        # Calculate aggregate metrics
        accuracy_scores = []
        latencies = []

        for benchmark in benchmarks:
            if benchmark.get("score") is not None:
                accuracy_scores.append(benchmark["score"])

            metrics = benchmark.get("metrics", {})
            if "p50_ms" in metrics:
                latencies.append(metrics["p50_ms"])

        # Update solver with calculated metrics
        updates = {}

        if accuracy_scores:
            # Use average of all benchmark scores as overall accuracy
            updates["estimated_accuracy"] = sum(accuracy_scores) / len(accuracy_scores)

        if latencies:
            # Use median latency
            sorted_latencies = sorted(latencies)
            median_idx = len(sorted_latencies) // 2
            updates["estimated_latency_ms"] = int(sorted_latencies[median_idx])

        if updates:
            # Update solver in registry
            solver = self.registry.get_solver(solver_id)
            if solver:
                for key, value in updates.items():
                    solver[key] = value
                # Note: SolverRegistry.update_solver() would be needed here
                # For now, metrics are stored in benchmark results
                logger.info(
                    "Updated capabilities for %s: %s",
                    solver_id,
                    updates,
                )


def create_benchmarking_integration(
    head_manager: HeadManager,
    registry_path: Path,
) -> BenchmarkingIntegration:
    """Factory function to create benchmarking integration.

    Args:
        head_manager: HeadManager instance
        registry_path: Path to solver registry database

    Returns:
        Configured BenchmarkingIntegration
    """
    from multihead.benchmarking import (
        BenchmarkRunner,
        GSM8KBenchmark,
        ImageClassificationBenchmark,
        LatencyBenchmark,
        MMLUBenchmark,
        SimpleReasoningBenchmark,
    )
    from multihead.registry.solver_registry import SolverRegistry

    # Create registry
    registry = SolverRegistry(registry_path)

    # Create benchmark runner
    benchmark_runner = BenchmarkRunner()
    benchmark_runner.register_benchmark(SimpleReasoningBenchmark())
    benchmark_runner.register_benchmark(MMLUBenchmark())
    benchmark_runner.register_benchmark(GSM8KBenchmark())
    benchmark_runner.register_benchmark(LatencyBenchmark())
    benchmark_runner.register_benchmark(ImageClassificationBenchmark())

    return BenchmarkingIntegration(
        head_manager=head_manager,
        benchmark_runner=benchmark_runner,
        registry=registry,
    )
