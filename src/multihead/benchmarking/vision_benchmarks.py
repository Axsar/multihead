"""Vision benchmarks (latency, image classification)."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable

from multihead.benchmarking.base import Benchmark, BenchmarkResult

logger = logging.getLogger(__name__)


class LatencyBenchmark(Benchmark):
    """Measures inference latency for solvers.

    Runs multiple inference passes and measures p50, p95, p99 latency.
    Applies to all solver types.
    """

    def __init__(self):
        """Initialize latency benchmark."""
        super().__init__(
            name="latency",
            solver_types=["llm", "vlm", "object_detection", "segmentation", "embedding"],
        )

    async def run(
        self,
        solver_id: str,
        generate_func: Callable,
        *,
        sample_limit: int | None = None,
        timeout_seconds: float = 300.0,
    ) -> BenchmarkResult:
        """Run latency benchmark.

        Args:
            solver_id: Solver identifier
            generate_func: Function to generate responses
            sample_limit: Number of inference passes (default 10)
            timeout_seconds: Total timeout

        Returns:
            BenchmarkResult with latency metrics
        """
        started_at = datetime.now(timezone.utc)
        num_passes = sample_limit if sample_limit else 10

        # Test prompts (simple to minimize variation)
        test_prompt = "Hello, how are you?"

        latencies: list[float] = []
        errors = 0

        try:
            async with asyncio.timeout(timeout_seconds):
                for _ in range(num_passes):
                    try:
                        start = time.perf_counter()
                        await generate_func(test_prompt)
                        end = time.perf_counter()

                        latency_ms = (end - start) * 1000
                        latencies.append(latency_ms)

                    except Exception as e:
                        logger.warning("Latency test failed for %s: %s", solver_id, e)
                        errors += 1

        except asyncio.TimeoutError:
            logger.warning("Latency benchmark timed out for %s", solver_id)

        if not latencies:
            # All tests failed
            return BenchmarkResult(
                benchmark_name=self.name,
                solver_id=solver_id,
                solver_type="unknown",
                score=0.0,
                error_count=errors,
                error_message="All latency tests failed",
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )

        # Calculate percentiles
        latencies.sort()
        p50_idx = len(latencies) // 2
        p95_idx = int(len(latencies) * 0.95)
        p99_idx = int(len(latencies) * 0.99)

        p50 = latencies[p50_idx]
        p95 = latencies[p95_idx] if p95_idx < len(latencies) else latencies[-1]
        p99 = latencies[p99_idx] if p99_idx < len(latencies) else latencies[-1]
        avg = sum(latencies) / len(latencies)

        # Score: 1.0 for < 100ms, 0.5 for 1s, 0.0 for > 10s
        # Logarithmic scale to handle wide latency ranges
        if p50 < 100:
            score = 1.0
        elif p50 < 1000:
            score = 1.0 - (p50 - 100) / 900 * 0.5
        elif p50 < 10000:
            score = 0.5 - (p50 - 1000) / 9000 * 0.5
        else:
            score = 0.0

        completed_at = datetime.now(timezone.utc)
        runtime = (completed_at - started_at).total_seconds()

        return BenchmarkResult(
            benchmark_name=self.name,
            solver_id=solver_id,
            solver_type="unknown",  # Latency applies to all types
            score=score,
            metrics={
                "p50_ms": p50,
                "p95_ms": p95,
                "p99_ms": p99,
                "avg_ms": avg,
                "min_ms": latencies[0],
                "max_ms": latencies[-1],
            },
            runtime_seconds=runtime,
            sample_count=len(latencies),
            error_count=errors,
            started_at=started_at,
            completed_at=completed_at,
        )


class ImageClassificationBenchmark(Benchmark):
    """Image classification benchmark for VLMs.

    Tests ability to classify/describe images correctly.
    This is a simplified version - production would use COCO or ImageNet.
    """

    # Sample image classification tasks (text descriptions for now)
    SAMPLE_TASKS = [
        {
            "image_description": "A photo of a cat sitting on a mat",
            "question": "What animal is in the image?",
            "answer": "cat",
        },
        {
            "image_description": "A red apple on a white table",
            "question": "What color is the apple?",
            "answer": "red",
        },
        {
            "image_description": "A person riding a bicycle on a road",
            "question": "What is the person riding?",
            "answer": "bicycle",
        },
    ]

    def __init__(self):
        """Initialize image classification benchmark."""
        super().__init__(
            name="image_classification",
            solver_types=["vlm"],
        )

    async def run(
        self,
        solver_id: str,
        generate_func: Callable,
        *,
        sample_limit: int | None = None,
        timeout_seconds: float = 300.0,
    ) -> BenchmarkResult:
        """Run image classification benchmark.

        Args:
            solver_id: Solver identifier
            generate_func: Function to generate responses
            sample_limit: Maximum tasks to test
            timeout_seconds: Total timeout

        Returns:
            BenchmarkResult with classification accuracy
        """
        started_at = datetime.now(timezone.utc)
        tasks = self.SAMPLE_TASKS[:sample_limit] if sample_limit else self.SAMPLE_TASKS

        correct = 0
        total = 0
        errors = 0

        try:
            async with asyncio.timeout(timeout_seconds):
                for task in tasks:
                    total += 1
                    try:
                        # Format prompt (in real version, would include actual image)
                        prompt = f"Image: {task['image_description']}\n{task['question']}"

                        # Generate response
                        response = await generate_func(prompt)
                        if isinstance(response, dict):
                            response_text = response.get("text", "")
                        else:
                            response_text = str(response)

                        # Check if answer is in response (case-insensitive)
                        if task['answer'].lower() in response_text.lower():
                            correct += 1

                    except Exception as e:
                        logger.warning("Image classification failed for %s: %s", solver_id, e)
                        errors += 1

        except asyncio.TimeoutError:
            logger.warning("Image classification timed out for %s", solver_id)

        score = correct / total if total > 0 else 0.0
        completed_at = datetime.now(timezone.utc)
        runtime = (completed_at - started_at).total_seconds()

        return BenchmarkResult(
            benchmark_name=self.name,
            solver_id=solver_id,
            solver_type="vlm",
            score=score,
            metrics={
                "correct": correct,
                "total": total,
                "accuracy": score,
            },
            runtime_seconds=runtime,
            sample_count=total,
            error_count=errors,
            started_at=started_at,
            completed_at=completed_at,
        )


class COCOBenchmark(Benchmark):
    """COCO object detection/segmentation benchmark.

    This is a placeholder - production would use actual COCO dataset
    and compute mAP (mean Average Precision) metrics.
    """

    def __init__(self):
        """Initialize COCO benchmark."""
        super().__init__(
            name="coco",
            solver_types=["object_detection", "segmentation", "vlm"],
        )

    async def run(
        self,
        solver_id: str,
        generate_func: Callable,
        *,
        sample_limit: int | None = None,
        timeout_seconds: float = 300.0,
    ) -> BenchmarkResult:
        """Run COCO benchmark.

        Args:
            solver_id: Solver identifier
            generate_func: Function to generate responses
            sample_limit: Maximum images to test
            timeout_seconds: Total timeout

        Returns:
            BenchmarkResult with mAP score
        """
        started_at = datetime.now(timezone.utc)

        # Placeholder implementation
        # In production, would:
        # 1. Load COCO validation set
        # 2. Run detector on each image
        # 3. Compute mAP across all categories
        # 4. Return comprehensive metrics (mAP@0.5, mAP@0.75, etc.)

        logger.warning("COCO benchmark not fully implemented - returning placeholder")

        completed_at = datetime.now(timezone.utc)
        runtime = (completed_at - started_at).total_seconds()

        return BenchmarkResult(
            benchmark_name=self.name,
            solver_id=solver_id,
            solver_type="object_detection",
            score=0.0,
            metrics={
                "status": "not_implemented",
                "note": "Use actual COCO dataset for production benchmarking",
            },
            runtime_seconds=runtime,
            sample_count=0,
            error_count=0,
            error_message="COCO benchmark requires full dataset implementation",
            started_at=started_at,
            completed_at=completed_at,
        )
