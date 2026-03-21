"""Tests for benchmarking engine."""

import pytest

from multihead.benchmarking.base import BenchmarkResult, BenchmarkRunner
from multihead.benchmarking.llm_benchmarks import (
    SimpleReasoningBenchmark,
    MMLUBenchmark,
    GSM8KBenchmark,
)
from multihead.benchmarking.vision_benchmarks import LatencyBenchmark, ImageClassificationBenchmark


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""

    def test_create_result(self):
        """Test creating a benchmark result."""
        result = BenchmarkResult(
            benchmark_name="test",
            solver_id="test-solver",
            solver_type="llm",
            score=0.85,
            metrics={"accuracy": 0.85},
        )

        assert result.score == 0.85
        assert result.benchmark_name == "test"

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = BenchmarkResult(
            benchmark_name="test",
            solver_id="test-solver",
            solver_type="llm",
            score=0.85,
            sample_count=10,
        )

        data = result.to_dict()
        assert data["score"] == 0.85
        assert data["sample_count"] == 10
        assert "started_at" in data


class TestSimpleReasoningBenchmark:
    """Test simple reasoning benchmark."""

    async def test_reasoning_benchmark_perfect_score(self):
        """Test benchmark with perfect answers."""
        # Mock generate function that always returns correct answer
        async def mock_generate(prompt: str) -> str:
            if "cats" in prompt.lower():
                return "Yes, all cats have hearts."
            elif "15 greater than 20" in prompt:
                return "No, 15 is not greater than 20."
            elif "raining" in prompt.lower():
                return "Maybe it is raining."
            elif "2 + 2" in prompt:
                return "4"
            elif "capital of France" in prompt:
                return "Paris"
            elif "square a rectangle" in prompt:
                return "Yes"
            elif "A > B" in prompt:
                return "Yes"
            elif "water H2O" in prompt:
                return "Yes"
            elif "10 / 2" in prompt:
                return "5"
            elif "sky green" in prompt:
                return "No"
            else:
                return "maybe"

        benchmark = SimpleReasoningBenchmark()
        result = await benchmark.run(
            solver_id="test-solver",
            generate_func=mock_generate,
            sample_limit=None,  # Use all questions
        )

        assert result.score == 1.0  # Perfect score
        assert result.sample_count == len(benchmark.QUESTIONS)
        assert result.error_count == 0

    async def test_reasoning_benchmark_partial_score(self):
        """Test benchmark with some incorrect answers."""
        # Mock that only answers "yes" to everything
        async def mock_generate(prompt: str) -> str:
            return "Yes"

        benchmark = SimpleReasoningBenchmark()
        result = await benchmark.run(
            solver_id="test-solver",
            generate_func=mock_generate,
            sample_limit=None,
        )

        # Should get some correct (questions with "yes" answer)
        assert 0.0 < result.score < 1.0
        assert result.sample_count == len(benchmark.QUESTIONS)

    async def test_reasoning_benchmark_with_sample_limit(self):
        """Test benchmark with sample limit."""
        async def mock_generate(prompt: str) -> str:
            return "Yes"

        benchmark = SimpleReasoningBenchmark()
        result = await benchmark.run(
            solver_id="test-solver",
            generate_func=mock_generate,
            sample_limit=3,  # Only test 3 questions
        )

        assert result.sample_count == 3


class TestMMLUBenchmark:
    """Test MMLU benchmark."""

    async def test_mmlu_benchmark(self):
        """Test MMLU with correct answers."""
        # Mock that returns correct answer letter
        async def mock_generate(prompt: str) -> str:
            if "powerhouse" in prompt.lower():
                return "B. Mitochondria"
            elif "closest to the Sun" in prompt:
                return "C. Mercury"
            elif "7 * 8" in prompt:
                return "B. 56"
            return "A"

        benchmark = MMLUBenchmark()
        result = await benchmark.run(
            solver_id="test-solver",
            generate_func=mock_generate,
        )

        assert result.score == 1.0
        assert result.metrics["correct"] == len(benchmark.SAMPLE_QUESTIONS)


class TestGSM8KBenchmark:
    """Test GSM8K benchmark."""

    async def test_gsm8k_benchmark(self):
        """Test GSM8K with correct numeric answers."""
        # Mock that returns correct numbers
        async def mock_generate(prompt: str) -> str:
            if "5 apples" in prompt:
                return "John has 8 apples."
            elif "rectangle" in prompt:
                return "The area is 24."
            elif "train" in prompt:
                return "The speed is 30 miles per hour."
            return "0"

        benchmark = GSM8KBenchmark()
        result = await benchmark.run(
            solver_id="test-solver",
            generate_func=mock_generate,
        )

        assert result.score == 1.0
        assert result.metrics["accuracy"] == 1.0


class TestLatencyBenchmark:
    """Test latency benchmark."""

    async def test_latency_benchmark_fast(self):
        """Test latency benchmark with fast solver."""
        # Mock that returns instantly
        call_count = 0

        async def mock_generate(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return "Hello"

        benchmark = LatencyBenchmark()
        result = await benchmark.run(
            solver_id="test-solver",
            generate_func=mock_generate,
            sample_limit=5,
        )

        assert call_count == 5
        assert result.sample_count == 5
        assert "p50_ms" in result.metrics
        assert "p95_ms" in result.metrics
        assert result.metrics["p50_ms"] < 1000  # Should be very fast

    async def test_latency_benchmark_slow(self):
        """Test latency benchmark with slow solver."""
        import asyncio

        async def mock_generate_slow(prompt: str) -> str:
            await asyncio.sleep(0.1)  # 100ms delay
            return "Hello"

        benchmark = LatencyBenchmark()
        result = await benchmark.run(
            solver_id="test-solver",
            generate_func=mock_generate_slow,
            sample_limit=3,
        )

        assert result.metrics["p50_ms"] > 90  # Should be ~100ms


class TestImageClassificationBenchmark:
    """Test image classification benchmark."""

    async def test_image_classification(self):
        """Test image classification with correct answers."""
        async def mock_generate(prompt: str) -> str:
            if "cat" in prompt.lower():
                return "The animal in the image is a cat."
            elif "apple" in prompt.lower():
                return "The apple is red."
            elif "bicycle" in prompt.lower():
                return "The person is riding a bicycle."
            return "Unknown"

        benchmark = ImageClassificationBenchmark()
        result = await benchmark.run(
            solver_id="test-solver",
            generate_func=mock_generate,
        )

        assert result.score == 1.0
        assert result.solver_type == "vlm"


class TestBenchmarkRunner:
    """Test benchmark runner."""

    async def test_register_and_run_benchmarks(self):
        """Test registering and running multiple benchmarks."""
        runner = BenchmarkRunner()

        # Register benchmarks
        runner.register_benchmark(SimpleReasoningBenchmark())
        runner.register_benchmark(LatencyBenchmark())

        assert len(runner.benchmarks) == 2

        # Mock generate function
        async def mock_generate(prompt: str) -> str:
            return "Yes"

        # Run all applicable benchmarks for LLM
        results = await runner.run_all_benchmarks(
            solver_id="test-llm",
            solver_type="llm",
            generate_func=mock_generate,
            sample_limit=3,
        )

        # Should run both benchmarks (both apply to LLM)
        assert len(results) == 2
        assert any(r.benchmark_name == "simple_reasoning" for r in results)
        assert any(r.benchmark_name == "latency" for r in results)

    async def test_benchmark_runner_filters_by_type(self):
        """Test that runner only runs applicable benchmarks."""
        runner = BenchmarkRunner()

        # Register LLM-only benchmark
        runner.register_benchmark(MMLUBenchmark())

        async def mock_generate(prompt: str) -> str:
            return "A"

        # Try to run on VLM (should skip MMLU)
        results = await runner.run_all_benchmarks(
            solver_id="test-vlm",
            solver_type="vlm",
            generate_func=mock_generate,
        )

        # MMLU doesn't apply to VLM, but runs anyway since VLM can do text
        # Actually, MMLU is solver_types=["llm"], so it shouldn't run
        # Let me check - if no applicable benchmarks, returns empty list
        assert len(results) == 0  # MMLU only applies to LLM

    def test_get_aggregate_score(self):
        """Test aggregate score calculation."""
        runner = BenchmarkRunner()

        results = [
            BenchmarkResult("test1", "solver", "llm", score=0.8),
            BenchmarkResult("test2", "solver", "llm", score=0.9),
            BenchmarkResult("test3", "solver", "llm", score=0.7),
        ]

        aggregate = runner.get_aggregate_score(results)
        assert aggregate == pytest.approx(0.8, abs=0.01)

    def test_compare_solvers(self):
        """Test solver comparison."""
        runner = BenchmarkRunner()

        results_a = [
            BenchmarkResult("test1", "solver_a", "llm", score=0.9),
            BenchmarkResult("test2", "solver_a", "llm", score=0.8),
        ]

        results_b = [
            BenchmarkResult("test1", "solver_b", "llm", score=0.7),
            BenchmarkResult("test2", "solver_b", "llm", score=0.9),
        ]

        comparison = runner.compare_solvers(results_a, results_b)

        assert comparison["aggregate_score_a"] == pytest.approx(0.85, abs=0.01)
        assert comparison["aggregate_score_b"] == pytest.approx(0.8, abs=0.01)
        assert comparison["winner"] == "a"
        assert "test1" in comparison["per_benchmark"]
        assert "test2" in comparison["per_benchmark"]
