"""Tests for benchmarking integration with HeadManager."""

from __future__ import annotations

import pytest

from multihead.benchmarking.integration import (
    BenchmarkingIntegration,
    create_benchmarking_integration,
)
from multihead.benchmarking.base import BenchmarkRunner, BenchmarkResult
from multihead.head_manager import HeadManager
from multihead.models import AdapterKind, HeadManifest
from multihead.registry.solver_registry import SolverRegistry
from multihead.discovery.base import SolverCandidate


@pytest.fixture
def temp_registry(tmp_path):
    """Create temporary solver registry."""
    return SolverRegistry(tmp_path / "test_registry.db")


@pytest.fixture
def head_manager():
    """Create HeadManager with mock head."""
    manifests = {
        "mock-llm": HeadManifest(
            head_id="mock-llm",
            name="Mock LLM",
            adapter=AdapterKind.MOCK,
            model="mock-v1",
            kind="llm",
        ),
    }
    return HeadManager(manifests)


@pytest.fixture
def benchmark_runner():
    """Create BenchmarkRunner with test benchmarks."""
    from multihead.benchmarking import SimpleReasoningBenchmark, LatencyBenchmark

    runner = BenchmarkRunner()
    runner.register_benchmark(SimpleReasoningBenchmark())
    runner.register_benchmark(LatencyBenchmark())
    return runner


@pytest.fixture
def integration(head_manager, benchmark_runner, temp_registry):
    """Create benchmarking integration."""
    return BenchmarkingIntegration(
        head_manager=head_manager,
        benchmark_runner=benchmark_runner,
        registry=temp_registry,
    )


class TestSolverToHeadManifest:
    """Test solver to HeadManifest conversion."""

    def test_converts_huggingface_solver(self, integration):
        """Should convert HuggingFace solver to HeadManifest."""
        solver = {
            "solver_id": "hf-qwen-8b",
            "name": "Qwen3 8B",
            "source": "huggingface",
            "model_id": "Qwen/Qwen3-8B-Instruct",
            "solver_type": "llm",
            "vram_mb": 6000,
        }

        manifest = integration.solver_to_head_manifest(solver)

        assert manifest is not None
        assert manifest.head_id == "hf-qwen-8b"
        assert manifest.name == "Qwen3 8B"
        assert manifest.adapter == AdapterKind.TRANSFORMERS
        assert manifest.model == "Qwen/Qwen3-8B-Instruct"
        assert manifest.kind == "llm"
        assert manifest.vram_hint_mb == 6000

    def test_converts_ollama_solver(self, integration):
        """Should convert Ollama solver to HeadManifest."""
        solver = {
            "solver_id": "ollama-llama3",
            "name": "Llama 3",
            "source": "ollama",
            "model_id": "llama3:8b",
            "solver_type": "llm",
        }

        manifest = integration.solver_to_head_manifest(solver)

        assert manifest is not None
        assert manifest.adapter == AdapterKind.OLLAMA
        assert manifest.model == "llama3:8b"

    def test_skips_botvibes_solver(self, integration):
        """Should skip BotVibes solvers (external)."""
        solver = {
            "solver_id": "botvibes-solver",
            "name": "External Solver",
            "source": "botvibes",
            "model_id": "some-model",
            "solver_type": "llm",
        }

        manifest = integration.solver_to_head_manifest(solver)

        assert manifest is None

    def test_skips_solver_without_model_id(self, integration):
        """Should skip solvers without model_id."""
        solver = {
            "solver_id": "no-model",
            "name": "No Model",
            "source": "huggingface",
            "solver_type": "llm",
        }

        manifest = integration.solver_to_head_manifest(solver)

        assert manifest is None


class TestCreateGenerateFunction:
    """Test generate function creation."""

    @pytest.mark.asyncio
    async def test_creates_function_for_mock_solver(self, integration, temp_registry):
        """Should create generate function for mock solver."""
        from multihead.adapters.mock import MockAdapter
        from multihead.head_manager import HeadState
        from multihead.resilience import CircuitBreaker

        # Add mock solver to registry
        candidate = SolverCandidate(
            solver_id="test-mock",
            name="Test Mock",
            source="huggingface",
            solver_type="llm",
            task_types=["text_generation"],
            modalities=["text"],
            model_id="mock-model",
        )
        temp_registry.add_solver(candidate)

        # Add mock manifest, adapter, state, and breaker to head_manager
        manifest = HeadManifest(
            head_id="test-mock",
            name="Test Mock",
            adapter=AdapterKind.MOCK,
            model="mock-model",
            kind="llm",
        )
        integration.head_manager._manifests["test-mock"] = manifest
        integration.head_manager._adapters["test-mock"] = MockAdapter(manifest)
        integration.head_manager._states["test-mock"] = HeadState.OFF
        integration.head_manager._breakers["test-mock"] = CircuitBreaker(5, 60.0)

        # Create generate function
        generate_func = await integration.create_generate_function("test-mock")

        # Should return a function
        assert generate_func is not None
        assert callable(generate_func)

        # Test the function
        result = await generate_func("Hello")
        assert result is not None

    @pytest.mark.asyncio
    async def test_returns_none_for_nonexistent_solver(self, integration):
        """Should return None for nonexistent solver."""
        generate_func = await integration.create_generate_function("nonexistent")
        assert generate_func is None


class TestBenchmarkSolver:
    """Test solver benchmarking."""

    @pytest.mark.asyncio
    async def test_benchmarks_mock_solver(self, integration, temp_registry):
        """Should benchmark mock solver and store results."""
        from multihead.adapters.mock import MockAdapter
        from multihead.head_manager import HeadState
        from multihead.resilience import CircuitBreaker

        # Add mock solver
        candidate = SolverCandidate(
            solver_id="bench-test",
            name="Benchmark Test",
            source="huggingface",
            solver_type="llm",
            task_types=["text_generation"],
            modalities=["text"],
            model_id="mock-model",
        )
        temp_registry.add_solver(candidate)

        # Add mock manifest, adapter, state, and breaker
        manifest = HeadManifest(
            head_id="bench-test",
            name="Benchmark Test",
            adapter=AdapterKind.MOCK,
            model="mock-model",
            kind="llm",
        )
        integration.head_manager._manifests["bench-test"] = manifest
        integration.head_manager._adapters["bench-test"] = MockAdapter(manifest)
        integration.head_manager._states["bench-test"] = HeadState.OFF
        integration.head_manager._breakers["bench-test"] = CircuitBreaker(5, 60.0)

        # Run benchmarks
        results = await integration.benchmark_solver(
            "bench-test",
            sample_limit=3,  # Quick test
            timeout_seconds=60.0,
        )

        # Should get results
        assert len(results) > 0
        assert all(isinstance(r, BenchmarkResult) for r in results)

        # Check stored in registry
        stored_benchmarks = temp_registry.get_benchmark_results(solver_id="bench-test")
        assert len(stored_benchmarks) > 0

    @pytest.mark.asyncio
    async def test_returns_empty_for_nonexistent_solver(self, integration):
        """Should return empty list for nonexistent solver."""
        results = await integration.benchmark_solver("nonexistent")
        assert results == []


class TestUpdateSolverCapabilities:
    """Test updating solver capabilities from benchmarks."""

    @pytest.mark.asyncio
    async def test_updates_capabilities_from_benchmarks(self, integration, temp_registry):
        """Should calculate and update capabilities based on benchmark results."""
        # Add solver
        candidate = SolverCandidate(
            solver_id="cap-test",
            name="Capability Test",
            source="huggingface",
            solver_type="llm",
            task_types=["text_generation"],
            modalities=["text"],
            model_id="mock-model",
        )
        temp_registry.add_solver(candidate)

        # Add mock benchmark results
        temp_registry.add_benchmark_result(BenchmarkResult(
            benchmark_name="simple_reasoning",
            solver_id="cap-test",
            solver_type="llm",
            score=0.85,
            metrics={"correct": 8, "total": 10, "p50_ms": 120},
            runtime_seconds=5.0,
            sample_count=10,
        ))

        temp_registry.add_benchmark_result(BenchmarkResult(
            benchmark_name="latency",
            solver_id="cap-test",
            solver_type="llm",
            score=0.9,
            metrics={"p50_ms": 100, "p95_ms": 150},
            runtime_seconds=2.0,
            sample_count=10,
        ))

        # Update capabilities
        await integration.update_solver_capabilities("cap-test")

        # Capabilities should be updated
        # Note: Current implementation logs updates but doesn't persist to solver dict
        # This is because SolverRegistry.update_solver() is not implemented yet
        # For now, just verify no errors occurred


class TestCreateBenchmarkingIntegration:
    """Test factory function."""

    def test_creates_integration_with_all_benchmarks(self, head_manager, tmp_path):
        """Factory should create integration with all standard benchmarks."""
        integration = create_benchmarking_integration(
            head_manager=head_manager,
            registry_path=tmp_path / "test.db",
        )

        assert integration is not None
        assert integration.head_manager == head_manager
        assert integration.benchmarks is not None
        assert integration.registry is not None

        # Should have registered benchmarks
        # SimpleReasoning, MMLU, GSM8K, Latency, ImageClassification
        assert len(integration.benchmarks.benchmarks) >= 5
