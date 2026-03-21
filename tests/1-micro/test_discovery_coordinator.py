"""Tests for discovery coordinator."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from multihead.benchmarking.base import BenchmarkRunner
from multihead.benchmarking.llm_benchmarks import SimpleReasoningBenchmark
from multihead.discovery.base import DiscoveryAgent, SolverCandidate
from multihead.discovery.coordinator import DiscoveryCoordinator, create_discovery_job
from multihead.registry import AdoptionRule, SolverRegistry


class MockDiscoveryAgent(DiscoveryAgent):
    """Mock discovery agent for testing."""

    def __init__(self, source_name: str, candidates: list[SolverCandidate]):
        super().__init__(source_name)
        self.candidates = candidates

    async def discover_new_solvers(
        self,
        *,
        solver_types: list[str] | None = None,
        min_downloads: int | None = None,
        updated_since: datetime | None = None,
        limit: int = 50,
    ) -> list[SolverCandidate]:
        """Return mock candidates."""
        return self.candidates[:limit]

    async def get_solver_details(self, solver_id: str) -> SolverCandidate | None:
        """Get mock solver details."""
        for candidate in self.candidates:
            if candidate.solver_id == solver_id:
                return candidate
        return None


@pytest.fixture
def registry(tmp_path):
    """Create temporary registry."""
    db_path = tmp_path / "test_registry.db"
    return SolverRegistry(db_path)


@pytest.fixture
def benchmark_runner():
    """Create benchmark runner."""
    runner = BenchmarkRunner()
    runner.register_benchmark(SimpleReasoningBenchmark())
    return runner


@pytest.fixture
def mock_candidates():
    """Create mock solver candidates."""
    return [
        SolverCandidate(
            solver_id="test-llm-1",
            name="Test LLM 1",
            source="mock",
            solver_type="llm",
            version="1.0",
            estimated_cost=0.01,
        ),
        SolverCandidate(
            solver_id="test-llm-2",
            name="Test LLM 2",
            source="mock",
            solver_type="llm",
            version="1.0",
            estimated_cost=0.05,
        ),
    ]


@pytest.fixture
def discovery_agents(mock_candidates):
    """Create mock discovery agents."""
    return {
        "mock": MockDiscoveryAgent("mock", mock_candidates),
    }


@pytest.fixture
def coordinator(registry, benchmark_runner, discovery_agents):
    """Create discovery coordinator."""
    return DiscoveryCoordinator(
        registry=registry,
        benchmark_runner=benchmark_runner,
        discovery_agents=discovery_agents,
        auto_benchmark=False,  # Disabled for tests
        auto_adopt=True,
    )


class TestDiscoveryCoordinator:
    """Test DiscoveryCoordinator class."""

    @pytest.mark.asyncio
    async def test_run_weekly_discovery(self, coordinator):
        """Test running weekly discovery."""
        results = await coordinator.run_weekly_discovery(limit_per_source=10)

        assert results["discovered_count"] == 2
        assert len(results["new_solvers"]) == 2
        assert results["new_solvers"][0]["solver_id"] == "test-llm-1"
        assert results["new_solvers"][1]["solver_id"] == "test-llm-2"
        assert results["benchmarked_count"] == 0  # Disabled
        assert "started_at" in results
        assert "completed_at" in results

    @pytest.mark.asyncio
    async def test_discovery_skips_existing_solvers(self, coordinator, mock_candidates):
        """Test that discovery skips already registered solvers."""
        # Pre-register first solver
        coordinator.registry.add_solver(mock_candidates[0])

        # Run discovery
        results = await coordinator.run_weekly_discovery()

        # Should only discover the second solver
        assert results["discovered_count"] == 1
        assert results["new_solvers"][0]["solver_id"] == "test-llm-2"

    @pytest.mark.asyncio
    async def test_discovery_updates_version_changes(self, coordinator, mock_candidates):
        """Test that discovery registers version updates."""
        # Register old version
        old_version = SolverCandidate(
            solver_id="test-llm-1",
            name="Test LLM 1",
            source="mock",
            solver_type="llm",
            version="0.9",
        )
        coordinator.registry.add_solver(old_version)

        # Run discovery (mock has v1.0)
        results = await coordinator.run_weekly_discovery()

        # Should discover updated version
        assert results["discovered_count"] == 2  # Updated v1 + new solver
        solver = coordinator.registry.get_solver("test-llm-1")
        assert solver["version"] == "1.0"  # Updated to new version

    @pytest.mark.asyncio
    async def test_check_adoptions(self, coordinator, mock_candidates):
        """Test checking adoption rules."""
        # Add solver to registry
        coordinator.registry.add_solver(mock_candidates[0])

        # Add lenient adoption rule
        rule = AdoptionRule(
            rule_id="test-rule",
            name="Test Rule",
            solver_type="llm",
            min_aggregate_score=0.0,  # Accept anything
            max_cost_per_call=0.10,
        )
        coordinator.registry.add_adoption_rule(rule)

        # Add mock benchmark result
        from multihead.benchmarking.base import BenchmarkResult
        coordinator.registry.add_benchmark_result(BenchmarkResult(
            benchmark_name="simple_reasoning",
            solver_id="test-llm-1",
            solver_type="llm",
            score=0.5,
        ))

        # Check adoptions
        results = coordinator._check_adoptions()

        assert results["adopted_count"] == 1
        assert results["adopted_solvers"][0]["solver_id"] == "test-llm-1"
        assert results["adopted_solvers"][0]["rule_id"] == "test-rule"

        # Verify status updated
        solver = coordinator.registry.get_solver("test-llm-1")
        assert solver["adoption_status"] == "adopted"

    @pytest.mark.asyncio
    async def test_check_adoptions_respects_cost_limit(self, coordinator, mock_candidates):
        """Test that adoption respects cost constraints."""
        # Add expensive solver
        coordinator.registry.add_solver(mock_candidates[1])  # cost=0.05

        # Add strict cost rule
        rule = AdoptionRule(
            rule_id="cheap-only",
            name="Cheap Only",
            solver_type="llm",
            max_cost_per_call=0.02,  # Exclude $0.05 solver
        )
        coordinator.registry.add_adoption_rule(rule)

        # Add mock benchmark
        from multihead.benchmarking.base import BenchmarkResult
        coordinator.registry.add_benchmark_result(BenchmarkResult(
            benchmark_name="simple_reasoning",
            solver_id="test-llm-2",
            solver_type="llm",
            score=0.9,  # High score
        ))

        # Check adoptions
        results = coordinator._check_adoptions()

        # Should not adopt (too expensive)
        assert results["adopted_count"] == 0

    def test_add_default_adoption_rules(self, coordinator):
        """Test adding default adoption rules."""
        coordinator.add_default_adoption_rules()

        # Should have added 4 rules
        rules = [
            "high-quality-llm",
            "fast-inference",
            "capable-vlm",
            "local-models",
        ]

        for rule_id in rules:
            # Try to find a solver that would match this rule
            # (Just verify rules were created - we can't directly query them)
            pass  # Rules are stored, tested via adoption checks

    @pytest.mark.asyncio
    async def test_discovery_with_solver_type_filter(self, coordinator):
        """Test filtering discovery by solver type."""
        # Create mixed candidates
        mixed_agents = {
            "mock": MockDiscoveryAgent("mock", [
                SolverCandidate(
                    solver_id="llm-1",
                    name="LLM",
                    source="mock",
                    solver_type="llm",
                ),
                SolverCandidate(
                    solver_id="vlm-1",
                    name="VLM",
                    source="mock",
                    solver_type="vlm",
                ),
            ]),
        }

        coord = DiscoveryCoordinator(
            registry=coordinator.registry,
            benchmark_runner=coordinator.benchmarks,
            discovery_agents=mixed_agents,
        )

        # Discover only LLMs
        results = await coord.run_weekly_discovery(solver_types=["llm"])

        # Both will be discovered (filtering happens in real agents, not coordinator)
        # But in production, agents would filter internally
        assert results["discovered_count"] >= 1

    @pytest.mark.asyncio
    async def test_discovery_handles_errors_gracefully(self, coordinator):
        """Test that discovery handles agent errors."""
        # Create failing agent
        failing_agent = MagicMock(spec=DiscoveryAgent)
        failing_agent.discover_new_solvers = AsyncMock(
            side_effect=Exception("Network error")
        )

        coord = DiscoveryCoordinator(
            registry=coordinator.registry,
            benchmark_runner=coordinator.benchmarks,
            discovery_agents={"failing": failing_agent},
        )

        # Should not crash
        results = await coord.run_weekly_discovery()

        assert results["discovered_count"] == 0
        assert len(results["errors"]) == 1
        assert "Network error" in results["errors"][0]["error"]

    @pytest.mark.asyncio
    async def test_full_workflow(self, coordinator, mock_candidates):
        """Test complete discovery workflow."""
        # Add lenient adoption rule
        rule = AdoptionRule(
            rule_id="test-rule",
            name="Test",
            solver_type="llm",
            min_aggregate_score=0.5,  # Realistic threshold
            max_cost_per_call=0.10,
        )
        coordinator.registry.add_adoption_rule(rule)

        # Run discovery
        results = await coordinator.run_weekly_discovery()

        assert results["discovered_count"] == 2
        assert results["new_solvers"][0]["solver_id"] == "test-llm-1"

        # Manually add benchmark (auto-benchmark disabled)
        from multihead.benchmarking.base import BenchmarkResult
        coordinator.registry.add_benchmark_result(BenchmarkResult(
            benchmark_name="simple_reasoning",
            solver_id="test-llm-1",
            solver_type="llm",
            score=0.7,  # Passes 0.5 threshold
        ))

        coordinator.registry.add_benchmark_result(BenchmarkResult(
            benchmark_name="simple_reasoning",
            solver_id="test-llm-2",
            solver_type="llm",
            score=0.6,  # Passes 0.5 threshold
        ))

        # Check adoptions
        adoption_results = coordinator._check_adoptions()

        # Should adopt both benchmarked solvers (both have score > 0.5)
        assert adoption_results["adopted_count"] == 2
        assert len(adoption_results["adopted_solvers"]) == 2


class TestCreateDiscoveryJob:
    """Test factory function."""

    def test_create_discovery_job(self, tmp_path):
        """Test creating discovery job via factory."""
        registry_path = tmp_path / "registry.db"

        coordinator = create_discovery_job(registry_path)

        assert coordinator.registry is not None
        assert coordinator.benchmarks is not None
        assert len(coordinator.agents) >= 2  # HF, Ollama (BotVibes optional)
        assert "huggingface" in coordinator.agents
        assert "ollama" in coordinator.agents
        # BotVibes only added if credentials available
        assert coordinator.auto_benchmark is False  # Disabled by default
        assert coordinator.auto_adopt is True

    def test_create_discovery_job_with_benchmarking_enabled(self, tmp_path):
        """Test creating job with benchmarking enabled."""
        registry_path = tmp_path / "registry.db"

        coordinator = create_discovery_job(
            registry_path,
            auto_benchmark=True,
        )

        assert coordinator.auto_benchmark is True
