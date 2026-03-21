"""Tests for solver registry."""

from datetime import datetime, timezone, timedelta

import pytest

from multihead.benchmarking.base import BenchmarkResult
from multihead.discovery.base import SolverCandidate
from multihead.registry import AdoptionRule, SolverRegistry


@pytest.fixture
def registry(tmp_path):
    """Create temporary solver registry."""
    db_path = tmp_path / "test_registry.db"
    return SolverRegistry(db_path)


@pytest.fixture
def sample_candidate():
    """Create sample solver candidate."""
    return SolverCandidate(
        solver_id="test-llm-1",
        name="Test LLM",
        source="huggingface",
        solver_type="llm",
        task_types=["text-generation", "question-answering"],
        modalities=["text"],
        benchmark_scores={"mmlu": 0.75},
        estimated_latency_ms=500,
        estimated_cost=0.01,
        model_id="org/model-name",
        version="v1.0",
        license="apache-2.0",
        description="A test language model",
        url="https://example.com/model",
        tags=["transformer", "instruct"],
    )


@pytest.fixture
def sample_benchmark_result():
    """Create sample benchmark result."""
    return BenchmarkResult(
        benchmark_name="mmlu",
        solver_id="test-llm-1",
        solver_type="llm",
        score=0.75,
        metrics={"correct": 75, "total": 100},
        runtime_seconds=120.5,
        sample_count=100,
        error_count=0,
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )


class TestSolverRegistry:
    """Test SolverRegistry class."""

    def test_create_registry(self, tmp_path):
        """Test creating a new registry."""
        db_path = tmp_path / "test.db"
        registry = SolverRegistry(db_path)

        assert db_path.exists()
        assert registry.db_path == db_path

    def test_add_solver(self, registry, sample_candidate):
        """Test adding a solver to registry."""
        registry.add_solver(sample_candidate)

        solver = registry.get_solver("test-llm-1")
        assert solver is not None
        assert solver["solver_id"] == "test-llm-1"
        assert solver["name"] == "Test LLM"
        assert solver["solver_type"] == "llm"
        assert solver["source"] == "huggingface"
        assert solver["adoption_status"] == "candidate"

    def test_add_solver_with_custom_status(self, registry, sample_candidate):
        """Test adding solver with custom adoption status."""
        registry.add_solver(sample_candidate, adoption_status="adopted")

        solver = registry.get_solver("test-llm-1")
        assert solver["adoption_status"] == "adopted"

    def test_get_nonexistent_solver(self, registry):
        """Test getting a solver that doesn't exist."""
        solver = registry.get_solver("nonexistent")
        assert solver is None

    def test_list_all_solvers(self, registry):
        """Test listing all solvers."""
        # Add multiple solvers
        for i in range(3):
            candidate = SolverCandidate(
                solver_id=f"solver-{i}",
                name=f"Solver {i}",
                source="huggingface",
                solver_type="llm",
            )
            registry.add_solver(candidate)

        solvers = registry.list_solvers()
        assert len(solvers) == 3

    def test_list_solvers_by_type(self, registry):
        """Test filtering solvers by type."""
        # Add LLM
        llm = SolverCandidate(
            solver_id="llm-1",
            name="LLM",
            source="huggingface",
            solver_type="llm",
        )
        registry.add_solver(llm)

        # Add VLM
        vlm = SolverCandidate(
            solver_id="vlm-1",
            name="VLM",
            source="huggingface",
            solver_type="vlm",
        )
        registry.add_solver(vlm)

        # Filter by type
        llms = registry.list_solvers(solver_type="llm")
        assert len(llms) == 1
        assert llms[0]["solver_id"] == "llm-1"

        vlms = registry.list_solvers(solver_type="vlm")
        assert len(vlms) == 1
        assert vlms[0]["solver_id"] == "vlm-1"

    def test_list_solvers_by_source(self, registry):
        """Test filtering solvers by source."""
        # Add HuggingFace solver
        hf = SolverCandidate(
            solver_id="hf-1",
            name="HF Model",
            source="huggingface",
            solver_type="llm",
        )
        registry.add_solver(hf)

        # Add Ollama solver
        ollama = SolverCandidate(
            solver_id="ollama-1",
            name="Ollama Model",
            source="ollama",
            solver_type="llm",
        )
        registry.add_solver(ollama)

        # Filter by source
        hf_solvers = registry.list_solvers(source="huggingface")
        assert len(hf_solvers) == 1
        assert hf_solvers[0]["solver_id"] == "hf-1"

    def test_list_solvers_by_adoption_status(self, registry):
        """Test filtering by adoption status."""
        candidate = SolverCandidate(
            solver_id="candidate-1",
            name="Candidate",
            source="huggingface",
            solver_type="llm",
        )
        registry.add_solver(candidate, adoption_status="candidate")

        adopted = SolverCandidate(
            solver_id="adopted-1",
            name="Adopted",
            source="huggingface",
            solver_type="llm",
        )
        registry.add_solver(adopted, adoption_status="adopted")

        # Filter
        candidates = registry.list_solvers(adoption_status="candidate")
        assert len(candidates) == 1
        assert candidates[0]["solver_id"] == "candidate-1"

        adopted_list = registry.list_solvers(adoption_status="adopted")
        assert len(adopted_list) == 1
        assert adopted_list[0]["solver_id"] == "adopted-1"

    def test_add_benchmark_result(self, registry, sample_candidate, sample_benchmark_result):
        """Test storing benchmark results."""
        # Add solver first
        registry.add_solver(sample_candidate)

        # Add benchmark result
        registry.add_benchmark_result(sample_benchmark_result)

        # Retrieve results
        results = registry.get_benchmark_results("test-llm-1")
        assert len(results) == 1
        assert results[0]["benchmark_name"] == "mmlu"
        assert results[0]["score"] == 0.75
        assert results[0]["solver_id"] == "test-llm-1"

    def test_get_benchmark_results_by_name(self, registry, sample_candidate):
        """Test filtering benchmark results by name."""
        registry.add_solver(sample_candidate)

        # Add multiple benchmark results
        for bench_name in ["mmlu", "gsm8k", "mmlu"]:
            result = BenchmarkResult(
                benchmark_name=bench_name,
                solver_id="test-llm-1",
                solver_type="llm",
                score=0.8,
            )
            registry.add_benchmark_result(result)

        # Filter by name
        mmlu_results = registry.get_benchmark_results("test-llm-1", benchmark_name="mmlu")
        assert len(mmlu_results) == 2

        gsm8k_results = registry.get_benchmark_results("test-llm-1", benchmark_name="gsm8k")
        assert len(gsm8k_results) == 1

    def test_get_benchmark_results_with_limit(self, registry, sample_candidate):
        """Test limiting benchmark results."""
        registry.add_solver(sample_candidate)

        # Add 5 results
        for i in range(5):
            result = BenchmarkResult(
                benchmark_name=f"bench-{i}",
                solver_id="test-llm-1",
                solver_type="llm",
                score=0.5 + i * 0.1,
            )
            registry.add_benchmark_result(result)

        # Get only 3 most recent
        results = registry.get_benchmark_results("test-llm-1", limit=3)
        assert len(results) == 3

    def test_aggregate_score_calculation(self, registry, sample_candidate):
        """Test calculating aggregate score from benchmarks."""
        registry.add_solver(sample_candidate)

        # Add benchmark results
        benchmarks = [
            ("mmlu", 0.8),
            ("gsm8k", 0.7),
            ("reasoning", 0.9),
        ]

        for bench_name, score in benchmarks:
            result = BenchmarkResult(
                benchmark_name=bench_name,
                solver_id="test-llm-1",
                solver_type="llm",
                score=score,
            )
            registry.add_benchmark_result(result)

        # Aggregate should be average: (0.8 + 0.7 + 0.9) / 3 = 0.8
        aggregate = registry._get_aggregate_score("test-llm-1")
        assert aggregate == pytest.approx(0.8, abs=0.01)

    def test_aggregate_score_uses_latest_results(self, registry, sample_candidate):
        """Test that aggregate uses only the latest result per benchmark."""
        registry.add_solver(sample_candidate)

        # Add older MMLU result
        old_result = BenchmarkResult(
            benchmark_name="mmlu",
            solver_id="test-llm-1",
            solver_type="llm",
            score=0.6,
            started_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
        registry.add_benchmark_result(old_result)

        # Add newer MMLU result
        new_result = BenchmarkResult(
            benchmark_name="mmlu",
            solver_id="test-llm-1",
            solver_type="llm",
            score=0.9,
            started_at=datetime.now(timezone.utc),
        )
        registry.add_benchmark_result(new_result)

        # Aggregate should use new score (0.9), not old (0.6)
        aggregate = registry._get_aggregate_score("test-llm-1")
        assert aggregate == pytest.approx(0.9, abs=0.01)

    def test_compare_solvers(self, registry):
        """Test comparing two solvers."""
        # Add two solvers
        solver_a = SolverCandidate(
            solver_id="solver-a",
            name="Solver A",
            source="huggingface",
            solver_type="llm",
        )
        solver_b = SolverCandidate(
            solver_id="solver-b",
            name="Solver B",
            source="ollama",
            solver_type="llm",
        )
        registry.add_solver(solver_a)
        registry.add_solver(solver_b)

        # Add benchmarks for solver A
        registry.add_benchmark_result(BenchmarkResult(
            benchmark_name="mmlu",
            solver_id="solver-a",
            solver_type="llm",
            score=0.85,
        ))
        registry.add_benchmark_result(BenchmarkResult(
            benchmark_name="gsm8k",
            solver_id="solver-a",
            solver_type="llm",
            score=0.75,
        ))

        # Add benchmarks for solver B
        registry.add_benchmark_result(BenchmarkResult(
            benchmark_name="mmlu",
            solver_id="solver-b",
            solver_type="llm",
            score=0.70,
        ))
        registry.add_benchmark_result(BenchmarkResult(
            benchmark_name="gsm8k",
            solver_id="solver-b",
            solver_type="llm",
            score=0.80,
        ))

        # Compare
        comparison = registry.compare_solvers("solver-a", "solver-b")

        assert comparison["winner"] == "a"  # A has higher aggregate
        assert comparison["aggregate_score_a"] == pytest.approx(0.8, abs=0.01)
        assert comparison["aggregate_score_b"] == pytest.approx(0.75, abs=0.01)
        assert "mmlu" in comparison["per_benchmark"]
        assert "gsm8k" in comparison["per_benchmark"]

    def test_add_adoption_rule(self, registry):
        """Test adding an adoption rule."""
        rule = AdoptionRule(
            rule_id="high-quality-llm",
            name="High Quality LLM",
            solver_type="llm",
            min_aggregate_score=0.85,
            required_benchmarks=["mmlu", "gsm8k"],
            min_benchmark_scores={"mmlu": 0.8},
            max_cost_per_call=0.05,
            max_latency_ms=1000,
            required_license=["apache-2.0", "mit"],
            auto_register=True,
            notify_user=True,
        )

        registry.add_adoption_rule(rule)

        # Verify it was stored (by checking if it matches anything)
        # We'll test matching in next test

    def test_check_adoption_rules_matching(self, registry):
        """Test checking if solver meets adoption rules."""
        # Add rule
        rule = AdoptionRule(
            rule_id="good-llm",
            name="Good LLM",
            solver_type="llm",
            min_aggregate_score=0.75,
            required_benchmarks=["mmlu"],
            min_benchmark_scores={"mmlu": 0.7},
            max_cost_per_call=0.10,
        )
        registry.add_adoption_rule(rule)

        # Add solver that meets criteria
        solver = SolverCandidate(
            solver_id="good-solver",
            name="Good Solver",
            source="huggingface",
            solver_type="llm",
            estimated_cost=0.05,
        )
        registry.add_solver(solver)

        # Add benchmark result
        registry.add_benchmark_result(BenchmarkResult(
            benchmark_name="mmlu",
            solver_id="good-solver",
            solver_type="llm",
            score=0.8,
        ))

        # Check rules
        matching_rules = registry.check_adoption_rules("good-solver")
        assert "good-llm" in matching_rules

    def test_check_adoption_rules_not_matching(self, registry):
        """Test solver that doesn't meet adoption criteria."""
        # Add strict rule
        rule = AdoptionRule(
            rule_id="strict-llm",
            name="Strict LLM",
            solver_type="llm",
            min_aggregate_score=0.95,  # Very high threshold
        )
        registry.add_adoption_rule(rule)

        # Add mediocre solver
        solver = SolverCandidate(
            solver_id="mediocre-solver",
            name="Mediocre Solver",
            source="huggingface",
            solver_type="llm",
        )
        registry.add_solver(solver)

        # Add low benchmark result
        registry.add_benchmark_result(BenchmarkResult(
            benchmark_name="mmlu",
            solver_id="mediocre-solver",
            solver_type="llm",
            score=0.6,
        ))

        # Should not match
        matching_rules = registry.check_adoption_rules("mediocre-solver")
        assert "strict-llm" not in matching_rules

    def test_adoption_rule_excludes_sources(self, registry):
        """Test adoption rule with excluded sources."""
        # Add rule that excludes BotVibes
        rule = AdoptionRule(
            rule_id="no-botvibes",
            name="No BotVibes",
            solver_type="llm",
            min_aggregate_score=0.5,
            excluded_sources=["botvibes"],
        )
        registry.add_adoption_rule(rule)

        # Add BotVibes solver
        solver = SolverCandidate(
            solver_id="botvibes-solver",
            name="BotVibes Solver",
            source="botvibes",
            solver_type="llm",
        )
        registry.add_solver(solver)

        # Add good benchmark
        registry.add_benchmark_result(BenchmarkResult(
            benchmark_name="mmlu",
            solver_id="botvibes-solver",
            solver_type="llm",
            score=0.9,
        ))

        # Should not match due to excluded source
        matching_rules = registry.check_adoption_rules("botvibes-solver")
        assert "no-botvibes" not in matching_rules

    def test_update_adoption_status(self, registry, sample_candidate):
        """Test updating solver adoption status."""
        registry.add_solver(sample_candidate)

        # Update to adopted
        registry.update_adoption_status(
            "test-llm-1",
            "adopted",
            rule_id="some-rule",
            notes="Automatically adopted due to high performance",
        )

        # Verify
        solver = registry.get_solver("test-llm-1")
        assert solver["adoption_status"] == "adopted"
        assert solver["adoption_rule_id"] == "some-rule"
        assert "high performance" in solver["notes"]

    def test_list_solvers_by_min_aggregate_score(self, registry):
        """Test filtering solvers by minimum aggregate score."""
        # Add solver 1 with high scores
        solver1 = SolverCandidate(
            solver_id="high-scorer",
            name="High Scorer",
            source="huggingface",
            solver_type="llm",
        )
        registry.add_solver(solver1)
        registry.add_benchmark_result(BenchmarkResult(
            benchmark_name="mmlu",
            solver_id="high-scorer",
            solver_type="llm",
            score=0.9,
        ))

        # Add solver 2 with low scores
        solver2 = SolverCandidate(
            solver_id="low-scorer",
            name="Low Scorer",
            source="huggingface",
            solver_type="llm",
        )
        registry.add_solver(solver2)
        registry.add_benchmark_result(BenchmarkResult(
            benchmark_name="mmlu",
            solver_id="low-scorer",
            solver_type="llm",
            score=0.4,
        ))

        # Filter by minimum score
        good_solvers = registry.list_solvers(min_aggregate_score=0.75)
        assert len(good_solvers) == 1
        assert good_solvers[0]["solver_id"] == "high-scorer"

    def test_solver_metadata_preservation(self, registry):
        """Test that all solver metadata is preserved."""
        candidate = SolverCandidate(
            solver_id="test-solver",
            name="Test Solver",
            source="huggingface",
            solver_type="llm",
            task_types=["text-generation", "qa"],
            modalities=["text", "image"],
            benchmark_scores={"mmlu": 0.8, "gsm8k": 0.7},
            estimated_latency_ms=250,
            estimated_cost=0.02,
            model_id="org/model-123",
            version="2.0",
            license="mit",
            description="A versatile model",
            url="https://example.com",
            tags=["transformer", "fine-tuned"],
            discovery_metadata={"downloads": 10000, "likes": 500},
        )

        registry.add_solver(candidate)
        retrieved = registry.get_solver("test-solver")

        # Verify all fields
        assert retrieved["task_types"] == ["text-generation", "qa"]
        assert retrieved["modalities"] == ["text", "image"]
        assert retrieved["benchmark_scores"] == {"mmlu": 0.8, "gsm8k": 0.7}
        assert retrieved["estimated_latency_ms"] == 250
        assert retrieved["estimated_cost"] == 0.02
        assert retrieved["model_id"] == "org/model-123"
        assert retrieved["version"] == "2.0"
        assert retrieved["license"] == "mit"
        assert retrieved["description"] == "A versatile model"
        assert retrieved["url"] == "https://example.com"
        assert retrieved["tags"] == ["transformer", "fine-tuned"]
        assert retrieved["discovery_metadata"]["downloads"] == 10000
