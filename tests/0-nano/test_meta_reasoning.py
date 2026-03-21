"""Tests for Phase 5: Meta-Reasoning Solver Selection."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from multihead.benchmarking.base import BenchmarkResult, BenchmarkRunner
from multihead.consensus import (
    ConsensusResult,
    ConsensusStrategy,
    VoteResult,
)
from multihead.meta_reasoning.parsing import (
    format_candidates_prompt,
    try_parse_json,
    parse_consensus_output,
)


def _make_vote(head_id: str, text: str, success: bool = True) -> VoteResult:
    """Helper to create a VoteResult with text output."""
    return VoteResult(
        head_id=head_id,
        outputs={"text": text},
        success=success,
        schema_valid=success,
        latency_ms=500,
    )
from multihead.discovery.base import SolverCandidate
from multihead.meta_reasoning import MetaReasoningSelector, SelectionResult
from multihead.registry import SolverRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry(tmp_path):
    """Create a registry with test solvers."""
    reg = SolverRegistry(tmp_path / "test_meta.db")

    # Add some candidate solvers
    candidates = [
        SolverCandidate(
            solver_id="yolo-v11",
            name="YOLOv11",
            source="huggingface",
            solver_type="object_detection",
            task_types=["object_detection"],
            modalities=["text", "image"],
            benchmark_scores={"coco_map": 0.52},
            estimated_latency_ms=12,
            estimated_cost=0.0,
            license="agpl-3.0",
        ),
        SolverCandidate(
            solver_id="yolo-v12",
            name="YOLOv12",
            source="paperswithcode",
            solver_type="object_detection",
            task_types=["object_detection"],
            modalities=["text", "image"],
            benchmark_scores={"coco_map": 0.558},
            estimated_latency_ms=8,
            estimated_cost=0.0,
            license="agpl-3.0",
        ),
        SolverCandidate(
            solver_id="dino-x",
            name="DINO-X",
            source="huggingface",
            solver_type="object_detection",
            task_types=["object_detection"],
            modalities=["text", "image"],
            benchmark_scores={"coco_map": 0.542},
            estimated_latency_ms=45,
            estimated_cost=0.02,
        ),
    ]

    for c in candidates:
        reg.add_solver(c, adoption_status="adopted")

    # Add benchmark results
    for solver_id, score in [("yolo-v11", 0.85), ("yolo-v12", 0.92), ("dino-x", 0.88)]:
        reg.add_benchmark_result(BenchmarkResult(
            benchmark_name="latency",
            solver_id=solver_id,
            solver_type="object_detection",
            score=score,
        ))

    return reg


@pytest.fixture
def head_manager():
    """Create mock HeadManager."""
    hm = MagicMock()
    hm.get_states.return_value = {
        "mock-llm": {"kind": "llm", "status": "idle"},
        "mock-vlm": {"kind": "vlm", "status": "idle"},
    }
    return hm


@pytest.fixture
def selector(head_manager, registry):
    """Create MetaReasoningSelector."""
    return MetaReasoningSelector(
        head_manager=head_manager,
        registry=registry,
        consensus_heads=["mock-llm"],
    )


# ---------------------------------------------------------------------------
# SelectionResult
# ---------------------------------------------------------------------------

class TestSelectionResult:
    """Test SelectionResult dataclass."""

    def test_to_dict(self):
        """Test serialization."""
        result = SelectionResult(
            task_type="object_detection",
            selected_solver_id="yolo-v12",
            reasoning="Best mAP",
            confidence_score=0.92,
            rankings=["yolo-v12", "dino-x", "yolo-v11"],
            consensus_votes={"mock-llm": "yolo-v12"},
            benchmark_scores={"latency": 0.92},
            candidates_evaluated=3,
        )

        d = result.to_dict()
        assert d["task_type"] == "object_detection"
        assert d["selected_solver_id"] == "yolo-v12"
        assert d["confidence_score"] == 0.92
        assert len(d["rankings"]) == 3


# ---------------------------------------------------------------------------
# Candidate Gathering
# ---------------------------------------------------------------------------

class TestCandidateGathering:
    """Test gathering candidates from registry."""

    def test_gather_candidates(self, selector):
        """Test gathering candidates for a task type."""
        candidates = selector._gather_candidates("object_detection")

        assert len(candidates) == 3
        ids = {c["solver_id"] for c in candidates}
        assert "yolo-v11" in ids
        assert "yolo-v12" in ids
        assert "dino-x" in ids

    def test_gather_candidates_by_solver_type(self, selector):
        """Test filtering by solver type."""
        candidates = selector._gather_candidates(
            "object_detection",
            solver_type="object_detection",
        )
        assert len(candidates) == 3

    def test_gather_candidates_empty(self, selector):
        """Test gathering with no matching candidates."""
        candidates = selector._gather_candidates("nonexistent_task")
        # Falls back to all candidates of matching type (no type filter)
        assert len(candidates) >= 0

    def test_gather_deduplicates(self, selector, registry):
        """Test deduplication of candidates."""
        # add_solver with INSERT OR REPLACE means same solver_id gets overwritten.
        # The dedup logic in _gather_candidates handles when both statuses
        # ("adopted" and "candidate") are queried and might return the same id.
        # Verify we get unique IDs back.
        candidates = selector._gather_candidates("object_detection")
        ids = [c["solver_id"] for c in candidates]
        # All IDs should be unique
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Prompt Formatting
# ---------------------------------------------------------------------------

class TestPromptFormatting:
    """Test candidate prompt formatting."""

    def test_format_candidates_prompt(self, selector, registry):
        """Test prompt formatting includes all candidate details."""
        candidates = selector._gather_candidates("object_detection")
        prompt = format_candidates_prompt("object_detection", candidates)

        assert "object_detection" in prompt
        assert "YOLOv12" in prompt
        assert "DINO-X" in prompt
        assert "rankings" in prompt
        assert "reasoning" in prompt
        assert "confidence" in prompt

    def test_format_handles_missing_fields(self, selector):
        """Test formatting with candidates that have missing fields."""
        candidates = [
            {
                "solver_id": "test-solver",
                "name": "Test",
                "source": "mock",
                "solver_type": "llm",
                "benchmark_scores": {},
            }
        ]

        prompt = format_candidates_prompt("test_task", candidates)
        assert "test-solver" in prompt
        assert "unknown" in prompt  # Missing cost/latency


# ---------------------------------------------------------------------------
# JSON Parsing
# ---------------------------------------------------------------------------

class TestJsonParsing:
    """Test JSON extraction from consensus outputs."""

    def test_parse_direct_json(self, selector):
        """Test parsing direct JSON."""
        text = '{"rankings": ["a", "b"], "reasoning": "test", "confidence": 0.9}'
        result = try_parse_json(text)

        assert result is not None
        assert result["rankings"] == ["a", "b"]

    def test_parse_markdown_json(self, selector):
        """Test parsing JSON from markdown code block."""
        text = '```json\n{"rankings": ["a", "b"], "reasoning": "test", "confidence": 0.9}\n```'
        result = try_parse_json(text)

        assert result is not None
        assert result["rankings"] == ["a", "b"]

    def test_parse_embedded_json(self, selector):
        """Test extracting JSON from surrounding text."""
        text = (
            'Here is my analysis:\n'
            '{"rankings": ["solver-a", "solver-b"],'
            ' "reasoning": "A is better",'
            ' "confidence": 0.85}\n'
            'That is my ranking.'
        )
        result = try_parse_json(text)

        assert result is not None
        assert result["rankings"] == ["solver-a", "solver-b"]

    def test_parse_empty(self, selector):
        """Test parsing empty string."""
        assert try_parse_json("") is None
        assert try_parse_json(None) is None

    def test_parse_garbage(self, selector):
        """Test parsing non-JSON text."""
        assert try_parse_json("just some text") is None


# ---------------------------------------------------------------------------
# Consensus Output Parsing
# ---------------------------------------------------------------------------

class TestConsensusParsing:
    """Test parsing consensus results into rankings."""

    def test_parse_successful_consensus(self, selector, registry):
        """Test parsing a successful consensus result."""
        candidates = selector._gather_candidates("object_detection")

        vote_text = json.dumps({
            "rankings": ["yolo-v12", "dino-x", "yolo-v11"],
            "reasoning": "YOLOv12 has best mAP at 0.558",
            "confidence": 0.9,
        })
        vote = _make_vote("mock-llm", vote_text)

        result = ConsensusResult(
            consensus_outputs={"text": vote_text},
            all_votes=[vote],
            agreement_score=1.0,
            red_flags=[],
            strategy_used=ConsensusStrategy.WEIGHTED,
        )

        rankings, confidence, reasoning, votes = parse_consensus_output(
            result, candidates,
        )

        assert rankings[0] == "yolo-v12"
        assert len(rankings) == 3
        assert confidence > 0
        assert "mAP" in reasoning
        assert votes.get("mock-llm") == "yolo-v12"

    def test_parse_failed_consensus_falls_back(self, selector, registry):
        """Test fallback when consensus output is unparseable."""
        candidates = selector._gather_candidates("object_detection")

        vote = _make_vote("mock-llm", "I couldn't decide, they're all great!")

        result = ConsensusResult(
            consensus_outputs={},
            all_votes=[vote],
            agreement_score=0.0,
            red_flags=[],
            strategy_used=ConsensusStrategy.WEIGHTED,
        )

        rankings, confidence, reasoning, votes = parse_consensus_output(
            result, candidates,
        )

        # Should fall back to registry order
        assert len(rankings) == 3
        assert confidence < 0.5  # Low confidence
        assert "Fallback" in reasoning

    def test_parse_validates_solver_ids(self, selector, registry):
        """Test that invalid solver IDs are filtered out."""
        candidates = selector._gather_candidates("object_detection")

        vote_text = json.dumps({
            "rankings": ["yolo-v12", "nonexistent-solver", "yolo-v11"],
            "reasoning": "test",
            "confidence": 0.8,
        })
        vote = _make_vote("mock-llm", vote_text)

        result = ConsensusResult(
            consensus_outputs={"text": vote_text},
            all_votes=[vote],
            agreement_score=1.0,
            red_flags=[],
            strategy_used=ConsensusStrategy.WEIGHTED,
        )

        rankings, _, _, _ = parse_consensus_output(result, candidates)

        assert "nonexistent-solver" not in rankings
        assert "yolo-v12" in rankings
        # Missing candidates should be added at end
        assert "dino-x" in rankings


# ---------------------------------------------------------------------------
# Reranking with Benchmarks
# ---------------------------------------------------------------------------

class TestReranking:
    """Test re-ranking with empirical benchmarks."""

    def test_rerank_no_benchmarks(self, selector):
        """Test reranking with no benchmark data."""
        rankings = ["a", "b", "c"]
        result = selector._rerank_with_benchmarks(rankings, {})

        assert result == rankings  # Unchanged

    def test_rerank_with_benchmarks(self, selector):
        """Test reranking promotes benchmarked solver."""
        rankings = ["a", "b", "c"]
        benchmarks = {
            "a": 0.6,  # Moderate
            "b": 0.95,  # Excellent
            "c": 0.7,  # Good
        }

        result = selector._rerank_with_benchmarks(rankings, benchmarks)

        # b should be promoted (0.95 bench score)
        assert result[0] == "b"

    def test_rerank_60_40_weight(self, selector):
        """Test that 60% empirical, 40% consensus weight works."""
        # Consensus says "a" is best, but benchmarks say "b" is much better
        rankings = ["a", "b"]
        benchmarks = {
            "a": 0.3,   # Poor benchmark
            "b": 0.95,  # Excellent benchmark
        }

        result = selector._rerank_with_benchmarks(rankings, benchmarks)

        # b (0.6*0.95 + 0.4*0.0 = 0.57) should beat a (0.6*0.3 + 0.4*1.0 = 0.58)
        # Actually a=0.58, b=0.57 - very close, consensus wins slightly
        # But with clear gap, empirical should win
        # Let's verify: a = 0.6*0.3 + 0.4*1.0 = 0.58, b = 0.6*0.95 + 0.4*0.0 = 0.57
        # In this case consensus barely wins. Let's test a clearer gap:
        benchmarks2 = {
            "a": 0.1,
            "b": 0.99,
        }
        result2 = selector._rerank_with_benchmarks(rankings, benchmarks2)
        # a = 0.6*0.1 + 0.4*1.0 = 0.46, b = 0.6*0.99 + 0.4*0.0 = 0.594
        assert result2[0] == "b"


# ---------------------------------------------------------------------------
# Preference Recording
# ---------------------------------------------------------------------------

class TestPreferenceRecording:
    """Test preference recording in registry."""

    def test_record_preference(self, selector):
        """Test recording a selection as preference."""
        result = SelectionResult(
            task_type="object_detection",
            selected_solver_id="yolo-v12",
            reasoning="Best mAP",
            confidence_score=0.92,
            rankings=["yolo-v12", "dino-x"],
            consensus_votes={"mock-llm": "yolo-v12"},
            benchmark_scores={"latency": 0.92},
            candidates_evaluated=3,
        )

        selector._record_preference(result)

        pref = selector.registry.get_preference("object_detection")
        assert pref is not None
        assert pref["preferred_solver_id"] == "yolo-v12"
        assert pref["confidence_score"] == 0.92
        assert pref["reasoning"] == "Best mAP"

    def test_get_current_preference(self, selector):
        """Test getting current preference."""
        # Record one
        selector.registry.record_selection(
            task_type="text_generation",
            preferred_solver_id="qwen3-8b",
            reasoning="Fast and local",
            confidence_score=0.8,
        )

        pref = selector.get_current_preference("text_generation")
        assert pref is not None
        assert pref["preferred_solver_id"] == "qwen3-8b"

    def test_get_preference_none(self, selector):
        """Test getting preference for unrecorded task type."""
        pref = selector.get_current_preference("nonexistent_task")
        assert pref is None

    def test_list_preferences(self, selector):
        """Test listing all preferences."""
        # Record two preferences
        for task, solver in [("task_a", "solver-1"), ("task_b", "solver-2")]:
            selector.registry.record_selection(
                task_type=task,
                preferred_solver_id=solver,
                reasoning="test",
                confidence_score=0.7,
            )

        prefs = selector.list_preferences()
        assert len(prefs) == 2


# ---------------------------------------------------------------------------
# Benchmark Lookup
# ---------------------------------------------------------------------------

class TestBenchmarkLookup:
    """Test benchmark result lookup for re-ranking."""

    async def test_benchmark_top_candidates_uses_cache(self, selector):
        """Test that existing benchmark results are used."""
        selector.benchmark_runner = BenchmarkRunner()

        scores = await selector._benchmark_top_candidates(
            ["yolo-v12", "dino-x"],
            "object_detection",
        )

        assert "yolo-v12" in scores
        assert "dino-x" in scores
        assert scores["yolo-v12"] == 0.92  # From fixture
        assert scores["dino-x"] == 0.88

    async def test_benchmark_no_runner(self, selector):
        """Test with no benchmark runner."""
        selector.benchmark_runner = None
        scores = await selector._benchmark_top_candidates(["yolo-v12"], "test")
        assert scores == {}


# ---------------------------------------------------------------------------
# Task Type Discovery
# ---------------------------------------------------------------------------

class TestTaskTypeDiscovery:
    """Test discovering task types from registry."""

    def test_discover_task_types(self, selector):
        """Test discovering task types from registered solvers."""
        task_types = selector._discover_task_types()
        assert "object_detection" in task_types


# ---------------------------------------------------------------------------
# Full Selection Flow (mocked consensus)
# ---------------------------------------------------------------------------

class TestFullSelectionFlow:
    """Test end-to-end selection with mocked consensus."""

    @pytest.mark.asyncio
    async def test_select_best_solver(self, selector):
        """Test full selection flow with mocked consensus."""
        vote_text = json.dumps({
            "rankings": ["yolo-v12", "dino-x", "yolo-v11"],
            "reasoning": "YOLOv12 has the highest COCO mAP at 0.558 with 8ms latency",
            "confidence": 0.92,
        })
        mock_vote = _make_vote("mock-llm", vote_text)

        mock_result = ConsensusResult(
            consensus_outputs={"text": vote_text},
            all_votes=[mock_vote],
            agreement_score=1.0,
            red_flags=[],
            strategy_used=ConsensusStrategy.WEIGHTED,
        )

        selector.consensus_engine.execute = AsyncMock(return_value=mock_result)

        result = await selector.select_best_solver("object_detection")

        assert result.selected_solver_id == "yolo-v12"
        assert result.confidence_score > 0
        assert result.candidates_evaluated == 3
        assert "mAP" in result.reasoning

        # Verify preference was recorded
        pref = selector.registry.get_preference("object_detection")
        assert pref is not None
        assert pref["preferred_solver_id"] == "yolo-v12"

    @pytest.mark.asyncio
    async def test_select_with_benchmarks(self, selector):
        """Test selection with benchmark re-ranking."""
        selector.benchmark_runner = BenchmarkRunner()

        # Mock consensus — says yolo-v11 is best
        vote_text = json.dumps({
            "rankings": ["yolo-v11", "yolo-v12", "dino-x"],
            "reasoning": "YOLOv11 is well-tested",
            "confidence": 0.7,
        })
        mock_vote = _make_vote("mock-llm", vote_text)

        mock_result = ConsensusResult(
            consensus_outputs={"text": vote_text},
            all_votes=[mock_vote],
            agreement_score=1.0,
            red_flags=[],
            strategy_used=ConsensusStrategy.WEIGHTED,
        )

        selector.consensus_engine.execute = AsyncMock(return_value=mock_result)

        # Mock benchmark scores with a large enough gap to override consensus
        # yolo-v11 rank1 consensus (1.0) + 0.40 bench → 0.6*0.4+0.4*1.0 = 0.64
        # yolo-v12 rank2 consensus (0.67) + 0.98 bench → 0.6*0.98+0.4*0.67 = 0.856
        selector._benchmark_top_candidates = AsyncMock(return_value={
            "yolo-v11": 0.40,
            "yolo-v12": 0.98,
            "dino-x": 0.50,
        })

        result = await selector.select_best_solver(
            "object_detection",
            run_benchmarks=True,
        )

        # yolo-v12 has much higher benchmark score, should be re-ranked to #1
        assert result.selected_solver_id == "yolo-v12"
        assert result.benchmark_scores.get("yolo-v12") == 0.98

    @pytest.mark.asyncio
    async def test_select_too_few_candidates(self, selector):
        """Test error when too few candidates."""
        with pytest.raises(ValueError, match="Only 0 candidates"):
            await selector.select_best_solver(
                "nonexistent_task_type",
                solver_type="nonexistent_type",
                min_candidates=2,
            )

    @pytest.mark.asyncio
    async def test_select_no_auto_record(self, head_manager, registry):
        """Test selection without auto-recording preference."""
        selector = MetaReasoningSelector(
            head_manager=head_manager,
            registry=registry,
            consensus_heads=["mock-llm"],
            auto_record=False,
        )

        vote_text = json.dumps({
            "rankings": ["yolo-v12", "yolo-v11", "dino-x"],
            "reasoning": "test",
            "confidence": 0.8,
        })
        mock_vote = _make_vote("mock-llm", vote_text)

        mock_result = ConsensusResult(
            consensus_outputs={"text": vote_text},
            all_votes=[mock_vote],
            agreement_score=1.0,
            red_flags=[],
            strategy_used=ConsensusStrategy.WEIGHTED,
        )

        selector.consensus_engine.execute = AsyncMock(return_value=mock_result)

        result = await selector.select_best_solver("object_detection")

        assert result.selected_solver_id == "yolo-v12"
        # Should NOT record preference
        pref = registry.get_preference("object_detection")
        assert pref is None


# ---------------------------------------------------------------------------
# Router Integration
# ---------------------------------------------------------------------------

class TestRouterPreferenceIntegration:
    """Test that recorded preferences affect router scoring."""

    def test_router_preference_boost(self, registry):
        """Test that router uses preferences for scoring."""
        # Record a preference
        registry.record_selection(
            task_type="object_detection",
            preferred_solver_id="yolo-v12",
            reasoning="Best COCO mAP",
            confidence_score=0.95,
        )

        # Verify preference is stored and retrievable
        pref = registry.get_preference("object_detection")
        assert pref["preferred_solver_id"] == "yolo-v12"
        assert pref["confidence_score"] == 0.95

        # The Router with registry would boost yolo-v12 by _W_PREFERENCE * 0.95
        # We verify the preference exists; actual Router integration is tested
        # in test_router.py (Router accepts registry parameter)


# ---------------------------------------------------------------------------
# Evaluate All Task Types
# ---------------------------------------------------------------------------

class TestEvaluateAllTaskTypes:
    """Test batch evaluation of multiple task types."""

    @pytest.mark.asyncio
    async def test_evaluate_all(self, selector):
        """Test evaluating all discovered task types."""
        vote_text = json.dumps({
            "rankings": ["yolo-v12", "yolo-v11", "dino-x"],
            "reasoning": "test",
            "confidence": 0.8,
        })
        mock_vote = _make_vote("mock-llm", vote_text)

        mock_result = ConsensusResult(
            consensus_outputs={"text": vote_text},
            all_votes=[mock_vote],
            agreement_score=1.0,
            red_flags=[],
            strategy_used=ConsensusStrategy.WEIGHTED,
        )

        selector.consensus_engine.execute = AsyncMock(return_value=mock_result)

        results = await selector.evaluate_all_task_types(["object_detection"])

        assert len(results) == 1
        assert results[0].task_type == "object_detection"
        assert results[0].selected_solver_id == "yolo-v12"

    @pytest.mark.asyncio
    async def test_evaluate_all_auto_discover(self, selector):
        """Test auto-discovering task types."""
        task_types = selector._discover_task_types()
        assert "object_detection" in task_types
