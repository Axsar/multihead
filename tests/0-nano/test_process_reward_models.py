"""Tests for Process Reward Models (PRM)."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from multihead.models import StageResult, StepStatus
from multihead.process_reward_models import (
    CompositePRMScorer,
    LLMPRMScorer,
    PathScore,
    PRMScore,
    PRMScorer,
    RubricPRMScorer,
    StepQuality,
)


class TestPRMScore:
    """Test PRMScore dataclass."""

    def test_creates_score(self):
        """Should create a PRM score."""
        score = PRMScore(
            step_id="step-1",
            quality=StepQuality.CORRECT,
            score=0.9,
            confidence=0.8,
            feedback="Step is logically sound",
        )

        assert score.step_id == "step-1"
        assert score.quality == StepQuality.CORRECT
        assert score.score == 0.9
        assert score.confidence == 0.8

    def test_is_acceptable_above_threshold(self):
        """Should accept steps above threshold."""
        score = PRMScore(
            step_id="step-1",
            quality=StepQuality.CORRECT,
            score=0.8,
            confidence=0.9,
            feedback="Good",
        )

        assert score.is_acceptable(threshold=0.6) is True
        assert score.is_acceptable(threshold=0.9) is False

    def test_is_acceptable_rejects_incorrect(self):
        """Should reject incorrect steps regardless of score."""
        score = PRMScore(
            step_id="step-1",
            quality=StepQuality.INCORRECT,
            score=0.8,  # High score but marked incorrect
            confidence=0.9,
            feedback="Logic error",
        )

        assert score.is_acceptable() is False


class TestPathScore:
    """Test path-level scoring."""

    def test_from_step_scores_min_aggregation(self):
        """Should use minimum score (weakest link) for path."""
        step_scores = [
            PRMScore("s1", StepQuality.CORRECT, 0.9, 0.8, "Good"),
            PRMScore("s2", StepQuality.CORRECT, 0.7, 0.8, "OK"),
            PRMScore("s3", StepQuality.CORRECT, 0.95, 0.9, "Excellent"),
        ]

        path_score = PathScore.from_step_scores(
            "path-1", step_scores, aggregation="min"
        )

        assert path_score.total_score == 0.7  # Minimum
        assert path_score.min_score == 0.7
        assert path_score.avg_score == 0.85
        assert path_score.num_correct == 3

    def test_from_step_scores_avg_aggregation(self):
        """Should use average score for path."""
        step_scores = [
            PRMScore("s1", StepQuality.CORRECT, 0.6, 0.8, "OK"),
            PRMScore("s2", StepQuality.CORRECT, 1.0, 0.9, "Perfect"),
        ]

        path_score = PathScore.from_step_scores(
            "path-1", step_scores, aggregation="avg"
        )

        assert path_score.total_score == 0.8  # Average of 0.6 and 1.0

    def test_from_step_scores_product_aggregation(self):
        """Should use product (probability) for path."""
        step_scores = [
            PRMScore("s1", StepQuality.CORRECT, 0.9, 0.8, "Good"),
            PRMScore("s2", StepQuality.CORRECT, 0.8, 0.8, "Good"),
        ]

        path_score = PathScore.from_step_scores(
            "path-1", step_scores, aggregation="product"
        )

        assert path_score.total_score == pytest.approx(0.72)  # 0.9 * 0.8

    def test_counts_quality_levels(self):
        """Should count steps by quality."""
        step_scores = [
            PRMScore("s1", StepQuality.CORRECT, 0.9, 0.8, "Good"),
            PRMScore("s2", StepQuality.PARTIALLY_CORRECT, 0.7, 0.7, "OK"),
            PRMScore("s3", StepQuality.INCORRECT, 0.2, 0.9, "Error"),
            PRMScore("s4", StepQuality.CORRECT, 0.95, 0.85, "Great"),
        ]

        path_score = PathScore.from_step_scores("path-1", step_scores)

        assert path_score.num_correct == 2
        assert path_score.num_partial == 1
        assert path_score.num_incorrect == 1

    def test_identifies_first_error(self):
        """Should identify where first error occurred."""
        step_scores = [
            PRMScore("s1", StepQuality.CORRECT, 0.9, 0.8, "Good"),
            PRMScore("s2", StepQuality.CORRECT, 0.85, 0.8, "Good"),
            PRMScore("s3", StepQuality.INCORRECT, 0.2, 0.9, "First error"),
            PRMScore("s4", StepQuality.INCORRECT, 0.1, 0.8, "Second error"),
        ]

        path_score = PathScore.from_step_scores("path-1", step_scores)

        assert path_score.first_error_at == "s3"

    def test_handles_empty_scores(self):
        """Should handle empty step scores list."""
        path_score = PathScore.from_step_scores("path-1", [])

        assert path_score.total_score == 0.0
        assert path_score.num_correct == 0


class TestLLMPRMScorer:
    """Test LLM-based PRM scorer."""

    @pytest.mark.asyncio
    async def test_scores_step(self):
        """Should score a step using LLM."""
        mock_generate = AsyncMock(return_value="""
Quality: correct
Score: 8
Confidence: 9
Feedback: Step correctly applies the formula
Issues: none
Strengths: Clear logical progression
""")

        scorer = LLMPRMScorer(mock_generate)

        result = StageResult(
            step_id="test-step",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
            outputs={"result": "x = 5"},
        )

        score = await scorer.score_step(result, {"goal": "Solve for x"})

        assert score.step_id == "test-step"
        assert score.quality == StepQuality.CORRECT
        assert score.score == 0.8  # 8/10 normalized
        assert score.confidence == 0.9  # 9/10 normalized

    @pytest.mark.asyncio
    async def test_parses_incorrect_quality(self):
        """Should parse incorrect quality from LLM."""
        mock_generate = AsyncMock(return_value="""
Quality: incorrect
Score: 2
Confidence: 8
Feedback: Logic error in step 3
Issues: Forgot to square both sides
""")

        scorer = LLMPRMScorer(mock_generate)
        result = StageResult(
            step_id="test",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
            outputs={"result": "error"},
        )

        score = await scorer.score_step(result, {})

        assert score.quality == StepQuality.INCORRECT
        assert score.score == 0.2

    @pytest.mark.asyncio
    async def test_handles_llm_error(self):
        """Should handle LLM errors gracefully."""
        mock_generate = AsyncMock(side_effect=Exception("LLM timeout"))

        scorer = LLMPRMScorer(mock_generate)
        result = StageResult(
            step_id="test",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
        )

        score = await scorer.score_step(result, {})

        assert score.quality == StepQuality.UNKNOWN
        assert score.score == 0.5  # Default
        assert score.confidence == 0.0
        assert "failed" in score.feedback.lower()

    @pytest.mark.asyncio
    async def test_includes_previous_steps_in_context(self):
        """Should include previous steps in LLM prompt."""
        mock_generate = AsyncMock(
            return_value="Quality: correct\nScore: 7\nConfidence: 8\nFeedback: Good"
        )

        scorer = LLMPRMScorer(mock_generate)

        prev_step = StageResult(
            step_id="prev",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
            outputs={"result": "Step 1 output"},
        )

        current_step = StageResult(
            step_id="current",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
            outputs={"result": "Step 2 output"},
        )

        await scorer.score_step(
            current_step,
            {"goal": "Test", "previous_steps": [prev_step]},
        )

        # Verify LLM was called with previous steps in prompt
        call_args = mock_generate.call_args[0][0]
        assert "Previous steps" in call_args
        assert "Step 1 output" in call_args


class TestRubricPRMScorer:
    """Test rubric-based PRM scorer."""

    @pytest.mark.asyncio
    async def test_scores_using_rubric(self):
        """Should score step using predefined rubric."""
        # Define rubric checks
        def has_output(result, ctx):
            return bool(result.outputs)

        def no_errors(result, ctx):
            return not result.error

        def reasonable_length(result, ctx):
            if not result.outputs:
                return False
            output = str(result.outputs.get("result", ""))
            return 10 <= len(output) <= 1000

        rubric = [
            ("Has output", has_output, 0.4),
            ("No errors", no_errors, 0.4),
            ("Reasonable length", reasonable_length, 0.2),
        ]

        scorer = RubricPRMScorer(rubric)

        result = StageResult(
            step_id="test",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
            outputs={"result": "This is a good answer with reasonable length"},
        )

        score = await scorer.score_step(result, {})

        assert score.score == 1.0  # Passed all criteria
        assert score.quality == StepQuality.CORRECT
        assert len(score.strengths) == 3

    @pytest.mark.asyncio
    async def test_partial_score_for_some_criteria(self):
        """Should give partial score when some criteria fail."""
        def has_output(result, ctx):
            return bool(result.outputs)

        def no_errors(result, ctx):
            return not result.error

        rubric = [
            ("Has output", has_output, 0.5),
            ("No errors", no_errors, 0.5),
        ]

        scorer = RubricPRMScorer(rubric)

        # Result with output but has error
        result = StageResult(
            step_id="test",
            head_id="mock-llm",
            status=StepStatus.FAILED,
            outputs={"result": "Some output"},
            error="Something failed",
        )

        score = await scorer.score_step(result, {})

        assert score.score == 0.5  # Passed 1/2 criteria
        assert score.quality == StepQuality.NEUTRAL
        assert len(score.strengths) == 1
        assert len(score.issues) == 1


class TestCompositePRMScorer:
    """Test composite PRM scorer."""

    @pytest.mark.asyncio
    async def test_combines_multiple_scorers(self):
        """Should combine scores from multiple scorers."""
        # Create mock scorers
        scorer1 = AsyncMock(spec=PRMScorer)
        scorer1.score_step = AsyncMock(return_value=PRMScore(
            "test", StepQuality.CORRECT, 0.8, 0.9, "Good from scorer 1"
        ))

        scorer2 = AsyncMock(spec=PRMScorer)
        scorer2.score_step = AsyncMock(return_value=PRMScore(
            "test", StepQuality.PARTIALLY_CORRECT, 0.6, 0.7, "OK from scorer 2"
        ))

        composite = CompositePRMScorer([
            (scorer1, 0.7),  # 70% weight
            (scorer2, 0.3),  # 30% weight
        ])

        result = StageResult(
            step_id="test",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
        )

        score = await composite.score_step(result, {})

        # Weighted average: 0.8*0.7 + 0.6*0.3 = 0.56 + 0.18 = 0.74
        assert score.score == pytest.approx(0.74)
        assert score.confidence == pytest.approx(0.84)  # 0.9*0.7 + 0.7*0.3

    @pytest.mark.asyncio
    async def test_min_aggregation(self):
        """Should use minimum score when aggregation is 'min'."""
        scorer1 = AsyncMock(spec=PRMScorer)
        scorer1.score_step = AsyncMock(return_value=PRMScore(
            "test", StepQuality.CORRECT, 0.9, 0.9, "Great"
        ))

        scorer2 = AsyncMock(spec=PRMScorer)
        scorer2.score_step = AsyncMock(return_value=PRMScore(
            "test", StepQuality.PARTIALLY_CORRECT, 0.5, 0.8, "Weak link"
        ))

        composite = CompositePRMScorer(
            [(scorer1, 1.0), (scorer2, 1.0)],
            aggregation="min"
        )

        result = StageResult(
            step_id="test",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
        )

        score = await composite.score_step(result, {})

        assert score.score == 0.5  # Minimum of 0.9 and 0.5

    @pytest.mark.asyncio
    async def test_aggregates_issues_and_strengths(self):
        """Should combine issues and strengths from all scorers."""
        scorer1 = AsyncMock(spec=PRMScorer)
        scorer1.score_step = AsyncMock(return_value=PRMScore(
            "test", StepQuality.CORRECT, 0.8, 0.9, "Good",
            issues=["Issue A"],
            strengths=["Strength A", "Strength B"]
        ))

        scorer2 = AsyncMock(spec=PRMScorer)
        scorer2.score_step = AsyncMock(return_value=PRMScore(
            "test", StepQuality.PARTIALLY_CORRECT, 0.6, 0.7, "OK",
            issues=["Issue B"],
            strengths=["Strength A"]  # Duplicate
        ))

        composite = CompositePRMScorer([(scorer1, 1.0), (scorer2, 1.0)])

        result = StageResult(
            step_id="test",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
        )

        score = await composite.score_step(result, {})

        # Should have both issues
        assert set(score.issues) == {"Issue A", "Issue B"}
        # Should deduplicate strengths
        assert set(score.strengths) == {"Strength A", "Strength B"}


class TestPathScoring:
    """Test scoring entire reasoning paths."""

    @pytest.mark.asyncio
    async def test_scores_path(self):
        """Should score entire path using PRM scorer."""
        mock_generate = AsyncMock(side_effect=[
            "Quality: correct\nScore: 9\nConfidence: 8\nFeedback: Good",
            "Quality: correct\nScore: 8\nConfidence: 9\nFeedback: Good",
            "Quality: partially_correct\nScore: 6\nConfidence: 7\nFeedback: OK",
        ])

        scorer = LLMPRMScorer(mock_generate)

        results = [
            StageResult(
                step_id=f"step-{i}",
                head_id="mock-llm",
                status=StepStatus.COMMITTED,
                outputs={"result": f"Output {i}"}
            )
            for i in range(3)
        ]

        path_score = await scorer.score_path(
            results,
            {"goal": "Test", "path_id": "test-path"},
            aggregation="min"
        )

        assert path_score.path_id == "test-path"
        assert len(path_score.step_scores) == 3
        assert path_score.total_score == 0.6  # Minimum (weakest link)
        assert path_score.num_correct == 2
        assert path_score.num_partial == 1
