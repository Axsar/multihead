"""Tests for PRM integration with orchestrator."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from multihead.models import StageResult, StepDef, StepStatus
from multihead.process_reward_models import PRMScore, PRMScorer, StepQuality
from multihead.prm_integration import (
    PRMOrchestrationHook,
    PRMPathSelector,
    create_rubric_for_code_steps,
    create_rubric_for_math_steps,
    get_prm_config,
    should_use_prm,
)


class TestShouldUsePRM:
    """Test PRM activation logic."""

    def test_returns_false_by_default(self):
        """Should return False for steps without PRM metadata."""
        step = StepDef(name="test", prompt_template="Test")
        assert should_use_prm(step) is False

    def test_returns_true_if_use_prm_in_extra(self):
        """Should return True if extra.use_prm is set."""
        step = StepDef(
            name="test",
            prompt_template="Test",
            extra={"use_prm": True},
        )
        assert should_use_prm(step) is True


class TestGetPRMConfig:
    """Test PRM configuration extraction."""

    def test_returns_defaults(self):
        """Should return default config if no metadata."""
        step = StepDef(name="test", prompt_template="Test")
        config = get_prm_config(step)

        assert config["scorer_type"] == "llm"
        assert config["threshold"] == 0.6
        assert config["fail_on_low_score"] is False

    def test_extracts_config_from_extra(self):
        """Should extract PRM config from extra."""
        step = StepDef(
            name="test",
            prompt_template="Test",
            extra={
                "prm_scorer": "rubric",
                "prm_threshold": 0.8,
                "prm_fail_on_low": True,
            },
        )
        config = get_prm_config(step)

        assert config["scorer_type"] == "rubric"
        assert config["threshold"] == 0.8
        assert config["fail_on_low_score"] is True


class TestPRMOrchestrationHook:
    """Test orchestration hook for PRM scoring."""

    @pytest.mark.asyncio
    async def test_scores_step_on_complete(self):
        """Should score step when it completes."""
        mock_scorer = AsyncMock(spec=PRMScorer)
        mock_scorer.score_step = AsyncMock(return_value=PRMScore(
            "step-1", StepQuality.CORRECT, 0.9, 0.85, "Good step"
        ))

        hook = PRMOrchestrationHook(mock_scorer)

        step = StepDef(step_id="step-1", name="Test", prompt_template="Test")
        result = StageResult(
            step_id="step-1",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
            outputs={"result": "Output"},
        )

        score = await hook.on_step_complete(step, result, {"goal": "Test"})

        assert score.step_id == "step-1"
        assert score.quality == StepQuality.CORRECT
        assert score.score == 0.9

    @pytest.mark.asyncio
    async def test_stores_score_in_result_metrics(self):
        """Should add PRM score to result metrics."""
        mock_scorer = AsyncMock(spec=PRMScorer)
        mock_scorer.score_step = AsyncMock(return_value=PRMScore(
            "step-1", StepQuality.CORRECT, 0.85, 0.9, "Good"
        ))

        hook = PRMOrchestrationHook(mock_scorer, store_scores=True)

        step = StepDef(step_id="step-1", name="Test", prompt_template="Test")
        result = StageResult(
            step_id="step-1",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
        )

        await hook.on_step_complete(step, result, {})

        assert result.metrics["prm_score"] == 0.85
        assert result.metrics["prm_quality"] == "correct"
        assert result.metrics["prm_confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_adds_issues_to_warnings(self):
        """Should add PRM issues to result warnings."""
        mock_scorer = AsyncMock(spec=PRMScorer)
        mock_scorer.score_step = AsyncMock(return_value=PRMScore(
            "step-1",
            StepQuality.PARTIALLY_CORRECT,
            0.6,
            0.8,
            "Has issues",
            issues=["Missing validation", "Incomplete logic"],
        ))

        hook = PRMOrchestrationHook(mock_scorer, store_scores=True)

        step = StepDef(step_id="step-1", name="Test", prompt_template="Test")
        result = StageResult(
            step_id="step-1",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
        )

        await hook.on_step_complete(step, result, {})

        assert "Missing validation" in result.warnings
        assert "Incomplete logic" in result.warnings

    @pytest.mark.asyncio
    async def test_get_path_score(self):
        """Should compute path score from stored step scores."""
        mock_scorer = AsyncMock(spec=PRMScorer)
        mock_scorer.score_step = AsyncMock(side_effect=[
            PRMScore("step-1", StepQuality.CORRECT, 0.9, 0.85, "Good"),
            PRMScore("step-2", StepQuality.CORRECT, 0.8, 0.9, "Good"),
            PRMScore("step-3", StepQuality.PARTIALLY_CORRECT, 0.6, 0.7, "OK"),
        ])

        hook = PRMOrchestrationHook(mock_scorer, store_scores=True)

        # Score three steps
        for i in range(1, 4):
            step = StepDef(step_id=f"step-{i}", name="Test", prompt_template="Test")
            result = StageResult(
                step_id=f"step-{i}",
                head_id="mock-llm",
                status=StepStatus.COMMITTED,
            )
            await hook.on_step_complete(step, result, {})

        # Get path score
        path_score = hook.get_path_score(["step-1", "step-2", "step-3"])

        assert path_score is not None
        assert len(path_score.step_scores) == 3
        assert path_score.total_score == 0.6  # Min of 0.9, 0.8, 0.6


class TestPRMPathSelector:
    """Test path selection based on PRM scores."""

    @pytest.mark.asyncio
    async def test_selects_best_path(self):
        """Should select path with highest PRM score."""
        mock_scorer = AsyncMock(spec=PRMScorer)

        # Path 1: lower quality
        # Path 2: higher quality (should be selected)
        mock_scorer.score_path = AsyncMock(side_effect=[
            AsyncMock(total_score=0.6, num_correct=2),  # Path 1
            AsyncMock(total_score=0.9, num_correct=3),  # Path 2
        ])

        selector = PRMPathSelector(mock_scorer)

        path1 = [
            StageResult(step_id="s1", head_id="mock-llm", status=StepStatus.COMMITTED),
            StageResult(step_id="s2", head_id="mock-llm", status=StepStatus.COMMITTED),
        ]

        path2 = [
            StageResult(step_id="s3", head_id="mock-llm", status=StepStatus.COMMITTED),
            StageResult(step_id="s4", head_id="mock-llm", status=StepStatus.COMMITTED),
        ]

        best_path, best_score = await selector.select_best_path(
            [path1, path2],
            {"goal": "Test"},
        )

        # Should select path2 (higher score)
        assert best_path == path2
        assert best_score.total_score == 0.9

    @pytest.mark.asyncio
    async def test_handles_single_path(self):
        """Should handle single path gracefully."""
        mock_scorer = AsyncMock(spec=PRMScorer)
        mock_scorer.score_path = AsyncMock(return_value=AsyncMock(
            total_score=0.8,
            num_correct=2
        ))

        selector = PRMPathSelector(mock_scorer)

        path = [StageResult(step_id="s1", head_id="mock-llm", status=StepStatus.COMMITTED)]

        best_path, best_score = await selector.select_best_path([path], {})

        assert best_path == path


class TestRubricCreation:
    """Test rubric creation helpers."""

    def test_creates_code_rubric(self):
        """Should create rubric for code steps."""
        rubric = create_rubric_for_code_steps()

        assert len(rubric) == 3
        assert rubric[0][0] == "Produces code"
        assert rubric[1][0] == "No syntax errors"
        assert rubric[2][0] == "Reasonable length"

        # Check weights sum to 1.0
        total_weight = sum(weight for _, _, weight in rubric)
        assert total_weight == pytest.approx(1.0)

    def test_creates_math_rubric(self):
        """Should create rubric for math steps."""
        rubric = create_rubric_for_math_steps()

        assert len(rubric) == 3
        assert rubric[0][0] == "Has numerical result"
        assert rubric[1][0] == "Shows work"
        assert rubric[2][0] == "No errors"

        # Check weights sum to 1.0
        total_weight = sum(weight for _, _, weight in rubric)
        assert total_weight == pytest.approx(1.0)

    def test_code_rubric_checks_work(self):
        """Should verify code rubric checks function."""
        rubric = create_rubric_for_code_steps()

        # Create result with code
        code_result = StageResult(
            step_id="test",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
            outputs={"result": "def hello():\n    print('hello')"},
        )

        # First check: has code
        has_code_check = rubric[0][1]
        assert has_code_check(code_result, {}) is True

        # Create result without code
        no_code_result = StageResult(
            step_id="test",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
            outputs={"result": "Just plain text"},
        )

        assert has_code_check(no_code_result, {}) is False

    def test_math_rubric_checks_work(self):
        """Should verify math rubric checks function."""
        rubric = create_rubric_for_math_steps()

        # Create result with math
        math_result = StageResult(
            step_id="test",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
            outputs={"result": "x = 5, so 2x + 3 = 13"},
        )

        # First check: has numerical result
        has_number_check = rubric[0][1]
        assert has_number_check(math_result, {}) is True

        # Second check: shows work
        shows_work_check = rubric[1][1]
        assert shows_work_check(math_result, {}) is True
