"""Tests for Tree-of-Thoughts integration with orchestrator."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from multihead.models import StageResult, StepDef, StepStatus
from multihead.tree_of_thoughts import SearchStrategy
from multihead.tot_integration import (
    StepStateEvaluator,
    execute_step_with_tot,
    get_tot_config,
    should_use_tot,
)


class TestShouldUseTot:
    """Test ToT activation logic."""

    def test_returns_false_by_default(self):
        """Should return False for steps without ToT metadata."""
        step = StepDef(name="test", prompt_template="Test")
        assert should_use_tot(step) is False

    def test_returns_true_if_use_tot_in_extra(self):
        """Should return True if extra.use_tot is set."""
        step = StepDef(
            name="test",
            prompt_template="Test",
            extra={"use_tot": True},
        )
        assert should_use_tot(step) is True

    def test_returns_false_if_use_tot_is_false(self):
        """Should return False if explicitly disabled."""
        step = StepDef(
            name="test",
            prompt_template="Test",
            extra={"use_tot": False},
        )
        assert should_use_tot(step) is False


class TestGetTotConfig:
    """Test ToT configuration extraction."""

    def test_returns_defaults(self):
        """Should return default config if no metadata."""
        step = StepDef(name="test", prompt_template="Test")
        config = get_tot_config(step)

        assert config["strategy"] == SearchStrategy.BFS
        assert config["num_alternatives"] == 3
        assert config["max_depth"] == 2

    def test_extracts_strategy_from_extra(self):
        """Should extract strategy from extra metadata."""
        step = StepDef(
            name="test",
            prompt_template="Test",
            extra={"tot_strategy": "dfs"},
        )
        config = get_tot_config(step)

        assert config["strategy"] == SearchStrategy.DFS

    def test_extracts_num_alternatives(self):
        """Should extract num_alternatives from extra."""
        step = StepDef(
            name="test",
            prompt_template="Test",
            extra={"tot_alternatives": 5},
        )
        config = get_tot_config(step)

        assert config["num_alternatives"] == 5

    def test_extracts_max_depth(self):
        """Should extract max_depth from extra."""
        step = StepDef(
            name="test",
            prompt_template="Test",
            extra={"tot_max_depth": 4},
        )
        config = get_tot_config(step)

        assert config["max_depth"] == 4

    def test_extracts_all_config(self):
        """Should extract full configuration."""
        step = StepDef(
            name="test",
            prompt_template="Test",
            extra={
                "tot_strategy": "beam",
                "tot_alternatives": 4,
                "tot_max_depth": 3,
            },
        )
        config = get_tot_config(step)

        assert config["strategy"] == SearchStrategy.BEAM
        assert config["num_alternatives"] == 4
        assert config["max_depth"] == 3

    def test_handles_invalid_strategy(self):
        """Should fall back to BFS for invalid strategy."""
        step = StepDef(
            name="test",
            prompt_template="Test",
            extra={"tot_strategy": "invalid"},
        )
        config = get_tot_config(step)

        assert config["strategy"] == SearchStrategy.BFS


class TestStepStateEvaluator:
    """Test state evaluation for step results."""

    @pytest.mark.asyncio
    async def test_evaluates_confidence(self):
        """Should use confidence from metrics if available."""
        evaluator = StepStateEvaluator()

        result = StageResult(
            step_id="test",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
            outputs={"result": "Test output"},
            metrics={"confidence": 0.85},
        )

        score = await evaluator.evaluate(result, {})

        assert score == 0.85

    @pytest.mark.asyncio
    async def test_low_score_for_errors(self):
        """Should return low score for error results."""
        evaluator = StepStateEvaluator()

        result = StageResult(
            step_id="test",
            head_id="mock-llm",
            status=StepStatus.FAILED,
            error="Something failed",
        )

        score = await evaluator.evaluate(result, {})

        assert score == 0.1

    @pytest.mark.asyncio
    async def test_scores_by_output_length(self):
        """Should score based on output length when no confidence."""
        evaluator = StepStateEvaluator()

        # Short output
        result1 = StageResult(
            step_id="test",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
            outputs={"result": "Short"},
        )
        score1 = await evaluator.evaluate(result1, {})

        # Long output
        long_output = "x" * 500
        result2 = StageResult(
            step_id="test",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
            outputs={"result": long_output},
        )
        score2 = await evaluator.evaluate(result2, {})

        # Longer output should score higher
        assert score2 > score1

    @pytest.mark.asyncio
    async def test_defaults_for_invalid_state(self):
        """Should return low score for non-StageResult objects."""
        evaluator = StepStateEvaluator()

        score = await evaluator.evaluate("not a result", {})

        assert score == 0.3

    @pytest.mark.asyncio
    async def test_defaults_for_empty_result(self):
        """Should return default score for empty result."""
        evaluator = StepStateEvaluator()

        result = StageResult(
            step_id="test",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
        )

        score = await evaluator.evaluate(result, {})

        assert score == 0.5


class TestExecuteStepWithTot:
    """Test step execution with ToT."""

    @pytest.mark.asyncio
    async def test_explores_alternatives(self):
        """Should explore multiple alternatives and return best."""
        best_result = StageResult(
            step_id="tot-step",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
            outputs={"result": "Best output from exploration"},
            metrics={"confidence": 0.95},
        )

        # Mock ToTEngine.solve to return a controlled result
        mock_solve_return = {
            "best_path": [],
            "best_state": best_result,
            "best_score": 0.95,
            "explored_count": 3,
            "all_nodes": [],
        }

        step = StepDef(
            step_id="tot-step",
            name="Explore alternatives",
            prompt_template="Solve the problem",
            extra={"use_tot": True, "tot_alternatives": 3},
        )

        execute_func = AsyncMock(return_value=best_result)

        with patch("multihead.tot_integration.ToTEngine") as MockEngine:
            mock_engine_instance = MagicMock()
            mock_engine_instance.solve = AsyncMock(return_value=mock_solve_return)
            MockEngine.return_value = mock_engine_instance

            result = await execute_step_with_tot(
                execute_func, step, {"goal": "test goal"}, num_alternatives=3
            )

        assert result == best_result
        assert result.outputs["result"] == "Best output from exploration"
        mock_engine_instance.solve.assert_called_once()

    @pytest.mark.asyncio
    async def test_uses_configured_strategy(self):
        """Should use the configured search strategy."""
        fallback_result = StageResult(
            step_id="tot-step",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
            outputs={"result": "DFS result"},
        )

        # Return None best_state to exercise fallback path too
        mock_solve_return = {
            "best_path": [],
            "best_state": None,
            "best_score": 0.0,
            "explored_count": 2,
            "all_nodes": [],
        }

        step = StepDef(
            step_id="tot-step",
            name="DFS explore",
            prompt_template="Explore with DFS",
            extra={"use_tot": True, "tot_strategy": "dfs"},
        )

        execute_func = AsyncMock(return_value=fallback_result)

        with patch("multihead.tot_integration.ToTEngine") as MockEngine:
            mock_engine_instance = MagicMock()
            mock_engine_instance.solve = AsyncMock(return_value=mock_solve_return)
            MockEngine.return_value = mock_engine_instance

            result = await execute_step_with_tot(
                execute_func, step, {}, strategy=SearchStrategy.DFS
            )

        # Verify DFS strategy was passed to ToTEngine constructor
        MockEngine.assert_called_once()
        call_kwargs = MockEngine.call_args
        assert (
            call_kwargs.kwargs.get("strategy")
            or call_kwargs[1].get("strategy") == SearchStrategy.DFS
        )

        # Since best_state was None, should fallback to execute_func
        assert result == fallback_result
        execute_func.assert_called_once_with(step)


class TestToTRecipeExample:
    """Test example recipe with ToT configuration."""

    def test_recipe_with_tot_step(self):
        """Should create a valid step with ToT metadata."""
        step = StepDef(
            step_id="tot-step",
            name="Creative problem solving",
            prompt_template="Solve this problem: {problem}",
            extra={
                "use_tot": True,
                "tot_strategy": "beam",
                "tot_alternatives": 4,
                "tot_max_depth": 3,
            },
        )

        assert should_use_tot(step) is True

        config = get_tot_config(step)
        assert config["strategy"] == SearchStrategy.BEAM
        assert config["num_alternatives"] == 4
        assert config["max_depth"] == 3

    def test_mixed_recipe_with_tot_and_regular_steps(self):
        """Should support mixing ToT and regular steps."""
        # Regular step
        step1 = StepDef(
            step_id="regular",
            name="Regular step",
            prompt_template="Do something",
        )

        # ToT step
        step2 = StepDef(
            step_id="tot-step",
            name="ToT step",
            prompt_template="Explore alternatives",
            extra={"use_tot": True},
        )

        # Another regular step
        step3 = StepDef(
            step_id="regular2",
            name="Regular step 2",
            prompt_template="Do something else",
        )

        assert should_use_tot(step1) is False
        assert should_use_tot(step2) is True
        assert should_use_tot(step3) is False
