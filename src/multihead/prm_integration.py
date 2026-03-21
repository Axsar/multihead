"""Integration of Process Reward Models with MultiHead orchestrator.

Enables step-level quality scoring in recipes:
- Score each step as it executes
- Use PRM scores to guide ToT exploration
- Identify reasoning errors early
- Provide detailed feedback for debugging

PRM scores can be used to:
1. Detect errors early (fail fast)
2. Rank alternative reasoning paths (best path selection)
3. Guide reflection (which steps need refinement)
4. Train better models (credit assignment)
"""

from __future__ import annotations

import logging
from typing import Any

from multihead.models import StageResult, StepDef
from multihead.process_reward_models import PathScore, PRMScore, PRMScorer

logger = logging.getLogger(__name__)


class PRMOrchestrationHook:
    """Hook for scoring steps during orchestration.

    Integrates with orchestrator to score each step as it completes.
    """

    def __init__(self, scorer: PRMScorer, *, store_scores: bool = True):
        """Initialize PRM orchestration hook.

        Args:
            scorer: PRM scorer to use
            store_scores: Whether to store scores in step results
        """
        self.scorer = scorer
        self.store_scores = store_scores
        self.scores: dict[str, PRMScore] = {}  # step_id -> score

    async def on_step_complete(
        self,
        step: StepDef,
        result: StageResult,
        context: dict[str, Any],
    ) -> PRMScore:
        """Called when a step completes execution.

        Args:
            step: Step definition
            result: Step execution result
            context: Execution context with previous steps

        Returns:
            PRMScore for the completed step
        """
        logger.debug("Scoring step %s with PRM", step.step_id)

        score = await self.scorer.score_step(result, context)

        if self.store_scores:
            self.scores[step.step_id] = score

            # Add score to result metrics
            if result.metrics is None:
                result.metrics = {}

            result.metrics["prm_score"] = score.score
            result.metrics["prm_quality"] = score.quality.value
            result.metrics["prm_confidence"] = score.confidence

            if score.issues:
                result.warnings.extend(score.issues)

        logger.info(
            "PRM score for %s: %.2f (%s) - %s",
            step.step_id,
            score.score,
            score.quality.value,
            score.feedback,
        )

        return score

    def get_path_score(
        self,
        step_ids: list[str],
        *,
        aggregation: str = "min",
    ) -> PathScore | None:
        """Get aggregate path score for a sequence of steps.

        Args:
            step_ids: List of step IDs in execution order
            aggregation: How to combine scores (min/avg/product)

        Returns:
            PathScore or None if scores not available
        """
        step_scores = []
        for step_id in step_ids:
            if step_id in self.scores:
                step_scores.append(self.scores[step_id])

        if not step_scores:
            return None

        return PathScore.from_step_scores(
            path_id="-".join(step_ids),
            step_scores=step_scores,
            aggregation=aggregation,
        )


def should_use_prm(step: StepDef) -> bool:
    """Check if a step should use PRM scoring.

    Args:
        step: Step definition

    Returns:
        True if step should be PRM-scored
    """
    # Check for explicit prm flag
    if hasattr(step, "extra") and step.extra:
        return step.extra.get("use_prm", False)

    return False


def get_prm_config(step: StepDef) -> dict[str, Any]:
    """Extract PRM configuration from step metadata.

    Args:
        step: Step definition

    Returns:
        Dict with PRM config (scorer_type, threshold, etc.)
    """
    config = {
        "scorer_type": "llm",  # llm, rubric, or composite
        "threshold": 0.6,  # Minimum acceptable score
        "fail_on_low_score": False,  # Whether to fail step if score too low
    }

    if not hasattr(step, "extra") or not step.extra:
        return config

    # Extract from extra
    if "prm_scorer" in step.extra:
        config["scorer_type"] = step.extra["prm_scorer"]

    if "prm_threshold" in step.extra:
        config["threshold"] = float(step.extra["prm_threshold"])

    if "prm_fail_on_low" in step.extra:
        config["fail_on_low_score"] = bool(step.extra["prm_fail_on_low"])

    return config


def integrate_prm_with_tot(
    prm_scorer: PRMScorer,
    tot_state_evaluator,
) -> Any:
    """Integrate PRM scoring with Tree-of-Thoughts state evaluation.

    Creates a state evaluator that uses PRM scores to guide ToT exploration.

    Args:
        prm_scorer: PRM scorer to use
        tot_state_evaluator: Existing ToT state evaluator to enhance

    Returns:
        Enhanced state evaluator that considers PRM scores
    """
    # This would create a wrapper around the ToT state evaluator
    # that incorporates PRM scores into path evaluation
    logger.info("Integrating PRM with ToT state evaluation")

    class PRMEnhancedStateEvaluator:
        """State evaluator enhanced with PRM scoring."""

        def __init__(self, base_evaluator, prm_scorer: PRMScorer):
            self.base = base_evaluator
            self.prm = prm_scorer

        async def evaluate(self, state: Any, context: dict[str, Any]) -> float:
            """Evaluate state using both base evaluator and PRM."""
            # Get base evaluation
            base_score = await self.base.evaluate(state, context)

            # Get PRM evaluation if state is a StageResult
            if isinstance(state, StageResult):
                prm_score = await self.prm.score_step(state, context)

                # Combine scores (weighted average)
                combined = 0.5 * base_score + 0.5 * prm_score.score

                logger.debug(
                    "Combined ToT evaluation: base=%.2f, prm=%.2f, combined=%.2f",
                    base_score,
                    prm_score.score,
                    combined,
                )

                return combined

            return base_score

    return PRMEnhancedStateEvaluator(tot_state_evaluator, prm_scorer)


class PRMPathSelector:
    """Selects best reasoning path based on PRM scores.

    Used to choose between multiple alternative solution paths.
    """

    def __init__(self, scorer: PRMScorer):
        """Initialize path selector.

        Args:
            scorer: PRM scorer to use
        """
        self.scorer = scorer

    async def select_best_path(
        self,
        paths: list[list[StageResult]],
        context: dict[str, Any],
        *,
        aggregation: str = "min",
    ) -> tuple[list[StageResult], PathScore]:
        """Select the best reasoning path from alternatives.

        Args:
            paths: List of paths (each path is a list of StageResults)
            context: Execution context
            aggregation: How to score paths (min/avg/product)

        Returns:
            (best_path, path_score) tuple
        """
        logger.info("Selecting best path from %d alternatives", len(paths))

        if not paths:
            raise ValueError("No paths to select from")

        if len(paths) == 1:
            path = paths[0]
            score = await self.scorer.score_path(path, context, aggregation=aggregation)
            return path, score

        # Score all paths
        path_scores = []
        for i, path in enumerate(paths):
            path_ctx = {**context, "path_id": f"path-{i}"}
            score = await self.scorer.score_path(path, path_ctx, aggregation=aggregation)
            path_scores.append((path, score))

            logger.info(
                "Path %d score: total=%.2f, min=%.2f, avg=%.2f, errors=%d",
                i,
                score.total_score,
                score.min_score,
                score.avg_score,
                score.num_incorrect,
            )

        # Select path with highest score
        best_path, best_score = max(path_scores, key=lambda x: x[1].total_score)

        logger.info(
            "Selected best path: score=%.2f, correct=%d/%d",
            best_score.total_score,
            best_score.num_correct,
            len(best_score.step_scores),
        )

        return best_path, best_score


def create_rubric_for_code_steps() -> list[tuple[str, Any, float]]:
    """Create a rubric for scoring code generation steps.

    Returns:
        Rubric suitable for RubricPRMScorer
    """
    def has_code_output(result: StageResult, ctx: dict[str, Any]) -> bool:
        """Check if step produced code output."""
        if not result.outputs:
            return False
        output = str(result.outputs.get("result", ""))
        # Simple heuristic: contains code-like patterns
        return any(keyword in output for keyword in ["def ", "class ", "import ", "function ", "const ", "let "])

    def no_syntax_errors(result: StageResult, ctx: dict[str, Any]) -> bool:
        """Check for syntax errors."""
        if result.error:
            return "syntax" not in result.error.lower()
        return True

    def has_reasonable_length(result: StageResult, ctx: dict[str, Any]) -> bool:
        """Check if code is reasonable length."""
        if not result.outputs:
            return False
        output = str(result.outputs.get("result", ""))
        return 20 <= len(output) <= 5000

    return [
        ("Produces code", has_code_output, 0.4),
        ("No syntax errors", no_syntax_errors, 0.4),
        ("Reasonable length", has_reasonable_length, 0.2),
    ]


def create_rubric_for_math_steps() -> list[tuple[str, Any, float]]:
    """Create a rubric for scoring mathematical reasoning steps.

    Returns:
        Rubric suitable for RubricPRMScorer
    """
    def has_numerical_result(result: StageResult, ctx: dict[str, Any]) -> bool:
        """Check if step produces a number."""
        if not result.outputs:
            return False
        output = str(result.outputs.get("result", ""))
        return any(c.isdigit() for c in output)

    def shows_work(result: StageResult, ctx: dict[str, Any]) -> bool:
        """Check if step shows reasoning."""
        if not result.outputs:
            return False
        output = str(result.outputs.get("result", ""))
        # Look for mathematical operations
        return any(op in output for op in ["=", "+", "-", "*", "/", "^"])

    def no_errors(result: StageResult, ctx: dict[str, Any]) -> bool:
        """Check for errors."""
        return not result.error

    return [
        ("Has numerical result", has_numerical_result, 0.3),
        ("Shows work", shows_work, 0.3),
        ("No errors", no_errors, 0.4),
    ]
