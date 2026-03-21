"""PRM scorer implementations.

Provides concrete scorer classes for evaluating reasoning step quality:
- PRMScorer: Abstract base class
- LLMPRMScorer: Uses an LLM to evaluate steps
- RubricPRMScorer: Uses a predefined rubric/checklist
- CompositePRMScorer: Combines multiple scorers
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

from multihead.models import StageResult
from multihead.process_reward_models.models import PRMScore, PathScore, StepQuality

logger = logging.getLogger(__name__)


class PRMScorer(ABC):
    """Base class for process reward model scorers."""

    @abstractmethod
    async def score_step(
        self,
        step_result: StageResult,
        context: dict[str, Any],
    ) -> PRMScore:
        """Score the quality of a single reasoning step.

        Args:
            step_result: Result from executing the step
            context: Additional context (previous steps, goal, etc.)

        Returns:
            PRMScore with quality assessment
        """
        pass

    async def score_path(
        self,
        step_results: list[StageResult],
        context: dict[str, Any],
        *,
        aggregation: str = "min",
    ) -> PathScore:
        """Score an entire reasoning path.

        Args:
            step_results: Results from all steps in path
            context: Additional context
            aggregation: How to combine scores (min/avg/product)

        Returns:
            PathScore with aggregate metrics
        """
        step_scores = []
        for result in step_results:
            score = await self.score_step(result, context)
            step_scores.append(score)

        path_id = context.get("path_id", "path")
        return PathScore.from_step_scores(path_id, step_scores, aggregation=aggregation)


class LLMPRMScorer(PRMScorer):
    """Uses an LLM to score step quality.

    Prompts the LLM to evaluate whether each step is logically sound,
    contributes to the solution, and follows best practices.
    """

    def __init__(self, generate_func: Callable[[str], Any]):
        """Initialize with LLM generate function.

        Args:
            generate_func: Async function that takes a prompt and returns text
        """
        self.generate = generate_func

    async def score_step(
        self,
        step_result: StageResult,
        context: dict[str, Any],
    ) -> PRMScore:
        """Score step using LLM evaluation.

        Args:
            step_result: Step execution result
            context: Context with goal, previous steps

        Returns:
            PRMScore from LLM evaluation
        """
        goal = context.get("goal", "solve the problem")
        previous_steps = context.get("previous_steps", [])

        # Build context of previous steps
        context_text = ""
        if previous_steps:
            context_text = "Previous steps:\n"
            for i, prev in enumerate(previous_steps, 1):
                prev_output = prev.outputs.get("result", str(prev.outputs)) if prev.outputs else "No output"
                context_text += f"{i}. {prev_output}\n"

        # Get current step output
        current_output = step_result.outputs.get("result", str(step_result.outputs)) if step_result.outputs else "No output"

        prompt = f"""Evaluate the quality of this reasoning step.

Goal: {goal}

{context_text}
Current step output: {current_output}

Evaluate this step on:
1. Logical soundness - Is the reasoning correct?
2. Progress toward goal - Does it move us closer to the solution?
3. Clarity - Is the step well-explained?

Respond in this format:
Quality: [correct/partially_correct/incorrect/neutral]
Score: [number from 0-10]
Confidence: [number from 0-10]
Feedback: [One sentence explaining your assessment]
Issues: [List any problems, or "none"]
Strengths: [List what was done well, or "none"]"""

        try:
            response = await self.generate(prompt)
            return self._parse_llm_response(response, step_result.step_id)

        except Exception as e:
            logger.error("LLM PRM scoring failed for step %s: %s", step_result.step_id, e)
            # Default to neutral score on error
            return PRMScore(
                step_id=step_result.step_id,
                quality=StepQuality.UNKNOWN,
                score=0.5,
                confidence=0.0,
                feedback=f"Scoring failed: {e}",
            )

    def _parse_llm_response(self, response: str, step_id: str) -> PRMScore:
        """Parse LLM response into PRMScore.

        Args:
            response: LLM response text
            step_id: Step identifier

        Returns:
            Parsed PRMScore
        """
        lines = response.strip().split("\n")

        # Defaults
        quality = StepQuality.NEUTRAL
        score = 0.5
        confidence = 0.5
        feedback = ""
        issues = []
        strengths = []

        for line in lines:
            line = line.strip()
            if line.lower().startswith("quality:"):
                quality_str = line.split(":", 1)[1].strip().lower()
                try:
                    quality = StepQuality(quality_str)
                except ValueError:
                    pass

            elif line.lower().startswith("score:"):
                try:
                    score_str = line.split(":", 1)[1].strip()
                    # Extract first number
                    for word in score_str.split():
                        try:
                            score_num = float(word)
                            score = max(0.0, min(10.0, score_num)) / 10.0  # Normalize to 0-1
                            break
                        except ValueError:
                            continue
                except Exception:
                    pass

            elif line.lower().startswith("confidence:"):
                try:
                    conf_str = line.split(":", 1)[1].strip()
                    for word in conf_str.split():
                        try:
                            conf_num = float(word)
                            confidence = max(0.0, min(10.0, conf_num)) / 10.0
                            break
                        except ValueError:
                            continue
                except Exception:
                    pass

            elif line.lower().startswith("feedback:"):
                feedback = line.split(":", 1)[1].strip()

            elif line.lower().startswith("issues:"):
                issues_text = line.split(":", 1)[1].strip()
                if issues_text.lower() not in ["none", "n/a"]:
                    issues = [issues_text]

            elif line.lower().startswith("strengths:"):
                strengths_text = line.split(":", 1)[1].strip()
                if strengths_text.lower() not in ["none", "n/a"]:
                    strengths = [strengths_text]

        return PRMScore(
            step_id=step_id,
            quality=quality,
            score=score,
            confidence=confidence,
            feedback=feedback if feedback else f"Step assessed as {quality.value}",
            issues=issues,
            strengths=strengths,
        )


class RubricPRMScorer(PRMScorer):
    """Scores steps using a predefined rubric/checklist.

    Useful for domain-specific evaluation where you know what
    makes a good step (e.g., code must compile, math must be valid).
    """

    def __init__(
        self,
        rubric: list[tuple[str, Callable[[StageResult, dict[str, Any]], bool], float]],
    ):
        """Initialize with scoring rubric.

        Args:
            rubric: List of (criterion_name, check_function, weight) tuples
                   check_function takes (step_result, context) and returns bool
                   weight is 0-1 importance of this criterion
        """
        self.rubric = rubric

    async def score_step(
        self,
        step_result: StageResult,
        context: dict[str, Any],
    ) -> PRMScore:
        """Score step using rubric criteria.

        Args:
            step_result: Step result
            context: Context

        Returns:
            PRMScore based on rubric evaluation
        """
        total_weight = sum(weight for _, _, weight in self.rubric)
        weighted_score = 0.0
        passed_criteria = []
        failed_criteria = []

        for criterion_name, check_func, weight in self.rubric:
            try:
                passed = check_func(step_result, context)
                if passed:
                    weighted_score += weight
                    passed_criteria.append(criterion_name)
                else:
                    failed_criteria.append(criterion_name)
            except Exception as e:
                logger.warning("Rubric check '%s' failed: %s", criterion_name, e)
                failed_criteria.append(f"{criterion_name} (error)")

        # Normalize score
        score = weighted_score / total_weight if total_weight > 0 else 0.5

        # Determine quality
        if score >= 0.9:
            quality = StepQuality.CORRECT
        elif score >= 0.6:
            quality = StepQuality.PARTIALLY_CORRECT
        elif score >= 0.3:
            quality = StepQuality.NEUTRAL
        else:
            quality = StepQuality.INCORRECT

        feedback = f"Passed {len(passed_criteria)}/{len(self.rubric)} criteria"

        return PRMScore(
            step_id=step_result.step_id,
            quality=quality,
            score=score,
            confidence=1.0,  # Rubric-based scoring is deterministic
            feedback=feedback,
            issues=failed_criteria,
            strengths=passed_criteria,
        )


class CompositePRMScorer(PRMScorer):
    """Combines multiple PRM scorers.

    Useful for getting diverse perspectives on step quality
    (e.g., LLM evaluation + rubric checks).
    """

    def __init__(
        self,
        scorers: list[tuple[PRMScorer, float]],
        *,
        aggregation: str = "weighted_avg",
    ):
        """Initialize with multiple scorers.

        Args:
            scorers: List of (scorer, weight) tuples
            aggregation: How to combine scores (weighted_avg, min, max)
        """
        self.scorers = scorers
        self.aggregation = aggregation

    async def score_step(
        self,
        step_result: StageResult,
        context: dict[str, Any],
    ) -> PRMScore:
        """Score step using all scorers and combine results.

        Args:
            step_result: Step result
            context: Context

        Returns:
            Combined PRMScore
        """
        scores = []
        for scorer, weight in self.scorers:
            score = await scorer.score_step(step_result, context)
            scores.append((score, weight))

        if not scores:
            return PRMScore(
                step_id=step_result.step_id,
                quality=StepQuality.UNKNOWN,
                score=0.5,
                confidence=0.0,
                feedback="No scorers available",
            )

        # Aggregate scores
        total_weight = sum(w for _, w in scores)

        if self.aggregation == "weighted_avg":
            combined_score = sum(s.score * w for s, w in scores) / total_weight
        elif self.aggregation == "min":
            combined_score = min(s.score for s, _ in scores)
        elif self.aggregation == "max":
            combined_score = max(s.score for s, _ in scores)
        else:
            combined_score = sum(s.score * w for s, w in scores) / total_weight

        # Determine quality from combined score
        if combined_score >= 0.9:
            quality = StepQuality.CORRECT
        elif combined_score >= 0.6:
            quality = StepQuality.PARTIALLY_CORRECT
        elif combined_score >= 0.3:
            quality = StepQuality.NEUTRAL
        else:
            quality = StepQuality.INCORRECT

        # Aggregate feedback
        all_issues = []
        all_strengths = []
        for score, _ in scores:
            all_issues.extend(score.issues)
            all_strengths.extend(score.strengths)

        return PRMScore(
            step_id=step_result.step_id,
            quality=quality,
            score=combined_score,
            confidence=sum(s.confidence * w for s, w in scores) / total_weight,
            feedback=f"Combined score from {len(scores)} scorers",
            issues=list(set(all_issues)),  # Deduplicate
            strengths=list(set(all_strengths)),
            metadata={"individual_scores": [s.score for s, _ in scores]},
        )
