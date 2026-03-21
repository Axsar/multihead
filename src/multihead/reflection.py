"""Reflection engine for self-correcting step execution.

Implements the Reflexion pattern (Shinn et al., 2023):
Actor → Evaluator → Self-Reflection → Memory → Refine → Re-attempt

This enables pipelines to learn from mistakes and iteratively improve outputs.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from multihead.models import StageResult, StepDef, StepStatus

logger = logging.getLogger(__name__)


@dataclass
class ReflectionResult:
    """Result of analyzing a failed or low-quality step execution.

    Attributes:
        step_id: Step that was reflected on
        attempt_number: Which attempt this reflection is for (1, 2, 3...)
        quality_score: Evaluator's assessment (0.0-1.0, higher = better)
        passed: Whether output meets quality threshold
        reflection_text: Verbal feedback for next attempt
        suggested_changes: Specific modifications to try
        should_retry: Whether to attempt again
        metadata: Additional evaluator metadata
    """
    step_id: str
    attempt_number: int
    quality_score: float
    passed: bool
    reflection_text: str
    suggested_changes: dict[str, Any]
    should_retry: bool
    metadata: dict[str, Any]
    created_at: datetime


class Evaluator(ABC):
    """Base class for step output evaluators.

    Evaluators score output quality and determine if reflection is needed.
    """

    @abstractmethod
    async def evaluate(
        self,
        step: StepDef,
        result: StageResult,
        context: dict[str, Any],
    ) -> ReflectionResult:
        """Evaluate step output quality and generate reflection.

        Args:
            step: Step definition
            result: Execution result to evaluate
            context: Run context (step_results, inputs, etc.)

        Returns:
            ReflectionResult with quality score and feedback
        """
        pass


class ConfidenceEvaluator(Evaluator):
    """Evaluates outputs based on confidence scores or quality metrics.

    Useful for consensus results or model outputs with confidence.
    """

    def __init__(self, confidence_threshold: float = 0.7):
        """Initialize confidence evaluator.

        Args:
            confidence_threshold: Minimum confidence to pass (0.0-1.0)
        """
        self.threshold = confidence_threshold

    async def evaluate(
        self,
        step: StepDef,
        result: StageResult,
        context: dict[str, Any],
    ) -> ReflectionResult:
        """Evaluate based on confidence score in result.

        Looks for 'confidence' in result.outputs or result.metrics.
        """
        # Extract confidence score
        confidence = (
            result.outputs.get("confidence") or
            result.metrics.get("confidence") or
            1.0  # Default: assume high confidence if not specified
        )

        passed = confidence >= self.threshold

        if passed:
            reflection_text = f"Output confidence {confidence:.2f} meets threshold {self.threshold:.2f}. No refinement needed."
            suggested_changes = {}
            should_retry = False
        else:
            reflection_text = (
                f"Output confidence {confidence:.2f} below threshold {self.threshold:.2f}. "
                f"The output may be uncertain or incomplete. Consider:\n"
                f"1. Providing more context in the prompt\n"
                f"2. Breaking the task into smaller steps\n"
                f"3. Using a different solver or head"
            )
            suggested_changes = {
                "add_context": True,
                "increase_detail": True,
                "try_alternative_approach": True,
            }
            should_retry = True

        return ReflectionResult(
            step_id=step.step_id,
            attempt_number=1,  # Will be updated by ReflectionEngine
            quality_score=confidence,
            passed=passed,
            reflection_text=reflection_text,
            suggested_changes=suggested_changes,
            should_retry=should_retry,
            metadata={"confidence": confidence, "threshold": self.threshold},
            created_at=datetime.now(timezone.utc),
        )


class ErrorEvaluator(Evaluator):
    """Evaluates based on execution errors.

    Reflects on failure reasons and suggests fixes.
    """

    async def evaluate(
        self,
        step: StepDef,
        result: StageResult,
        context: dict[str, Any],
    ) -> ReflectionResult:
        """Evaluate based on execution status and errors."""
        if result.status == StepStatus.COMMITTED:
            # Success - no reflection needed
            return ReflectionResult(
                step_id=step.step_id,
                attempt_number=1,
                quality_score=1.0,
                passed=True,
                reflection_text="Step executed successfully without errors.",
                suggested_changes={},
                should_retry=False,
                metadata={},
                created_at=datetime.now(timezone.utc),
            )

        # Failed - analyze error
        error_msg = result.error or "Unknown error"

        # Classify error type
        is_validation_error = "validation" in error_msg.lower()
        is_timeout = "timeout" in error_msg.lower()
        is_format_error = "json" in error_msg.lower() or "schema" in error_msg.lower()

        reflection_text = f"Step failed with error: {error_msg}\n\n"
        suggested_changes = {}

        if is_validation_error:
            reflection_text += (
                "Validation failed. Suggestions:\n"
                "1. Review the validation criteria and ensure the prompt addresses them\n"
                "2. Add explicit instructions about required output format\n"
                "3. Provide examples of valid outputs"
            )
            suggested_changes = {
                "add_validation_context": True,
                "add_examples": True,
                "clarify_requirements": True,
            }
        elif is_timeout:
            reflection_text += (
                "Execution timed out. Suggestions:\n"
                "1. Simplify the prompt to reduce generation time\n"
                "2. Increase timeout threshold\n"
                "3. Break into smaller sub-steps"
            )
            suggested_changes = {
                "simplify_prompt": True,
                "decompose_further": True,
            }
        elif is_format_error:
            reflection_text += (
                "Output format error. Suggestions:\n"
                "1. Add explicit JSON schema to prompt\n"
                "2. Provide a template of expected output\n"
                "3. Use structured output mode if available"
            )
            suggested_changes = {
                "add_schema": True,
                "add_template": True,
                "use_structured_output": True,
            }
        else:
            reflection_text += (
                "General execution failure. Suggestions:\n"
                "1. Verify the input data is valid\n"
                "2. Check if the head/model is appropriate for this task\n"
                "3. Review the prompt for ambiguity or missing context"
            )
            suggested_changes = {
                "verify_inputs": True,
                "review_head_selection": True,
                "clarify_prompt": True,
            }

        return ReflectionResult(
            step_id=step.step_id,
            attempt_number=1,
            quality_score=0.0,
            passed=False,
            reflection_text=reflection_text,
            suggested_changes=suggested_changes,
            should_retry=True,
            metadata={"error": error_msg},
            created_at=datetime.now(timezone.utc),
        )


class CompositeEvaluator(Evaluator):
    """Combines multiple evaluators (all must pass).

    Useful for checking both confidence and error conditions.
    """

    def __init__(self, evaluators: list[Evaluator]):
        """Initialize composite evaluator.

        Args:
            evaluators: List of evaluators to run
        """
        self.evaluators = evaluators

    async def evaluate(
        self,
        step: StepDef,
        result: StageResult,
        context: dict[str, Any],
    ) -> ReflectionResult:
        """Run all evaluators and combine results."""
        results = []
        for evaluator in self.evaluators:
            reflection = await evaluator.evaluate(step, result, context)
            results.append(reflection)

        # Aggregate: pass only if all pass
        all_passed = all(r.passed for r in results)
        min_quality = min(r.quality_score for r in results)
        any_should_retry = any(r.should_retry for r in results)

        # Combine reflections
        reflection_texts = [r.reflection_text for r in results if not r.passed]
        combined_text = "\n\n".join(reflection_texts) if reflection_texts else "All evaluators passed."

        # Merge suggested changes
        merged_changes = {}
        for r in results:
            merged_changes.update(r.suggested_changes)

        return ReflectionResult(
            step_id=step.step_id,
            attempt_number=1,
            quality_score=min_quality,
            passed=all_passed,
            reflection_text=combined_text,
            suggested_changes=merged_changes,
            should_retry=any_should_retry,
            metadata={"evaluator_count": len(results)},
            created_at=datetime.now(timezone.utc),
        )


class ReflectionMemory:
    """Stores reflection history for a run.

    Memory enables learning across attempts within the same run.
    """

    def __init__(self):
        """Initialize empty memory."""
        self.reflections: dict[str, list[ReflectionResult]] = {}

    def add(self, reflection: ReflectionResult) -> None:
        """Store a reflection result.

        Args:
            reflection: Reflection to store
        """
        if reflection.step_id not in self.reflections:
            self.reflections[reflection.step_id] = []
        self.reflections[reflection.step_id].append(reflection)

    def get_history(self, step_id: str) -> list[ReflectionResult]:
        """Get all reflections for a step.

        Args:
            step_id: Step to get history for

        Returns:
            List of reflections in chronological order
        """
        return self.reflections.get(step_id, [])

    def get_attempt_count(self, step_id: str) -> int:
        """Get number of attempts for a step.

        Args:
            step_id: Step to count

        Returns:
            Number of attempts
        """
        return len(self.reflections.get(step_id, []))


class ReflectionEngine:
    """Orchestrates the Actor-Evaluator-Reflect-Memory cycle.

    Wraps step execution with reflection logic, enabling self-correction.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        *,
        max_attempts: int = 3,
        memory: ReflectionMemory | None = None,
    ):
        """Initialize reflection engine.

        Args:
            evaluator: Evaluator to use for quality assessment
            max_attempts: Maximum refinement attempts per step
            memory: Optional shared memory (created if not provided)
        """
        self.evaluator = evaluator
        self.max_attempts = max_attempts
        self.memory = memory or ReflectionMemory()

    def get_refinement_context(self, step_id: str) -> str:
        """Build context from previous reflections.

        Args:
            step_id: Step to get context for

        Returns:
            Context string for prompt augmentation
        """
        history = self.memory.get_history(step_id)
        if not history:
            return ""

        # Build summary of previous attempts
        context_parts = ["\n=== Previous Attempts and Reflections ===\n"]
        for i, reflection in enumerate(history, 1):
            context_parts.append(
                f"\nAttempt {i}:\n"
                f"Quality Score: {reflection.quality_score:.2f}\n"
                f"Reflection: {reflection.reflection_text}\n"
            )

        context_parts.append("\n=== Current Attempt ===\n")
        context_parts.append("Please incorporate the above feedback and avoid previous mistakes.\n")

        return "".join(context_parts)

    async def should_refine(
        self,
        step: StepDef,
        result: StageResult,
        context: dict[str, Any],
        attempt: int,
    ) -> ReflectionResult | None:
        """Check if step needs refinement.

        Args:
            step: Step definition
            result: Execution result
            context: Run context
            attempt: Current attempt number

        Returns:
            ReflectionResult if refinement needed, None if passed
        """
        reflection = await self.evaluator.evaluate(step, result, context)
        reflection.attempt_number = attempt

        self.memory.add(reflection)

        if reflection.passed:
            logger.info(
                "Step %s passed evaluation (quality=%.2f, attempt=%d)",
                step.step_id, reflection.quality_score, attempt
            )
            return None

        if attempt >= self.max_attempts:
            logger.warning(
                "Step %s failed after %d attempts (quality=%.2f)",
                step.step_id, self.max_attempts, reflection.quality_score
            )
            return None  # Give up after max attempts

        if not reflection.should_retry:
            logger.info(
                "Step %s failed but evaluator recommends not retrying",
                step.step_id
            )
            return None

        logger.info(
            "Step %s needs refinement (quality=%.2f, attempt=%d/%d)",
            step.step_id, reflection.quality_score, attempt, self.max_attempts
        )
        return reflection
