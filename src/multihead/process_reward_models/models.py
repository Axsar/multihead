"""Data models for Process Reward Models (PRM).

Defines the core data structures used for step-level quality scoring:
- StepQuality enum for categorical quality levels
- PRMScore for individual step assessments
- PathScore for aggregate path evaluations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class StepQuality(str, Enum):
    """Step quality levels for PRM scoring."""

    CORRECT = "correct"  # Step is logically sound and moves toward solution
    PARTIALLY_CORRECT = "partially_correct"  # Step has merit but issues
    INCORRECT = "incorrect"  # Step contains errors or is counterproductive
    NEUTRAL = "neutral"  # Step is valid but doesn't add much value
    UNKNOWN = "unknown"  # Cannot determine quality


@dataclass
class PRMScore:
    """Process reward model score for a single step.

    Contains both a numeric score and categorical quality assessment,
    plus detailed feedback for debugging and improvement.
    """

    step_id: str
    quality: StepQuality
    score: float  # 0.0 (bad) to 1.0 (perfect)
    confidence: float  # How confident is the scorer (0.0-1.0)
    feedback: str  # Human-readable explanation
    issues: list[str] = field(default_factory=list)  # Specific problems found
    strengths: list[str] = field(default_factory=list)  # What the step did well
    metadata: dict[str, Any] = field(default_factory=dict)
    scored_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def is_acceptable(self, threshold: float = 0.6) -> bool:
        """Check if step quality meets threshold.

        Args:
            threshold: Minimum score to be acceptable (default 0.6)

        Returns:
            True if step is acceptable quality
        """
        return self.score >= threshold and self.quality != StepQuality.INCORRECT


@dataclass
class PathScore:
    """Aggregate PRM score for an entire reasoning path.

    Combines scores from all steps to evaluate overall path quality.
    """

    path_id: str
    step_scores: list[PRMScore]
    total_score: float  # Aggregate score for entire path
    min_score: float  # Lowest step score (weakest link)
    avg_score: float  # Average step score
    num_correct: int
    num_incorrect: int
    num_partial: int
    first_error_at: str | None = None  # Step ID where first error occurred
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_step_scores(
        cls,
        path_id: str,
        step_scores: list[PRMScore],
        *,
        aggregation: str = "min",  # min, avg, or product
    ) -> PathScore:
        """Create path score from individual step scores.

        Args:
            path_id: Identifier for this path
            step_scores: List of PRM scores for each step
            aggregation: How to combine step scores (min/avg/product)

        Returns:
            PathScore with aggregated metrics
        """
        if not step_scores:
            return cls(
                path_id=path_id,
                step_scores=[],
                total_score=0.0,
                min_score=0.0,
                avg_score=0.0,
                num_correct=0,
                num_incorrect=0,
                num_partial=0,
            )

        scores = [s.score for s in step_scores]
        min_score = min(scores)
        avg_score = sum(scores) / len(scores)

        # Aggregate based on strategy
        if aggregation == "min":
            total_score = min_score  # Weakest link
        elif aggregation == "avg":
            total_score = avg_score  # Average quality
        elif aggregation == "product":
            # Product of probabilities (each step must succeed)
            total_score = 1.0
            for score in scores:
                total_score *= score
        else:
            total_score = avg_score

        # Count quality levels
        num_correct = sum(1 for s in step_scores if s.quality == StepQuality.CORRECT)
        num_incorrect = sum(1 for s in step_scores if s.quality == StepQuality.INCORRECT)
        num_partial = sum(1 for s in step_scores if s.quality == StepQuality.PARTIALLY_CORRECT)

        # Find first error
        first_error_at = None
        for score in step_scores:
            if score.quality == StepQuality.INCORRECT:
                first_error_at = score.step_id
                break

        return cls(
            path_id=path_id,
            step_scores=step_scores,
            total_score=total_score,
            min_score=min_score,
            avg_score=avg_score,
            num_correct=num_correct,
            num_incorrect=num_incorrect,
            num_partial=num_partial,
            first_error_at=first_error_at,
        )
