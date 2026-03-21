"""State evaluators for Tree-of-Thoughts exploration.

Evaluators assess how promising a given reasoning state is,
guiding the search toward high-quality solutions.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

logger = logging.getLogger(__name__)


class StateEvaluator(ABC):
    """Base class for evaluating the promise of a thought state."""

    @abstractmethod
    async def evaluate(
        self,
        state: Any,
        context: dict[str, Any],
    ) -> float:
        """Evaluate how promising this state is.

        Args:
            state: The state to evaluate
            context: Additional context (goal, constraints, etc.)

        Returns:
            Score from 0.0 (unpromising) to 1.0 (very promising)
        """
        pass


class LLMStateEvaluator(StateEvaluator):
    """Evaluates state quality using an LLM."""

    def __init__(self, generate_func: Callable[[str], Any]):
        """Initialize with an LLM generate function.

        Args:
            generate_func: Async function that takes a prompt and returns text
        """
        self.generate = generate_func

    async def evaluate(
        self,
        state: Any,
        context: dict[str, Any],
    ) -> float:
        """Evaluate state quality using LLM.

        Args:
            state: State to evaluate
            context: Context with goal

        Returns:
            Score 0.0-1.0
        """
        goal = context.get("goal", "solve the problem")

        prompt = f"""Goal: {goal}

Current state: {state}

On a scale of 0-10, how promising is this state for achieving the goal?
Consider:
- Progress toward the goal
- Likelihood of success if we continue this path
- Quality and correctness so far

Respond with ONLY a number from 0-10, nothing else."""

        try:
            response = await self.generate(prompt)
            # Extract first number from response
            response_str = str(response).strip()
            # Try to parse as float
            for word in response_str.split():
                try:
                    score = float(word)
                    # Normalize to 0-1
                    normalized = max(0.0, min(10.0, score)) / 10.0
                    return normalized
                except ValueError:
                    continue

            # Default to mid-range if parsing fails
            logger.warning("Could not parse evaluation score from: %s", response_str)
            return 0.5

        except Exception as e:
            logger.error("Evaluation failed: %s", e)
            return 0.5
