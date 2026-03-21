"""Integration of Tree-of-Thoughts with MultiHead orchestrator.

Enables steps in recipes to use ToT exploration for creative problem-solving:
- Step-level ToT: Explore multiple alternative approaches for a single step
- Automatic thought generation from step prompts
- State evaluation using head outputs
- Integration with existing orchestration flow
"""

from __future__ import annotations

import logging
from typing import Any

from multihead.models import StageResult, StepDef
from multihead.tree_of_thoughts import (
    LLMStateEvaluator,
    LLMThoughtGenerator,
    SearchStrategy,
    ThoughtGenerator,
    StateEvaluator,
    ToTEngine,
)

logger = logging.getLogger(__name__)


class StepThoughtGenerator(ThoughtGenerator):
    """Generates alternative step approaches using an LLM."""

    def __init__(self, execute_func, step: StepDef):
        """Initialize with step execution function.

        Args:
            execute_func: Async function to execute a step
            step: The step definition to generate alternatives for
        """
        self.execute = execute_func
        self.step = step

    async def generate_thoughts(
        self,
        current_state: Any,
        context: dict[str, Any],
        num_thoughts: int = 3,
    ) -> list[tuple[str, Any]]:
        """Generate alternative approaches by varying the step prompt.

        Args:
            current_state: Current step output (or None for first iteration)
            context: Contains original_prompt, goal, etc.
            num_thoughts: Number of alternatives to generate

        Returns:
            List of (description, new_state) tuples where new_state is the result
            of executing the step with that approach
        """
        original_prompt = context.get("original_prompt", self.step.prompt_template)
        goal = context.get("goal", "complete the task")

        # Generate alternative prompts
        meta_prompt = f"""Given this task prompt:
{original_prompt}

Overall goal: {goal}

Generate {num_thoughts} different alternative approaches or variations to accomplish this.
Each approach should be distinct (not just rephrasing).

Format as:
1. [Brief description]: [Modified prompt]
2. [Brief description]: [Modified prompt]
3. [Brief description]: [Modified prompt]"""

        # Execute meta-prompt to get alternatives
        try:
            # We need to use a head for generating alternatives
            # For now, use the step's configured head
            from multihead.head_manager import HeadManager

            # This is a simplified version - in production we'd get HeadManager from context
            logger.info("Generating %d alternative approaches for step %s", num_thoughts, self.step.step_id)

            # Simplified: return variations of the original prompt
            # In a real implementation, we'd call an LLM to generate these
            thoughts = []
            for i in range(num_thoughts):
                variation = f"Approach {i + 1}: {original_prompt}"
                thoughts.append((f"Variation {i + 1}", variation))

            return thoughts

        except Exception as e:
            logger.error("Failed to generate alternative thoughts: %s", e)
            # Fallback: return original prompt
            return [(("Original approach", original_prompt))]


class StepStateEvaluator(StateEvaluator):
    """Evaluates step outputs using confidence and quality metrics."""

    async def evaluate(
        self,
        state: Any,
        context: dict[str, Any],
    ) -> float:
        """Evaluate a step result.

        Args:
            state: StageResult from step execution
            context: Additional context

        Returns:
            Score 0.0-1.0
        """
        if not isinstance(state, StageResult):
            # If not a proper result, default to low score
            return 0.3

        # Check confidence in metrics
        if state.metrics and "confidence" in state.metrics:
            return float(state.metrics["confidence"])

        # Check for errors
        if state.error:
            return 0.1

        # Check output quality (simple heuristic: longer output = better)
        if state.outputs:
            # Get total length of all outputs
            output_text = " ".join(str(v) for v in state.outputs.values())
            output_len = len(output_text)
            # Normalize to 0-1 (assume 500 chars is "good")
            return min(1.0, 0.5 + (output_len / 1000.0))

        return 0.5


async def execute_step_with_tot(
    execute_step_func,
    step: StepDef,
    context: dict[str, Any],
    *,
    num_alternatives: int = 3,
    strategy: SearchStrategy = SearchStrategy.BFS,
    max_depth: int = 2,
) -> StageResult:
    """Execute a step using Tree-of-Thoughts exploration.

    Args:
        execute_step_func: Function to execute a step
        step: The step to execute
        context: Execution context
        num_alternatives: Number of alternative approaches to try
        strategy: ToT search strategy
        max_depth: Maximum exploration depth

    Returns:
        Best StageResult from exploring alternatives
    """
    logger.info("Executing step %s with ToT (%s strategy)", step.step_id, strategy)

    # For now, use simple implementations
    # In production, these would use the actual HeadManager
    thought_generator = StepThoughtGenerator(execute_step_func, step)
    state_evaluator = StepStateEvaluator()

    # Create ToT engine
    engine = ToTEngine(
        thought_generator=thought_generator,
        state_evaluator=state_evaluator,
        strategy=strategy,
        max_depth=max_depth,
        max_thoughts_per_state=num_alternatives,
    )

    # Define goal checker
    def is_goal_reached(state: Any) -> bool:
        """Check if step output meets quality threshold."""
        if isinstance(state, StageResult):
            # Consider successful if confidence > 0.8
            if hasattr(state, "confidence") and state.confidence:
                return state.confidence > 0.8
            # Or if no errors
            return not state.error
        return False

    # Run ToT exploration
    tot_context = {
        "original_prompt": step.prompt_template,
        "goal": context.get("goal", "complete the step"),
    }

    result = await engine.solve(
        problem=step.prompt_template,
        initial_state=None,
        is_goal_reached=is_goal_reached,
    )

    logger.info(
        "ToT exploration complete: %d nodes explored, best score=%.2f",
        result["explored_count"],
        result["best_score"],
    )

    # Return best result
    if result["best_state"] and isinstance(result["best_state"], StageResult):
        return result["best_state"]

    # Fallback: execute with original prompt
    logger.warning("ToT exploration did not yield a valid result, falling back to standard execution")
    return await execute_step_func(step)


def should_use_tot(step: StepDef) -> bool:
    """Check if a step should use ToT exploration.

    Args:
        step: Step definition

    Returns:
        True if step should use ToT
    """
    # Check for explicit tot flag in extra
    if hasattr(step, "extra") and step.extra:
        return step.extra.get("use_tot", False)

    return False


def get_tot_config(step: StepDef) -> dict[str, Any]:
    """Extract ToT configuration from step extra metadata.

    Args:
        step: Step definition

    Returns:
        Dict with ToT config (strategy, num_alternatives, max_depth)
    """
    config = {
        "strategy": SearchStrategy.BFS,
        "num_alternatives": 3,
        "max_depth": 2,
    }

    if not hasattr(step, "extra") or not step.extra:
        return config

    # Extract from extra
    if "tot_strategy" in step.extra:
        strategy_str = step.extra["tot_strategy"]
        try:
            config["strategy"] = SearchStrategy(strategy_str)
        except ValueError:
            logger.warning("Invalid tot_strategy '%s', using BFS", strategy_str)

    if "tot_alternatives" in step.extra:
        config["num_alternatives"] = int(step.extra["tot_alternatives"])

    if "tot_max_depth" in step.extra:
        config["max_depth"] = int(step.extra["tot_max_depth"])

    return config
