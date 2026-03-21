"""Complete recipe learning workflow."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from ._learner import RecipeLearner

logger = logging.getLogger(__name__)


async def learn_recipe_workflow(
    goal: str,
    requirements: dict[str, Any],
    test_cases: list[dict[str, Any]],
    learner: RecipeLearner,
    *,
    save_name: str | None = None,
    task_type: str | None = None,
) -> dict[str, Any]:
    """Complete recipe learning workflow.

    Args:
        goal: What the recipe should accomplish
        requirements: Recipe requirements
        test_cases: Test data for benchmarking
        learner: RecipeLearner instance
        save_name: Optional name for saving adopted recipe
        task_type: Task type for registry tracking

    Returns:
        Workflow results with decision and saved recipe path
    """
    logger.info("Starting recipe learning workflow for: %s", goal)

    # Step 1: Query expert
    proposed_recipe = await learner.query_expert_recipe(goal, requirements)
    if not proposed_recipe:
        return {
            "success": False,
            "error": "Failed to get recipe from expert",
        }

    # Step 2: Benchmark
    benchmark_results = await learner.benchmark_recipe(proposed_recipe, test_cases)

    # Step 3: Evaluate (consensus if available, fallback to benchmark comparison)
    evaluation = await learner.evaluate_recipe(proposed_recipe, benchmark_results)

    # Step 4: Adopt if approved
    saved_path = None
    if evaluation["action"] == "adopt":
        name = save_name or f"learned-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        saved_path = learner.save_recipe(
            proposed_recipe,
            name,
            metadata={
                "source": "botvibes_expert",
                "goal": goal,
                "decision": evaluation["action"],
                "confidence": evaluation["confidence"],
            },
            task_type=task_type,
            benchmark_results=benchmark_results,
        )

        # Mark as adopted in registry
        if learner.registry and task_type:
            versions = learner.registry.list_recipe_versions(recipe_id=name)
            if versions:
                learner.registry.adopt_recipe_version(name, versions[0]["version"])

        # Record evaluation votes if available
        if learner.registry and evaluation.get("votes"):
            versions = learner.registry.list_recipe_versions(recipe_id=name)
            if versions:
                learner.record_evaluation_votes(
                    name, versions[0]["version"], evaluation["votes"],
                )

        # Step 5: Share success
        await learner.share_success(proposed_recipe, benchmark_results)

    return {
        "success": True,
        "proposed_recipe": proposed_recipe,
        "benchmark_results": benchmark_results,
        "evaluation": evaluation,
        "saved_path": str(saved_path) if saved_path else None,
    }
