"""Core RecipeLearner class - learns optimal recipes from BotVibes experts.

This module enables MultiHead to improve its own recipes by:
1. Querying BotVibes experts for recipe design
2. Benchmarking proposed recipes on test data
3. Evaluating via consensus whether to adopt
4. Tracking recipe versions in SolverRegistry
5. Sharing successes back to the knowledge network

The key insight: "BotVibes knows better" - external experts can design
better recipes than manual ones.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from multihead.acp_bridge import ACPBridge
from multihead.consensus import ConsensusEngine
from multihead.orchestrator import Orchestrator

from ._benchmarking import benchmark_recipe as _benchmark_recipe
from ._evaluation import (
    build_evaluation_prompt,
    evaluate_by_benchmarks,
    evaluate_with_consensus,
)
from ._parsing import parse_recipe_from_response
from ._persistence import (
    record_evaluation_votes as _record_evaluation_votes,
    save_recipe as _save_recipe,
    share_success as _share_success,
)

if TYPE_CHECKING:
    from multihead.head_manager import HeadManager
    from multihead.registry.solver_registry import SolverRegistry

logger = logging.getLogger(__name__)


class RecipeLearner:
    """Learns optimal recipes from BotVibes experts.

    Workflow:
    1. Query BotVibes for recipe design (capability: recipe_design)
    2. Parse and validate proposed recipe
    3. Benchmark on test data
    4. Consensus evaluation (adopt, modify, reject)
    5. Deploy if approved + track version
    6. Share successes back to knowledge network
    """

    def __init__(
        self,
        acp_bridge: ACPBridge,
        recipes_dir: Path,
        test_data_dir: Path | None = None,
        orchestrator: Orchestrator | None = None,
        head_manager: HeadManager | None = None,
        registry: SolverRegistry | None = None,
    ):
        """Initialize recipe learner.

        Args:
            acp_bridge: ACPBridge for querying BotVibes experts
            recipes_dir: Directory containing local recipes
            test_data_dir: Optional directory with test data for benchmarking
            orchestrator: Optional Orchestrator for executing recipes during benchmarking
            head_manager: Optional HeadManager for consensus evaluation
            registry: Optional SolverRegistry for recipe version tracking
        """
        self.acp = acp_bridge
        self.recipes_dir = Path(recipes_dir)
        self.test_data_dir = Path(test_data_dir) if test_data_dir else None
        self.orchestrator = orchestrator
        self.head_manager = head_manager
        self.registry = registry
        self.consensus_engine: ConsensusEngine | None = None
        if head_manager:
            self.consensus_engine = ConsensusEngine(head_manager)
        self.recipes_dir.mkdir(parents=True, exist_ok=True)

    async def query_expert_recipe(
        self,
        goal: str,
        requirements: dict[str, Any],
        *,
        conversation_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Query BotVibes expert for recipe design.

        Args:
            goal: What the recipe should accomplish
            requirements: Constraints and requirements for the recipe
            conversation_id: Optional conversation ID for multi-turn

        Returns:
            Proposed recipe as dict, or None if query failed
        """
        # Build prompt for recipe design
        prompt = self._build_recipe_design_prompt(goal, requirements)

        # Create ACP task for recipe design
        logger.info("Querying BotVibes expert for recipe: %s", goal)

        try:
            task_id = await self.acp.create_task(
                capability="recipe_design",  # Specialized capability
                payload_ref=prompt,
                priority="normal",
                conversation_id=conversation_id,
            )

            # Wait for expert response
            result = await self.acp.poll_for_completion(
                task_id,
                timeout_seconds=300.0,  # 5 min for recipe design
                poll_interval=5.0,
            )

            if result and result.get("status") == "complete":
                output_ref = result.get("output_ref")
                if output_ref:
                    # Parse the recipe YAML from output
                    recipe = parse_recipe_from_response(output_ref)
                    logger.info("Received recipe from expert: %s", recipe.get("goal"))
                    return recipe

            logger.warning("Expert query failed or timed out")
            return None

        except Exception as e:
            logger.error("Failed to query expert recipe: %s", e)
            return None

    def _build_recipe_design_prompt(
        self,
        goal: str,
        requirements: dict[str, Any],
    ) -> str:
        """Build prompt for recipe design request.

        Args:
            goal: Recipe goal
            requirements: Requirements and constraints

        Returns:
            Formatted prompt
        """
        prompt_parts = [
            "Design a multi-step recipe (YAML WorkOrder) for the following task:\n",
            f"\nGoal: {goal}\n",
            "\nRequirements:",
        ]

        for key, value in requirements.items():
            prompt_parts.append(f"- {key}: {value}")

        prompt_parts.extend([
            "\n\nYour recipe should:",
            "1. Use atomic, composable steps",
            "2. Specify dependencies clearly (depends_on)",
            "3. Include appropriate task_types for routing",
            "4. Consider privacy constraints if handling sensitive data",
            "5. Include validation/consensus where appropriate",
            "\n\nReturn a valid YAML WorkOrder with goal and steps.",
        ])

        return "\n".join(prompt_parts)

    async def benchmark_recipe(
        self,
        recipe: dict[str, Any],
        test_cases: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Benchmark a recipe on test data.

        Args:
            recipe: Recipe to benchmark
            test_cases: List of test cases with inputs and expected outputs

        Returns:
            Benchmark results with success rate, metrics, etc.
        """
        return await _benchmark_recipe(recipe, test_cases, self.orchestrator)

    async def evaluate_recipe(
        self,
        proposed_recipe: dict[str, Any],
        benchmark_results: dict[str, Any],
        current_recipe: dict[str, Any] | None = None,
        current_benchmark: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate whether to adopt proposed recipe.

        Uses multi-head consensus when ConsensusEngine is available,
        falls back to benchmark comparison otherwise.

        Args:
            proposed_recipe: Recipe from expert
            benchmark_results: Benchmark results for proposed recipe
            current_recipe: Current recipe (if exists)
            current_benchmark: Benchmark results for current recipe (if exists)

        Returns:
            Evaluation decision with action, rationale, confidence, votes
        """
        # Build evaluation prompt
        prompt = build_evaluation_prompt(
            proposed_recipe, benchmark_results, current_recipe, current_benchmark,
        )

        # Try consensus evaluation first
        if self.consensus_engine and self.head_manager:
            try:
                consensus_decision = await evaluate_with_consensus(
                    prompt, self.consensus_engine, self.head_manager,
                )
                if consensus_decision:
                    return consensus_decision
            except Exception as e:
                logger.warning("Consensus evaluation failed, falling back: %s", e)

        # Fallback: simple benchmark comparison
        return evaluate_by_benchmarks(benchmark_results, current_benchmark)

    def save_recipe(
        self,
        recipe: dict[str, Any],
        name: str,
        *,
        metadata: dict[str, Any] | None = None,
        task_type: str | None = None,
        benchmark_results: dict[str, Any] | None = None,
    ) -> Path:
        """Save recipe to recipes directory and optionally track in registry.

        Args:
            recipe: Recipe to save
            name: Recipe filename (without .yaml extension)
            metadata: Optional metadata to include in file header
            task_type: Task type for registry tracking
            benchmark_results: Benchmark results for registry tracking

        Returns:
            Path to saved recipe
        """
        return _save_recipe(
            recipe,
            name,
            self.recipes_dir,
            metadata=metadata,
            task_type=task_type,
            benchmark_results=benchmark_results,
            registry=self.registry,
        )

    def record_evaluation_votes(
        self,
        recipe_id: str,
        version: int,
        votes: dict[str, str],
        confidences: dict[str, float] | None = None,
        rationales: dict[str, str] | None = None,
    ) -> None:
        """Record consensus evaluation votes in registry.

        Args:
            recipe_id: Recipe identifier
            version: Recipe version number
            votes: {head_id: action} mapping
            confidences: {head_id: confidence} mapping
            rationales: {head_id: rationale} mapping
        """
        if not self.registry:
            return

        _record_evaluation_votes(
            recipe_id,
            version,
            votes,
            self.registry,
            confidences=confidences,
            rationales=rationales,
        )

    async def share_success(
        self,
        recipe: dict[str, Any],
        benchmark_results: dict[str, Any],
    ) -> bool:
        """Share successful recipe back to BotVibes knowledge network.

        Args:
            recipe: Recipe that performed well
            benchmark_results: Benchmark results showing success

        Returns:
            True if shared successfully
        """
        return await _share_success(recipe, benchmark_results, self.acp)
