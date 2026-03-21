"""Main ToT engine for orchestrating Tree-of-Thoughts exploration.

The ToTEngine ties together thought generation, state evaluation,
and tree search to solve problems through multi-path reasoning.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from multihead.tree_of_thoughts.evaluators import StateEvaluator
from multihead.tree_of_thoughts.generators import ThoughtGenerator
from multihead.tree_of_thoughts.models import SearchStrategy, ThoughtNode
from multihead.tree_of_thoughts.searcher import ToTSearcher

logger = logging.getLogger(__name__)


class ToTEngine:
    """Main engine for Tree-of-Thoughts exploration.

    Orchestrates thought generation, evaluation, and search to find
    optimal reasoning paths through creative exploration.
    """

    def __init__(
        self,
        thought_generator: ThoughtGenerator,
        state_evaluator: StateEvaluator,
        strategy: SearchStrategy = SearchStrategy.BFS,
        max_depth: int = 5,
        max_thoughts_per_state: int = 3,
        beam_width: int = 3,
    ):
        """Initialize ToT engine.

        Args:
            thought_generator: Generator for alternative thoughts
            state_evaluator: Evaluator for state quality
            strategy: Search strategy (BFS, DFS, beam)
            max_depth: Maximum tree depth
            max_thoughts_per_state: Alternatives per state
            beam_width: For beam search, paths to keep
        """
        self.searcher = ToTSearcher(
            thought_generator=thought_generator,
            state_evaluator=state_evaluator,
            strategy=strategy,
            max_depth=max_depth,
            max_thoughts_per_state=max_thoughts_per_state,
            beam_width=beam_width,
        )

    async def solve(
        self,
        problem: str,
        initial_state: Any = None,
        is_goal_reached: Callable[[Any], bool] | None = None,
    ) -> dict[str, Any]:
        """Solve a problem using Tree-of-Thoughts exploration.

        Args:
            problem: Problem description/goal
            initial_state: Optional starting state (None = empty)
            is_goal_reached: Optional goal check function (default: always False)

        Returns:
            Dict with:
            - best_path: List of ThoughtNodes from root to best solution
            - best_state: The final state of the best path
            - best_score: Evaluation score of best path
            - explored_count: Total nodes explored
            - all_nodes: All explored nodes (for visualization)
        """
        logger.info("Starting ToT search: %s", problem)

        # Default goal checker (never satisfied, explore full tree)
        if is_goal_reached is None:
            is_goal_reached = lambda state: False

        # Run search
        best_node, all_nodes = await self.searcher.search(
            initial_state=initial_state or "",
            goal=problem,
            is_goal_reached=is_goal_reached,
        )

        if best_node is None:
            logger.warning("Search found no solution")
            return {
                "best_path": [],
                "best_state": None,
                "best_score": 0.0,
                "explored_count": len(all_nodes),
                "all_nodes": all_nodes,
            }

        path = best_node.get_path()
        logger.info(
            "ToT search complete: %d nodes explored, best score=%.2f, depth=%d",
            len(all_nodes),
            best_node.evaluation_score,
            best_node.depth,
        )

        return {
            "best_path": path,
            "best_state": best_node.state,
            "best_score": best_node.evaluation_score,
            "explored_count": len(all_nodes),
            "all_nodes": all_nodes,
            "path_description": best_node.get_path_description(),
        }
