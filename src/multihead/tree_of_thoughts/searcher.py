"""Search strategies for Tree-of-Thoughts exploration.

Implements BFS, DFS, and beam search over the thought tree,
combining thought generation and state evaluation to find
optimal reasoning paths.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from multihead.tree_of_thoughts.evaluators import StateEvaluator
from multihead.tree_of_thoughts.generators import ThoughtGenerator
from multihead.tree_of_thoughts.models import SearchStrategy, ThoughtNode

logger = logging.getLogger(__name__)


class ToTSearcher:
    """Implements tree search strategies for exploring thought trees."""

    def __init__(
        self,
        thought_generator: ThoughtGenerator,
        state_evaluator: StateEvaluator,
        strategy: SearchStrategy = SearchStrategy.BFS,
        max_depth: int = 5,
        max_thoughts_per_state: int = 3,
        beam_width: int = 3,
    ):
        """Initialize ToT searcher.

        Args:
            thought_generator: Generator for alternative thoughts
            state_evaluator: Evaluator for state quality
            strategy: Search strategy (BFS, DFS, beam)
            max_depth: Maximum depth to explore
            max_thoughts_per_state: Number of alternatives per state
            beam_width: For beam search, how many paths to keep
        """
        self.generator = thought_generator
        self.evaluator = state_evaluator
        self.strategy = strategy
        self.max_depth = max_depth
        self.max_thoughts_per_state = max_thoughts_per_state
        self.beam_width = beam_width

    async def search(
        self,
        initial_state: Any,
        goal: str,
        is_goal_reached: Callable[[Any], bool],
    ) -> tuple[ThoughtNode | None, list[ThoughtNode]]:
        """Search the thought tree for a solution.

        Args:
            initial_state: Starting state
            goal: Goal description
            is_goal_reached: Function to check if a state achieves the goal

        Returns:
            (best_node, all_explored_nodes) tuple
        """
        context = {"goal": goal}

        # Create root node
        root = ThoughtNode(
            node_id="root",
            state=initial_state,
            step_description="Initial state",
            depth=0,
        )
        root.evaluation_score = await self.evaluator.evaluate(initial_state, context)

        explored_nodes = [root]

        if self.strategy == SearchStrategy.BFS:
            return await self._search_bfs(root, context, is_goal_reached, explored_nodes)
        elif self.strategy == SearchStrategy.DFS:
            return await self._search_dfs(root, context, is_goal_reached, explored_nodes)
        elif self.strategy == SearchStrategy.BEAM:
            return await self._search_beam(root, context, is_goal_reached, explored_nodes)
        else:
            raise ValueError(f"Unknown search strategy: {self.strategy}")

    async def _search_bfs(
        self,
        root: ThoughtNode,
        context: dict[str, Any],
        is_goal_reached: Callable[[Any], bool],
        explored_nodes: list[ThoughtNode],
    ) -> tuple[ThoughtNode | None, list[ThoughtNode]]:
        """Breadth-first search: explore all alternatives at each level."""
        queue = [root]
        best_terminal = None
        best_score = -1.0

        while queue:
            current = queue.pop(0)

            # Check if goal reached
            if is_goal_reached(current.state):
                current.is_terminal = True
                if current.evaluation_score > best_score:
                    best_score = current.evaluation_score
                    best_terminal = current
                continue

            # Don't expand beyond max depth
            if current.depth >= self.max_depth:
                current.is_terminal = True
                if current.evaluation_score > best_score:
                    best_score = current.evaluation_score
                    best_terminal = current
                continue

            # Generate alternative thoughts
            try:
                thoughts = await self.generator.generate_thoughts(
                    current.state,
                    context,
                    self.max_thoughts_per_state,
                )

                for i, (description, new_state) in enumerate(thoughts):
                    child = ThoughtNode(
                        node_id=f"{current.node_id}-{i}",
                        state=new_state,
                        step_description=description,
                    )
                    child.evaluation_score = await self.evaluator.evaluate(new_state, context)

                    current.add_child(child)
                    explored_nodes.append(child)
                    queue.append(child)

            except Exception as e:
                logger.error("Failed to expand node %s: %s", current.node_id, e)
                current.is_terminal = True

        # If no goal reached, return highest-scoring explored node
        if best_terminal is None and explored_nodes:
            best_terminal = max(explored_nodes, key=lambda n: n.evaluation_score)

        return best_terminal, explored_nodes

    async def _search_dfs(
        self,
        root: ThoughtNode,
        context: dict[str, Any],
        is_goal_reached: Callable[[Any], bool],
        explored_nodes: list[ThoughtNode],
    ) -> tuple[ThoughtNode | None, list[ThoughtNode]]:
        """Depth-first search: follow one path to completion before backtracking."""
        stack = [root]
        best_terminal = None
        best_score = -1.0

        while stack:
            current = stack.pop()  # LIFO for DFS

            # Check if goal reached
            if is_goal_reached(current.state):
                current.is_terminal = True
                if current.evaluation_score > best_score:
                    best_score = current.evaluation_score
                    best_terminal = current
                continue

            # Don't expand beyond max depth
            if current.depth >= self.max_depth:
                current.is_terminal = True
                if current.evaluation_score > best_score:
                    best_score = current.evaluation_score
                    best_terminal = current
                continue

            # Generate alternative thoughts
            try:
                thoughts = await self.generator.generate_thoughts(
                    current.state,
                    context,
                    self.max_thoughts_per_state,
                )

                # Add in reverse order so first thought is explored first
                for i in reversed(range(len(thoughts))):
                    description, new_state = thoughts[i]
                    child = ThoughtNode(
                        node_id=f"{current.node_id}-{i}",
                        state=new_state,
                        step_description=description,
                    )
                    child.evaluation_score = await self.evaluator.evaluate(new_state, context)

                    current.add_child(child)
                    explored_nodes.append(child)
                    stack.append(child)

            except Exception as e:
                logger.error("Failed to expand node %s: %s", current.node_id, e)
                current.is_terminal = True

        # If no goal reached, return highest-scoring explored node
        if best_terminal is None and explored_nodes:
            best_terminal = max(explored_nodes, key=lambda n: n.evaluation_score)

        return best_terminal, explored_nodes

    async def _search_beam(
        self,
        root: ThoughtNode,
        context: dict[str, Any],
        is_goal_reached: Callable[[Any], bool],
        explored_nodes: list[ThoughtNode],
    ) -> tuple[ThoughtNode | None, list[ThoughtNode]]:
        """Beam search: keep top-k most promising paths at each level."""
        current_beam = [root]
        best_terminal = None
        best_score = -1.0

        for depth in range(self.max_depth):
            if not current_beam:
                break

            next_beam = []

            for current in current_beam:
                # Check if goal reached
                if is_goal_reached(current.state):
                    current.is_terminal = True
                    if current.evaluation_score > best_score:
                        best_score = current.evaluation_score
                        best_terminal = current
                    continue

                # Generate alternative thoughts
                try:
                    thoughts = await self.generator.generate_thoughts(
                        current.state,
                        context,
                        self.max_thoughts_per_state,
                    )

                    for i, (description, new_state) in enumerate(thoughts):
                        child = ThoughtNode(
                            node_id=f"{current.node_id}-{i}",
                            state=new_state,
                            step_description=description,
                        )
                        child.evaluation_score = await self.evaluator.evaluate(new_state, context)

                        current.add_child(child)
                        explored_nodes.append(child)
                        next_beam.append(child)

                except Exception as e:
                    logger.error("Failed to expand node %s: %s", current.node_id, e)
                    current.is_terminal = True

            # Keep only top-k nodes for next iteration
            next_beam.sort(key=lambda n: n.evaluation_score, reverse=True)
            current_beam = next_beam[: self.beam_width]

        # Mark remaining beam nodes as terminal
        for node in current_beam:
            node.is_terminal = True
            if node.evaluation_score > best_score:
                best_score = node.evaluation_score
                best_terminal = node

        # If no goal reached, return highest-scoring explored node
        if best_terminal is None and explored_nodes:
            best_terminal = max(explored_nodes, key=lambda n: n.evaluation_score)

        return best_terminal, explored_nodes
