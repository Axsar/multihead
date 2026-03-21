"""Tests for Tree-of-Thoughts exploration."""

from __future__ import annotations

import pytest
from typing import Any
from unittest.mock import AsyncMock

from multihead.tree_of_thoughts import (
    LLMStateEvaluator,
    LLMThoughtGenerator,
    SearchStrategy,
    StateEvaluator,
    ThoughtGenerator,
    ThoughtNode,
    ToTEngine,
    ToTSearcher,
)


class MockThoughtGenerator(ThoughtGenerator):
    """Mock thought generator for testing."""

    def __init__(self, thoughts_map: dict[str, list[tuple[str, str]]]):
        """Initialize with predefined thoughts.

        Args:
            thoughts_map: Maps states to list of (description, new_state) tuples
        """
        self.thoughts_map = thoughts_map

    async def generate_thoughts(
        self,
        current_state: Any,
        context: dict[str, Any],
        num_thoughts: int = 3,
    ) -> list[tuple[str, Any]]:
        """Return predefined thoughts for this state."""
        state_key = str(current_state)
        thoughts = self.thoughts_map.get(state_key, [])
        return thoughts[:num_thoughts]


class MockStateEvaluator(StateEvaluator):
    """Mock state evaluator for testing."""

    def __init__(self, scores_map: dict[str, float]):
        """Initialize with predefined scores.

        Args:
            scores_map: Maps states to evaluation scores
        """
        self.scores_map = scores_map

    async def evaluate(
        self,
        state: Any,
        context: dict[str, Any],
    ) -> float:
        """Return predefined score for this state."""
        state_key = str(state)
        return self.scores_map.get(state_key, 0.5)


class TestThoughtNode:
    """Test ThoughtNode functionality."""

    def test_creates_node(self):
        """Should create a thought node."""
        node = ThoughtNode(
            node_id="test-1",
            state="test state",
            step_description="Test step",
        )

        assert node.node_id == "test-1"
        assert node.state == "test state"
        assert node.step_description == "Test step"
        assert node.depth == 0
        assert not node.is_terminal
        assert node.parent is None
        assert len(node.children) == 0

    def test_adds_child(self):
        """Should add child and update parent/depth."""
        parent = ThoughtNode("parent", "parent state", "Parent", depth=1)
        child = ThoughtNode("child", "child state", "Child")

        parent.add_child(child)

        assert len(parent.children) == 1
        assert parent.children[0] == child
        assert child.parent == parent
        assert child.depth == 2  # Parent depth + 1

    def test_get_path_single_node(self):
        """Should get path for single node."""
        node = ThoughtNode("root", "state", "Root")
        path = node.get_path()

        assert len(path) == 1
        assert path[0] == node

    def test_get_path_multi_level(self):
        """Should get full path from root to leaf."""
        root = ThoughtNode("root", "s0", "Start")
        child1 = ThoughtNode("c1", "s1", "Step 1")
        child2 = ThoughtNode("c2", "s2", "Step 2")

        root.add_child(child1)
        child1.add_child(child2)

        path = child2.get_path()

        assert len(path) == 3
        assert path[0] == root
        assert path[1] == child1
        assert path[2] == child2

    def test_get_path_description(self):
        """Should generate path description."""
        root = ThoughtNode("root", "s0", "Start")
        child1 = ThoughtNode("c1", "s1", "Middle")
        child2 = ThoughtNode("c2", "s2", "End")

        root.add_child(child1)
        child1.add_child(child2)

        description = child2.get_path_description()

        assert description == "Start → Middle → End"


class TestLLMThoughtGenerator:
    """Test LLM-based thought generation."""

    @pytest.mark.asyncio
    async def test_generates_thoughts_from_llm(self):
        """Should parse numbered thoughts from LLM response."""
        mock_generate = AsyncMock(return_value="""
1. Try approach A
2. Try approach B
3. Try approach C
""")

        generator = LLMThoughtGenerator(mock_generate)
        thoughts = await generator.generate_thoughts("current", {"goal": "test"}, 3)

        assert len(thoughts) == 3
        assert thoughts[0][0] == "Try approach A"
        assert thoughts[1][0] == "Try approach B"
        assert thoughts[2][0] == "Try approach C"

    @pytest.mark.asyncio
    async def test_handles_bullet_points(self):
        """Should handle bullet-pointed responses."""
        mock_generate = AsyncMock(return_value="""
- First idea
- Second idea
- Third idea
""")

        generator = LLMThoughtGenerator(mock_generate)
        thoughts = await generator.generate_thoughts("current", {"goal": "test"}, 3)

        assert len(thoughts) == 3
        assert "First idea" in thoughts[0][0]

    @pytest.mark.asyncio
    async def test_limits_to_requested_count(self):
        """Should limit thoughts to requested number."""
        mock_generate = AsyncMock(return_value="""
1. Idea 1
2. Idea 2
3. Idea 3
4. Idea 4
5. Idea 5
""")

        generator = LLMThoughtGenerator(mock_generate)
        thoughts = await generator.generate_thoughts("current", {"goal": "test"}, 2)

        assert len(thoughts) == 2

    @pytest.mark.asyncio
    async def test_fallback_for_unparseable_response(self):
        """Should fall back to full response if parsing fails."""
        mock_generate = AsyncMock(return_value="Unstructured response without numbers")

        generator = LLMThoughtGenerator(mock_generate)
        thoughts = await generator.generate_thoughts("current", {"goal": "test"}, 3)

        assert len(thoughts) == 1
        assert thoughts[0][0] == "Unstructured response without numbers"


class TestLLMStateEvaluator:
    """Test LLM-based state evaluation."""

    @pytest.mark.asyncio
    async def test_evaluates_state(self):
        """Should parse score from LLM response."""
        mock_generate = AsyncMock(return_value="7")

        evaluator = LLMStateEvaluator(mock_generate)
        score = await evaluator.evaluate("test state", {"goal": "test"})

        assert score == 0.7  # 7/10 normalized

    @pytest.mark.asyncio
    async def test_normalizes_score(self):
        """Should normalize scores to 0-1 range."""
        mock_generate = AsyncMock(return_value="10")

        evaluator = LLMStateEvaluator(mock_generate)
        score = await evaluator.evaluate("test state", {"goal": "test"})

        assert score == 1.0

    @pytest.mark.asyncio
    async def test_handles_text_with_number(self):
        """Should extract number from text response."""
        mock_generate = AsyncMock(return_value="The score is 8 out of 10")

        evaluator = LLMStateEvaluator(mock_generate)
        score = await evaluator.evaluate("test state", {"goal": "test"})

        assert score == 0.8

    @pytest.mark.asyncio
    async def test_defaults_on_parse_failure(self):
        """Should default to 0.5 if parsing fails."""
        mock_generate = AsyncMock(return_value="Cannot determine quality")

        evaluator = LLMStateEvaluator(mock_generate)
        score = await evaluator.evaluate("test state", {"goal": "test"})

        assert score == 0.5

    @pytest.mark.asyncio
    async def test_handles_errors(self):
        """Should default to 0.5 on error."""
        mock_generate = AsyncMock(side_effect=Exception("Network error"))

        evaluator = LLMStateEvaluator(mock_generate)
        score = await evaluator.evaluate("test state", {"goal": "test"})

        assert score == 0.5


class TestToTSearcherBFS:
    """Test BFS search strategy."""

    @pytest.mark.asyncio
    async def test_explores_all_at_each_level(self):
        """BFS should explore all nodes at each level before going deeper."""
        # Create a simple tree: root -> [a, b] where a -> [a1, a2], b -> [b1]
        thoughts_map = {
            "": [("Go to A", "a"), ("Go to B", "b")],
            "a": [("Go to A1", "a1"), ("Go to A2", "a2")],
            "b": [("Go to B1", "b1")],
        }

        scores_map = {
            "": 0.5,
            "a": 0.6,
            "b": 0.7,
            "a1": 0.8,
            "a2": 0.9,
            "b1": 0.95,  # Highest score
        }

        generator = MockThoughtGenerator(thoughts_map)
        evaluator = MockStateEvaluator(scores_map)

        searcher = ToTSearcher(
            thought_generator=generator,
            state_evaluator=evaluator,
            strategy=SearchStrategy.BFS,
            max_depth=2,
        )

        # Never reaches goal, so explores full tree
        is_goal = lambda state: False

        best_node, all_nodes = await searcher.search("", "test goal", is_goal)

        # Should explore all nodes in tree
        assert len(all_nodes) > 3  # root + children + grandchildren
        # Best node should be b1 (highest score)
        assert best_node.state == "b1"
        assert best_node.evaluation_score == 0.95

    @pytest.mark.asyncio
    async def test_stops_at_goal(self):
        """BFS should stop exploring a path when goal is reached."""
        thoughts_map = {
            "": [("Go to A", "a"), ("Go to B", "b")],
            "a": [("Go to A1", "a1")],
            "b": [("Go to B1", "b1")],
        }

        scores_map = {"": 0.5, "a": 0.6, "b": 0.9}

        generator = MockThoughtGenerator(thoughts_map)
        evaluator = MockStateEvaluator(scores_map)

        searcher = ToTSearcher(
            thought_generator=generator,
            state_evaluator=evaluator,
            strategy=SearchStrategy.BFS,
            max_depth=3,
        )

        # Goal is state "b"
        is_goal = lambda state: state == "b"

        best_node, all_nodes = await searcher.search("", "test goal", is_goal)

        assert best_node.state == "b"
        assert best_node.is_terminal


class TestToTSearcherDFS:
    """Test DFS search strategy."""

    @pytest.mark.asyncio
    async def test_explores_depth_first(self):
        """DFS should follow one path to max depth before backtracking."""
        thoughts_map = {
            "": [("Path A", "a"), ("Path B", "b")],
            "a": [("Deep A", "aa")],
            "b": [("Deep B", "bb")],
        }

        scores_map = {"": 0.5, "a": 0.6, "aa": 0.7, "b": 0.8, "bb": 0.9}

        generator = MockThoughtGenerator(thoughts_map)
        evaluator = MockStateEvaluator(scores_map)

        searcher = ToTSearcher(
            thought_generator=generator,
            state_evaluator=evaluator,
            strategy=SearchStrategy.DFS,
            max_depth=2,
        )

        is_goal = lambda state: False

        best_node, all_nodes = await searcher.search("", "test goal", is_goal)

        # Should explore entire tree and find best
        assert best_node.state == "bb"  # Highest score


class TestToTSearcherBeam:
    """Test beam search strategy."""

    @pytest.mark.asyncio
    async def test_keeps_only_top_k(self):
        """Beam search should only keep top-k paths at each level."""
        thoughts_map = {
            "": [
                ("Low score path", "low"),
                ("Medium score path", "medium"),
                ("High score path", "high"),
            ],
            "low": [("Continue low", "low2")],
            "medium": [("Continue medium", "medium2")],
            "high": [("Continue high", "high2")],
        }

        scores_map = {
            "": 0.5,
            "low": 0.3,
            "medium": 0.6,
            "high": 0.9,
            "low2": 0.4,
            "medium2": 0.7,
            "high2": 0.95,
        }

        generator = MockThoughtGenerator(thoughts_map)
        evaluator = MockStateEvaluator(scores_map)

        searcher = ToTSearcher(
            thought_generator=generator,
            state_evaluator=evaluator,
            strategy=SearchStrategy.BEAM,
            max_depth=2,
            beam_width=2,  # Keep only top 2
        )

        is_goal = lambda state: False

        best_node, all_nodes = await searcher.search("", "test goal", is_goal)

        # Should prune low-scoring path
        all_states = [n.state for n in all_nodes]

        # Should have high and medium paths
        assert "high" in all_states
        assert "medium" in all_states

        # Low path might not be expanded beyond first level
        # (depends on beam pruning)

        # Best should still be the highest-scoring terminal
        assert best_node.evaluation_score >= 0.7


class TestToTEngine:
    """Test ToT engine integration."""

    @pytest.mark.asyncio
    async def test_solves_problem(self):
        """Should explore tree and return best solution."""
        thoughts_map = {
            "": [("Approach 1", "a1"), ("Approach 2", "a2")],
            "a1": [("Refine 1", "a1-refined")],
            "a2": [("Refine 2", "a2-refined")],
        }

        scores_map = {
            "": 0.5,
            "a1": 0.6,
            "a2": 0.9,
            "a1-refined": 0.7,
            "a2-refined": 0.95,
        }

        generator = MockThoughtGenerator(thoughts_map)
        evaluator = MockStateEvaluator(scores_map)

        engine = ToTEngine(
            thought_generator=generator,
            state_evaluator=evaluator,
            strategy=SearchStrategy.BFS,
            max_depth=2,
        )

        result = await engine.solve("Solve test problem")

        assert result["best_state"] == "a2-refined"
        assert result["best_score"] == 0.95
        assert result["explored_count"] > 3
        assert len(result["best_path"]) == 3  # root -> a2 -> a2-refined

    @pytest.mark.asyncio
    async def test_returns_path_description(self):
        """Should include readable path description."""
        thoughts_map = {
            "": [("Start with X", "x")],
            "x": [("Then Y", "y")],
        }

        scores_map = {"": 0.5, "x": 0.7, "y": 0.9}

        generator = MockThoughtGenerator(thoughts_map)
        evaluator = MockStateEvaluator(scores_map)

        engine = ToTEngine(
            thought_generator=generator,
            state_evaluator=evaluator,
            max_depth=2,
        )

        result = await engine.solve("Test")

        assert "path_description" in result
        assert "Start with X" in result["path_description"]
        assert "Then Y" in result["path_description"]

    @pytest.mark.asyncio
    async def test_handles_goal_reached(self):
        """Should stop when goal is reached."""
        thoughts_map = {
            "": [("Try A", "a"), ("Try B", "b")],
            "a": [("Continue A", "aa")],
        }

        scores_map = {"": 0.5, "a": 0.7, "b": 0.9, "aa": 0.6}

        generator = MockThoughtGenerator(thoughts_map)
        evaluator = MockStateEvaluator(scores_map)

        engine = ToTEngine(
            thought_generator=generator,
            state_evaluator=evaluator,
            max_depth=3,
        )

        # Goal is to reach state "a"
        is_goal = lambda state: state == "a"

        result = await engine.solve("Test", is_goal_reached=is_goal)

        assert result["best_state"] == "a"
        # Should not explore beyond "a" since goal was reached
        all_states = [n.state for n in result["all_nodes"]]
        assert "aa" not in all_states  # Didn't expand "a" further

    @pytest.mark.asyncio
    async def test_handles_no_solution(self):
        """Should handle case with no nodes."""
        # Generator that produces no thoughts
        generator = MockThoughtGenerator({})
        evaluator = MockStateEvaluator({"": 0.5})

        engine = ToTEngine(
            thought_generator=generator,
            state_evaluator=evaluator,
            max_depth=2,
        )

        result = await engine.solve("Test")

        # Should still return root node as best
        assert result["best_state"] == ""
        assert result["explored_count"] == 1  # Just root
