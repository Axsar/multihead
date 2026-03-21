"""Tests for consensus-based decomposition."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from multihead.auto_decomposition import AutoDecomposer
from multihead.consensus import ConsensusStrategy
from multihead.decomposer import DecompositionPlan, TaskDecomposer, TaskNode


class TestConsensusDecomposition:
    """Test multi-head consensus decomposition."""

    @pytest.fixture
    def mock_head_manager(self):
        """Create a mock HeadManager."""
        mgr = MagicMock()
        mgr.get_states.return_value = {
            "mock-llm": {
                "head_id": "mock-llm",
                "state": "active",
                "kind": "llm",
                "is_active": True,
            }
        }
        mgr.get_manifest.return_value = {"kind": "llm"}
        registry = MagicMock()
        registry.list_heads.return_value = ["mock-llm", "qwen-llm", "openai-gpt4o"]
        registry.get_head.return_value = MagicMock(kind="llm")
        mgr.registry = registry
        return mgr

    @pytest.fixture
    def mock_task_decomposer(self, mock_head_manager):
        """Create a mock TaskDecomposer."""
        decomposer = MagicMock(spec=TaskDecomposer)
        decomposer.heads = mock_head_manager
        return decomposer

    @pytest.fixture
    def auto_decomposer(self, mock_task_decomposer):
        """Create AutoDecomposer with mocked base decomposer."""
        decomposer = AutoDecomposer(head_manager=mock_task_decomposer.heads)
        decomposer._base_decomposer = mock_task_decomposer
        return decomposer

    def create_mock_plan(
        self,
        goal: str,
        num_steps: int = 5,
        complexity: str = "moderate",
    ) -> DecompositionPlan:
        """Create a mock DecompositionPlan."""
        phases = [
            TaskNode(
                id="1",
                goal="Phase 1",
                children=[
                    TaskNode(
                        id=f"1.{i}",
                        goal=f"Step {i}",
                        action_type="edit",
                    )
                    for i in range(1, num_steps + 1)
                ],
            )
        ]
        return DecompositionPlan(
            goal=goal,
            complexity=complexity,
            phases=phases,
        )

    @pytest.mark.asyncio
    async def test_decompose_with_consensus_basic(
        self, auto_decomposer, mock_task_decomposer
    ):
        """Should run decomposition with multiple heads and vote on best."""
        goal = "Build a web scraper"

        # Mock decomposition results from different heads
        plan1 = self.create_mock_plan(goal, num_steps=5, complexity="moderate")
        plan2 = self.create_mock_plan(goal, num_steps=6, complexity="moderate")
        plan3 = self.create_mock_plan(goal, num_steps=5, complexity="moderate")

        mock_task_decomposer.decompose = AsyncMock(
            side_effect=[plan1, plan2, plan3]
        )

        # Run consensus decomposition
        winner, metadata = await auto_decomposer.decompose_with_consensus(
            goal=goal,
            heads=["mock-llm", "qwen-llm", "openai-gpt4o"],
            strategy=ConsensusStrategy.MAJORITY,
        )

        # Should have called decompose for each head
        assert mock_task_decomposer.decompose.call_count == 3

        # Should return a plan
        assert isinstance(winner, DecompositionPlan)
        assert winner.goal == goal

        # Should include consensus metadata
        assert "agreement_score" in metadata
        assert "winner_head" in metadata
        assert "num_votes" in metadata
        assert metadata["num_votes"] == 3

    @pytest.mark.asyncio
    async def test_consensus_weighted_strategy(
        self, auto_decomposer, mock_task_decomposer
    ):
        """Should apply weights when using WEIGHTED strategy."""
        goal = "Implement user authentication"

        # Create plans with different quality scores
        plan1 = self.create_mock_plan(goal, num_steps=3, complexity="simple")
        plan2 = self.create_mock_plan(goal, num_steps=10, complexity="moderate")
        plan3 = self.create_mock_plan(goal, num_steps=8, complexity="moderate")

        mock_task_decomposer.decompose = AsyncMock(
            side_effect=[plan1, plan2, plan3]
        )

        # Run with weights favoring second head
        weights = {
            "mock-llm": 1.0,
            "qwen-llm": 2.0,  # Higher weight
            "openai-gpt4o": 1.0,
        }

        winner, metadata = await auto_decomposer.decompose_with_consensus(
            goal=goal,
            heads=["mock-llm", "qwen-llm", "openai-gpt4o"],
            strategy=ConsensusStrategy.WEIGHTED,
            weights=weights,
        )

        assert isinstance(winner, DecompositionPlan)
        assert metadata["winner_head"] in weights

    @pytest.mark.asyncio
    async def test_consensus_first_to_ahead_strategy(
        self, auto_decomposer, mock_task_decomposer
    ):
        """Should use quality scoring for FIRST_TO_AHEAD."""
        goal = "Create data pipeline"

        # Create plans with different quality
        plan1 = self.create_mock_plan(goal, num_steps=100, complexity="simple")  # Too many steps
        plan2 = self.create_mock_plan(goal, num_steps=15, complexity="moderate")  # Good
        plan3 = self.create_mock_plan(goal, num_steps=2, complexity="complex")  # Too few

        mock_task_decomposer.decompose = AsyncMock(
            side_effect=[plan1, plan2, plan3]
        )

        winner, metadata = await auto_decomposer.decompose_with_consensus(
            goal=goal,
            heads=["mock-llm", "qwen-llm", "openai-gpt4o"],
            strategy=ConsensusStrategy.FIRST_TO_AHEAD,
        )

        # Winner should be plan2 (best quality score)
        assert winner.total_steps == 15

    @pytest.mark.asyncio
    async def test_consensus_handles_failures(
        self, auto_decomposer, mock_task_decomposer
    ):
        """Should handle when some heads fail to decompose."""
        goal = "Deploy application"

        plan1 = self.create_mock_plan(goal, num_steps=8)

        # First head succeeds, others fail
        mock_task_decomposer.decompose = AsyncMock(
            side_effect=[
                plan1,
                Exception("Head failed"),
                Exception("Another failure"),
            ]
        )

        winner, metadata = await auto_decomposer.decompose_with_consensus(
            goal=goal,
            heads=["mock-llm", "qwen-llm", "openai-gpt4o"],
            strategy=ConsensusStrategy.MAJORITY,
        )

        # Should still work with 1 successful plan
        assert isinstance(winner, DecompositionPlan)
        assert metadata["num_votes"] == 1

    @pytest.mark.asyncio
    async def test_consensus_all_heads_fail(
        self, auto_decomposer, mock_task_decomposer
    ):
        """Should raise error when all heads fail."""
        goal = "Impossible task"

        # All heads fail
        mock_task_decomposer.decompose = AsyncMock(
            side_effect=Exception("Decomposition failed")
        )

        with pytest.raises(ValueError, match="No heads successfully decomposed"):
            await auto_decomposer.decompose_with_consensus(
                goal=goal,
                heads=["mock-llm", "qwen-llm"],
                strategy=ConsensusStrategy.MAJORITY,
            )

    @pytest.mark.asyncio
    async def test_plan_quality_scoring(self, auto_decomposer):
        """Should score plan quality correctly."""
        # Good plan: moderate complexity, reasonable step count
        good_plan = self.create_mock_plan(
            "Build API",
            num_steps=12,
            complexity="moderate",
        )
        good_score = auto_decomposer._score_plan_quality(good_plan)

        # Too many steps plan
        too_many_plan = self.create_mock_plan(
            "Build API",
            num_steps=80,
            complexity="moderate",
        )
        too_many_score = auto_decomposer._score_plan_quality(too_many_plan)

        # Too few steps plan
        too_few_plan = self.create_mock_plan(
            "Build API",
            num_steps=2,
            complexity="moderate",
        )
        too_few_score = auto_decomposer._score_plan_quality(too_few_plan)

        # Good plan should score highest
        assert good_score > too_many_score
        assert good_score > too_few_score

    @pytest.mark.asyncio
    async def test_plan_signature_similarity(self, auto_decomposer):
        """Should generate same signature for similar plans."""
        plan1 = self.create_mock_plan("Task", num_steps=5, complexity="moderate")
        plan2 = self.create_mock_plan("Task", num_steps=5, complexity="moderate")
        plan3 = self.create_mock_plan("Task", num_steps=10, complexity="complex")

        sig1 = auto_decomposer._plan_signature(plan1)
        sig2 = auto_decomposer._plan_signature(plan2)
        sig3 = auto_decomposer._plan_signature(plan3)

        # Similar plans should have same signature
        assert sig1 == sig2

        # Different plans should have different signatures
        assert sig1 != sig3

    @pytest.mark.asyncio
    async def test_consensus_with_validation(
        self, auto_decomposer, mock_task_decomposer
    ):
        """Should run validation on each plan during consensus."""
        goal = "Write unit tests"

        plan = self.create_mock_plan(goal, num_steps=6)

        mock_task_decomposer.decompose = AsyncMock(return_value=plan)

        winner, metadata = await auto_decomposer.decompose_with_consensus(
            goal=goal,
            heads=["mock-llm"],
            strategy=ConsensusStrategy.MAJORITY,
            auto_validate=True,  # Enable validation
        )

        # Plan should have validation metadata attached
        assert hasattr(winner, "_validation_issues")

    @pytest.mark.asyncio
    async def test_consensus_with_research_features(
        self, auto_decomposer, mock_task_decomposer
    ):
        """Should enable research features on winning plan."""
        goal = "Optimize database queries"

        plan = self.create_mock_plan(goal, num_steps=5)

        mock_task_decomposer.decompose = AsyncMock(return_value=plan)

        winner, metadata = await auto_decomposer.decompose_with_consensus(
            goal=goal,
            heads=["mock-llm"],
            strategy=ConsensusStrategy.MAJORITY,
            enable_research_features=True,
        )

        # Winner should have feature configs attached
        assert hasattr(winner, "_feature_configs")
