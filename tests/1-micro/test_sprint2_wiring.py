"""Tests for Sprint 2: ConsensusEngine.rank_proposals() and AgentExecutor execution wiring."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from multihead.consensus import ConsensusEngine, ConsensusStrategy, VoteResult
from multihead.head_manager import HeadManager
from multihead.models import AdapterKind, HeadManifest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_head_manager():
    """Minimal HeadManager with mock heads."""
    manifests = {
        "mock-llm": HeadManifest(
            head_id="mock-llm", name="Mock LLM", adapter=AdapterKind.MOCK,
            model="mock-v1", kind="llm", gpu_required=False,
        ),
    }
    return HeadManager(manifests)


@pytest.fixture
def engine(mock_head_manager):
    return ConsensusEngine(mock_head_manager)


# ---------------------------------------------------------------------------
# Tests: ConsensusEngine.rank_proposals()
# ---------------------------------------------------------------------------


class TestRankProposalsEmpty:
    def test_no_votes_returns_red_flag(self, engine):
        result = engine.rank_proposals(votes=[])
        assert len(result.red_flags) == 1
        assert result.red_flags[0]["type"] == "no_valid_votes"
        assert result.agreement_score == 0.0

    def test_all_failed_votes_returns_red_flag(self, engine):
        votes = [
            VoteResult(head_id="a", success=False, error="timeout"),
            VoteResult(head_id="b", success=False, error="crash"),
        ]
        result = engine.rank_proposals(votes=votes)
        assert any(f["type"] == "no_valid_votes" for f in result.red_flags)


class TestRankProposalsMajority:
    def test_single_vote_wins(self, engine):
        votes = [VoteResult(head_id="a", outputs={"text": "plan A"})]
        result = engine.rank_proposals(votes=votes, strategy=ConsensusStrategy.MAJORITY)
        assert result.consensus_outputs["text"] == "plan A"
        assert result.agreement_score == 1.0

    def test_majority_picks_most_common(self, engine):
        votes = [
            VoteResult(head_id="a", outputs={"text": "plan A"}),
            VoteResult(head_id="b", outputs={"text": "plan B"}),
            VoteResult(head_id="c", outputs={"text": "plan A"}),
        ]
        result = engine.rank_proposals(votes=votes, strategy=ConsensusStrategy.MAJORITY)
        assert result.consensus_outputs["text"] == "plan A"
        assert result.agreement_score == pytest.approx(2 / 3)

    def test_tie_picks_first(self, engine):
        votes = [
            VoteResult(head_id="a", outputs={"text": "plan A"}),
            VoteResult(head_id="b", outputs={"text": "plan B"}),
        ]
        result = engine.rank_proposals(votes=votes, strategy=ConsensusStrategy.MAJORITY)
        # Counter.most_common picks one; either is valid
        assert result.consensus_outputs["text"] in ("plan A", "plan B")
        assert result.agreement_score == 0.5


class TestRankProposalsWeighted:
    def test_weighted_higher_weight_wins(self, engine):
        votes = [
            VoteResult(head_id="a", outputs={"text": "plan A"}),
            VoteResult(head_id="b", outputs={"text": "plan B"}),
        ]
        weights = {"a": 0.3, "b": 0.9}
        result = engine.rank_proposals(
            votes=votes, strategy=ConsensusStrategy.WEIGHTED, weights=weights,
        )
        assert result.consensus_outputs["text"] == "plan B"

    def test_weighted_no_weights_defaults(self, engine):
        votes = [
            VoteResult(head_id="a", outputs={"text": "same"}),
            VoteResult(head_id="b", outputs={"text": "same"}),
        ]
        result = engine.rank_proposals(
            votes=votes, strategy=ConsensusStrategy.WEIGHTED,
        )
        assert result.consensus_outputs["text"] == "same"
        assert result.agreement_score == 1.0


class TestRankProposalsUnanimous:
    def test_unanimous_all_agree(self, engine):
        votes = [
            VoteResult(head_id="a", outputs={"text": "same"}),
            VoteResult(head_id="b", outputs={"text": "same"}),
        ]
        result = engine.rank_proposals(
            votes=votes, strategy=ConsensusStrategy.UNANIMOUS,
        )
        assert result.agreement_score == 1.0

    def test_unanimous_disagree(self, engine):
        votes = [
            VoteResult(head_id="a", outputs={"text": "plan A"}),
            VoteResult(head_id="b", outputs={"text": "plan B"}),
        ]
        result = engine.rank_proposals(
            votes=votes, strategy=ConsensusStrategy.UNANIMOUS,
        )
        assert result.agreement_score == 0.5


class TestRankProposalsFTA:
    def test_fta_single_bucket(self, engine):
        votes = [
            VoteResult(head_id="a", outputs={"text": "plan A"}),
            VoteResult(head_id="b", outputs={"text": "plan A"}),
        ]
        result = engine.rank_proposals(
            votes=votes, strategy=ConsensusStrategy.FIRST_TO_AHEAD,
        )
        assert result.consensus_outputs["text"] == "plan A"
        assert result.agreement_score == 1.0
        assert result.strategy_used == ConsensusStrategy.FIRST_TO_AHEAD

    def test_fta_two_buckets(self, engine):
        votes = [
            VoteResult(head_id="a", outputs={"text": "plan A"}),
            VoteResult(head_id="b", outputs={"text": "plan B"}),
            VoteResult(head_id="c", outputs={"text": "plan A"}),
        ]
        result = engine.rank_proposals(
            votes=votes, strategy=ConsensusStrategy.FIRST_TO_AHEAD,
        )
        assert result.consensus_outputs["text"] == "plan A"
        assert result.metrics["unique_buckets"] == 2
        assert result.metrics["leader_count"] == 2


class TestRankProposalsMetrics:
    def test_metrics_include_mode(self, engine):
        votes = [VoteResult(head_id="a", outputs={"text": "plan"})]
        result = engine.rank_proposals(votes=votes)
        assert result.metrics["mode"] == "rank_proposals"
        assert result.metrics["total_proposals"] == 1
        assert result.metrics["valid_proposals"] == 1

    def test_strategy_used_set(self, engine):
        votes = [VoteResult(head_id="a", outputs={"text": "plan"})]
        result = engine.rank_proposals(
            votes=votes, strategy=ConsensusStrategy.WEIGHTED,
        )
        assert result.strategy_used == ConsensusStrategy.WEIGHTED


# ---------------------------------------------------------------------------
# Tests: SolveCoordinator._vote_on_proposals
# ---------------------------------------------------------------------------


class TestSolveCoordinatorVoting:
    """Test that SolveCoordinator.vote uses ConsensusEngine properly."""

    def _make_proposal_claim(self, agent_id: str, statement: str, confidence: float = 0.85):
        """Build a minimal Claim-like object for testing."""
        from multihead.knowledge_models import (
            Claim, ClaimCanonical, ClaimScope, ClaimStatus, ClaimType,
            EntityRef, Provenance, ScopeType, ValueObject,
        )
        return Claim(
            claim_id=f"clm_test_{agent_id}",
            claim_type=ClaimType.FACT,
            claim_status=ClaimStatus.ACCEPTED,
            scope=ClaimScope(
                scope_type=ScopeType.PROJECT,
                scope_id="test",
                visibility="project",
                valid_from=datetime.now(timezone.utc),
            ),
            canonical=ClaimCanonical(
                claim_key=f"solve.proposal.{agent_id}",
                subject=EntityRef(entity_type="proposal", entity_id=f"prop_{agent_id}"),
                predicate="proposes_solution",
                object=ValueObject(value_type="string", value="test task"),
            ),
            statement=statement,
            confidence=confidence,
            provenance=Provenance(produced_by={"id": agent_id, "method": "test"}),
        )

    @pytest.mark.asyncio
    async def test_single_proposal_returns_it(self, mock_head_manager):
        """Single proposal should be returned without voting."""
        from multihead.solve import SolveCoordinator, SolveConfig
        from multihead.knowledge_store import KnowledgeStore

        tmpdir = tempfile.mkdtemp()
        ks = KnowledgeStore(Path(tmpdir) / "test.db")
        orchestrator = MagicMock()

        config = SolveConfig(project_id="test", session_id="test-coord")
        coord = SolveCoordinator(ks, mock_head_manager, orchestrator, config)

        proposal = self._make_proposal_claim("agent-1", "My plan: do X")
        # Single proposal skips voting in solve(), but test _vote directly
        result = await coord._vote_on_proposals([proposal])
        assert result.claim_id == proposal.claim_id

    @pytest.mark.asyncio
    async def test_multiple_proposals_votes(self, mock_head_manager):
        """Multiple identical proposals should agree."""
        from multihead.solve import SolveCoordinator, SolveConfig
        from multihead.knowledge_store import KnowledgeStore

        tmpdir = tempfile.mkdtemp()
        ks = KnowledgeStore(Path(tmpdir) / "test.db")
        orchestrator = MagicMock()

        config = SolveConfig(
            project_id="test", session_id="test-coord",
            consensus_strategy=ConsensusStrategy.MAJORITY,
        )
        coord = SolveCoordinator(ks, mock_head_manager, orchestrator, config)

        p1 = self._make_proposal_claim("agent-1", "Same plan")
        p2 = self._make_proposal_claim("agent-2", "Same plan")
        p3 = self._make_proposal_claim("agent-3", "Different plan")

        result = await coord._vote_on_proposals([p1, p2, p3])
        # Majority should pick "Same plan" (2 vs 1)
        assert result.statement == "Same plan"

    @pytest.mark.asyncio
    async def test_weighted_strategy_uses_confidence(self, mock_head_manager):
        """Weighted strategy should prefer higher-confidence proposals."""
        from multihead.solve import SolveCoordinator, SolveConfig
        from multihead.knowledge_store import KnowledgeStore

        tmpdir = tempfile.mkdtemp()
        ks = KnowledgeStore(Path(tmpdir) / "test.db")
        orchestrator = MagicMock()

        config = SolveConfig(
            project_id="test", session_id="test-coord",
            consensus_strategy=ConsensusStrategy.WEIGHTED,
        )
        coord = SolveCoordinator(ks, mock_head_manager, orchestrator, config)

        p1 = self._make_proposal_claim("agent-1", "Low confidence plan", confidence=0.3)
        p2 = self._make_proposal_claim("agent-2", "High confidence plan", confidence=0.95)

        result = await coord._vote_on_proposals([p1, p2])
        assert result.statement == "High confidence plan"


# ---------------------------------------------------------------------------
# Tests: AgentExecutor work order extraction
# ---------------------------------------------------------------------------


class TestAgentExecutorWorkOrder:
    def _make_executor(self):
        from multihead.agent_executor import AgentExecutor
        from multihead.knowledge_store import KnowledgeStore

        tmpdir = tempfile.mkdtemp()
        ks = KnowledgeStore(Path(tmpdir) / "test.db")
        hm = MagicMock()
        hm.get_states.return_value = {}
        return AgentExecutor(ks, hm, "test-agent"), ks

    def test_extract_work_order_from_json_value(self):
        """Work order stored in canonical.object.value should be extractable."""
        from multihead.knowledge_models import (
            Claim, ClaimCanonical, ClaimScope, ClaimStatus, ClaimType,
            EntityRef, Provenance, ScopeType, ValueObject,
        )

        executor, _ = self._make_executor()

        wo_data = {
            "goal": "Test goal",
            "steps": [
                {"step_id": "step_1", "goal": "Explore code", "action_type": "explore"},
                {"step_id": "step_2", "goal": "Implement fix", "action_type": "implement"},
            ],
        }

        proposal = Claim(
            claim_id="clm_test_prop",
            claim_type=ClaimType.FACT,
            claim_status=ClaimStatus.ACCEPTED,
            scope=ClaimScope(
                scope_type=ScopeType.PROJECT, scope_id="test",
                visibility="project", valid_from=datetime.now(timezone.utc),
            ),
            canonical=ClaimCanonical(
                claim_key="solve.proposal.test",
                subject=EntityRef(entity_type="proposal", entity_id="prop_test"),
                predicate="proposes_solution",
                object=ValueObject(
                    value_type="json",
                    value={"task": "Fix the bug", "work_order": wo_data},
                ),
            ),
            statement="TASK PROPOSAL",
            confidence=0.85,
            provenance=Provenance(produced_by={"id": "test-agent"}),
        )

        plan = executor._extract_work_order(proposal)
        assert plan is not None
        assert plan["goal"] == "Test goal"
        assert len(plan["phases"]) == 1
        assert len(plan["phases"][0]["children"]) == 2
        assert plan["phases"][0]["children"][0]["action_type"] == "explore"

    def test_extract_work_order_with_phases(self):
        """Work order already in plan format (with phases key) passes through."""
        from multihead.knowledge_models import (
            Claim, ClaimCanonical, ClaimScope, ClaimStatus, ClaimType,
            EntityRef, Provenance, ScopeType, ValueObject,
        )

        executor, _ = self._make_executor()

        plan_data = {
            "goal": "Test",
            "phases": [{"id": "p1", "goal": "Phase 1", "children": []}],
        }

        proposal = Claim(
            claim_id="clm_test_plan",
            claim_type=ClaimType.FACT,
            claim_status=ClaimStatus.ACCEPTED,
            scope=ClaimScope(
                scope_type=ScopeType.PROJECT, scope_id="test",
                visibility="project", valid_from=datetime.now(timezone.utc),
            ),
            canonical=ClaimCanonical(
                claim_key="solve.proposal.test",
                subject=EntityRef(entity_type="proposal", entity_id="prop_test"),
                predicate="proposes_solution",
                object=ValueObject(
                    value_type="json",
                    value={"task": "Test", "work_order": plan_data},
                ),
            ),
            statement="TASK PROPOSAL",
            confidence=0.85,
            provenance=Provenance(produced_by={"id": "test-agent"}),
        )

        plan = executor._extract_work_order(proposal)
        assert plan is not None
        assert "phases" in plan
        assert plan["phases"][0]["id"] == "p1"

    def test_extract_work_order_returns_none_for_old_format(self):
        """Old proposals without work_order in value return None."""
        from multihead.knowledge_models import (
            Claim, ClaimCanonical, ClaimScope, ClaimStatus, ClaimType,
            EntityRef, Provenance, ScopeType, ValueObject,
        )

        executor, _ = self._make_executor()

        proposal = Claim(
            claim_id="clm_old",
            claim_type=ClaimType.FACT,
            claim_status=ClaimStatus.ACCEPTED,
            scope=ClaimScope(
                scope_type=ScopeType.PROJECT, scope_id="test",
                visibility="project", valid_from=datetime.now(timezone.utc),
            ),
            canonical=ClaimCanonical(
                claim_key="solve.proposal.old",
                subject=EntityRef(entity_type="proposal", entity_id="prop_old"),
                predicate="proposes_solution",
                object=ValueObject(value_type="string", value="just a task string"),
            ),
            statement="TASK PROPOSAL",
            confidence=0.85,
            provenance=Provenance(produced_by={"id": "test-agent"}),
        )

        plan = executor._extract_work_order(proposal)
        assert plan is None

    def test_extract_task_from_proposal_json(self):
        """Task extracted from JSON object value."""
        from multihead.knowledge_models import (
            Claim, ClaimCanonical, ClaimScope, ClaimStatus, ClaimType,
            EntityRef, Provenance, ScopeType, ValueObject,
        )

        executor, _ = self._make_executor()

        proposal = Claim(
            claim_id="clm_t",
            claim_type=ClaimType.FACT,
            claim_status=ClaimStatus.ACCEPTED,
            scope=ClaimScope(
                scope_type=ScopeType.PROJECT, scope_id="test",
                visibility="project", valid_from=datetime.now(timezone.utc),
            ),
            canonical=ClaimCanonical(
                claim_key="solve.proposal.t",
                subject=EntityRef(entity_type="proposal", entity_id="prop_t"),
                predicate="proposes_solution",
                object=ValueObject(
                    value_type="json",
                    value={"task": "Fix the authentication bug", "work_order": {}},
                ),
            ),
            statement="TASK PROPOSAL",
            confidence=0.85,
            provenance=Provenance(produced_by={"id": "test-agent"}),
        )

        task = executor._extract_task_from_proposal(proposal)
        assert task == "Fix the authentication bug"


class TestAgentExecutorExecution:
    """Test execute_assignment end-to-end with mocked executor."""

    @pytest.mark.asyncio
    async def test_execute_assignment_calls_autonomous_executor(self):
        """execute_assignment should use AutonomousExecutor instead of placeholder."""
        from multihead.agent_executor import AgentExecutor
        from multihead.autonomous_executor import ExecutionReport
        from multihead.knowledge_store import KnowledgeStore
        from multihead.knowledge_models import (
            Claim, ClaimCanonical, ClaimScope, ClaimStatus, ClaimType,
            EntityRef, Provenance, ScopeType, ValueObject,
        )

        tmpdir = tempfile.mkdtemp()
        ks = KnowledgeStore(Path(tmpdir) / "test.db")
        hm = MagicMock()
        hm.get_states.return_value = {}

        executor = AgentExecutor(ks, hm, "test-agent")

        # Create proposal with work order
        wo_data = {
            "goal": "Test",
            "steps": [
                {"step_id": "s1", "goal": "Do thing", "action_type": "explore"},
            ],
        }
        proposal = Claim(
            claim_id="clm_proposal_123",
            claim_type=ClaimType.FACT,
            claim_status=ClaimStatus.ACCEPTED,
            scope=ClaimScope(
                scope_type=ScopeType.PROJECT, scope_id="test",
                visibility="project", valid_from=datetime.now(timezone.utc),
            ),
            canonical=ClaimCanonical(
                claim_key="solve.proposal.test",
                subject=EntityRef(entity_type="proposal", entity_id="prop_test"),
                predicate="proposes_solution",
                object=ValueObject(
                    value_type="json",
                    value={"task": "Test task", "work_order": wo_data},
                ),
            ),
            statement="TASK PROPOSAL",
            confidence=0.85,
            provenance=Provenance(produced_by={"id": "test-agent"}),
        )
        ks.insert_claim(proposal)

        # Create assignment
        assignment = Claim(
            claim_id="clm_assignment_456",
            claim_type=ClaimType.DECISION,
            claim_status=ClaimStatus.ACCEPTED,
            scope=ClaimScope(
                scope_type=ScopeType.PROJECT, scope_id="test",
                visibility="project", valid_from=datetime.now(timezone.utc),
            ),
            canonical=ClaimCanonical(
                claim_key="solve.assignment.test",
                subject=EntityRef(entity_type="assignment", entity_id="assign_test"),
                predicate="assigned_to",
                object=ValueObject(value_type="string", value="test-agent"),
            ),
            statement="WORK ASSIGNMENT\n\nFROM: coord\nTO: test-agent\nPROPOSAL: clm_proposal_123",
            confidence=0.95,
            provenance=Provenance(produced_by={"id": "coord"}),
            related_claim_ids=["clm_proposal_123"],
        )

        # Mock AutonomousExecutor.execute
        mock_report = ExecutionReport(
            goal="Test task",
            strategy="LocalLLMStrategy",
            total_steps=1,
            completed_steps=1,
            failed_steps=0,
            skipped_steps=0,
            total_cost_usd=0.0,
            total_duration_secs=1.5,
        )

        with patch(
            "multihead.agent_executor.AutonomousExecutor.execute",
            new_callable=AsyncMock,
            return_value=mock_report,
        ):
            result_id = await executor.execute_assignment(assignment)

        assert result_id.startswith("clm_")

        # Verify a result claim was posted
        result_claim = ks.get_claim(result_id)
        assert result_claim is not None
        assert "SUCCESS" in result_claim.statement

    @pytest.mark.asyncio
    async def test_execute_assignment_handles_missing_work_order(self):
        """Old-format proposals without work_order should post error result."""
        from multihead.agent_executor import AgentExecutor
        from multihead.knowledge_store import KnowledgeStore
        from multihead.knowledge_models import (
            Claim, ClaimCanonical, ClaimScope, ClaimStatus, ClaimType,
            EntityRef, Provenance, ScopeType, ValueObject,
        )

        tmpdir = tempfile.mkdtemp()
        ks = KnowledgeStore(Path(tmpdir) / "test.db")
        hm = MagicMock()
        hm.get_states.return_value = {}

        executor = AgentExecutor(ks, hm, "test-agent")

        # Old-format proposal without work_order
        proposal = Claim(
            claim_id="clm_old_prop",
            claim_type=ClaimType.FACT,
            claim_status=ClaimStatus.ACCEPTED,
            scope=ClaimScope(
                scope_type=ScopeType.PROJECT, scope_id="test",
                visibility="project", valid_from=datetime.now(timezone.utc),
            ),
            canonical=ClaimCanonical(
                claim_key="solve.proposal.old",
                subject=EntityRef(entity_type="proposal", entity_id="prop_old"),
                predicate="proposes_solution",
                object=ValueObject(value_type="string", value="just text"),
            ),
            statement="TASK PROPOSAL\nTASK: old task",
            confidence=0.85,
            provenance=Provenance(produced_by={"id": "test-agent"}),
        )
        ks.insert_claim(proposal)

        assignment = Claim(
            claim_id="clm_assign_old",
            claim_type=ClaimType.DECISION,
            claim_status=ClaimStatus.ACCEPTED,
            scope=ClaimScope(
                scope_type=ScopeType.PROJECT, scope_id="test",
                visibility="project", valid_from=datetime.now(timezone.utc),
            ),
            canonical=ClaimCanonical(
                claim_key="solve.assignment.old",
                subject=EntityRef(entity_type="assignment", entity_id="assign_old"),
                predicate="assigned_to",
                object=ValueObject(value_type="string", value="test-agent"),
            ),
            statement="WORK ASSIGNMENT\n\nFROM: coord\nTO: test-agent\nPROPOSAL: clm_old_prop",
            confidence=0.95,
            provenance=Provenance(produced_by={"id": "coord"}),
            related_claim_ids=["clm_old_prop"],
        )

        result_id = await executor.execute_assignment(assignment)
        result_claim = ks.get_claim(result_id)
        assert "ERROR" in result_claim.statement


# ---------------------------------------------------------------------------
# Tests: CLI --strategy flag
# ---------------------------------------------------------------------------


class TestSolveCLIStrategyFlag:
    def test_strategy_param_exists_on_distributed_solve(self):
        """The distributed solve command should accept --strategy."""
        from multihead.cli import main

        # Get the solve command (last registered wins)
        solve_cmd = main.commands.get("solve")
        assert solve_cmd is not None

        param_names = [p.name for p in solve_cmd.params]
        assert "strategy" in param_names

    def test_strategy_choices_valid(self):
        """The strategy param should have the right choices."""
        import click
        from multihead.cli import main

        solve_cmd = main.commands.get("solve")
        for param in solve_cmd.params:
            if param.name == "strategy":
                assert isinstance(param.type, click.Choice)
                assert "majority" in param.type.choices
                assert "weighted" in param.type.choices
                assert "first_to_ahead" in param.type.choices
                break
        else:
            pytest.fail("No strategy param found")
