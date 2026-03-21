"""Slow tests for SolveCoordinator — tests that run the solve pipeline
(async polling with timeouts)."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from multihead.solve import (
    SolveConfig,
    SolveCoordinator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_ks():
    """Mock KnowledgeStore for unit tests."""
    ks = MagicMock()
    ks.insert_claim = MagicMock()
    ks.list_claims = MagicMock(return_value=[])
    ks.get_responses_to_claim = MagicMock(return_value=[])
    ks.get_claim = MagicMock(return_value=None)
    return ks


@pytest.fixture
def mock_hm():
    """Mock HeadManager."""
    return MagicMock()


@pytest.fixture
def mock_orch():
    """Mock Orchestrator."""
    return MagicMock()


@pytest.fixture
def config():
    """Default SolveConfig for tests."""
    return SolveConfig(
        project_id="test-project",
        session_id="test-session",
        proposal_timeout_seconds=5.0,  # Short for tests
        min_proposals=1,
        max_proposals=5,
    )


def _make_coordinator(mock_ks, mock_hm, mock_orch, config):
    """Create a SolveCoordinator with mocked session state."""
    with patch("multihead.solve._load_seen_sessions", return_value=set()), \
         patch("multihead.solve._save_seen_sessions"), \
         patch("multihead.solve.signal.signal"), \
         patch("multihead.solve.atexit.register"):
        return SolveCoordinator(
            knowledge_store=mock_ks,
            head_manager=mock_hm,
            orchestrator=mock_orch,
            config=config,
        )


# ---------------------------------------------------------------------------
# Full solve flow
# ---------------------------------------------------------------------------


class TestSolveFlow:
    @pytest.mark.asyncio
    async def test_no_proposals_returns_failure(self, mock_ks, mock_hm, mock_orch, config):
        """When no proposals arrive in multi-agent mode, returns success=False."""
        config.proposal_timeout_seconds = 0.1  # Very short
        coord = _make_coordinator(mock_ks, mock_hm, mock_orch, config)

        # Force multi-agent mode: other sessions exist and user chooses to wait
        fake_session = [{
            "session_id": "other-1", "capabilities": [],
            "last_seen": datetime.now(timezone.utc),
        }]
        with patch(
            "multihead.solve.coordinator.discover_active_sessions",
            return_value=fake_session), \
             patch("multihead.solve.coordinator.prompt_multi_session", return_value=False), \
             patch("multihead.solve.coordinator._load_seen_sessions", return_value=set()), \
             patch("multihead.solve.coordinator._save_seen_sessions"), \
             patch("multihead.solve.coordinator._show_onboarding_messages"):
            result = await coord.solve("build a thing")

        assert result.success is False
        assert result.proposals_received == 0
        assert "No proposals" in result.error

    @pytest.mark.asyncio
    async def test_single_proposal_skips_voting(self, mock_ks, mock_hm, mock_orch, config):
        """Single proposal is auto-selected without consensus vote (multi-agent mode)."""
        from multihead.knowledge_models import (
            Claim, ClaimCanonical, ClaimScope, ClaimStatus, ClaimType,
            EntityRef, Provenance, ScopeType, ValueObject,
        )

        # Create a mock proposal claim
        proposal = Claim(
            claim_type=ClaimType.FACT,
            claim_status=ClaimStatus.PROPOSED,
            scope=ClaimScope(scope_type=ScopeType.PROJECT, scope_id="test-project",
                             visibility="project", valid_from=datetime.now(timezone.utc)),
            canonical=ClaimCanonical(
                claim_key="proposal.1",
                subject=EntityRef(entity_type="task", entity_id="t1"),
                predicate="proposes",
                object=ValueObject(value_type="string", value="my plan"),
            ),
            statement="I propose: step 1, step 2",
            confidence=0.9,
            provenance=Provenance(produced_by={"id": "agent-a", "method": "decompose"}),
        )

        config.proposal_timeout_seconds = 0.1
        coord = _make_coordinator(mock_ks, mock_hm, mock_orch, config)

        mock_ks.get_responses_to_claim = MagicMock(return_value=[proposal])

        # Force multi-agent mode: other sessions exist and user chooses to wait
        fake_session = [{
            "session_id": "other-1", "capabilities": [],
            "last_seen": datetime.now(timezone.utc),
        }]
        with patch(
            "multihead.solve.coordinator.discover_active_sessions",
            return_value=fake_session), \
             patch("multihead.solve.coordinator.prompt_multi_session", return_value=False), \
             patch("multihead.solve.coordinator._load_seen_sessions", return_value=set()), \
             patch("multihead.solve.coordinator._save_seen_sessions"), \
             patch("multihead.solve.coordinator._show_onboarding_messages"):
            result = await coord.solve("build a thing")

        assert result.success is True
        assert result.proposals_received == 1
        assert result.winning_proposal_id == proposal.claim_id
        assert result.assigned_agent == "agent-a"

    @pytest.mark.asyncio
    async def test_multiple_proposals_triggers_consensus(self, mock_ks, mock_hm, mock_orch, config):
        """Multiple proposals triggers consensus voting (multi-agent mode)."""
        from multihead.knowledge_models import (
            Claim, ClaimCanonical, ClaimScope, ClaimStatus, ClaimType,
            EntityRef, Provenance, ScopeType, ValueObject,
        )

        proposals = []
        for i, agent in enumerate(["agent-a", "agent-b"]):
            p = Claim(
                claim_type=ClaimType.FACT,
                claim_status=ClaimStatus.PROPOSED,
                scope=ClaimScope(scope_type=ScopeType.PROJECT, scope_id="test-project",
                                 visibility="project", valid_from=datetime.now(timezone.utc)),
                canonical=ClaimCanonical(
                    claim_key=f"proposal.{i}",
                    subject=EntityRef(entity_type="task", entity_id="t1"),
                    predicate="proposes",
                    object=ValueObject(value_type="string", value=f"plan {i}"),
                ),
                statement=f"Plan from {agent}",
                confidence=0.8 + i * 0.1,
                provenance=Provenance(produced_by={"id": agent, "method": "decompose"}),
            )
            proposals.append(p)

        config.proposal_timeout_seconds = 0.1
        coord = _make_coordinator(mock_ks, mock_hm, mock_orch, config)
        mock_ks.get_responses_to_claim = MagicMock(return_value=proposals)

        # Force multi-agent mode: other sessions exist and user chooses to wait
        fake_session = [{
            "session_id": "other-1", "capabilities": [],
            "last_seen": datetime.now(timezone.utc),
        }]
        with patch(
            "multihead.solve.coordinator.discover_active_sessions",
            return_value=fake_session), \
             patch("multihead.solve.coordinator.prompt_multi_session", return_value=False), \
             patch("multihead.solve.coordinator._load_seen_sessions", return_value=set()), \
             patch("multihead.solve.coordinator._save_seen_sessions"), \
             patch("multihead.solve.coordinator._show_onboarding_messages"), \
             patch("multihead.solve.coordinator.ConsensusEngine") as MockEngine:

            # Mock consensus to return first proposal as winner
            mock_result = MagicMock()
            mock_result.consensus_outputs = {"text": proposals[0].statement}
            mock_result.agreement_score = 0.9
            mock_result.strategy_used = config.consensus_strategy
            MockEngine.return_value.rank_proposals = MagicMock(return_value=mock_result)

            result = await coord.solve("task")

        assert result.success is True
        assert result.proposals_received == 2
        assert result.winning_proposal_id == proposals[0].claim_id

    @pytest.mark.asyncio
    async def test_solo_mode_auto_approves(self, mock_ks, mock_hm, mock_orch, config):
        """Solo mode (no other sessions) self-proposes and returns plan to caller."""
        config.proposal_timeout_seconds = 0.1
        coord = _make_coordinator(mock_ks, mock_hm, mock_orch, config)

        # Mock search_claims_fts used by _self_propose for RAG context
        mock_ks.search_claims_fts = MagicMock(return_value=[])

        with patch("multihead.solve.coordinator.discover_active_sessions", return_value=[]), \
             patch("multihead.solve.coordinator._load_seen_sessions", return_value=set()), \
             patch("multihead.solve.coordinator._save_seen_sessions"), \
             patch("multihead.solve.coordinator._show_onboarding_messages"):
            result = await coord.solve("solo task")

        assert result.success is True
        # Solo mode: caller is the executor, so execution_started is False
        assert result.execution_started is False
        assert coord.config.auto_approve is True
        assert result.proposals_received == 1
        assert result.decomposition is not None
        assert "solo task" in result.decomposition

    @pytest.mark.asyncio
    async def test_explicit_auto_approve_overrides(self, mock_ks, mock_hm, mock_orch, config):
        """explicit_auto_approve=False overrides solo detection."""
        config.proposal_timeout_seconds = 0.1

        with patch("multihead.solve._load_seen_sessions", return_value=set()), \
             patch("multihead.solve._save_seen_sessions"), \
             patch("multihead.solve.signal.signal"), \
             patch("multihead.solve.atexit.register"):
            coord = SolveCoordinator(
                knowledge_store=mock_ks,
                head_manager=mock_hm,
                orchestrator=mock_orch,
                config=config,
                explicit_auto_approve=False,
            )

        with patch("multihead.solve.coordinator.discover_active_sessions", return_value=[]), \
             patch("multihead.solve.coordinator._load_seen_sessions", return_value=set()), \
             patch("multihead.solve.coordinator._save_seen_sessions"), \
             patch("multihead.solve.coordinator._show_onboarding_messages"):
            result = await coord.solve("task")

        assert result.execution_started is False
        assert coord.config.auto_approve is False

    @pytest.mark.asyncio
    async def test_consensus_fallback_first_proposal(self, mock_ks, mock_hm, mock_orch, config):
        """When consensus match fails, falls back to first proposal (multi-agent mode)."""
        from multihead.knowledge_models import (
            Claim, ClaimCanonical, ClaimScope, ClaimStatus, ClaimType,
            EntityRef, Provenance, ScopeType, ValueObject,
        )

        proposals = []
        for i in range(2):
            p = Claim(
                claim_type=ClaimType.FACT,
                claim_status=ClaimStatus.PROPOSED,
                scope=ClaimScope(scope_type=ScopeType.PROJECT, scope_id="test-project",
                                 visibility="project", valid_from=datetime.now(timezone.utc)),
                canonical=ClaimCanonical(
                    claim_key=f"proposal.{i}",
                    subject=EntityRef(entity_type="task", entity_id="t1"),
                    predicate="proposes",
                    object=ValueObject(value_type="string", value=f"plan {i}"),
                ),
                statement=f"Plan {i}",
                confidence=0.8,
                provenance=Provenance(produced_by={"id": f"agent-{i}", "method": "decompose"}),
            )
            proposals.append(p)

        config.proposal_timeout_seconds = 0.1
        coord = _make_coordinator(mock_ks, mock_hm, mock_orch, config)
        mock_ks.get_responses_to_claim = MagicMock(return_value=proposals)

        # Force multi-agent mode: other sessions exist and user chooses to wait
        fake_session = [{
            "session_id": "other-1", "capabilities": [],
            "last_seen": datetime.now(timezone.utc),
        }]
        with patch(
            "multihead.solve.coordinator.discover_active_sessions",
            return_value=fake_session), \
             patch("multihead.solve.coordinator.prompt_multi_session", return_value=False), \
             patch("multihead.solve.coordinator._load_seen_sessions", return_value=set()), \
             patch("multihead.solve.coordinator._save_seen_sessions"), \
             patch("multihead.solve.coordinator._show_onboarding_messages"), \
             patch("multihead.solve.coordinator.ConsensusEngine") as MockEngine:

            # Consensus returns text that doesn't match any proposal
            mock_result = MagicMock()
            mock_result.consensus_outputs = {"text": "non-matching text"}
            mock_result.agreement_score = 0.5
            mock_result.strategy_used = config.consensus_strategy
            MockEngine.return_value.rank_proposals = MagicMock(return_value=mock_result)

            result = await coord.solve("task")

        assert result.success is True
        # Fallback to first proposal
        assert result.winning_proposal_id == proposals[0].claim_id
