"""Tests for the distributed SolveCoordinator (fast unit tests only)."""

from unittest.mock import MagicMock, patch

import pytest

from multihead.solve import (
    SolveConfig,
    SolveCoordinator,
    SolveResult,
    discover_active_sessions,
    mark_session_offline,
    write_presence_claim,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def knowledge_store(tmp_path):
    """Real KnowledgeStore backed by temp SQLite DB."""
    from multihead.knowledge_store import KnowledgeStore
    return KnowledgeStore(tmp_path / "knowledge.db")


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
# Model tests
# ---------------------------------------------------------------------------


class TestSolveConfig:
    def test_defaults(self):
        cfg = SolveConfig()
        assert cfg.project_id == "multihead"
        assert cfg.session_id == "multihead-coordinator"
        assert cfg.proposal_timeout_seconds == 300.0
        assert cfg.min_proposals == 1
        assert cfg.max_proposals == 10
        assert cfg.auto_approve is False

    def test_custom(self):
        from multihead.consensus import ConsensusStrategy
        cfg = SolveConfig(
            project_id="custom",
            session_id="my-session",
            proposal_timeout_seconds=60.0,
            min_proposals=2,
            consensus_strategy=ConsensusStrategy.WEIGHTED,
            auto_approve=True,
        )
        assert cfg.project_id == "custom"
        assert cfg.min_proposals == 2
        assert cfg.consensus_strategy == ConsensusStrategy.WEIGHTED
        assert cfg.auto_approve is True


class TestSolveResult:
    def test_success(self):
        r = SolveResult(
            task="build it",
            request_id="req-1",
            proposals_received=3,
            winning_proposal_id="prop-2",
            assigned_agent="agent-a",
            execution_started=True,
            result_claim_id="result-1",
            success=True,
        )
        assert r.success is True
        assert r.proposals_received == 3
        assert r.error is None

    def test_failure(self):
        r = SolveResult(
            task="fail task",
            request_id="req-2",
            proposals_received=0,
            winning_proposal_id=None,
            assigned_agent=None,
            execution_started=False,
            result_claim_id=None,
            success=False,
            error="No proposals",
        )
        assert r.success is False
        assert r.error == "No proposals"


# ---------------------------------------------------------------------------
# Presence management
# ---------------------------------------------------------------------------


class TestPresenceClaims:
    def test_write_presence_claim(self, knowledge_store):
        """Writes a presence FACT claim."""
        claim_id = write_presence_claim(
            knowledge_store, "sess-1", "test-proj", ["llm", "vlm"],
        )
        assert claim_id
        # Should be retrievable
        claims = knowledge_store.list_claims(claim_type="fact", scope_id="test-proj")
        assert len(claims) >= 1
        found = [c for c in claims if c.canonical.predicate == "available"]
        assert len(found) == 1
        assert "sess-1" in found[0].canonical.claim_key

    def test_mark_session_offline(self, knowledge_store):
        """Marking offline writes a new claim with unique key."""
        write_presence_claim(knowledge_store, "sess-1", "test-proj")
        offline_id = mark_session_offline(knowledge_store, "sess-1", "test-proj")
        assert offline_id
        # Should have the offline claim
        claims = knowledge_store.list_claims(claim_type="fact", scope_id="test-proj")
        offline_claims = [c for c in claims if c.canonical.predicate == "offline"]
        assert len(offline_claims) == 1

    def test_mark_offline_without_presence(self, knowledge_store):
        """Marking offline without prior presence should still work."""
        offline_id = mark_session_offline(knowledge_store, "new-sess", "test-proj")
        assert offline_id


class TestSessionDiscovery:
    def test_discover_no_sessions(self, knowledge_store):
        """Empty DB returns no sessions."""
        sessions = discover_active_sessions(
            knowledge_store, "test-proj", "my-session",
        )
        assert sessions == []

    def test_discover_excludes_self(self, knowledge_store):
        """Own session is excluded from discovery."""
        write_presence_claim(knowledge_store, "my-session", "test-proj")
        sessions = discover_active_sessions(
            knowledge_store, "test-proj", "my-session",
        )
        assert len(sessions) == 0

    def test_discover_finds_other_sessions(self, knowledge_store):
        """Discovers other active sessions."""
        write_presence_claim(knowledge_store, "other-1", "test-proj", ["llm"])
        write_presence_claim(knowledge_store, "other-2", "test-proj", ["vlm"])
        sessions = discover_active_sessions(
            knowledge_store, "test-proj", "my-session",
        )
        assert len(sessions) == 2
        ids = {s["session_id"] for s in sessions}
        assert ids == {"other-1", "other-2"}

    def test_discover_filters_by_project(self, knowledge_store):
        """Only finds sessions in the same project."""
        write_presence_claim(knowledge_store, "other-1", "proj-a")
        write_presence_claim(knowledge_store, "other-2", "proj-b")
        sessions = discover_active_sessions(
            knowledge_store, "proj-a", "my-session",
        )
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "other-1"


# ---------------------------------------------------------------------------
# SolveCoordinator init & cleanup
# ---------------------------------------------------------------------------


class TestCoordinatorInit:
    def test_init_writes_presence(self, mock_ks, mock_hm, mock_orch, config):
        """Constructor writes a presence claim."""
        coord = _make_coordinator(mock_ks, mock_hm, mock_orch, config)
        mock_ks.insert_claim.assert_called_once()
        claim = mock_ks.insert_claim.call_args[0][0]
        assert claim.canonical.predicate == "available"
        assert "test-session" in claim.canonical.claim_key

    def test_cleanup_marks_offline(self, mock_ks, mock_hm, mock_orch, config):
        """_cleanup() marks session offline."""
        coord = _make_coordinator(mock_ks, mock_hm, mock_orch, config)
        mock_ks.reset_mock()
        coord._cleanup()
        mock_ks.insert_claim.assert_called_once()
        claim = mock_ks.insert_claim.call_args[0][0]
        assert claim.canonical.predicate == "offline"


# ---------------------------------------------------------------------------
# Internal methods
# ---------------------------------------------------------------------------


class TestPostTaskRequest:
    @pytest.mark.asyncio
    async def test_posts_question_claim(self, mock_ks, mock_hm, mock_orch, config):
        """_post_task_request creates a QUESTION claim."""
        coord = _make_coordinator(mock_ks, mock_hm, mock_orch, config)
        mock_ks.reset_mock()

        request_id = await coord._post_task_request("build a widget")

        assert request_id.startswith("clm_")
        mock_ks.insert_claim.assert_called_once()
        claim = mock_ks.insert_claim.call_args[0][0]
        assert claim.claim_type.value == "question"
        assert "build a widget" in claim.statement
        assert "test-session" in claim.statement

    @pytest.mark.asyncio
    async def test_claim_id_filled_in_statement(self, mock_ks, mock_hm, mock_orch, config):
        """Statement template has claim_id filled in."""
        coord = _make_coordinator(mock_ks, mock_hm, mock_orch, config)
        mock_ks.reset_mock()

        request_id = await coord._post_task_request("task")

        claim = mock_ks.insert_claim.call_args[0][0]
        # The claim_id should appear in the statement (template filled)
        assert request_id in claim.statement


class TestCollectProposals:
    @pytest.mark.asyncio
    async def test_returns_empty_on_timeout(self, mock_ks, mock_hm, mock_orch, config):
        """Returns [] when no proposals arrive before timeout."""
        config.proposal_timeout_seconds = 0.1
        coord = _make_coordinator(mock_ks, mock_hm, mock_orch, config)
        mock_ks.get_responses_to_claim = MagicMock(return_value=[])

        proposals = await coord._collect_proposals("req-1")

        assert proposals == []

    @pytest.mark.asyncio
    async def test_returns_proposals_found(self, mock_ks, mock_hm, mock_orch, config):
        """Returns proposals when found."""
        config.proposal_timeout_seconds = 0.1
        coord = _make_coordinator(mock_ks, mock_hm, mock_orch, config)

        mock_proposal = MagicMock()
        mock_ks.get_responses_to_claim = MagicMock(return_value=[mock_proposal])

        proposals = await coord._collect_proposals("req-1")

        assert len(proposals) == 1

    @pytest.mark.asyncio
    async def test_fallback_on_db_error(self, mock_ks, mock_hm, mock_orch, config):
        """Falls back to raw SQL when get_responses_to_claim raises."""
        config.proposal_timeout_seconds = 0.1
        coord = _make_coordinator(mock_ks, mock_hm, mock_orch, config)

        # Simulate DB error followed by successful fallback
        mock_ks.get_responses_to_claim = MagicMock(side_effect=ValueError("bad stability"))
        mock_ks.db_path = ":memory:"

        # The fallback does raw SQL — mock sqlite3
        with patch("sqlite3.connect") as mock_conn:
            mock_conn.return_value.row_factory = None
            mock_conn.return_value.execute.return_value.fetchall.return_value = []
            proposals = await coord._collect_proposals("req-1")

        assert proposals == []


class TestAssignWork:
    @pytest.mark.asyncio
    async def test_creates_decision_claim(self, mock_ks, mock_hm, mock_orch, config):
        """_assign_work creates a DECISION claim."""
        coord = _make_coordinator(mock_ks, mock_hm, mock_orch, config)
        mock_ks.reset_mock()

        mock_proposal = MagicMock()
        mock_proposal.claim_id = "prop-1"

        assignment_id = await coord._assign_work(mock_proposal, "agent-a")

        assert assignment_id.startswith("clm_")
        claim = mock_ks.insert_claim.call_args[0][0]
        assert claim.claim_type.value == "decision"
        assert "agent-a" in claim.statement
        assert claim.related_claim_ids == ["prop-1"]


class TestMonitorExecution:
    @pytest.mark.asyncio
    async def test_returns_result_claim_id(self, mock_ks, mock_hm, mock_orch, config):
        """Returns result claim_id when execution completes."""
        config.proposal_timeout_seconds = 0.1
        coord = _make_coordinator(mock_ks, mock_hm, mock_orch, config)

        mock_result = MagicMock()
        mock_result.claim_id = "result-1"
        mock_ks.get_responses_to_claim = MagicMock(return_value=[mock_result])

        result_id = await coord._monitor_execution("assign-1", "agent-a")

        assert result_id == "result-1"

    @pytest.mark.asyncio
    async def test_returns_none_on_timeout(self, mock_ks, mock_hm, mock_orch, config):
        """Returns None when monitoring times out."""
        config.proposal_timeout_seconds = 0.05  # Monitor timeout = 2x = 0.1s
        coord = _make_coordinator(mock_ks, mock_hm, mock_orch, config)
        mock_ks.get_responses_to_claim = MagicMock(return_value=[])

        result_id = await coord._monitor_execution("assign-1", "agent-a")

        assert result_id is None
