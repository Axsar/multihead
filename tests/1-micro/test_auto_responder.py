"""Tests for auto-responder cross-session collaboration."""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch

from multihead.knowledge_store import KnowledgeStore
from multihead.knowledge_models import (
    Claim,
    ClaimStatus,
    ClaimType,
    ClaimScope,
    ScopeType,
    ClaimCanonical,
    EntityRef,
    ValueObject,
    Provenance,
)
from multihead import auto_responder


@pytest.fixture
def knowledge_store(tmp_path):
    """Create a temporary knowledge store."""
    db_path = tmp_path / "test_knowledge.db"
    return KnowledgeStore(db_path)


@pytest.fixture
def sample_request():
    """Create a sample decomposition request."""
    return Claim(
        claim_id="clm_req_001",
        claim_status=ClaimStatus.PROPOSED,
        claim_type=ClaimType.QUESTION,
        scope=ClaimScope(
            scope_type=ScopeType.PROJECT,
            scope_id="multihead",
            visibility="project",
            valid_from=datetime.now(timezone.utc),
        ),
        canonical=ClaimCanonical(
            claim_key="test.decomposition.request",
            subject=EntityRef(entity_type="task", entity_id="task_1", label="Web Scraper"),
            predicate="needs_decomposition",
            object=ValueObject(value_type="string", value="Build a web scraper"),
        ),
        statement="Need help decomposing: Build a web scraper for product prices",
        rationale="Request for cross-session collaboration",
        confidence=0.9,
        provenance=Provenance(
            produced_by={"id": "session-a", "method": "manual"},
        ),
    )


@pytest.fixture(autouse=True)
def reset_dedup():
    """Reset the dedup set before each test."""
    auto_responder._responded_request_ids.clear()


@pytest.mark.asyncio
async def test_check_for_pending_messages_no_messages(knowledge_store):
    """Test check when no pending messages exist."""
    count = await auto_responder.check_for_pending_messages(
        knowledge_store,
        session_id="session-b",
        project_id="multihead",
    )

    assert count == 0


@pytest.mark.asyncio
async def test_check_for_pending_messages_finds_messages(knowledge_store, sample_request):
    """Test check finds pending messages."""
    knowledge_store.insert_claim(sample_request)

    count = await auto_responder.check_for_pending_messages(
        knowledge_store,
        session_id="session-b",
        project_id="multihead",
    )

    assert count == 1


@pytest.mark.asyncio
async def test_check_for_pending_messages_timeout():
    """Test that check respects timeout and doesn't block."""
    # Create mock that hangs
    mock_ks = Mock()
    mock_ks.get_pending_messages = Mock(side_effect=lambda *args, **kwargs: asyncio.sleep(10))

    # Should timeout at 2s and return 0
    start = asyncio.get_event_loop().time()
    count = await auto_responder.check_for_pending_messages(
        mock_ks,
        session_id="session-b",
        project_id="multihead",
        timeout_seconds=0.5,
    )
    elapsed = asyncio.get_event_loop().time() - start

    assert count == 0
    assert elapsed < 1.0  # Should timeout quickly


@pytest.mark.asyncio
async def test_check_for_pending_messages_exception_handling():
    """Test that exceptions are caught and don't crash."""
    # Create mock that raises
    mock_ks = Mock()
    mock_ks.get_pending_messages = Mock(side_effect=Exception("Database error"))

    # Should catch exception and return 0
    count = await auto_responder.check_for_pending_messages(
        mock_ks,
        session_id="session-b",
        project_id="multihead",
    )

    assert count == 0


def test_get_pending_count_notification_zero():
    """Test notification for zero messages."""
    notification = auto_responder.get_pending_count_notification(0)
    assert notification == ""


def test_get_pending_count_notification_one():
    """Test notification for one message."""
    notification = auto_responder.get_pending_count_notification(1)
    assert notification == "\n\n[collab] 1 request pending — type /collab to review"


def test_get_pending_count_notification_multiple():
    """Test notification for multiple messages."""
    notification = auto_responder.get_pending_count_notification(5)
    assert notification == "\n\n[collab] 5 requests pending — type /collab to review"


@pytest.mark.asyncio
async def test_handle_collab_command_no_messages(knowledge_store):
    """Test /collab command when no messages."""
    result = await auto_responder.handle_collab_command(
        knowledge_store,
        session_id="session-b",
        project_id="multihead",
        head_manager=Mock(),
    )

    assert result == "No pending collaboration requests."


@pytest.mark.asyncio
async def test_handle_collab_command_shows_messages(knowledge_store, sample_request):
    """Test /collab command shows pending messages."""
    knowledge_store.insert_claim(sample_request)

    result = await auto_responder.handle_collab_command(
        knowledge_store,
        session_id="session-b",
        project_id="multihead",
        head_manager=Mock(),
    )

    assert "[Cross-Session Collaboration] 1 pending request" in result
    assert "FROM: session-a" in result
    assert "Build a web scraper" in result
    assert "/collab-respond" in result
    assert "/collab-ignore" in result


@pytest.mark.asyncio
async def test_handle_collab_command_limits_to_10(knowledge_store):
    """Test /collab command limits display to 10 requests."""
    # Create 15 requests
    for i in range(15):
        request = Claim(
            claim_id=f"clm_req_{i:03d}",
            claim_status=ClaimStatus.PROPOSED,
            claim_type=ClaimType.QUESTION,
            scope=ClaimScope(
                scope_type=ScopeType.PROJECT,
                scope_id="multihead",
                visibility="project",
                valid_from=datetime.now(timezone.utc),
            ),
            canonical=ClaimCanonical(
                claim_key=f"test.request.{i}",
                subject=EntityRef(entity_type="task", entity_id=f"t{i}", label="Task"),
                predicate="needs_help",
                object=ValueObject(value_type="string", value="help"),
            ),
            statement=f"Request {i}",
            provenance=Provenance(
                produced_by={"id": "session-a", "method": "manual"},
            ),
        )
        knowledge_store.insert_claim(request)

    result = await auto_responder.handle_collab_command(
        knowledge_store,
        session_id="session-b",
        project_id="multihead",
        head_manager=Mock(),
    )

    assert "15 pending request" in result
    assert "and 5 more" in result


def test_ignore_request():
    """Test ignoring a request adds to dedup set."""
    request_id = "clm_req_001"

    result = auto_responder.ignore_request(request_id)

    assert "ignored" in result.lower()
    assert request_id in auto_responder._responded_request_ids


@pytest.mark.asyncio
async def test_dedup_prevents_reprompting(knowledge_store, sample_request):
    """Test that dedup prevents showing same request twice."""
    knowledge_store.insert_claim(sample_request)

    # First check - should find message
    count1 = await auto_responder.check_for_pending_messages(
        knowledge_store,
        session_id="session-b",
        project_id="multihead",
    )
    assert count1 == 1

    # Mark as responded
    auto_responder._responded_request_ids.add(sample_request.claim_id)

    # Second check - should not find message (deduped)
    count2 = await auto_responder.check_for_pending_messages(
        knowledge_store,
        session_id="session-b",
        project_id="multihead",
    )
    assert count2 == 0


@pytest.mark.asyncio
async def test_respond_to_request_not_found(knowledge_store):
    """Test responding to non-existent request."""
    mock_head_manager = Mock()

    result = await auto_responder.respond_to_request(
        "clm_nonexistent",
        knowledge_store,
        mock_head_manager,
        session_id="session-b",
        project_id="multihead",
    )

    assert "not found" in result.lower()


@pytest.mark.asyncio
async def test_respond_to_request_adds_to_dedup(knowledge_store, sample_request):
    """Test that responding adds request to dedup set."""
    knowledge_store.insert_claim(sample_request)

    # Mock the decomposer and session_poller
    with patch("multihead.decomposer.TaskDecomposer") as mock_decomposer_class, \
         patch("multihead.session_poller") as mock_poller:

        mock_decomposer = AsyncMock()
        mock_decomposer.decompose = AsyncMock(return_value=Mock())
        mock_decomposer_class.return_value = mock_decomposer

        mock_poller.get_request_task = Mock(return_value="Build a web scraper")
        mock_poller.submit_decomposition_proposal = Mock(return_value="clm_response_001")

        mock_head_manager = Mock()

        result = await auto_responder.respond_to_request(
            sample_request.claim_id[:16],
            knowledge_store,
            mock_head_manager,
            session_id="session-b",
            project_id="multihead",
        )

        assert "submitted" in result.lower()
        assert sample_request.claim_id in auto_responder._responded_request_ids


@pytest.mark.asyncio
async def test_scope_filtering_in_pending_messages(knowledge_store):
    """Test that pending messages are filtered by scope."""
    # Create request in different scope
    h2v_request = Claim(
        claim_id="clm_h2v_req",
        claim_status=ClaimStatus.PROPOSED,
        claim_type=ClaimType.QUESTION,
        scope=ClaimScope(
            scope_type=ScopeType.PROJECT,
            scope_id="h2v",
            visibility="project",
            valid_from=datetime.now(timezone.utc),
        ),
        canonical=ClaimCanonical(
            claim_key="h2v.request",
            subject=EntityRef(entity_type="task", entity_id="t1", label="Task"),
            predicate="needs_help",
            object=ValueObject(value_type="string", value="help"),
        ),
        statement="Project question",
        provenance=Provenance(
            produced_by={"id": "session-a", "method": "manual"},
        ),
    )

    knowledge_store.insert_claim(h2v_request)

    # Query for multihead scope - should not see h2v request
    count = await auto_responder.check_for_pending_messages(
        knowledge_store,
        session_id="session-b",
        project_id="multihead",
    )

    assert count == 0


@pytest.mark.asyncio
async def test_filters_own_requests(knowledge_store, sample_request):
    """Test that session doesn't see its own requests."""
    knowledge_store.insert_claim(sample_request)

    # Query as same session that posted the request
    count = await auto_responder.check_for_pending_messages(
        knowledge_store,
        session_id="session-a",  # Same as request.provenance.produced_by.id
        project_id="multihead",
    )

    assert count == 0
