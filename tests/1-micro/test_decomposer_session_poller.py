"""Tests for TaskDecomposer + session_poller integration (Gap #2)."""

import pytest
from unittest.mock import Mock, AsyncMock

from multihead.decomposer import TaskDecomposer, DecompositionPlan
from multihead.knowledge_store import KnowledgeStore
from multihead.knowledge_models import (
    Claim,
    ClaimCanonical,
    ClaimScope,
    ClaimStatus,
    ClaimType,
    EntityRef,
    Provenance,
    ScopeType,
    ValueObject,
)


@pytest.fixture
def mock_head_manager():
    """Create mock HeadManager."""
    manager = Mock()
    manager.get_states.return_value = {
        "qwen-llm": {
            "head_id": "qwen-llm",
            "kind": "llm",
            "state": "active",
            "is_active": True,
        },
    }

    # Mock get_manifest to return a manifest-like dict
    manager.get_manifest.return_value = {"kind": "llm", "model": "mock-model"}

    # Mock generate to return a valid decomposition plan
    async def mock_generate(head_id, prompt):
        return {
            "text": """{
                "complexity": "simple",
                "phases": [
                    {
                        "id": "1",
                        "goal": "Test phase",
                        "action_type": "explore",
                        "children": [
                            {
                                "id": "1.1",
                                "goal": "Test step",
                                "action_type": "test"
                            }
                        ]
                    }
                ]
            }"""
        }

    manager.generate = AsyncMock(side_effect=mock_generate)
    return manager


@pytest.fixture
def mock_knowledge_store(tmp_path):
    """Create a real KnowledgeStore for testing."""
    db_path = tmp_path / "test_knowledge.db"
    store = KnowledgeStore(db_path)
    return store


def test_decomposer_init_with_session_params(mock_head_manager, mock_knowledge_store):
    """Test TaskDecomposer accepts session_id and project_id parameters."""
    decomposer = TaskDecomposer(
        mock_head_manager,
        knowledge_store=mock_knowledge_store,
        session_id="test-session",
        project_id="test-project",
    )

    assert decomposer.session_id == "test-session"
    assert decomposer.project_id == "test-project"
    assert decomposer.knowledge is mock_knowledge_store


def test_decomposer_init_without_session_params(mock_head_manager):
    """Test TaskDecomposer works without session params (backward compatible)."""
    decomposer = TaskDecomposer(mock_head_manager)

    assert decomposer.session_id is None
    assert decomposer.project_id == "multihead"  # Default


@pytest.mark.asyncio
async def test_decompose_checks_for_pending_requests(
    mock_head_manager,
    mock_knowledge_store,
    caplog,
):
    """Test decompose() calls session_poller to check for pending requests."""
    import logging
    caplog.set_level(logging.INFO)

    # Create a decomposition request in knowledge store
    request_claim = Claim(
        claim_type=ClaimType.QUESTION,
        claim_status=ClaimStatus.PROPOSED,
        scope=ClaimScope(
            scope_type=ScopeType.PROJECT,
            scope_id="test-project",
            visibility="private",
        ),
        canonical=ClaimCanonical(
            claim_key="decomp.request.req_123",
            subject=EntityRef(
                entity_type="decomposition_request",
                entity_id="req_123",
                entity_label="Request",
            ),
            predicate="needs_decomposition",
            object=ValueObject(value_type="text", value="Test task"),
        ),
        statement="DECOMP_REQUEST: Test task from other session",
        rationale="Test request",
        confidence=1.0,
        provenance=Provenance(produced_by={"kind": "session", "id": "other-session"}),
    )
    mock_knowledge_store.insert_claim(request_claim)

    # Create decomposer with session_id
    decomposer = TaskDecomposer(
        mock_head_manager,
        knowledge_store=mock_knowledge_store,
        session_id="my-session",
        project_id="test-project",
    )

    # Run decompose
    plan = await decomposer.decompose("My goal")

    # Should log pending requests
    assert "Found 1 pending decomposition requests from other sessions" in caplog.text
    assert "other-session" in caplog.text
    assert "Test task from other session" in caplog.text


@pytest.mark.asyncio
async def test_decompose_skips_own_requests(
    mock_head_manager,
    mock_knowledge_store,
    caplog,
):
    """Test decompose() filters out requests from the same session."""
    import logging
    caplog.set_level(logging.INFO)

    # Create a decomposition request from SAME session
    request_claim = Claim(
        claim_type=ClaimType.QUESTION,
        claim_status=ClaimStatus.PROPOSED,
        scope=ClaimScope(
            scope_type=ScopeType.PROJECT,
            scope_id="test-project",
            visibility="private",
        ),
        canonical=ClaimCanonical(
            claim_key="decomp.request.req_456",
            subject=EntityRef(
                entity_type="decomposition_request",
                entity_id="req_456",
                entity_label="Request",
            ),
            predicate="needs_decomposition",
            object=ValueObject(value_type="text", value="My own task"),
        ),
        statement="DECOMP_REQUEST: My own task",
        rationale="Test request",
        confidence=1.0,
        provenance=Provenance(produced_by={"kind": "session", "id": "my-session"}),
    )
    mock_knowledge_store.insert_claim(request_claim)

    # Create decomposer with SAME session_id
    decomposer = TaskDecomposer(
        mock_head_manager,
        knowledge_store=mock_knowledge_store,
        session_id="my-session",
        project_id="test-project",
    )

    # Run decompose
    plan = await decomposer.decompose("My goal")

    # Should NOT log pending requests (filtered out own request)
    assert "pending decomposition requests" not in caplog.text


@pytest.mark.asyncio
async def test_decompose_without_knowledge_store_skips_poller(
    mock_head_manager,
    caplog,
):
    """Test decompose() gracefully skips session poller if no knowledge_store."""
    import logging
    caplog.set_level(logging.INFO)

    # Create decomposer WITHOUT knowledge_store
    decomposer = TaskDecomposer(
        mock_head_manager,
        knowledge_store=None,
        session_id="my-session",
    )

    # Run decompose (should not crash)
    plan = await decomposer.decompose("My goal")

    # Should not log pending requests
    assert "pending decomposition requests" not in caplog.text


@pytest.mark.asyncio
async def test_decompose_without_session_id_skips_poller(
    mock_head_manager,
    mock_knowledge_store,
    caplog,
):
    """Test decompose() skips session poller if no session_id."""
    import logging
    caplog.set_level(logging.INFO)

    # Create decomposer WITHOUT session_id
    decomposer = TaskDecomposer(
        mock_head_manager,
        knowledge_store=mock_knowledge_store,
        session_id=None,
    )

    # Run decompose (should not crash)
    plan = await decomposer.decompose("My goal")

    # Should not log pending requests
    assert "pending decomposition requests" not in caplog.text


@pytest.mark.asyncio
async def test_decompose_multiple_pending_requests(
    mock_head_manager,
    mock_knowledge_store,
    caplog,
):
    """Test decompose() handles multiple pending requests."""
    import logging
    caplog.set_level(logging.INFO)

    # Create multiple decomposition requests
    for i in range(5):
        request_claim = Claim(
            claim_type=ClaimType.QUESTION,
            claim_status=ClaimStatus.PROPOSED,
            scope=ClaimScope(
                scope_type=ScopeType.PROJECT,
                scope_id="test-project",
                visibility="private",
            ),
            canonical=ClaimCanonical(
                claim_key=f"decomp.request.req_{i}",
                subject=EntityRef(
                    entity_type="decomposition_request",
                    entity_id=f"req_{i}",
                    entity_label="Request",
                ),
                predicate="needs_decomposition",
                object=ValueObject(value_type="text", value=f"Task {i}"),
            ),
            statement=f"DECOMP_REQUEST: Task {i}",
            rationale="Test request",
            confidence=1.0,
            provenance=Provenance(produced_by={"kind": "session", "id": f"session-{i}"}),
        )
        mock_knowledge_store.insert_claim(request_claim)

    # Create decomposer
    decomposer = TaskDecomposer(
        mock_head_manager,
        knowledge_store=mock_knowledge_store,
        session_id="my-session",
        project_id="test-project",
    )

    # Run decompose
    plan = await decomposer.decompose("My goal")

    # Should log all 5 pending requests
    assert "Found 5 pending decomposition requests" in caplog.text
    # Should log first 3 in detail (any 3 tasks, order doesn't matter)
    task_count = sum(1 for i in range(5) if f"Task {i}" in caplog.text)
    assert task_count == 3, f"Expected 3 tasks logged, got {task_count}"
    # Should log "and 2 more"
    assert "and 2 more" in caplog.text


@pytest.mark.asyncio
async def test_decompose_session_poller_error_doesnt_fail_decomposition(
    mock_head_manager,
    caplog,
):
    """Test decompose() continues if session poller fails."""
    import logging
    caplog.set_level(logging.WARNING)

    # Create a broken knowledge store
    broken_store = Mock()
    broken_store.list_claims.side_effect = Exception("Database error")

    # Create decomposer
    decomposer = TaskDecomposer(
        mock_head_manager,
        knowledge_store=broken_store,
        session_id="my-session",
        project_id="test-project",
    )

    # Run decompose (should not crash despite poller error)
    plan = await decomposer.decompose("My goal")

    # Should log warning but still succeed
    assert "Session poller check failed" in caplog.text
    assert isinstance(plan, DecompositionPlan)
    assert plan.goal == "My goal"


@pytest.mark.asyncio
async def test_decompose_returns_valid_plan_with_session_poller(
    mock_head_manager,
    mock_knowledge_store,
):
    """Test decompose() still returns valid plan after checking session poller."""
    decomposer = TaskDecomposer(
        mock_head_manager,
        knowledge_store=mock_knowledge_store,
        session_id="my-session",
        project_id="test-project",
    )

    plan = await decomposer.decompose("Build a classifier")

    # Verify plan structure
    assert isinstance(plan, DecompositionPlan)
    assert plan.goal == "Build a classifier"
    assert plan.complexity == "simple"
    assert len(plan.phases) == 1
    assert plan.total_steps >= 1
