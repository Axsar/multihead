"""Tests for HeadManager + ClaudeSessionAdapter integration (Gap #1)."""

import pytest

from multihead.head_manager import HeadManager, _create_adapter
from multihead.models import HeadManifest, AdapterKind
from multihead.knowledge_store import KnowledgeStore
from multihead.adapters.claude_session import ClaudeSessionAdapter


@pytest.fixture
def mock_knowledge_store(tmp_path):
    """Create a real KnowledgeStore for testing."""
    db_path = tmp_path / "test_knowledge.db"
    store = KnowledgeStore(db_path)
    return store


def test_create_adapter_claude_session_with_knowledge_store(mock_knowledge_store):
    """Test _create_adapter creates ClaudeSessionAdapter when given knowledge_store."""
    manifest = HeadManifest(
        head_id="claude-session-1",
        name="Claude Session",
        kind="llm",
        model="claude-session",
        adapter=AdapterKind.CLAUDE_SESSION,
        extra={
            "session_id": "test-session",
            "project_id": "test-project",
            "timeout_seconds": 60.0,
        },
    )

    adapter = _create_adapter(manifest, knowledge_store=mock_knowledge_store)

    assert isinstance(adapter, ClaudeSessionAdapter)
    assert adapter.knowledge_store is mock_knowledge_store
    assert adapter.session_id == "test-session"
    assert adapter.project_id == "test-project"
    assert adapter.timeout_seconds == 60.0


def test_create_adapter_claude_session_without_knowledge_store():
    """Test _create_adapter raises error when CLAUDE_SESSION needs knowledge_store."""
    manifest = HeadManifest(
        head_id="claude-session-1",
        name="Claude Session",
        kind="llm",
        model="claude-session",
        adapter=AdapterKind.CLAUDE_SESSION,
    )

    with pytest.raises(ValueError, match="ClaudeSessionAdapter requires knowledge_store"):
        _create_adapter(manifest, knowledge_store=None)


def test_create_adapter_claude_session_defaults(mock_knowledge_store):
    """Test ClaudeSessionAdapter uses defaults when extra is not provided."""
    manifest = HeadManifest(
        head_id="claude-session-1",
        name="Claude Session",
        kind="llm",
        model="claude-session",
        adapter=AdapterKind.CLAUDE_SESSION,
        # No extra config
    )

    adapter = _create_adapter(manifest, knowledge_store=mock_knowledge_store)

    assert isinstance(adapter, ClaudeSessionAdapter)
    assert adapter.session_id == "claude-main"  # Default
    assert adapter.project_id == "multihead"  # Default
    assert adapter.timeout_seconds == 300.0  # Default (5 min)
    assert adapter.min_responses == 1  # Default
    assert adapter.poll_interval == 2.0  # Default


def test_head_manager_init_with_knowledge_store(mock_knowledge_store):
    """Test HeadManager.__init__() accepts knowledge_store parameter."""
    manifests = {
        "claude-session-1": HeadManifest(
            head_id="claude-session-1",
            name="Claude Session",
            kind="llm",
            model="claude-session",
            adapter=AdapterKind.CLAUDE_SESSION,
        ),
        "mock-llm": HeadManifest(
            head_id="mock-llm",
            name="Mock LLM",
            kind="llm",
            model="mock",
            adapter=AdapterKind.MOCK,
        ),
    }

    manager = HeadManager(manifests, knowledge_store=mock_knowledge_store)

    assert manager._knowledge_store is mock_knowledge_store

    # Verify Claude session adapter was created
    adapter = manager._adapters["claude-session-1"]
    assert isinstance(adapter, ClaudeSessionAdapter)
    assert adapter.knowledge_store is mock_knowledge_store


def test_head_manager_init_without_knowledge_store_skip_claude_session():
    """Test HeadManager init fails gracefully when CLAUDE_SESSION head but no knowledge_store."""
    manifests = {
        "claude-session-1": HeadManifest(
            head_id="claude-session-1",
            name="Claude Session",
            kind="llm",
            model="claude-session",
            adapter=AdapterKind.CLAUDE_SESSION,
        ),
    }

    # Should raise error because ClaudeSessionAdapter requires knowledge_store
    with pytest.raises(ValueError, match="ClaudeSessionAdapter requires knowledge_store"):
        HeadManager(manifests, knowledge_store=None)


def test_head_manager_mixed_adapters_with_knowledge_store(mock_knowledge_store):
    """Test HeadManager with mix of adapters (some need knowledge_store, some don't)."""
    manifests = {
        "claude-session-1": HeadManifest(
            head_id="claude-session-1",
            name="Claude Session 1",
            kind="llm",
            model="claude-session",
            adapter=AdapterKind.CLAUDE_SESSION,
        ),
        "claude-session-2": HeadManifest(
            head_id="claude-session-2",
            name="Claude Session 2",
            kind="llm",
            model="claude-session",
            adapter=AdapterKind.CLAUDE_SESSION,
            extra={"session_id": "session-2"},
        ),
        "mock-llm": HeadManifest(
            head_id="mock-llm",
            name="Mock LLM",
            kind="llm",
            model="mock",
            adapter=AdapterKind.MOCK,
        ),
    }

    manager = HeadManager(manifests, knowledge_store=mock_knowledge_store)

    # Verify all adapters created
    assert len(manager._adapters) == 3

    # Claude session adapters should have knowledge_store
    assert isinstance(manager._adapters["claude-session-1"], ClaudeSessionAdapter)
    assert manager._adapters["claude-session-1"].knowledge_store is mock_knowledge_store
    assert manager._adapters["claude-session-1"].session_id == "claude-main"

    assert isinstance(manager._adapters["claude-session-2"], ClaudeSessionAdapter)
    assert manager._adapters["claude-session-2"].knowledge_store is mock_knowledge_store
    assert manager._adapters["claude-session-2"].session_id == "session-2"

    # Mock adapter shouldn't need knowledge_store
    assert manager._adapters["mock-llm"].__class__.__name__ == "MockAdapter"


def test_head_manager_backward_compatibility_no_knowledge_store():
    """Test HeadManager still works without knowledge_store for non-CLAUDE_SESSION adapters."""
    manifests = {
        "mock-llm": HeadManifest(
            head_id="mock-llm",
            name="Mock LLM",
            kind="llm",
            model="mock",
            adapter=AdapterKind.MOCK,
        ),
    }

    # Should work fine without knowledge_store
    manager = HeadManager(manifests)

    assert manager._knowledge_store is None
    assert len(manager._adapters) == 1


@pytest.mark.asyncio
async def test_claude_session_adapter_via_head_manager(mock_knowledge_store):
    """Test end-to-end: HeadManager → ClaudeSessionAdapter → generate()."""
    manifests = {
        "claude-session-1": HeadManifest(
            head_id="claude-session-1",
            name="Claude Session",
            kind="llm",
            model="claude-session",
            adapter=AdapterKind.CLAUDE_SESSION,
            extra={
                "timeout_seconds": 1.0,  # Short timeout for testing
                "min_responses": 1,
            },
        ),
    }

    manager = HeadManager(manifests, knowledge_store=mock_knowledge_store)

    # Get the adapter
    adapter = manager._adapters["claude-session-1"]

    # Call generate (will timeout with no responses, but should not crash)
    result = await adapter.generate("Test task decomposition")

    # Should return empty with no responses
    assert "text" in result
    assert "responses" in result
    assert result["responses"] == []


def test_head_manager_get_states_includes_claude_session(mock_knowledge_store):
    """Test get_states() includes claude_session heads."""
    manifests = {
        "claude-session-1": HeadManifest(
            head_id="claude-session-1",
            name="Claude Session",
            kind="llm",
            model="claude-session",
            adapter=AdapterKind.CLAUDE_SESSION,
        ),
    }

    manager = HeadManager(manifests, knowledge_store=mock_knowledge_store)

    states = manager.get_states()

    assert "claude-session-1" in states
    assert states["claude-session-1"]["kind"] == "llm"
    assert states["claude-session-1"]["model"] == "claude-session"


def test_create_adapter_other_adapters_ignore_knowledge_store(mock_knowledge_store):
    """Test other adapters ignore knowledge_store parameter gracefully."""
    mock_manifest = HeadManifest(
        head_id="mock-llm",
        name="Mock LLM",
        kind="llm",
        model="mock",
        adapter=AdapterKind.MOCK,
    )

    # Should work fine even with knowledge_store passed (it's ignored for MOCK)
    adapter = _create_adapter(mock_manifest, knowledge_store=mock_knowledge_store)

    assert adapter is not None
    assert adapter.__class__.__name__ == "MockAdapter"
