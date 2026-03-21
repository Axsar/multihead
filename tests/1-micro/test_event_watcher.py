"""Tests for EventWatcher — background event detection for the shell."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from multihead.event_watcher import EventWatcher, ShellEvent


# ---------------------------------------------------------------------------
# ShellEvent
# ---------------------------------------------------------------------------


class TestShellEvent:
    def test_defaults(self):
        e = ShellEvent(source="acp", event_type="task_available", summary="test task")
        assert e.source == "acp"
        assert e.event_type == "task_available"
        assert e.summary == "test task"
        assert e.detail == {}
        assert e.auto_actionable is False
        assert e.event_id == ""
        assert e.timestamp > 0

    def test_with_detail(self):
        e = ShellEvent(
            source="knowledge",
            event_type="collab_request",
            summary="from agent-x: fix bug",
            detail={"claim_id": "abc123", "requester": "agent-x"},
            auto_actionable=True,
            event_id="abc123",
        )
        assert e.detail["claim_id"] == "abc123"
        assert e.auto_actionable is True
        assert e.event_id == "abc123"


# ---------------------------------------------------------------------------
# EventWatcher — init and config
# ---------------------------------------------------------------------------


class TestEventWatcherInit:
    def test_defaults(self):
        ew = EventWatcher()
        assert ew._poll_interval == 15
        assert ew._watch_acp is True
        assert ew._watch_knowledge is True
        assert ew._running is False
        assert ew.pending_count == 0
        assert ew.history == []

    def test_custom_config(self):
        ew = EventWatcher(
            poll_interval=30,
            watch_acp=False,
            watch_knowledge=True,
            session_id="test-session",
            project_id="test-project",
        )
        assert ew._poll_interval == 30
        assert ew._watch_acp is False
        assert ew._session_id == "test-session"
        assert ew._project_id == "test-project"


# ---------------------------------------------------------------------------
# Queue operations
# ---------------------------------------------------------------------------


class TestQueueOperations:
    def test_enqueue_new_event(self):
        ew = EventWatcher()
        event = ShellEvent(
            source="acp", event_type="task", summary="test",
            event_id="id-1",
        )
        assert ew._enqueue(event) is True
        assert ew.pending_count == 1
        assert len(ew.history) == 1

    def test_dedup_same_event_id(self):
        ew = EventWatcher()
        e1 = ShellEvent(source="acp", event_type="task", summary="a", event_id="id-1")
        e2 = ShellEvent(source="acp", event_type="task", summary="b", event_id="id-1")
        assert ew._enqueue(e1) is True
        assert ew._enqueue(e2) is False
        assert ew.pending_count == 1

    def test_no_dedup_for_empty_event_id(self):
        ew = EventWatcher()
        e1 = ShellEvent(source="acp", event_type="task", summary="a")
        e2 = ShellEvent(source="acp", event_type="task", summary="b")
        assert ew._enqueue(e1) is True
        assert ew._enqueue(e2) is True
        assert ew.pending_count == 2

    def test_get_pending_drains_queue(self):
        ew = EventWatcher()
        for i in range(5):
            ew._enqueue(ShellEvent(
                source="test", event_type="t", summary=f"e{i}", event_id=f"id-{i}",
            ))
        assert ew.pending_count == 5
        events = ew.get_pending()
        assert len(events) == 5
        assert ew.pending_count == 0

    def test_get_pending_empty(self):
        ew = EventWatcher()
        assert ew.get_pending() == []

    def test_history_capped(self):
        ew = EventWatcher()
        ew._max_history = 5
        for i in range(10):
            ew._enqueue(ShellEvent(
                source="test", event_type="t", summary=f"e{i}", event_id=f"id-{i}",
            ))
        assert len(ew.history) == 5
        # Should keep the last 5
        assert ew.history[0].summary == "e5"
        assert ew.history[-1].summary == "e9"

    def test_clear_history(self):
        ew = EventWatcher()
        ew._enqueue(ShellEvent(source="t", event_type="t", summary="e"))
        assert len(ew.history) == 1
        ew.clear_history()
        assert len(ew.history) == 0


# ---------------------------------------------------------------------------
# Knowledge inbox check
# ---------------------------------------------------------------------------


class TestCheckKnowledgeInbox:
    @pytest.mark.asyncio
    async def test_no_knowledge_store(self):
        """Should not crash when knowledge store is None."""
        ew = EventWatcher(knowledge_store=None)
        await ew._check_knowledge_inbox()
        assert ew.pending_count == 0

    @pytest.mark.asyncio
    async def test_with_pending_messages(self):
        """Should enqueue events for pending collaboration requests."""
        # Mock a claim
        claim = MagicMock()
        claim.claim_id = "claim-abc-123"
        claim.statement = "DECOMP_REQUEST: Fix the bug"
        claim.provenance = MagicMock()
        claim.provenance.produced_by = {"id": "agent-x"}
        claim.canonical = MagicMock()
        claim.canonical.claim_type = "question"

        ks = MagicMock()
        ks.get_unhandled_claims = MagicMock(return_value=[claim])
        ks.record_interaction = MagicMock()

        ew = EventWatcher(knowledge_store=ks, session_id="s1", project_id="multihead")
        await ew._check_knowledge_inbox()

        assert ew.pending_count == 1
        events = ew.get_pending()
        assert events[0].source == "knowledge"
        assert events[0].event_type == "collab_request"
        assert "agent-x" in events[0].summary
        assert events[0].event_id == "claim-abc-123"

    @pytest.mark.asyncio
    async def test_dedup_across_checks(self):
        """Same claim_id should not produce duplicate events."""
        claim = MagicMock()
        claim.claim_id = "claim-xyz"
        claim.statement = "Do something"
        claim.provenance = MagicMock()
        claim.provenance.produced_by = {"id": "agent-y"}
        claim.canonical = MagicMock()
        claim.canonical.claim_type = "request"

        ks = MagicMock()
        ks.get_unhandled_claims = MagicMock(return_value=[claim])
        ks.record_interaction = MagicMock()

        ew = EventWatcher(knowledge_store=ks)
        await ew._check_knowledge_inbox()
        await ew._check_knowledge_inbox()  # Second check
        assert ew.pending_count == 1  # Still just 1

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Should handle timeout gracefully."""
        async def slow_query(*args, **kwargs):
            await asyncio.sleep(10)
            return []

        ks = MagicMock()
        ks.get_unhandled_claims = MagicMock(side_effect=lambda *a, **k: asyncio.sleep(10))
        ks.record_interaction = MagicMock()

        ew = EventWatcher(knowledge_store=ks)
        # Should not hang — timeout kicks in
        await asyncio.wait_for(ew._check_knowledge_inbox(), timeout=5)
        assert ew.pending_count == 0


# ---------------------------------------------------------------------------
# ACP task check
# ---------------------------------------------------------------------------


class TestCheckACPTasks:
    @pytest.mark.asyncio
    async def test_no_acp_url(self):
        """Should skip when ACP_URL not set."""
        ew = EventWatcher()
        with patch.dict("os.environ", {}, clear=True):
            await ew._check_acp_tasks()
        assert ew.pending_count == 0

    @pytest.mark.asyncio
    async def test_no_httpx(self):
        """Should skip gracefully when httpx not installed."""
        ew = EventWatcher()
        env = {"ACP_URL": "http://localhost:8000", "ACP_SESSION_KEY": "test"}
        with patch.dict("os.environ", env):
            with patch.dict("sys.modules", {"httpx": None}):
                await ew._check_acp_tasks()
        assert ew.pending_count == 0


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------


class TestRunLoop:
    @pytest.mark.asyncio
    async def test_stop(self):
        """Watcher should stop when signaled."""
        ew = EventWatcher(poll_interval=1, watch_acp=False, watch_knowledge=False)
        task = asyncio.create_task(ew.run())
        await asyncio.sleep(0.1)
        assert ew._running is True
        await ew.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert ew._running is False

    @pytest.mark.asyncio
    async def test_cancel(self):
        """Watcher should handle cancellation gracefully."""
        ew = EventWatcher(poll_interval=1, watch_acp=False, watch_knowledge=False)
        task = asyncio.create_task(ew.run())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert ew._running is False


# ---------------------------------------------------------------------------
# EventWatcherConfig in RuntimeConfig
# ---------------------------------------------------------------------------


class TestEventWatcherConfig:
    def test_default_config(self):
        from multihead.runtime_config import RuntimeConfig
        config = RuntimeConfig()
        ew_cfg = config.pipeline.event_watcher
        assert ew_cfg.enabled is True
        assert ew_cfg.poll_interval == 15
        assert ew_cfg.auto_handle is False
        assert ew_cfg.watch_acp is True
        assert ew_cfg.watch_knowledge is True

    def test_set_value_enabled(self):
        from multihead.runtime_config import RuntimeConfig
        config = RuntimeConfig()
        result = config.set_value("pipeline.event_watcher.enabled", "false")
        assert "False" in result or "false" in result.lower()
        assert config.pipeline.event_watcher.enabled is False

    def test_set_value_poll_interval(self):
        from multihead.runtime_config import RuntimeConfig
        config = RuntimeConfig()
        result = config.set_value("pipeline.event_watcher.poll_interval", "30")
        assert "30" in result
        assert config.pipeline.event_watcher.poll_interval == 30

    def test_set_value_auto_handle(self):
        from multihead.runtime_config import RuntimeConfig
        config = RuntimeConfig()
        config.set_value("pipeline.event_watcher.auto_handle", "true")
        assert config.pipeline.event_watcher.auto_handle is True

    def test_set_value_unknown_field(self):
        from multihead.runtime_config import RuntimeConfig
        config = RuntimeConfig()
        with pytest.raises(ValueError, match="Unknown event_watcher"):
            config.set_value("pipeline.event_watcher.nonexistent", "x")
