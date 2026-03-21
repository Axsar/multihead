"""Round 16 tests: event store replay resilience,
knowledge store JSON safety, WS timeout, session truncation logging."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from multihead.models import (
    EventKind,
    RunEvent,
    RunStatus,
    StepDef,
    WorkOrder,
)


# ---------------------------------------------------------------------------
# Event store replay resilience
# ---------------------------------------------------------------------------


class TestEventStoreReplayResilience:
    def test_replay_survives_corrupted_work_order(self, tmp_path):
        """Replay should not crash on corrupted work_order data."""
        from multihead.event_store import EventStore

        db_path = tmp_path / "test.db"
        store = EventStore(tmp_path / "runs", db_path)

        # Write a RUN_CREATED event with corrupted work_order
        run_id = "test-run-corrupt"
        event = RunEvent(
            run_id=run_id,
            kind=EventKind.RUN_CREATED,
            data={"work_order": {"invalid": "not a real work order"}},
        )
        store.append(event)

        # Replay should not crash
        state = store.replay(run_id)
        assert state.run_id == run_id
        assert state.status == RunStatus.QUEUED
        # work_order should be None since parsing failed
        assert state.work_order is None

    def test_replay_survives_valid_work_order(self, tmp_path):
        """Replay should correctly parse valid work_order data."""
        from multihead.event_store import EventStore

        db_path = tmp_path / "test.db"
        store = EventStore(tmp_path / "runs", db_path)

        run_id = "test-run-valid"
        wo = WorkOrder(goal="test goal", steps=[
            StepDef(name="s1", head_id="h1", prompt_template="test"),
        ])
        event = RunEvent(
            run_id=run_id,
            kind=EventKind.RUN_CREATED,
            data={"work_order": wo.model_dump(mode="json")},
        )
        store.append(event)

        state = store.replay(run_id)
        assert state.work_order is not None
        assert state.work_order.goal == "test goal"

    def test_replay_logs_corrupted_work_order(self, tmp_path, caplog):
        """Replay should log a warning for corrupted work_order."""
        from multihead.event_store import EventStore

        db_path = tmp_path / "test.db"
        store = EventStore(tmp_path / "runs", db_path)

        run_id = "test-run-log"
        event = RunEvent(
            run_id=run_id,
            kind=EventKind.RUN_CREATED,
            data={"work_order": "not_a_dict"},
        )
        store.append(event)

        with caplog.at_level(logging.WARNING):
            store.replay(run_id)
        assert "Corrupted work_order" in caplog.text

    def test_replay_fallback_to_arg_work_order(self, tmp_path):
        """Replay should use arg work_order if event one is corrupted."""
        from multihead.event_store import EventStore

        db_path = tmp_path / "test.db"
        store = EventStore(tmp_path / "runs", db_path)

        run_id = "test-run-fallback"
        event = RunEvent(
            run_id=run_id,
            kind=EventKind.RUN_CREATED,
            data={"work_order": {"bad": True}},
        )
        store.append(event)

        fallback = WorkOrder(goal="fallback goal")
        state = store.replay(run_id, work_order=fallback)
        assert state.work_order is not None
        assert state.work_order.goal == "fallback goal"


# ---------------------------------------------------------------------------
# Knowledge store JSON parse safety
# ---------------------------------------------------------------------------


class TestKnowledgeStoreJSONSafety:
    def test_safe_json_loads_valid(self):
        """_safe_json_loads should parse valid JSON."""
        from multihead.knowledge_store import _safe_json_loads

        assert _safe_json_loads('["a", "b"]', []) == ["a", "b"]
        assert _safe_json_loads('{"k": 1}', {}) == {"k": 1}

    def test_safe_json_loads_corrupted(self):
        """_safe_json_loads should return default on corrupted JSON."""
        from multihead.knowledge_store import _safe_json_loads

        assert _safe_json_loads("not json", []) == []
        assert _safe_json_loads("{broken", {}) == {}

    def test_safe_json_loads_none(self):
        """_safe_json_loads should return default on None input."""
        from multihead.knowledge_store import _safe_json_loads

        assert _safe_json_loads(None, []) == []
        assert _safe_json_loads(None, {}) == {}

    def test_safe_json_loads_logs_warning(self, caplog):
        """_safe_json_loads should log a warning on corrupted JSON."""
        from multihead.knowledge_store import _safe_json_loads

        with caplog.at_level(logging.WARNING):
            _safe_json_loads("broken", [], context="test.field")
        assert "Corrupted JSON" in caplog.text
        assert "test.field" in caplog.text

    def test_row_to_event_uses_safe_json(self):
        """_row_to_event should use _safe_json_loads for all JSON fields."""
        import inspect
        from multihead.knowledge_store import KnowledgeStore

        source = inspect.getsource(KnowledgeStore._row_to_event)
        # Should use _safe_json_loads, not raw json.loads
        assert "_safe_json_loads" in source
        assert "json.loads" not in source

    def test_row_to_claim_uses_safe_json(self):
        """_row_to_claim should use _safe_json_loads for all JSON fields."""
        import inspect
        from multihead.knowledge_store import KnowledgeStore

        source = inspect.getsource(KnowledgeStore._row_to_claim)
        # Should use _safe_json_loads, not raw json.loads
        assert "_safe_json_loads" in source
        assert "json.loads" not in source

    def test_knowledge_store_has_logger(self):
        """knowledge_store helpers module should have a logger."""
        from multihead.knowledge_store._helpers import logger
        assert logger is not None


# ---------------------------------------------------------------------------
# WebSocket idle timeout
# ---------------------------------------------------------------------------


class TestWebSocketTimeout:
    def test_ws_has_idle_timeout_constant(self):
        """routes_ws should define WS_MAX_IDLE_SECONDS."""
        from multihead.api.routes_ws import WS_MAX_IDLE_SECONDS

        assert WS_MAX_IDLE_SECONDS > 0
        assert WS_MAX_IDLE_SECONDS <= 600  # At most 10 min

    def test_ws_has_max_duration_constant(self):
        """routes_ws should define WS_MAX_DURATION_SECONDS."""
        from multihead.api.routes_ws import WS_MAX_DURATION_SECONDS

        assert WS_MAX_DURATION_SECONDS > 0
        assert WS_MAX_DURATION_SECONDS <= 7200  # At most 2 hours

    def test_ws_run_events_has_timeout_logic(self):
        """ws_run_events should check both idle and absolute timeouts."""
        import inspect
        from multihead.api.routes_ws import ws_run_events

        source = inspect.getsource(ws_run_events)
        assert "WS_MAX_IDLE_SECONDS" in source
        assert "WS_MAX_DURATION_SECONDS" in source
        assert "idle_timeout" in source
        assert "max_duration_exceeded" in source

    def test_ws_updates_last_activity_on_new_events(self):
        """WebSocket should update last_activity when new events arrive."""
        import inspect
        from multihead.api.routes_ws import ws_run_events

        source = inspect.getsource(ws_run_events)
        assert "last_activity" in source


# ---------------------------------------------------------------------------
# Session truncation logging
# ---------------------------------------------------------------------------


class TestSessionTruncationLogging:
    def test_session_has_logger(self):
        """session module should have a logger."""
        from multihead import session
        assert hasattr(session, "logger")

    def test_truncation_logs_warning(self, caplog):
        """Truncating an oversized message should log a warning."""
        from multihead.session import SessionManager

        with tempfile.TemporaryDirectory() as d:
            mgr = SessionManager(Path(d), max_message_size=10)
            session = mgr.create_session()

            with caplog.at_level(logging.WARNING):
                mgr.add_message(session.session_id, "user", "x" * 50)

            assert "Truncating message" in caplog.text
            assert "50 bytes" in caplog.text
            assert "10 bytes" in caplog.text

    def test_no_warning_for_normal_message(self, caplog):
        """Normal-sized messages should not trigger truncation warning."""
        from multihead.session import SessionManager

        with tempfile.TemporaryDirectory() as d:
            mgr = SessionManager(Path(d), max_message_size=1000)
            session = mgr.create_session()

            with caplog.at_level(logging.WARNING):
                mgr.add_message(session.session_id, "user", "short message")

            assert "Truncating" not in caplog.text

    def test_truncated_message_content_is_correct(self):
        """Truncated message should have exactly max_message_size content."""
        from multihead.session import SessionManager

        with tempfile.TemporaryDirectory() as d:
            mgr = SessionManager(Path(d), max_message_size=20)
            session = mgr.create_session()
            msg = mgr.add_message(session.session_id, "user", "a" * 100)
            assert len(msg.content) == 20
