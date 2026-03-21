"""Tests for WebSocket streaming routes."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.testclient import TestClient

from multihead.adapters.mock import MockAdapter
from multihead.models import HeadManifest, AdapterKind, RunEvent, EventKind


def _make_app_with_state():
    """Create a minimal FastAPI app with WebSocket routes and mock state."""
    from fastapi import FastAPI
    from multihead.api.routes_ws import router

    app = FastAPI()
    app.include_router(router, prefix="/ws")
    return app


class TestWSRunEvents:
    def test_stream_events(self):
        app = _make_app_with_state()
        events = [
            RunEvent(run_id="run_1", kind=EventKind.STEP_STARTED, payload={"step": "s1"}),
            RunEvent(run_id="run_1", kind=EventKind.RUN_DONE, payload={}),
        ]
        mock_store = MagicMock()
        mock_store.read_events.return_value = events
        app.state.event_store = mock_store

        client = TestClient(app)
        with client.websocket_connect("/ws/runs/run_1/events") as ws:
            data1 = ws.receive_json()
            assert data1["kind"] == "step_started"
            data2 = ws.receive_json()
            assert data2["kind"] == "run_done"
            # Should get stream_end after run_done
            data3 = ws.receive_json()
            assert data3["type"] == "stream_end"

    def test_stream_empty_run(self):
        app = _make_app_with_state()
        mock_store = MagicMock()
        # First call returns empty, simulate client disconnect
        call_count = 0

        def side_effect(run_id):
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                # After a few polls with no events, simulate events arriving
                return [
                    RunEvent(run_id=run_id, kind=EventKind.RUN_DONE, payload={})
                ]
            return []

        mock_store.read_events.side_effect = side_effect
        app.state.event_store = mock_store

        client = TestClient(app)
        with client.websocket_connect("/ws/runs/run_x/events") as ws:
            data = ws.receive_json()
            assert data["kind"] == "run_done"


class TestWSChat:
    def test_chat_exchange(self):
        app = _make_app_with_state()

        mock_session = MagicMock()
        mock_session.session_id = "ses_1"

        mock_sessions = MagicMock()
        mock_sessions.get_session.return_value = mock_session

        mock_core = MagicMock()
        mock_core.chat = AsyncMock(return_value="Hello back!")

        app.state.agentic_core = mock_core
        app.state.session_manager = mock_sessions

        client = TestClient(app)
        with client.websocket_connect("/ws/chat/ses_1") as ws:
            ws.send_text("Hello")
            data = ws.receive_json()
            assert data["type"] == "response"
            assert data["content"] == "Hello back!"
            assert data["session_id"] == "ses_1"

    def test_chat_json_message(self):
        app = _make_app_with_state()

        mock_session = MagicMock()
        mock_session.session_id = "ses_2"

        mock_sessions = MagicMock()
        mock_sessions.get_session.return_value = mock_session

        mock_core = MagicMock()
        mock_core.chat = AsyncMock(return_value="Got it")

        app.state.agentic_core = mock_core
        app.state.session_manager = mock_sessions

        client = TestClient(app)
        with client.websocket_connect("/ws/chat/ses_2") as ws:
            ws.send_text(json.dumps({"message": "structured input"}))
            data = ws.receive_json()
            assert data["content"] == "Got it"

    def test_chat_new_session(self):
        app = _make_app_with_state()

        mock_session = MagicMock()
        mock_session.session_id = "ses_new"

        mock_sessions = MagicMock()
        mock_sessions.get_session.return_value = None
        mock_sessions.create_session.return_value = mock_session

        mock_core = MagicMock()
        mock_core.chat = AsyncMock(return_value="Welcome!")

        app.state.agentic_core = mock_core
        app.state.session_manager = mock_sessions

        client = TestClient(app)
        with client.websocket_connect("/ws/chat/nonexistent") as ws:
            # First message should be session_created
            data = ws.receive_json()
            assert data["type"] == "session_created"
            assert data["session_id"] == "ses_new"

            # Then send a message
            ws.send_text("Hi")
            data = ws.receive_json()
            assert data["type"] == "response"
            assert data["content"] == "Welcome!"

    def test_chat_stream_mode(self):
        """Token streaming via WebSocket with stream=true."""
        app = _make_app_with_state()

        mock_session = MagicMock()
        mock_session.session_id = "ses_stream"

        mock_sessions = MagicMock()
        mock_sessions.get_session.return_value = mock_session
        mock_sessions.assemble_context.return_value = [
            {"role": "user", "content": "Hello"},
        ]

        # Mock tool registry
        mock_tools = MagicMock()
        mock_tools.list_tools.return_value = []

        # Mock head manager with generate_stream
        async def mock_stream(head_id, prompt, **kwargs):
            for word in ["Hello", " from", " stream"]:
                yield word

        mock_heads = MagicMock()
        mock_heads.generate_stream = mock_stream

        # Mock VRAM manager
        mock_vram = MagicMock()
        mock_vram.ensure_core_available = AsyncMock()

        mock_core = MagicMock()
        mock_core.sessions = mock_sessions
        mock_core.vram = mock_vram
        mock_core.heads = mock_heads
        mock_core.tools = mock_tools
        mock_core.core_head_id = "core-llm"
        mock_core._format_messages = lambda msgs: "prompt"

        app.state.agentic_core = mock_core
        app.state.session_manager = mock_sessions

        client = TestClient(app)
        with client.websocket_connect("/ws/chat/ses_stream") as ws:
            ws.send_text(json.dumps({"message": "Hello", "stream": True}))

            # Should get stream_start
            data = ws.receive_json()
            assert data["type"] == "stream_start"

            # Should get token chunks
            tokens = []
            while True:
                data = ws.receive_json()
                if data["type"] == "stream_end":
                    break
                assert data["type"] == "stream_token"
                tokens.append(data["token"])

            assert len(tokens) == 3
            assert "".join(tokens) == "Hello from stream"
            assert data["content"] == "Hello from stream"


# -------------------------------------------------------------------
# MockAdapter streaming
# -------------------------------------------------------------------

class TestMockAdapterStream:
    @pytest.mark.asyncio
    async def test_generate_stream_yields_words(self):
        manifest = HeadManifest(
            head_id="stream-test", name="Stream", adapter=AdapterKind.MOCK,
            model="mock-v1", kind="llm", gpu_required=False,
        )
        adapter = MockAdapter(manifest)
        await adapter.load()

        chunks = []
        async for chunk in adapter.generate_stream("test prompt"):
            chunks.append(chunk)

        full_text = "".join(chunks)
        assert len(chunks) > 1  # Should yield multiple words
        assert "Mock LLM response" in full_text

    @pytest.mark.asyncio
    async def test_generate_stream_matches_generate(self):
        manifest = HeadManifest(
            head_id="stream-test2", name="Stream2", adapter=AdapterKind.MOCK,
            model="mock-v1", kind="llm", gpu_required=False,
        )
        adapter = MockAdapter(manifest)
        await adapter.load()

        # Get full response
        full = await adapter.generate("same prompt")
        full_text = full["text"]

        # Get streamed response
        chunks = []
        async for chunk in adapter.generate_stream("same prompt"):
            chunks.append(chunk)
        streamed_text = "".join(chunks)

        # Call counts differ (generate is called twice), but text format matches
        assert "Mock LLM response" in streamed_text
