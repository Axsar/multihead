"""Tests for session management."""

import pytest

from multihead.session import Message, Session, SessionManager


@pytest.fixture
def manager(tmp_path):
    return SessionManager(tmp_path / "sessions")


class TestSession:
    def test_auto_id(self):
        s = Session()
        assert s.session_id.startswith("ses_")

    def test_messages_default_empty(self):
        s = Session()
        assert len(s.messages) == 0


class TestSessionManager:
    def test_create_session(self, manager):
        session = manager.create_session()
        assert session.session_id.startswith("ses_")
        # Saved to disk
        path = manager.sessions_dir / f"{session.session_id}.json"
        assert path.exists()

    def test_add_message(self, manager):
        session = manager.create_session()
        msg = manager.add_message(session.session_id, "user", "Hello!")
        assert msg.role == "user"
        assert msg.content == "Hello!"

        reloaded = manager.get_session(session.session_id)
        assert len(reloaded.messages) == 1

    def test_add_message_missing_session(self, manager):
        with pytest.raises(KeyError):
            manager.add_message("nonexistent", "user", "Hello!")

    def test_load_session(self, manager):
        session = manager.create_session()
        manager.add_message(session.session_id, "user", "Test")

        # Clear in-memory cache
        manager._sessions.clear()

        loaded = manager.load_session(session.session_id)
        assert loaded is not None
        assert len(loaded.messages) == 1

    def test_load_session_missing(self, manager):
        assert manager.load_session("nonexistent") is None

    def test_list_sessions(self, manager):
        manager.create_session()
        manager.create_session()
        listing = manager.list_sessions()
        assert len(listing) == 2

    def test_assemble_context_basic(self, manager):
        session = manager.create_session()
        manager.add_message(session.session_id, "user", "Hello")
        manager.add_message(session.session_id, "assistant", "Hi there!")

        ctx = manager.assemble_context(session.session_id, system_prompt="You are helpful.")
        assert len(ctx) == 3  # system + user + assistant
        assert ctx[0]["role"] == "system"
        assert ctx[1]["role"] == "user"
        assert ctx[2]["role"] == "assistant"

    def test_assemble_context_no_system(self, manager):
        session = manager.create_session()
        manager.add_message(session.session_id, "user", "Hello")

        ctx = manager.assemble_context(session.session_id)
        assert len(ctx) == 1
        assert ctx[0]["role"] == "user"

    def test_assemble_context_missing_session(self, manager):
        ctx = manager.assemble_context("nonexistent")
        assert ctx == []

    def test_trim_messages(self):
        msgs = [
            Message(role="user", content="a" * 400),
            Message(role="assistant", content="b" * 400),
            Message(role="user", content="c" * 400),
        ]
        # Budget for ~200 tokens = ~800 chars = 2 messages
        trimmed = SessionManager._trim_messages(msgs, 200)
        assert len(trimmed) == 2
        assert trimmed[0].content.startswith("b")
        assert trimmed[1].content.startswith("c")
