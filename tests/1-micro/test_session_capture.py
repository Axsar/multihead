"""Tests for SessionCapture — read Claude SDK JSONL transcripts."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from multihead.session_capture import SessionCapture, SessionRecord, ingest_session_to_knowledge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_session(tmp_path: Path, session_id: str, records: list[dict]) -> Path:
    """Write a JSONL session file to tmp_path."""
    path = tmp_path / f"{session_id}.jsonl"
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return path


def _user_record(content: str, session_id: str = "test-session") -> dict:
    return {
        "type": "user",
        "sessionId": session_id,
        "message": {"role": "user", "content": content},
        "timestamp": "2026-03-01T10:00:00Z",
        "uuid": "user-1",
    }


def _assistant_text_record(content: str, session_id: str = "test-session") -> dict:
    return {
        "type": "assistant",
        "sessionId": session_id,
        "message": {
            "role": "assistant",
            "model": "claude-opus-4-6",
            "content": [{"type": "text", "text": content}],
        },
        "timestamp": "2026-03-01T10:00:01Z",
        "uuid": "asst-1",
    }


def _assistant_tool_record(
    tool_name: str, text: str = "", session_id: str = "test-session",
) -> dict:
    content = [{"type": "tool_use", "name": tool_name, "id": "tu-1", "input": {}}]
    if text:
        content.insert(0, {"type": "text", "text": text})
    return {
        "type": "assistant",
        "sessionId": session_id,
        "message": {
            "role": "assistant",
            "model": "claude-opus-4-6",
            "content": content,
        },
        "timestamp": "2026-03-01T10:00:02Z",
        "uuid": "asst-tool-1",
    }


def _compact_record(session_id: str = "test-session") -> dict:
    return {
        "type": "system",
        "subtype": "compact_boundary",
        "sessionId": session_id,
        "content": "Conversation compacted",
        "timestamp": "2026-03-01T10:30:00Z",
        "uuid": "compact-1",
    }


def _queue_record(session_id: str = "test-session") -> dict:
    return {
        "type": "queue-operation",
        "operation": "dequeue",
        "sessionId": session_id,
        "timestamp": "2026-03-01T09:59:59Z",
    }


# ---------------------------------------------------------------------------
# SessionRecord
# ---------------------------------------------------------------------------


class TestSessionRecord:
    def test_user_record(self):
        rec = SessionRecord(_user_record("Hello"))
        assert rec.is_user
        assert not rec.is_assistant
        assert rec.content == "Hello"
        assert rec.has_text

    def test_assistant_text(self):
        rec = SessionRecord(_assistant_text_record("Hi there"))
        assert rec.is_assistant
        assert rec.content == "Hi there"
        assert rec.tools_used == []

    def test_assistant_tool_use(self):
        rec = SessionRecord(_assistant_tool_record("Read", "Let me check"))
        assert rec.is_assistant
        assert rec.content == "Let me check"
        assert rec.tools_used == ["Read"]

    def test_compact_boundary(self):
        rec = SessionRecord(_compact_record())
        assert rec.is_compact_boundary
        assert not rec.is_user

    def test_empty_content(self):
        rec = SessionRecord({"type": "progress"})
        assert not rec.has_text
        assert rec.content == ""

    def test_queue_operation(self):
        rec = SessionRecord(_queue_record())
        assert rec.type == "queue-operation"
        assert not rec.is_user
        assert not rec.is_assistant


# ---------------------------------------------------------------------------
# SessionCapture — list_sessions
# ---------------------------------------------------------------------------


class TestListSessions:
    def test_list_sessions(self, tmp_path):
        _write_session(tmp_path, "sess-1", [
            _user_record("Hello", "sess-1"),
            _assistant_text_record("Hi", "sess-1"),
        ] * 20)  # Make it big enough
        capture = SessionCapture(tmp_path)
        sessions = capture.list_sessions(min_size=100)
        assert len(sessions) >= 1
        assert sessions[0]["session_id"] == "sess-1"

    def test_list_empty_dir(self, tmp_path):
        capture = SessionCapture(tmp_path)
        assert capture.list_sessions() == []

    def test_list_nonexistent_dir(self, tmp_path):
        capture = SessionCapture(tmp_path / "nonexistent")
        assert capture.list_sessions() == []

    def test_min_size_filter(self, tmp_path):
        _write_session(tmp_path, "tiny", [_queue_record()])
        capture = SessionCapture(tmp_path)
        # File is tiny, should be filtered out
        sessions = capture.list_sessions(min_size=10000)
        assert len(sessions) == 0


# ---------------------------------------------------------------------------
# SessionCapture — read_session
# ---------------------------------------------------------------------------


class TestReadSession:
    def test_read_session(self, tmp_path):
        _write_session(tmp_path, "test-1", [
            _queue_record("test-1"),
            _user_record("Hello", "test-1"),
            _assistant_text_record("World", "test-1"),
        ])
        capture = SessionCapture(tmp_path)
        records = capture.read_session("test-1")
        assert len(records) == 3

    def test_read_nonexistent(self, tmp_path):
        capture = SessionCapture(tmp_path)
        with pytest.raises(FileNotFoundError):
            capture.read_session("nonexistent")


# ---------------------------------------------------------------------------
# SessionCapture — extract_conversation
# ---------------------------------------------------------------------------


class TestExtractConversation:
    def test_basic_conversation(self, tmp_path):
        _write_session(tmp_path, "conv-1", [
            _queue_record("conv-1"),
            _user_record("What is 1+1?", "conv-1"),
            _assistant_text_record("2", "conv-1"),
        ])
        capture = SessionCapture(tmp_path)
        msgs = capture.extract_conversation("conv-1")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "What is 1+1?"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "2"

    def test_skips_queue_operations(self, tmp_path):
        _write_session(tmp_path, "conv-2", [
            _queue_record("conv-2"),
            _queue_record("conv-2"),
            _user_record("Hi", "conv-2"),
        ])
        capture = SessionCapture(tmp_path)
        msgs = capture.extract_conversation("conv-2")
        assert len(msgs) == 1

    def test_includes_compact_boundary(self, tmp_path):
        _write_session(tmp_path, "conv-3", [
            _user_record("Part 1", "conv-3"),
            _assistant_text_record("Reply 1", "conv-3"),
            _compact_record("conv-3"),
            _user_record("Part 2", "conv-3"),
        ])
        capture = SessionCapture(tmp_path)
        msgs = capture.extract_conversation("conv-3")
        assert any(m["role"] == "system" for m in msgs)

    def test_tool_calls_excluded_by_default(self, tmp_path):
        _write_session(tmp_path, "conv-4", [
            _user_record("Check file", "conv-4"),
            _assistant_tool_record("Read", session_id="conv-4"),  # tool only, no text
        ])
        capture = SessionCapture(tmp_path)
        msgs = capture.extract_conversation("conv-4", include_tools=False)
        # Tool-only message with no text should be excluded
        assert len(msgs) == 1

    def test_tool_calls_included(self, tmp_path):
        _write_session(tmp_path, "conv-5", [
            _user_record("Check file", "conv-5"),
            _assistant_tool_record("Read", session_id="conv-5"),
        ])
        capture = SessionCapture(tmp_path)
        msgs = capture.extract_conversation("conv-5", include_tools=True)
        assert len(msgs) == 2
        assert msgs[1]["tools"] == ["Read"]


# ---------------------------------------------------------------------------
# SessionCapture — export_markdown
# ---------------------------------------------------------------------------


class TestExportMarkdown:
    def test_export(self, tmp_path):
        _write_session(tmp_path, "export-1", [
            _user_record("Hello world", "export-1"),
            _assistant_text_record("Hi!", "export-1"),
        ])
        capture = SessionCapture(tmp_path)
        output = tmp_path / "export.md"
        path = capture.export_markdown("export-1", output_path=output)
        assert path.exists()
        content = path.read_text()
        assert "Hello world" in content
        assert "Hi!" in content
        assert "# Session Transcript" in content

    def test_export_default_path(self, tmp_path):
        _write_session(tmp_path, "export-2", [
            _user_record("Test", "export-2"),
            _assistant_text_record("OK", "export-2"),
        ])
        capture = SessionCapture(tmp_path)
        path = capture.export_markdown("export-2")
        assert path.exists()
        assert "export-2.md" in path.name

    def test_export_empty_session_raises(self, tmp_path):
        _write_session(tmp_path, "empty", [_queue_record("empty")])
        capture = SessionCapture(tmp_path)
        with pytest.raises(ValueError, match="no messages"):
            capture.export_markdown("empty")


# ---------------------------------------------------------------------------
# SessionCapture — get_session_stats
# ---------------------------------------------------------------------------


class TestSessionStats:
    def test_stats(self, tmp_path):
        _write_session(tmp_path, "stats-1", [
            _user_record("Hello", "stats-1"),
            _assistant_text_record("World", "stats-1"),
            _assistant_tool_record("Edit", "Editing", "stats-1"),
            _compact_record("stats-1"),
        ])
        capture = SessionCapture(tmp_path)
        stats = capture.get_session_stats("stats-1")
        assert stats["user_messages"] == 1
        assert stats["assistant_messages"] == 2
        assert stats["compactions"] == 1
        assert "Edit" in stats["unique_tools"]
        assert stats["total_chars"] > 0


# ---------------------------------------------------------------------------
# ingest_session_to_knowledge
# ---------------------------------------------------------------------------


class TestIngestSession:
    def test_ingests_decisions(self, tmp_path):
        _write_session(tmp_path, "ingest-1", [
            _user_record("Let's use FTS5 for all search queries from now on", "ingest-1"),
            _assistant_text_record("Great decision. I'll implement FTS5 search.", "ingest-1"),
        ])
        mock_ks = MagicMock()
        mock_ks.insert_claim = MagicMock()
        ids = ingest_session_to_knowledge("ingest-1", mock_ks, sessions_dir=tmp_path)
        assert len(ids) >= 1
        assert mock_ks.insert_claim.called

    def test_skips_short_messages(self, tmp_path):
        _write_session(tmp_path, "ingest-2", [
            _user_record("ok", "ingest-2"),
            _assistant_text_record("sure", "ingest-2"),
        ])
        mock_ks = MagicMock()
        ids = ingest_session_to_knowledge("ingest-2", mock_ks, sessions_dir=tmp_path)
        assert len(ids) == 0

    def test_respects_max_claims(self, tmp_path):
        records = []
        for i in range(100):
            records.append(_user_record(
                f"Let's always use approach {i} for the implementation", f"ingest-3",
            ))
        _write_session(tmp_path, "ingest-3", records)
        mock_ks = MagicMock()
        ids = ingest_session_to_knowledge(
            "ingest-3", mock_ks, sessions_dir=tmp_path, max_claims=5,
        )
        assert len(ids) <= 5

    def test_empty_session(self, tmp_path):
        _write_session(tmp_path, "ingest-4", [_queue_record("ingest-4")])
        mock_ks = MagicMock()
        ids = ingest_session_to_knowledge("ingest-4", mock_ks, sessions_dir=tmp_path)
        assert ids == []
