"""Tests for ConversationHarvester."""

import json
from unittest.mock import MagicMock

import pytest

from multihead.conversation_harvester import (
    ConversationHarvester,
    ConversationHarvestResult,
    Exchange,
)


def _make_jsonl_line(line_type: str, **kwargs) -> str:
    """Create a JSONL line."""
    data = {"type": line_type, **kwargs}
    return json.dumps(data)


def _make_user_line(text: str, timestamp: str = "2026-03-10T12:00:00Z") -> str:
    return _make_jsonl_line(
        "user",
        timestamp=timestamp,
        message={"role": "user", "content": text},
    )


def _make_assistant_line(text: str) -> str:
    return _make_jsonl_line(
        "assistant",
        message={"role": "assistant", "content": [{"type": "text", "text": text}]},
    )


def _make_assistant_thinking(thinking: str, text: str) -> str:
    return _make_jsonl_line(
        "assistant",
        message={
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": thinking},
                {"type": "text", "text": text},
            ],
        },
    )


def _make_progress_line() -> str:
    return _make_jsonl_line("progress", data={"tool": "Read", "status": "running"})


def _make_tool_use_line() -> str:
    return _make_jsonl_line(
        "assistant",
        message={
            "role": "assistant",
            "content": [{"type": "tool_use", "name": "Read", "input": {"path": "/foo"}}],
        },
    )


def _make_tool_result_line(result: str) -> str:
    return _make_jsonl_line(
        "user",
        message={
            "role": "user",
            "content": [{"type": "tool_result", "content": result}],
        },
    )


@pytest.fixture
def tmp_claude_home(tmp_path):
    """Create a fake ~/.claude/projects/ structure with JSONL files."""
    projects_dir = tmp_path / "projects"
    proj = projects_dir / "-mnt-d-DevD-TestProject"
    proj.mkdir(parents=True)

    # Create a simple session file
    session_file = proj / "abc12345-1234-5678-abcd-123456789012.jsonl"
    lines = [
        _make_jsonl_line("file-history-snapshot", data={}),
        _make_user_line("How do I fix the login bug?"),
        _make_progress_line(),
        _make_assistant_line("The login bug is caused by a missing null check in auth.py line 42."),
        _make_user_line("Can you fix it?"),
        _make_tool_use_line(),
        _make_progress_line(),
        _make_tool_result_line("File updated successfully"),
        _make_assistant_line("Fixed! I added a null check before accessing user.email."),
    ]
    session_file.write_text("\n".join(lines), encoding="utf-8")

    return tmp_path


@pytest.fixture
def harvester(tmp_claude_home, tmp_path):
    """Create a ConversationHarvester with mocked stores."""
    record_store = MagicMock()
    record_store.ingest_text = MagicMock(return_value=MagicMock(sha256="abc123"))
    knowledge_store = MagicMock()

    return ConversationHarvester(
        record_store=record_store,
        knowledge_store=knowledge_store,
        claude_home=tmp_claude_home,
        data_dir=tmp_path / "data",
        max_files_per_run=50,
    )


class TestScanSessions:
    def test_finds_jsonl_files(self, harvester, tmp_claude_home):
        sessions = harvester.scan_sessions()
        assert len(sessions) == 1
        assert sessions[0].session_id == "abc12345-1234-5678-abcd-123456789012"
        assert sessions[0].project_name == "-mnt-d-DevD-TestProject"
        assert sessions[0].scope_id == "testproject"

    def test_empty_dir(self, tmp_path):
        h = ConversationHarvester(
            record_store=MagicMock(),
            knowledge_store=MagicMock(),
            claude_home=tmp_path / "nonexistent",
        )
        assert h.scan_sessions() == []

    def test_project_filter(self, harvester):
        assert len(harvester.scan_sessions(project_filter="TestProject")) == 1
        assert len(harvester.scan_sessions(project_filter="Nonexistent")) == 0


class TestExchangeExtraction:
    def test_basic_exchange(self, harvester, tmp_claude_home):
        proj = tmp_claude_home / "projects" / "-mnt-d-DevD-TestProject"
        session_file = list(proj.glob("*.jsonl"))[0]

        exchanges = list(harvester._iter_exchanges(session_file, "test-session"))
        assert len(exchanges) == 2

        # First exchange
        assert "login bug" in exchanges[0].user_text
        assert "null check" in exchanges[0].assistant_text
        assert exchanges[0].turn_index == 0

        # Second exchange
        assert "fix it" in exchanges[1].user_text
        assert "Fixed" in exchanges[1].assistant_text
        assert exchanges[1].turn_index == 1

    def test_thinking_blocks_kept(self, harvester, tmp_path):
        session = tmp_path / "thinking.jsonl"
        lines = [
            _make_user_line("What is 2+2?"),
            _make_assistant_thinking("Let me think... 2+2=4", "The answer is 4."),
        ]
        session.write_text("\n".join(lines), encoding="utf-8")

        exchanges = list(harvester._iter_exchanges(session, "test"))
        assert len(exchanges) == 1
        assert "[Thinking]" in exchanges[0].assistant_text
        assert "The answer is 4" in exchanges[0].assistant_text

    def test_tool_use_skipped(self, harvester, tmp_path):
        session = tmp_path / "tools.jsonl"
        lines = [
            _make_user_line("Read the file"),
            _make_tool_use_line(),
            _make_tool_result_line("short result"),
            _make_assistant_line("Here's what the file contains."),
        ]
        session.write_text("\n".join(lines), encoding="utf-8")

        exchanges = list(harvester._iter_exchanges(session, "test"))
        assert len(exchanges) == 1
        # Tool result < 500 chars should be kept
        assert (
            "short result" in exchanges[0].user_text
            or "file contains" in exchanges[0].assistant_text
        )

    def test_progress_lines_skipped(self, harvester, tmp_path):
        session = tmp_path / "progress.jsonl"
        lines = [
            _make_progress_line(),
            _make_progress_line(),
            _make_user_line("Hello"),
            _make_progress_line(),
            _make_assistant_line("Hi there!"),
            _make_progress_line(),
        ]
        session.write_text("\n".join(lines), encoding="utf-8")

        exchanges = list(harvester._iter_exchanges(session, "test"))
        assert len(exchanges) == 1
        assert "Hello" in exchanges[0].user_text


class TestTextCleaning:
    def test_system_reminders_removed(self, harvester):
        text = "Hello <system-reminder>secret stuff here</system-reminder> world"
        cleaned = harvester._clean_text(text)
        assert "secret" not in cleaned
        assert "Hello" in cleaned
        assert "world" in cleaned

    def test_base64_replaced(self, harvester):
        text = "Image: " + "A" * 200 + " done"
        cleaned = harvester._clean_text(text)
        assert "[base64-data]" in cleaned
        assert "A" * 200 not in cleaned

    def test_short_text_unchanged(self, harvester):
        text = "Simple text with no noise"
        assert harvester._clean_text(text) == text

    def test_empty_text(self, harvester):
        assert harvester._clean_text("") == ""
        assert harvester._clean_text(None) == ""


class TestShouldSkipLine:
    @pytest.mark.parametrize("line_type", [
        "progress", "queue-operation", "file-history-snapshot", "system", "last-prompt",
    ])
    def test_skip_types(self, line_type):
        assert ConversationHarvester._should_skip_line(line_type) is True

    @pytest.mark.parametrize("line_type", ["user", "assistant"])
    def test_keep_types(self, line_type):
        assert ConversationHarvester._should_skip_line(line_type) is False


class TestHarvestAll:
    def test_basic_harvest(self, harvester):
        result = harvester.harvest_all()
        assert isinstance(result, ConversationHarvestResult)
        assert result.files_scanned == 1
        assert result.files_processed == 1
        assert result.exchanges_ingested == 2
        assert result.records_created == 2
        assert result.duration_seconds >= 0

    def test_skips_unchanged_files(self, harvester):
        # First run
        result1 = harvester.harvest_all()
        assert result1.files_processed == 1

        # Second run — same files, should skip
        result2 = harvester.harvest_all()
        assert result2.files_processed == 0
        assert result2.files_skipped == 1

    def test_max_files_limit(self, harvester, tmp_claude_home):
        # Create more files
        proj = tmp_claude_home / "projects" / "-mnt-d-DevD-TestProject"
        for i in range(5):
            f = proj / f"session-{i:04d}-0000-0000-0000-000000000000.jsonl"
            lines = [
                _make_user_line(f"Question {i}"),
                _make_assistant_line(f"Answer {i}"),
            ]
            f.write_text("\n".join(lines), encoding="utf-8")

        harvester._max_files_per_run = 3
        result = harvester.harvest_all()
        assert result.files_processed == 3  # Capped at limit

    def test_handles_malformed_jsonl(self, harvester, tmp_claude_home):
        proj = tmp_claude_home / "projects" / "-mnt-d-DevD-TestProject"
        bad = proj / "bad-session-0000-0000-0000-000000000000.jsonl"
        bad.write_text("not json\n{broken\n", encoding="utf-8")

        result = harvester.harvest_all()
        # Should not crash — bad file produces 0 exchanges
        assert result.files_scanned >= 1


class TestManifest:
    def test_manifest_persists(self, harvester, tmp_path):
        harvester.harvest_all()
        manifest_path = tmp_path / "data" / "sessions" / "conversation_manifest.json"
        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text())
        assert manifest["last_harvest"] is not None
        assert len(manifest["files"]) == 1

    def test_file_hash_fast(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        h = ConversationHarvester._file_hash_fast(f)
        assert ":" in h  # format is mtime_ns:size


class TestScopeId:
    @pytest.mark.parametrize("folder,expected", [
        ("-mnt-d-DevD-Multihead", "multihead"),
        ("-home-user-projects-myapp", "myapp"),
        ("-mnt-d-DevD-Vibebots", "vibebots"),
        ("unknown-project", "project"),
    ])
    def test_derive_scope_id(self, folder, expected):
        assert ConversationHarvester._derive_scope_id(folder) == expected


class TestStatus:
    def test_status_before_harvest(self, harvester):
        status = harvester.status()
        assert status["total_files"] == 1
        assert status["files_processed"] == 0
        assert status["files_remaining"] == 1

    def test_status_after_harvest(self, harvester):
        harvester.harvest_all()
        status = harvester.status()
        assert status["files_processed"] == 1
        assert status["files_remaining"] == 0
        assert status["total_exchanges"] == 2


class TestFormatExchange:
    def test_basic_format(self):
        ex = Exchange(
            user_text="Hello",
            assistant_text="Hi there",
            timestamp="2026-03-10T12:00:00Z",
            session_id="test",
            turn_index=0,
        )
        text = ConversationHarvester._format_exchange(ex)
        assert "[User]: Hello" in text
        assert "[Assistant]: Hi there" in text
        assert "[Timestamp:" in text

    def test_no_timestamp(self):
        ex = Exchange(
            user_text="Q", assistant_text="A",
            timestamp="", session_id="test", turn_index=0,
        )
        text = ConversationHarvester._format_exchange(ex)
        assert "[Timestamp" not in text
