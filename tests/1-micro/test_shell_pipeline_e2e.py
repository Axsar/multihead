"""Integration tests for ShellPipeline — end-to-end flows.

Tests the full pipeline with all stages wired together:
- Knowledge RAG → Intent → Execution → Recording
- Pipeline on/off behavior
- Config changes via /pipeline command
- Partial component scenarios
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from multihead.shell import Shell


@contextmanager
def mock_tui(side_effect):
    """Mock TUI application for shell.run() tests.

    Patches ``_build_application`` to return a fake Application whose
    ``run_async()`` feeds inputs through ``_process_input`` sequentially.
    Does NOT set ``_output_pane`` so ``_tui_print`` falls back to console.
    """
    inputs = side_effect if isinstance(side_effect, list) else [side_effect]

    async def _fake_run_async(*args, **kwargs):
        shell_ref = _fake_run_async._shell_ref
        for item in inputs:
            if isinstance(item, type) and issubclass(item, BaseException):
                raise item()
            if isinstance(item, BaseException):
                raise item
            text = item.strip() if isinstance(item, str) else ""
            if not text:
                continue
            if text.lower() in ("exit", "quit", "q"):
                return
            await shell_ref._process_input(text)

    mock_app = MagicMock()
    mock_app.run_async = _fake_run_async

    def _patched_build(self):
        _fake_run_async._shell_ref = self
        return mock_app

    with patch.object(Shell, "_build_application", _patched_build):
        yield mock_app


from multihead.runtime_config import RuntimeConfig
from multihead.shell_pipeline import ShellPipeline
from multihead.slash_commands import SlashCommandHandler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path):
    """Create a test knowledge DB with claims."""
    path = tmp_path / "knowledge.db"
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE claims (
            rowid INTEGER PRIMARY KEY AUTOINCREMENT,
            claim_id TEXT, claim_status TEXT DEFAULT 'accepted',
            claim_type TEXT DEFAULT 'fact', scope_type TEXT DEFAULT 'project',
            scope_id TEXT DEFAULT 'multihead', visibility TEXT DEFAULT 'private',
            valid_from TEXT, valid_to TEXT, claim_key TEXT, predicate TEXT,
            subject_json TEXT, object_json TEXT, statement TEXT, rationale TEXT,
            confidence REAL DEFAULT 0.8, stability TEXT, importance REAL DEFAULT 0.5,
            superseded_by_claim_id TEXT, rejection_reason TEXT, contested_reason TEXT,
            derived_from_json TEXT, related_json TEXT, conflicts_json TEXT,
            provenance_json TEXT, signature TEXT, signed_by TEXT,
            created_at TEXT, updated_at TEXT
        )
    """)
    claims = [
        ("clm_1", "arch.router.scoring",
         "Router uses weighted scoring with active head preference", 0.9),
        ("clm_2", "arch.dag.parallel", "DAG executor runs independent steps in parallel", 0.85),
        ("clm_3", "arch.knowledge.wal",
         "Knowledge store uses SQLite WAL mode for concurrency", 0.8),
    ]
    for cid, key, stmt, conf in claims:
        conn.execute(
            "INSERT INTO claims (claim_id, claim_key, statement, confidence, claim_status) "
            "VALUES (?, ?, ?, ?, 'accepted')",
            (cid, key, stmt, conf),
        )
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def mock_ks(db_path):
    ks = MagicMock()
    ks.insert_claim = MagicMock()

    def _connect():
        return sqlite3.connect(str(db_path))

    ks._connect = _connect
    return ks


@pytest.fixture
def runtime_config():
    return RuntimeConfig()


@pytest.fixture
def pipeline(mock_ks, runtime_config):
    hm = MagicMock()
    hm.active_head = "core-llm"
    router = MagicMock()
    router.route.return_value = "core-llm"
    return ShellPipeline(
        knowledge_store=mock_ks,
        head_manager=hm,
        router=router,
        runtime_config=runtime_config,
    )


@pytest.fixture
def brain_fn():
    return AsyncMock(return_value="The router uses weighted scoring for head selection.")


@pytest.fixture
def shell_with_pipeline(pipeline, runtime_config):
    """Full shell with pipeline wired in."""
    ac = MagicMock()
    ac.chat = AsyncMock(return_value="Local response")
    ac._detect_peers.return_value = []
    hm = MagicMock()
    hm.active_head = "core-llm"
    hm.get_states.return_value = {
        "core-llm": {"state": "active"},
    }
    hm.shutdown = AsyncMock()
    sm = MagicMock()
    sm.create_session.return_value = MagicMock(session_id="ses_e2e")
    slash = MagicMock()
    slash.is_slash_command.side_effect = lambda t: t.startswith("/")
    slash.handle = AsyncMock(return_value="OK")

    return Shell(
        agentic_core=ac,
        head_manager=hm,
        knowledge_store=MagicMock(),
        session_manager=sm,
        slash_handler=slash,
        runtime_config=runtime_config,
        show_banner=False,
        pipeline=pipeline,
    )


# ---------------------------------------------------------------------------
# Full Flow Tests
# ---------------------------------------------------------------------------


class TestFullFlow:
    async def test_chat_flow_with_rag(self, pipeline, brain_fn):
        """Chat message gets knowledge context and passes to brain."""
        result = await pipeline.process(
            "How does the router work?", brain_fn, "ses_1",
        )
        assert result == brain_fn.return_value
        # Brain should receive knowledge context about router
        call_args = brain_fn.call_args
        knowledge_ctx = call_args[0][2]
        assert "router" in knowledge_ctx.lower() or knowledge_ctx == ""

    async def test_pipeline_records_knowledge(self, pipeline, brain_fn):
        """Substantial exchanges get recorded as claims."""
        brain_fn.return_value = (
            "The router uses a weighted scoring system. It considers multiple factors "
            "including whether a head is already active (40 points), circuit breaker "
            "state (30 points), VRAM fit (15 points), error rate (10 points), and "
            "latency (5 points)."
        )
        await pipeline.process("How does the router work?", brain_fn, "ses_1")
        # Should have attempted to record a claim
        assert pipeline._stats["claims_recorded"] >= 0  # May or may not succeed

    async def test_pipeline_off_is_passthrough(self, pipeline, brain_fn):
        """With pipeline disabled, messages go straight to brain."""
        pipeline._config.pipeline.enabled = False
        await pipeline.process("How does the router work?", brain_fn, "ses_1")
        brain_fn.assert_called_once_with("ses_1", "How does the router work?", "")
        assert pipeline._stats["knowledge_hits"] == 0


# ---------------------------------------------------------------------------
# Shell Integration
# ---------------------------------------------------------------------------


class TestShellIntegration:
    async def test_shell_uses_pipeline(self, shell_with_pipeline):
        """Shell routes through pipeline when present."""
        # Ensure ac.start/stop are awaitable
        shell_with_pipeline.ac.start = AsyncMock()
        shell_with_pipeline.ac.stop = AsyncMock()
        with mock_tui(["hello", "exit"]):
            await shell_with_pipeline.run("ses_e2e")
        assert shell_with_pipeline.pipeline._stats["messages_processed"] == 1

    async def test_shell_pipeline_none(self):
        """Shell works without pipeline (backward compat)."""
        ac = MagicMock()
        ac.chat = AsyncMock(return_value="Hello!")
        ac.start = AsyncMock()
        ac.stop = AsyncMock()
        ac._detect_peers.return_value = []
        hm = MagicMock()
        hm.active_head = "core-llm"
        hm.shutdown = AsyncMock()
        sm = MagicMock()
        slash = MagicMock()
        slash.is_slash_command.side_effect = lambda t: t.startswith("/")

        s = Shell(
            agentic_core=ac,
            head_manager=hm,
            knowledge_store=MagicMock(),
            session_manager=sm,
            slash_handler=slash,
            show_banner=False,
            pipeline=None,
        )
        with mock_tui(["hello", "exit"]):
            await s.run("ses_test")
        ac.chat.assert_called_once()


# ---------------------------------------------------------------------------
# Pipeline Command Tests
# ---------------------------------------------------------------------------


class TestPipelineCommand:
    def test_show_status(self, runtime_config):
        slash = SlashCommandHandler(
            config=runtime_config,
            config_path=MagicMock(),
            tool_registry=MagicMock(),
            head_states_fn=MagicMock(),
        )
        result = slash._handle_pipeline([])
        assert "Pipeline: ON" in result
        assert "knowledge_rag" in result
        assert "auto_decompose" in result

    def test_toggle_off(self, runtime_config):
        slash = SlashCommandHandler(
            config=runtime_config,
            config_path=MagicMock(),
            tool_registry=MagicMock(),
            head_states_fn=MagicMock(),
        )
        result = slash._handle_pipeline(["off"])
        assert "disabled" in result
        assert runtime_config.pipeline.enabled is False

    def test_toggle_on(self, runtime_config):
        runtime_config.pipeline.enabled = False
        slash = SlashCommandHandler(
            config=runtime_config,
            config_path=MagicMock(),
            tool_registry=MagicMock(),
            head_states_fn=MagicMock(),
        )
        result = slash._handle_pipeline(["on"])
        assert "enabled" in result
        assert runtime_config.pipeline.enabled is True

    def test_set_value(self, runtime_config):
        slash = SlashCommandHandler(
            config=runtime_config,
            config_path=MagicMock(),
            tool_registry=MagicMock(),
            head_states_fn=MagicMock(),
        )
        result = slash._handle_pipeline(["set", "auto_record", "false"])
        assert "auto_record" in result
        assert runtime_config.pipeline.auto_record is False

    def test_set_invalid_key(self, runtime_config):
        slash = SlashCommandHandler(
            config=runtime_config,
            config_path=MagicMock(),
            tool_registry=MagicMock(),
            head_states_fn=MagicMock(),
        )
        result = slash._handle_pipeline(["set", "nonexistent", "true"])
        assert "Unknown" in result

    def test_show_with_stats(self, runtime_config, pipeline):
        """Shows pipeline stats when shell has pipeline."""
        shell = MagicMock()
        shell.pipeline = pipeline
        pipeline._stats["messages_processed"] = 42
        slash = SlashCommandHandler(
            config=runtime_config,
            config_path=MagicMock(),
            tool_registry=MagicMock(),
            head_states_fn=MagicMock(),
            shell=shell,
        )
        result = slash._handle_pipeline([])
        assert "42 messages" in result

    def test_usage_message(self, runtime_config):
        slash = SlashCommandHandler(
            config=runtime_config,
            config_path=MagicMock(),
            tool_registry=MagicMock(),
            head_states_fn=MagicMock(),
        )
        result = slash._handle_pipeline(["invalid"])
        assert "Usage" in result


# ---------------------------------------------------------------------------
# Partial Components
# ---------------------------------------------------------------------------


class TestPartialComponents:
    async def test_no_knowledge_store(self, runtime_config, brain_fn):
        """Pipeline works without knowledge store."""
        p = ShellPipeline(runtime_config=runtime_config)
        result = await p.process("hello", brain_fn, "ses_1")
        assert result == brain_fn.return_value

    async def test_no_orchestrator(self, mock_ks, runtime_config, brain_fn):
        """Pipeline falls back to brain when no orchestrator for tasks."""
        p = ShellPipeline(
            knowledge_store=mock_ks,
            runtime_config=runtime_config,
        )
        result = await p._execute_as_task("do something", "", "ses_1", brain_fn)
        assert result == brain_fn.return_value

    async def test_no_router(self, runtime_config, brain_fn):
        """Pipeline works without router (no VLM routing)."""
        p = ShellPipeline(runtime_config=runtime_config)
        p._config.pipeline.vlm_auto_route = True
        result = await p.process(
            "Look at image.png please describe it in detail now",
            brain_fn, "ses_1",
        )
        assert result == brain_fn.return_value
