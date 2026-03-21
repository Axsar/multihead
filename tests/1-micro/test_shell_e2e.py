"""End-to-end integration tests for the MultiHead Agent Terminal.

Tests the full shell lifecycle: startup → command → chat → exit,
including head management, knowledge queries, process management,
session resume, and brain-swap (local ↔ Claude SDK).
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from multihead.shell import BRAIN_CLAUDE, BRAIN_LOCAL, Shell
from multihead.process_manager import ProcessManager
from multihead.slash_commands import SlashCommandHandler
from multihead.runtime_config import RuntimeConfig
from multihead.tool_registry import ToolRegistry


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


# ---------------------------------------------------------------------------
# Fixtures — full integration (real SlashCommandHandler, real ToolRegistry)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_hm():
    """HeadManager mock with realistic state."""
    hm = MagicMock()
    hm.active_head = "mock-llm"
    hm.get_states.return_value = {
        "mock-llm": {"state": "active", "name": "Mock LLM", "adapter": "mock"},
        "mock-vlm": {"state": "off", "name": "Mock VLM", "adapter": "mock"},
    }
    hm.shutdown = AsyncMock()
    hm.wake_head = AsyncMock()
    hm.sleep_head = AsyncMock()
    hm.ensure_active = AsyncMock()
    return hm


@pytest.fixture
def mock_ks():
    """KnowledgeStore mock with claims."""
    ks = MagicMock()
    claim = MagicMock()
    claim.statement = "The engine supports hot-swapping GPU models"
    claim.canonical = MagicMock()
    claim.canonical.claim_key = "engine.hotswap"
    ks.list_claims.return_value = [claim]
    ks.search_claims_fts.return_value = [
        ("engine.hotswap", "The engine supports hot-swapping GPU models", 0.9),
    ]
    return ks


@pytest.fixture
def mock_sm():
    """SessionManager mock."""
    sm = MagicMock()
    sm.create_session.return_value = MagicMock(session_id="ses_e2e")
    sm.get_session.return_value = MagicMock(
        session_id="ses_e2e", messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    )
    sm.list_sessions.return_value = [
        MagicMock(session_id="ses_001", messages=["a", "b"]),
        MagicMock(session_id="ses_002", messages=["c"]),
    ]
    return sm


@pytest.fixture
def mock_ac():
    """AgenticCore mock."""
    ac = MagicMock()
    ac.chat = AsyncMock(return_value="Sure, I can help with that!")
    ac.start = AsyncMock()
    ac.stop = AsyncMock()
    ac._detect_peers.return_value = []
    tools = MagicMock()
    tools.list_tools.return_value = []
    ac.tools = tools
    return ac


@pytest.fixture
def tool_registry():
    return ToolRegistry()


@pytest.fixture
def slash_handler(mock_hm, mock_ks, tool_registry):
    """Real SlashCommandHandler wired to mocks."""
    from pathlib import Path
    return SlashCommandHandler(
        config=RuntimeConfig(),
        config_path=Path("/tmp/e2e_test_config.json"),
        tool_registry=tool_registry,
        head_states_fn=mock_hm.get_states,
        knowledge_store=mock_ks,
        session_id="ses_e2e",
        project_id="multihead",
        head_manager=mock_hm,
    )


@pytest.fixture
def e2e_shell(mock_ac, mock_hm, mock_ks, mock_sm, slash_handler):
    """Full Shell with real SlashCommandHandler."""
    return Shell(
        agentic_core=mock_ac,
        head_manager=mock_hm,
        knowledge_store=mock_ks,
        session_manager=mock_sm,
        slash_handler=slash_handler,
        show_banner=False,
    )


# ---------------------------------------------------------------------------
# Full lifecycle
# ---------------------------------------------------------------------------


class TestShellLifecycle:
    """Start → interact → exit."""

    async def test_start_chat_exit(self, e2e_shell, mock_ac):
        """Full lifecycle: start → chat → exit."""
        with mock_tui(["Hello there", "exit"]):
            await e2e_shell.run("ses_e2e")
        mock_ac.start.assert_called_once()
        mock_ac.chat.assert_called_once()
        mock_ac.stop.assert_called_once()

    async def test_multiple_messages(self, e2e_shell, mock_ac):
        with mock_tui(["msg1", "msg2", "msg3", "exit"]):
            await e2e_shell.run("ses_e2e")
        assert mock_ac.chat.call_count == 3

    async def test_mixed_slash_and_chat(self, e2e_shell, mock_ac, mock_hm):
        with mock_tui(["/status", "Hello", "/help", "exit"]):
            await e2e_shell.run("ses_e2e")
        # Only "Hello" should reach the agent
        assert mock_ac.chat.call_count == 1

    async def test_empty_then_chat_then_exit(self, e2e_shell, mock_ac):
        with mock_tui(["", "", "Hi", "exit"]):
            await e2e_shell.run("ses_e2e")
        assert mock_ac.chat.call_count == 1


# ---------------------------------------------------------------------------
# Head management through shell
# ---------------------------------------------------------------------------


class TestHeadManagementE2E:
    """Head wake/sleep/swap via slash commands in the shell."""

    async def test_wake_through_shell(self, e2e_shell, mock_hm):
        with mock_tui(["/wake mock-vlm", "exit"]):
            await e2e_shell.run("ses_e2e")
        mock_hm.wake_head.assert_called_once_with("mock-vlm")

    async def test_sleep_through_shell(self, e2e_shell, mock_hm):
        with mock_tui(["/sleep mock-llm", "exit"]):
            await e2e_shell.run("ses_e2e")
        mock_hm.sleep_head.assert_called_once_with("mock-llm")

    async def test_swap_through_shell(self, e2e_shell, mock_hm):
        with mock_tui(["/swap mock-vlm", "exit"]):
            await e2e_shell.run("ses_e2e")
        mock_hm.ensure_active.assert_called_once_with("mock-vlm")

    async def test_status_shows_head_states(self, e2e_shell):
        from io import StringIO
        from rich.console import Console
        output = StringIO()
        e2e_shell.console = Console(file=output, force_terminal=True)
        with mock_tui(["/status", "exit"]):
            await e2e_shell.run("ses_e2e")
        text = output.getvalue()
        assert "mock-llm" in text


# ---------------------------------------------------------------------------
# Knowledge integration
# ---------------------------------------------------------------------------


class TestKnowledgeE2E:
    """Knowledge RAG context injection + /knowledge command."""

    async def test_knowledge_context_injected(self, e2e_shell, mock_ac):
        """When user asks about a topic matching claims, knowledge is injected."""
        with mock_tui(["Tell me about engine hotswap", "exit"]):
            await e2e_shell.run("ses_e2e")
        # The chat call should include knowledge context
        call_args = mock_ac.chat.call_args[0]
        assert "Knowledge context" in call_args[1]

    async def test_knowledge_slash_command(self, e2e_shell):
        from io import StringIO
        from rich.console import Console
        output = StringIO()
        e2e_shell.console = Console(file=output, force_terminal=True)
        with mock_tui(["/knowledge engine", "exit"]):
            await e2e_shell.run("ses_e2e")
        text = output.getvalue()
        # Should show search results or "no claims"
        assert "claim" in text.lower() or "engine" in text.lower() or "No claims" in text

    async def test_no_knowledge_for_irrelevant_query(self, e2e_shell, mock_ac):
        """Unrelated queries should NOT inject knowledge context."""
        e2e_shell.ks.search_claims_fts.return_value = []  # No FTS5 matches
        with mock_tui(["What time is it?", "exit"]):
            await e2e_shell.run("ses_e2e")
        call_args = mock_ac.chat.call_args[0]
        assert "Knowledge context" not in call_args[1]


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------


class TestProcessE2E:
    """Process spawn/list/kill through the shell."""

    async def test_spawn_and_ps(self):
        """Spawn a process via ProcessManager, list it."""
        import sys
        pm = ProcessManager()
        # Use a simple python command that doesn't need nested quotes
        proc = await pm.spawn(f'"{sys.executable}" -c pass')
        assert proc.pid > 0
        procs = pm.list_processes()
        assert len(procs) >= 1
        await asyncio.sleep(0.3)
        await pm.cleanup()

    async def test_shell_cleans_up_processes(self, mock_ac, mock_hm, mock_ks,
                                             mock_sm, slash_handler):
        """Shell should cleanup ProcessManager on exit."""
        pm = ProcessManager()
        shell_with_pm = Shell(
            agentic_core=mock_ac,
            head_manager=mock_hm,
            knowledge_store=mock_ks,
            session_manager=mock_sm,
            slash_handler=slash_handler,
            show_banner=False,
            process_manager=pm,
        )
        import sys
        proc = await pm.spawn(f'"{sys.executable}" -c __import__(\'time\').sleep(60)')
        assert proc.status == "running"
        with mock_tui(["exit"]):
            await shell_with_pm.run("ses_e2e")
        # After exit, processes should be cleaned up
        assert len(pm.list_processes()) == 0


# ---------------------------------------------------------------------------
# Session resume
# ---------------------------------------------------------------------------


class TestSessionResumeE2E:
    """Resume previous sessions."""

    async def test_sessions_command(self, e2e_shell):
        from io import StringIO
        from rich.console import Console
        output = StringIO()
        e2e_shell.console = Console(file=output, force_terminal=True)
        with mock_tui(["/sessions", "exit"]):
            await e2e_shell.run("ses_e2e")
        text = output.getvalue()
        assert "ses_e2e" in text or "Sessions" in text

    async def test_session_info_command(self, e2e_shell):
        from io import StringIO
        from rich.console import Console
        output = StringIO()
        e2e_shell.console = Console(file=output, force_terminal=True)
        with mock_tui(["/session", "exit"]):
            await e2e_shell.run("ses_e2e")
        text = output.getvalue()
        assert "ses_e2e" in text


# ---------------------------------------------------------------------------
# Banner integration
# ---------------------------------------------------------------------------


class TestBannerE2E:
    """Banner display with real data flow."""

    async def test_banner_shown_by_default(self, mock_ac, mock_hm, mock_ks,
                                           mock_sm, slash_handler):
        from io import StringIO
        from rich.console import Console
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        shell_with_banner = Shell(
            agentic_core=mock_ac,
            head_manager=mock_hm,
            knowledge_store=mock_ks,
            session_manager=mock_sm,
            slash_handler=slash_handler,
            show_banner=True,
            console=console,
        )
        with mock_tui(["exit"]):
            await shell_with_banner.run("ses_e2e")
        text = output.getvalue()
        assert "MultiHead" in text
        assert "PLUR" in text

    async def test_no_banner_flag(self, mock_ac, mock_hm, mock_ks,
                                  mock_sm, slash_handler):
        from io import StringIO
        from rich.console import Console
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        shell_no_banner = Shell(
            agentic_core=mock_ac,
            head_manager=mock_hm,
            knowledge_store=mock_ks,
            session_manager=mock_sm,
            slash_handler=slash_handler,
            show_banner=False,
            console=console,
        )
        with mock_tui(["exit"]):
            await shell_no_banner.run("ses_e2e")
        text = output.getvalue()
        # Should NOT have the banner panel
        assert "multihead shell" not in text


# ---------------------------------------------------------------------------
# PLUR prompt validation
# ---------------------------------------------------------------------------


class TestPLURIntegration:
    """Verify PLUR principles are enforced in system prompt."""

    def test_system_prompt_includes_all_principles(self, e2e_shell):
        prompt = e2e_shell.build_system_prompt()
        assert "Peace" in prompt
        assert "Love" in prompt
        assert "Unity" in prompt
        assert "Respect" in prompt

    def test_system_prompt_warns_about_destructive_actions(self, e2e_shell):
        prompt = e2e_shell.build_system_prompt()
        assert "destructive" in prompt.lower()
        assert "confirmation" in prompt.lower()

    def test_system_prompt_includes_dynamic_stats(self, e2e_shell):
        prompt = e2e_shell.build_system_prompt()
        # Placeholders should be replaced with actual values
        assert "{claim_count}" not in prompt
        assert "{tool_count}" not in prompt
        assert "{peer_count}" not in prompt


# ---------------------------------------------------------------------------
# Brain-swap integration
# ---------------------------------------------------------------------------


class TestBrainSwapE2E:
    """Full brain-swap lifecycle through the shell."""

    async def test_brain_command_shows_current(self, mock_ac, mock_hm, mock_ks,
                                                mock_sm, tool_registry):
        """'/brain' with no args shows current brain mode."""
        from io import StringIO
        from pathlib import Path
        from rich.console import Console

        output = StringIO()
        console = Console(file=output, force_terminal=True)

        shell = Shell(
            agentic_core=mock_ac,
            head_manager=mock_hm,
            knowledge_store=mock_ks,
            session_manager=mock_sm,
            slash_handler=None,
            show_banner=False,
            console=console,
            brain=BRAIN_LOCAL,
        )
        slash = SlashCommandHandler(
            config=RuntimeConfig(),
            config_path=Path("/tmp/test.json"),
            tool_registry=tool_registry,
            head_states_fn=mock_hm.get_states,
            knowledge_store=mock_ks,
            session_id="ses_e2e",
            head_manager=mock_hm,
            shell=shell,
        )
        shell.slash = slash

        with mock_tui(["/brain", "exit"]):
            await shell.run("ses_e2e")
        text = output.getvalue()
        assert "local" in text

    async def test_brain_switch_to_claude_and_back(self, mock_ac, mock_hm,
                                                     mock_ks, mock_sm,
                                                     tool_registry):
        """Switch from local → claude → local using /brain commands."""
        from io import StringIO
        from pathlib import Path
        from rich.console import Console

        output = StringIO()
        console = Console(file=output, force_terminal=True)

        mock_claude = MagicMock()
        mock_claude._loaded = True
        mock_claude._mcp_servers = {"multihead": MagicMock()}
        mock_claude.generate = AsyncMock(return_value={
            "text": "Claude says hi!",
            "session_id": "ses_c1",
            "cost_usd": 0.02,
            "turns": 1,
        })

        shell = Shell(
            agentic_core=mock_ac,
            head_manager=mock_hm,
            knowledge_store=mock_ks,
            session_manager=mock_sm,
            slash_handler=None,
            show_banner=False,
            console=console,
            brain=BRAIN_LOCAL,
            claude_adapter=mock_claude,
        )
        slash = SlashCommandHandler(
            config=RuntimeConfig(),
            config_path=Path("/tmp/test.json"),
            tool_registry=tool_registry,
            head_states_fn=mock_hm.get_states,
            knowledge_store=mock_ks,
            session_id="ses_e2e",
            head_manager=mock_hm,
            shell=shell,
        )
        shell.slash = slash

        with mock_tui([
            "/brain claude",    # Switch to Claude
            "Hello Claude!",    # Chat via Claude
            "/brain local",     # Switch back to local
            "Hello local!",     # Chat via local
            "exit",
        ]):
            await shell.run("ses_e2e")

        text = output.getvalue()
        assert "Claude" in text
        # Verify both brains were used
        mock_claude.generate.assert_called_once()
        mock_ac.chat.assert_called_once()

    async def test_claude_brain_full_lifecycle(self, mock_ac, mock_hm,
                                                mock_ks, mock_sm,
                                                tool_registry):
        """Start in claude mode, chat, exit."""
        from io import StringIO
        from pathlib import Path
        from rich.console import Console

        output = StringIO()
        console = Console(file=output, force_terminal=True)

        mock_claude = MagicMock()
        mock_claude._loaded = True
        mock_claude._mcp_servers = {"multihead": MagicMock()}
        mock_claude.generate = AsyncMock(return_value={
            "text": "I'm Claude running inside MultiHead!",
            "session_id": "ses_c2",
            "cost_usd": 0.03,
            "turns": 2,
        })

        shell = Shell(
            agentic_core=mock_ac,
            head_manager=mock_hm,
            knowledge_store=mock_ks,
            session_manager=mock_sm,
            slash_handler=None,
            show_banner=True,
            console=console,
            brain=BRAIN_CLAUDE,
            claude_adapter=mock_claude,
        )
        slash = SlashCommandHandler(
            config=RuntimeConfig(),
            config_path=Path("/tmp/test.json"),
            tool_registry=tool_registry,
            head_states_fn=mock_hm.get_states,
            knowledge_store=mock_ks,
            session_id="ses_e2e",
            head_manager=mock_hm,
            shell=shell,
        )
        shell.slash = slash

        with mock_tui(["Say hello", "exit"]):
            await shell.run("ses_e2e")

        text = output.getvalue()
        # Banner should show Claude brain
        assert "Claude SDK" in text
        # Response from Claude
        assert "Claude running inside MultiHead" in text
        # Local AC should NOT have been called
        mock_ac.chat.assert_not_called()
        # Claude should have been called
        mock_claude.generate.assert_called_once()

    async def test_brain_swap_prompt_changes(self, mock_ac, mock_hm,
                                              mock_ks, mock_sm, tool_registry):
        """Prompt reflects brain mode changes."""
        mock_claude = MagicMock()
        mock_claude._loaded = True
        mock_claude._mcp_servers = {"multihead": MagicMock()}

        shell = Shell(
            agentic_core=mock_ac,
            head_manager=mock_hm,
            knowledge_store=mock_ks,
            session_manager=mock_sm,
            slash_handler=MagicMock(),
            show_banner=False,
            brain=BRAIN_LOCAL,
            claude_adapter=mock_claude,
        )

        # Local prompt shows head name
        assert "mock-llm" in shell._get_prompt_text()

        # Switch to Claude
        await shell.switch_brain("claude")
        assert "claude-sdk" in shell._get_prompt_text()

        # Switch back
        await shell.switch_brain("local")
        assert "mock-llm" in shell._get_prompt_text()
