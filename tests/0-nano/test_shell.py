"""Tests for the MultiHead Agent Terminal (shell.py)."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from multihead.shell import SHELL_SYSTEM_PROMPT, Shell


@contextmanager
def mock_tui(side_effect):
    """Mock TUI application for shell.run() tests.

    Patches ``_build_application`` to return a fake Application whose
    ``run_async()`` feeds inputs through ``_process_input`` sequentially,
    exactly as the real TUI would via ``_on_input_accept``.

    Does NOT set ``_output_pane`` so that ``_tui_print`` falls back to
    ``self.console.print()`` — letting tests capture output the usual way
    via ``Console(file=StringIO())``.
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
        # Capture the shell instance so _fake_run_async can call _process_input
        _fake_run_async._shell_ref = self
        # Do NOT set self._output_pane — keeps console fallback for test assertions
        return mock_app

    with patch.object(Shell, "_build_application", _patched_build):
        yield mock_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_head_manager():
    hm = MagicMock()
    hm.active_head = "core-llm"
    hm.get_states.return_value = {
        "core-llm": {"state": "active", "name": "Core LLM", "adapter": "transformers"},
        "vision-vlm": {"state": "off", "name": "Vision VLM", "adapter": "transformers"},
    }
    hm.shutdown = AsyncMock()
    hm.wake_head = AsyncMock()
    hm.sleep_head = AsyncMock()
    hm.ensure_active = AsyncMock()
    return hm


@pytest.fixture
def mock_knowledge_store():
    ks = MagicMock()
    claim = MagicMock()
    claim.statement = "The mesh protocol uses Ed25519 for node identity signing"
    claim.canonical = MagicMock()
    claim.canonical.claim_key = "mesh.security.ed25519"
    ks.list_claims.return_value = [claim]
    # FTS5 search returns list of (claim_key, statement, confidence) tuples
    ks.search_claims_fts.return_value = [
        ("mesh.security.ed25519", "The mesh protocol uses Ed25519 for node identity signing", 0.9),
    ]
    return ks


@pytest.fixture
def mock_session_manager():
    sm = MagicMock()
    sm.create_session.return_value = MagicMock(session_id="ses_test123")
    sm.get_session.return_value = MagicMock(
        session_id="ses_test123",
        messages=[],
    )
    return sm


@pytest.fixture
def mock_agentic_core():
    ac = MagicMock()
    ac.chat = AsyncMock(return_value="Hello! I'm MultiHead.")
    ac.start = AsyncMock()
    ac.stop = AsyncMock()
    ac._detect_peers.return_value = []
    # Tool registry
    tools = MagicMock()
    tool_spec = MagicMock()
    tool_spec.name = "files.read"
    tool_spec.description = "Read a file"
    tool_spec.params_schema = {"path": {"type": "string"}}
    tools.list_tools.return_value = [tool_spec]
    ac.tools = tools
    return ac


@pytest.fixture
def mock_slash():
    slash = MagicMock()
    slash.is_slash_command.side_effect = lambda t: t.startswith("/")
    slash.handle = AsyncMock(return_value="Command output")
    return slash


@pytest.fixture
def shell(mock_agentic_core, mock_head_manager, mock_knowledge_store,
          mock_session_manager, mock_slash):
    return Shell(
        agentic_core=mock_agentic_core,
        head_manager=mock_head_manager,
        knowledge_store=mock_knowledge_store,
        session_manager=mock_session_manager,
        slash_handler=mock_slash,
        show_banner=False,
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestShellInit:
    def test_creates_with_all_components(self, shell):
        assert shell.ac is not None
        assert shell.hm is not None
        assert shell.ks is not None
        assert shell.sessions is not None
        assert shell.slash is not None

    def test_default_console(self, shell):
        assert shell.console is not None

    def test_custom_console(self, mock_agentic_core, mock_head_manager,
                           mock_knowledge_store, mock_session_manager, mock_slash):
        from rich.console import Console
        custom = Console()
        s = Shell(mock_agentic_core, mock_head_manager, mock_knowledge_store,
                  mock_session_manager, mock_slash, console=custom)
        assert s.console is custom

    def test_banner_flag(self, mock_agentic_core, mock_head_manager,
                        mock_knowledge_store, mock_session_manager, mock_slash):
        s = Shell(mock_agentic_core, mock_head_manager, mock_knowledge_store,
                  mock_session_manager, mock_slash, show_banner=True)
        assert s.show_banner is True


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


class TestPrompt:
    def test_prompt_with_active_head(self, shell):
        prompt = shell._get_prompt_text()
        assert "core-llm" in prompt
        assert "you>" in prompt

    def test_prompt_without_active_head(self, shell):
        shell.hm.active_head = None
        prompt = shell._get_prompt_text()
        assert "you>" in prompt


# ---------------------------------------------------------------------------
# Multi-line mode
# ---------------------------------------------------------------------------


class TestMultilineMode:
    def test_default_is_single_line(self, shell):
        assert shell._multiline_mode is False

    def test_toggle_multiline_on(self, shell):
        shell._multiline_mode = True
        assert shell._multiline_mode is True

    def test_toolbar_shows_multiline_hint_single(self, shell):
        toolbar = shell._get_toolbar()
        assert "Ctrl+E" in toolbar.value

    def test_toolbar_shows_multiline_indicator(self, shell):
        shell._multiline_mode = True
        toolbar = shell._get_toolbar()
        assert "MULTILINE" in toolbar.value
        assert "Alt+Enter" in toolbar.value

    def test_toolbar_status_overrides_mode(self, shell):
        """Status text takes priority over mode indicator."""
        shell._set_status("Thinking...")
        toolbar = shell._get_toolbar()
        assert "Thinking..." in toolbar.value
        assert "MULTILINE" not in toolbar.value


    def test_build_key_bindings(self, shell):
        kb = shell._build_key_bindings()
        # Should have at least 2 bindings (toggle + submit + ctrl-c + ctrl-d)
        assert len(kb.bindings) >= 2


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------


class TestBanner:
    def test_banner_includes_heads(self, shell):
        """Banner should list heads with their states."""
        from io import StringIO
        from rich.console import Console
        output = StringIO()
        shell.console = Console(file=output, force_terminal=True)
        shell._print_banner()
        text = output.getvalue()
        assert "core-llm" in text
        assert "vision-vlm" in text

    def test_banner_includes_plur(self, shell):
        from io import StringIO
        from rich.console import Console
        output = StringIO()
        shell.console = Console(file=output, force_terminal=True)
        shell._print_banner()
        text = output.getvalue()
        assert "PLUR" in text

    def test_banner_includes_knowledge_count(self, shell):
        from io import StringIO
        from rich.console import Console
        output = StringIO()
        shell.console = Console(file=output, force_terminal=True)
        shell._print_banner()
        text = output.getvalue()
        assert "claims" in text

    def test_banner_with_no_heads(self, shell):
        shell.hm.get_states.return_value = {}
        from io import StringIO
        from rich.console import Console
        output = StringIO()
        shell.console = Console(file=output, force_terminal=True)
        shell._print_banner()
        text = output.getvalue()
        assert "none loaded" in text


# ---------------------------------------------------------------------------
# PLUR System Prompt
# ---------------------------------------------------------------------------


class TestPLURPrompt:
    def test_prompt_contains_plur_principles(self):
        assert "Peace" in SHELL_SYSTEM_PROMPT
        assert "Love" in SHELL_SYSTEM_PROMPT
        assert "Unity" in SHELL_SYSTEM_PROMPT
        assert "Respect" in SHELL_SYSTEM_PROMPT

    def test_prompt_has_safety_instructions(self):
        assert "destructive" in SHELL_SYSTEM_PROMPT
        assert "confirmation" in SHELL_SYSTEM_PROMPT

    def test_prompt_has_superpower_sections(self):
        assert "Knowledge Store" in SHELL_SYSTEM_PROMPT
        assert "knowledge.db" in SHELL_SYSTEM_PROMPT
        assert "multihead solve" in SHELL_SYSTEM_PROMPT
        assert "{gpu_info}" in SHELL_SYSTEM_PROMPT

    def test_prompt_has_claim_count_placeholder(self):
        assert "{claim_count}" in SHELL_SYSTEM_PROMPT

    def test_build_system_prompt_fills_placeholders(self, shell):
        prompt = shell.build_system_prompt()
        assert "{claim_count}" not in prompt
        assert "{gpu_info}" not in prompt
        assert "{data_dir}" not in prompt
        assert "{knowledge_db}" not in prompt
        assert "{config_dir}" not in prompt
        assert "Knowledge Store" in prompt


# ---------------------------------------------------------------------------
# Response Display
# ---------------------------------------------------------------------------


class TestResponseDisplay:
    def test_display_plain_text(self, shell):
        from io import StringIO
        from rich.console import Console
        output = StringIO()
        shell.console = Console(file=output, force_terminal=True)
        shell._display_response("Hello world")
        assert "Hello world" in output.getvalue()

    def test_display_markdown(self, shell):
        from io import StringIO
        from rich.console import Console
        output = StringIO()
        shell.console = Console(file=output, force_terminal=True)
        shell._display_response("## Header\n\n```python\nprint('hi')\n```")
        assert "print" in output.getvalue()

    def test_display_empty(self, shell):
        from io import StringIO
        from rich.console import Console
        output = StringIO()
        shell.console = Console(file=output, force_terminal=True)
        shell._display_response("")
        assert output.getvalue() == ""


# ---------------------------------------------------------------------------
# Knowledge RAG
# ---------------------------------------------------------------------------


class TestKnowledgeRAG:
    def test_finds_relevant_claims(self, shell):
        ctx = shell._build_knowledge_context("Tell me about mesh security")
        assert "Ed25519" in ctx
        assert "Knowledge context" in ctx

    def test_no_results_for_unrelated_query(self, shell):
        shell.ks.search_claims_fts.return_value = []
        ctx = shell._build_knowledge_context("Tell me about cooking")
        assert ctx == ""

    def test_skips_short_words(self, shell):
        """Short words (<=3 chars) should be filtered out via FTS5."""
        shell.ks.search_claims_fts.return_value = []
        ctx = shell._build_knowledge_context("is it ok")
        assert ctx == ""

    def test_no_knowledge_store(self, shell):
        shell.ks = None
        ctx = shell._build_knowledge_context("mesh security")
        assert ctx == ""


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------


class TestRunLoop:
    async def test_exit_command(self, shell):
        with mock_tui(["exit"]):
            await shell.run("ses_test")
        shell.ac.start.assert_called_once()
        shell.ac.stop.assert_called_once()

    async def test_quit_command(self, shell):
        with mock_tui(["quit"]):
            await shell.run("ses_test")
        shell.ac.stop.assert_called_once()

    async def test_eof_exits(self, shell):
        with mock_tui(EOFError):
            await shell.run("ses_test")
        shell.ac.stop.assert_called_once()

    async def test_keyboard_interrupt_exits(self, shell):
        with mock_tui(KeyboardInterrupt):
            await shell.run("ses_test")
        shell.ac.stop.assert_called_once()

    async def test_empty_input_skipped(self, shell):
        with mock_tui(["", "exit"]):
            await shell.run("ses_test")
        shell.ac.chat.assert_not_called()

    async def test_slash_command_dispatched(self, shell):
        with mock_tui(["/help", "exit"]):
            await shell.run("ses_test")
        shell.slash.handle.assert_called_once_with("/help")
        shell.ac.chat.assert_not_called()

    async def test_regular_message_sent_to_agent(self, shell):
        with mock_tui(["Hello agent", "exit"]):
            await shell.run("ses_test")
        shell.ac.chat.assert_called_once()
        call_args = shell.ac.chat.call_args
        assert "ses_test" == call_args[0][0]

    async def test_shutdown_on_exit(self, shell):
        with mock_tui(["exit"]):
            await shell.run("ses_test")
        shell.hm.shutdown.assert_called_once()


# ---------------------------------------------------------------------------
# Multi-line input (deprecated methods — still tested for backward compat)
# ---------------------------------------------------------------------------


class TestMultiLineInput:
    """Tests for _read_multiline (prompt_toolkit-based multi-line input)."""

    def _setup_prompt_session(self, shell, responses):
        """Set up a mock PromptSession on the shell for testing."""
        from prompt_toolkit import PromptSession
        mock_session = MagicMock(spec=PromptSession)
        mock_session.prompt_async = AsyncMock(side_effect=responses)
        shell._prompt_session = mock_session

    async def test_single_line(self, shell):
        result = await shell._read_multiline("hello world")
        assert result == "hello world"

    async def test_backslash_continuation(self, shell):
        self._setup_prompt_session(shell, ["second line"])
        result = await shell._read_multiline("first line\\")
        assert result == "first line\nsecond line"

    async def test_backslash_multi_continuation(self, shell):
        self._setup_prompt_session(shell, ["line two\\", "line three"])
        result = await shell._read_multiline("line one\\")
        assert result == "line one\nline two\nline three"

    async def test_backslash_eof_submits_partial(self, shell):
        self._setup_prompt_session(shell, [EOFError])
        result = await shell._read_multiline("start\\")
        assert result == "start"

    async def test_triple_backtick_block(self, shell):
        self._setup_prompt_session(shell, ["line one", "line two", "```"])
        result = await shell._read_multiline("```")
        assert result == "line one\nline two"

    async def test_triple_backtick_with_content_on_first_line(self, shell):
        self._setup_prompt_session(shell, ["more content", "```"])
        result = await shell._read_multiline("```some start")
        assert result == "some start\nmore content"

    async def test_triple_backtick_preserves_blank_lines(self, shell):
        self._setup_prompt_session(shell, ["line one", "", "line three", "```"])
        result = await shell._read_multiline("```")
        assert result == "line one\n\nline three"

    async def test_triple_backtick_eof_submits_partial(self, shell):
        self._setup_prompt_session(shell, ["only line", EOFError])
        result = await shell._read_multiline("```")
        assert result == "only line"

    async def test_triple_backtick_empty_block_returns_none(self, shell):
        self._setup_prompt_session(shell, ["```"])
        result = await shell._read_multiline("```")
        assert result is None

    async def test_paste_detection_joins_lines(self, shell):
        """Pasted multi-line text should be joined into one message."""
        with patch.object(
            Shell, "_collect_paste_lines",
            return_value=["second line", "third line"],
        ):
            result = await shell._read_multiline("first line")
        assert result == "first line\nsecond line\nthird line"

    async def test_paste_detection_no_extra_lines(self, shell):
        """Single typed line — no paste detected."""
        with patch.object(Shell, "_collect_paste_lines", return_value=[]):
            result = await shell._read_multiline("just one line")
        assert result == "just one line"

    async def test_paste_not_triggered_for_backtick_block(self, shell):
        """Triple-backtick block should bypass paste detection."""
        self._setup_prompt_session(shell, ["pasted", "```"])
        with patch.object(Shell, "_collect_paste_lines") as mock_paste:
            result = await shell._read_multiline("```")
        mock_paste.assert_not_called()
        assert result == "pasted"

    async def test_paste_not_triggered_for_backslash(self, shell):
        """Backslash continuation should bypass paste detection."""
        self._setup_prompt_session(shell, ["end"])
        with patch.object(Shell, "_collect_paste_lines") as mock_paste:
            result = await shell._read_multiline("start\\")
        mock_paste.assert_not_called()
        assert result == "start\nend"


class TestCollectPasteLines:
    """Tests for the static _collect_paste_lines method."""

    def test_no_data_returns_empty(self):
        """When stdin has no buffered data, returns empty list."""
        with patch("select.select", return_value=([], [], [])):
            result = Shell._collect_paste_lines()
        assert result == []

    def test_collects_buffered_lines(self):
        """When stdin has buffered paste data, collects all lines."""
        import io
        fake_stdin = io.StringIO("line two\nline three\n")
        call_count = [0]
        def mock_select(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                return ([fake_stdin], [], [])
            return ([], [], [])

        with patch("select.select", side_effect=mock_select):
            with patch("sys.stdin", fake_stdin):
                result = Shell._collect_paste_lines()
        assert result == ["line two", "line three"]

    def test_handles_oserror(self):
        """Should handle OSError gracefully (e.g., Windows)."""
        with patch("select.select", side_effect=OSError("not supported")):
            result = Shell._collect_paste_lines()
        assert result == []

    def test_handles_value_error(self):
        """Should handle ValueError (e.g., closed stdin)."""
        with patch("select.select", side_effect=ValueError("closed")):
            result = Shell._collect_paste_lines()
        assert result == []
