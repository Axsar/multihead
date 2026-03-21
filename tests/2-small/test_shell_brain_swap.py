"""Tests for Shell brain-swap integration (local ↔ Claude SDK).

Verifies that the shell can operate in two brain modes:
- local: AgenticCore with local GPU models
- claude: Claude Agent SDK with in-process MCP tools
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from multihead.shell import BRAIN_CLAUDE, BRAIN_LOCAL, Shell


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_head_manager():
    hm = MagicMock()
    hm.active_head = "core-llm"
    hm.get_states.return_value = {
        "core-llm": {"state": "active", "name": "Core LLM", "adapter": "transformers"},
        "claude-sdk": {"state": "off", "name": "Claude SDK", "adapter": "claude_agent_sdk"},
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
    claim.statement = "The mesh protocol uses Ed25519 for node identity"
    claim.canonical = MagicMock()
    claim.canonical.claim_key = "mesh.ed25519"
    ks.list_claims.return_value = [claim]
    # FTS5 search: by default return results; tests override for empty cases
    ks.search_claims_fts.return_value = [
        ("mesh.ed25519", "The mesh protocol uses Ed25519 for node identity", 0.9),
    ]
    return ks


@pytest.fixture
def mock_session_manager():
    sm = MagicMock()
    sm.create_session.return_value = MagicMock(session_id="ses_test123")
    sm.add_message = MagicMock()
    return sm


@pytest.fixture
def mock_agentic_core():
    ac = MagicMock()
    ac.chat = AsyncMock(return_value="Local response")
    ac.start = AsyncMock()
    ac.stop = AsyncMock()
    ac._detect_peers.return_value = []
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
def mock_claude_adapter():
    adapter = MagicMock()
    adapter._loaded = True
    adapter._mcp_servers = {}
    adapter.load = AsyncMock()
    adapter.generate = AsyncMock(return_value={
        "text": "Claude response",
        "session_id": "ses_claude_001",
        "cost_usd": 0.05,
        "turns": 2,
    })
    adapter.set_mcp_servers = MagicMock()
    return adapter


@pytest.fixture
def shell_local(mock_agentic_core, mock_head_manager, mock_knowledge_store,
                mock_session_manager, mock_slash):
    """Shell in local brain mode."""
    return Shell(
        agentic_core=mock_agentic_core,
        head_manager=mock_head_manager,
        knowledge_store=mock_knowledge_store,
        session_manager=mock_session_manager,
        slash_handler=mock_slash,
        show_banner=False,
        brain=BRAIN_LOCAL,
    )


@pytest.fixture
def shell_claude(mock_agentic_core, mock_head_manager, mock_knowledge_store,
                 mock_session_manager, mock_slash, mock_claude_adapter):
    """Shell in claude brain mode."""
    return Shell(
        agentic_core=mock_agentic_core,
        head_manager=mock_head_manager,
        knowledge_store=mock_knowledge_store,
        session_manager=mock_session_manager,
        slash_handler=mock_slash,
        show_banner=False,
        brain=BRAIN_CLAUDE,
        claude_adapter=mock_claude_adapter,
    )


# ---------------------------------------------------------------------------
# Brain mode initialization
# ---------------------------------------------------------------------------


class TestBrainInit:
    """Brain mode initialization."""

    def test_default_brain_is_local(self, shell_local):
        assert shell_local.brain == BRAIN_LOCAL

    def test_claude_brain_set(self, shell_claude):
        assert shell_claude.brain == BRAIN_CLAUDE

    def test_invalid_brain_falls_back_to_local(self, mock_agentic_core,
                                                mock_head_manager,
                                                mock_knowledge_store,
                                                mock_session_manager,
                                                mock_slash):
        s = Shell(
            agentic_core=mock_agentic_core,
            head_manager=mock_head_manager,
            knowledge_store=mock_knowledge_store,
            session_manager=mock_session_manager,
            slash_handler=mock_slash,
            brain="invalid",
        )
        assert s.brain == BRAIN_LOCAL

    def test_claude_adapter_stored(self, shell_claude, mock_claude_adapter):
        assert shell_claude._claude_adapter is mock_claude_adapter

    def test_local_has_no_claude_adapter(self, shell_local):
        assert shell_local._claude_adapter is None


# ---------------------------------------------------------------------------
# Local brain routing
# ---------------------------------------------------------------------------


class TestLocalBrain:
    """Local brain (AgenticCore) routing."""

    async def test_chat_via_local(self, shell_local, mock_agentic_core):
        result = await shell_local._chat_via_local("ses_1", "hello")
        assert result == "Local response"
        mock_agentic_core.chat.assert_called_once()

    async def test_local_injects_knowledge(self, shell_local, mock_agentic_core):
        result = await shell_local._chat_via_local("ses_1", "tell me about mesh protocol")
        call_args = mock_agentic_core.chat.call_args[0]
        assert "Knowledge context" in call_args[1]

    async def test_local_no_knowledge_for_short_input(self, shell_local, mock_agentic_core):
        shell_local.ks.search_claims_fts.return_value = []
        result = await shell_local._chat_via_local("ses_1", "hi")
        call_args = mock_agentic_core.chat.call_args[0]
        assert "Knowledge" not in call_args[1]


# ---------------------------------------------------------------------------
# Claude brain routing
# ---------------------------------------------------------------------------


class TestClaudeBrain:
    """Claude brain (Agent SDK) routing."""

    async def test_chat_via_claude(self, shell_claude, mock_claude_adapter):
        result = await shell_claude._chat_via_claude("ses_1", "hello world")
        assert result == "Claude response"
        mock_claude_adapter.generate.assert_called_once()

    async def test_claude_passes_system_prompt(self, shell_claude, mock_claude_adapter):
        await shell_claude._chat_via_claude("ses_1", "hello")
        call_kwargs = mock_claude_adapter.generate.call_args[1]
        assert "system_prompt" in call_kwargs
        assert "PLUR" in call_kwargs["system_prompt"]

    async def test_claude_passes_conversation_id(self, shell_claude, mock_claude_adapter):
        await shell_claude._chat_via_claude("ses_1", "hello")
        call_kwargs = mock_claude_adapter.generate.call_args[1]
        assert "conversation_id" in call_kwargs

    async def test_claude_records_local_session(self, shell_claude, mock_session_manager):
        await shell_claude._chat_via_claude("ses_1", "hello")
        assert mock_session_manager.add_message.call_count == 2  # user + assistant

    async def test_claude_without_adapter_returns_error(self, shell_local):
        """Local shell (no claude adapter) returns error on claude path."""
        result = await shell_local._chat_via_claude("ses_1", "hello")
        assert "error" in result.lower()

    async def test_claude_error_handling(self, shell_claude, mock_claude_adapter):
        mock_claude_adapter.generate.side_effect = RuntimeError("API error")
        result = await shell_claude._chat_via_claude("ses_1", "hello")
        assert "Claude error" in result


# ---------------------------------------------------------------------------
# Brain switching
# ---------------------------------------------------------------------------


class TestBrainSwitch:
    """Brain switching mid-session."""

    async def test_switch_to_claude(self, shell_local, mock_claude_adapter):
        shell_local._claude_adapter = mock_claude_adapter
        result = await shell_local.switch_brain("claude")
        assert shell_local.brain == BRAIN_CLAUDE
        assert "Claude" in result

    async def test_switch_to_local(self, shell_claude):
        result = await shell_claude.switch_brain("local")
        assert shell_claude.brain == BRAIN_LOCAL
        assert "local" in result.lower()

    async def test_switch_same_mode_noop(self, shell_local):
        result = await shell_local.switch_brain("local")
        assert "Already" in result

    async def test_switch_claude_already_claude(self, shell_claude):
        result = await shell_claude.switch_brain("claude")
        assert "Already" in result

    async def test_switch_invalid_mode(self, shell_local):
        result = await shell_local.switch_brain("gpt")
        assert "Unknown" in result

    async def test_switch_claude_without_adapter(self, shell_local):
        result = await shell_local.switch_brain("claude")
        assert "not configured" in result.lower()


# ---------------------------------------------------------------------------
# Prompt reflects brain mode
# ---------------------------------------------------------------------------


class TestPrompt:
    """Prompt changes based on brain mode."""

    def test_local_prompt_shows_head(self, shell_local):
        prompt = shell_local._get_prompt_text()
        assert "core-llm" in prompt

    def test_claude_prompt_shows_claude(self, shell_claude):
        prompt = shell_claude._get_prompt_text()
        assert "claude-sdk" in prompt


# ---------------------------------------------------------------------------
# MCP tool injection
# ---------------------------------------------------------------------------


class TestMCPInjection:
    """In-process MCP tool injection for Claude brain."""

    def test_inject_mcp_tools(self, shell_claude, mock_claude_adapter):
        with patch("multihead.sdk_mcp_tools.build_sdk_mcp_server") as mock_build:
            mock_build.return_value = {"type": "sdk", "tools": []}
            shell_claude._inject_mcp_tools()
            mock_build.assert_called_once()
            mock_claude_adapter.set_mcp_servers.assert_called_once()

    async def test_ensure_claude_ready_loads_adapter(self, shell_claude, mock_claude_adapter):
        mock_claude_adapter._loaded = False
        await shell_claude._ensure_claude_ready()
        mock_claude_adapter.load.assert_called_once()

    async def test_ensure_claude_ready_skips_mcp(self, shell_claude, mock_claude_adapter):
        """MCP injection disabled due to SDK bug — ensure_claude_ready skips it."""
        mock_claude_adapter._mcp_servers = {}
        ready = await shell_claude._ensure_claude_ready()
        assert ready is True
        # MCP not injected (disabled in shell due to SDK v0.1.44 bug)
        assert mock_claude_adapter._mcp_servers == {}

    async def test_ensure_claude_falls_back_on_failure(self, shell_claude, mock_claude_adapter):
        mock_claude_adapter._loaded = False
        mock_claude_adapter.load.side_effect = RuntimeError("SDK not installed")
        ready = await shell_claude._ensure_claude_ready()
        assert ready is False


# ---------------------------------------------------------------------------
# Slash command /brain
# ---------------------------------------------------------------------------


class TestBrainSlashCommand:
    """The /brain slash command."""

    async def test_brain_show_current(self, shell_local, mock_slash):
        from multihead.slash_commands import SlashCommandHandler
        handler = SlashCommandHandler(
            config=MagicMock(),
            config_path=MagicMock(),
            tool_registry=MagicMock(),
            head_states_fn=MagicMock(),
            shell=shell_local,
        )
        result = await handler._handle_brain([])
        assert "local" in result

    async def test_brain_switch_claude(self, shell_local, mock_claude_adapter):
        from multihead.slash_commands import SlashCommandHandler
        shell_local._claude_adapter = mock_claude_adapter
        handler = SlashCommandHandler(
            config=MagicMock(),
            config_path=MagicMock(),
            tool_registry=MagicMock(),
            head_states_fn=MagicMock(),
            shell=shell_local,
        )
        result = await handler._handle_brain(["claude"])
        assert "Claude" in result

    async def test_brain_no_shell_returns_error(self):
        from multihead.slash_commands import SlashCommandHandler
        handler = SlashCommandHandler(
            config=MagicMock(),
            config_path=MagicMock(),
            tool_registry=MagicMock(),
            head_states_fn=MagicMock(),
        )
        result = await handler._handle_brain(["claude"])
        assert "requires" in result.lower()
