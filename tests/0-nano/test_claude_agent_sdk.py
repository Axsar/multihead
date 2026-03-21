"""Tests for Claude Agent SDK adapter.

Tests the adapter interface, configuration, session management,
and integration with HeadManager — all without calling the real SDK.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from multihead.adapters.claude_agent_sdk import ClaudeAgentSDKAdapter
from multihead.models import AdapterKind, HeadManifest

# Module path for patching SDK symbols imported at module level
_MOD = "multihead.adapters.claude_agent_sdk"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manifest():
    """Basic head manifest for Claude Agent SDK."""
    return HeadManifest(
        head_id="claude-sdk",
        name="Claude Agent SDK",
        adapter=AdapterKind.CLAUDE_AGENT_SDK,
        model="claude-sonnet-4-20250514",
        kind="llm",
        gpu_required=False,
        extra={
            "max_turns": 10,
            "max_budget_usd": 2.0,
            "permission_mode": "bypassPermissions",
            "effort": "high",
            "cwd": "/tmp/test",
        },
    )


@pytest.fixture
def adapter(manifest):
    """Adapter instance (not loaded)."""
    return ClaudeAgentSDKAdapter(manifest)


@pytest.fixture
async def loaded_adapter(adapter):
    """Adapter loaded (SDK is installed in test env)."""
    await adapter.load()
    return adapter


# ---------------------------------------------------------------------------
# Fake SDK message types for mocking
# ---------------------------------------------------------------------------


@dataclass
class FakeTextBlock:
    text: str


@dataclass
class FakeAssistantMessage:
    content: list
    model: str = "claude-sonnet-4-20250514"


@dataclass
class FakeResultMessage:
    session_id: str = "ses_abc123"
    total_cost_usd: float = 0.05
    num_turns: int = 3
    is_error: bool = False
    result: str | None = None
    duration_ms: int = 5000
    duration_api_ms: int = 4000
    subtype: str = "success"


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    """Adapter initialization from manifest."""

    def test_reads_extra_config(self, adapter):
        assert adapter._max_turns == 10
        assert adapter._max_budget == 2.0
        assert adapter._permission_mode == "bypassPermissions"
        assert adapter._effort == "high"
        assert adapter._cwd == "/tmp/test"

    def test_defaults_without_extra(self):
        manifest = HeadManifest(
            head_id="claude-sdk",
            name="Claude SDK",
            adapter=AdapterKind.CLAUDE_AGENT_SDK,
            model="claude-sonnet-4-20250514",
            kind="llm",
        )
        adapter = ClaudeAgentSDKAdapter(manifest)
        assert adapter._max_turns == 25
        assert adapter._max_budget == 5.0
        assert adapter._permission_mode == "bypassPermissions"
        assert adapter._effort == "high"

    def test_not_loaded_initially(self, adapter):
        assert adapter._loaded is False

    def test_adapter_kind_registered(self):
        assert AdapterKind.CLAUDE_AGENT_SDK == "claude_agent_sdk"


# ---------------------------------------------------------------------------
# Load / Unload
# ---------------------------------------------------------------------------


class TestLoadUnload:
    """Load and unload lifecycle."""

    async def test_load_succeeds(self, adapter):
        """SDK is installed in test env, so load should work."""
        await adapter.load()
        assert adapter._loaded is True
        assert adapter._sdk_available is True

    async def test_load_without_sdk_raises(self, adapter):
        with patch(f"{_MOD}._SDK_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="claude-agent-sdk not installed"):
                await adapter.load()

    async def test_unload(self, loaded_adapter):
        await loaded_adapter.unload()
        assert loaded_adapter._loaded is False

    async def test_unload_clears_sessions(self, loaded_adapter):
        loaded_adapter._sessions["conv1"] = "ses_123"
        await loaded_adapter.unload()
        assert len(loaded_adapter._sessions) == 0


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------


class TestGenerate:
    """One-shot generation via query()."""

    async def test_generate_returns_text(self, loaded_adapter):
        async def fake_query(prompt, options):
            yield FakeAssistantMessage(content=[FakeTextBlock("Hello world")])
            yield FakeResultMessage(session_id="ses_001")

        with patch(f"{_MOD}.query", fake_query), \
             patch(f"{_MOD}.AssistantMessage", FakeAssistantMessage), \
             patch(f"{_MOD}.TextBlock", FakeTextBlock), \
             patch(f"{_MOD}.ResultMessage", FakeResultMessage):
            result = await loaded_adapter.generate("Say hello")

        assert result["text"] == "Hello world"
        assert result["session_id"] == "ses_001"

    async def test_generate_not_loaded_raises(self, adapter):
        with pytest.raises(RuntimeError, match="not loaded"):
            await adapter.generate("test")

    async def test_generate_tracks_session(self, loaded_adapter):
        async def fake_query(prompt, options):
            yield FakeAssistantMessage(content=[FakeTextBlock("ok")])
            yield FakeResultMessage(session_id="ses_tracked")

        with patch(f"{_MOD}.query", fake_query), \
             patch(f"{_MOD}.AssistantMessage", FakeAssistantMessage), \
             patch(f"{_MOD}.TextBlock", FakeTextBlock), \
             patch(f"{_MOD}.ResultMessage", FakeResultMessage):
            await loaded_adapter.generate("test")

        assert loaded_adapter.get_session_id("default") == "ses_tracked"

    async def test_generate_error_result(self, loaded_adapter):
        async def fake_query(prompt, options):
            yield FakeResultMessage(session_id="ses_err", is_error=True, result="Something failed")

        with patch(f"{_MOD}.query", fake_query), \
             patch(f"{_MOD}.AssistantMessage", FakeAssistantMessage), \
             patch(f"{_MOD}.TextBlock", FakeTextBlock), \
             patch(f"{_MOD}.ResultMessage", FakeResultMessage):
            result = await loaded_adapter.generate("test")

        assert "[ERROR]" in result["text"]

    async def test_generate_multi_block_response(self, loaded_adapter):
        async def fake_query(prompt, options):
            yield FakeAssistantMessage(content=[
                FakeTextBlock("Part 1"),
                FakeTextBlock("Part 2"),
            ])
            yield FakeResultMessage(session_id="ses_multi")

        with patch(f"{_MOD}.query", fake_query), \
             patch(f"{_MOD}.AssistantMessage", FakeAssistantMessage), \
             patch(f"{_MOD}.TextBlock", FakeTextBlock), \
             patch(f"{_MOD}.ResultMessage", FakeResultMessage):
            result = await loaded_adapter.generate("test")

        assert "Part 1" in result["text"]
        assert "Part 2" in result["text"]

    async def test_generate_cost_tracking(self, loaded_adapter):
        async def fake_query(prompt, options):
            yield FakeResultMessage(session_id="ses_cost", total_cost_usd=1.23, num_turns=7)

        with patch(f"{_MOD}.query", fake_query), \
             patch(f"{_MOD}.AssistantMessage", FakeAssistantMessage), \
             patch(f"{_MOD}.TextBlock", FakeTextBlock), \
             patch(f"{_MOD}.ResultMessage", FakeResultMessage):
            result = await loaded_adapter.generate("test")

        assert result["cost_usd"] == 1.23
        assert result["turns"] == 7


# ---------------------------------------------------------------------------
# Chat (multi-turn)
# ---------------------------------------------------------------------------


class TestChat:
    """Multi-turn chat via ClaudeSDKClient."""

    async def test_chat_not_loaded_raises(self, adapter):
        with pytest.raises(RuntimeError, match="not loaded"):
            await adapter.chat([{"role": "user", "content": "hi"}])

    async def test_chat_extracts_last_user_message(self, loaded_adapter):
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "response"},
            {"role": "user", "content": "follow up"},
        ]

        captured_prompt = []

        class FakeClient:
            def __init__(self, options):
                self.options = options

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def query(self, prompt):
                captured_prompt.append(prompt)

            async def receive_response(self):
                yield FakeAssistantMessage(content=[FakeTextBlock("reply")])
                yield FakeResultMessage(session_id="ses_chat")

        with patch(f"{_MOD}.ClaudeSDKClient", FakeClient), \
             patch(f"{_MOD}.AssistantMessage", FakeAssistantMessage), \
             patch(f"{_MOD}.TextBlock", FakeTextBlock), \
             patch(f"{_MOD}.ResultMessage", FakeResultMessage):
            result = await loaded_adapter.chat(messages)

        assert captured_prompt[0] == "follow up"
        assert result["text"] == "reply"

    async def test_chat_empty_messages(self, loaded_adapter):
        result = await loaded_adapter.chat([])
        assert result["text"] == ""

    async def test_chat_resumes_session(self, loaded_adapter):
        """Second chat call should resume the previous session."""
        loaded_adapter._sessions["default"] = "ses_existing"

        captured_options = []

        class FakeClient:
            def __init__(self, options):
                captured_options.append(options)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def query(self, prompt):
                pass

            async def receive_response(self):
                yield FakeResultMessage(session_id="ses_existing")

        with patch(f"{_MOD}.ClaudeSDKClient", FakeClient), \
             patch(f"{_MOD}.AssistantMessage", FakeAssistantMessage), \
             patch(f"{_MOD}.TextBlock", FakeTextBlock), \
             patch(f"{_MOD}.ResultMessage", FakeResultMessage):
            await loaded_adapter.chat(
                [{"role": "user", "content": "hi"}]
            )

        assert captured_options[0].resume == "ses_existing"

    async def test_chat_tracks_new_session(self, loaded_adapter):
        class FakeClient:
            def __init__(self, options):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def query(self, prompt):
                pass

            async def receive_response(self):
                yield FakeResultMessage(session_id="ses_new_from_chat")

        with patch(f"{_MOD}.ClaudeSDKClient", FakeClient), \
             patch(f"{_MOD}.AssistantMessage", FakeAssistantMessage), \
             patch(f"{_MOD}.TextBlock", FakeTextBlock), \
             patch(f"{_MOD}.ResultMessage", FakeResultMessage):
            await loaded_adapter.chat(
                [{"role": "user", "content": "hi"}]
            )

        assert loaded_adapter.get_session_id("default") == "ses_new_from_chat"


# ---------------------------------------------------------------------------
# Options building
# ---------------------------------------------------------------------------


class TestOptionsBuilding:
    """Test _build_options configuration."""

    def test_default_options(self, loaded_adapter):
        opts = loaded_adapter._build_options()
        assert opts.model == "claude-sonnet-4-20250514"
        assert opts.max_turns == 10
        assert opts.max_budget_usd == 2.0
        assert opts.permission_mode == "bypassPermissions"
        assert opts.effort == "high"

    def test_per_call_overrides(self, loaded_adapter):
        opts = loaded_adapter._build_options(model="claude-opus-4-6", max_turns=50)
        assert opts.model == "claude-opus-4-6"
        assert opts.max_turns == 50

    def test_system_prompt_appended(self, loaded_adapter):
        opts = loaded_adapter._build_options(system_prompt="Be helpful")
        assert opts.system_prompt == "Be helpful"

    def test_mcp_servers_injected(self, loaded_adapter):
        mock_server = MagicMock()
        loaded_adapter.set_mcp_servers({"multihead": mock_server})
        opts = loaded_adapter._build_options()
        assert "multihead" in opts.mcp_servers

    def test_resume_passed(self, loaded_adapter):
        opts = loaded_adapter._build_options(resume="ses_resume_123")
        assert opts.resume == "ses_resume_123"

    def test_allowed_tools(self, loaded_adapter):
        opts = loaded_adapter._build_options(allowed_tools=["Read", "Edit"])
        assert opts.allowed_tools == ["Read", "Edit"]


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


class TestSessionManagement:
    """Session tracking and resume."""

    def test_get_session_none_initially(self, adapter):
        assert adapter.get_session_id("conv1") is None

    def test_get_session_after_tracking(self, adapter):
        adapter._sessions["conv1"] = "ses_abc"
        assert adapter.get_session_id("conv1") == "ses_abc"

    def test_multiple_conversations(self, adapter):
        adapter._sessions["conv1"] = "ses_1"
        adapter._sessions["conv2"] = "ses_2"
        assert adapter.get_session_id("conv1") == "ses_1"
        assert adapter.get_session_id("conv2") == "ses_2"


# ---------------------------------------------------------------------------
# MCP server injection
# ---------------------------------------------------------------------------


class TestMCPServers:
    """In-process MCP tool injection."""

    def test_set_mcp_servers(self, adapter):
        servers = {"knowledge": MagicMock(), "heads": MagicMock()}
        adapter.set_mcp_servers(servers)
        assert adapter._mcp_servers == servers

    def test_mcp_servers_empty_by_default(self, adapter):
        assert adapter._mcp_servers == {}


# ---------------------------------------------------------------------------
# Healthcheck
# ---------------------------------------------------------------------------


class TestHealthcheck:
    """Health checking."""

    async def test_healthcheck_without_sdk(self, adapter):
        adapter._sdk_available = False
        result = await adapter.healthcheck()
        assert result is False

    async def test_healthcheck_with_claude_cli(self, loaded_adapter):
        with patch("shutil.which", return_value="/usr/bin/claude"):
            result = await loaded_adapter.healthcheck()
        assert result is True

    async def test_healthcheck_without_claude_cli(self, loaded_adapter):
        with patch("shutil.which", return_value=None):
            result = await loaded_adapter.healthcheck()
        assert result is False


# ---------------------------------------------------------------------------
# Sleep / Wake
# ---------------------------------------------------------------------------


class TestSleepWake:
    """Sleep/wake for API-based adapter."""

    async def test_sleep_is_noop(self, loaded_adapter):
        await loaded_adapter.sleep()
        assert loaded_adapter._loaded is True  # Still loaded

    async def test_wake_reloads(self, adapter):
        await adapter.wake()
        assert adapter._loaded is True


# ---------------------------------------------------------------------------
# HeadManager integration
# ---------------------------------------------------------------------------


class TestHeadManagerIntegration:
    """Adapter creates correctly via HeadManager factory."""

    def test_factory_creates_adapter(self):
        from multihead.head_manager import _create_adapter

        manifest = HeadManifest(
            head_id="claude-sdk-test",
            name="Claude SDK Test",
            adapter=AdapterKind.CLAUDE_AGENT_SDK,
            model="claude-sonnet-4-20250514",
            kind="llm",
        )
        adapter = _create_adapter(manifest)
        assert isinstance(adapter, ClaudeAgentSDKAdapter)
        assert adapter.manifest.head_id == "claude-sdk-test"

    def test_adapter_kind_in_enum(self):
        assert "claude_agent_sdk" in [k.value for k in AdapterKind]


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestStreaming:
    """Streaming generation."""

    async def test_generate_stream_not_loaded(self, adapter):
        with pytest.raises(RuntimeError, match="not loaded"):
            async for _ in adapter.generate_stream("test"):
                pass

    async def test_generate_stream_yields_text(self, loaded_adapter):
        async def fake_query(prompt, options):
            yield FakeAssistantMessage(content=[FakeTextBlock("chunk1")])
            yield FakeAssistantMessage(content=[FakeTextBlock("chunk2")])

        chunks = []
        with patch(f"{_MOD}.query", fake_query), \
             patch(f"{_MOD}.AssistantMessage", FakeAssistantMessage), \
             patch(f"{_MOD}.TextBlock", FakeTextBlock):
            async for chunk in loaded_adapter.generate_stream("test"):
                chunks.append(chunk)

        assert chunks == ["chunk1", "chunk2"]
