"""Tests for shell-specific slash commands (/wake, /sleep, /swap, /status, etc.)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from multihead.runtime_config import RuntimeConfig
from multihead.slash_commands import SlashCommandHandler
from multihead.tool_registry import ToolRegistry


@pytest.fixture
def handler():
    """Create a SlashCommandHandler with mocked dependencies."""
    config = RuntimeConfig()
    config_path = Path("/tmp/test_config.json")
    tr = ToolRegistry()
    hm = MagicMock()
    hm.get_states.return_value = {
        "core-llm": {"state": "active", "name": "Core LLM", "adapter": "transformers"},
        "vision-vlm": {"state": "off", "name": "Vision VLM", "adapter": "transformers"},
    }
    hm.wake_head = AsyncMock()
    hm.sleep_head = AsyncMock()
    hm.ensure_active = AsyncMock()

    ks = MagicMock()
    claim = MagicMock()
    claim.statement = "The router uses weighted scoring for head selection"
    claim.canonical = MagicMock()
    claim.canonical.claim_key = "router.scoring.weights"
    ks.list_claims.return_value = [claim]
    ks.get_presence_peers.return_value = []

    return SlashCommandHandler(
        config=config,
        config_path=config_path,
        tool_registry=tr,
        head_states_fn=hm.get_states,
        knowledge_store=ks,
        session_id="ses_test",
        project_id="multihead",
        head_manager=hm,
    )


# ---------------------------------------------------------------------------
# /wake
# ---------------------------------------------------------------------------


class TestWake:
    async def test_wake_head(self, handler):
        result = await handler.handle("/wake core-llm")
        handler.head_manager.wake_head.assert_called_once_with("core-llm")
        assert "woken up" in result

    async def test_wake_no_args(self, handler):
        result = await handler.handle("/wake")
        assert "Usage" in result

    async def test_wake_failure(self, handler):
        handler.head_manager.wake_head = AsyncMock(side_effect=RuntimeError("OOM"))
        result = await handler.handle("/wake core-llm")
        assert "Failed" in result
        assert "OOM" in result

    async def test_wake_no_head_manager(self, handler):
        handler.head_manager = None
        result = await handler.handle("/wake core-llm")
        assert "not available" in result


# ---------------------------------------------------------------------------
# /sleep
# ---------------------------------------------------------------------------


class TestSleep:
    async def test_sleep_head(self, handler):
        result = await handler.handle("/sleep core-llm")
        handler.head_manager.sleep_head.assert_called_once_with("core-llm")
        assert "sleep" in result

    async def test_sleep_no_args(self, handler):
        result = await handler.handle("/sleep")
        assert "Usage" in result

    async def test_sleep_failure(self, handler):
        handler.head_manager.sleep_head = AsyncMock(side_effect=RuntimeError("not loaded"))
        result = await handler.handle("/sleep core-llm")
        assert "Failed" in result


# ---------------------------------------------------------------------------
# /swap
# ---------------------------------------------------------------------------


class TestSwap:
    async def test_swap_head(self, handler):
        result = await handler.handle("/swap vision-vlm")
        handler.head_manager.ensure_active.assert_called_once_with("vision-vlm")
        assert "Swapped" in result
        assert "vision-vlm" in result

    async def test_swap_no_args(self, handler):
        result = await handler.handle("/swap")
        assert "Usage" in result

    async def test_swap_failure(self, handler):
        handler.head_manager.ensure_active = AsyncMock(
            side_effect=RuntimeError("head not found"),
        )
        result = await handler.handle("/swap unknown")
        assert "Failed" in result


# ---------------------------------------------------------------------------
# /status
# ---------------------------------------------------------------------------


class TestStatus:
    async def test_status_shows_heads(self, handler):
        result = await handler.handle("/status")
        assert "core-llm" in result
        assert "vision-vlm" in result
        assert "active" in result

    async def test_status_shows_knowledge(self, handler):
        result = await handler.handle("/status")
        assert "Knowledge" in result
        assert "claim" in result.lower()

    async def test_status_no_knowledge(self, handler):
        handler.knowledge_store = None
        result = await handler.handle("/status")
        assert "not connected" in result


# ---------------------------------------------------------------------------
# /knowledge
# ---------------------------------------------------------------------------


class TestKnowledge:
    async def test_knowledge_list(self, handler):
        result = await handler.handle("/knowledge")
        assert "router.scoring.weights" in result

    async def test_knowledge_search(self, handler):
        result = await handler.handle("/knowledge router")
        assert "router" in result.lower()

    async def test_knowledge_no_match(self, handler):
        handler.knowledge_store.list_claims.return_value = []
        result = await handler.handle("/knowledge foobar")
        assert "No claims" in result

    async def test_knowledge_no_store(self, handler):
        handler.knowledge_store = None
        result = await handler.handle("/knowledge")
        assert "not available" in result


# ---------------------------------------------------------------------------
# /session
# ---------------------------------------------------------------------------


class TestSession:
    async def test_session_info(self, handler):
        result = await handler.handle("/session")
        assert "ses_test" in result

    async def test_session_no_active(self, handler):
        handler.session_id = None
        result = await handler.handle("/session")
        assert "No active session" in result


# ---------------------------------------------------------------------------
# /sessions
# ---------------------------------------------------------------------------


class TestSessions:
    async def test_sessions_list(self, handler):
        result = await handler.handle("/sessions")
        assert "ses_test" in result
        assert "--session" in result


# ---------------------------------------------------------------------------
# /mesh
# ---------------------------------------------------------------------------


class TestMesh:
    async def test_mesh_no_peers(self, handler):
        result = await handler.handle("/mesh")
        assert "No mesh peers" in result

    async def test_mesh_with_peers(self, handler):
        peer = MagicMock()
        peer.node_id = "node-desktop"
        peer.status = "online"
        handler.knowledge_store.get_presence_peers.return_value = [peer]
        result = await handler.handle("/mesh")
        assert "node-desktop" in result
        assert "online" in result

    async def test_mesh_no_store(self, handler):
        handler.knowledge_store = None
        result = await handler.handle("/mesh")
        assert "not available" in result


# ---------------------------------------------------------------------------
# /help — updated list
# ---------------------------------------------------------------------------


class TestHelpUpdated:
    async def test_help_includes_new_commands(self, handler):
        result = await handler.handle("/help")
        assert "/wake" in result
        assert "/sleep" in result
        assert "/swap" in result
        assert "/status" in result
        assert "/knowledge" in result
        assert "/session" in result
        assert "/sessions" in result
        assert "/mesh" in result
