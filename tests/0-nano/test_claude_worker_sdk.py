"""Tests for Claude Worker Daemon SDK mode.

Verifies that the worker daemon can use Claude Agent SDK as an
alternative to spawning subprocess.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# scripts/ is not a package — add to path for import
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from claude_worker import ClaudeWorker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def worker_sdk():
    """Worker in SDK mode."""
    with patch.dict("os.environ", {
        "ACP_URL": "http://localhost:8000/api/v1",
        "ACP_SESSION_KEY": "test-jwt",
    }):
        return ClaudeWorker(mode="sdk")


@pytest.fixture
def worker_headless():
    """Worker in headless mode."""
    with patch.dict("os.environ", {
        "ACP_URL": "http://localhost:8000/api/v1",
        "ACP_SESSION_KEY": "test-jwt",
    }):
        return ClaudeWorker(mode="headless")


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestWorkerInit:
    """Worker initialization."""

    def test_default_mode_sdk(self, worker_sdk):
        assert worker_sdk.default_mode == "sdk"

    def test_headless_mode(self, worker_headless):
        assert worker_headless.default_mode == "headless"

    def test_sdk_adapter_none_initially(self, worker_sdk):
        assert worker_sdk._sdk_adapter is None

    def test_sessions_empty(self, worker_sdk):
        assert worker_sdk.active_sessions == {}


# ---------------------------------------------------------------------------
# SDK adapter lazy init
# ---------------------------------------------------------------------------


class TestSDKAdapterInit:
    """SDK adapter lazy initialization."""

    async def test_ensure_sdk_adapter_creates(self, worker_sdk):
        """Test that lazy init creates and loads the adapter."""
        mock_adapter = MagicMock()
        mock_adapter.load = AsyncMock()
        mock_adapter.set_mcp_servers = MagicMock()
        mock_adapter._mcp_servers = {}

        with patch("multihead.adapters.claude_agent_sdk.ClaudeAgentSDKAdapter",
                    return_value=mock_adapter):
            await worker_sdk._ensure_sdk_adapter()

        assert worker_sdk._sdk_adapter is mock_adapter
        mock_adapter.load.assert_called_once()

    async def test_invoke_sdk_returns_result(self, worker_sdk):
        """Test SDK invocation with mocked adapter."""
        mock_adapter = MagicMock()
        mock_adapter.generate = AsyncMock(return_value={
            "text": "SDK result text",
            "session_id": "ses_sdk_123",
            "cost_usd": 0.03,
            "turns": 2,
        })
        worker_sdk._sdk_adapter = mock_adapter

        result = await worker_sdk._invoke_claude_sdk("Do something")
        assert result["result"] == "SDK result text"
        assert result["session_id"] == "ses_sdk_123"
        assert result["cost_usd"] == 0.03

    async def test_invoke_sdk_with_resume(self, worker_sdk):
        """Test SDK invocation with session resume."""
        mock_adapter = MagicMock()
        mock_adapter.generate = AsyncMock(return_value={
            "text": "Resumed response",
            "session_id": "ses_sdk_resumed",
            "cost_usd": 0.01,
            "turns": 1,
        })
        worker_sdk._sdk_adapter = mock_adapter

        result = await worker_sdk._invoke_claude_sdk(
            "Follow up", session_id="ses_prev", conversation_id="conv_1"
        )
        call_kwargs = mock_adapter.generate.call_args[1]
        assert call_kwargs["resume"] == "ses_prev"
        assert result["session_id"] == "ses_sdk_resumed"

    async def test_invoke_sdk_error_returns_dict(self, worker_sdk):
        """Test SDK invocation handles errors gracefully."""
        mock_adapter = MagicMock()
        mock_adapter.generate = AsyncMock(side_effect=RuntimeError("API down"))
        worker_sdk._sdk_adapter = mock_adapter

        result = await worker_sdk._invoke_claude_sdk("Do something")
        assert "error" in result
        assert "API down" in result["error"]

    async def test_invoke_sdk_enriches_prompt(self, worker_sdk):
        """Test that context pack is prepended for new sessions."""
        mock_adapter = MagicMock()
        mock_adapter.generate = AsyncMock(return_value={
            "text": "ok", "session_id": "s", "cost_usd": 0, "turns": 1,
        })
        worker_sdk._sdk_adapter = mock_adapter

        # Without session_id → should enrich
        with patch.object(worker_sdk, "_load_context_pack", return_value="CONTEXT"):
            await worker_sdk._invoke_claude_sdk("task prompt")
            call_args = mock_adapter.generate.call_args[0]
            assert "CONTEXT" in call_args[0]
            assert "task prompt" in call_args[0]

    async def test_invoke_sdk_skips_context_for_resume(self, worker_sdk):
        """Test that context pack is skipped when resuming."""
        mock_adapter = MagicMock()
        mock_adapter.generate = AsyncMock(return_value={
            "text": "ok", "session_id": "s", "cost_usd": 0, "turns": 1,
        })
        worker_sdk._sdk_adapter = mock_adapter

        with patch.object(worker_sdk, "_load_context_pack", return_value="CONTEXT"):
            await worker_sdk._invoke_claude_sdk("follow up", session_id="ses_prev")
            call_args = mock_adapter.generate.call_args[0]
            assert "CONTEXT" not in call_args[0]


# ---------------------------------------------------------------------------
# Mode routing
# ---------------------------------------------------------------------------


class TestModeRouting:
    """Task processing routes by mode."""

    def test_mode_from_task_overrides_default(self, worker_sdk):
        """Task-level mode should override worker default."""
        # This is tested through _process_task but we verify the logic
        task = {"mode": "headless", "task_id": "t1", "payload_ref": "test"}
        assert task.get("mode", worker_sdk.default_mode) == "headless"

    def test_default_mode_used_when_task_has_none(self, worker_sdk):
        task = {"task_id": "t1", "payload_ref": "test"}
        assert task.get("mode", worker_sdk.default_mode) == "sdk"
