"""Tests for the Claude Code ACP Worker Daemon."""

from __future__ import annotations

import json
import os
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Set required env vars before import
os.environ.setdefault("ACP_URL", "http://localhost:8000/api/v1")
os.environ.setdefault("ACP_SESSION_KEY", "test-jwt-token")

from scripts.claude_worker import ClaudeWorker


def _mock_async_client(post_responses):
    """Create a properly mocked httpx.AsyncClient for async with."""
    mock_client = MagicMock()
    mock_client.post = AsyncMock(side_effect=post_responses)

    async def aenter(*args, **kwargs):
        return mock_client

    async def aexit(*args, **kwargs):
        return False

    mock_client.__aenter__ = aenter
    mock_client.__aexit__ = aexit
    return mock_client


@pytest.fixture
def worker():
    return ClaudeWorker(mode="headless", tmux_target="test-claude")


class TestClaudeWorkerInit:
    def test_default_config(self, worker):
        assert worker.agent_id == "claude-session-agent"
        assert worker.default_mode == "headless"
        assert worker.tmux_target == "test-claude"
        assert worker.active_sessions == {}
        assert worker._seen == set()

    def test_interactive_mode(self):
        w = ClaudeWorker(mode="interactive")
        assert w.default_mode == "interactive"


class TestWSUrl:
    def test_strips_api_v1(self, worker):
        worker.acp_url = "http://localhost:8000/api/v1"
        url = worker._ws_url()
        assert url.startswith("ws://localhost:8000/ws/agents/claude-session-agent?token=")
        assert "/api/v1/" not in url

    def test_https_to_wss(self, worker):
        worker.acp_url = "https://example.com/api/v1"
        url = worker._ws_url()
        assert url.startswith("wss://example.com/ws/agents/")


class TestHeaders:
    def test_auth_header(self, worker):
        worker.api_key = "my-jwt"
        headers = worker._headers()
        assert headers["Authorization"] == "Bearer my-jwt"
        assert headers["Content-Type"] == "application/json"


class TestInvokeClaude:
    @patch("shutil.which")
    def test_claude_not_found(self, mock_which, worker):
        mock_which.return_value = None
        result = worker._invoke_claude("test prompt")
        assert "error" in result
        assert "not found" in result["error"]

    @patch("subprocess.run")
    @patch("shutil.which", return_value="/usr/bin/claude")
    def test_successful_invocation(self, mock_which, mock_run, worker):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "session_id": "ses-123",
                "result": "The answer is 4",
                "cost_usd": 0.01,
            }),
            stderr="",
        )
        result = worker._invoke_claude("What is 2+2?")
        assert result["session_id"] == "ses-123"
        assert result["result"] == "The answer is 4"

        # Verify the command was constructed correctly
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/usr/bin/claude"
        assert "-p" in cmd
        # Prompt is the arg after -p; may include context pack prefix
        p_idx = cmd.index("-p")
        prompt_arg = cmd[p_idx + 1]
        assert "What is 2+2?" in prompt_arg
        assert "--output-format" in cmd
        assert "json" in cmd
        assert "--permission-mode" in cmd
        assert "acceptEdits" in cmd

    @patch("subprocess.run")
    @patch("shutil.which", return_value="/usr/bin/claude")
    def test_resume_session(self, mock_which, mock_run, worker):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"session_id": "ses-123", "result": "Continued"}),
            stderr="",
        )
        result = worker._invoke_claude("follow up", session_id="ses-123")
        cmd = mock_run.call_args[0][0]
        assert "--resume" in cmd
        assert "ses-123" in cmd

    @patch("subprocess.run")
    @patch("shutil.which", return_value="/usr/bin/claude")
    def test_nonzero_exit(self, mock_which, mock_run, worker):
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Something went wrong",
        )
        result = worker._invoke_claude("bad prompt")
        assert "error" in result
        assert "Exit 1" in result["error"]

    @patch("subprocess.run")
    @patch("shutil.which", return_value="/usr/bin/claude")
    def test_timeout(self, mock_which, mock_run, worker):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=300)
        result = worker._invoke_claude("slow prompt")
        assert "error" in result
        assert "timed out" in result["error"]

    @patch("subprocess.run")
    @patch("shutil.which", return_value="/usr/bin/claude")
    def test_empty_output(self, mock_which, mock_run, worker):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = worker._invoke_claude("prompt")
        assert "error" in result
        assert "Empty" in result["error"]

    @patch("subprocess.run")
    @patch("shutil.which", return_value="/usr/bin/claude")
    def test_non_json_output(self, mock_which, mock_run, worker):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Some text output that is not JSON",
            stderr="",
        )
        result = worker._invoke_claude("prompt")
        assert "result" in result or "error" in result

    @patch("subprocess.run")
    @patch("shutil.which", return_value="/usr/bin/claude")
    def test_list_output_format(self, mock_which, mock_run, worker):
        """claude -p --output-format json returns a list of message objects."""
        messages = [
            {"type": "system", "subtype": "init", "session_id": "ses-456", "cwd": "/tmp"},
            {"type": "assistant", "message": {
                "content": [{"type": "text", "text": "The answer is 4."}],
            }},
            {"type": "result", "result": "The answer is 4.",
             "cost_usd": 0.01, "is_error": False,
             "session_id": "ses-456"},
        ]
        mock_run.return_value = MagicMock(
            returncode=0, stdout=json.dumps(messages), stderr="",
        )
        result = worker._invoke_claude("What is 2+2?")
        assert result["session_id"] == "ses-456"
        assert result["result"] == "The answer is 4."
        assert result["cost_usd"] == 0.01

    @patch("subprocess.run")
    @patch("shutil.which", return_value="/usr/bin/claude")
    def test_list_output_no_result_msg(self, mock_which, mock_run, worker):
        """List format without a 'result' type message uses assistant content."""
        messages = [
            {"type": "system", "subtype": "init", "session_id": "ses-789"},
            {"type": "assistant", "message": {
                "content": [{"type": "text", "text": "Hello world"}],
            }},
        ]
        mock_run.return_value = MagicMock(
            returncode=0, stdout=json.dumps(messages), stderr="",
        )
        result = worker._invoke_claude("say hi")
        assert result["session_id"] == "ses-789"
        assert result["result"] == "Hello world"

    @patch("subprocess.run")
    @patch("shutil.which", return_value="/usr/bin/claude")
    def test_env_claudecode_unset(self, mock_which, mock_run, worker):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"result": "ok"}),
            stderr="",
        )
        worker._invoke_claude("test")
        env = mock_run.call_args[1].get("env", {})
        assert "CLAUDECODE" not in env


class TestNormalizeCloudeOutput:
    def test_with_result_message(self):
        messages = [
            {"type": "system", "subtype": "init", "session_id": "s1"},
            {"type": "assistant", "message": {
                "content": [{"type": "text", "text": "thinking..."}],
            }},
            {"type": "result", "result": "Final answer", "cost_usd": 0.02, "is_error": False},
        ]
        out = ClaudeWorker._normalize_claude_output(messages)
        assert out["session_id"] == "s1"
        assert out["result"] == "Final answer"
        assert out["cost_usd"] == 0.02
        assert out["is_error"] is False

    def test_without_result_message(self):
        messages = [
            {"type": "system", "subtype": "init", "session_id": "s2"},
            {"type": "assistant", "message": {"content": [
                {"type": "text", "text": "Part 1"},
                {"type": "text", "text": "Part 2"},
            ]}},
        ]
        out = ClaudeWorker._normalize_claude_output(messages)
        assert out["session_id"] == "s2"
        assert out["result"] == "Part 1\nPart 2"

    def test_empty_messages(self):
        out = ClaudeWorker._normalize_claude_output([])
        assert out.get("result") == "No output"


class TestWakeInteractive:
    @patch("subprocess.run")
    def test_success(self, mock_run, worker):
        # has-session succeeds, send-keys succeeds
        mock_run.return_value = MagicMock(returncode=0)
        result = worker._wake_interactive("check inbox")
        assert "result" in result
        assert "tmux" in result["result"].lower() or "Sent" in result["result"]

    @patch("subprocess.run")
    def test_session_not_found(self, mock_run, worker):
        mock_run.return_value = MagicMock(returncode=1)
        result = worker._wake_interactive("test")
        assert "error" in result
        assert "not found" in result["error"]

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_tmux_not_installed(self, mock_run, worker):
        result = worker._wake_interactive("test")
        assert "error" in result
        assert "not installed" in result["error"]


class TestPollOnceSync:
    @patch("httpx.Client")
    def test_successful_poll(self, mock_client_class, worker):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"task_id": "t1", "payload_ref": "hello"}]
        mock_client.get.return_value = mock_resp
        mock_client_class.return_value = mock_client

        tasks = worker._poll_once_sync()
        assert len(tasks) == 1
        assert tasks[0]["task_id"] == "t1"

    @patch("httpx.Client")
    def test_empty_poll(self, mock_client_class, worker):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        mock_client.get.return_value = mock_resp
        mock_client_class.return_value = mock_client

        tasks = worker._poll_once_sync()
        assert tasks == []

    @patch("httpx.Client")
    def test_auth_error(self, mock_client_class, worker):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"
        mock_client.get.return_value = mock_resp
        mock_client_class.return_value = mock_client

        tasks = worker._poll_once_sync()
        assert tasks == []


class TestProcessTask:
    @pytest.mark.asyncio
    @patch.object(ClaudeWorker, "_invoke_claude")
    async def test_headless_task(self, mock_invoke, worker):
        mock_invoke.return_value = {
            "session_id": "ses-abc",
            "result": "Done! Changed 3 files.",
            "cost_usd": 0.05,
        }

        task = {
            "task_id": "task-001-full-uuid",
            "payload_ref": "refactor the auth module",
            "conversation_id": "conv-001",
        }

        # Reserve, dispatch, result submission responses
        reserve_resp = MagicMock(status_code=200)
        reserve_resp.json.return_value = {"accepted": True}
        dispatch_resp = MagicMock(status_code=200)
        result_resp = MagicMock(status_code=200)

        mock_client = _mock_async_client([reserve_resp, dispatch_resp, result_resp])

        with patch("scripts.claude_worker.httpx.AsyncClient", return_value=mock_client):
            await worker._process_task(task)

        # Verify session tracking
        assert worker.active_sessions["conv-001"] == "ses-abc"
        assert "task-001-full-uuid" in worker._seen

    @pytest.mark.asyncio
    @patch.object(ClaudeWorker, "_wake_interactive")
    async def test_interactive_task(self, mock_wake, worker):
        worker.default_mode = "interactive"
        mock_wake.return_value = {"result": "Sent to tmux session 'claude'"}

        task = {
            "task_id": "task-002-full-uuid",
            "payload_ref": "check your inbox",
        }

        reserve_resp = MagicMock(status_code=200)
        reserve_resp.json.return_value = {"accepted": True}
        dispatch_resp = MagicMock(status_code=200)
        result_resp = MagicMock(status_code=200)

        mock_client = _mock_async_client([reserve_resp, dispatch_resp, result_resp])

        with patch("scripts.claude_worker.httpx.AsyncClient", return_value=mock_client):
            await worker._process_task(task)

        assert "task-002-full-uuid" in worker._seen
        mock_wake.assert_called_once_with("check your inbox")

    @pytest.mark.asyncio
    async def test_skip_seen_task(self, worker):
        worker._seen.add("task-already-done")
        task = {"task_id": "task-already-done", "payload_ref": "test"}
        # Should return without doing anything
        await worker._process_task(task)
        # No error raised = success
