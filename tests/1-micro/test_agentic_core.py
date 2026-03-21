"""Tests for VRAM policy and Agentic Core."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from multihead.agentic_core import AgenticCore
from multihead.artifact_store import ArtifactStore
from multihead.event_store import EventStore
from multihead.head_manager import HeadManager
from multihead.models import AdapterKind, HeadManifest
from multihead.orchestrator import Orchestrator
from multihead.session import SessionManager
from multihead.tool_registry import ToolRegistry
from multihead.vram_policy import VRAMManager, VRAMPolicy


@pytest.fixture
def head_manager():
    manifests = {
        "core-llm": HeadManifest(
            head_id="core-llm", name="Core", adapter=AdapterKind.MOCK,
            model="mock-v1", kind="llm", gpu_required=False,
        ),
        "worker-llm": HeadManifest(
            head_id="worker-llm", name="Worker", adapter=AdapterKind.MOCK,
            model="mock-v1", kind="llm", gpu_required=False,
        ),
    }
    return HeadManager(manifests)


@pytest.fixture
def vram_manager(head_manager):
    policy = VRAMPolicy(core_mode="keep_loaded")
    return VRAMManager(head_manager, policy, core_head_id="core-llm")


@pytest.fixture
def core(head_manager, vram_manager, tmp_path):
    artifact_store = ArtifactStore(tmp_path / "artifacts", tmp_path / "artifacts.db")
    event_store = EventStore(tmp_path / "runs", tmp_path / "state.db")
    orchestrator = Orchestrator(event_store, artifact_store, head_manager, tmp_path / "runs")
    tool_registry = ToolRegistry()
    session_manager = SessionManager(tmp_path / "sessions")

    # On Windows, bypass _normalize_path WSL conversion so approval-flow
    # tests that write files to tmp_path can actually reach them.
    if sys.platform == "win32":
        @staticmethod
        def _identity_path(raw: str) -> Path:
            return Path(raw.strip().strip('"').strip("'"))

        patcher = patch.object(ToolRegistry, "_normalize_path", _identity_path)
        patcher.start()

    c = AgenticCore(
        head_manager=head_manager,
        orchestrator=orchestrator,
        tool_registry=tool_registry,
        session_manager=session_manager,
        vram_manager=vram_manager,
        core_head_id="core-llm",
    )
    yield c

    if sys.platform == "win32":
        patcher.stop()


# -------------------------------------------------------------------
# VRAM Policy
# -------------------------------------------------------------------

class TestVRAMPolicy:
    def test_policy_defaults(self):
        policy = VRAMPolicy()
        assert policy.core_mode == "keep_loaded"
        assert policy.worker_load_policy == "per_stage"

    @pytest.mark.asyncio
    async def test_ensure_core_available(self, vram_manager):
        await vram_manager.ensure_core_available()
        assert vram_manager.heads.active_head == "core-llm"

    @pytest.mark.asyncio
    async def test_prepare_for_batch_keep_loaded(self, vram_manager):
        await vram_manager.ensure_core_available()
        await vram_manager.prepare_for_batch("worker-llm")
        # With keep_loaded, worker gets loaded (swapping core due to GPU mutex)
        assert vram_manager.heads.active_head == "worker-llm"

    @pytest.mark.asyncio
    async def test_restore_after_batch(self, head_manager):
        policy = VRAMPolicy(core_mode="unload_during_batch")
        vm = VRAMManager(head_manager, policy, core_head_id="core-llm")

        await vm.ensure_core_available()
        assert head_manager.active_head == "core-llm"

        await vm.prepare_for_batch("worker-llm")
        assert head_manager.active_head == "worker-llm"

        await vm.restore_after_batch()
        assert head_manager.active_head == "core-llm"

    def test_get_vram_status(self, vram_manager):
        status = vram_manager.get_vram_status()
        assert status["core_head_id"] == "core-llm"
        assert "policy" in status
        assert "heads" in status


# -------------------------------------------------------------------
# Agentic Core
# -------------------------------------------------------------------

class TestAgenticCore:
    @pytest.mark.asyncio
    async def test_chat_basic(self, core):
        session = core.sessions.create_session()
        response = await core.chat(session.session_id, "Hello!")
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_chat_adds_messages(self, core):
        session = core.sessions.create_session()
        await core.chat(session.session_id, "Hello!")

        reloaded = core.sessions.get_session(session.session_id)
        # user message + assistant response
        assert len(reloaded.messages) >= 2
        assert reloaded.messages[0].role == "user"
        assert reloaded.messages[-1].role == "assistant"

    @pytest.mark.asyncio
    async def test_core_loop_depth_limit(self, core):
        """The core should stop after MAX_TOOL_LOOP_DEPTH."""
        session = core.sessions.create_session()
        # Just verify it doesn't hang — mock adapter always returns SAY-like text
        response = await core.chat(session.session_id, "Do something complex")
        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_format_messages(self):
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
        ]
        result = AgenticCore._format_messages(messages)
        assert "[system]" in result
        assert "[user]" in result
        assert "Be helpful" in result

    @pytest.mark.asyncio
    async def test_format_question(self):
        from multihead.action_types import PauseAndAskAction
        action = PauseAndAskAction(question="Pick one:", options=["A", "B"])
        result = AgenticCore._format_question(action)
        assert "Pick one:" in result
        assert "1. A" in result
        assert "2. B" in result

    @pytest.mark.asyncio
    async def test_monitor_nonexistent_run(self, core):
        from multihead.action_types import MonitorWorkOrderAction
        action = MonitorWorkOrderAction(run_id="nonexistent")
        result = await core._handle_monitor_workorder(action)
        assert "not found" in result.lower()


# -------------------------------------------------------------------
# Tool Approval Flow
# -------------------------------------------------------------------


class TestApprovalFlow:
    @pytest.mark.asyncio
    async def test_approval_requested_for_dangerous_tool(self, core):
        """Tools with requires_approval should ask instead of executing."""
        from multihead.action_types import CallToolAction

        session = core.sessions.create_session()

        # Directly test the execute path with a dangerous tool
        action = CallToolAction(tool="shell.run", params={"command": "echo hi"})
        result = await core._execute_action(session.session_id, action, depth=0)

        assert "requires approval" in result.lower()
        assert "shell.run" in result

    @pytest.mark.asyncio
    async def test_pending_approval_stored(self, core):
        """After requesting approval, pending action is in session metadata."""
        from multihead.action_types import CallToolAction

        session = core.sessions.create_session()
        action = CallToolAction(tool="files.write", params={"path": "/tmp/x", "content": "y"})
        await core._execute_action(session.session_id, action, depth=0)

        pending = core._get_pending_approval(session.session_id)
        assert pending is not None
        assert pending["tool"] == "files.write"

    @pytest.mark.asyncio
    async def test_approval_accepted(self, core, tmp_path):
        """User approving should execute the tool."""
        from multihead.action_types import CallToolAction

        session = core.sessions.create_session()
        target = tmp_path / "approved.txt"

        # Request approval for files.write
        action = CallToolAction(
            tool="files.write",
            params={"path": str(target), "content": "written!"},
        )
        await core._execute_action(session.session_id, action, depth=0)

        # Verify pending exists
        assert core._get_pending_approval(session.session_id) is not None

        # User says "yes"
        response = await core.chat(session.session_id, "yes")

        # File should have been written
        assert target.exists()
        assert target.read_text() == "written!"

    @pytest.mark.asyncio
    async def test_approval_rejected(self, core, tmp_path):
        """User rejecting should cancel the tool."""
        from multihead.action_types import CallToolAction

        session = core.sessions.create_session()
        target = tmp_path / "rejected.txt"

        action = CallToolAction(tool="files.write", params={"path": str(target), "content": "nope"})
        await core._execute_action(session.session_id, action, depth=0)

        # User says "no"
        response = await core.chat(session.session_id, "no")

        assert "cancelled" in response.lower()
        assert not target.exists()

    @pytest.mark.asyncio
    async def test_safe_tool_no_approval(self, core):
        """Tools without requires_approval should execute immediately."""
        from multihead.action_types import CallToolAction

        session = core.sessions.create_session()
        # files.read doesn't require approval
        action = CallToolAction(tool="files.read", params={"path": "/nonexistent"})
        result = await core._execute_action(session.session_id, action, depth=0)

        # Should NOT ask for approval (file doesn't exist, but should try to read)
        assert "requires approval" not in result.lower()

    @pytest.mark.asyncio
    async def test_no_pending_on_fresh_session(self, core):
        """Fresh session should have no pending approval."""
        session = core.sessions.create_session()
        assert core._get_pending_approval(session.session_id) is None

    @pytest.mark.asyncio
    async def test_approval_words(self, core, tmp_path):
        """Various approval words should be recognized."""
        from multihead.action_types import CallToolAction

        for word in ["yes", "approve", "confirm", "ok", "proceed"]:
            session = core.sessions.create_session()
            target = tmp_path / f"approved_{word}.txt"

            action = CallToolAction(
                tool="files.write",
                params={"path": str(target), "content": word},
            )
            await core._execute_action(session.session_id, action, depth=0)

            response = await core.chat(session.session_id, word)
            assert target.exists(), f"'{word}' should be recognized as approval"
