"""Tests for the process manager (spawn/list/output/kill)."""

from __future__ import annotations

import asyncio
import sys

import pytest

from multihead.process_manager import ManagedProcess, ProcessManager, register_process_tools
from multihead.tool_registry import ToolRegistry

_PY = f'"{sys.executable}"'  # Quoted for shlex.split on Windows
_ECHO_HELLO = f'{_PY} -c print(42)'
_SLEEP_CMD = f"{_PY} -c __import__('time').sleep(60)"


# ---------------------------------------------------------------------------
# ManagedProcess
# ---------------------------------------------------------------------------


class TestManagedProcess:
    def test_to_dict(self):
        proc = ManagedProcess(pid=123, command="echo hello", status="running")
        d = proc.to_dict()
        assert d["pid"] == 123
        assert d["command"] == "echo hello"
        assert d["status"] == "running"
        assert "elapsed_seconds" in d

    def test_default_status(self):
        proc = ManagedProcess(pid=1, command="test")
        assert proc.status == "running"
        assert proc.exit_code is None


# ---------------------------------------------------------------------------
# Spawn
# ---------------------------------------------------------------------------


class TestSpawn:
    async def test_spawn_echo(self):
        pm = ProcessManager()
        proc = await pm.spawn(_ECHO_HELLO)
        assert proc.pid > 0
        assert proc.status == "running"
        # Give it time to finish
        await asyncio.sleep(0.2)
        assert proc.status == "exited"
        await pm.cleanup()

    async def test_spawn_tracks_process(self):
        pm = ProcessManager()
        proc = await pm.spawn(_ECHO_HELLO)
        assert pm.get(proc.pid) is proc
        await pm.cleanup()

    async def test_spawn_max_limit(self):
        pm = ProcessManager(max_processes=2)
        p1 = await pm.spawn(_SLEEP_CMD)
        p2 = await pm.spawn(_SLEEP_CMD)
        with pytest.raises(RuntimeError, match="Max 2"):
            await pm.spawn(_SLEEP_CMD)
        await pm.cleanup()

    async def test_spawn_bad_command(self):
        pm = ProcessManager()
        # With exec mode (no shell), nonexistent commands raise OSError immediately
        with pytest.raises(OSError, match="Failed to start"):
            await pm.spawn("nonexistent_command_xyz_12345")
        await pm.cleanup()


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


class TestList:
    async def test_list_empty(self):
        pm = ProcessManager()
        assert pm.list_processes() == []

    async def test_list_with_processes(self):
        pm = ProcessManager()
        await pm.spawn(_ECHO_HELLO)
        await pm.spawn(_ECHO_HELLO)
        procs = pm.list_processes()
        assert len(procs) == 2
        await pm.cleanup()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


class TestOutput:
    async def test_output_captures_stdout(self):
        pm = ProcessManager()
        proc = await pm.spawn(f'{_PY} -c print(42)')
        await asyncio.sleep(0.3)
        text = pm.output(proc.pid)
        assert "42" in text
        await pm.cleanup()

    async def test_output_unknown_pid(self):
        pm = ProcessManager()
        text = pm.output(99999)
        assert "No process" in text

    async def test_output_limit_lines(self):
        pm = ProcessManager()
        # Print 50 lines
        proc = await pm.spawn(f"""{_PY} -c "exec('for i in range(50):\\n print(i)')" """.strip())
        await asyncio.sleep(0.3)
        text = pm.output(proc.pid, lines=5)
        assert "stdout" in text
        await pm.cleanup()


# ---------------------------------------------------------------------------
# Kill
# ---------------------------------------------------------------------------


class TestKill:
    async def test_kill_running_process(self):
        pm = ProcessManager()
        proc = await pm.spawn(_SLEEP_CMD)
        result = pm.kill(proc.pid)
        assert "killed" in result.lower()
        assert proc.status == "killed"
        await pm.cleanup()

    async def test_kill_unknown_pid(self):
        pm = ProcessManager()
        result = pm.kill(99999)
        assert "No process" in result

    async def test_kill_already_exited(self):
        pm = ProcessManager()
        proc = await pm.spawn(_ECHO_HELLO)
        await asyncio.sleep(0.3)
        result = pm.kill(proc.pid)
        assert "not running" in result
        await pm.cleanup()


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    async def test_cleanup_kills_all(self):
        pm = ProcessManager()
        p1 = await pm.spawn(_SLEEP_CMD)
        p2 = await pm.spawn(_SLEEP_CMD)
        await pm.cleanup()
        assert len(pm.list_processes()) == 0


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    def test_register_tools(self):
        registry = ToolRegistry()
        pm = ProcessManager()
        register_process_tools(registry, pm)
        tools = registry.list_all_tools()
        names = [t.name for t in tools]
        assert "process.spawn" in names
        assert "process.list" in names
        assert "process.output" in names
        assert "process.kill" in names
        assert "process.send" in names

    def test_spawn_requires_approval(self):
        registry = ToolRegistry()
        pm = ProcessManager()
        register_process_tools(registry, pm)
        spec = registry.get_spec("process.spawn")
        assert spec.requires_approval is True

    def test_kill_requires_approval(self):
        registry = ToolRegistry()
        pm = ProcessManager()
        register_process_tools(registry, pm)
        spec = registry.get_spec("process.kill")
        assert spec.requires_approval is True

    def test_list_does_not_require_approval(self):
        registry = ToolRegistry()
        pm = ProcessManager()
        register_process_tools(registry, pm)
        spec = registry.get_spec("process.list")
        assert spec.requires_approval is False


# ---------------------------------------------------------------------------
# Slash command integration
# ---------------------------------------------------------------------------


class TestSlashProcessCommands:
    """Test process slash commands via SlashCommandHandler."""

    async def test_spawn_command(self):
        from multihead.slash_commands import SlashCommandHandler
        from multihead.runtime_config import RuntimeConfig
        from pathlib import Path

        pm = ProcessManager()
        handler = SlashCommandHandler(
            config=RuntimeConfig(),
            config_path=Path("/tmp/test.json"),
            tool_registry=ToolRegistry(),
            head_states_fn=lambda: {},
            process_manager=pm,
        )
        result = await handler.handle(f"/spawn {_PY} -c pass")
        assert "PID" in result
        await pm.cleanup()

    async def test_ps_empty(self):
        from multihead.slash_commands import SlashCommandHandler
        from multihead.runtime_config import RuntimeConfig
        from pathlib import Path

        pm = ProcessManager()
        handler = SlashCommandHandler(
            config=RuntimeConfig(),
            config_path=Path("/tmp/test.json"),
            tool_registry=ToolRegistry(),
            head_states_fn=lambda: {},
            process_manager=pm,
        )
        result = await handler.handle("/ps")
        assert "No tracked" in result

    async def test_ps_no_manager(self):
        from multihead.slash_commands import SlashCommandHandler
        from multihead.runtime_config import RuntimeConfig
        from pathlib import Path

        handler = SlashCommandHandler(
            config=RuntimeConfig(),
            config_path=Path("/tmp/test.json"),
            tool_registry=ToolRegistry(),
            head_states_fn=lambda: {},
        )
        result = await handler.handle("/ps")
        assert "not available" in result

    async def test_help_includes_process_commands(self):
        from multihead.slash_commands import SlashCommandHandler
        from multihead.runtime_config import RuntimeConfig
        from pathlib import Path

        handler = SlashCommandHandler(
            config=RuntimeConfig(),
            config_path=Path("/tmp/test.json"),
            tool_registry=ToolRegistry(),
            head_states_fn=lambda: {},
        )
        result = await handler.handle("/help")
        assert "/spawn" in result
        assert "/ps" in result
        assert "/output" in result
        assert "/kill" in result
