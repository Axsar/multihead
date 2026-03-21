"""Process manager for the MultiHead Agent Terminal.

Tracks subprocesses spawned from the shell (e.g., claude -p sessions,
scripts, build commands). Provides spawn/list/output/kill operations
with output buffering and automatic cleanup.
"""

from __future__ import annotations

import asyncio
import logging
import shlex
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .tool_registry import ToolRegistry, ToolSpec

logger = logging.getLogger(__name__)

MAX_PROCESSES = 10
OUTPUT_BUFFER_SIZE = 100


@dataclass
class ManagedProcess:
    """A tracked subprocess."""

    pid: int
    command: str
    started_at: float = field(default_factory=time.time)
    status: str = "running"  # running, exited, killed, failed
    exit_code: int | None = None
    stdout_lines: deque[str] = field(default_factory=lambda: deque(maxlen=OUTPUT_BUFFER_SIZE))
    stderr_lines: deque[str] = field(default_factory=lambda: deque(maxlen=OUTPUT_BUFFER_SIZE))
    _process: asyncio.subprocess.Process | None = field(default=None, repr=False)
    _reader_task: asyncio.Task[None] | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for display."""
        elapsed = time.time() - self.started_at
        return {
            "pid": self.pid,
            "command": self.command,
            "status": self.status,
            "exit_code": self.exit_code,
            "elapsed_seconds": round(elapsed, 1),
            "stdout_lines": len(self.stdout_lines),
        }


class ProcessManager:
    """Manage subprocesses spawned from the shell.

    Usage:
        pm = ProcessManager()
        proc = await pm.spawn("echo hello")
        procs = pm.list_processes()
        output = pm.output(proc.pid)
        pm.kill(proc.pid)
        await pm.cleanup()
    """

    def __init__(self, max_processes: int = MAX_PROCESSES) -> None:
        self._processes: dict[int, ManagedProcess] = {}
        self._max = max_processes

    async def spawn(self, command: str, cwd: str = ".") -> ManagedProcess:
        """Start a subprocess and track it.

        Raises:
            RuntimeError: If max process limit reached.
            OSError: If command fails to start.
        """
        # Clean up finished processes first
        self._reap()

        running = sum(1 for p in self._processes.values() if p.status == "running")
        if running >= self._max:
            raise RuntimeError(
                f"Max {self._max} concurrent processes. Kill one first (/kill <pid>)."
            )

        try:
            args = shlex.split(command)
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
        except ValueError as e:
            raise OSError(f"Invalid command syntax: {e}") from e
        except Exception as e:
            raise OSError(f"Failed to start: {e}") from e

        managed = ManagedProcess(
            pid=proc.pid,
            command=command,
            _process=proc,
        )
        self._processes[proc.pid] = managed

        # Start background reader for stdout/stderr
        managed._reader_task = asyncio.create_task(
            self._read_output(managed),
        )

        logger.info("Spawned process %d: %s", proc.pid, command)
        return managed

    def list_processes(self) -> list[ManagedProcess]:
        """List all tracked processes."""
        self._reap()
        return list(self._processes.values())

    def get(self, pid: int) -> ManagedProcess | None:
        """Get a managed process by PID."""
        return self._processes.get(pid)

    def output(self, pid: int, lines: int = 20) -> str:
        """Read recent output from a process."""
        proc = self._processes.get(pid)
        if proc is None:
            return f"No process with PID {pid}."

        self._check_status(proc)

        stdout = list(proc.stdout_lines)[-lines:] if proc.stdout_lines else []
        stderr = list(proc.stderr_lines)[-lines:] if proc.stderr_lines else []

        parts = []
        if stdout:
            parts.append("stdout:\n" + "".join(stdout))
        if stderr:
            parts.append("stderr:\n" + "".join(stderr))
        if not parts:
            return f"[PID {pid}] No output yet."

        status_line = f"[PID {pid} | {proc.status}]"
        return status_line + "\n" + "\n".join(parts)

    def kill(self, pid: int) -> str:
        """Kill a tracked process."""
        proc = self._processes.get(pid)
        if proc is None:
            return f"No process with PID {pid}."

        if proc.status != "running" or proc._process is None:
            return f"Process {pid} is not running (status: {proc.status})."

        try:
            proc._process.kill()
            proc.status = "killed"
            logger.info("Killed process %d", pid)
            return f"Process {pid} killed."
        except Exception as e:
            return f"Failed to kill {pid}: {e}"

    async def send(self, pid: int, text: str) -> str:
        """Send input to a running process."""
        proc = self._processes.get(pid)
        if proc is None:
            return f"No process with PID {pid}."

        if proc.status != "running" or proc._process is None:
            return f"Process {pid} is not running."

        if proc._process.stdin is None:
            return f"Process {pid} has no stdin (not interactive)."

        try:
            proc._process.stdin.write((text + "\n").encode())
            await proc._process.stdin.drain()
            return f"Sent to PID {pid}: {text}"
        except Exception as e:
            return f"Failed to send to {pid}: {e}"

    async def cleanup(self) -> None:
        """Kill all running processes. Called on shell exit."""
        for proc in list(self._processes.values()):
            if proc.status == "running" and proc._process:
                try:
                    proc._process.kill()
                    proc.status = "killed"
                except Exception:
                    pass
            if proc._reader_task and not proc._reader_task.done():
                proc._reader_task.cancel()
                try:
                    await proc._reader_task
                except asyncio.CancelledError:
                    pass
        self._processes.clear()

    # -------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------

    def _reap(self) -> None:
        """Check and update status of running processes."""
        for proc in self._processes.values():
            self._check_status(proc)

    def _check_status(self, proc: ManagedProcess) -> None:
        """Update status from subprocess returncode."""
        if proc.status != "running" or proc._process is None:
            return
        rc = proc._process.returncode
        if rc is not None:
            proc.exit_code = rc
            proc.status = "exited"

    async def _read_output(self, proc: ManagedProcess) -> None:
        """Background task to read stdout/stderr into buffers."""
        try:
            if proc._process is None:
                return

            async def _drain(stream: asyncio.StreamReader | None, buf: deque[str]) -> None:
                if stream is None:
                    return
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    buf.append(line.decode(errors="replace"))

            await asyncio.gather(
                _drain(proc._process.stdout, proc.stdout_lines),
                _drain(proc._process.stderr, proc.stderr_lines),
            )

            # Wait for process to finish
            await proc._process.wait()
            proc.exit_code = proc._process.returncode
            proc.status = "exited"

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug("Output reader error for PID %d: %s", proc.pid, e)
            proc.status = "failed"


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

_PROCESS_SPAWN_SPEC = ToolSpec(
    name="process.spawn",
    description="Start a background subprocess",
    params_schema={
        "command": {"type": "string", "required": True, "description": "Shell command to run"},
        "cwd": {"type": "string", "description": "Working directory (default: current)"},
    },
    requires_approval=True,
)

_PROCESS_LIST_SPEC = ToolSpec(
    name="process.list",
    description="List tracked subprocesses",
    params_schema={},
)

_PROCESS_OUTPUT_SPEC = ToolSpec(
    name="process.output",
    description="Read recent output from a subprocess",
    params_schema={
        "pid": {"type": "integer", "required": True, "description": "Process ID"},
        "lines": {"type": "integer", "description": "Number of lines (default: 20)"},
    },
)

_PROCESS_KILL_SPEC = ToolSpec(
    name="process.kill",
    description="Kill a tracked subprocess",
    params_schema={
        "pid": {"type": "integer", "required": True, "description": "Process ID"},
    },
    requires_approval=True,
)

_PROCESS_SEND_SPEC = ToolSpec(
    name="process.send",
    description="Send input to a running subprocess",
    params_schema={
        "pid": {"type": "integer", "required": True, "description": "Process ID"},
        "text": {"type": "string", "required": True, "description": "Text to send"},
    },
)


def register_process_tools(registry: ToolRegistry, pm: ProcessManager) -> None:
    """Register process management tools in the tool registry."""

    async def _spawn(params: dict[str, Any]) -> Any:
        from .tool_registry import ToolResult
        command = params.get("command", "")
        cwd = params.get("cwd", ".")
        try:
            proc = await pm.spawn(command, cwd)
            return ToolResult(
                tool="process.spawn", success=True,
                output=f"Started PID {proc.pid}: {command}",
            )
        except Exception as e:
            return ToolResult(tool="process.spawn", success=False, error=str(e))

    async def _list(params: dict[str, Any]) -> Any:
        from .tool_registry import ToolResult
        procs = pm.list_processes()
        if not procs:
            return ToolResult(tool="process.list", success=True, output="No processes.")
        lines = [f"PID {p.pid}: {p.command} [{p.status}]" for p in procs]
        return ToolResult(tool="process.list", success=True, output="\n".join(lines))

    async def _output(params: dict[str, Any]) -> Any:
        from .tool_registry import ToolResult
        pid = int(params.get("pid", 0))
        lines = int(params.get("lines", 20))
        text = pm.output(pid, lines)
        return ToolResult(tool="process.output", success=True, output=text)

    async def _kill(params: dict[str, Any]) -> Any:
        from .tool_registry import ToolResult
        pid = int(params.get("pid", 0))
        text = pm.kill(pid)
        return ToolResult(tool="process.kill", success=True, output=text)

    async def _send(params: dict[str, Any]) -> Any:
        from .tool_registry import ToolResult
        pid = int(params.get("pid", 0))
        text = params.get("text", "")
        result = await pm.send(pid, text)
        return ToolResult(tool="process.send", success=True, output=result)

    registry.register(_PROCESS_SPAWN_SPEC, _spawn)
    registry.register(_PROCESS_LIST_SPEC, _list)
    registry.register(_PROCESS_OUTPUT_SPEC, _output)
    registry.register(_PROCESS_KILL_SPEC, _kill)
    registry.register(_PROCESS_SEND_SPEC, _send)
