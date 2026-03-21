"""Core service manager for background services within multihead shell.

Manages named background services (auto-responder, worker daemon, etc.)
as asyncio tasks with start/stop lifecycle, health monitoring, and
config-driven auto-start. Standalone scripts remain available — this
provides in-process management for the unified shell experience.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, Literal

from ..runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)

# Grace period for shutdown (seconds)
SHUTDOWN_TIMEOUT = 5.0


@dataclass
class ServiceEntry:
    """A registered background service."""

    name: str
    factory: Callable[[], Coroutine[Any, Any, None]]
    description: str = ""
    auto_start: bool = False
    status: Literal["stopped", "running", "failed", "stopping"] = "stopped"
    started_at: float | None = None
    error: str | None = None


class ServiceManager:
    """Manages background services within the shell event loop.

    Each service is a named coroutine with start/stop lifecycle,
    health monitoring, and config-driven auto-start.
    """

    def __init__(self, runtime_config: RuntimeConfig) -> None:
        self._config = runtime_config
        self._services: dict[str, ServiceEntry] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        # Shared data for cross-service/UI access (e.g. marketplace stats)
        self.shared_data: dict[str, Any] = {}

    def register(
        self,
        name: str,
        factory: Callable[[], Coroutine[Any, Any, None]],
        description: str = "",
        auto_start: bool = False,
    ) -> None:
        """Register a service factory (not started yet)."""
        self._services[name] = ServiceEntry(
            name=name,
            factory=factory,
            description=description,
            auto_start=auto_start,
        )

    async def start(self, name: str) -> str:
        """Start a registered service. Returns status message."""
        entry = self._services.get(name)
        if entry is None:
            return f"Unknown service: {name}"

        if entry.status == "running" and name in self._tasks:
            task = self._tasks[name]
            if not task.done():
                return f"Service '{name}' is already running."

        # Launch the coroutine as a task
        entry.error = None
        try:
            coro = entry.factory()
            task = asyncio.create_task(coro, name=f"svc-{name}")
            self._tasks[name] = task
            entry.status = "running"
            entry.started_at = time.time()

            # Add a done callback to detect failures
            task.add_done_callback(lambda t, n=name: self._on_task_done(n, t))

            return f"Service '{name}' started."
        except Exception as e:
            entry.status = "failed"
            entry.error = str(e)
            return f"Failed to start '{name}': {e}"

    def _on_task_done(self, name: str, task: asyncio.Task[None]) -> None:
        """Called when a service task completes (normally or with error)."""
        entry = self._services.get(name)
        if entry is None:
            return

        if entry.status == "stopping":
            entry.status = "stopped"
            return

        if task.cancelled():
            entry.status = "stopped"
            return

        exc = task.exception()
        if exc is not None:
            entry.status = "failed"
            entry.error = str(exc)
            logger.warning("Service '%s' failed: %s", name, exc)
        else:
            entry.status = "stopped"

    async def stop(self, name: str) -> str:
        """Stop a running service gracefully."""
        entry = self._services.get(name)
        if entry is None:
            return f"Unknown service: {name}"

        task = self._tasks.get(name)
        if task is None or task.done():
            entry.status = "stopped"
            return f"Service '{name}' is not running."

        entry.status = "stopping"
        task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=SHUTDOWN_TIMEOUT)
        except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
            pass

        entry.status = "stopped"
        entry.started_at = None
        return f"Service '{name}' stopped."

    async def auto_start_all(self) -> list[str]:
        """Start all services that should auto-start based on config.

        Returns list of status messages.
        """
        messages: list[str] = []
        svc_config = getattr(self._config, "services", None)

        for name, entry in self._services.items():
            should_start = entry.auto_start
            # Override from ServicesConfig if available
            if svc_config is not None:
                config_flag = getattr(svc_config, name.replace("-", "_"), None)
                if config_flag is not None:
                    should_start = config_flag

            if should_start:
                msg = await self.start(name)
                messages.append(msg)

        return messages

    async def shutdown_all(self) -> None:
        """Graceful shutdown of all running services."""
        running = [
            name for name, entry in self._services.items()
            if entry.status == "running"
        ]
        if not running:
            return

        # Cancel all running tasks
        for name in running:
            task = self._tasks.get(name)
            if task and not task.done():
                self._services[name].status = "stopping"
                task.cancel()

        # Wait for all to finish with timeout
        tasks = [self._tasks[n] for n in running if n in self._tasks]
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                for name in running:
                    logger.warning("Service '%s' did not stop within timeout", name)

        for name in running:
            self._services[name].status = "stopped"
            self._services[name].started_at = None

    def status(self) -> list[dict[str, Any]]:
        """Return status of all registered services."""
        result = []
        for name, entry in self._services.items():
            info: dict[str, Any] = {
                "name": name,
                "status": entry.status,
                "description": entry.description,
                "auto_start": entry.auto_start,
            }
            if entry.started_at is not None:
                info["uptime_seconds"] = round(time.time() - entry.started_at, 1)
            if entry.error:
                info["error"] = entry.error
            result.append(info)
        return result

    def status_line(self) -> str:
        """One-line summary for banner display."""
        if not self._services:
            return "Services: none registered"

        parts = []
        for name, entry in self._services.items():
            if entry.status == "running":
                parts.append(f"{name} (running)")
            elif entry.status == "failed":
                parts.append(f"{name} (FAILED)")
            else:
                parts.append(f"{name} (stopped)")

        return f"Services: {', '.join(parts)}"

    @property
    def registered_names(self) -> list[str]:
        """List of registered service names."""
        return list(self._services.keys())
