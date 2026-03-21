"""Service manager for background services within multihead shell.

Manages named background services (auto-responder, worker daemon, etc.)
as asyncio tasks with start/stop lifecycle, health monitoring, and
config-driven auto-start. Standalone scripts remain available — this
provides in-process management for the unified shell experience.
"""

from .manager import SHUTDOWN_TIMEOUT, ServiceEntry, ServiceManager
from .services import (
    auto_responder_service,
    cloud_marketplace_service,
    event_watcher_service,
    night_shift_service,
    session_harvester_service,
    worker_daemon_service,
)

__all__ = [
    "SHUTDOWN_TIMEOUT",
    "ServiceEntry",
    "ServiceManager",
    "auto_responder_service",
    "cloud_marketplace_service",
    "event_watcher_service",
    "night_shift_service",
    "session_harvester_service",
    "worker_daemon_service",
]
