"""API routes for system operations (logs, shutdown, restart)."""

from __future__ import annotations

import asyncio
import collections
import logging
import os
import subprocess
import sys
import time

from fastapi import APIRouter, Query, Request

logger = logging.getLogger(__name__)

router = APIRouter()

_start_time = time.time()


@router.get("/logs")
async def get_logs(
    request: Request,
    lines: int = Query(default=30, ge=1, le=200),
):
    """Return the last N lines from the multihead log file."""
    settings = request.app.state.settings
    log_file = settings.data_dir / "multihead.log"

    result_lines: list[str] = []
    total_lines = 0

    if log_file.is_file():
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            tail = collections.deque(maxlen=lines)
            for total_lines, line in enumerate(f, 1):  # noqa: B007
                tail.append(line.rstrip("\n"))
            result_lines = list(tail)

    return {
        "lines": result_lines,
        "total_lines": total_lines,
        "log_file": str(log_file),
    }


@router.post("/shutdown")
async def shutdown():
    """Gracefully shut down the MultiHead daemon."""
    logger.info("Shutdown requested via API")
    loop = asyncio.get_event_loop()
    loop.call_later(1.0, os._exit, 0)
    return {"status": "shutting_down"}


@router.post("/restart")
async def restart():
    """Restart the MultiHead daemon by spawning a new process then exiting."""
    logger.info("Restart requested via API")
    # Project root: go up from multihead/api/routes_system.py → multihead/ → src/ → project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    # Spawn a new server process directly (bypasses mh.cmd issues with nested shells)
    flags = (
        subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        if sys.platform == "win32"
        else 0
    )
    subprocess.Popen(
        [sys.executable, "-m", "multihead", "serve"],
        cwd=project_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
        start_new_session=True,
    )
    # Give the new process a moment to bind, then exit
    loop = asyncio.get_event_loop()
    loop.call_later(1.5, os._exit, 0)
    return {"status": "restarting"}


def get_uptime_seconds() -> float:
    """Return seconds since this module was first imported."""
    return round(time.time() - _start_time, 1)
