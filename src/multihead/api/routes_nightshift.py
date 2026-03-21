"""API routes for Night Shift pipeline."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Query, Request

router = APIRouter()

# Track running night shift task (protected by lock)
_nightshift_lock = asyncio.Lock()
_nightshift_status: dict[str, Any] = {
    "running": False,
    "last_report": None,
    "current_stage": None,
    "progress": deque(maxlen=100),
}


@router.post("/trigger")
async def trigger_nightshift(
    request: Request,
    background_tasks: BackgroundTasks,
    debug: bool = Query(False, description="Enable progress tracking"),
    concurrency: int = Query(1, description="Parallel LLM calls per stage (1=sequential)"),
) -> dict[str, Any]:
    """Trigger a Night Shift run in the background."""
    async with _nightshift_lock:
        if _nightshift_status["running"]:
            return {"status": "already_running"}
        _nightshift_status["running"] = True
        _nightshift_status["current_stage"] = None
        _nightshift_status["progress"] = deque(maxlen=100)

    night_shift = request.app.state.night_shift
    night_shift.config.concurrency = max(1, concurrency)

    if debug:
        def _progress_cb(evt: dict) -> None:
            _nightshift_status["progress"].append(evt)
            if evt.get("event") == "stage_start":
                _nightshift_status["current_stage"] = evt.get("stage")
            elif evt.get("event") == "complete":
                _nightshift_status["current_stage"] = None
        night_shift.on_progress = _progress_cb
    else:
        night_shift.on_progress = None

    async def _run():
        try:
            report = await night_shift.run()
            async with _nightshift_lock:
                _nightshift_status["last_report"] = report.model_dump(mode="json")
        finally:
            async with _nightshift_lock:
                _nightshift_status["running"] = False
                _nightshift_status["current_stage"] = None

    background_tasks.add_task(_run)
    return {"status": "triggered"}


@router.get("/status")
async def nightshift_status() -> dict[str, Any]:
    """Get Night Shift status including live progress."""
    return {
        "running": _nightshift_status["running"],
        "has_last_report": _nightshift_status["last_report"] is not None,
        "current_stage": _nightshift_status["current_stage"],
        "progress": list(_nightshift_status["progress"]),
    }


@router.get("/report")
async def nightshift_report() -> dict[str, Any]:
    """Get the last Night Shift report."""
    report = _nightshift_status["last_report"]
    if report is None:
        return {"status": "no_report"}
    return report
