"""API routes for Conversation Harvester."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class HarvestRunRequest(BaseModel):
    max_files: int = 50


@router.get("/harvester/status")
async def harvester_status(request: Request) -> dict[str, Any]:
    """Return current harvester status (file counts, processed state, etc.)."""
    from multihead.conversation_harvester import ConversationHarvester

    harvester = ConversationHarvester(
        record_store=request.app.state.record_store,
        knowledge_store=request.app.state.knowledge_store,
        data_dir=request.app.state.settings.data_dir,
    )
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, harvester.status)


@router.post("/harvester/run")
async def harvester_run(request: Request, body: HarvestRunRequest | None = None) -> dict[str, Any]:
    """Trigger a conversation harvest run."""
    from multihead.conversation_harvester import ConversationHarvester

    max_files = body.max_files if body else 50

    harvester = ConversationHarvester(
        record_store=request.app.state.record_store,
        knowledge_store=request.app.state.knowledge_store,
        data_dir=request.app.state.settings.data_dir,
        max_files_per_run=max_files,
    )
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, harvester.harvest_all)
    return asdict(result)


# --- Backlog Sweep ---


class BacklogRunRequest(BaseModel):
    batch_size: int = 500
    reset: bool = False


@router.get("/backlog/status")
async def backlog_status(request: Request) -> dict[str, Any]:
    """Return backlog sweep cursor position and total claims."""
    night_shift = request.app.state.night_shift
    offset = night_shift._get_backlog_cursor()
    total = night_shift.knowledge.count_claims(status="accepted")
    # Find last run timestamp from marker file
    marker = night_shift.output_dir / ".last_backlog_sweep"
    last_run = None
    if marker.exists():
        import os
        from datetime import datetime, timezone
        mtime = datetime.fromtimestamp(os.path.getmtime(marker), tz=timezone.utc)
        last_run = mtime.isoformat()

    return {
        "cursor_offset": offset,
        "total_claims": total,
        "last_run": last_run,
    }


@router.post("/backlog/run")
async def backlog_run(request: Request, body: BacklogRunRequest | None = None) -> dict[str, Any]:
    """Trigger a backlog sweep run."""
    night_shift = request.app.state.night_shift
    batch_size = body.batch_size if body else 500
    reset = body.reset if body else False
    result = await night_shift.run_backlog(batch_size=batch_size, reset=reset)
    return result
