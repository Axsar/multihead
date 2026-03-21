"""ACP marketplace execution routes: execute contracts, track progress, history."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ._common import _get_acp

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# In-memory execution state
# ---------------------------------------------------------------------------

# contractId -> execution entry dict
_executions: dict[str, dict[str, Any]] = {}

# Strong references to background tasks (prevent GC)
_running_tasks: set[asyncio.Task] = set()  # type: ignore[type-arg]

# Completed executions moved here for history queries
_execution_history: list[dict[str, Any]] = []

_MAX_HISTORY = 200


class ExecuteContractRequest(BaseModel):
    """Body for POST /marketplace/contracts/{contractId}/execute."""

    capability_id: str = ""
    description: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _phase_record(phase: str, start_ms: float) -> dict:
    return {"phase": phase, "duration_ms": round((time.monotonic() * 1000) - start_ms)}


async def _run_execution(
    contract_id: str,
    capability_id: str,
    description: str,
    request_app_state: Any,
) -> None:
    """Background task: find matching skill, create session, run, track phases."""

    entry = _executions[contract_id]
    phase_durations: list[dict] = []

    try:
        # Phase 1: match skill
        phase_start = time.monotonic() * 1000
        entry["phase"] = "matching_skill"
        skill_registry = getattr(request_app_state, "skill_registry", None)
        matched_skill = None

        if skill_registry and capability_id:
            for skill in skill_registry.list_skills():
                if skill.capability_id == capability_id:
                    matched_skill = skill
                    break

        # Fallback: try partial match on capability_id
        if not matched_skill and skill_registry and capability_id:
            cap_lower = capability_id.lower()
            for skill in skill_registry.list_skills():
                if cap_lower in (skill.capability_id or "").lower():
                    matched_skill = skill
                    break

        phase_durations.append(_phase_record("matching_skill", phase_start))

        if matched_skill:
            entry["skill_name"] = matched_skill.name
            entry["model"] = getattr(matched_skill, "metadata", {}).get("model", "")
            entry["skill_instructions_preview"] = (matched_skill.instructions or "")[:200]
        else:
            entry["skill_name"] = None

        # Phase 2: create session
        phase_start = time.monotonic() * 1000
        entry["phase"] = "creating_session"

        sessions = getattr(request_app_state, "session_manager", None)
        core = getattr(request_app_state, "agentic_core", None)

        if not sessions or not core:
            raise RuntimeError("Session manager or agentic core not available")

        session = sessions.create_session()
        entry["session_id"] = session.session_id
        phase_durations.append(_phase_record("creating_session", phase_start))

        # Phase 3: build prompt
        phase_start = time.monotonic() * 1000
        entry["phase"] = "building_prompt"

        prompt_text = description or f"Execute capability: {capability_id}"
        entry["prompt_preview"] = prompt_text[:200]

        if matched_skill:
            skill_prompt = matched_skill.build_system_prompt()
        else:
            skill_prompt = None

        phase_durations.append(_phase_record("building_prompt", phase_start))

        # Phase 4: LLM execution
        phase_start = time.monotonic() * 1000
        entry["phase"] = "executing"

        if matched_skill and skill_prompt:
            response = await core.chat_with_skill(session.session_id, prompt_text, skill_prompt)
        else:
            response = await core.chat(session.session_id, prompt_text)

        phase_durations.append(_phase_record("executing", phase_start))

        # Phase 5: delivery
        phase_start = time.monotonic() * 1000
        entry["phase"] = "delivering"

        # Try to deliver to BotVibes if bridge is connected
        bridge = getattr(request_app_state, "acp_bridge", None)
        if bridge and bridge.connected:
            try:
                import hashlib

                output_bytes = response.encode("utf-8") if response else b""
                output_hash = hashlib.sha256(output_bytes).hexdigest()
                receipt = {
                    "receipt_type": "completion",
                    "artifact_hashes": [
                        {
                            "ref": f"contract-{contract_id[:8]}-output",
                            "sha256": output_hash,
                            "size_bytes": len(output_bytes),
                        }
                    ],
                    "metrics": {
                        "outcome": "success",
                        "confidence": 0.85,
                        "latency_ms": int(
                            (time.monotonic() * 1000) - phase_start
                        ),
                        "output_preview": (response or "")[:500],
                    },
                }
                await bridge.request(
                    "POST",
                    f"/marketplace/contracts/{contract_id}/receipts",
                    json=receipt,
                )
            except Exception as deliver_err:
                logger.warning("Delivery to BotVibes failed (non-fatal): %s", deliver_err)

        phase_durations.append(_phase_record("delivering", phase_start))

        # Done
        entry["status"] = "done"
        entry["output"] = response
        entry["partial_output"] = response
        entry["finished_at"] = _now_iso()
        entry["phase"] = "complete"
        entry["phase_durations"] = phase_durations

    except Exception as exc:
        logger.error("Execution failed for contract %s: %s", contract_id, exc)
        entry["status"] = "error"
        entry["error"] = str(exc)
        entry["finished_at"] = _now_iso()
        entry["phase"] = "failed"
        entry["phase_durations"] = phase_durations

    # Archive to history
    _archive_if_finished(contract_id)


def _archive_if_finished(contract_id: str) -> None:
    """Copy finished execution to history list."""
    entry = _executions.get(contract_id)
    if not entry or entry.get("status") == "running":
        return
    history_entry = {
        "contract_id": contract_id,
        "capability_id": entry.get("capability_id", ""),
        "skill_name": entry.get("skill_name"),
        "model": entry.get("model"),
        "status": entry["status"],
        "started_at": entry["started_at"],
        "finished_at": entry.get("finished_at"),
        "duration_ms": None,
        "phase_durations": entry.get("phase_durations"),
        "output_preview": (entry.get("output") or "")[:200],
        "error": entry.get("error"),
    }
    # Compute duration
    try:
        t0 = datetime.fromisoformat(entry["started_at"])
        t1 = datetime.fromisoformat(entry.get("finished_at", entry["started_at"]))
        history_entry["duration_ms"] = int((t1 - t0).total_seconds() * 1000)
    except Exception:
        pass

    _execution_history.append(history_entry)
    # Trim history
    while len(_execution_history) > _MAX_HISTORY:
        _execution_history.pop(0)
    # Remove from active dict
    _executions.pop(contract_id, None)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/marketplace/contracts/{contract_id}/execute")
async def execute_contract(
    contract_id: str,
    body: ExecuteContractRequest,
    request: Request,
):
    """Start server-side execution of a marketplace contract.

    Creates a session, finds a matching skill, runs the LLM, and delivers.
    Execution runs in the background — poll GET /executions/{contractId} for status.
    """
    # Prevent double-execution
    existing = _executions.get(contract_id)
    if existing and existing.get("status") == "running":
        return {"contract_id": contract_id, "status": "already_running"}

    entry: dict[str, Any] = {
        "status": "running",
        "started_at": _now_iso(),
        "finished_at": None,
        "phase": "queued",
        "output": None,
        "error": None,
        "session_id": None,
        "skill_name": None,
        "run_id": None,
        "capability_id": body.capability_id,
        "description": body.description,
        "partial_output": None,
        "model": None,
        "prompt_preview": None,
        "skill_instructions_preview": None,
        "phase_durations": None,
    }
    _executions[contract_id] = entry

    # Launch background task (store reference to prevent GC)
    task = asyncio.create_task(
        _run_execution(
            contract_id,
            body.capability_id,
            body.description,
            request.app.state,
        ),
        name=f"exec-{contract_id[:8]}",
    )
    _running_tasks.add(task)
    task.add_done_callback(_running_tasks.discard)

    return {"contract_id": contract_id, "status": "running"}


@router.get("/marketplace/executions")
async def list_executions(request: Request):
    """List all running and recent executions."""
    return {"executions": dict(_executions)}


@router.get("/marketplace/executions/history")
async def execution_history(
    request: Request,
    capability_id: str | None = None,
    limit: int = 50,
):
    """Get execution history, optionally filtered by capability_id."""
    items = _execution_history
    if capability_id:
        items = [h for h in items if h.get("capability_id") == capability_id]
    # Most recent first
    items = list(reversed(items))[:limit]
    return {"history": items, "total": len(items)}


@router.get("/marketplace/executions/{contract_id}")
async def get_execution(contract_id: str, request: Request):
    """Get the status of a single execution."""
    entry = _executions.get(contract_id)
    if not entry:
        # Check history
        for h in reversed(_execution_history):
            if h.get("contract_id") == contract_id:
                return {
                    "status": h["status"],
                    "started_at": h["started_at"],
                    "finished_at": h.get("finished_at"),
                    "output_preview": h.get("output_preview"),
                    "error": h.get("error"),
                    "capability_id": h.get("capability_id"),
                    "skill_name": h.get("skill_name"),
                    "model": h.get("model"),
                    "phase_durations": h.get("phase_durations"),
                }
        raise HTTPException(404, f"No execution found for contract {contract_id}")
    return entry
