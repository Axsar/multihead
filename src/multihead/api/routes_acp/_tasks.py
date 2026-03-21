"""ACP task routes: list, create, get, claim, submit result."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, Request

from ._common import ClaimRequest, CreateTaskRequest, TaskResultRequest, _get_acp

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/tasks")
async def get_available_tasks(
    request: Request,
    capability: str = "com.claude.code",
    limit: int = 10,
):
    """Poll BotVibes for available tasks matching a capability."""
    bridge = _get_acp(request)
    try:
        result = await bridge.request(
            "GET",
            "/tasks/available",
            params={"capability": capability, "limit": limit},
        )
        # Trim each task to essential fields to avoid oversized responses
        if isinstance(result, list):
            result = [
                {
                    "task_id": t.get("task_id", ""),
                    "capability_id": t.get("capability_id", t.get("required_capability", "")),
                    "status": t.get("status", t.get("state", "")),
                    "created_at": t.get("created_at", ""),
                    "agent_id": t.get("agent_id", t.get("target_agent_id", "")),
                }
                for t in result
                if isinstance(t, dict)
            ]
        return result
    except Exception as e:
        logger.error("Failed to fetch ACP tasks: %s", e)
        raise HTTPException(502, f"BotVibes error: {e}")


@router.post("/tasks")
async def create_task(body: CreateTaskRequest, request: Request):
    """Create a new task on BotVibes for any agent to pick up."""
    bridge = _get_acp(request)
    try:
        payload: dict = {
            "project_id": bridge.project_id,
            "required_capability": body.required_capability,
            "payload_ref": body.payload_ref,
            "input_schema": body.input_schema,
            "output_schema": body.output_schema,
            "priority": body.priority,
        }
        if body.target_agent_id:
            payload["target_agent_id"] = body.target_agent_id
        if body.conversation_id:
            payload["conversation_id"] = body.conversation_id
        result = await bridge.request("POST", "/tasks", json=payload)
        return result
    except Exception as e:
        logger.error("Failed to create ACP task: %s", e)
        raise HTTPException(502, f"BotVibes error: {e}")


@router.post("/tasks/{task_id}/claim")
async def claim_task(task_id: str, request: Request, body: ClaimRequest | None = None):
    """Atomically reserve + dispatch a task (two calls in one)."""
    bridge = _get_acp(request)
    agent_id = body.agent_id if body else "claude_code_proxy"

    try:
        # Reserve
        reserve_resp = await bridge.request(
            "POST",
            f"/tasks/{task_id}/reserve",
            json={"agent_id": agent_id, "reservation_ttl_ms": 15000},
        )
        if not reserve_resp.get("accepted"):
            return {"claimed": False, "reason": reserve_resp.get("reason", "reservation rejected")}

        # Dispatch
        dispatch_resp = await bridge.request(
            "POST",
            f"/tasks/{task_id}/dispatch",
            json={"agent_id": agent_id},
        )
        return {"claimed": True, "task_id": task_id, "state": dispatch_resp.get("state", "in_progress")}
    except Exception as e:
        logger.error("Failed to claim ACP task %s: %s", task_id, e)
        raise HTTPException(502, f"BotVibes error: {e}")


@router.post("/tasks/{task_id}/result")
async def submit_result(task_id: str, body: TaskResultRequest, request: Request):
    """Submit results for a completed task."""
    bridge = _get_acp(request)
    payload: dict = {"status": body.status}
    if body.status == "complete":
        payload["output_ref"] = body.output_ref
        if body.confidence is not None:
            payload["confidence"] = body.confidence
        if body.latency_ms is not None:
            payload["latency_ms"] = body.latency_ms
        payload["subtasks"] = []
    else:
        payload["error_code"] = body.error_code or "UNKNOWN"
        payload["message"] = body.message or "Task failed"

    try:
        result = await bridge.request("POST", f"/tasks/{task_id}/result", json=payload)
        return result
    except Exception as e:
        logger.error("Failed to submit ACP result for %s: %s", task_id, e)
        raise HTTPException(502, f"BotVibes error: {e}")


@router.get("/tasks/{task_id}")
async def get_task(task_id: str, request: Request, full: bool = Query(default=False)):
    """Get details of a specific task."""
    bridge = _get_acp(request)
    try:
        data = await bridge.request("GET", f"/tasks/{task_id}")
        # Truncate potentially large fields (unless full=True)
        if not full and isinstance(data, dict):
            for field in ("payload_ref", "output_ref"):
                val = data.get(field)
                if isinstance(val, str) and len(val) > 500:
                    data[field] = val[:500] + "...[truncated]"
        return data
    except Exception as e:
        logger.error("Failed to get ACP task %s: %s", task_id, e)
        raise HTTPException(502, f"BotVibes error: {e}")
