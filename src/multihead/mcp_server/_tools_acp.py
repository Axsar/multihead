"""ACP task and marketplace tool logic."""

from __future__ import annotations

import json

import httpx

from ._core import _request


async def _check_tasks(capability: str = "com.claude.code") -> str:
    try:
        result = await _request("GET", "/acp/tasks", params={"capability": capability, "limit": 10})
        # Trim each task to essential fields only
        if isinstance(result, list):
            result = [
                {
                    "task_id": t.get("task_id"),
                    "capability_id": t.get("capability_id") or t.get("required_capability"),
                    "status": t.get("status"),
                    "created_at": t.get("created_at"),
                }
                for t in result
            ]
        elif isinstance(result, dict) and "tasks" in result:
            result["tasks"] = [
                {
                    "task_id": t.get("task_id"),
                    "capability_id": t.get("capability_id") or t.get("required_capability"),
                    "status": t.get("status"),
                    "created_at": t.get("created_at"),
                }
                for t in result["tasks"]
            ]
        return json.dumps(result, indent=2)
    except httpx.ConnectError:
        return "Error: MultiHead server not running. Start it with: multihead serve"
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 503:
            return "ACP bridge not connected to BotVibes. Set ACP_URL and ACP_API_KEY in .env."
        return f"Error: {e}"
    except Exception as e:
        return f"Error: {e}"


async def _claim_task(task_id: str, agent_id: str = "claude_code_proxy") -> str:
    try:
        result = await _request("POST", f"/acp/tasks/{task_id}/claim", json={"agent_id": agent_id})
        return json.dumps(result, indent=2)
    except httpx.ConnectError:
        return "Error: MultiHead server not running."
    except Exception as e:
        return f"Error: {e}"


async def _complete_task(
    task_id: str,
    output_ref: str,
    status: str = "complete",
    confidence: float | None = None,
    error_message: str | None = None,
) -> str:
    payload: dict = {"status": status, "output_ref": output_ref}
    if confidence is not None:
        payload["confidence"] = confidence
    if error_message:
        payload["message"] = error_message
        payload["error_code"] = "EXECUTION_ERROR"
    try:
        result = await _request("POST", f"/acp/tasks/{task_id}/result", json=payload)
        return json.dumps(result, indent=2)
    except httpx.ConnectError:
        return "Error: MultiHead server not running."
    except Exception as e:
        return f"Error: {e}"


async def _create_task(capability: str, payload_ref: str, priority: str = "normal", target_agent_id: str | None = None, conversation_id: str | None = None) -> str:
    try:
        payload: dict = {"required_capability": capability, "payload_ref": payload_ref, "priority": priority}
        if target_agent_id:
            payload["target_agent_id"] = target_agent_id
        if conversation_id:
            payload["conversation_id"] = conversation_id
        result = await _request("POST", "/acp/tasks", json=payload)
        return json.dumps(result, indent=2)
    except httpx.ConnectError:
        return "Error: MultiHead server not running."
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 503:
            return "ACP bridge not connected to BotVibes."
        return f"Error: {e}"
    except Exception as e:
        return f"Error: {e}"


async def _delegate_claude(prompt: str, conversation_id: str | None = None, priority: str = "normal") -> str:
    try:
        payload: dict = {
            "required_capability": "com.claude.code",
            "payload_ref": prompt,
            "target_agent_id": "claude-session-agent",
            "priority": priority,
        }
        if conversation_id:
            payload["conversation_id"] = conversation_id
        result = await _request("POST", "/acp/tasks", json=payload)
        return json.dumps(result, indent=2)
    except httpx.ConnectError:
        return "Error: MultiHead server not running."
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 503:
            return "ACP bridge not connected to BotVibes."
        return f"Error: {e}"
    except Exception as e:
        return f"Error: {e}"
