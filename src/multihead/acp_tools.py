"""ACP tools for the Agentic Core's tool registry.

Gives the local LLM the ability to create tasks on BotVibes for other
agents (e.g. Claude Code). This is how Qwen initiates bidirectional
communication through the ACP task queue.
"""

from __future__ import annotations

import logging
from typing import Any

from .tool_registry import ToolRegistry, ToolResult, ToolSpec

logger = logging.getLogger(__name__)


ACP_CREATE_TASK_SPEC = ToolSpec(
    name="acp.create_task",
    description=(
        "Create a task on BotVibes for another agent. "
        "Use target_agent_id='claude-session-agent' to send directly to Claude Code. "
        "Capability examples: 'reasoning.complex', 'code.edit', 'code.refactor'."
    ),
    params_schema={
        "capability": {"type": "string", "required": True},
        "prompt": {"type": "string", "required": True},
        "target_agent_id": {"type": "string", "default": "claude-session-agent"},
        "priority": {"type": "string", "default": "normal"},
        "conversation_id": {"type": "string", "description": "Thread ID for related tasks"},
    },
    requires_approval=False,
)


async def _tool_acp_create_task(params: dict[str, Any]) -> ToolResult:
    """Create a task on BotVibes via the local ACP proxy endpoint."""
    capability = params.get("capability", "")
    prompt = params.get("prompt", "") or params.get("description", "")
    target_agent_id = params.get("target_agent_id", "") or params.get("recipient", "claude-session-agent") or "claude-session-agent"
    # Normalize common recipient names
    if target_agent_id.lower() in ("claude code", "claude", "claude-code"):
        target_agent_id = "claude-session-agent"
    priority = params.get("priority", "normal")

    conversation_id = params.get("conversation_id")

    if not capability or not prompt:
        return ToolResult(
            tool="acp.create_task",
            success=False,
            error="Both 'capability' and 'prompt' are required",
        )

    import httpx

    try:
        payload: dict = {
            "required_capability": capability,
            "payload_ref": prompt,
            "priority": priority,
        }
        if target_agent_id:
            payload["target_agent_id"] = target_agent_id
        if conversation_id:
            payload["conversation_id"] = conversation_id
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "http://127.0.0.1:7337/acp/tasks",
                json=payload,
            )
            resp.raise_for_status()
            result = resp.json()
            task_id = result.get("task_id", "unknown")
            return ToolResult(
                tool="acp.create_task",
                success=True,
                output=f"Task created: {task_id} (capability={capability}, priority={priority})",
            )
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 503:
            return ToolResult(
                tool="acp.create_task",
                success=False,
                error="ACP bridge not connected to BotVibes",
            )
        return ToolResult(tool="acp.create_task", success=False, error=str(e))
    except Exception as e:
        return ToolResult(tool="acp.create_task", success=False, error=str(e))


def register_acp_tools(registry: ToolRegistry) -> None:
    """Register ACP tools in the tool registry."""
    registry.register(ACP_CREATE_TASK_SPEC, _tool_acp_create_task)
