"""MCP-over-HTTP: core setup, health, heads, chat, config, and nightshift tools.

This module owns the FastMCP instance and _ctx dict. Other sub-modules import
``mcp`` and ``_ctx`` from here to register their tools on the same instance.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastmcp import FastMCP

logger = logging.getLogger(__name__)

# The FastMCP instance -- tools are registered at import time,
# but they access core objects via the _ctx dict which is wired later.
mcp = FastMCP(
    "multihead",
    instructions=(
        "MultiHead: local multimodal task-runner with hot-swappable specialist models. "
        "MCP-native endpoint -- all operations available as tools."
    ),
)

# Core object references, populated by wire_mcp()
_ctx: dict[str, Any] = {}


def wire_mcp(app_state: Any) -> None:
    """Wire live app.state references into the MCP tools."""
    _ctx["head_manager"] = app_state.head_manager
    _ctx["orchestrator"] = app_state.orchestrator
    _ctx["event_store"] = app_state.event_store
    _ctx["artifact_store"] = app_state.artifact_store
    _ctx["knowledge_store"] = app_state.knowledge_store
    _ctx["record_store"] = app_state.record_store
    _ctx["pack_builder"] = app_state.pack_builder
    _ctx["night_shift"] = app_state.night_shift
    _ctx["session_manager"] = app_state.session_manager
    _ctx["agentic_core"] = app_state.agentic_core
    _ctx["tool_registry"] = app_state.tool_registry
    _ctx["acp_bridge"] = app_state.acp_bridge
    _ctx["skill_registry"] = app_state.skill_registry
    _ctx["settings"] = app_state.settings
    _ctx["runtime_config"] = app_state.runtime_config
    _ctx["skill_catalog"] = app_state.skill_catalog
    _ctx["metrics"] = app_state.metrics
    _ctx["resource_monitor"] = app_state.resource_monitor
    logger.info("MCP tools wired to live core objects")


# -------------------------------------------------------------------
# Health & Status
# -------------------------------------------------------------------

@mcp.tool()
async def health() -> str:
    """Check MultiHead health: heads loaded, API status, BotVibes connection."""
    hm = _ctx["head_manager"]
    bridge = _ctx.get("acp_bridge")
    skills = _ctx.get("skill_registry")
    states = hm.get_states()
    active = sum(1 for s in states.values() if s.get("status") == "active")
    return json.dumps({
        "status": "ok",
        "heads_total": len(states),
        "heads_active": active,
        "heads": list(states.keys()),
        "botvibes_connected": bridge.connected if bridge else False,
        "skills_loaded": len(skills) if skills else 0,
    })


# -------------------------------------------------------------------
# Heads
# -------------------------------------------------------------------

@mcp.tool()
async def list_heads() -> str:
    """List all registered model heads and their current states."""
    states = _ctx["head_manager"].get_states()
    return json.dumps(states, default=str)


@mcp.tool()
async def wake_head(head_id: str) -> str:
    """Wake/load a model head into memory.

    Args:
        head_id: The head to wake (e.g. 'qwen-llm', 'claude-sonnet').
    """
    result = await _ctx["head_manager"].wake(head_id)
    return json.dumps({"head_id": head_id, "status": "waking", "result": str(result)})


@mcp.tool()
async def sleep_head(head_id: str) -> str:
    """Unload a model head from GPU memory.

    Args:
        head_id: The head to sleep.
    """
    await _ctx["head_manager"].sleep(head_id)
    return json.dumps({"head_id": head_id, "status": "sleeping"})


@mcp.tool()
async def generate(head_id: str, prompt: str, temperature: float | None = None, max_tokens: int | None = None) -> str:
    """Generate text directly through a specific model head.

    Args:
        head_id: The head to use (e.g. 'qwen-llm', 'openai-gpt4o').
        prompt: The prompt text.
        temperature: Optional sampling temperature.
        max_tokens: Optional max tokens to generate.
    """
    adapter = _ctx["head_manager"].get_adapter(head_id)
    kwargs: dict[str, Any] = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    result = await adapter.generate(prompt, **kwargs)
    return result.get("text", str(result))


# -------------------------------------------------------------------
# Chat
# -------------------------------------------------------------------

@mcp.tool()
async def chat(message: str, session_id: str | None = None) -> str:
    """Send a message to MultiHead's Agentic Core (local LLM with tools).

    Args:
        message: The message to send.
        session_id: Optional session ID to continue a conversation.
    """
    core = _ctx["agentic_core"]
    if session_id is None:
        session_id = _ctx["session_manager"].create_session()
    response = await core.chat(session_id, message)
    return json.dumps({"session_id": session_id, "response": response})


@mcp.tool()
async def list_sessions() -> str:
    """List all chat sessions."""
    sessions = _ctx["session_manager"].list_sessions()
    return json.dumps(sessions, default=str)


# -------------------------------------------------------------------
# NightShift
# -------------------------------------------------------------------

@mcp.tool()
async def nightshift(action: str = "status") -> str:
    """Control the Night Shift knowledge pipeline.

    Args:
        action: 'status' to check, 'trigger' to start a run.
    """
    ns = _ctx["night_shift"]
    if action == "trigger":
        asyncio.create_task(ns.run())
        return json.dumps({"status": "triggered"})
    return json.dumps(ns.get_status(), default=str)


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

@mcp.tool()
async def config(action: str = "show", key: str | None = None, value: str | None = None) -> str:
    """Get or set runtime configuration.

    Args:
        action: 'show' to view config, 'set' to change a value.
        key: Config key (for 'set' action).
        value: New value (for 'set' action).
    """
    rc = _ctx["runtime_config"]
    if action == "set" and key and value is not None:
        rc.set(key, value)
        from ..runtime_config import RuntimeConfig
        rc.save(_ctx["settings"].data_dir / "runtime_config.json")
        return json.dumps({"status": "updated", "key": key, "value": value})
    return json.dumps(rc.to_dict(), default=str)
