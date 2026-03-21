"""Structured action types for the Agentic Core."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field


class SayAction(BaseModel):
    """Text response, no side effects."""
    action: Literal["SAY"] = "SAY"
    content: str


class CallToolAction(BaseModel):
    """Single tool invocation."""
    action: Literal["CALL_TOOL"] = "CALL_TOOL"
    tool: str
    params: dict[str, Any] = Field(default_factory=dict)


class CreateWorkOrderAction(BaseModel):
    """Spawn a pipeline work order."""
    action: Literal["CREATE_WORKORDER"] = "CREATE_WORKORDER"
    goal: str
    steps: list[dict[str, Any]] = Field(default_factory=list)
    inputs: dict[str, Any] = Field(default_factory=dict)


class MonitorWorkOrderAction(BaseModel):
    """Check and optionally intervene on a running work order."""
    action: Literal["MONITOR_WORKORDER"] = "MONITOR_WORKORDER"
    run_id: str
    decision: str = "check"  # check | cancel | retry


class PauseAndAskAction(BaseModel):
    """Need user input before continuing."""
    action: Literal["PAUSE_AND_ASK"] = "PAUSE_AND_ASK"
    question: str
    options: list[str] = Field(default_factory=list)


CoreAction = SayAction | CallToolAction | CreateWorkOrderAction | MonitorWorkOrderAction | PauseAndAskAction

_ACTION_MAP: dict[str, type[BaseModel]] = {
    "SAY": SayAction,
    "CALL_TOOL": CallToolAction,
    "CREATE_WORKORDER": CreateWorkOrderAction,
    "MONITOR_WORKORDER": MonitorWorkOrderAction,
    "PAUSE_AND_ASK": PauseAndAskAction,
}


def _extract_json(text: str) -> dict[str, Any] | None:
    """Try to extract a JSON object from text."""
    # Try direct parse
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Try markdown code blocks
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try embedded JSON object
    m = re.search(r"\{[^{}]*\"action\"[^{}]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


def parse_action(raw: str) -> CoreAction:
    """Parse an action from raw LLM output.

    Falls back to SayAction if parsing fails.
    """
    data = _extract_json(raw)

    if data and "action" in data:
        action_type = data["action"].upper()
        cls = _ACTION_MAP.get(action_type)
        if cls:
            try:
                return cls.model_validate(data)
            except Exception as e:
                logger.debug("Action validation failed for %s: %s", action_type, e)

        # Fuzzy match: if unknown action but has "tool" key, treat as CALL_TOOL
        if not cls and "tool" in data:
            try:
                return CallToolAction.model_validate({
                    "action": "CALL_TOOL",
                    "tool": data["tool"],
                    "params": data.get("params", {}),
                })
            except Exception as e:
                logger.debug("Fuzzy CALL_TOOL fallback failed: %s", e)

    # Fallback: treat entire text as a SAY action
    return SayAction(content=raw)
