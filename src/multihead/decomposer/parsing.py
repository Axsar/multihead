"""JSON parsing helpers for LLM decomposition output."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .models import DecompositionPlan, TaskNode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "is", "and",
    "or", "with", "from", "by", "it", "this", "that", "be", "are", "was",
    "do", "does", "did", "not", "no", "we", "i", "my", "our", "us",
})


def _extract_keywords(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 2]


# ---------------------------------------------------------------------------
# LLM output stripping
# ---------------------------------------------------------------------------

def _strip_llm_wrapper(raw: str) -> str:
    """Strip markdown fences, thinking tags, and find the JSON payload."""
    text = raw.strip()

    # Strip thinking tags
    if "<think>" in text:
        idx = text.rfind("</think>")
        if idx >= 0:
            text = text[idx + 8:].strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        text = "\n".join(lines)
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()

    return text


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def parse_json_object(raw: str) -> dict[str, Any]:
    """Extract and parse a JSON object from LLM output.

    Tries multiple strategies to handle LLM output variations:
    1. Parse stripped text directly
    2. Find first complete JSON object using bracket matching
    3. Extract from first { to last } (legacy fallback)
    """
    text = _strip_llm_wrapper(raw)

    # Strategy 1: Try parsing directly
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Find first complete JSON object using bracket matching
    start = text.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escape_next = False

        for i in range(start, len(text)):
            char = text[i]

            # Handle string escaping
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue

            # Track whether we're inside a string
            if char == '"':
                in_string = not in_string
                continue

            # Only count braces outside of strings
            if not in_string:
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    # Found matching closing brace
                    if depth == 0:
                        try:
                            return json.loads(text[start:i+1])
                        except json.JSONDecodeError as e:
                            logger.warning(f"Bracket-matched JSON failed to parse: {e}")
                            break

    # Strategy 3: Legacy fallback - first { to last }
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError(f"No JSON object found in: {text[:200]}")

    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError as e:
        # If all strategies fail, provide helpful error
        raise ValueError(
            f"Failed to parse JSON from LLM output. Error: {e}\n"
            f"Attempted to parse: {text[start:min(end, start+500)]}..."
        )


def parse_json_array(raw: str) -> list[dict[str, Any]]:
    """Extract and parse a JSON array from LLM output.

    Tries multiple strategies to handle LLM output variations:
    1. Parse stripped text directly
    2. Find first complete JSON array using bracket matching
    3. Extract from first [ to last ] (legacy fallback)
    """
    text = _strip_llm_wrapper(raw)

    # Strategy 1: Try parsing directly
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Find first complete JSON array using bracket matching
    start = text.find("[")
    if start >= 0:
        depth = 0
        in_string = False
        escape_next = False

        for i in range(start, len(text)):
            char = text[i]

            # Handle string escaping
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue

            # Track whether we're inside a string
            if char == '"':
                in_string = not in_string
                continue

            # Only count brackets outside of strings
            if not in_string:
                if char == "[":
                    depth += 1
                elif char == "]":
                    depth -= 1
                    # Found matching closing bracket
                    if depth == 0:
                        try:
                            return json.loads(text[start:i+1])
                        except json.JSONDecodeError as e:
                            logger.warning(f"Bracket-matched JSON failed to parse: {e}")
                            break

    # Strategy 3: Legacy fallback - first [ to last ]
    start = text.find("[")
    end = text.rfind("]") + 1
    if start < 0 or end <= start:
        raise ValueError(f"No JSON array found in: {text[:200]}")

    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError as e:
        # If all strategies fail, provide helpful error
        raise ValueError(
            f"Failed to parse JSON array from LLM output. Error: {e}\n"
            f"Attempted to parse: {text[start:min(end, start+500)]}..."
        )


# ---------------------------------------------------------------------------
# Node / plan parsing
# ---------------------------------------------------------------------------

def parse_node(data: dict[str, Any]) -> TaskNode:
    """Recursively parse a node dict into TaskNode."""
    children_data = (
        data.get("children")
        or data.get("steps")
        or data.get("substeps")
        or []
    )
    children = [parse_node(c) for c in children_data]

    return TaskNode(
        id=str(data.get("id", "")),
        goal=data.get("goal", data.get("name", "")),
        rationale=data.get("rationale", ""),
        action_type=data.get("action_type", ""),
        target_files=data.get("target_files") or [],
        expected_output=data.get("expected_output", ""),
        children=children,
    )


def parse_plan(raw: str, goal: str, context_keys: list[str] | None = None) -> DecompositionPlan:
    """Parse LLM JSON output into a DecompositionPlan."""
    data = parse_json_object(raw)
    phases = [parse_node(p) for p in data.get("phases", [])]
    return DecompositionPlan(
        goal=goal,
        complexity=data.get("complexity", "moderate"),
        phases=phases,
        context_used=context_keys or [],
    )
