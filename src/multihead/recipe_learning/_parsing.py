"""YAML and JSON parsing utilities for recipe learning."""

from __future__ import annotations

import json
import logging
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def extract_yaml(text: str) -> str | None:
    """Extract YAML from text (handles markdown code blocks).

    Args:
        text: Text that may contain YAML

    Returns:
        Extracted YAML string, or None if not found
    """
    # Check for markdown code blocks
    if "```yaml" in text or "```yml" in text:
        # Extract from code block
        start_markers = ["```yaml", "```yml"]
        for marker in start_markers:
            if marker in text:
                start = text.index(marker) + len(marker)
                end = text.index("```", start)
                return text[start:end].strip()

    # Check if entire text is YAML
    if text.strip().startswith("goal:") or text.strip().startswith("steps:"):
        return text.strip()

    # Try to find YAML-like content
    lines = text.split("\n")
    yaml_lines = []
    in_yaml = False

    for line in lines:
        if line.strip().startswith("goal:") or line.strip().startswith("steps:"):
            in_yaml = True

        if in_yaml:
            yaml_lines.append(line)

    if yaml_lines:
        return "\n".join(yaml_lines)

    return None


def parse_recipe_from_response(response: str) -> dict[str, Any] | None:
    """Parse recipe YAML from expert response.

    Args:
        response: Expert response (may contain explanation + YAML)

    Returns:
        Parsed recipe dict, or None if parsing failed
    """
    try:
        # Extract YAML from response (may have markdown code blocks)
        yaml_text = extract_yaml(response)

        if not yaml_text:
            logger.warning("No YAML found in expert response")
            return None

        # Parse YAML
        recipe = yaml.safe_load(yaml_text)

        # Validate basic structure
        if not isinstance(recipe, dict):
            logger.error("Recipe is not a dict")
            return None

        if "goal" not in recipe:
            logger.warning("Recipe missing 'goal' field")
            recipe["goal"] = "Expert-designed recipe"

        if "steps" not in recipe or not recipe["steps"]:
            logger.error("Recipe missing 'steps' field")
            return None

        return recipe

    except yaml.YAMLError as e:
        logger.error("Failed to parse recipe YAML: %s", e)
        return None


def try_parse_vote_json(text: str) -> dict[str, Any]:
    """Parse JSON from a vote output."""
    # Direct JSON
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # Markdown code block
    if "```" in (text or ""):
        try:
            start = text.index("```") + 3
            if text[start:].startswith("json"):
                start += 4
            end = text.index("```", start)
            return json.loads(text[start:end].strip())
        except (ValueError, json.JSONDecodeError):
            pass

    # Search for embedded JSON
    for ch in ["{", "["]:
        idx = (text or "").find(ch)
        if idx >= 0:
            try:
                return json.loads(text[idx:])
            except json.JSONDecodeError:
                pass

    return {"action": "reject", "confidence": 0.3, "rationale": ""}
