"""Agentic Core: always-on LLM brain with structured action dispatch."""

from ._constants import (
    _APPROVAL_WORDS,
    _MESH_TUTORIAL,
    _TOOL_CAPABLE_ADAPTERS,
    MAX_TOOL_LOOP_DEPTH,
    MAX_VALIDATION_RETRIES,
    SIMPLE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)
from ._core import AgenticCore

__all__ = [
    "AgenticCore",
    "SYSTEM_PROMPT",
    "SIMPLE_SYSTEM_PROMPT",
    "_TOOL_CAPABLE_ADAPTERS",
    "MAX_VALIDATION_RETRIES",
    "MAX_TOOL_LOOP_DEPTH",
    "_APPROVAL_WORDS",
    "_MESH_TUTORIAL",
]
