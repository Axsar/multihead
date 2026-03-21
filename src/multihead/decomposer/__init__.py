"""Task Decomposer — breaks high-level goals into hierarchical execution plans.

Takes a goal + context -> produces a tree of phases/steps/substeps,
each leaf being a single concrete action that an agent (Claude Code, local LLM)
can execute.
"""

# Re-export everything that was in the original decomposer.py module
# so all existing imports continue to work unchanged.

from .models import (
    DecompositionPlan,
    TaskNode,
    _action_icon,
    _leaf_to_prompt,
    _render_node,
    _trim_node,
)
from .parsing import (
    _STOP_WORDS,
    _extract_keywords,
    _strip_llm_wrapper,
    parse_json_array,
    parse_json_object,
    parse_node,
    parse_plan,
)
from .prompts import DECOMPOSE_PROMPT, REFINE_PROMPT
from .task_decomposer import TaskDecomposer

__all__ = [
    # Models
    "TaskNode",
    "DecompositionPlan",
    # Parsing
    "parse_json_object",
    "parse_json_array",
    "parse_node",
    "parse_plan",
    "_extract_keywords",
    "_strip_llm_wrapper",
    "_STOP_WORDS",
    # Prompts
    "DECOMPOSE_PROMPT",
    "REFINE_PROMPT",
    # Decomposer
    "TaskDecomposer",
    # Tree helpers (used internally but re-exported for completeness)
    "_leaf_to_prompt",
    "_render_node",
    "_action_icon",
    "_trim_node",
]
