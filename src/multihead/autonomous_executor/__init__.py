"""Autonomous Executor — executes decomposed plans via Claude Code sessions.

Alternative execution backend for the auto-responder poller. Instead of
just posting plans to knowledge.db (plan-only mode), this spawns `claude -p`
subprocesses per step, with role-specific tool restrictions, context chaining,
and quality-gated reflection loops.

Strategies:
- LocalLLMStrategy: plan-only (current default — posts plan to knowledge.db)
- ClaudeSessionStrategy: spawns claude -p per step with role-based tools
"""

from .dag import (
    _flatten_leaves,
    _infer_layer_dependencies,
    _summarize_plan,
    _topological_layers,
)
from .executor import AutonomousExecutor
from .models import ExecutionReport, StepContext, StepExecutionResult
from .strategies import (
    DEFAULT_ROLE_PROMPT,
    DEFAULT_TOOLS,
    ClaudeSessionStrategy,
    ExecutionStrategy,
    LocalLLMStrategy,
    ROLE_PROMPTS,
    ROLE_TOOL_MAP,
    _normalize_claude_output,
)

__all__ = [
    # Models
    "StepExecutionResult",
    "ExecutionReport",
    "StepContext",
    # Strategies
    "ExecutionStrategy",
    "LocalLLMStrategy",
    "ClaudeSessionStrategy",
    # Constants
    "ROLE_TOOL_MAP",
    "DEFAULT_TOOLS",
    "ROLE_PROMPTS",
    "DEFAULT_ROLE_PROMPT",
    # DAG helpers
    "_flatten_leaves",
    "_infer_layer_dependencies",
    "_topological_layers",
    "_summarize_plan",
    # Internal
    "_normalize_claude_output",
    # Main class
    "AutonomousExecutor",
]
