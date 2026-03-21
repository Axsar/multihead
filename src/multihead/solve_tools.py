"""Solve pipeline tool for the Agentic Core.

Lets the LLM invoke the solve pipeline from chat when it detects
a complex multi-step task that needs decomposition and routing.
"""

from __future__ import annotations

import logging
from typing import Any

from .tool_registry import ToolRegistry, ToolResult, ToolSpec

logger = logging.getLogger(__name__)


SOLVE_SPEC = ToolSpec(
    name="solve.run",
    description=(
        "Run the autonomous solve pipeline for complex multi-step tasks. "
        "Decomposes the task into steps, routes each to the best head, "
        "and executes them (potentially in parallel). Use this when the user "
        "asks you to build, create, or solve something that requires multiple steps."
    ),
    params_schema={
        "task": {"type": "string", "required": True},
        "max_steps": {"type": "integer", "default": 20},
        "timeout": {"type": "integer", "default": 240},
        "dry_run": {
            "type": "boolean",
            "default": False,
            "description": "If true, plan only — don't execute",
        },
    },
    requires_approval=False,
)


def register_solve_tools(
    registry: ToolRegistry,
    head_manager: Any,
    event_store: Any,
    artifact_store: Any,
    knowledge_store: Any,
    runs_dir: Any,
) -> None:
    """Register solve pipeline tools."""

    async def _tool_solve_run(params: dict[str, Any]) -> ToolResult:
        from .solve_pipeline import SolveConstraints, SolvePipeline

        task = params.get("task", "")
        if not task:
            return ToolResult(tool="solve.run", success=False, error="No task provided")

        constraints = SolveConstraints(
            max_steps=params.get("max_steps", 20),
            timeout_seconds=params.get("timeout", 240),
        )
        dry_run = params.get("dry_run", False)

        pipeline = SolvePipeline(
            head_manager=head_manager,
            event_store=event_store,
            artifact_store=artifact_store,
            knowledge_store=knowledge_store,
            runs_dir=runs_dir,
        )

        try:
            result = await pipeline.solve(task, constraints=constraints, dry_run=dry_run)
            summary = (
                f"**Solve complete** ({result.status})\n"
                f"- Steps: {result.steps_succeeded}/{result.steps_total} succeeded\n"
                f"- Duration: {result.duration_seconds:.1f}s\n"
                f"- Confidence: {result.confidence:.0%}\n"
            )
            if dry_run:
                summary = f"**Dry run** — {result.plan_steps} steps planned\n" + summary
            summary += f"\n---\n\n{result.output}"
            return ToolResult(tool="solve.run", success=True, output=summary)
        except Exception as e:
            logger.error("Solve tool error: %s", e, exc_info=True)
            return ToolResult(tool="solve.run", success=False, error=str(e))

    registry.register(SOLVE_SPEC, _tool_solve_run)
