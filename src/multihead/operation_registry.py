"""Registry of all maintenance LLM operations in MultiHead.

Catalogs every place the system invokes an LLM for internal/maintenance work
(as opposed to user-facing chat). Each entry records the operation name,
description, source file, line range, invocation type, tier, category, and
the default model used.

This registry is static metadata — it does not execute anything. The
routes_operations module exposes it via REST for the Cortex Operations UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class OperationDef:
    """Definition of a single maintenance LLM operation."""

    name: str
    description: str
    file: str
    lines: str  # e.g. "54-56" or "407"
    invocation: Literal["cli_subprocess", "sdk_direct", "head_routed", "rest_api"]
    tier: Literal["direct", "routed"]
    default_model: str
    category: str


# ---------------------------------------------------------------------------
# Static catalog — hand-curated from codebase grep of heads.generate(),
# head_manager.generate(), adapter.generate(), and claude -p subprocess calls.
# ---------------------------------------------------------------------------

_OPERATIONS: list[OperationDef] = [
    # -- Night Shift pipeline --
    OperationDef(
        name="nightshift.open_loops",
        description="Extract unresolved questions, TODOs, and pending decisions from ingested text chunks",
        file="night_shift/stages_late.py",
        lines="42-60",
        invocation="head_routed",
        tier="routed",
        default_model="qwen-llm",
        category="night_shift",
    ),
    OperationDef(
        name="nightshift.daily_brief",
        description="Generate a daily brief summarizing events and claims created during Night Shift",
        file="night_shift/stages_late.py",
        lines="69-88",
        invocation="head_routed",
        tier="routed",
        default_model="qwen-llm",
        category="night_shift",
    ),
    OperationDef(
        name="nightshift.weekly_rollup",
        description="Write a weekly rollup summarizing trends, progress, and blockers",
        file="night_shift/stages_late.py",
        lines="90-117",
        invocation="head_routed",
        tier="routed",
        default_model="qwen-llm",
        category="night_shift",
    ),
    OperationDef(
        name="nightshift.entity_linking",
        description="Link extracted entities to existing knowledge graph entries",
        file="night_shift/stages_late.py",
        lines="119-145",
        invocation="head_routed",
        tier="routed",
        default_model="qwen-llm",
        category="night_shift",
    ),
    OperationDef(
        name="nightshift.ingestion_llm",
        description="LLM-assisted extraction during Night Shift text ingestion pipeline",
        file="night_shift/pipeline.py",
        lines="497-510",
        invocation="head_routed",
        tier="routed",
        default_model="qwen-llm",
        category="night_shift",
    ),

    # -- Cloud Marketplace fulfillment --
    OperationDef(
        name="marketplace.code_review",
        description="LLM-powered code review for marketplace contract fulfillment (code.review.v1)",
        file="cloud_marketplace/_fulfillment.py",
        lines="210-224",
        invocation="head_routed",
        tier="routed",
        default_model="llm",
        category="marketplace",
    ),
    OperationDef(
        name="marketplace.vision_describe",
        description="VLM image description for marketplace contracts (image.describe.v1)",
        file="cloud_marketplace/_fulfillment_vision.py",
        lines="86-95",
        invocation="head_routed",
        tier="routed",
        default_model="vlm",
        category="marketplace",
    ),
    OperationDef(
        name="marketplace.vision_detect",
        description="VLM object detection for marketplace contracts (image.detect.objects.v1)",
        file="cloud_marketplace/_fulfillment_vision.py",
        lines="155-160",
        invocation="head_routed",
        tier="routed",
        default_model="vlm",
        category="marketplace",
    ),

    # -- Consensus engine --
    OperationDef(
        name="consensus.head_query",
        description="Send a query to a head as part of multi-head consensus (fan-out phase)",
        file="consensus/engine.py",
        lines="136-139",
        invocation="head_routed",
        tier="routed",
        default_model="(per-head)",
        category="consensus",
    ),

    # -- Orchestrator step execution --
    OperationDef(
        name="orchestrator.step_execute",
        description="Execute a single DAG step via head generate (standard single-head path)",
        file="orchestrator/_execution.py",
        lines="248-250",
        invocation="head_routed",
        tier="routed",
        default_model="(per-step)",
        category="orchestrator",
    ),

    # -- Task decomposer --
    OperationDef(
        name="decomposer.decompose",
        description="Decompose a complex goal into a tree of sub-tasks using LLM",
        file="decomposer/task_decomposer.py",
        lines="94-117",
        invocation="head_routed",
        tier="routed",
        default_model="qwen-llm",
        category="orchestrator",
    ),
    OperationDef(
        name="decomposer.refine",
        description="Refine a decomposition node by expanding it into child sub-tasks",
        file="decomposer/task_decomposer.py",
        lines="151-154",
        invocation="head_routed",
        tier="routed",
        default_model="qwen-llm",
        category="orchestrator",
    ),

    # -- Test generation --
    OperationDef(
        name="testgen.generate",
        description="Generate unit tests for a given code file using LLM",
        file="test_generation.py",
        lines="254-262",
        invocation="head_routed",
        tier="routed",
        default_model="qwen-llm",
        category="testing",
    ),
    OperationDef(
        name="testgen.refine",
        description="Refine generated tests based on execution failures",
        file="test_generation.py",
        lines="302-310",
        invocation="head_routed",
        tier="routed",
        default_model="qwen-llm",
        category="testing",
    ),
    OperationDef(
        name="testgen.doctest",
        description="Generate doctests for Python functions using LLM",
        file="test_generation.py",
        lines="334-340",
        invocation="head_routed",
        tier="routed",
        default_model="qwen-llm",
        category="testing",
    ),

    # -- Benchmarking --
    OperationDef(
        name="benchmark.evaluate",
        description="Run benchmark evaluation prompts through a head for quality scoring",
        file="benchmarking/integration.py",
        lines="129-135",
        invocation="head_routed",
        tier="routed",
        default_model="(per-benchmark)",
        category="benchmarking",
    ),

    # -- Agentic core --
    OperationDef(
        name="agentic.tool_call",
        description="Execute an LLM tool-calling loop in the agentic core",
        file="agentic_core/_core.py",
        lines="405-407",
        invocation="head_routed",
        tier="routed",
        default_model="(core-head)",
        category="agentic",
    ),

    # -- MCP server decompose --
    OperationDef(
        name="mcp.decompose_claude",
        description="Decompose a task via claude -p subprocess (MCP tool fallback path)",
        file="mcp_server/_tools_decompose.py",
        lines="137-180",
        invocation="cli_subprocess",
        tier="direct",
        default_model="claude-sonnet-4-6",
        category="mcp",
    ),

    # -- Claude adapter (CLI subprocess) --
    OperationDef(
        name="adapter.claude_cli",
        description="Generate text via claude -p CLI subprocess (ClaudeAdapter)",
        file="adapters/claude_adapter.py",
        lines="50-81",
        invocation="cli_subprocess",
        tier="direct",
        default_model="claude-sonnet-4-6",
        category="adapter",
    ),

    # -- Shell brain --
    OperationDef(
        name="shell.brain",
        description="Interactive shell brain: processes user messages via Claude SDK adapter",
        file="shell/brain.py",
        lines="78-85",
        invocation="sdk_direct",
        tier="direct",
        default_model="claude-sdk",
        category="shell",
    ),

    # -- Shell pipeline execution --
    OperationDef(
        name="shell.pipeline_exec",
        description="Shell pipeline message execution via adapter.generate with optional image routing",
        file="shell_pipeline/execution.py",
        lines="53-62",
        invocation="sdk_direct",
        tier="direct",
        default_model="(auto-routed)",
        category="shell",
    ),

    # -- MCP mount core --
    OperationDef(
        name="mcp.generate",
        description="LLM generate via MCP-over-HTTP mounted tool (streamable-http transport)",
        file="mcp_mount/_core.py",
        lines="124-126",
        invocation="rest_api",
        tier="routed",
        default_model="(per-head)",
        category="mcp",
    ),

    # -- Mesh routes --
    OperationDef(
        name="mesh.task_execute",
        description="Execute a mesh task submission through adapter.generate (v1/tasks endpoint)",
        file="mesh/mesh_routes.py",
        lines="79-81",
        invocation="rest_api",
        tier="routed",
        default_model="(per-capability)",
        category="mesh",
    ),

    # -- Heads API route --
    OperationDef(
        name="api.heads_generate",
        description="Direct head generate via REST API (POST /heads/{id}/generate)",
        file="api/routes_heads.py",
        lines="123-125",
        invocation="rest_api",
        tier="routed",
        default_model="(per-head)",
        category="api",
    ),

    # -- SDK MCP tools --
    OperationDef(
        name="sdk.mcp_generate",
        description="Generate via SDK MCP tool registration (sdk_mcp_tools.py)",
        file="sdk_mcp_tools.py",
        lines="355-357",
        invocation="sdk_direct",
        tier="routed",
        default_model="(per-head)",
        category="sdk",
    ),

    # -- Extractors (base) --
    OperationDef(
        name="extractor.base_extract",
        description="Run extraction on text chunks using an LLM adapter (base extractor)",
        file="extractors/base.py",
        lines="35-42",
        invocation="head_routed",
        tier="routed",
        default_model="(per-extractor)",
        category="extraction",
    ),

    # -- Autonomous executor --
    OperationDef(
        name="autonomous.claude_session",
        description="Spawn claude -p subprocess per execution step with role-specific tools",
        file="autonomous_executor/strategies.py",
        lines="161-203",
        invocation="cli_subprocess",
        tier="direct",
        default_model="claude-sonnet-4-6",
        category="autonomous",
    ),
]


def get_all_operations() -> list[OperationDef]:
    """Return the full list of registered operations."""
    return list(_OPERATIONS)


def get_operation(name: str) -> OperationDef | None:
    """Lookup a single operation by name."""
    for op in _OPERATIONS:
        if op.name == name:
            return op
    return None
