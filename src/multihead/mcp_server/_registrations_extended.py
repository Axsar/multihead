"""MCP tool registrations — ACP, marketplace, solve, decompose, harvest, delegate."""

from __future__ import annotations

import json

import httpx

from ._core import _request, mcp
from ._tools_acp import (
    _check_tasks,
    _claim_task,
    _complete_task,
    _create_task,
    _delegate_claude,
)
from ._tools_solve import _solve, _solve_self, _solve_consensus, _collect_and_vote, _report_solve_step, _finalize_solve
from ._tools_decompose import _decompose, _refine_step
from ._tools_harvest import _harvest


# -------------------------------------------------------------------
# ACP task tools (bidirectional via BotVibes)
# -------------------------------------------------------------------

@mcp.tool()
async def multihead_check_tasks(capability: str = "com.claude.code") -> str:
    """Check BotVibes for tasks waiting in Claude Code's inbox.

    Polls the ACP server for available tasks matching the given capability.
    Call this to see if MultiHead's local LLM has delegated work to you.

    Args:
        capability: Capability to filter by (default: com.claude.code).
    """
    return await _check_tasks(capability)


@mcp.tool()
async def multihead_claim_task(task_id: str) -> str:
    """Claim an ACP task from BotVibes to work on.

    Atomically reserves and dispatches the task. Once claimed, complete it
    with multihead_complete_task when done.

    Args:
        task_id: The task UUID from multihead_check_tasks.
    """
    return await _claim_task(task_id)


@mcp.tool()
async def multihead_complete_task(
    task_id: str,
    output_ref: str,
    status: str = "complete",
    confidence: float | None = None,
    error_message: str | None = None,
) -> str:
    """Submit results for a completed ACP task.

    Args:
        task_id: The task UUID being completed.
        output_ref: Result description or artifact reference.
        status: 'complete' or 'failed'.
        confidence: Optional confidence score (0.0-1.0).
        error_message: Error description if status is 'failed'.
    """
    return await _complete_task(task_id, output_ref, status, confidence, error_message)


@mcp.tool()
async def multihead_create_task(capability: str, payload_ref: str, priority: str = "normal", target_agent_id: str | None = None, conversation_id: str | None = None) -> str:
    """Create a task on BotVibes for another agent to handle.

    Use this to delegate work to MultiHead's local LLM or other ACP agents.
    Use target_agent_id for direct targeting (skips capability discovery).
    Use conversation_id to thread related tasks in a conversation.

    Args:
        capability: Required capability (e.g. 'llm.generate', 'reasoning.complex').
        payload_ref: Task description or input data as a string.
        priority: 'high', 'normal', or 'batch'.
        target_agent_id: Direct target agent (e.g. 'multihead-agent' or 'claude-session-agent').
        conversation_id: Optional conversation thread ID for related tasks.
    """
    return await _create_task(capability, payload_ref, priority, target_agent_id, conversation_id)


@mcp.tool()
async def multihead_marketplace_procure(
    capability: str,
    payload: str,
    max_price: float | None = None,
    quote_timeout: float = 30.0,
) -> str:
    """Procure work from BotVibes cloud marketplace.

    Submits an RFQ (Request for Quote), waits for provider quotes,
    selects the best one, and accepts it — returning a contract.

    Args:
        capability: Required capability (e.g. 'text_generation', 'visual_reasoning').
        payload: Task description or requirements.
        max_price: Maximum acceptable price per call (optional).
        quote_timeout: Seconds to wait for quotes (default 30).
    """
    try:
        body: dict = {"capability": capability, "payload": payload, "quote_timeout": quote_timeout}
        if max_price is not None:
            body["max_price"] = max_price
        result = await _request("POST", "/acp/marketplace/procure", json=body)
        return json.dumps(result, indent=2)
    except httpx.ConnectError:
        return "Error: MultiHead server not running. Start it with: multihead serve"
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 503:
            return "Cloud marketplace not configured. Set ACP_CLOUD_URL and ACP_CLOUD_API_KEY in .env."
        if e.response.status_code == 422:
            return f"Marketplace error: {e.response.text}"
        return f"Error: {e}"
    except Exception as e:
        return f"Error: {e}"


# -------------------------------------------------------------------
# Solve pipeline
# -------------------------------------------------------------------

@mcp.tool()
async def multihead_solve(
    task: str,
    strategy: str = "first_to_ahead",
    max_steps: int = 20,
    enable_marketplace: bool = False,
    dry_run: bool = False,
    mode: str = "self",
    min_proposals: int = 3,
    timeout_hours: float = 24.0,
    scope_id: str = "default",
) -> str:
    """Run the autonomous solve pipeline: decompose, route, execute.

    Takes a high-level task and autonomously decomposes it into steps,
    routes each step to the best available head, executes with DAG
    parallelism, and aggregates results.

    Args:
        task: The task to solve (e.g. "Build a hello world app").
        strategy: Consensus strategy for decomposition (first_to_ahead, majority, weighted).
        max_steps: Maximum steps in the execution plan.
        enable_marketplace: Allow marketplace delegation for unroutable steps.
        dry_run: Stop after decomposition and return the plan without executing.
        mode: "self" (default) — caller is the executor, returns plan + context.
              "consensus" — post request to knowledge.db, collect proposals from other sessions, vote.
              "server" — proxy to multihead serve for GPU head execution.
        min_proposals: Minimum proposals before voting (consensus mode, default 3).
        timeout_hours: Hours to wait for proposals (consensus mode, default 24).
        scope_id: Project scope for consensus request (default "default").
    """
    if mode == "self":
        return await _solve_self(task, strategy, max_steps, dry_run=dry_run)
    if mode == "consensus":
        return await _solve_consensus(task, strategy, min_proposals, 10, timeout_hours, scope_id)
    return await _solve(task, strategy, max_steps, enable_marketplace, dry_run=dry_run)


@mcp.tool()
async def multihead_complete_step(
    run_id: str,
    step_id: str,
    output: str = "",
    status: str = "completed",
) -> str:
    """Report completion of a self-solve step.

    Call this after executing each step from a self-solve plan.
    Stores the output as an artifact and records the event.

    Args:
        run_id: The run_id from multihead_solve (self mode).
        step_id: The step ID (e.g. "1.1", "2.1").
        output: Summary of what was done.
        status: "completed" or "failed".
    """
    return await _report_solve_step(run_id, step_id, status, output)


@mcp.tool()
async def multihead_finalize_solve(run_id: str) -> str:
    """Finalize a self-solve run after all steps are done.

    Marks the run as complete and records final metrics.

    Args:
        run_id: The run_id from multihead_solve (self mode).
    """
    return await _finalize_solve(run_id)


@mcp.tool()
async def multihead_collect_votes(
    request_id: str,
    force_vote: bool = False,
) -> str:
    """Check proposal status and run consensus vote for a consensus solve.

    Call this after multihead_solve(mode="consensus") to check how many
    proposals have been submitted. When min_proposals is met, automatically
    runs the consensus vote and returns the winner.

    Args:
        request_id: The request_id from multihead_solve (consensus mode).
        force_vote: Vote with whatever proposals are available, even if below minimum.
    """
    return await _collect_and_vote(request_id, force_vote)


# -------------------------------------------------------------------
# Decomposition tools
# -------------------------------------------------------------------

@mcp.tool()
async def multihead_decompose(
    goal: str,
    context: str = "",
    head_id: str | None = None,
    max_depth: int = 4,
) -> str:
    """Decompose a complex task into a hierarchical execution plan.

    Takes a high-level goal and uses an LLM (grounded by knowledge store
    claims) to produce a tree of phases, steps, and substeps. Each leaf
    step is a single concrete action.

    Three strategies (tried in order if earlier ones fail):
      1. "self" (default) — returns the decomposition template + knowledge
         context so YOU (the calling session) can decompose with your own
         loaded codebase context. Best quality since you can read actual code.
      2. "claude-p" — spawns a claude -p subprocess to decompose.
      3. "qwen" — routes through local Qwen LLM via MultiHead API.

    Args:
        goal: The high-level task to decompose (e.g. "Fix auth token refresh logic").
        context: Optional additional context about the task.
        head_id: Specific model head to use. If not set, uses the active LLM head.
        max_depth: Maximum tree depth (default 4).
    """
    return await _decompose(goal, context, head_id, max_depth, strategy="self")


@mcp.tool()
async def multihead_refine_step(
    node_id: str,
    node_goal: str,
    action_type: str = "",
    target_files: list[str] | None = None,
    exploration_result: str = "",
    head_id: str | None = None,
) -> str:
    """Refine a decomposition step into smaller sub-steps.

    Use after multihead_decompose to drill into a complex step.
    Provide exploration_result with what you learned to get better sub-steps.

    Args:
        node_id: The step ID to refine (e.g. "2.1").
        node_goal: What the step does.
        action_type: Current action type (explore, read, edit, etc.).
        target_files: Files relevant to this step.
        exploration_result: What you learned when exploring this step.
        head_id: Specific model head to use.
    """
    return await _refine_step(node_id, node_goal, action_type, target_files, exploration_result, head_id)


# -------------------------------------------------------------------
# Session Harvester
# -------------------------------------------------------------------

@mcp.tool()
async def multihead_harvest(action: str = "status") -> str:
    """Trigger or check session harvester status.

    Scans all Claude Code project folders (~/.claude/projects/),
    reads MEMORY.md and CLAUDE.md files, and deposits extracted
    claims into knowledge.db for cross-context awareness.

    Args:
        action: 'status' (default), 'run' (trigger harvest), or 'list' (show projects).
    """
    return await _harvest(action)


# -------------------------------------------------------------------
# Delegate to Claude Worker Daemon
# -------------------------------------------------------------------

@mcp.tool()
async def multihead_delegate_claude(prompt: str, conversation_id: str | None = None, priority: str = "normal") -> str:
    """Delegate a task to the Claude Worker Daemon for autonomous execution.

    Creates an ACP task targeting claude-session-agent. The worker daemon
    picks it up and spawns a headless Claude Code subprocess to handle it.
    Use conversation_id for multi-turn exchanges (the daemon tracks sessions
    and uses --resume for continuity).

    Args:
        prompt: Task description for Claude Code to execute.
        conversation_id: Optional thread ID for multi-turn conversations.
        priority: 'high', 'normal', or 'batch'.
    """
    return await _delegate_claude(prompt, conversation_id, priority)
