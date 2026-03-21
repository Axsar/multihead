"""MCP tool registrations — @mcp.tool() decorated wrappers.

Core tools are defined here; ACP, marketplace, solve, decompose,
harvest, and delegate tools live in _registrations_extended.
"""

from __future__ import annotations

import json

from ._core import _get_ks, mcp
from ._tools_core import (
    _briefing,
    _chat,
    _config,
    _deposit_action_claim,
    _deposit_claim,
    _file_briefing,
    _generate,
    _heads,
    _knowledge,
    _report_event,
    _run_recipe,
    _run_status,
    _swap_head,
)

# Re-export extended tool registrations so importing _registrations
# still triggers all @mcp.tool() decorations.
from ._registrations_extended import (  # noqa: F401
    multihead_check_tasks,
    multihead_claim_task,
    multihead_collect_votes,
    multihead_complete_task,
    multihead_create_task,
    multihead_decompose,
    multihead_delegate_claude,
    multihead_complete_step,
    multihead_finalize_solve,
    multihead_harvest,
    multihead_marketplace_procure,
    multihead_refine_step,
    multihead_solve,
)


# -------------------------------------------------------------------
# Core tool wrappers
# -------------------------------------------------------------------

@mcp.tool()
async def multihead_chat(message: str, session_id: str | None = None) -> str:
    """Send a message to MultiHead's local LLM and get a response.

    The local LLM (e.g. Qwen3-8B on GPU) processes the message using the
    Agentic Core, which can use tools, search the web, and access knowledge.

    Args:
        message: The message to send to the local LLM.
        session_id: Optional session ID to continue a conversation.
    """
    return await _chat(message, session_id)


@mcp.tool()
async def multihead_generate(head_id: str, prompt: str, temperature: float | None = None, max_tokens: int | None = None) -> str:
    """Generate text directly through a specific model head.

    Bypasses the Agentic Core and sends the prompt directly to the specified
    model. Useful for raw inference without tool use or conversation context.

    Args:
        head_id: The head to use (e.g. 'qwen-llm', 'qwen-vlm', 'openai-gpt4o').
        prompt: The prompt text to send to the model.
        temperature: Optional sampling temperature (0.0 to 1.0).
        max_tokens: Optional maximum tokens to generate.
    """
    return await _generate(head_id, prompt, temperature, max_tokens)


@mcp.tool()
async def multihead_heads() -> str:
    """List all available model heads and their current states.

    Shows which models are registered, their adapter type, kind (LLM/VLM),
    and current state (OFF, ACTIVE, SLEEPING, etc.).
    """
    return await _heads()


@mcp.tool()
async def multihead_swap_head(head_id: str, action: str = "wake") -> str:
    """Load, unload, or sleep a model head.

    Controls which model is active on the GPU. Only one GPU-heavy head
    can be active at a time (GPU mutex).

    Args:
        head_id: The head to control (e.g. 'qwen-llm', 'qwen-vlm').
        action: One of 'wake' (load/activate), 'sleep' (low-power), or 'unload' (free GPU).
    """
    return await _swap_head(head_id, action)


@mcp.tool()
async def multihead_run_recipe(recipe: str, inputs: dict | None = None) -> str:
    """Execute a pipeline recipe on MultiHead.

    Recipes are YAML-defined multi-step pipelines that chain model calls,
    tool use, and data transformations.

    Args:
        recipe: Recipe name (without .yaml extension).
        inputs: Optional input data for the pipeline.
    """
    return await _run_recipe(recipe, inputs)


@mcp.tool()
async def multihead_run_status(run_id: str) -> str:
    """Check the status of a pipeline run.

    Args:
        run_id: The run ID returned by multihead_run_recipe.
    """
    return await _run_status(run_id)


@mcp.tool()
async def multihead_knowledge(query_type: str = "claims", status: str | None = None, limit: int = 10) -> str:
    """Query MultiHead's knowledge store for claims or events.

    The knowledge store contains facts learned from conversations and
    Night Shift processing.

    Args:
        query_type: Either 'claims' or 'events'.
        status: Optional filter by status (e.g. 'accepted', 'pending', 'confirmed').
        limit: Maximum number of results (default 10, max 50).
    """
    limit = min(limit, 50)
    return await _knowledge(query_type, status, limit)


@mcp.tool()
async def multihead_deposit_claim(
    claim_key: str, statement: str, produced_by: str = "claude_code",
    scope_id: str = "default", claim_type: str = "fact", confidence: float = 0.9,
) -> str:
    """Deposit a claim into MultiHead's knowledge store.

    Use this to record facts, decisions, or observations that should persist
    across sessions. Claims are the building blocks of institutional memory.

    IMPORTANT — For action items (work orders, votes, consensus requests,
    progress updates, results, blockers), use multihead_deposit_action instead.
    It enforces the canonical key pattern action.{scope}.{type}.{id} and sets
    deadlines automatically. Using this tool for action items will trigger a
    warning.

    Args:
        claim_key: Dot-separated key (e.g. 'project.component.detail').
        statement: Human-readable statement of the claim.
        produced_by: Who/what produced this claim (e.g. 'claude-multihead-main', 'claude_code').
        scope_id: Project scope (e.g. 'multihead', 'default', 'vibebots').
        claim_type: Type: fact, decision, constraint, preference, plan, etc.
        confidence: Confidence score 0.0-1.0 (default 0.9).
    """
    # Detect action-like claims that should use multihead_deposit_action
    _action_hints = ["vote", "consensus", "work_order", "solve.request", "solve.proposal"]
    key_lower = claim_key.lower()
    warning = ""
    if any(hint in key_lower for hint in _action_hints) and not claim_key.startswith("action."):
        warning = (
            "\n\nWARNING: This claim looks like an action item (contains "
            f"'{next(h for h in _action_hints if h in key_lower)}') but doesn't use the "
            "canonical action.{scope}.{type}.{id} key pattern. "
            "Consider using multihead_deposit_action instead for proper inbox visibility "
            "and deadline tracking."
        )

    # Auto-infer scope if caller left it as default
    if scope_id == "default":
        from multihead.scope_inference import infer_scope
        scope_id = infer_scope(claim_key, statement)

    result = await _deposit_claim(claim_key, statement, produced_by, scope_id, claim_type, confidence)
    if warning:
        # Append warning to the JSON result
        try:
            data = json.loads(result)
            data["convention_warning"] = warning.strip()
            return json.dumps(data, indent=2)
        except (json.JSONDecodeError, TypeError):
            return result + warning
    return result


@mcp.tool()
async def multihead_deposit_action(
    scope_id: str,
    action_type: str,
    short_id: str,
    statement: str,
    produced_by: str = "claude-multihead-main",
    deadline_hours: int = 48,
) -> str:
    """Deposit an action claim with enforced key convention and deadline.

    Creates claim with key action.{scope_id}.{action_type}.{short_id} and
    sets valid_to = now + deadline_hours. Use for work orders, consensus
    requests, votes, progress updates, and results.

    Args:
        scope_id: Project scope (e.g. 'multihead', 'default').
        action_type: One of: work_order, consensus, vote, progress, result, blocker, proposal, assignment.
        short_id: Short identifier (e.g. 'inbox-action-layer', 'scena-dsl').
        statement: Human-readable description of the action item.
        produced_by: Who produced this (default: claude-multihead-main).
        deadline_hours: Hours until expiry (default 48).
    """
    return await _deposit_action_claim(
        scope_id, action_type, short_id, statement, produced_by, deadline_hours,
    )


@mcp.tool()
async def multihead_report_event(
    title: str, summary: str = "", event_type: str = "note",
    produced_by: str = "claude_code",
) -> str:
    """Report a knowledge event to MultiHead.

    Events track what happened — pipeline runs, task completions, decisions.
    They form the timeline that Night Shift audits.

    Args:
        title: Short title of what happened.
        summary: Longer description.
        event_type: Type: note, task_completed, decision, commit, milestone, etc.
        produced_by: Who/what produced this event.
    """
    return await _report_event(title, summary, event_type, produced_by)


@mcp.tool()
async def multihead_briefing(component: str, scope_id: str = "default") -> str:
    """Get a briefing for a component — what it needs to know before running.

    Returns direct claims (key matches component), related claims
    (statement mentions component), and recent events. Components call
    this at startup to learn context from the knowledge store.

    Args:
        component: Component name (e.g. 'auth', 'payments', 'deploy').
        scope_id: Project scope (default 'default').
    """
    return await _briefing(component, scope_id)


@mcp.tool()
async def multihead_file_briefing(file_path: str) -> str:
    """Get knowledge briefing for a file BEFORE editing it.

    Returns what the knowledge base knows about this file:
    - CONSTRAINTS: corroborated facts (independently verified — don't violate)
    - WARNINGS: stale claims (things that changed — verify before assuming)
    - SIGNALS: contested claims (channels disagree — be careful)
    - HISTORY: superseded claims (failed approaches — don't repeat)
    - UNVERIFIED: claimed fixes not confirmed by code

    Call this before editing any file to avoid repeating known mistakes
    and to respect verified invariants.

    Args:
        file_path: Path to the file you're about to edit (absolute or relative).
    """
    return await _file_briefing(file_path)


@mcp.tool()
async def multihead_check_inbox(
    agent_id: str = "claude-multihead-main",
    scope_id: str | None = None,
    max_age_hours: int = 48,
    limit: int = 10,
) -> str:
    """Check knowledge.db inbox for messages directed at this agent.

    Returns unhandled claims (questions, requests, plans, action items) from
    other agents that this agent hasn't yet read or responded to. Claims are
    marked as 'read' after retrieval so they won't appear again.

    Args:
        agent_id: Identity to check inbox for (default: claude-multihead-main).
        scope_id: Project scope to filter. None = all scopes (default).
        max_age_hours: Ignore claims older than this (default 48).
        limit: Maximum results (default 10).
    """
    try:
        ks = _get_ks()
        items: list[dict] = []
        shown_ids: list[str] = []
        self_ids = {"claude-multihead-main", "claude_code_main", agent_id}

        # Action prefixes surface work orders, consensus, and votes
        # Covers both canonical (action.*) and legacy key patterns
        action_prefixes = [
            "action.",              # canonical: action.{scope}.{type}.{id}
            "solve.consensus.",     # legacy solve consensus requests
            "solve.request.",       # legacy solve decomposition requests
            "solve.vote.",          # legacy solve votes
            "solve.proposal.",      # legacy solve proposals
            "vote.",                # legacy bare vote prefix
            "consensus.",           # legacy bare consensus prefix
        ]

        for claim_type_list in [["question", "request"], ["plan"]]:
            if hasattr(ks, "get_unhandled_claims"):
                claims = ks.get_unhandled_claims(
                    agent_id=agent_id,
                    claim_types=claim_type_list,
                    scope_id=scope_id,
                    max_age_hours=max_age_hours,
                    limit=limit,
                    key_prefixes=action_prefixes,
                )
            elif hasattr(ks, "get_pending_messages"):
                claims = ks.get_pending_messages(
                    session_id=agent_id,
                    scope_id=scope_id,
                    max_age_hours=max_age_hours,
                    limit=limit,
                )
            else:
                return json.dumps({"inbox": [], "note": "Knowledge store has no inbox method"})

            for claim in claims:
                sender = claim.provenance.produced_by.get("id", "unknown")
                if sender in self_ids:
                    shown_ids.append(claim.claim_id)
                    continue
                # Check expiry via valid_to
                expired = False
                valid_to_str = ""
                if hasattr(claim, "valid_to") and claim.valid_to:
                    valid_to_str = claim.valid_to.isoformat() if hasattr(claim.valid_to, "isoformat") else str(claim.valid_to)
                    try:
                        from datetime import datetime as _dt
                        vt = _dt.fromisoformat(valid_to_str) if isinstance(claim.valid_to, str) else claim.valid_to
                        expired = vt < datetime.now(timezone.utc) if vt.tzinfo else False
                    except Exception:
                        pass
                items.append({
                    "claim_id": claim.claim_id,
                    "from": sender,
                    "scope": claim.canonical.scope_id if hasattr(claim.canonical, "scope_id") else "",
                    "type": claim.claim_type.value if hasattr(claim.claim_type, "value") else str(claim.claim_type),
                    "key": claim.canonical.claim_key,
                    "statement": claim.statement[:300],
                    "deadline": valid_to_str,
                    "expired": expired,
                    "created_at": claim.provenance.created_at.isoformat() if claim.provenance.created_at else "",
                })
                shown_ids.append(claim.claim_id)

        # Mark all shown claims as 'read' so they don't reappear
        if shown_ids and hasattr(ks, "record_interaction"):
            for cid in shown_ids:
                try:
                    ks.record_interaction(cid, agent_id, "read")
                except Exception:
                    pass  # Duplicate interaction is fine

        return json.dumps({"inbox": items, "count": len(items)}, indent=2)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
async def multihead_config(action: str = "show", key: str | None = None, value: str | None = None) -> str:
    """View or modify MultiHead's runtime configuration.

    Controls tool enable/disable, generation defaults, web access, etc.

    Args:
        action: 'show' to view config, 'set' to change a value.
        key: Config key for 'set' action (e.g. 'generation.temperature', 'web_tools_enabled').
        value: New value for 'set' action.
    """
    return await _config(action, key, value)
