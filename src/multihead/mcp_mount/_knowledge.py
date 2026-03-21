"""MCP tools for knowledge, runs/orchestration, solve/decompose, packs, and consensus."""

from __future__ import annotations

import json
from typing import Any

from ._core import mcp, _ctx


# -------------------------------------------------------------------
# Knowledge
# -------------------------------------------------------------------

@mcp.tool()
async def query_knowledge(query_type: str = "claims", status: str | None = None, limit: int = 25) -> str:
    """Query the knowledge store for claims or events.

    Args:
        query_type: 'claims' or 'events'.
        status: Optional filter by status (e.g. 'accepted', 'pending').
        limit: Maximum results to return (max 25).
    """
    limit = min(limit, 25)
    ks = _ctx["knowledge_store"]
    if query_type == "events":
        items = await ks.list_events(limit=limit)
        result = [
            {
                "event_id": getattr(i, "event_id", None),
                "event_type": str(getattr(i, "event_type", "")),
                "title": (getattr(i, "title", "") or "")[:200],
                "severity": getattr(i, "severity", None),
                "status": str(getattr(i, "event_status", "")),
                "timestamp": str(getattr(i, "provenance", {}).created_at) if hasattr(getattr(i, "provenance", None), "created_at") else None,
            }
            for i in items
        ]
    else:
        kwargs: dict[str, Any] = {"limit": limit}
        if status:
            kwargs["claim_status"] = status
        items = await ks.list_claims(**kwargs)
        result = [
            {
                "claim_id": getattr(i, "claim_id", None),
                "claim_key": getattr(i.canonical, "claim_key", None) if hasattr(i, "canonical") else None,
                "statement": (getattr(i, "statement", "") or "")[:200],
                "confidence": getattr(i, "confidence", None),
                "status": str(getattr(i, "claim_status", "")),
                "timestamp": str(getattr(i, "provenance", {}).created_at) if hasattr(getattr(i, "provenance", None), "created_at") else None,
            }
            for i in items
        ]
    return json.dumps(result, default=str)


@mcp.tool()
async def create_claim(claim_key: str, statement: str, claim_type: str = "fact", confidence: float = 0.8) -> str:
    """Create a new knowledge claim.

    Args:
        claim_key: Short key for the claim (e.g. 'python.version').
        statement: The claim text.
        claim_type: Type: 'fact', 'preference', 'capability'.
        confidence: Confidence score 0.0-1.0.
    """
    ks = _ctx["knowledge_store"]
    claim_id = await ks.upsert_claim(
        claim_key=claim_key,
        statement=statement,
        claim_type=claim_type,
        confidence=confidence,
    )
    return json.dumps({"claim_id": claim_id, "claim_key": claim_key})


@mcp.tool()
async def create_event(title: str, summary: str, event_type: str = "observation", tags: list[str] | None = None) -> str:
    """Record a knowledge event.

    Args:
        title: Short event title.
        summary: Detailed description.
        event_type: Type: 'observation', 'decision', 'discovery', 'milestone'.
        tags: Optional tags for categorization.
    """
    ks = _ctx["knowledge_store"]
    event_id = await ks.append_event(
        title=title,
        summary=summary,
        event_type=event_type,
        tags=tags or [],
    )
    return json.dumps({"event_id": event_id, "title": title})


@mcp.tool()
async def briefing(component: str, scope_id: str | None = None) -> str:
    """Get a knowledge briefing for a component -- relevant claims, events, context.

    Args:
        component: Component name (e.g. 'multihead', 'cortex').
        scope_id: Optional scope filter.
    """
    ks = _ctx["knowledge_store"]
    result = await ks.briefing(component=component, scope_id=scope_id)
    return json.dumps(result, default=str)


# -------------------------------------------------------------------
# Runs & Orchestration
# -------------------------------------------------------------------

@mcp.tool()
async def list_runs(limit: int = 20) -> str:
    """List recent orchestration runs.

    Args:
        limit: Maximum runs to return (default 20).
    """
    runs = _ctx["event_store"].list_runs()
    runs = runs[:limit]
    # Keep only essential fields
    slim = [
        {
            "run_id": r.get("run_id"),
            "status": r.get("status"),
            "goal": r.get("goal"),
            "created_at": r.get("created_at"),
            "updated_at": r.get("updated_at"),
        }
        for r in runs
    ]
    return json.dumps(slim, default=str)


@mcp.tool()
async def run_recipe(recipe: str) -> str:
    """Start a new orchestration run from a recipe name.

    Args:
        recipe: Recipe name (e.g. 'summarize', 'analyze').
    """
    from ..config import load_recipe
    work_order = load_recipe(_ctx["settings"].config_dir, recipe)
    run = await _ctx["orchestrator"].start_run(work_order)
    return json.dumps({"run_id": run.run_id, "status": run.status}, default=str)


@mcp.tool()
async def run_status(run_id: str) -> str:
    """Get status and progress of a specific run.

    Args:
        run_id: The run ID to check.
    """
    run = _ctx["event_store"].get_run(run_id)
    return json.dumps(run, default=str)


# -------------------------------------------------------------------
# Solve & Decompose
# -------------------------------------------------------------------

@mcp.tool()
async def solve(task: str, strategy: str = "auto", max_steps: int = 10) -> str:
    """Run the autonomous solve pipeline: decompose -> route -> execute.

    Args:
        task: The task description to solve.
        strategy: Strategy: 'auto', 'single', 'parallel'.
        max_steps: Maximum execution steps.
    """
    from ..solve_engine import SolveEngine
    engine = SolveEngine(
        head_manager=_ctx["head_manager"],
        event_store=_ctx["event_store"],
        artifact_store=_ctx["artifact_store"],
        knowledge_store=_ctx["knowledge_store"],
        runs_dir=_ctx["settings"].runs_dir,
    )
    result = await engine.solve(task, strategy=strategy, max_steps=max_steps)
    return json.dumps(result, default=str)


@mcp.tool()
async def decompose(goal: str, context: str = "") -> str:
    """Decompose a goal into an execution plan.

    Args:
        goal: The high-level goal to decompose.
        context: Optional additional context.
    """
    from ..decompose import Decomposer
    decomposer = Decomposer(_ctx["head_manager"])
    result = await decomposer.decompose(goal, context=context)
    return json.dumps(result, default=str)


# -------------------------------------------------------------------
# Context Packs
# -------------------------------------------------------------------

@mcp.tool()
async def list_packs() -> str:
    """List all built context packs."""
    pb = _ctx["pack_builder"]
    packs = pb.list_packs()
    return json.dumps(packs, default=str)


@mcp.tool()
async def build_pack(purpose: str, max_tokens: int = 4000) -> str:
    """Build a new context pack for a specific purpose.

    Args:
        purpose: What the pack is for (e.g. 'onboarding', 'debugging').
        max_tokens: Token budget for the pack.
    """
    pb = _ctx["pack_builder"]
    pack = await pb.build(purpose=purpose, budgets={"max_tokens": max_tokens, "max_items": 50})
    return json.dumps(pack, default=str)


# -------------------------------------------------------------------
# Consensus
# -------------------------------------------------------------------

@mcp.tool()
async def consensus(query: str, head_ids: list[str], strategy: str = "majority") -> str:
    """Execute a consensus query across multiple model heads.

    Args:
        query: The question to ask all heads.
        head_ids: List of head IDs to query.
        strategy: Consensus strategy: 'majority', 'weighted', 'debate'.
    """
    from ..consensus import ConsensusEngine
    engine = ConsensusEngine(_ctx["head_manager"])
    result = await engine.execute(query, head_ids, strategy)
    # Slim down response -- strip full vote text
    if hasattr(result, "model_dump"):
        rd = result.model_dump()
    else:
        rd = result if isinstance(result, dict) else {"raw": str(result)}
    final_answer = str(rd.get("consensus_outputs", ""))[:1000]
    votes = rd.get("all_votes", [])
    head_summary = [
        {
            "head_id": v.get("head_id"),
            "agreed": v.get("success", False) and v.get("schema_valid", True),
            "latency_ms": v.get("latency_ms"),
        }
        for v in votes
    ]
    slim = {
        "agreement_score": rd.get("agreement_score"),
        "final_answer": final_answer,
        "heads": head_summary,
    }
    return json.dumps(slim, default=str)
