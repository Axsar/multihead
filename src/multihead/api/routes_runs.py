"""API routes for run management."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from pydantic import BaseModel

from ..config import load_recipe
from ..models import WorkOrder

router = APIRouter()


class RunCreateRequest(BaseModel):
    recipe: str | None = None
    goal: str | None = None
    inputs: dict[str, Any] = {}
    work_order: dict[str, Any] | None = None


class RunResponse(BaseModel):
    run_id: str
    status: str
    goal: str = ""
    current_step_index: int = 0
    total_steps: int = 0
    created_at: str = ""
    started_at: str | None = None
    ended_at: str | None = None


@router.post("", response_model=RunResponse)
async def create_run(body: RunCreateRequest, request: Request, background_tasks: BackgroundTasks):
    """Create and start a new run from a recipe or inline work order."""
    orchestrator = request.app.state.orchestrator
    settings = request.app.state.settings

    if body.recipe:
        work_order = load_recipe(settings.config_dir, body.recipe)
        if body.inputs:
            work_order.inputs.update(body.inputs)
    elif body.work_order:
        work_order = WorkOrder.model_validate(body.work_order)
    elif body.goal:
        work_order = WorkOrder(goal=body.goal, inputs=body.inputs)
    else:
        raise HTTPException(400, "Must provide 'recipe', 'goal', or 'work_order'")

    # Validate head IDs exist
    head_manager = request.app.state.head_manager
    available_heads = set(head_manager.get_states().keys())
    for step in work_order.steps:
        if step.head_id not in available_heads:
            raise HTTPException(
                400,
                f"Unknown head '{step.head_id}' in step '{step.name}'. "
                f"Available: {sorted(available_heads)}",
            )

    state = await orchestrator.create_run(work_order)

    # Execute in background
    background_tasks.add_task(orchestrator.execute_run, state.run_id)

    return RunResponse(
        run_id=state.run_id,
        status=state.status.value,
        goal=work_order.goal,
        total_steps=len(work_order.steps),
        created_at=state.created_at.isoformat(),
    )


@router.get("/{run_id}", response_model=RunResponse)
async def get_run(run_id: str, request: Request):
    """Get run status and progress."""
    orchestrator = request.app.state.orchestrator
    state = orchestrator.get_run_state(run_id)
    if state is None:
        raise HTTPException(404, f"Run not found: {run_id}")

    total_steps = len(state.work_order.steps) if state.work_order else 0

    return RunResponse(
        run_id=state.run_id,
        status=state.status.value,
        goal=state.work_order.goal if state.work_order else "",
        current_step_index=state.current_step_index,
        total_steps=total_steps,
        created_at=state.created_at.isoformat(),
        started_at=state.started_at.isoformat() if state.started_at else None,
        ended_at=state.ended_at.isoformat() if state.ended_at else None,
    )


@router.get("/{run_id}/results")
async def get_run_results(run_id: str, request: Request):
    """Get detailed step results including consensus metrics."""
    orchestrator = request.app.state.orchestrator
    state = orchestrator.get_run_state(run_id)
    if state is None:
        raise HTTPException(404, f"Run not found: {run_id}")

    results = []
    for step_id, result in state.step_results.items():
        step_data: dict[str, Any] = {
            "step_id": step_id,
            "head_id": result.head_id,
            "status": result.status.value,
            "outputs": result.outputs,
            "metrics": result.metrics,
            "warnings": result.warnings,
            "error": result.error,
            "started_at": result.started_at.isoformat(),
            "ended_at": result.ended_at.isoformat() if result.ended_at else None,
        }
        # Include consensus info if present
        if "consensus_agreement" in result.metrics:
            step_data["consensus"] = {
                "agreement_score": result.metrics.get("consensus_agreement", 0),
                "red_flags_count": result.metrics.get("consensus_red_flags", 0),
            }
        results.append(step_data)

    return {
        "run_id": run_id,
        "status": state.status.value,
        "total_steps": len(state.work_order.steps) if state.work_order else 0,
        "completed_steps": len(state.step_results),
        "results": results,
    }


@router.get("/{run_id}/events")
async def get_run_events(run_id: str, request: Request, tail: int = 50):
    """Get run events (JSONL)."""
    event_store = request.app.state.event_store
    events = event_store.read_events(run_id)
    if not events:
        raise HTTPException(404, f"No events for run: {run_id}")

    events = events[-tail:]
    return [e.model_dump(mode="json") for e in events]


@router.post("/{run_id}/cancel")
async def cancel_run(run_id: str, request: Request):
    """Cancel a running run."""
    orchestrator = request.app.state.orchestrator
    state = await orchestrator.cancel_run(run_id)
    if state is None:
        raise HTTPException(404, f"Run not found: {run_id}")
    return {"run_id": run_id, "status": state.status.value}


@router.post("/{run_id}/replay")
async def replay_run(run_id: str, request: Request, background_tasks: BackgroundTasks, step_id: str | None = None):
    """Replay a run from a specific step (or from the beginning)."""
    orchestrator = request.app.state.orchestrator
    state = orchestrator.get_run_state(run_id)
    if state is None:
        raise HTTPException(404, f"Run not found: {run_id}")

    if step_id and state.work_order:
        # Find step index
        for i, s in enumerate(state.work_order.steps):
            if s.step_id == step_id:
                state.current_step_index = i
                break

    background_tasks.add_task(orchestrator.execute_run, run_id)
    return {"run_id": run_id, "status": "replaying", "from_step": state.current_step_index}


@router.get("")
async def list_runs(request: Request, limit: int = Query(default=20, le=100)):
    """List all runs (most recent first)."""
    event_store = request.app.state.event_store
    runs = event_store.list_runs()
    return runs[-limit:] if len(runs) > limit else runs
