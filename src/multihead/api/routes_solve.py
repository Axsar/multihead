"""API routes for the autonomous solve pipeline."""

from __future__ import annotations

import logging
import uuid
from collections import OrderedDict
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

_MAX_TRACKED = 200


class SolveRequest(BaseModel):
    task: str
    strategy: str = "first_to_ahead"
    max_steps: int = 20
    max_depth: int = 3
    timeout: float = 240.0
    enable_marketplace: bool = False
    enable_tests: bool = False
    dry_run: bool = False
    # Phase 1-2: capability routing + privacy
    task_types: list[str] | None = None
    privacy: str | None = None  # "public", "internal", "confidential", "restricted"


class SolveResponse(BaseModel):
    run_id: str
    status: str
    output: str
    confidence: float
    steps_total: int = 0
    steps_succeeded: int = 0
    steps_failed: int = 0
    duration_seconds: float = 0.0
    plan_steps: int = 0
    parallel_steps: int = 0
    dry_run: bool = False


class SolveStartResponse(BaseModel):
    """Slim response returned when background=true."""

    run_id: str
    status: str
    plan_steps: int = 0
    parallel_steps: int = 0


class SolveStatusResponse(BaseModel):
    """Status response for polling a background solve."""

    run_id: str
    status: str  # "running", "done", "failed"
    output: str = ""
    confidence: float = 0.0
    steps_total: int = 0
    steps_succeeded: int = 0
    steps_failed: int = 0
    duration_seconds: float = 0.0
    plan_steps: int = 0
    parallel_steps: int = 0
    error: str | None = None


# ---------------------------------------------------------------------------
# Background solve tracking (capped LRU)
# ---------------------------------------------------------------------------

_solve_runs: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _track(run_id: str, data: dict[str, Any]) -> None:
    """Upsert a solve run entry, evicting oldest if over cap."""
    _solve_runs[run_id] = data
    _solve_runs.move_to_end(run_id)
    while len(_solve_runs) > _MAX_TRACKED:
        _solve_runs.popitem(last=False)


async def _run_solve_background(
    head_manager: Any,
    event_store: Any,
    artifact_store: Any,
    knowledge_store: Any,
    runs_dir: Any,
    task: str,
    constraints: Any,
    dry_run: bool,
    run_id: str,
) -> None:
    """Background task: run solve pipeline and track progress."""
    from ..solve_pipeline import SolvePipeline

    _track(run_id, {"status": "running", "run_id": run_id})

    pipeline = SolvePipeline(
        head_manager=head_manager,
        event_store=event_store,
        artifact_store=artifact_store,
        knowledge_store=knowledge_store,
        runs_dir=runs_dir,
    )

    result = await pipeline.solve(task, constraints=constraints, dry_run=dry_run)

    # pipeline.solve() never raises — it returns status="failed" on error
    _track(run_id, {
        "status": result.status,
        "run_id": run_id,
        "real_run_id": result.run_id,
        "output": result.output,
        "confidence": result.confidence,
        "steps_total": result.steps_total,
        "steps_succeeded": result.steps_succeeded,
        "steps_failed": result.steps_failed,
        "duration_seconds": result.duration_seconds,
        "plan_steps": result.plan_steps,
        "parallel_steps": result.parallel_steps,
        "dry_run": result.dry_run,
    })


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("", response_model=SolveResponse)
async def solve(
    body: SolveRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    background: bool = Query(False),
):
    """Run the autonomous solve pipeline: decompose -> route -> execute.

    Pass ?background=true to return immediately with a run_id.
    Poll GET /solve/{run_id} for progress.
    """
    from ..solve_pipeline import SolveConstraints

    constraints = SolveConstraints(
        max_steps=body.max_steps,
        max_depth=body.max_depth,
        timeout_seconds=body.timeout,
        strategy=body.strategy,
        enable_test_generation=body.enable_tests,
        enable_marketplace_delegation=body.enable_marketplace,
    )

    if background:
        run_id = f"solve-{uuid.uuid4().hex[:12]}"
        # Pass raw state objects so the background task builds its own pipeline
        background_tasks.add_task(
            _run_solve_background,
            head_manager=request.app.state.head_manager,
            event_store=request.app.state.event_store,
            artifact_store=request.app.state.artifact_store,
            knowledge_store=getattr(request.app.state, "knowledge_store", None),
            runs_dir=request.app.state.settings.runs_dir,
            task=body.task,
            constraints=constraints,
            dry_run=body.dry_run,
            run_id=run_id,
        )
        return SolveStartResponse(run_id=run_id, status="running")

    # Synchronous (blocking) path — existing behaviour
    from ..solve_pipeline import SolvePipeline

    hm = request.app.state.head_manager
    es = request.app.state.event_store
    art = request.app.state.artifact_store
    ks = getattr(request.app.state, "knowledge_store", None)
    settings = request.app.state.settings

    pipeline = SolvePipeline(
        head_manager=hm,
        event_store=es,
        artifact_store=art,
        knowledge_store=ks,
        runs_dir=settings.runs_dir,
    )

    try:
        result = await pipeline.solve(body.task, constraints=constraints, dry_run=body.dry_run)
    except Exception as e:
        logger.error("Solve pipeline error: %s", e, exc_info=True)
        raise HTTPException(500, f"Solve pipeline error: {e}")

    return SolveResponse(
        run_id=result.run_id,
        status=result.status,
        output=result.output,
        confidence=result.confidence,
        steps_total=result.steps_total,
        steps_succeeded=result.steps_succeeded,
        steps_failed=result.steps_failed,
        duration_seconds=result.duration_seconds,
        plan_steps=result.plan_steps,
        parallel_steps=result.parallel_steps,
        dry_run=result.dry_run,
    )


@router.get("/{run_id}")
async def get_solve_status(run_id: str):
    """Poll background solve progress."""
    entry = _solve_runs.get(run_id)
    if not entry:
        raise HTTPException(404, f"Solve run not found: {run_id}")
    return SolveStatusResponse(
        run_id=run_id,
        status=entry["status"],
        output=entry.get("output", ""),
        confidence=entry.get("confidence", 0.0),
        steps_total=entry.get("steps_total", 0),
        steps_succeeded=entry.get("steps_succeeded", 0),
        steps_failed=entry.get("steps_failed", 0),
        duration_seconds=entry.get("duration_seconds", 0.0),
        plan_steps=entry.get("plan_steps", 0),
        parallel_steps=entry.get("parallel_steps", 0),
        error=entry.get("error"),
    )
