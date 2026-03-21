"""API routes for task decomposition."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..decomposer import TaskDecomposer

logger = logging.getLogger(__name__)

router = APIRouter()


class DecomposeRequest(BaseModel):
    goal: str
    context: str = ""
    head_id: str | None = None
    max_depth: int = 4


class RefineRequest(BaseModel):
    node_id: str
    node_goal: str
    action_type: str = ""
    target_files: list[str] = []
    exploration_result: str = ""
    head_id: str | None = None


@router.post("")
async def decompose(body: DecomposeRequest, request: Request):
    """Decompose a high-level goal into a hierarchical execution plan."""
    hm = request.app.state.head_manager
    ks = getattr(request.app.state, "knowledge_store", None)
    metrics = getattr(request.app.state, "metrics", None)

    decomposer = TaskDecomposer(hm, knowledge_store=ks, metrics=metrics)

    try:
        plan = await decomposer.decompose(
            goal=body.goal,
            context=body.context,
            head_id=body.head_id,
            max_depth=body.max_depth,
        )
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(422, f"Failed to parse LLM response: {e}")

    return {
        "goal": plan.goal,
        "complexity": plan.complexity,
        "total_steps": plan.total_steps,
        "max_depth": plan.max_depth,
        "context_used": plan.context_used,
        "phases": [p.model_dump() for p in plan.phases],
        "tree": plan.render_tree(),
    }


@router.post("/refine")
async def refine(body: RefineRequest, request: Request):
    """Refine a single step into sub-steps."""
    from ..decomposer import TaskNode

    hm = request.app.state.head_manager
    ks = getattr(request.app.state, "knowledge_store", None)
    metrics = getattr(request.app.state, "metrics", None)

    decomposer = TaskDecomposer(hm, knowledge_store=ks, metrics=metrics)

    node = TaskNode(
        id=body.node_id,
        goal=body.node_goal,
        action_type=body.action_type,
        target_files=body.target_files,
    )

    try:
        children = await decomposer.refine(
            node=node,
            exploration_result=body.exploration_result,
            head_id=body.head_id,
        )
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(422, f"Failed to parse LLM response: {e}")

    return {
        "parent_id": body.node_id,
        "children": [c.model_dump() for c in children],
    }
