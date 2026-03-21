"""API routes for head management."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class GenerateRequest(BaseModel):
    prompt: str
    temperature: float | None = None
    max_tokens: int | None = None
    images: list[str] | None = None  # base64-encoded images for VLM heads


@router.get("")
async def list_heads(request: Request):
    """List all registered heads and their states."""
    head_manager = request.app.state.head_manager
    return head_manager.get_states()


@router.get("/{head_id}")
async def get_head(head_id: str, request: Request):
    """Get a specific head's state."""
    head_manager = request.app.state.head_manager
    states = head_manager.get_states()
    if head_id not in states:
        raise HTTPException(404, f"Head not found: {head_id}")
    return states[head_id]


@router.post("/{head_id}/wake")
async def wake_head(head_id: str, request: Request):
    """Wake/load a head."""
    head_manager = request.app.state.head_manager
    try:
        await head_manager.ensure_active(head_id)
    except KeyError:
        raise HTTPException(404, f"Head not found: {head_id}")
    except Exception as e:
        logger.exception("Failed to wake head %s", head_id)
        raise HTTPException(500, f"Failed to wake head: {head_id}")
    return {"head_id": head_id, "state": head_manager.get_state(head_id).value}


@router.post("/{head_id}/sleep")
async def sleep_head(head_id: str, request: Request, level: int = 1):
    """Put a head to sleep."""
    head_manager = request.app.state.head_manager
    try:
        await head_manager.sleep_head(head_id, level)
    except KeyError:
        raise HTTPException(404, f"Head not found: {head_id}")
    return {"head_id": head_id, "state": head_manager.get_state(head_id).value}


@router.post("/{head_id}/unload")
async def unload_head(head_id: str, request: Request):
    """Fully unload a head."""
    head_manager = request.app.state.head_manager
    try:
        await head_manager.unload_head(head_id)
    except KeyError:
        raise HTTPException(404, f"Head not found: {head_id}")
    return {"head_id": head_id, "state": head_manager.get_state(head_id).value}


class RouteRequest(BaseModel):
    task_types: list[str]
    privacy: str | None = None  # "public", "internal", "confidential", "restricted"


@router.post("/route")
async def route_by_task(body: RouteRequest, request: Request):
    """Route to the best head for given task types (Phase 1).

    Uses capability-aware routing with privacy constraints.
    """
    from ..router import Router

    head_manager = request.app.state.head_manager
    registry = getattr(request.app.state, "solver_registry", None)
    r = Router(head_manager, registry=registry)

    privacy = None
    if body.privacy:
        from ..models import DataSensitivity, PrivacyConstraint
        try:
            sensitivity = DataSensitivity(body.privacy)
            privacy = PrivacyConstraint(data_sensitivity=sensitivity)
        except ValueError:
            raise HTTPException(400, f"Invalid privacy level: {body.privacy}")

    ranked = r.rank_by_task(body.task_types, privacy=privacy)
    if not ranked:
        raise HTTPException(404, f"No head can handle task types: {body.task_types}")

    return {
        "selected": ranked[0][0],
        "score": ranked[0][1],
        "rankings": [{"head_id": hid, "score": s} for hid, s in ranked],
    }


@router.post("/{head_id}/generate")
async def generate(head_id: str, body: GenerateRequest, request: Request):
    """Generate text through a specific head."""
    head_manager = request.app.state.head_manager
    kwargs = {}
    if body.temperature is not None:
        kwargs["temperature"] = body.temperature
    if body.max_tokens is not None:
        kwargs["max_tokens"] = body.max_tokens
    if body.images is not None:
        kwargs["images"] = body.images
    try:
        await head_manager.ensure_active(head_id)
        result = await head_manager.generate(head_id, body.prompt, **kwargs)
        return result
    except KeyError:
        raise HTTPException(404, f"Head not found: {head_id}")
    except Exception as e:
        logger.exception("Generate failed for head %s", head_id)
        raise HTTPException(500, f"Generation failed: {e}")
