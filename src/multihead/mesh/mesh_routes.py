"""API routes for mesh protocol."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


async def verify_mesh_token(request: Request) -> None:
    """Validate bearer token on protected mesh routes.

    Reads the MeshSecurity instance from app.state.mesh_security.
    If no security is configured (mesh_security is None), all requests pass.
    """
    security = getattr(request.app.state, "mesh_security", None)
    if security is None:
        return  # No auth configured

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header")

    token = auth_header[7:]  # Strip "Bearer "
    if not security.auth.validate(token):
        raise HTTPException(403, "Invalid mesh token")


class TaskSubmitRequest(BaseModel):
    """Request to submit a task to a mesh node."""
    task_id: str = ""
    capability_kind: str  # llm | vlm | embed
    model: str = ""
    prompt: str = ""
    params: dict[str, Any] = Field(default_factory=dict)


@router.get("/capabilities", dependencies=[Depends(verify_mesh_token)])
async def list_capabilities(request: Request, kind: str | None = None) -> list[dict[str, Any]]:
    """List capabilities of this node."""
    registry = request.app.state.capability_registry
    caps = registry.list_capabilities(kind)
    return [c.model_dump(mode="json") for c in caps]


@router.post("/tasks", dependencies=[Depends(verify_mesh_token)])
async def submit_task(body: TaskSubmitRequest, request: Request) -> dict[str, Any]:
    """Submit a task to this node."""
    registry = request.app.state.capability_registry
    available = registry.find_available(body.capability_kind)

    if not available:
        raise HTTPException(503, f"No available {body.capability_kind} capabilities")

    cap = available[0]
    head_manager = request.app.state.head_manager

    # Find matching head
    matching_head = None
    for head_id, manifest in head_manager._manifests.items():
        if manifest.kind == body.capability_kind:
            if not body.model or manifest.model == body.model:
                matching_head = head_id
                break

    if not matching_head:
        raise HTTPException(404, "No matching head found")

    try:
        registry.update_status(cap.capability_id, "busy")
        await head_manager.ensure_active(matching_head)
        adapter = head_manager.get_adapter(matching_head)
        response = await adapter.generate(body.prompt)
        registry.update_status(cap.capability_id, "available")

        return {
            "task_id": body.task_id,
            "status": "completed",
            "result": response,
            "head_id": matching_head,
        }
    except Exception as e:
        registry.update_status(cap.capability_id, "available")
        raise HTTPException(500, f"Task execution failed: {e}")


@router.get("/node", dependencies=[Depends(verify_mesh_token)])
async def node_info(request: Request) -> dict[str, Any]:
    """Get this node's information."""
    registry = request.app.state.capability_registry
    caps = registry.list_capabilities()
    return {
        "node_id": getattr(request.app.state, "node_id", "local"),
        "version": "1.0.0",
        "capabilities": len(caps),
        "available": len(registry.find_available()),
    }


@router.get("/claims", dependencies=[Depends(verify_mesh_token)])
async def list_claims(
    request: Request,
    since: str = "",
    scope_id: str = "",
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return shared claims for mesh replication.

    Only returns claims with visibility='shared'.
    Supports delta sync via `since` (ISO timestamp).
    """
    from datetime import datetime, timezone

    store = getattr(request.app.state, "knowledge_store", None)
    if store is None:
        return []

    since_dt = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since)
        except ValueError:
            raise HTTPException(400, f"Invalid 'since' timestamp: {since}")

    claims = store.get_shared_claims_since(
        since=since_dt,
        scope_id=scope_id or None,
        limit=min(limit, 500),
    )
    return [c.model_dump(mode="json") for c in claims]


class ClaimImportRequest(BaseModel):
    """Request to import claims from a peer."""
    claims: list[dict[str, Any]]


@router.post("/claims/import", dependencies=[Depends(verify_mesh_token)])
async def import_claims(body: ClaimImportRequest, request: Request) -> dict[str, Any]:
    """Receive and import claims from a peer node.

    Verifies signatures, skips duplicates, and records conflicts.
    """
    store = getattr(request.app.state, "knowledge_store", None)
    if store is None:
        raise HTTPException(503, "Knowledge store not available")

    imported = 0
    skipped = 0
    errors: list[str] = []

    from ..knowledge_models import Claim, ClaimStatus

    for claim_dict in body.claims:
        claim_id = claim_dict.get("claim_id", "")
        try:
            # Skip duplicates
            if store.get_claim(claim_id):
                skipped += 1
                continue

            # Only import shared claims
            scope_data = claim_dict.get("scope", {})
            if scope_data.get("visibility", "private") != "shared":
                skipped += 1
                continue

            claim = Claim.model_validate(claim_dict)

            # Verify signature
            sig_result = store.verify_claim(claim)
            if sig_result is False:
                errors.append(f"Tampered signature on {claim_id}")
                continue

            # Import as proposed
            claim.claim_status = ClaimStatus.PROPOSED
            store.insert_claim(claim)
            imported += 1

        except Exception as e:
            errors.append(f"{claim_id}: {e}")

    return {
        "imported": imported,
        "skipped": skipped,
        "errors": errors,
    }


@router.get("/health")
async def mesh_health(request: Request) -> dict[str, Any]:
    """Mesh-specific health check."""
    return {"status": "ok", "mesh_version": "1.0.0"}
