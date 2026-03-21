"""API routes for context packs."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class PackBuildRequest(BaseModel):
    purpose: str = "Custom Pack"
    filters: dict[str, Any] = {}
    budgets: dict[str, int] = {"max_tokens": 4000, "max_items": 50}


@router.get("")
async def list_packs(request: Request) -> list[dict[str, Any]]:
    """List all built context packs."""
    pb = request.app.state.pack_builder
    return pb.list_packs()


@router.post("/build")
async def build_pack(body: PackBuildRequest, request: Request) -> dict[str, Any]:
    """Build a new context pack."""
    pb = request.app.state.pack_builder
    pack = pb.build_pack(
        purpose=body.purpose,
        filters=body.filters,
        budgets=body.budgets,
    )
    return {
        "pack_id": pack.pack_id,
        "purpose": pack.purpose,
        "item_count": len(pack.items),
        "metrics": pack.metrics,
        "created_at": pack.created_at.isoformat(),
    }


@router.post("/build-standard")
async def build_standard_packs(request: Request) -> list[dict[str, Any]]:
    """Build all 5 standard Night Shift packs."""
    pb = request.app.state.pack_builder
    packs = pb.build_standard_packs()
    return [
        {
            "pack_id": p.pack_id,
            "purpose": p.purpose,
            "item_count": len(p.items),
            "created_at": p.created_at.isoformat(),
        }
        for p in packs
    ]


@router.get("/{pack_id}")
async def get_pack(pack_id: str, request: Request) -> dict[str, Any]:
    """Get a specific pack's contents."""
    pb = request.app.state.pack_builder
    pack = pb.load_pack(pack_id)
    if pack is None:
        raise HTTPException(404, f"Pack not found: {pack_id}")
    return pack.model_dump(mode="json")
