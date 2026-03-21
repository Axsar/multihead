"""API routes for the operations registry.

Exposes the catalog of maintenance LLM operations so Cortex can display
and configure model overrides for each operation.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..operation_registry import get_all_operations, get_operation

router = APIRouter()


def _op_to_dict(op, runtime_config) -> dict:
    """Serialize an OperationDef to the shape Cortex expects."""
    # Check for a model override in runtime_config.operations
    operations_cfg = getattr(runtime_config, "operations", None)
    overrides = getattr(operations_cfg, "model_overrides", {}) if operations_cfg else {}
    configured = overrides.get(op.name, "")
    return {
        "name": op.name,
        "description": op.description,
        "file": op.file,
        "lines": op.lines,
        "invocation": op.invocation,
        "tier": op.tier,
        "default_model": op.default_model,
        "category": op.category,
        "configured_model": configured or op.default_model,
        "has_override": bool(configured),
    }


@router.get("")
async def list_operations(
    request: Request,
    tier: str | None = None,
    category: str | None = None,
):
    """List all registered maintenance LLM operations.

    Query params:
        tier: filter by 'direct' or 'routed'
        category: filter by category (e.g. 'night_shift', 'marketplace')
    """
    rc = getattr(request.app.state, "runtime_config", None)
    ops = get_all_operations()

    if tier:
        ops = [o for o in ops if o.tier == tier]
    if category:
        ops = [o for o in ops if o.category == category]

    items = [_op_to_dict(o, rc) for o in ops]
    return {"operations": items, "total": len(items)}


@router.get("/{name}")
async def get_operation_detail(name: str, request: Request):
    """Get a single operation by name."""
    op = get_operation(name)
    if not op:
        raise HTTPException(404, f"Operation not found: {name}")

    rc = getattr(request.app.state, "runtime_config", None)
    return _op_to_dict(op, rc)
