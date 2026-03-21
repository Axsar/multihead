"""API routes for runtime configuration."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class ConfigSetRequest(BaseModel):
    key: str
    value: str


@router.get("")
async def get_config(request: Request):
    """Get current runtime config."""
    config = request.app.state.runtime_config
    return config.model_dump()


@router.post("/set")
async def set_config(body: ConfigSetRequest, request: Request):
    """Set a runtime config value."""
    config = request.app.state.runtime_config
    config_path = request.app.state.runtime_config_path
    try:
        result = config.set_value(body.key, body.value)
        config.save(config_path)
        return {"key": body.key, "value": body.value, "message": result}
    except ValueError as e:
        raise HTTPException(400, str(e))
