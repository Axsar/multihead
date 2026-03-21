"""Shared models and helpers for ACP route modules."""

from __future__ import annotations

import logging

from fastapi import HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CreateTaskRequest(BaseModel):
    """Request to create a new ACP task."""

    required_capability: str
    payload_ref: str  # JSON string or artifact reference
    target_agent_id: str | None = None  # Direct targeting (skips discovery)
    conversation_id: str | None = None  # Thread tasks in a conversation
    input_schema: str = "application/json"
    output_schema: str = "application/json"
    priority: str = "normal"


class ClaimRequest(BaseModel):
    """Request body for claiming a task."""

    agent_id: str = "claude_code_proxy"


class TaskResultRequest(BaseModel):
    """Request to submit task results."""

    status: str = "complete"  # complete | failed
    output_ref: str = ""
    confidence: float | None = None
    latency_ms: int | None = None
    error_code: str | None = None
    message: str | None = None


class TokenLoginRequest(BaseModel):
    """Login to BotVibes to get a fresh JWT."""
    email: str
    password: str


class ProcureRequest(BaseModel):
    """Request to procure a service from the cloud marketplace."""

    capability: str
    payload: str
    max_price: float | None = None
    max_latency_ms: int | None = None
    min_quality: float | None = None
    quote_timeout: float = 30.0


class PostRFQRequest(BaseModel):
    """Post an RFQ to the marketplace."""

    capability_id: str
    payload_ref: str = ""
    budget_max: float = 5.0
    max_latency_ms: int = 30000
    description: str | None = None


class SubmitQuoteRequest(BaseModel):
    """Submit a quote on an RFQ."""

    listing_id: str
    unit_price: float = 0.5
    estimated_latency_ms: int = 30000
    message: str = ""


class CreateListingRequest(BaseModel):
    """Create a marketplace listing."""

    capability_id: str
    name: str = ""
    description: str = ""
    pricing_model: str = "per_call"
    unit_price: float = 0.50
    quality_score: float = 0.85


def _get_acp(request: Request):
    """Get ACP bridge from app state, raise if not connected."""
    bridge = getattr(request.app.state, "acp_bridge", None)
    if not bridge or not bridge.connected:
        raise HTTPException(503, "ACP bridge not connected to BotVibes")
    return bridge
