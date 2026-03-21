"""API routes for consensus execution and inspection."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

from ..consensus import (
    ConsensusConfig,
    ConsensusEngine,
    ConsensusResult,
    ConsensusStrategy,
    FirstToAheadConfig,
    HeadTask,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class HeadTaskRequest(BaseModel):
    head_id: str
    prompt_template: str = ""
    weight: float = Field(default=1.0, gt=0, le=100)
    required: bool = True
    extract_fields: list[str] = Field(default_factory=list)


class FirstToAheadRequest(BaseModel):
    k_margin: int = Field(default=3, ge=1, le=50)
    max_samples: int = Field(default=25, ge=3, le=100)
    min_samples: int = Field(default=3, ge=1, le=50)
    stall_threshold: int = Field(default=9, ge=1, le=100)
    red_flag_max_tokens: int = Field(default=700, ge=10, le=10000)
    red_flag_must_parse: bool = False


class ConsensusExecuteRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=100_000)
    heads: list[HeadTaskRequest] = Field(..., min_length=1, max_length=50)
    strategy: str = "majority"
    threshold: float = Field(default=0.5, ge=0, le=1)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    cross_modal: bool = False
    fail_on_disagreement: bool = False
    timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    first_to_ahead: FirstToAheadRequest | None = None


class VoteResponse(BaseModel):
    head_id: str
    outputs: dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error: str | None = None
    schema_valid: bool = True
    latency_ms: float = 0.0


class ConsensusResponse(BaseModel):
    consensus_outputs: dict[str, Any] = Field(default_factory=dict)
    all_votes: list[VoteResponse] = Field(default_factory=list)
    agreement_score: float = 0.0
    red_flags: list[dict[str, Any]] = Field(default_factory=list)
    strategy_used: str = "majority"
    metrics: dict[str, Any] = Field(default_factory=dict)


class StrategyInfo(BaseModel):
    name: str
    description: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/execute", response_model=ConsensusResponse)
async def execute_consensus(body: ConsensusExecuteRequest, request: Request):
    """Execute a consensus query across multiple heads."""
    head_manager = request.app.state.head_manager
    metrics = getattr(request.app.state, "metrics", None)

    # Validate strategy
    try:
        strategy = ConsensusStrategy(body.strategy)
    except ValueError:
        valid = [s.value for s in ConsensusStrategy]
        raise HTTPException(400, f"Invalid strategy '{body.strategy}'. Valid: {valid}")

    # Validate head IDs exist
    available_heads = set(head_manager.get_states().keys())
    for ht in body.heads:
        if ht.head_id not in available_heads:
            raise HTTPException(
                404, f"Head '{ht.head_id}' not found. Available: {sorted(available_heads)}"
            )

    # Build FTA config if provided
    fta_config = None
    if body.first_to_ahead:
        fta_config = FirstToAheadConfig(
            k_margin=body.first_to_ahead.k_margin,
            max_samples=body.first_to_ahead.max_samples,
            min_samples=body.first_to_ahead.min_samples,
            stall_threshold=body.first_to_ahead.stall_threshold,
            red_flag_max_tokens=body.first_to_ahead.red_flag_max_tokens,
            red_flag_must_parse=body.first_to_ahead.red_flag_must_parse,
        )

    config = ConsensusConfig(
        heads=[
            HeadTask(
                head_id=ht.head_id,
                prompt_template=ht.prompt_template,
                weight=ht.weight,
                required=ht.required,
                extract_fields=ht.extract_fields,
            )
            for ht in body.heads
        ],
        strategy=strategy,
        threshold=body.threshold,
        output_schema=body.output_schema,
        cross_modal=body.cross_modal,
        fail_on_disagreement=body.fail_on_disagreement,
        timeout_seconds=body.timeout_seconds,
        first_to_ahead=fta_config,
    )

    engine = ConsensusEngine(head_manager, metrics=metrics)

    try:
        result = await engine.execute(config, body.prompt)
    except ValueError as e:
        logger.warning("Consensus execution failed: %s", e)
        raise HTTPException(422, "Consensus execution failed: invalid configuration")

    return ConsensusResponse(
        consensus_outputs=result.consensus_outputs,
        all_votes=[
            VoteResponse(
                head_id=v.head_id,
                outputs=v.outputs,
                success=v.success,
                error=v.error,
                schema_valid=v.schema_valid,
                latency_ms=v.latency_ms,
            )
            for v in result.all_votes
        ],
        agreement_score=result.agreement_score,
        red_flags=result.red_flags,
        strategy_used=result.strategy_used.value,
        metrics=result.metrics,
    )


@router.get("/strategies", response_model=list[StrategyInfo])
async def list_strategies():
    """List available consensus strategies."""
    return [
        StrategyInfo(
            name="majority",
            description="Simple majority vote: most common output wins.",
        ),
        StrategyInfo(
            name="weighted",
            description="Weighted vote: heads with higher weight have more influence.",
        ),
        StrategyInfo(
            name="unanimous",
            description="All heads must produce identical output. Score is 1.0 or fraction agreeing.",
        ),
        StrategyInfo(
            name="threshold",
            description="Majority vote with minimum agreement percentage required.",
        ),
        StrategyInfo(
            name="first_to_ahead",
            description=(
                "MAKER-style dynamic sampling: samples until one answer leads by k_margin votes. "
                "Discards red-flagged outputs pre-vote. Escalates temperature on stall."
            ),
        ),
    ]
