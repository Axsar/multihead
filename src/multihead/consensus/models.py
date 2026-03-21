"""Consensus data models: enums, configs, and result types."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ConsensusStrategy(str, Enum):
    MAJORITY = "majority"
    WEIGHTED = "weighted"
    UNANIMOUS = "unanimous"
    THRESHOLD = "threshold"
    FIRST_TO_AHEAD = "first_to_ahead"


# ---------------------------------------------------------------------------
# Config models
# ---------------------------------------------------------------------------


class HeadTask(BaseModel):
    """Configuration for one head in a consensus step."""

    head_id: str
    prompt_template: str = ""  # Empty = use base prompt; set for cross-modal
    weight: float = 1.0
    required: bool = True
    extract_fields: list[str] = Field(default_factory=list)


class FirstToAheadConfig(BaseModel):
    """Configuration for FIRST_TO_AHEAD (MAKER-style) dynamic voting.

    The algorithm samples candidates sequentially, discards red-flagged
    outputs, clusters remaining into equivalence buckets via canonical
    hashing, and stops when one bucket leads by k_margin votes.
    """

    k_margin: int = 3           # leader must be ahead by this many
    max_samples: int = 25       # hard cap on total samples
    min_samples: int = 3        # don't stop on early luck
    stall_threshold: int = 9    # escalate after this many without winner
    red_flag_max_tokens: int = 700   # discard outputs longer than this (token approx)
    red_flag_must_parse: bool = False  # discard non-JSON when output_schema expects it


class ConsensusConfig(BaseModel):
    """Configuration for multi-head consensus execution."""

    heads: list[HeadTask]
    strategy: ConsensusStrategy = ConsensusStrategy.MAJORITY
    threshold: float = 0.5  # For THRESHOLD strategy
    output_schema: dict[str, Any] = Field(default_factory=dict)
    cross_modal: bool = False
    fail_on_disagreement: bool = False
    timeout_seconds: float = 30.0  # Per-head execution timeout
    first_to_ahead: FirstToAheadConfig | None = None  # Config for FTA strategy


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class VoteResult(BaseModel):
    """Result from a single head in consensus."""

    head_id: str
    outputs: dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error: str | None = None
    schema_valid: bool = True
    schema_errors: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0


class ConsensusResult(BaseModel):
    """Final consensus output with audit trail."""

    consensus_outputs: dict[str, Any] = Field(default_factory=dict)
    all_votes: list[VoteResult] = Field(default_factory=list)
    agreement_score: float = 0.0
    red_flags: list[dict[str, Any]] = Field(default_factory=list)
    strategy_used: ConsensusStrategy = ConsensusStrategy.MAJORITY
    metrics: dict[str, Any] = Field(default_factory=dict)
