"""Core data models for MultiHead orchestration."""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from ulid import ULID


def new_id(prefix: str = "") -> str:
    """Generate a ULID-based ID with optional prefix."""
    uid = str(ULID())
    return f"{prefix}{uid}" if prefix else uid


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class HeadState(str, enum.Enum):
    OFF = "off"
    LOADING = "loading"
    ACTIVE = "active"
    SLEEPING = "sleeping"
    UNLOADING = "unloading"
    ERROR = "error"


class RunStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, enum.Enum):
    PENDING = "pending"
    STARTED = "started"
    OUTPUT_WRITTEN = "output_written"
    COMMITTED = "committed"
    FAILED = "failed"
    SKIPPED = "skipped"


class EventKind(str, enum.Enum):
    RUN_CREATED = "run_created"
    STEP_PLANNED = "step_planned"
    STEP_STARTED = "step_started"
    STEP_OUTPUT_WRITTEN = "step_output_written"
    STEP_COMMITTED = "step_committed"
    STEP_FAILED = "step_failed"
    STEP_REFLECTION = "step_reflection"  # Reflection loop triggered
    STEP_REFINED = "step_refined"  # Step retried with refined prompt
    STEP_TOT_STARTED = "step_tot_started"  # Tree-of-Thoughts exploration started
    STEP_TOT_COMPLETE = "step_tot_complete"  # Tree-of-Thoughts exploration finished
    STEP_PRM_SCORED = "step_prm_scored"  # Process Reward Model scored the step
    RUN_DONE = "run_done"
    RUN_FAILED = "run_failed"
    RUN_CANCELLED = "run_cancelled"


class AdapterKind(str, enum.Enum):
    OLLAMA = "ollama"
    VLLM = "vllm"
    TRANSFORMERS = "transformers"
    OPENAI = "openai"
    EMBEDDING = "embedding"
    CLAUDE = "claude"
    CLAUDE_SESSION = "claude_session"  # Multi-session consensus via KB
    MOCK = "mock"
    DETERMINISTIC = "deterministic"  # Pure Python functions
    MESH = "mesh"  # Remote mesh peer
    CLAUDE_AGENT_SDK = "claude_agent_sdk"  # Claude via Agent SDK
    ANTHROPIC = "anthropic"  # Anthropic Messages API (real-time + batch)
    BOTVIBES = "botvibes"  # BotVibes marketplace


class SolverType(str, enum.Enum):
    """Types of solvers MultiHead can route to."""
    LLM = "llm"
    VLM = "vlm"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    EMBEDDING = "embedding"
    DETERMINISTIC = "deterministic"
    VALIDATOR = "validator"
    TOOL = "tool"
    EXTERNAL_SERVICE = "external_service"
    HUMAN = "human"


class DataSensitivity(str, enum.Enum):
    """Data sensitivity levels for privacy-aware routing."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class CheckpointMode(str, enum.Enum):
    SYNC = "sync"
    ASYNC = "async"


# ---------------------------------------------------------------------------
# Solver capabilities (Phase 1: capability-aware routing)
# ---------------------------------------------------------------------------

class Capability(BaseModel):
    """Fine-grained capability description for a solver."""

    solver_type: SolverType
    input_modalities: list[str] = Field(default_factory=list)  # ["text", "image", "code"]
    output_modalities: list[str] = Field(default_factory=list)  # ["text", "json", "image"]
    task_types: list[str] = Field(default_factory=list)  # Specific tasks solver can do

    # Quality benchmarks
    accuracy_score: float | None = None  # 0.0-1.0
    latency_p50_ms: float | None = None
    cost_per_call: float | None = None  # USD

    # Resource requirements
    requires_gpu: bool = False
    vram_mb: int = 0

    # Constraints
    max_input_tokens: int | None = None
    max_image_size: tuple[int, int] | None = None
    supported_formats: list[str] = Field(default_factory=list)

    # Additional benchmarks (flexible)
    benchmarks: dict[str, float] = Field(default_factory=dict)


class PrivacyConstraint(BaseModel):
    """Privacy constraints for solver selection."""

    data_sensitivity: DataSensitivity = DataSensitivity.PUBLIC
    allowed_providers: list[str] | None = None  # Whitelist
    blocked_providers: list[str] | None = None  # Blacklist
    require_encryption: bool = False
    max_data_retention_days: int = 0
    geographic_restrictions: list[str] = Field(default_factory=list)


class BudgetConstraint(BaseModel):
    """Budget constraints for solver selection (Phase 6: RFQ)."""

    max_cost_per_step: float | None = None  # USD per step
    max_total_cost: float | None = None  # Total budget cap (USD)
    max_latency_ms: int | None = None  # Max latency per step (milliseconds)
    max_total_time_s: int | None = None  # Max total time (seconds)


# ---------------------------------------------------------------------------
# Head manifest (from heads.yaml)
# ---------------------------------------------------------------------------

class HeadManifest(BaseModel):
    head_id: str
    name: str
    adapter: AdapterKind
    model: str  # e.g. "qwen3:8b" for Ollama, "Qwen/Qwen3-VL-32B-Thinking" for transformers
    kind: str = "llm"  # llm | vlm | embed | tool
    endpoint: str = ""  # base URL for remote adapters
    gpu_required: bool = True
    vram_hint_mb: int = 0
    cpu_ram_hint_mb: int = 0
    quantization: str | None = None  # "4bit", "8bit", etc. for transformers
    extra: dict[str, Any] = Field(default_factory=dict)

    # Phase 1: Capability-aware routing
    capabilities: Capability | None = None  # Detailed capability description
    is_local: bool = True  # False for external APIs, BotVibes
    privacy_level: str = "local"  # "local" | "encrypted" | "external"


# ---------------------------------------------------------------------------
# WorkOrder & Steps (recipe definition)
# ---------------------------------------------------------------------------

class StepDef(BaseModel):
    step_id: str = ""
    name: str
    head_id: str = ""  # Empty = resolved by Router when required_kind/task_types set
    prompt_template: str = ""  # Jinja2-ish template or raw prompt
    input_refs: list[str] = Field(default_factory=list)  # artifact refs or "user_input"
    depends_on: list[str] = Field(default_factory=list)  # explicit step_id dependencies
    required_kind: str | None = None  # "llm", "vlm", etc. — triggers routing
    output_schema: dict[str, Any] = Field(default_factory=dict)
    retry_policy: dict[str, Any] = Field(default_factory=lambda: {"max_attempts": 1, "backoff_ms": 1000})
    checkpoint_mode: CheckpointMode = CheckpointMode.SYNC
    extra: dict[str, Any] = Field(default_factory=dict)
    consensus: Any = None  # ConsensusConfig from consensus.py (avoids circular import)
    fallback: list[str] = Field(default_factory=list)  # alternative head_ids on primary failure

    # Phase 1: Capability-aware routing
    task_types: list[str] = Field(default_factory=list)  # Specific tasks needed
    privacy: PrivacyConstraint | None = None  # Privacy constraints
    budget: BudgetConstraint | None = None  # Budget constraints (Phase 6)
    validator: Any = Field(default=None, exclude=True)  # Validator instance (from validators.py, excluded from serialization)
    deterministic_function: str | None = None  # For deterministic solver: "module.function"

    # Reflection loop configuration
    enable_reflection: bool = False  # Enable Actor-Evaluator-Reflect-Memory cycle
    evaluator: Any = Field(default=None, exclude=True)  # Evaluator instance (from reflection.py, excluded from serialization)
    max_reflection_attempts: int = 3  # Maximum refinement attempts via reflection

    def model_post_init(self, __context: Any) -> None:
        if not self.step_id:
            self.step_id = new_id("step_")


class WorkOrder(BaseModel):
    run_id: str = ""
    goal: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    steps: list[StepDef] = Field(default_factory=list)
    budgets: dict[str, Any] = Field(default_factory=dict)
    run_timeout_seconds: float = 300.0  # Max execution time (5 min default)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def model_post_init(self, __context: Any) -> None:
        if not self.run_id:
            self.run_id = new_id("run_")


# ---------------------------------------------------------------------------
# Stage result (what each step returns)
# ---------------------------------------------------------------------------

class ArtifactRef(BaseModel):
    artifact_id: str
    uri: str = ""
    name: str = ""
    media_type: str = ""
    size_bytes: int = 0


class StageResult(BaseModel):
    step_id: str
    head_id: str
    status: StepStatus
    outputs: dict[str, Any] = Field(default_factory=dict)
    output_artifacts: list[ArtifactRef] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None


# ---------------------------------------------------------------------------
# Run events (event-sourced state)
# ---------------------------------------------------------------------------

class RunEvent(BaseModel):
    event_id: str = ""
    run_id: str
    kind: EventKind
    step_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def model_post_init(self, __context: Any) -> None:
        if not self.event_id:
            self.event_id = new_id("evt_")


# ---------------------------------------------------------------------------
# Run state (reconstructed from events)
# ---------------------------------------------------------------------------

class RunState(BaseModel):
    run_id: str
    status: RunStatus = RunStatus.QUEUED
    work_order: WorkOrder | None = None
    current_step_index: int = 0
    step_results: dict[str, StageResult] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    ended_at: datetime | None = None
