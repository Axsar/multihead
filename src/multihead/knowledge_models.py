"""Knowledge layer models for Night Shift, Claims, Events, and Context Packs."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from multihead.models import new_id


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EventStatus(str, enum.Enum):
    DRAFT = "draft"
    CONFIRMED = "confirmed"
    CORRECTED = "corrected"
    RETRACTED = "retracted"


class ClaimStatus(str, enum.Enum):
    PROPOSED = "proposed"
    CORROBORATED = "corroborated"  # Second independent channel agrees
    ACCEPTED = "accepted"
    CONTESTED = "contested"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"
    DEPRECATED = "deprecated"
    RESOLVED = "resolved"
    STALE = "stale"  # Source file changed since claim was derived


class ClaimType(str, enum.Enum):
    DEFINITION = "definition"
    DECISION = "decision"
    FACT = "fact"
    CONSTRAINT = "constraint"
    PREFERENCE = "preference"
    PLAN = "plan"
    ASSUMPTION = "assumption"
    RISK = "risk"
    QUESTION = "question"


class EventType(str, enum.Enum):
    DECISION = "decision"
    TASK_CREATED = "task_created"
    TASK_COMPLETED = "task_completed"
    TOOL_RUN = "tool_run"
    MEETING = "meeting"
    COMMIT = "commit"
    DEPLOYMENT = "deployment"
    PAYMENT = "payment"
    NOTE = "note"
    MILESTONE = "milestone"
    QUESTION = "question"
    ANSWER = "answer"
    SPEC_CHANGE = "spec_change"
    INCIDENT = "incident"


class ScopeType(str, enum.Enum):
    GLOBAL = "global"
    PROJECT = "project"
    REPO = "repo"
    PERSON = "person"
    SESSION = "session"
    AGENT = "agent"


class TimePrecision(str, enum.Enum):
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    UNKNOWN = "unknown"


class Stability(str, enum.Enum):
    VOLATILE = "volatile"
    TEMPORARY = "temporary"
    MEDIUM = "medium"
    HIGH = "high"
    STABLE = "stable"


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SpanRef(BaseModel):
    start: int
    end: int
    unit: str = "chars"  # chars | bytes | tokens


class LocatorRef(BaseModel):
    page: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    json_path: str | None = None


class EvidencePointer(BaseModel):
    evidence_id: str = ""
    record_id: str
    uri: str = ""
    sha256: str = ""
    span: SpanRef | None = None
    locator: LocatorRef | None = None
    quote: str = ""
    captured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def model_post_init(self, __context: Any) -> None:
        if not self.evidence_id:
            self.evidence_id = new_id("evp_")


class ActorRef(BaseModel):
    actor_type: str  # user | person | agent | service
    actor_id: str
    display: str = ""


class EntityRef(BaseModel):
    entity_type: str  # project | repo | file | model | tool | company | etc.
    entity_id: str
    label: str = ""


class ValueObject(BaseModel):
    value_type: str  # enum | string | number | boolean | json
    value: Any


class TimeBlock(BaseModel):
    happened_at: datetime
    ended_at: datetime | None = None
    timezone: str = "UTC"
    time_precision: TimePrecision = TimePrecision.UNKNOWN


class Provenance(BaseModel):
    produced_by: dict[str, str]  # {"kind": "extractor"|"user"|"agent", "id": "...", "participant_id": "par_..."}
    toolchain: list[dict[str, str]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Grounding metadata (triangulation pattern)
    observation_method: str = ""  # bash_output|user_statement|assistant_statement|code_read|git_history|document_read|inference
    source_anchor: dict[str, str] = Field(default_factory=dict)  # {"file_path": "...", "symbol": "...", "git_sha": "...", "line": "..."}
    speaker: str = ""  # human|assistant|tool — who originated this claim
    valence: str = ""  # frustrated|correction|excited|neutral|low_signal — user emotional signal
    evidence: list[dict[str, str]] = Field(default_factory=list)  # [{"type": "code_read", "file": "...", "line": "...", "text": "..."}]


class Participant(BaseModel):
    """Registered participant in the knowledge base.

    Each unique environment (repo + machine) gets a stable participant
    identity that persists across shell restarts.
    """

    participant_id: str = ""
    name: str = ""
    agent_type: str = "shell"  # shell, claude-code, worker, external, service
    context_hash: str = ""  # SHA-256 of environment fingerprint
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, str] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if not self.participant_id:
            self.participant_id = new_id("par_")


# ---------------------------------------------------------------------------
# Record (immutable evidence source)
# ---------------------------------------------------------------------------

class Record(BaseModel):
    record_id: str = ""
    uri: str
    sha256: str = ""
    mime: str = ""
    captured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def model_post_init(self, __context: Any) -> None:
        if not self.record_id:
            self.record_id = new_id("rec_")


# ---------------------------------------------------------------------------
# Knowledge Event (semantic events — distinct from RunEvent)
# ---------------------------------------------------------------------------

class KnowledgeEvent(BaseModel):
    """Semantic event: 'what happened' with evidence pointers."""
    event_id: str = ""
    event_status: EventStatus = EventStatus.DRAFT
    event_type: EventType
    title: str
    summary: str = ""
    time: TimeBlock
    actors: list[ActorRef] = Field(default_factory=list)
    entities: list[EntityRef] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    topic_ids: list[str] = Field(default_factory=list)
    evidence_supports: list[EvidencePointer] = Field(default_factory=list)
    evidence_related: list[EvidencePointer] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    caused_by_event_ids: list[str] = Field(default_factory=list)
    supersedes_event_ids: list[str] = Field(default_factory=list)
    duplicates_event_ids: list[str] = Field(default_factory=list)
    provenance: Provenance

    def model_post_init(self, __context: Any) -> None:
        if not self.event_id:
            self.event_id = new_id("evt_")


# ---------------------------------------------------------------------------
# Claim (something asserted as true, with evidence and lifecycle)
# ---------------------------------------------------------------------------

class ClaimScope(BaseModel):
    scope_type: ScopeType
    scope_id: str
    visibility: str = "private"  # private | shared
    valid_from: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    valid_to: datetime | None = None


class ClaimCanonical(BaseModel):
    claim_key: str  # dot-separated stable path, e.g. "project.multihead.core_model.mode"
    subject: EntityRef
    predicate: str
    object: ValueObject


class Claim(BaseModel):
    claim_id: str = ""
    claim_status: ClaimStatus = ClaimStatus.PROPOSED
    claim_type: ClaimType
    scope: ClaimScope
    canonical: ClaimCanonical
    statement: str
    rationale: str = ""
    evidence_supports: list[EvidencePointer] = Field(default_factory=list)
    evidence_opposes: list[EvidencePointer] = Field(default_factory=list)
    confidence: float = 0.0
    stability: Stability = Stability.MEDIUM
    importance: float = 0.0
    derived_from_event_ids: list[str] = Field(default_factory=list)
    related_claim_ids: list[str] = Field(default_factory=list)
    conflicts_with_claim_ids: list[str] = Field(default_factory=list)
    superseded_by_claim_id: str | None = None
    rejection_reason: str | None = None
    contested_reason: str | None = None
    provenance: Provenance
    signature: str = ""
    signed_by: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.claim_id:
            self.claim_id = new_id("clm_")

    def canonical_json_for_signing(self) -> str:
        """Return deterministic JSON of identity-bearing fields for HMAC signing."""
        import json
        return json.dumps({
            "claim_id": self.claim_id,
            "claim_key": self.canonical.claim_key,
            "predicate": self.canonical.predicate,
            "subject": self.canonical.subject.model_dump(mode="json"),
            "object": self.canonical.object.model_dump(mode="json"),
            "statement": self.statement,
        }, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Link (cross-topic connection)
# ---------------------------------------------------------------------------

class Link(BaseModel):
    link_id: str = ""
    from_entity: EntityRef
    to_entity: EntityRef
    reason_type: str  # shared_entity | shared_constraint | repeated_blocker | etc.
    reason: str = ""
    score: float = 0.0
    evidence: list[EvidencePointer] = Field(default_factory=list)
    link_status: str = "draft"  # draft | confirmed
    provenance: Provenance

    def model_post_init(self, __context: Any) -> None:
        if not self.link_id:
            self.link_id = new_id("lnk_")


# ---------------------------------------------------------------------------
# Context Pack models
# ---------------------------------------------------------------------------

class PackItem(BaseModel):
    type: str  # claim | event | snippet | artifact
    text: str
    priority: float = 0.0
    evidence_refs: list[str] = Field(default_factory=list)
    token_estimate: int = 0
    why_included: str = ""
    source_id: str = ""  # claim_id or event_id
    status: str = ""  # claim_status or event_status
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    embedding: list[float] | None = Field(default=None, exclude=True)


class ContextPack(BaseModel):
    pack_id: str = ""
    purpose: str = ""
    budgets: dict[str, int] = Field(default_factory=lambda: {"max_tokens": 4000, "max_items": 50})
    items: list[PackItem] = Field(default_factory=list)
    dropped: list[dict[str, str]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def model_post_init(self, __context: Any) -> None:
        if not self.pack_id:
            self.pack_id = new_id("pack_")


# ---------------------------------------------------------------------------
# Night Shift config and report
# ---------------------------------------------------------------------------

class NightShiftConfig(BaseModel):
    max_tokens_per_day: int = 100_000
    max_pack_tokens: int = 4000
    max_claims_per_day: int = 100000
    input_window_hours: int = 24
    auto_accept_confidence: float = 0.85
    auto_accept_min_supports: int = 2
    head_id: str = "qwen-llm"  # which head to use for extraction (local Qwen, not Claude)
    concurrency: int = 1  # parallel LLM calls (1=sequential, >1=async semaphore)
    reasoning_head_id: str | None = None  # head for reasoning-heavy stages (if different)
    reasoning_stages: list[str] = Field(default_factory=lambda: [
        "consistency_check", "open_loops", "daily_brief",
        "weekly_rollup", "cross_topic_links", "backlog_sweep",
    ])
    batch_mode: bool = False  # Use Anthropic Batch API (50% cheaper, async)
    no_wait: bool = False  # Submit batches and exit — don't poll, resume next run


class NightShiftReport(BaseModel):
    report_id: str = ""
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    stages_completed: list[str] = Field(default_factory=list)
    stages_skipped: list[str] = Field(default_factory=list)
    stages_failed: list[str] = Field(default_factory=list)
    records_processed: int = 0
    events_created: int = 0
    claims_created: int = 0
    packs_built: list[str] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        if not self.report_id:
            self.report_id = new_id("nsr_")
