"""API routes for knowledge store (events, claims, records)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from multihead.knowledge_models import (
    Claim,
    ClaimCanonical,
    ClaimScope,
    ClaimStatus,
    ClaimType,
    EntityRef,
    EventStatus,
    EventType,
    KnowledgeEvent,
    Provenance,
    ScopeType,
    Stability,
    TimeBlock,
    TimePrecision,
    ValueObject,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request bodies for POST endpoints
# ---------------------------------------------------------------------------

class CreateClaimBody(BaseModel):
    """Simplified claim creation — server fills in full model details."""
    claim_key: str  # e.g. "project.auth.status"
    statement: str
    subject_type: str = "component"  # entity_type
    subject_id: str = ""  # entity_id
    subject_label: str = ""
    predicate: str = "has_state"
    value: Any = True
    value_type: str = "string"  # enum | string | number | boolean | json
    claim_type: str = "fact"
    claim_status: str = "accepted"
    scope_type: str = "project"
    scope_id: str = "default"
    confidence: float = 0.9
    stability: str = "medium"
    importance: float = 0.5
    rationale: str = ""
    produced_by: str = "external"  # who created this claim
    tags: list[str] = Field(default_factory=list)


class UpdateClaimBody(BaseModel):
    """Update a claim's status and/or confidence."""
    claim_status: str | None = None  # proposed | accepted | disputed | retracted
    confidence: float | None = None  # 0.0 – 1.0


class CreateEventBody(BaseModel):
    """Simplified event creation — server fills in full model details."""
    title: str
    summary: str = ""
    event_type: str = "note"
    event_status: str = "confirmed"
    tags: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    produced_by: str = "external"
    entities: list[dict[str, str]] = Field(default_factory=list)  # [{"type": "component", "id": "auth"}]


@router.get("/events")
async def list_events(
    request: Request,
    status: str | None = None,
    event_type: str | None = None,
    limit: int = Query(default=50, le=100),
) -> list[dict[str, Any]]:
    """List knowledge events with optional filtering."""
    ks = request.app.state.knowledge_store
    events = ks.list_events(status=status, event_type=event_type, limit=limit)
    return [
        {
            "event_id": e.event_id,
            "event_type": e.event_type.value,
            "event_status": e.event_status.value,
            "title": e.title,
            "summary": e.summary,
            "created_at": e.provenance.created_at.isoformat(),
        }
        for e in events
    ]


@router.get("/claims")
async def list_claims(
    request: Request,
    status: str | None = None,
    claim_type: str | None = None,
    scope_id: str | None = None,
    limit: int = Query(default=50, le=100),
) -> list[dict[str, Any]]:
    """List claims with optional filtering."""
    ks = request.app.state.knowledge_store
    claims = ks.list_claims(status=status, claim_type=claim_type, scope_id=scope_id, limit=limit)
    return [
        {
            "claim_id": c.claim_id,
            "claim_type": c.claim_type.value,
            "claim_status": c.claim_status.value,
            "statement": c.statement,
            "confidence": c.confidence,
            "scope_id": c.scope.scope_id,
            "claim_key": c.canonical.claim_key,
            "created_at": c.provenance.created_at.isoformat(),
        }
        for c in claims
    ]


@router.get("/records")
async def list_records(
    request: Request,
    limit: int = Query(default=100, le=500),
) -> list[dict[str, Any]]:
    """List ingested records."""
    ks = request.app.state.knowledge_store
    records = ks.list_records(limit=limit)
    return [
        {
            "record_id": r.record_id,
            "uri": r.uri,
            "mime": r.mime,
            "captured_at": r.captured_at.isoformat(),
        }
        for r in records
    ]


@router.get("/events/{event_id}")
async def get_event(event_id: str, request: Request) -> dict[str, Any]:
    """Get a specific knowledge event."""
    ks = request.app.state.knowledge_store
    e = ks.get_event_by_id(event_id)
    if e is None:
        raise HTTPException(404, f"Event not found: {event_id}")
    return {
        "event_id": e.event_id,
        "event_type": e.event_type.value,
        "event_status": e.event_status.value,
        "title": e.title,
        "summary": e.summary,
        "created_at": e.provenance.created_at.isoformat(),
    }


@router.get("/claims/{claim_id}")
async def get_claim(claim_id: str, request: Request) -> dict[str, Any]:
    """Get a specific claim."""
    ks = request.app.state.knowledge_store
    c = ks.get_claim_by_id(claim_id)
    if c is None:
        raise HTTPException(404, f"Claim not found: {claim_id}")
    return {
        "claim_id": c.claim_id,
        "claim_type": c.claim_type.value,
        "claim_status": c.claim_status.value,
        "statement": c.statement,
        "confidence": c.confidence,
        "scope_id": c.scope.scope_id,
        "claim_key": c.canonical.claim_key,
        "created_at": c.provenance.created_at.isoformat(),
    }


@router.patch("/claims/{claim_id}")
async def update_claim(claim_id: str, body: UpdateClaimBody, request: Request) -> dict[str, Any]:
    """Update a claim's status and/or confidence.

    Preserves immutability audit trail: the original claim is marked superseded
    and a new claim is created with the updated fields.
    """
    ks = request.app.state.knowledge_store
    original = ks.get_claim_by_id(claim_id)
    if original is None:
        raise HTTPException(404, f"Claim not found: {claim_id}")

    new_status = ClaimStatus(body.claim_status) if body.claim_status else original.claim_status
    new_confidence = body.confidence if body.confidence is not None else original.confidence

    # If only confidence changed (no status change), update in-place
    if new_status == original.claim_status and body.confidence is not None:
        with ks._connect() as conn:
            conn.execute(
                "UPDATE claims SET confidence = ?, updated_at = ? WHERE claim_id = ?",
                (new_confidence, datetime.now(timezone.utc).isoformat(), claim_id),
            )
        updated = ks.get_claim_by_id(claim_id)
        return {
            "claim_id": updated.claim_id,
            "claim_status": updated.claim_status.value,
            "confidence": updated.confidence,
            "supersedes": None,
        }

    # Status changed: create superseding claim to preserve audit trail
    now = datetime.now(timezone.utc)
    new_claim = Claim(
        claim_status=new_status,
        claim_type=original.claim_type,
        scope=original.scope,
        canonical=original.canonical,
        statement=original.statement,
        rationale=original.rationale,
        confidence=new_confidence,
        stability=original.stability,
        importance=original.importance,
        related_claim_ids=[original.claim_id],
        provenance=Provenance(
            produced_by={"kind": "curation", "id": "cortex"},
        ),
    )
    new_claim = ks.insert_claim(new_claim)

    # Mark original as superseded
    ks.update_claim_status(claim_id, ClaimStatus.SUPERSEDED, superseded_by=new_claim.claim_id)

    return {
        "claim_id": new_claim.claim_id,
        "claim_status": new_claim.claim_status.value,
        "confidence": new_claim.confidence,
        "supersedes": claim_id,
    }


# ---------------------------------------------------------------------------
# Briefing endpoint — "what should I know?"
# ---------------------------------------------------------------------------

@router.get("/briefing")
async def briefing(
    request: Request,
    component: str = Query(..., description="Component name, e.g. 'auth', 'payments', 'deploy'"),
    scope_id: str = Query(default="default", description="Project scope"),
    include_events: bool = Query(default=True, description="Include recent events"),
    max_claims: int = Query(default=20, le=100),
    max_events: int = Query(default=10, le=50),
) -> dict[str, Any]:
    """Get a briefing for a component — relevant claims, recent events, and upstream context.

    This is the "read" endpoint. A component calls this at startup to learn
    what it needs to know before running.
    """
    ks = request.app.state.knowledge_store

    # 1. Claims whose key starts with the component prefix
    all_claims = ks.list_claims(scope_id=scope_id, limit=500)
    component_claims = []
    related_claims = []
    for c in all_claims:
        key = c.canonical.claim_key
        # Direct match: component name in key
        if component in key:
            component_claims.append(c)
        # Also grab claims from upstream/downstream that mention this component in statement
        elif component.replace("_", " ") in c.statement.lower() or component in c.statement.lower():
            related_claims.append(c)

    component_claims = component_claims[:max_claims]
    related_claims = related_claims[:max_claims]

    # 2. Recent events mentioning this component
    events_out = []
    if include_events:
        all_events = ks.list_events(limit=200)
        for e in all_events:
            # Check tags, entities, title, summary for component mention
            matched = (
                component in (e.summary or "").lower()
                or component in e.title.lower()
                or any(component in t.lower() for t in e.tags)
                or any(component in ent.entity_id for ent in e.entities)
            )
            if matched:
                events_out.append(e)
            if len(events_out) >= max_events:
                break

    return {
        "component": component,
        "scope_id": scope_id,
        "claims": [
            {
                "claim_id": c.claim_id,
                "claim_key": c.canonical.claim_key,
                "statement": c.statement,
                "claim_type": c.claim_type.value,
                "confidence": c.confidence,
                "status": c.claim_status.value,
            }
            for c in component_claims
        ],
        "related_claims": [
            {
                "claim_id": c.claim_id,
                "claim_key": c.canonical.claim_key,
                "statement": c.statement,
                "claim_type": c.claim_type.value,
                "confidence": c.confidence,
                "status": c.claim_status.value,
            }
            for c in related_claims
        ],
        "recent_events": [
            {
                "event_id": e.event_id,
                "title": e.title,
                "summary": e.summary,
                "event_type": e.event_type.value,
                "created_at": e.provenance.created_at.isoformat(),
            }
            for e in events_out
        ],
        "summary": f"{len(component_claims)} direct claims, {len(related_claims)} related, {len(events_out)} events",
    }


# ---------------------------------------------------------------------------
# POST endpoints
# ---------------------------------------------------------------------------

@router.post("/claims")
async def create_claim(body: CreateClaimBody, request: Request) -> dict[str, Any]:
    """Create a new claim from a simplified body."""
    ks = request.app.state.knowledge_store
    now = datetime.now(timezone.utc)
    try:
        claim = Claim(
            claim_status=ClaimStatus(body.claim_status),
            claim_type=ClaimType(body.claim_type),
            scope=ClaimScope(
                scope_type=ScopeType(body.scope_type),
                scope_id=body.scope_id,
                valid_from=now,
            ),
            canonical=ClaimCanonical(
                claim_key=body.claim_key,
                subject=EntityRef(
                    entity_type=body.subject_type,
                    entity_id=body.subject_id or body.claim_key.split(".")[0],
                    label=body.subject_label,
                ),
                predicate=body.predicate,
                object=ValueObject(value_type=body.value_type, value=body.value),
            ),
            statement=body.statement,
            rationale=body.rationale,
            confidence=body.confidence,
            stability=Stability(body.stability),
            importance=body.importance,
            provenance=Provenance(
                produced_by={"kind": "external", "id": body.produced_by},
            ),
        )
        claim = ks.insert_claim(claim)
    except Exception as e:
        raise HTTPException(400, str(e))
    return {
        "claim_id": claim.claim_id,
        "claim_key": claim.canonical.claim_key,
        "statement": claim.statement,
        "claim_status": claim.claim_status.value,
    }


@router.post("/events")
async def create_event(body: CreateEventBody, request: Request) -> dict[str, Any]:
    """Create a new knowledge event from a simplified body."""
    ks = request.app.state.knowledge_store
    now = datetime.now(timezone.utc)
    try:
        entities = [
            EntityRef(entity_type=e.get("type", "component"), entity_id=e["id"], label=e.get("label", ""))
            for e in body.entities
        ]
        event = KnowledgeEvent(
            event_status=EventStatus(body.event_status),
            event_type=EventType(body.event_type),
            title=body.title,
            summary=body.summary,
            time=TimeBlock(happened_at=now, time_precision=TimePrecision.SECOND),
            tags=body.tags,
            entities=entities,
            metrics=body.metrics,
            provenance=Provenance(
                produced_by={"kind": "external", "id": body.produced_by},
            ),
        )
        event = ks.insert_event(event)
    except Exception as e:
        raise HTTPException(400, str(e))
    return {
        "event_id": event.event_id,
        "event_type": event.event_type.value,
        "title": event.title,
        "event_status": event.event_status.value,
    }
