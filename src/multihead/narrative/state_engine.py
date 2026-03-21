"""Cross-session narrative state engine.

Builds persistent entity continuity across sessions, tracking
developers, modules, agents, and concepts with stable IDs.

Key principles:
- Deterministic: same inputs → same state
- Non-destructive: conflicts fork, never overwrite
- Evidence-driven: every state change cites sources
- Auditable: full delta log for replay
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from multihead.knowledge_models import (
    Claim,
    ClaimStatus,
    EntityRef,
    KnowledgeEvent,
)
from multihead.models import new_id
from multihead.narrative.confidence import EPISTEMIC_CEILING, ConfidenceCalibrator
from multihead.narrative.fusion import FusedNarrative

logger = logging.getLogger(__name__)

# Jaccard similarity threshold for deduplication
DEDUP_THRESHOLD = 0.70

# Confidence adjustments for state transitions
VERIFIED_BONUS = 0.05
PASS_WITH_CORRECTIONS_BONUS = 0.03
FAILED_PENALTY = -0.10


@dataclass
class TrackedEntity:
    """A persistent entity tracked across sessions."""

    entity_sid: str  # Stable ID (e.g., DEV_001, MOD_003, AGT_005)
    entity_type: str  # developer, module, agent, concept
    canonical_name: str
    aliases: list[str] = field(default_factory=list)
    first_seen_session: str = ""
    last_seen_session: str = ""
    appearance_count: int = 0
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StateDelta:
    """A single state transition (for audit trail)."""

    delta_id: str
    session_id: str
    timestamp: datetime
    action: str  # "entity_created", "entity_updated", "confidence_adjusted", "conflict_forked"
    target_sid: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class NarrativeState:
    """Complete narrative state across all sessions."""

    entities: dict[str, TrackedEntity] = field(default_factory=dict)  # entity_sid → TrackedEntity
    deltas: list[StateDelta] = field(default_factory=list)
    sessions_processed: list[str] = field(default_factory=list)
    total_claims_ingested: int = 0

    # Counters for stable ID generation
    _counters: dict[str, int] = field(default_factory=lambda: {
        "developer": 0,
        "module": 0,
        "agent": 0,
        "concept": 0,
        "file": 0,
        "unknown": 0,
    })

    def allocate_sid(self, entity_type: str) -> str:
        """Generate a new stable ID for an entity type."""
        prefix_map = {
            "developer": "DEV",
            "module": "MOD",
            "agent": "AGT",
            "concept": "CON",
            "file": "FIL",
            "unknown": "UNK",
        }
        prefix = prefix_map.get(entity_type, "ENT")
        counter = self._counters.get(entity_type, 0) + 1
        self._counters[entity_type] = counter
        return f"{prefix}_{counter:03d}"


class NarrativeStateEngine:
    """Builds cross-session narrative continuity.

    Consumes FusedNarrative units and updates persistent state:
    - Tracks entities (developers, modules, agents) with stable IDs
    - Deduplicates via name similarity (Jaccard threshold)
    - Adjusts confidence based on verification results
    - Maintains full audit trail for replay

    Usage:
        engine = NarrativeStateEngine()
        state = NarrativeState()

        # Process each session/PR/sprint narrative
        state = engine.apply(fused_narrative, state)
        state = engine.apply(next_narrative, state)

        # State accumulates across all applications
        print(state.entities)  # All tracked entities
        print(state.deltas)    # Full audit trail
    """

    def __init__(self):
        self.calibrator = ConfidenceCalibrator()

    def apply(
        self,
        narrative: FusedNarrative,
        state: NarrativeState,
    ) -> NarrativeState:
        """Apply a fused narrative to the current state.

        This is the core state transition function. It:
        1. Extracts entities from claims and events
        2. Matches against existing entities (by name)
        3. Creates new entities or updates existing ones
        4. Adjusts confidence based on verification
        5. Logs all changes to the audit trail

        Args:
            narrative: FusedNarrative from the fusion layer.
            state: Current NarrativeState (mutated in place).

        Returns:
            Updated NarrativeState.
        """
        session_id = narrative.unit_id
        now = datetime.now(timezone.utc)

        # Process accepted claims
        for claim in narrative.accepted_claims:
            self._process_claim(claim, session_id, now, state)

        # Process events for entity tracking
        # (Events from the fusion bundle, if available)

        # Adjust confidence based on verification
        if narrative.verification:
            self._apply_verification_adjustments(
                narrative, state, session_id, now,
            )

        # Handle conflicts
        for conflict in narrative.conflicts:
            self._log_conflict(conflict, session_id, now, state)

        state.sessions_processed.append(session_id)
        state.total_claims_ingested += narrative.total_claims

        return state

    def _process_claim(
        self,
        claim: Claim,
        session_id: str,
        now: datetime,
        state: NarrativeState,
    ) -> None:
        """Process a single claim for entity tracking."""
        # Extract entity references from claim
        subject = claim.canonical.subject
        self._track_entity(subject, session_id, now, state, claim.confidence)

    def _track_entity(
        self,
        entity_ref: EntityRef,
        session_id: str,
        now: datetime,
        state: NarrativeState,
        confidence: float,
    ) -> TrackedEntity:
        """Track an entity, creating or updating as needed."""
        # Try to find existing entity by name
        existing = self._lookup_by_name(
            entity_ref.label or entity_ref.entity_id, state,
        )

        if existing:
            # Update existing entity
            existing.last_seen_session = session_id
            existing.appearance_count += 1
            # Weighted confidence update
            existing.confidence = min(
                EPISTEMIC_CEILING,
                existing.confidence * 0.9 + confidence * 0.1,
            )

            state.deltas.append(StateDelta(
                delta_id=new_id("dlt_"),
                session_id=session_id,
                timestamp=now,
                action="entity_updated",
                target_sid=existing.entity_sid,
                details={
                    "appearance_count": existing.appearance_count,
                    "confidence": round(existing.confidence, 4),
                },
            ))
            return existing

        # Create new entity
        entity_type = self._map_entity_type(entity_ref.entity_type)
        sid = state.allocate_sid(entity_type)

        tracked = TrackedEntity(
            entity_sid=sid,
            entity_type=entity_type,
            canonical_name=entity_ref.label or entity_ref.entity_id,
            aliases=[entity_ref.entity_id] if entity_ref.entity_id != (entity_ref.label or "") else [],
            first_seen_session=session_id,
            last_seen_session=session_id,
            appearance_count=1,
            confidence=confidence,
        )
        state.entities[sid] = tracked

        state.deltas.append(StateDelta(
            delta_id=new_id("dlt_"),
            session_id=session_id,
            timestamp=now,
            action="entity_created",
            target_sid=sid,
            details={
                "canonical_name": tracked.canonical_name,
                "entity_type": entity_type,
                "confidence": round(confidence, 4),
            },
        ))

        return tracked

    def _lookup_by_name(
        self,
        name: str,
        state: NarrativeState,
    ) -> TrackedEntity | None:
        """Find an existing entity by canonical name or alias."""
        name_lower = name.lower().strip()
        if not name_lower:
            return None

        for entity in state.entities.values():
            # Exact match on canonical name
            if entity.canonical_name.lower().strip() == name_lower:
                return entity
            # Alias match
            for alias in entity.aliases:
                if alias.lower().strip() == name_lower:
                    return entity
            # Jaccard similarity for fuzzy matching
            if self._jaccard_similarity(
                entity.canonical_name.lower(), name_lower,
            ) >= DEDUP_THRESHOLD:
                return entity

        return None

    def _jaccard_similarity(self, a: str, b: str) -> float:
        """Compute Jaccard similarity between two strings (word-level)."""
        words_a = set(a.split())
        words_b = set(b.split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    def _apply_verification_adjustments(
        self,
        narrative: FusedNarrative,
        state: NarrativeState,
        session_id: str,
        now: datetime,
    ) -> None:
        """Adjust entity confidence based on verification results."""
        if not narrative.verification:
            return

        bonus = (
            VERIFIED_BONUS if narrative.verification.passed
            else PASS_WITH_CORRECTIONS_BONUS if narrative.verification.claims_corrected > 0
            else FAILED_PENALTY
        )

        # Apply to all entities seen in this session
        for entity in state.entities.values():
            if entity.last_seen_session == session_id:
                old_confidence = entity.confidence
                entity.confidence = max(
                    0.0,
                    min(EPISTEMIC_CEILING, entity.confidence + bonus),
                )
                if abs(old_confidence - entity.confidence) > 0.001:
                    state.deltas.append(StateDelta(
                        delta_id=new_id("dlt_"),
                        session_id=session_id,
                        timestamp=now,
                        action="confidence_adjusted",
                        target_sid=entity.entity_sid,
                        details={
                            "old": round(old_confidence, 4),
                            "new": round(entity.confidence, 4),
                            "reason": "verification_" + (
                                "passed" if narrative.verification.passed
                                else "corrected" if narrative.verification.claims_corrected > 0
                                else "failed"
                            ),
                        },
                    ))

    def _log_conflict(
        self,
        conflict: Any,
        session_id: str,
        now: datetime,
        state: NarrativeState,
    ) -> None:
        """Log a conflict as a state delta (non-destructive)."""
        state.deltas.append(StateDelta(
            delta_id=new_id("dlt_"),
            session_id=session_id,
            timestamp=now,
            action="conflict_forked",
            target_sid=conflict.conflict_id,
            details={
                "claim_a": conflict.claim_a_id,
                "claim_b": conflict.claim_b_id,
                "resolution": conflict.resolution,
                "description": conflict.description[:200],
            },
        ))

    def _map_entity_type(self, raw_type: str) -> str:
        """Map raw entity types to tracked entity types."""
        mapping = {
            "person": "developer",
            "user": "developer",
            "repo": "module",
            "project": "module",
            "file": "file",
            "model": "agent",
            "agent": "agent",
            "service": "agent",
            "tool": "agent",
            "session": "concept",
        }
        return mapping.get(raw_type, "concept")
