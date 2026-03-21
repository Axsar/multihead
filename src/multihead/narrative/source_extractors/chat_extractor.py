"""Extract narrative claims from chat/conversation transcripts.

Handles both interactive Claude sessions and non-interactive agent logs.
Produces Records, Events (decisions, questions), and Claims (agreements, action items).
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multihead.knowledge_models import (
    ActorRef,
    Claim,
    ClaimCanonical,
    ClaimScope,
    ClaimType,
    EntityRef,
    EvidencePointer,
    EventType,
    KnowledgeEvent,
    Provenance,
    Record,
    ScopeType,
    SpanRef,
    Stability,
    TimeBlock,
    TimePrecision,
    ValueObject,
)
from multihead.models import new_id
from multihead.narrative.confidence import ConfidenceCalibrator, SourcePriority

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from multihead.artifact_store import ArtifactStore

logger = logging.getLogger(__name__)

_PROVENANCE = Provenance(
    produced_by={"kind": "extractor", "id": "narrative.chat_extractor"},
    toolchain=[],
)


class ChatExtractor:
    """Extract narrative evidence from chat transcripts (JSONL format)."""

    def __init__(self, project_id: str = "default", artifact_store: ArtifactStore | None = None):
        self.project_id = project_id
        self.artifact_store = artifact_store
        self.calibrator = ConfidenceCalibrator()

    def extract_from_jsonl(
        self,
        transcript_path: Path,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Extract narrative artifacts from a JSONL chat transcript.

        Expected JSONL format (one message per line):
        {"role": "user"|"assistant", "content": "...", "timestamp": "ISO-8601"}

        Returns list of dicts with record, event, claims, evidence.
        """
        if not transcript_path.exists():
            logger.warning("Transcript not found: %s", transcript_path)
            return []

        session_id = session_id or transcript_path.stem
        messages = self._load_messages(transcript_path)

        # Create a Record for the entire transcript
        content_bytes = transcript_path.read_bytes()
        if self.artifact_store:
            ref = self.artifact_store.store(content_bytes, name=transcript_path.name, media_type="application/x-jsonl")
            sha = ref.artifact_id.removeprefix("sha256:")
        else:
            sha = hashlib.sha256(content_bytes).hexdigest()

        transcript_record = Record(
            uri=f"chat://{session_id}/{transcript_path.name}",
            sha256=sha,
            mime="application/x-jsonl",
        )

        artifacts = []

        # Extract decision events and claims from message patterns
        for i, msg in enumerate(messages):
            extracted = self._analyze_message(
                msg, i, messages, transcript_record, session_id,
            )
            if extracted:
                artifacts.extend(extracted)

        return artifacts

    def extract_from_messages(
        self,
        messages: list[dict[str, Any]],
        session_id: str,
        source_uri: str = "",
    ) -> list[dict[str, Any]]:
        """Extract from an in-memory message list (for live sessions)."""
        content_bytes = json.dumps(messages, default=str).encode("utf-8")
        if self.artifact_store:
            ref = self.artifact_store.store(content_bytes, name=f"chat_{session_id}", media_type="application/json")
            sha = ref.artifact_id.removeprefix("sha256:")
        else:
            sha = hashlib.sha256(content_bytes).hexdigest()

        record = Record(
            uri=source_uri or f"chat://{session_id}",
            sha256=sha,
            mime="application/json",
        )

        artifacts = []
        for i, msg in enumerate(messages):
            extracted = self._analyze_message(
                msg, i, messages, record, session_id,
            )
            if extracted:
                artifacts.extend(extracted)

        return artifacts

    def _load_messages(self, path: Path) -> list[dict[str, Any]]:
        """Load messages from JSONL file."""
        messages = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                messages.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return messages

    def _analyze_message(
        self,
        msg: dict[str, Any],
        index: int,
        all_messages: list[dict[str, Any]],
        record: Record,
        session_id: str,
    ) -> list[dict[str, Any]]:
        """Analyze a single message for narrative-worthy content."""
        content = msg.get("content", "")
        role = msg.get("role", "unknown")
        timestamp_str = msg.get("timestamp", "")

        if not content or len(content) < 10:
            return []

        try:
            timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now(timezone.utc)
        except ValueError:
            timestamp = datetime.now(timezone.utc)

        results = []

        # Create evidence pointer for this message
        evidence = EvidencePointer(
            record_id=record.record_id,
            uri=record.uri,
            span=SpanRef(start=index, end=index, unit="chars"),
            quote=content[:200],
        )

        # Detect decisions
        if self._is_decision(content, role):
            results.append(self._make_decision_artifact(
                content, role, timestamp, evidence, session_id, record,
            ))

        # Detect action items / tasks
        if self._is_action_item(content, role):
            results.append(self._make_task_artifact(
                content, role, timestamp, evidence, session_id, record,
            ))

        # Detect architecture/design agreements
        if self._is_architecture_claim(content, role):
            results.append(self._make_architecture_artifact(
                content, role, timestamp, evidence, session_id, record,
            ))

        return results

    def _is_decision(self, content: str, role: str) -> bool:
        """Detect decision-making language."""
        lower = content.lower()
        markers = [
            "let's go with", "we decided", "the decision is",
            "agreed to", "we'll use", "switching to",
            "going with", "chosen approach", "final answer",
            "yes, do that", "approved", "let's do",
        ]
        return any(m in lower for m in markers)

    def _is_action_item(self, content: str, role: str) -> bool:
        """Detect action items and tasks."""
        lower = content.lower()
        markers = [
            "todo:", "action item:", "need to",
            "should we", "next step", "let's plan",
            "can you", "please implement", "git commit",
        ]
        return any(m in lower for m in markers)

    def _is_architecture_claim(self, content: str, role: str) -> bool:
        """Detect architecture/design statements."""
        lower = content.lower()
        markers = [
            "architecture", "design pattern", "schema",
            "the pipeline", "data flow", "module structure",
            "api design", "database schema", "interface",
        ]
        return any(m in lower for m in markers)

    def _make_decision_artifact(
        self, content: str, role: str, timestamp: datetime,
        evidence: EvidencePointer, session_id: str, record: Record,
    ) -> dict[str, Any]:
        """Create a decision event + claim."""
        cal = self.calibrator.calibrate(0.85, SourcePriority.CHAT_MESSAGE_AUTHOR)

        event = KnowledgeEvent(
            event_type=EventType.DECISION,
            title=f"Decision in session {session_id[:8]}",
            summary=content[:300],
            time=TimeBlock(happened_at=timestamp, time_precision=TimePrecision.SECOND),
            actors=[ActorRef(actor_type="agent" if role == "assistant" else "user", actor_id=role)],
            evidence_supports=[evidence],
            provenance=_PROVENANCE,
        )

        claim = Claim(
            claim_type=ClaimType.DECISION,
            scope=ClaimScope(scope_type=ScopeType.SESSION, scope_id=session_id),
            canonical=ClaimCanonical(
                claim_key=f"session.{session_id}.decision.{evidence.evidence_id}",
                subject=EntityRef(entity_type="session", entity_id=session_id),
                predicate="decided",
                object=ValueObject(value_type="string", value=content[:500]),
            ),
            statement=content[:300],
            confidence=cal.calibrated,
            stability=Stability.MEDIUM,
            evidence_supports=[evidence],
            provenance=_PROVENANCE,
        )

        return {"record": record, "event": event, "claims": [claim], "evidence": [evidence]}

    def _make_task_artifact(
        self, content: str, role: str, timestamp: datetime,
        evidence: EvidencePointer, session_id: str, record: Record,
    ) -> dict[str, Any]:
        """Create a task event + claim."""
        cal = self.calibrator.calibrate(0.80, SourcePriority.CHAT_MESSAGE_AUTHOR)

        event = KnowledgeEvent(
            event_type=EventType.TASK_CREATED,
            title=f"Task from session {session_id[:8]}",
            summary=content[:300],
            time=TimeBlock(happened_at=timestamp, time_precision=TimePrecision.SECOND),
            actors=[ActorRef(actor_type="agent" if role == "assistant" else "user", actor_id=role)],
            evidence_supports=[evidence],
            provenance=_PROVENANCE,
        )

        claim = Claim(
            claim_type=ClaimType.PLAN,
            scope=ClaimScope(scope_type=ScopeType.SESSION, scope_id=session_id),
            canonical=ClaimCanonical(
                claim_key=f"session.{session_id}.task.{evidence.evidence_id}",
                subject=EntityRef(entity_type="session", entity_id=session_id),
                predicate="planned",
                object=ValueObject(value_type="string", value=content[:500]),
            ),
            statement=content[:300],
            confidence=cal.calibrated,
            stability=Stability.VOLATILE,
            evidence_supports=[evidence],
            provenance=_PROVENANCE,
        )

        return {"record": record, "event": event, "claims": [claim], "evidence": [evidence]}

    def _make_architecture_artifact(
        self, content: str, role: str, timestamp: datetime,
        evidence: EvidencePointer, session_id: str, record: Record,
    ) -> dict[str, Any]:
        """Create an architecture claim."""
        cal = self.calibrator.calibrate(0.75, SourcePriority.CHAT_MESSAGE_AUTHOR)

        claim = Claim(
            claim_type=ClaimType.DEFINITION,
            scope=ClaimScope(scope_type=ScopeType.PROJECT, scope_id=self.project_id),
            canonical=ClaimCanonical(
                claim_key=f"arch.{self.project_id}.{evidence.evidence_id}",
                subject=EntityRef(entity_type="project", entity_id=self.project_id),
                predicate="architecture_decision",
                object=ValueObject(value_type="string", value=content[:500]),
            ),
            statement=content[:300],
            confidence=cal.calibrated,
            stability=Stability.MEDIUM,
            evidence_supports=[evidence],
            provenance=_PROVENANCE,
        )

        return {"record": record, "event": None, "claims": [claim], "evidence": [evidence]}
