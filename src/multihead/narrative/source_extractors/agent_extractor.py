"""Extract narrative claims from agent task results.

Handles outputs from spawned agents (Claude subagents, Qwen sessions,
headless daemon runs). Tracks what was attempted, what succeeded/failed.
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
    produced_by={"kind": "extractor", "id": "narrative.agent_extractor"},
    toolchain=[],
)


class AgentExtractor:
    """Extract narrative evidence from agent/LLM session results."""

    def __init__(self, project_id: str = "default", artifact_store: ArtifactStore | None = None):
        self.project_id = project_id
        self.artifact_store = artifact_store
        self.calibrator = ConfidenceCalibrator()

    def extract_from_result(
        self,
        result: dict[str, Any],
        agent_id: str,
        agent_type: str = "claude",
        parent_session_id: str | None = None,
    ) -> dict[str, Any]:
        """Extract narrative artifacts from a single agent result.

        Args:
            result: Agent output dict. Expected keys:
                - task: str (what was asked)
                - output: str (what was produced)
                - status: str (success/failure/partial)
                - started_at: ISO-8601 (optional)
                - completed_at: ISO-8601 (optional)
                - model: str (optional, e.g. "Qwen3-VL-8B")
                - tokens_used: int (optional)
            agent_id: Identifier for the agent instance.
            agent_type: "claude", "qwen", "daemon", etc.
            parent_session_id: ID of the session that spawned this agent.

        Returns:
            Dict with record, event, claims, evidence.
        """
        task = result.get("task", "unknown task")
        output = result.get("output", "")
        status = result.get("status", "unknown")
        model = result.get("model", agent_type)

        # Record
        content = json.dumps(result, default=str)
        content_bytes = content.encode("utf-8")
        if self.artifact_store:
            ref = self.artifact_store.store(content_bytes, name=f"agent_{agent_id[:12]}", media_type="application/json")
            sha = ref.artifact_id.removeprefix("sha256:")
        else:
            sha = hashlib.sha256(content_bytes).hexdigest()

        record = Record(
            uri=f"agent://{agent_id}/{new_id()}",
            sha256=sha,
            mime="application/json",
        )

        # Evidence
        evidence = EvidencePointer(
            record_id=record.record_id,
            uri=record.uri,
            quote=f"Task: {task[:100]} | Status: {status}",
        )

        # Timestamps
        started = self._parse_time(result.get("started_at"))
        completed = self._parse_time(result.get("completed_at"))

        # Event
        event_type = (
            EventType.TASK_COMPLETED if status == "success"
            else EventType.INCIDENT if status == "failure"
            else EventType.TOOL_RUN
        )

        event = KnowledgeEvent(
            event_type=event_type,
            title=f"Agent {agent_id[:12]}: {task[:60]}",
            summary=f"[{status}] {output[:300]}" if output else f"[{status}] {task[:300]}",
            time=TimeBlock(
                happened_at=completed or started or datetime.now(timezone.utc),
                ended_at=completed,
                time_precision=TimePrecision.SECOND,
            ),
            actors=[
                ActorRef(actor_type="agent", actor_id=agent_id, display=f"{agent_type}/{model}"),
            ],
            entities=[
                EntityRef(entity_type="model", entity_id=model, label=model),
            ],
            tags=[agent_type, status, "agent-result"],
            evidence_supports=[evidence],
            metrics=self._extract_metrics(result),
            provenance=_PROVENANCE,
        )

        # Link to parent session
        if parent_session_id:
            event.caused_by_event_ids.append(parent_session_id)

        # Claims
        claims = self._extract_claims(
            task, output, status, agent_id, agent_type, model,
            record, evidence,
        )

        return {
            "record": record,
            "event": event,
            "claims": claims,
            "evidence": [evidence],
        }

    def extract_from_run_events(
        self,
        events_path: Path,
        agent_id: str,
    ) -> list[dict[str, Any]]:
        """Extract from a MultiHead run's events.jsonl file."""
        if not events_path.exists():
            return []

        artifacts = []
        for line in events_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                event_data = json.loads(line)
            except json.JSONDecodeError:
                continue

            kind = event_data.get("kind", "")
            if kind in ("STEP_COMMITTED", "STEP_FAILED", "RUN_DONE", "RUN_FAILED"):
                artifact = self.extract_from_result(
                    result={
                        "task": event_data.get("step_id", "unknown"),
                        "output": json.dumps(event_data.get("payload", {})),
                        "status": "success" if "COMMITTED" in kind or "DONE" in kind else "failure",
                        "started_at": event_data.get("ts"),
                        "completed_at": event_data.get("ts"),
                    },
                    agent_id=agent_id,
                    agent_type="multihead",
                )
                artifacts.append(artifact)

        return artifacts

    def _extract_claims(
        self,
        task: str,
        output: str,
        status: str,
        agent_id: str,
        agent_type: str,
        model: str,
        record: Record,
        evidence: EvidencePointer,
    ) -> list[Claim]:
        """Extract claims from agent result."""
        claims = []
        scope = ClaimScope(
            scope_type=ScopeType.PROJECT,
            scope_id=self.project_id,
        )

        # Determine source priority based on agent type
        if agent_type in ("claude", "gpt"):
            source = SourcePriority.LLM_INFERENCE
        elif agent_type in ("qwen-vlm", "vlm"):
            source = SourcePriority.VLM_INFERENCE
        else:
            source = SourcePriority.AGENT_INTERPRETATION

        cal = self.calibrator.calibrate(
            0.70 if status == "success" else 0.40,
            source,
        )

        # Task completion claim
        claims.append(Claim(
            claim_type=ClaimType.FACT,
            scope=scope,
            canonical=ClaimCanonical(
                claim_key=f"agent.{agent_id}.{record.record_id}.result",
                subject=EntityRef(entity_type="agent", entity_id=agent_id, label=agent_type),
                predicate="completed" if status == "success" else "attempted",
                object=ValueObject(value_type="string", value=task[:500]),
            ),
            statement=f"Agent {agent_id[:12]} ({model}) {'completed' if status == 'success' else 'failed'}: {task[:200]}",
            confidence=cal.calibrated,
            stability=Stability.STABLE,
            evidence_supports=[evidence],
            provenance=_PROVENANCE,
        ))

        # If there's meaningful output, create a finding claim
        if output and len(output) > 20 and status == "success":
            cal_output = self.calibrator.calibrate(0.65, source)
            claims.append(Claim(
                claim_type=ClaimType.FACT,
                scope=scope,
                canonical=ClaimCanonical(
                    claim_key=f"agent.{agent_id}.{record.record_id}.finding",
                    subject=EntityRef(entity_type="agent", entity_id=agent_id),
                    predicate="found",
                    object=ValueObject(value_type="string", value=output[:500]),
                ),
                statement=f"Finding: {output[:300]}",
                confidence=cal_output.calibrated,
                stability=Stability.VOLATILE,
                evidence_supports=[evidence],
                provenance=_PROVENANCE,
            ))

        return claims

    def _extract_metrics(self, result: dict[str, Any]) -> dict[str, float]:
        """Extract numeric metrics from result."""
        metrics: dict[str, float] = {}
        if "tokens_used" in result:
            metrics["tokens_used"] = float(result["tokens_used"])
        if "duration_ms" in result:
            metrics["duration_ms"] = float(result["duration_ms"])
        return metrics

    def _parse_time(self, val: Any) -> datetime | None:
        """Safely parse a timestamp."""
        if not val:
            return None
        if isinstance(val, datetime):
            return val
        try:
            return datetime.fromisoformat(str(val))
        except (ValueError, TypeError):
            return None
