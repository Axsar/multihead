"""Real-time knowledge extraction hooks for orchestrator.

Captures learnings during step execution and writes them to knowledge DB inline,
making knowledge available immediately for subsequent steps in the same run.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multihead.knowledge_models import (
    Claim,
    ClaimCanonical,
    ClaimScope,
    ClaimStatus,
    ClaimType,
    EntityRef,
    EvidencePointer,
    Provenance,
    Record,
    ScopeType,
    Stability,
    ValueObject,
)
from multihead.knowledge_store import KnowledgeStore
from multihead.models import StageResult, StepDef, StepStatus

logger = logging.getLogger(__name__)


class KnowledgeHook:
    """Hook for extracting and persisting knowledge during step execution.

    Integrates with orchestrator to capture learnings in real-time:
    - What worked (successful patterns)
    - What failed (errors and fixes)
    - Performance insights (latency, token usage)
    - Quality indicators (warnings, validation)

    All learnings are immediately written to knowledge DB, making them
    available for subsequent steps in the same run.
    """

    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        project_id: str = "multihead",
        *,
        min_confidence: float = 0.7,
        capture_successes: bool = True,
        capture_failures: bool = True,
        capture_performance: bool = True,
    ):
        """Initialize knowledge hook.

        Args:
            knowledge_store: KnowledgeStore instance for persistence
            project_id: Project scope for claims
            min_confidence: Minimum confidence for auto-accepting claims
            capture_successes: Whether to capture successful patterns
            capture_failures: Whether to capture failure patterns
            capture_performance: Whether to capture performance metrics
        """
        self.knowledge_store = knowledge_store
        self.project_id = project_id
        self.min_confidence = min_confidence
        self.capture_successes = capture_successes
        self.capture_failures = capture_failures
        self.capture_performance = capture_performance

        # Track claims created this session
        self.claims_created: list[str] = []
        self.patterns_learned: dict[str, int] = {}

    async def on_step_complete(
        self,
        step: StepDef,
        result: StageResult,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Called when a step completes execution.

        Extracts learnings and writes claims to knowledge DB.

        Args:
            step: Step definition
            result: Step execution result
            context: Execution context with previous steps and run metadata

        Returns:
            Dict with knowledge extraction metadata:
            - claims_created: List of claim IDs
            - patterns_detected: List of pattern types
            - learning_summary: Human-readable summary
        """
        logger.debug("Extracting knowledge from step %s", step.step_id)

        claims_created = []
        patterns_detected = []

        # Create record for this step execution
        record = self._create_execution_record(step, result, context)

        # Extract learnings based on result
        if result.status == StepStatus.COMMITTED:
            if self.capture_successes:
                success_claims = await self._extract_success_patterns(
                    step, result, context, record
                )
                claims_created.extend(success_claims)
                if success_claims:
                    patterns_detected.append("success_pattern")

            if self.capture_performance:
                perf_claims = await self._extract_performance_insights(
                    step, result, context, record
                )
                claims_created.extend(perf_claims)
                if perf_claims:
                    patterns_detected.append("performance_insight")

        elif result.status == StepStatus.FAILED:
            if self.capture_failures:
                failure_claims = await self._extract_failure_patterns(
                    step, result, context, record
                )
                claims_created.extend(failure_claims)
                if failure_claims:
                    patterns_detected.append("failure_pattern")

        # Update tracking
        self.claims_created.extend(claims_created)
        for pattern in patterns_detected:
            self.patterns_learned[pattern] = self.patterns_learned.get(pattern, 0) + 1

        # Generate summary
        summary = self._generate_learning_summary(
            step, result, claims_created, patterns_detected
        )

        logger.info(
            "Knowledge extracted from %s: %d claims, patterns=%s",
            step.step_id,
            len(claims_created),
            patterns_detected,
        )

        return {
            "claims_created": claims_created,
            "patterns_detected": patterns_detected,
            "learning_summary": summary,
        }

    def _create_execution_record(
        self,
        step: StepDef,
        result: StageResult,
        context: dict[str, Any],
    ) -> Record:
        """Create a record for this step execution."""
        now = datetime.now(timezone.utc)

        # Create URI from run_id and step_id
        run_id = context.get("run_id", "unknown")
        uri = f"multihead://runs/{run_id}/steps/{step.step_id}"

        record = Record(
            uri=uri,
            sha256="",  # Could hash result outputs
            mime="application/json",
            captured_at=now,
            created_at=now,
        )

        record = self.knowledge_store.insert_record(record)
        return record

    async def _extract_success_patterns(
        self,
        step: StepDef,
        result: StageResult,
        context: dict[str, Any],
        record: Record,
    ) -> list[str]:
        """Extract patterns from successful step execution.

        Captures:
        - Head-task compatibility (which heads work well for which tasks)
        - Prompt patterns that succeed
        - Output quality indicators
        """
        claims_created = []

        # Pattern: head-task compatibility
        if result.head_id and step.required_kind:
            claim_id = await self._create_claim(
                claim_key=f"head.{result.head_id}.compatible_with.{step.required_kind}",
                subject=EntityRef(
                    entity_type="head",
                    entity_id=result.head_id,
                    label=f"Head {result.head_id}",
                ),
                predicate="successfully_executed",
                object=ValueObject(
                    value_type="string",
                    value=f"{step.required_kind} task: {step.name}",
                ),
                statement=f"Head {result.head_id} successfully executed {step.required_kind} task: {step.name}",
                claim_type=ClaimType.FACT,
                confidence=0.8,
                importance=0.7,
                record=record,
            )
            if claim_id:
                claims_created.append(claim_id)

        # Pattern: output quality
        if result.metrics and result.metrics.get("prm_quality") == "correct":
            claim_id = await self._create_claim(
                claim_key=f"step.{step.step_id}.quality.high",
                subject=EntityRef(
                    entity_type="step",
                    entity_id=step.step_id,
                    label=step.name,
                ),
                predicate="produced_quality",
                object=ValueObject(value_type="string", value="high"),
                statement=f"Step {step.name} produced high-quality output (PRM verified)",
                claim_type=ClaimType.FACT,
                confidence=0.9,
                importance=0.6,
                record=record,
            )
            if claim_id:
                claims_created.append(claim_id)

        return claims_created

    async def _extract_failure_patterns(
        self,
        step: StepDef,
        result: StageResult,
        context: dict[str, Any],
        record: Record,
    ) -> list[str]:
        """Extract patterns from failed step execution.

        Captures:
        - Error types and frequencies
        - Head limitations
        - Prompt issues
        """
        claims_created = []

        if not result.error:
            return claims_created

        # Pattern: head limitation
        if result.head_id:
            # Extract error type from error message
            error_type = self._classify_error(result.error)

            claim_id = await self._create_claim(
                claim_key=f"head.{result.head_id}.error.{error_type}",
                subject=EntityRef(
                    entity_type="head",
                    entity_id=result.head_id,
                    label=f"Head {result.head_id}",
                ),
                predicate="encountered_error",
                object=ValueObject(
                    value_type="string",
                    value=f"{error_type}: {result.error[:100]}",
                ),
                statement=f"Head {result.head_id} encountered {error_type} error: {result.error[:100]}",
                claim_type=ClaimType.FACT,
                confidence=0.9,
                importance=0.8,
                record=record,
            )
            if claim_id:
                claims_created.append(claim_id)

        return claims_created

    async def _extract_performance_insights(
        self,
        step: StepDef,
        result: StageResult,
        context: dict[str, Any],
        record: Record,
    ) -> list[str]:
        """Extract performance insights from step execution.

        Captures:
        - Token efficiency
        - Latency patterns
        - Resource usage
        """
        claims_created = []

        if not result.metrics:
            return claims_created

        # Pattern: token efficiency
        tokens_in = result.metrics.get("tokens_in", 0)
        tokens_out = result.metrics.get("tokens_out", 0)

        if tokens_in > 0 and tokens_out > 0:
            ratio = tokens_out / tokens_in

            if ratio < 0.5:  # Very concise output
                claim_id = await self._create_claim(
                    claim_key=f"step.{step.step_id}.token_efficiency.high",
                    subject=EntityRef(
                        entity_type="step",
                        entity_id=step.step_id,
                        label=step.name,
                    ),
                    predicate="has_token_efficiency",
                    object=ValueObject(value_type="number", value=ratio),
                    statement=f"Step {step.name} has high token efficiency (ratio={ratio:.2f})",
                    claim_type=ClaimType.FACT,
                    confidence=0.7,
                    importance=0.5,
                    record=record,
                )
                if claim_id:
                    claims_created.append(claim_id)

        return claims_created

    async def _create_claim(
        self,
        claim_key: str,
        subject: EntityRef,
        predicate: str,
        object: ValueObject,
        statement: str,
        claim_type: ClaimType,
        confidence: float,
        importance: float,
        record: Record,
    ) -> str | None:
        """Create and persist a claim to knowledge DB.

        Args:
            claim_key: Unique key for claim
            subject: Subject entity
            predicate: Relationship
            object: Object value
            statement: Human-readable statement
            claim_type: Type of claim
            confidence: Confidence score (0-1)
            importance: Importance score (0-1)
            record: Evidence record

        Returns:
            Claim ID if created and accepted, None otherwise
        """
        try:
            now = datetime.now(timezone.utc)

            # Create claim
            claim = Claim(
                claim_status=ClaimStatus.PROPOSED,
                claim_type=claim_type,
                scope=ClaimScope(
                    scope_type=ScopeType.PROJECT,
                    scope_id=self.project_id,
                    visibility="private",
                    valid_from=now,
                ),
                canonical=ClaimCanonical(
                    claim_key=claim_key,
                    subject=subject,
                    predicate=predicate,
                    object=object,
                ),
                statement=statement,
                rationale="Extracted from real-time step execution",
                confidence=confidence,
                stability=Stability.MEDIUM,  # Needs more observations to be stable
                importance=importance,
                provenance=Provenance(
                    produced_by={"kind": "hook", "id": "knowledge_hook"},
                    toolchain=[{"tool": "KnowledgeHook", "version": "1.0"}],
                ),
            )

            # Insert claim
            claim = self.knowledge_store.insert_claim(claim)

            # Create evidence pointer
            evidence = EvidencePointer(
                record_id=record.record_id,
                uri=record.uri,
                quote=statement[:200],
                captured_at=now,
            )
            evidence = self.knowledge_store.insert_evidence(evidence)

            # Link evidence
            self.knowledge_store.link_claim_evidence(
                claim.claim_id, evidence.evidence_id, stance="supports"
            )

            # Auto-accept if confidence is high enough
            if confidence >= self.min_confidence:
                self.knowledge_store.accept_claim(claim.claim_id)

            logger.debug(
                "Created claim %s: %s (confidence=%.2f)",
                claim.claim_id,
                statement[:60],
                confidence,
            )

            return claim.claim_id

        except Exception as e:
            logger.warning("Failed to create claim: %s", e)
            return None

    def _classify_error(self, error_message: str) -> str:
        """Classify error type from error message."""
        error_lower = error_message.lower()

        if "timeout" in error_lower or "timed out" in error_lower:
            return "timeout"
        elif "out of memory" in error_lower or "oom" in error_lower:
            return "out_of_memory"
        elif "syntax" in error_lower or "parse" in error_lower:
            return "syntax_error"
        elif "context length" in error_lower or "token limit" in error_lower:
            return "context_length"
        elif "connection" in error_lower or "network" in error_lower:
            return "network"
        else:
            return "unknown"

    def _generate_learning_summary(
        self,
        step: StepDef,
        result: StageResult,
        claims_created: list[str],
        patterns_detected: list[str],
    ) -> str:
        """Generate human-readable summary of learnings."""
        if not claims_created:
            return f"No new learnings extracted from {step.name}"

        summary_parts = [f"Extracted {len(claims_created)} learnings from {step.name}:"]

        if "success_pattern" in patterns_detected:
            summary_parts.append(f"  ✓ Success: {result.head_id} works well for {step.required_kind} tasks")

        if "failure_pattern" in patterns_detected:
            error_type = self._classify_error(result.error or "")
            summary_parts.append(f"  ✗ Failure: {error_type} error encountered")

        if "performance_insight" in patterns_detected:
            summary_parts.append(f"  ⚡ Performance: efficient token usage detected")

        return "\n".join(summary_parts)

    def get_session_summary(self) -> dict[str, Any]:
        """Get summary of all knowledge extracted this session.

        Returns:
            Dict with:
            - total_claims: Number of claims created
            - patterns_learned: Dict of pattern types and counts
            - top_patterns: Most common patterns
        """
        top_patterns = sorted(
            self.patterns_learned.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        return {
            "total_claims": len(self.claims_created),
            "patterns_learned": dict(self.patterns_learned),
            "top_patterns": top_patterns,
            "claim_ids": self.claims_created,
        }
