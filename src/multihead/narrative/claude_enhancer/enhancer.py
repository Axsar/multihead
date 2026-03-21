"""ClaudeEnhancer: main orchestrator for Claude-enhanced document extraction.

Phase 1 (heuristic) extracts bullet points from markdown structure.
Phase 2 (Claude) sends each H2 section to a Claude worker daemon for
deep semantic extraction -- identifying implicit claims, relationships,
dependencies, and risks that heuristic parsing misses.

Usage:
    enhancer = ClaudeEnhancer(acp_url="http://localhost:8000/api/v1", api_key="...")
    claims = await enhancer.enhance_document(Path("PLAN.md"), doc_type="plan")

Architecture:
    1. Split markdown into H2-level sections
    2. Create one ACP task per section (parallel submission)
    3. Poll all tasks for completion
    4. Parse structured JSON results into Claim objects
    5. Merge with heuristic claims (dedup by claim_key)
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multihead.knowledge_models import (
    Claim,
    ClaimScope,
    EntityRef,
    EvidencePointer,
    EventType,
    KnowledgeEvent,
    Record,
    ScopeType,
    TimeBlock,
    TimePrecision,
)
from multihead.narrative.confidence import ConfidenceCalibrator

from .client import ACPClient
from .constants import (
    _DOC_STABILITY,
    _PROVENANCE,
    _SECTION_PROMPT_TEMPLATE,
    _SYNTHESIS_PROMPT_TEMPLATE,
    Stability,
    logger,
)
from .parsing import (
    convert_to_claims,
    merge_claims,
    parse_claude_output,
    split_sections,
)


class ClaudeEnhancer:
    """Enhance markdown extraction with Claude Code worker daemons.

    Creates one ACP task per document section, polls for results,
    and converts Claude's structured output into Claim objects.
    """

    def __init__(
        self,
        acp_url: str,
        api_key: str,
        project_id: str = "multihead",
        acp_project_id: str | None = None,
        poll_interval: float = 10.0,
        max_wait: float = 600.0,
        max_concurrent: int = 5,
    ):
        self.acp_url = acp_url.rstrip("/")
        self.api_key = api_key
        self.project_id = project_id
        # ACP project UUID (different from knowledge store project_id)
        self.acp_project_id = acp_project_id or os.environ.get(
            "ACP_PROJECT_ID", ""
        )
        self.poll_interval = poll_interval
        self.max_wait = max_wait
        self.max_concurrent = max_concurrent
        self.calibrator = ConfidenceCalibrator()
        self._client = ACPClient(
            acp_url=self.acp_url,
            api_key=self.api_key,
            acp_project_id=self.acp_project_id,
            poll_interval=self.poll_interval,
            max_wait=self.max_wait,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def enhance_document(
        self,
        doc_path: Path,
        doc_type: str = "plan",
        source_project: str | None = None,
        heuristic_claims: list[Claim] | None = None,
        synthesize: bool = True,
    ) -> list[dict[str, Any]]:
        """Send document sections to Claude for deep extraction.

        Args:
            doc_path: Path to markdown file.
            doc_type: Document type (plan, status, fixes, etc.).
            source_project: Override project scope ID.
            heuristic_claims: Claims from Phase 1 (MarkdownExtractor).
            synthesize: Run a synthesis pass for cross-section claims.

        Returns:
            List of artifact dicts with {record, event, claims, evidence}.
        """
        if not doc_path.exists():
            logger.warning("Document not found: %s", doc_path)
            return []

        scope_id = source_project or self.project_id
        content = doc_path.read_text(encoding="utf-8", errors="replace")
        doc_id = doc_path.stem.lower().replace(" ", "_")
        doc_name = doc_path.stem

        # Parse into H2 sections
        sections = split_sections(content)
        if not sections:
            logger.info("No sections to enhance in %s", doc_path.name)
            return []

        logger.info(
            "Enhancing %s: %d sections via Claude daemons",
            doc_path.name, len(sections),
        )

        # Phase 2a: Call MultiHead for each section (sequential to avoid GPU contention)
        tasks: list[dict[str, Any]] = []

        for i, section in enumerate(sections):
            prompt = self._build_section_prompt(
                section["heading"], section["text"], doc_type, doc_name,
            )
            try:
                task_id, response = await self._client.create_task(prompt)
                tasks.append({
                    "task_id": task_id,
                    "section": section,
                    "response": response,
                })
                logger.info(
                    "Completed section %d/%d: %s (%d chars)",
                    i + 1, len(sections), section["heading"][:50], len(response),
                )
            except Exception as e:
                logger.error("Section %d failed: %s", i + 1, e)
                continue

        if not tasks:
            logger.warning("No sections completed for %s", doc_path.name)
            return []

        logger.info("Completed %d sections, parsing claims...", len(tasks))

        # Create record FIRST so we have a valid record_id for evidence pointers
        content_bytes = content.encode("utf-8")
        sha = hashlib.sha256(content_bytes).hexdigest()
        record = Record(
            uri=f"markdown+claude://{scope_id}/{doc_path.name}",
            sha256=sha,
            mime="text/markdown",
        )

        # Phase 2b: Parse results into claims (no polling needed - we have responses)
        all_claude_claims: list[Claim] = []
        section_summaries: list[str] = []
        stability = _DOC_STABILITY.get(doc_type, Stability.MEDIUM)
        scope = ClaimScope(scope_type=ScopeType.PROJECT, scope_id=scope_id)

        for task_info in tasks:
            task_id = task_info["task_id"]
            section = task_info["section"]
            output = task_info.get("response", "")

            if not output:
                logger.warning(
                    "No response for section '%s'",
                    section["heading"],
                )
                continue

            parsed = parse_claude_output(output)
            section_claims = convert_to_claims(
                parsed, doc_id, scope_id, scope, stability, section["heading"],
                calibrator=self.calibrator,
                record_id=record.record_id,
            )
            all_claude_claims.extend(section_claims)

            # Build summary for synthesis
            if parsed:
                summary = f"### {section['heading']}\n"
                for c in parsed[:5]:  # Cap per section
                    summary += f"- [{c.get('claim_type', '?')}] {c.get('text', '')[:100]}\n"
                section_summaries.append(summary)

        logger.info(
            "Extracted %d Claude claims from %d sections",
            len(all_claude_claims), len(tasks),
        )

        # Phase 2d: Synthesis pass (cross-section claims)
        if synthesize and section_summaries:
            synthesis_claims = await self._run_synthesis(
                doc_name, doc_type, scope_id, scope, stability,
                doc_id, "\n".join(section_summaries), record.record_id,
            )
            all_claude_claims.extend(synthesis_claims)
            logger.info("Synthesis added %d cross-section claims", len(synthesis_claims))

        # Merge with heuristic claims (dedup by claim_key)
        merged = merge_claims(heuristic_claims or [], all_claude_claims)

        # Build evidence pointers (record was already created earlier)
        all_evidence: list[EvidencePointer] = []
        for claim in merged:
            for ep in claim.evidence_supports:
                if ep not in all_evidence:
                    all_evidence.append(ep)

        # Build event
        event = KnowledgeEvent(
            event_type=EventType.SPEC_CHANGE,
            title=f"Claude-enhanced ingestion of {doc_path.name} ({doc_type})",
            summary=(
                f"Phase 1 heuristic: {len(heuristic_claims or [])} claims. "
                f"Phase 2 Claude: {len(all_claude_claims)} claims. "
                f"Merged total: {len(merged)} unique claims."
            ),
            time=TimeBlock(
                happened_at=datetime.now(timezone.utc),
                time_precision=TimePrecision.SECOND,
            ),
            entities=[
                EntityRef(
                    entity_type="document",
                    entity_id=doc_id,
                    label=doc_name,
                )
            ],
            tags=[doc_type, "markdown", "claude-enhanced", scope_id],
            evidence_supports=all_evidence[:5],
            provenance=_PROVENANCE,
        )

        return [
            {
                "record": record,
                "event": event,
                "claims": merged,
                "evidence": all_evidence,
            }
        ]

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_section_prompt(
        heading: str,
        section_text: str,
        doc_type: str,
        doc_name: str,
    ) -> str:
        """Build the extraction prompt for one section."""
        return _SECTION_PROMPT_TEMPLATE.format(
            doc_type=doc_type,
            doc_name=doc_name,
            section_heading=heading,
            section_text=section_text[:6000],  # Cap section size
        )

    # ------------------------------------------------------------------
    # Synthesis pass
    # ------------------------------------------------------------------

    async def _run_synthesis(
        self,
        doc_name: str,
        doc_type: str,
        scope_id: str,
        scope: ClaimScope,
        stability: Stability,
        doc_id: str,
        section_summaries: str,
        record_id: str,
    ) -> list[Claim]:
        """Run synthesis pass for cross-section claims."""
        prompt = _SYNTHESIS_PROMPT_TEMPLATE.format(
            doc_type=doc_type,
            doc_name=doc_name,
            section_summaries=section_summaries,
        )

        try:
            task_id, response = await self._client.create_task(prompt)
            if response:
                parsed = parse_claude_output(response)
                return convert_to_claims(
                    parsed, doc_id, scope_id, scope, stability, "synthesis",
                    calibrator=self.calibrator,
                    record_id=record_id,
                )
        except Exception as e:
            logger.warning("Synthesis pass failed: %s", e)

        return []
