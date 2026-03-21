"""Extract narrative claims from Markdown planning/status documents.

Parses markdown structure (headings, bullets, checkboxes) into Records,
KnowledgeEvents, and Claims. Heuristic-based like GitExtractor and
ChatExtractor — no LLM required for basic extraction.

Usage:
    extractor = MarkdownExtractor(project_id="myproject")
    artifacts = extractor.extract_from_file(
        Path("EXPANSION_PLAN.md"), doc_type="plan",
    )
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multihead.knowledge_models import (
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
    produced_by={"kind": "extractor", "id": "narrative.markdown_extractor"},
    toolchain=[{"name": "markdown", "version": "heuristic"}],
)

# Map doc_type to default claim type
_DOC_TYPE_MAP: dict[str, ClaimType] = {
    "plan": ClaimType.PLAN,
    "status": ClaimType.FACT,
    "recovery": ClaimType.PLAN,
    "fixes": ClaimType.PLAN,
    "decision": ClaimType.DECISION,
    "constraint": ClaimType.CONSTRAINT,
}

# Map doc_type to stability
_DOC_STABILITY: dict[str, Stability] = {
    "plan": Stability.MEDIUM,
    "status": Stability.VOLATILE,
    "recovery": Stability.MEDIUM,
    "fixes": Stability.MEDIUM,
    "decision": Stability.STABLE,
    "constraint": Stability.STABLE,
}


def _slugify(text: str) -> str:
    """Convert heading text to a slug for claim keys."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:60] if slug else "untitled"


class MarkdownExtractor:
    """Extract narrative evidence from Markdown documents."""

    def __init__(
        self,
        project_id: str = "default",
        artifact_store: ArtifactStore | None = None,
    ):
        self.project_id = project_id
        self.artifact_store = artifact_store
        self.calibrator = ConfidenceCalibrator()

    def extract_from_file(
        self,
        doc_path: Path,
        doc_type: str = "plan",
        source_project: str | None = None,
    ) -> list[dict[str, Any]]:
        """Extract claims from a markdown file.

        Args:
            doc_path: Path to the markdown file.
            doc_type: One of "plan", "status", "recovery", "fixes", "decision", "constraint".
            source_project: Project scope ID (defaults to self.project_id).

        Returns:
            List of artifact dicts with {record, event, claims, evidence}.
        """
        if not doc_path.exists():
            logger.warning("Markdown file not found: %s", doc_path)
            return []

        scope_id = source_project or self.project_id
        content = doc_path.read_text(encoding="utf-8", errors="replace")
        doc_id = doc_path.stem.lower().replace(" ", "_")

        # 1. Record (immutable evidence source)
        content_bytes = content.encode("utf-8")
        if self.artifact_store:
            ref = self.artifact_store.store(
                content_bytes,
                name=doc_path.name,
                media_type="text/markdown",
            )
            sha = ref.artifact_id.removeprefix("sha256:")
        else:
            sha = hashlib.sha256(content_bytes).hexdigest()

        record = Record(
            uri=f"markdown://{scope_id}/{doc_path.name}",
            sha256=sha,
            mime="text/markdown",
        )

        # 2. Parse sections
        sections = self._parse_sections(content)
        if not sections:
            logger.info("No sections found in %s", doc_path.name)
            return []

        # 3. Extract claims from sections
        all_claims: list[Claim] = []
        all_evidence: list[EvidencePointer] = []

        default_claim_type = _DOC_TYPE_MAP.get(doc_type, ClaimType.FACT)
        stability = _DOC_STABILITY.get(doc_type, Stability.MEDIUM)
        scope = ClaimScope(scope_type=ScopeType.PROJECT, scope_id=scope_id)

        # Check if any sections have extractable items
        has_items = False
        for section in sections:
            if self._extract_items(section):
                has_items = True
                break
        if not has_items:
            return []

        for section in sections:
            items = self._extract_items(section)
            section_slug = _slugify(section["heading"])

            for item in items:
                # Determine claim type from content and checkbox
                claim_type = self._classify_claim_type(
                    item["text"], item.get("checkbox"), default_claim_type,
                )

                # Determine predicate from checkbox state
                predicate = self._classify_predicate(
                    item["text"], item.get("checkbox"), doc_type,
                )

                # Confidence based on checkbox presence
                raw_conf = 0.90 if item.get("checkbox") is not None else 0.85
                cal = self.calibrator.calibrate(
                    raw_conf, SourcePriority.PLANNING_DOCUMENT,
                )

                # Short hash for unique claim key
                item_hash = hashlib.sha256(
                    item["text"].encode()
                ).hexdigest()[:8]

                evidence = EvidencePointer(
                    record_id=record.record_id,
                    uri=record.uri,
                    sha256=record.sha256,
                    line_start=item["line"],
                    line_end=item["line"],
                    quote=item["text"][:200],
                )

                claim = Claim(
                    claim_type=claim_type,
                    scope=scope,
                    canonical=ClaimCanonical(
                        claim_key=f"doc.{scope_id}.{doc_id}.{section_slug}.{item_hash}",
                        subject=EntityRef(
                            entity_type="document",
                            entity_id=doc_id,
                            label=doc_path.stem,
                        ),
                        predicate=predicate,
                        object=ValueObject(
                            value_type="string",
                            value=item["text"][:500],
                        ),
                    ),
                    statement=item["text"][:300],
                    confidence=cal.calibrated,
                    stability=stability,
                    evidence_supports=[evidence],
                    provenance=_PROVENANCE,
                )

                all_claims.append(claim)
                all_evidence.append(evidence)

        # 4. Document-level event
        event = KnowledgeEvent(
            event_type=EventType.SPEC_CHANGE,
            title=f"Ingested {doc_path.name} ({doc_type})",
            summary=(
                f"Extracted {len(all_claims)} claims from "
                f"{len(sections)} sections in {doc_path.name}"
            ),
            time=TimeBlock(
                happened_at=datetime.now(timezone.utc),
                time_precision=TimePrecision.SECOND,
            ),
            entities=[
                EntityRef(
                    entity_type="document",
                    entity_id=doc_id,
                    label=doc_path.stem,
                )
            ],
            tags=[doc_type, "markdown", "ingestion", scope_id],
            evidence_supports=all_evidence[:5],  # Link first few
            provenance=_PROVENANCE,
        )

        return [
            {
                "record": record,
                "event": event,
                "claims": all_claims,
                "evidence": all_evidence,
            }
        ]

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_sections(self, text: str) -> list[dict[str, Any]]:
        """Parse markdown into sections by heading."""
        sections: list[dict[str, Any]] = []
        current: dict[str, Any] = {
            "heading": "preamble",
            "level": 0,
            "lines": [],
            "start_line": 0,
        }

        for i, line in enumerate(text.split("\n")):
            if line.startswith("#"):
                if current["lines"]:
                    sections.append(current)
                level = len(line) - len(line.lstrip("#"))
                heading = line.lstrip("# ").strip()
                current = {
                    "heading": heading,
                    "level": level,
                    "lines": [],
                    "start_line": i,
                }
            else:
                current["lines"].append((i, line))

        if current["lines"]:
            sections.append(current)

        return sections

    def _extract_items(self, section: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract actionable items from a section's lines."""
        items: list[dict[str, Any]] = []

        for line_num, line in section["lines"]:
            stripped = line.strip()

            # Checked checkbox: - [x] or - [X]
            if re.match(r"^-\s*\[[xX]\]\s+", stripped):
                text = re.sub(r"^-\s*\[[xX]\]\s+", "", stripped).strip()
                if text and len(text) > 5:
                    items.append({
                        "text": text,
                        "checkbox": "checked",
                        "line": line_num,
                    })
            # Unchecked checkbox: - [ ]
            elif re.match(r"^-\s*\[\s\]\s+", stripped):
                text = re.sub(r"^-\s*\[\s\]\s+", "", stripped).strip()
                if text and len(text) > 5:
                    items.append({
                        "text": text,
                        "checkbox": "unchecked",
                        "line": line_num,
                    })
            # Regular bullet: - item or * item
            elif re.match(r"^[-*]\s+", stripped):
                text = re.sub(r"^[-*]\s+", "", stripped).strip()
                if text and len(text) > 10:  # Skip very short bullets
                    items.append({
                        "text": text,
                        "checkbox": None,
                        "line": line_num,
                    })
            # Numbered list: 1. item, 1) item
            elif re.match(r"^\d+[.)]\s+", stripped):
                text = re.sub(r"^\d+[.)]\s+", "", stripped).strip()
                if text and len(text) > 10:
                    items.append({
                        "text": text,
                        "checkbox": None,
                        "line": line_num,
                    })

        return items

    def _classify_claim_type(
        self,
        text: str,
        checkbox: str | None,
        default: ClaimType,
    ) -> ClaimType:
        """Classify claim type from content."""
        lower = text.lower()

        # Checked items are facts (completed)
        if checkbox == "checked":
            return ClaimType.FACT

        # Keywords indicating decisions
        if any(w in lower for w in ("decided", "chose", "selected", "architecture")):
            return ClaimType.DECISION

        # Keywords indicating constraints
        if any(w in lower for w in ("must", "cannot", "constraint", "requirement", "mandatory")):
            return ClaimType.CONSTRAINT

        # Keywords indicating risks
        if any(w in lower for w in ("risk", "broken", "bug", "issue", "problem", "invalid")):
            return ClaimType.FACT  # Risks are facts about problems

        # Keywords indicating plans
        if any(w in lower for w in ("implement", "add", "create", "fix", "phase", "step")):
            return ClaimType.PLAN

        return default

    def _classify_predicate(
        self,
        text: str,
        checkbox: str | None,
        doc_type: str,
    ) -> str:
        """Classify predicate from content and checkbox state."""
        if checkbox == "checked":
            return "completed_per_doc"
        if checkbox == "unchecked":
            return "planned_pending"

        lower = text.lower()
        if doc_type == "fixes":
            return "fix_required"
        if any(w in lower for w in ("broken", "invalid", "bug", "issue")):
            return "has_issue"
        if any(w in lower for w in ("decided", "chose")):
            return "decided"
        if any(w in lower for w in ("must", "require")):
            return "requires"

        return "planned"
