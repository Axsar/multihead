"""Claim extraction and deposit logic for the session harvester."""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import PROVENANCE_ID, ProjectInfo

logger = logging.getLogger(__name__)


def extract_claims(
    content: str,
    scope_id: str,
    project: ProjectInfo,
    file_path: Path,
    confidence: float,
) -> list[dict[str, Any]]:
    """Extract structured claims from markdown content.

    Reuses heuristic patterns from narrative/markdown_extractor:
    - Headings become topics
    - Bullets become facts
    - Checkboxes become status items
    - Key-value patterns become facts
    """
    claims: list[dict[str, Any]] = []
    current_heading = ""
    lines = content.split("\n")

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Track headings for topic context
        heading_match = re.match(r"^(#{1,4})\s+(.+)", stripped)
        if heading_match:
            current_heading = heading_match.group(2).strip()
            continue

        # Skip code blocks
        if stripped.startswith("```"):
            continue

        # Extract bullet items as claims
        bullet_match = re.match(r"^[-*]\s+(.+)", stripped)
        checkbox_match = re.match(r"^[-*]\s+\[([ xX\u2713])\]\s+(.+)", stripped)

        if checkbox_match:
            done = checkbox_match.group(1).lower() in ("x", "\u2713")
            item_text = checkbox_match.group(2).strip()
            claim_type = "fact"
            predicate = "completed" if done else "planned"

            claims.append(_make_claim_dict(
                statement=item_text,
                claim_type=claim_type,
                predicate=predicate,
                scope_id=scope_id,
                topic=current_heading,
                source_file=str(file_path),
                project_name=project.name,
                confidence=confidence,
            ))
        elif bullet_match:
            item_text = bullet_match.group(1).strip()
            # Skip very short items or markdown formatting artifacts
            if len(item_text) < 10:
                continue
            # Skip items that are just links or references
            if item_text.startswith("http") or item_text.startswith("["):
                if len(item_text) < 30:
                    continue

            claim_type = classify_claim_type(item_text)
            claims.append(_make_claim_dict(
                statement=item_text,
                claim_type=claim_type,
                predicate="states",
                scope_id=scope_id,
                topic=current_heading,
                source_file=str(file_path),
                project_name=project.name,
                confidence=confidence,
            ))
        else:
            # Key-value patterns like "**Key**: value"
            kv_match = re.match(r"^\*\*(.+?)\*\*:\s*(.+)", stripped)
            if kv_match:
                key = kv_match.group(1).strip()
                value = kv_match.group(2).strip()
                if len(value) >= 5:
                    claims.append(_make_claim_dict(
                        statement=f"{key}: {value}",
                        claim_type="fact",
                        predicate="defines",
                        scope_id=scope_id,
                        topic=current_heading,
                        source_file=str(file_path),
                        project_name=project.name,
                        confidence=confidence,
                    ))

    return claims


def classify_claim_type(text: str) -> str:
    """Classify a bullet item into a claim type."""
    text_lower = text.lower()

    decision_markers = {"decision:", "decided", "chose", "selected", "agreed"}
    constraint_markers = {"must", "never", "always", "required", "constraint:"}
    preference_markers = {"prefer", "preference:", "default to", "use "}
    plan_markers = {"plan:", "roadmap", "todo", "milestone", "will "}

    if any(m in text_lower for m in decision_markers):
        return "decision"
    if any(m in text_lower for m in constraint_markers):
        return "constraint"
    if any(m in text_lower for m in preference_markers):
        return "preference"
    if any(m in text_lower for m in plan_markers):
        return "plan"
    return "fact"


def deposit_claims(claims: list[dict[str, Any]], knowledge_store: Any) -> int:
    """Deposit extracted claims into knowledge.db. Returns count deposited."""
    deposited = 0
    for claim_dict in claims:
        try:
            from ..knowledge_models import (
                Claim, ClaimCanonical, ClaimScope, ClaimStatus, ClaimType,
                EntityRef, Provenance, ScopeType, ValueObject,
            )

            ct = ClaimType(claim_dict["claim_type"]) if claim_dict["claim_type"] in ClaimType._value2member_map_ else ClaimType.FACT
            now = datetime.now(timezone.utc)

            claim = Claim(
                claim_status=ClaimStatus.PROPOSED,
                claim_type=ct,
                scope=ClaimScope(
                    scope_type=ScopeType.PROJECT,
                    scope_id=claim_dict["scope_id"],
                    valid_from=now,
                ),
                canonical=ClaimCanonical(
                    claim_key=claim_dict["claim_key"],
                    subject=EntityRef(
                        entity_type="session_memory",
                        entity_id=claim_dict["project_name"],
                        label=claim_dict.get("topic", ""),
                    ),
                    predicate=claim_dict["predicate"],
                    object=ValueObject(value_type="string", value=True),
                ),
                statement=claim_dict["statement"][:500],
                confidence=claim_dict["confidence"],
                provenance=Provenance(
                    produced_by={
                        "kind": "harvester",
                        "id": PROVENANCE_ID,
                    },
                    toolchain=[{
                        "name": "session_harvester",
                        "version": "1.0",
                        "source_file": claim_dict["source_file"],
                    }],
                ),
            )
            knowledge_store.insert_claim(claim)
            deposited += 1
        except Exception as e:
            logger.debug("Failed to deposit claim: %s", e)

    return deposited


def _make_claim_dict(
    statement: str,
    claim_type: str,
    predicate: str,
    scope_id: str,
    topic: str,
    source_file: str,
    project_name: str,
    confidence: float,
) -> dict[str, Any]:
    """Build a claim dict for later deposit."""
    # Stable claim key from content hash
    content_hash = hashlib.sha256(
        f"{scope_id}:{source_file}:{statement}".encode()
    ).hexdigest()[:12]
    topic_slug = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")[:30] if topic else "general"

    return {
        "statement": statement,
        "claim_type": claim_type,
        "predicate": predicate,
        "scope_id": scope_id,
        "topic": topic,
        "source_file": source_file,
        "project_name": project_name,
        "confidence": confidence,
        "claim_key": f"session.{scope_id}.{topic_slug}.{content_hash}",
    }
