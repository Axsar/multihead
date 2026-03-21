"""Markdown section splitting, Claude output parsing, and claim conversion."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from multihead.knowledge_models import (
    Claim,
    ClaimCanonical,
    ClaimScope,
    ClaimType,
    EntityRef,
    EvidencePointer,
    Stability,
    ValueObject,
)
from multihead.narrative.confidence import ConfidenceCalibrator, SourcePriority

from .constants import _CLAIM_TYPE_MAP, _PROVENANCE, logger


def split_sections(content: str) -> list[dict[str, Any]]:
    """Split markdown into H2-level sections.

    Groups content under each ## heading. Sections with < 50 chars
    of content are skipped (too short for meaningful extraction).
    """
    sections: list[dict[str, Any]] = []
    current_heading = ""
    current_lines: list[str] = []

    for line in content.split("\n"):
        # Detect H1 or H2 headings
        if line.startswith("## ") or (line.startswith("# ") and not line.startswith("### ")):
            # Save previous section
            if current_heading and current_lines:
                text = "\n".join(current_lines).strip()
                if len(text) >= 50:
                    sections.append({
                        "heading": current_heading,
                        "text": text,
                    })
            current_heading = line.lstrip("# ").strip()
            current_lines = []
        elif line.startswith("### ") or line.startswith("#### "):
            # Include sub-headings as content
            current_lines.append(line)
        else:
            current_lines.append(line)

    # Last section
    if current_heading and current_lines:
        text = "\n".join(current_lines).strip()
        if len(text) >= 50:
            sections.append({
                "heading": current_heading,
                "text": text,
            })

    return sections


def parse_claude_output(output: str) -> list[dict[str, Any]]:
    """Extract claims list from Claude's JSON output.

    Handles various output formats: raw JSON, markdown code blocks,
    text with embedded JSON.
    """
    if not output:
        return []

    # Try direct JSON parse
    try:
        data = json.loads(output)
        if isinstance(data, dict) and "claims" in data:
            return data["claims"]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code block
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", output, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            if isinstance(data, dict) and "claims" in data:
                return data["claims"]
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    # Try finding JSON object in text
    for match in re.finditer(r'\{[^{}]*"claims"\s*:\s*\[.*?\]\s*\}', output, re.DOTALL):
        try:
            data = json.loads(match.group())
            if isinstance(data, dict) and "claims" in data:
                return data["claims"]
        except json.JSONDecodeError:
            continue

    logger.warning("Could not parse Claude output: %s...", output[:200])
    return []


def convert_to_claims(
    parsed: list[dict[str, Any]],
    doc_id: str,
    scope_id: str,
    scope: ClaimScope,
    stability: Stability,
    section_heading: str,
    calibrator: ConfidenceCalibrator,
    record_id: str | None = None,
) -> list[Claim]:
    """Convert parsed JSON claims into Claim objects.

    Args:
        record_id: The record_id to link evidence pointers to. If None, uses a placeholder.
    """
    claims: list[Claim] = []
    section_slug = re.sub(r"[^a-z0-9]+", "_", section_heading.lower()).strip("_")[:60]

    for item in parsed:
        text = item.get("text", "").strip()
        if not text or len(text) < 10:
            continue

        claim_type_str = item.get("claim_type", "fact").lower()
        claim_type = _CLAIM_TYPE_MAP.get(claim_type_str, ClaimType.FACT)
        predicate = item.get("predicate", "states")
        raw_conf = item.get("confidence", 0.75)

        # Calibrate with LLM_INFERENCE source (Claude output)
        cal = calibrator.calibrate(raw_conf, SourcePriority.LLM_INFERENCE)

        # Unique hash for claim key
        item_hash = hashlib.sha256(text.encode()).hexdigest()[:8]

        evidence = EvidencePointer(
            record_id=record_id or f"claude_enhanced_{doc_id}",
            uri=f"claude://{scope_id}/{doc_id}/{section_slug}",
            sha256=hashlib.sha256(text.encode()).hexdigest(),
            quote=text[:200],
        )

        claim = Claim(
            claim_type=claim_type,
            scope=scope,
            canonical=ClaimCanonical(
                claim_key=f"claude.{scope_id}.{doc_id}.{section_slug}.{item_hash}",
                subject=EntityRef(
                    entity_type="document",
                    entity_id=doc_id,
                    label=doc_id.replace("_", " ").title(),
                ),
                predicate=predicate,
                object=ValueObject(
                    value_type="string",
                    value=text[:500],
                ),
            ),
            statement=text[:300],
            confidence=cal.calibrated,
            stability=stability,
            evidence_supports=[evidence],
            provenance=_PROVENANCE,
        )
        claims.append(claim)

    return claims


def merge_claims(
    heuristic: list[Claim],
    claude: list[Claim],
) -> list[Claim]:
    """Merge heuristic and Claude claims, deduplicating by claim_key.

    Heuristic claims win on key collision (higher confidence cap).
    Claude claims with unique keys are added.
    """
    seen_keys: set[str] = set()
    merged: list[Claim] = []

    # Heuristic claims first (higher priority source)
    for claim in heuristic:
        key = claim.canonical.claim_key
        if key not in seen_keys:
            seen_keys.add(key)
            merged.append(claim)

    # Claude claims (lower priority but deeper extraction)
    for claim in claude:
        key = claim.canonical.claim_key
        if key not in seen_keys:
            seen_keys.add(key)
            merged.append(claim)

    return merged
