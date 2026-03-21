"""Reference Bank — auto-generated knowledge files for Claude Code memory.

Queries knowledge.db, ranks claims by type/recency/confidence, and writes
topic-specific .md files to Claude Code's project memory directory.
These files survive SDK context compaction because they're loaded fresh
on every session start.

Usage:
    multihead refresh-refs              # CLI command
    RefBankBuilder(ks, memory_dir).refresh_all()  # Python API
    Night Shift stage: ref_bank_sync    # Automated nightly
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Recency half-life in days (matches context_packs.py)
_RECENCY_HALF_LIFE = 7.0


def _recency_score(updated_at: datetime | None) -> float:
    """Exponential decay with 7-day half-life."""
    if not updated_at:
        return 0.3
    now = datetime.now(timezone.utc)
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    age_days = max(0, (now - updated_at).total_seconds() / 86400)
    return math.exp(-age_days * math.log(2) / _RECENCY_HALF_LIFE)


# ---------------------------------------------------------------------------
# Ref file definitions
# ---------------------------------------------------------------------------

REF_FILES: list[dict[str, Any]] = [
    {
        "filename": "ref-decisions.md",
        "title": "Recent Decisions",
        "description": "Key decisions from knowledge.db (last 30 days, high confidence)",
        "query": "decision OR plan OR architecture OR chose OR adopted",
        "claim_types": ["decision", "plan"],
        "max_items": 25,
        "max_lines": 120,
    },
    {
        "filename": "ref-constraints.md",
        "title": "Active Constraints & Invariants",
        "description": "Rules, constraints, and invariants that must always hold",
        "query": "constraint OR must OR never OR always OR invariant OR rule",
        "claim_types": ["constraint", "preference"],
        "max_items": 20,
        "max_lines": 80,
    },
    {
        "filename": "ref-architecture.md",
        "title": "Architecture & Patterns",
        "description": "Architecture facts, patterns, and component descriptions",
        "query": "architecture OR adapter OR pipeline OR router OR manager OR component",
        "claim_types": ["fact", "definition"],
        "max_items": 30,
        "max_lines": 150,
    },
    {
        "filename": "ref-open-loops.md",
        "title": "Open Loops & Questions",
        "description": "Unresolved questions, pending items, and open issues",
        "query": "question OR pending OR unresolved OR TODO OR investigate",
        "claim_types": ["question", "risk", "assumption"],
        "max_items": 15,
        "max_lines": 80,
    },
    {
        "filename": "ref-recent-activity.md",
        "title": "Recent Activity (Last 7 Days)",
        "description": "What changed, what was learned, what was built recently",
        "query": None,  # Use recency, not keyword search
        "claim_types": None,  # All types
        "max_items": 30,
        "max_lines": 150,
        "recency_days": 7,
    },
    {
        "filename": "ref-capabilities.md",
        "title": "Head Capabilities & Benchmarks",
        "description": "Model capabilities, performance benchmarks, routing rules",
        "query": "head OR model OR VRAM OR benchmark OR capability OR accuracy OR mAP",
        "claim_types": ["fact", "definition"],
        "max_items": 20,
        "max_lines": 100,
    },
]


class RefBankBuilder:
    """Generates reference bank .md files from knowledge.db claims."""

    def __init__(
        self,
        knowledge_store: Any,
        memory_dir: Path,
    ) -> None:
        self.ks = knowledge_store
        self.memory_dir = memory_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def refresh_all(self) -> list[dict[str, Any]]:
        """Regenerate all reference bank files.

        Returns list of dicts with filename, items_count, lines for reporting.
        """
        results = []
        for ref_def in REF_FILES:
            result = self._build_ref_file(ref_def)
            results.append(result)
        return results

    def _build_ref_file(self, ref_def: dict[str, Any]) -> dict[str, Any]:
        """Build a single ref file from its definition."""
        filename = ref_def["filename"]
        title = ref_def["title"]
        max_items = ref_def["max_items"]
        max_lines = ref_def["max_lines"]

        # Gather claims
        claims = self._gather_claims(ref_def)

        # Score and rank
        scored = []
        for c in claims:
            score = self._score_claim(c)
            scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:max_items]

        # Build markdown
        lines = self._format_ref_file(title, ref_def["description"], top)

        # Trim to max lines
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines.append("\n> ... truncated (run `multihead refresh-refs` for full update)")

        # Write
        path = self.memory_dir / filename
        content = "\n".join(lines)
        path.write_text(content, encoding="utf-8")

        result = {
            "filename": filename,
            "path": str(path),
            "items_count": len(top),
            "lines": len(lines),
        }
        logger.info("Ref bank: wrote %s (%d items, %d lines)", filename, len(top), len(lines))
        return result

    def _gather_claims(self, ref_def: dict[str, Any]) -> list[Any]:
        """Gather candidate claims for a ref file."""
        query = ref_def.get("query")
        claim_types = ref_def.get("claim_types")
        recency_days = ref_def.get("recency_days")

        # Strategy 1: FTS5 search by query
        if query:
            try:
                results = self.ks.search_claims_fts(query, limit=100)
                # results are (claim_key, statement, confidence) tuples
                # Convert to lightweight claim-like objects
                claims = [
                    _ClaimProxy(key=r[0], statement=r[1], confidence=r[2])
                    for r in results
                ]
            except Exception:
                claims = []
        else:
            claims = []

        # Strategy 2: Filter by claim_type (supplement FTS results)
        if claim_types:
            for ct in claim_types:
                try:
                    typed_claims = self.ks.list_claims(
                        status="accepted", claim_type=ct, limit=50,
                    )
                    for c in typed_claims:
                        claims.append(_ClaimProxy(
                            key=getattr(c.canonical, "claim_key", ""),
                            statement=c.statement,
                            confidence=c.confidence,
                            claim_type=ct,
                            updated_at=getattr(c, "updated_at", None),
                        ))
                except Exception:
                    pass

        # Strategy 3: Recent claims by time window
        if recency_days and not query:
            try:
                recent = self.ks.list_claims(status="accepted", limit=200)
                now = datetime.now(timezone.utc)
                for c in recent:
                    updated = getattr(c, "updated_at", None)
                    if updated:
                        if updated.tzinfo is None:
                            updated = updated.replace(tzinfo=timezone.utc)
                        age = (now - updated).total_seconds() / 86400
                        if age <= recency_days:
                            claims.append(_ClaimProxy(
                                key=getattr(c.canonical, "claim_key", ""),
                                statement=c.statement,
                                confidence=c.confidence,
                                updated_at=updated,
                            ))
            except Exception:
                pass

        # Deduplicate by statement (first 100 chars)
        seen: set[str] = set()
        unique = []
        for c in claims:
            sig = c.statement[:100].lower()
            if sig not in seen:
                seen.add(sig)
                unique.append(c)

        return unique

    def _score_claim(self, claim: _ClaimProxy) -> float:
        """Score a claim for ranking (higher = more relevant)."""
        confidence = claim.confidence if claim.confidence else 0.5
        recency = _recency_score(claim.updated_at)
        # Weighted: 50% confidence, 40% recency, 10% base
        return 0.5 * confidence + 0.4 * recency + 0.1

    def _format_ref_file(
        self,
        title: str,
        description: str,
        scored_claims: list[tuple[float, _ClaimProxy]],
    ) -> list[str]:
        """Format a ref file as markdown."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            f"# {title}",
            f"> Auto-generated from knowledge.db | {now}",
            f"> {description}",
            f"> Items: {len(scored_claims)} | Run `multihead refresh-refs` to update",
            "",
        ]

        if not scored_claims:
            lines.append("*No matching claims found.*")
            return lines

        for score, claim in scored_claims:
            conf = f"{claim.confidence:.0%}" if claim.confidence else "?"
            # Truncate long statements
            stmt = claim.statement[:300]
            if len(claim.statement) > 300:
                stmt += "..."
            lines.append(f"- [{conf}] {stmt}")

        return lines


class _ClaimProxy:
    """Lightweight claim-like object for ref bank scoring."""

    __slots__ = ("key", "statement", "confidence", "claim_type", "updated_at")

    def __init__(
        self,
        key: str = "",
        statement: str = "",
        confidence: float = 0.5,
        claim_type: str = "fact",
        updated_at: datetime | None = None,
    ) -> None:
        self.key = key
        self.statement = statement
        self.confidence = confidence
        self.claim_type = claim_type
        self.updated_at = updated_at
