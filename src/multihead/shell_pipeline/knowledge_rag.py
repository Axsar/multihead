"""Stage 1: Knowledge RAG — query knowledge.db for relevant context.

Includes:
- Full-text search (FTS5) with SQL LIKE fallback
- Inbox context (pending claims/requests from knowledge.db)
- Keyword extraction
"""

from __future__ import annotations

import logging
from typing import Any

from .constants import AGENT_ID, SELF_IDENTITIES, _STOPWORDS

logger = logging.getLogger(__name__)


def extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from text (>3 chars, no stopwords)."""
    words = []
    for w in text.split():
        cleaned = w.lower().strip(".,!?;:'\"()[]{}#@")
        if len(cleaned) > 3 and cleaned not in _STOPWORDS:
            words.append(cleaned)
    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for w in words:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return unique


def query_claims_fts(
    ks: Any,
    keywords: list[str],
    limit: int = 10,
) -> list[tuple[str, str, float]]:
    """Query claims using FTS5 MATCH, falling back to LIKE.

    Returns list of (claim_key, statement, confidence) tuples,
    ranked by relevance.
    """
    if not ks or not keywords:
        return []

    # Try KnowledgeStore.search_claims_fts first (uses FTS5 with fallback)
    if hasattr(ks, "search_claims_fts") and callable(
        getattr(ks, "search_claims_fts", None)
    ):
        try:
            query = " ".join(keywords[:8])
            result = ks.search_claims_fts(
                query, limit=limit, min_confidence=0.3, max_age_days=90,
            )
            # Validate result is actually a list of tuples
            if isinstance(result, list) and all(
                isinstance(r, (tuple, list)) for r in result
            ):
                return result
        except Exception:
            pass

    # Direct SQL fallback
    return query_claims_by_keywords(ks, keywords, limit)


def query_claims_by_keywords(
    ks: Any,
    keywords: list[str],
    limit: int = 10,
) -> list[tuple[str, str, float]]:
    """Fallback: Query claims using SQL LIKE for each keyword.

    Returns list of (claim_key, statement, confidence) tuples,
    ordered by number of keyword matches (most relevant first).
    """
    if not ks or not keywords:
        return []

    if not hasattr(ks, "_connect"):
        return []

    try:
        with ks._connect() as conn:
            # Build query: count matches per claim across keywords
            case_parts = []
            params: list[str] = []
            for kw in keywords[:8]:
                case_parts.append(
                    "(CASE WHEN LOWER(statement) LIKE ? THEN 1 ELSE 0 END)"
                )
                params.append(f"%{kw}%")

            score_expr = " + ".join(case_parts) if case_parts else "0"

            sql = f"""
                SELECT claim_key, statement, confidence,
                       ({score_expr}) AS relevance
                FROM claims
                WHERE claim_status = 'accepted'
                  AND ({" OR ".join("LOWER(statement) LIKE ?" for _ in keywords[:8])})
                ORDER BY relevance DESC, confidence DESC
                LIMIT ?
            """
            params.extend(f"%{kw}%" for kw in keywords[:8])
            params.append(str(limit))

            rows = conn.execute(sql, params).fetchall()
            return [(row[0] or "", row[1] or "", row[2] or 0.0) for row in rows]

    except Exception as e:
        logger.warning("SQL knowledge query error: %s", e)
        return []


def build_knowledge_context(ks: Any, user_message: str) -> str:
    """Query knowledge.db for claims relevant to user's message.

    Uses direct SQL queries for speed (26k+ claims). Returns formatted
    context block with claim keys for traceability.
    """
    if not ks:
        return ""

    try:
        keywords = extract_keywords(user_message)
        if not keywords:
            return ""

        claims = query_claims_fts(ks, keywords, limit=10)
        if not claims:
            return ""

        lines = ["[Knowledge context from knowledge.db]"]
        for claim_key, statement, _confidence in claims:
            lines.append(f"- [{claim_key}] {statement[:200]}")
        return "\n".join(lines)

    except Exception as e:
        logger.debug("Knowledge context error: %s", e)
        return ""


def build_inbox_context(ks: Any, session_id: str) -> str:
    """Build a brief inbox summary of pending claims/requests.

    Uses claim_interactions table to only show claims this agent
    hasn't already handled (read/responded/dismissed). Once shown,
    claims are marked as 'read' so they won't reappear.

    Uses stable AGENT_ID for interaction tracking so dedup persists
    across shell restarts. Filters out self-produced claims.
    """
    if not ks:
        return ""

    try:
        items: list[str] = []
        shown_claim_ids: list[str] = []

        # Use interaction-aware query if available, fall back to legacy
        if hasattr(ks, "get_unhandled_claims"):
            # Get claims we haven't interacted with yet (stable identity)
            unhandled = ks.get_unhandled_claims(
                agent_id=AGENT_ID,
                claim_types=["question", "request"],
                scope_id="multihead",
                max_age_hours=48,
                limit=5,
            )
            for claim in unhandled:
                # Skip self-produced claims
                requester = claim.provenance.produced_by.get("id", "unknown")
                if requester in SELF_IDENTITIES:
                    shown_claim_ids.append(claim.claim_id)
                    continue
                stmt = claim.statement[:150].replace("\n", " ")
                items.append(f"- [{requester}] {stmt}")
                shown_claim_ids.append(claim.claim_id)

            # Also check for unhandled proposed plans
            unhandled_plans = ks.get_unhandled_claims(
                agent_id=AGENT_ID,
                claim_types=["plan"],
                scope_id="multihead",
                max_age_hours=48,
                limit=3,
            )
            for plan in unhandled_plans:
                author = plan.provenance.produced_by.get("id", "unknown")
                if author in SELF_IDENTITIES:
                    shown_claim_ids.append(plan.claim_id)
                    continue
                stmt = plan.statement[:150].replace("\n", " ")
                items.append(f"- [plan from {author}] {stmt}")
                shown_claim_ids.append(plan.claim_id)

        elif hasattr(ks, "get_pending_messages"):
            # Legacy fallback (no interaction tracking)
            pending = ks.get_pending_messages(
                session_id=AGENT_ID,
                scope_id="multihead",
                max_age_hours=48,
                limit=5,
            )
            for claim in pending:
                requester = claim.provenance.produced_by.get("id", "unknown")
                if requester in SELF_IDENTITIES:
                    continue
                stmt = claim.statement[:150].replace("\n", " ")
                items.append(f"- [{requester}] {stmt}")

            # Legacy plan check
            proposed_plans = ks.list_claims(
                claim_type="plan", status="proposed", scope_id="multihead", limit=3,
            )
            for plan in proposed_plans:
                author = plan.provenance.produced_by.get("id", "unknown")
                if author in SELF_IDENTITIES:
                    continue
                stmt = plan.statement[:150].replace("\n", " ")
                items.append(f"- [plan from {author}] {stmt}")

        if not items:
            # Still mark self-produced claims as read to prevent resurfacing
            if shown_claim_ids and hasattr(ks, "record_interaction"):
                for cid in shown_claim_ids:
                    try:
                        ks.record_interaction(
                            claim_id=cid, agent_id=AGENT_ID, action="read",
                        )
                    except Exception:
                        pass
            return ""

        # Mark shown claims as 'read' so they don't reappear
        if shown_claim_ids and hasattr(ks, "record_interaction"):
            for cid in shown_claim_ids:
                try:
                    ks.record_interaction(
                        claim_id=cid, agent_id=AGENT_ID, action="read",
                    )
                except Exception:
                    pass  # Non-critical — don't break inbox on tracking failure

        header = f"[Inbox: {len(items)} pending item(s) — address if relevant]"
        return header + "\n" + "\n".join(items[:5])

    except Exception as e:
        logger.debug("Inbox context error: %s", e)
        return ""
