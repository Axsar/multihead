"""Knowledge store tools for the Agentic Core's tool registry.

Gives the local LLM the ability to query claims and events from the
knowledge store built by Night Shift.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .knowledge_store import KnowledgeStore
from .tool_registry import ToolRegistry, ToolResult, ToolSpec

logger = logging.getLogger(__name__)


def register_knowledge_tools(registry: ToolRegistry, store: KnowledgeStore) -> None:
    """Register knowledge store query tools."""

    async def _query_claims(params: dict[str, Any]) -> ToolResult:
        status = params.get("status")
        claim_type = params.get("claim_type")
        scope_id = params.get("scope_id")
        limit = min(int(params.get("limit", 20)), 100)
        search = params.get("search", "")

        try:
            claims = store.list_claims(
                status=status, claim_type=claim_type,
                scope_id=scope_id, limit=limit,
            )
            # Filter by search term if provided
            if search:
                search_lower = search.lower()
                claims = [
                    c for c in claims
                    if search_lower in c.statement.lower()
                    or search_lower in c.canonical.claim_key.lower()
                    or search_lower in c.canonical.subject.entity_id.lower()
                ]

            items = []
            for c in claims:
                items.append({
                    "claim_id": c.claim_id,
                    "type": c.claim_type.value,
                    "status": c.claim_status.value,
                    "key": c.canonical.claim_key,
                    "statement": c.statement[:200],
                    "confidence": c.confidence,
                    "subject": c.canonical.subject.entity_id,
                })
            return ToolResult(
                tool="knowledge.claims",
                success=True,
                output=json.dumps(items, indent=2),
            )
        except Exception as e:
            return ToolResult(tool="knowledge.claims", success=False, error=str(e))

    async def _query_events(params: dict[str, Any]) -> ToolResult:
        event_type = params.get("event_type")
        status = params.get("status")
        limit = min(int(params.get("limit", 20)), 100)
        search = params.get("search", "")

        try:
            events = store.list_events(
                event_type=event_type, status=status, limit=limit,
            )
            # Filter by search term if provided
            if search:
                search_lower = search.lower()
                events = [
                    e for e in events
                    if search_lower in e.title.lower()
                    or search_lower in e.summary.lower()
                ]

            items = []
            for e in events:
                items.append({
                    "event_id": e.event_id,
                    "type": e.event_type.value,
                    "title": e.title[:160],
                    "summary": e.summary[:200],
                })
            return ToolResult(
                tool="knowledge.events",
                success=True,
                output=json.dumps(items, indent=2),
            )
        except Exception as e:
            return ToolResult(tool="knowledge.events", success=False, error=str(e))

    async def _knowledge_stats(params: dict[str, Any]) -> ToolResult:
        try:
            claims = store.list_claims(limit=10000)
            events = store.list_events(limit=10000)

            from collections import Counter
            claim_types = Counter(c.claim_type.value for c in claims)
            claim_statuses = Counter(c.claim_status.value for c in claims)
            event_types = Counter(e.event_type.value for e in events)

            stats = {
                "total_claims": len(claims),
                "total_events": len(events),
                "claim_types": dict(claim_types.most_common()),
                "claim_statuses": dict(claim_statuses.most_common()),
                "event_types": dict(event_types.most_common()),
            }
            return ToolResult(
                tool="knowledge.stats",
                success=True,
                output=json.dumps(stats, indent=2),
            )
        except Exception as e:
            return ToolResult(tool="knowledge.stats", success=False, error=str(e))

    registry.register(
        ToolSpec(
            name="knowledge.claims",
            description="Search claims in the knowledge store. Filter by status, claim_type, scope_id, or free-text search.",
            params_schema={
                "search": {"type": "string", "description": "Free-text search in statement, claim_key, or subject"},
                "status": {"type": "string", "description": "Filter: proposed, accepted, contested, superseded"},
                "claim_type": {"type": "string", "description": "Filter: fact, decision, definition, constraint, plan, risk, assumption, preference"},
                "limit": {"type": "integer", "description": "Max results (default 20, max 100)"},
            },
        ),
        _query_claims,
    )

    registry.register(
        ToolSpec(
            name="knowledge.events",
            description="Search events in the knowledge store. Filter by event_type, status, or free-text search.",
            params_schema={
                "search": {"type": "string", "description": "Free-text search in title or summary"},
                "event_type": {"type": "string", "description": "Filter: decision, task_created, task_completed, tool_run, commit, note, milestone, spec_change"},
                "limit": {"type": "integer", "description": "Max results (default 20, max 100)"},
            },
        ),
        _query_events,
    )

    registry.register(
        ToolSpec(
            name="knowledge.stats",
            description="Get summary statistics of the knowledge store (total claims, events, breakdowns by type and status).",
            params_schema={},
        ),
        _knowledge_stats,
    )
