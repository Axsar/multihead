"""Core tool logic: chat, generate, heads, knowledge, config, etc."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx

from multihead.knowledge_models import (
    Claim,
    ClaimCanonical,
    ClaimScope,
    ClaimStatus,
    ClaimType,
    EntityRef,
    EventStatus,
    EventType,
    KnowledgeEvent,
    Provenance,
    ScopeType,
    Stability,
    TimeBlock,
    TimePrecision,
    ValueObject,
)

from ._core import _get_ks, _request


async def _chat(message: str, session_id: str | None = None) -> str:
    payload = {"message": message}
    if session_id:
        payload["session_id"] = session_id
    try:
        result = await _request("POST", "/chat", json=payload)
        return json.dumps(result, indent=2)
    except httpx.ConnectError:
        return "Error: MultiHead server not running. Start it with: multihead serve"
    except Exception as e:
        return f"Error: {e}"


async def _generate(head_id: str, prompt: str, temperature: float | None = None, max_tokens: int | None = None) -> str:
    payload: dict = {"prompt": prompt}
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    try:
        result = await _request("POST", f"/heads/{head_id}/generate", json=payload)
        return json.dumps(result, indent=2)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return f"Error: Head '{head_id}' not found. Use multihead_heads() to list available heads."
        return f"Error: {e}"
    except httpx.ConnectError:
        return "Error: MultiHead server not running. Start it with: multihead serve"
    except Exception as e:
        return f"Error: {e}"


async def _heads() -> str:
    try:
        result = await _request("GET", "/heads")
        return json.dumps(result, indent=2)
    except httpx.ConnectError:
        return "Error: MultiHead server not running. Start it with: multihead serve"
    except Exception as e:
        return f"Error: {e}"


async def _swap_head(head_id: str, action: str = "wake") -> str:
    if action not in ("wake", "sleep", "unload"):
        return f"Error: action must be 'wake', 'sleep', or 'unload', got '{action}'"
    try:
        result = await _request("POST", f"/heads/{head_id}/{action}")
        return json.dumps(result, indent=2)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return f"Error: Head '{head_id}' not found."
        return f"Error: {e}"
    except httpx.ConnectError:
        return "Error: MultiHead server not running. Start it with: multihead serve"
    except Exception as e:
        return f"Error: {e}"


async def _run_recipe(recipe: str, inputs: dict | None = None) -> str:
    payload: dict = {"recipe": recipe}
    if inputs:
        payload["inputs"] = inputs
    try:
        result = await _request("POST", "/runs", json=payload)
        return json.dumps(result, indent=2)
    except httpx.ConnectError:
        return "Error: MultiHead server not running. Start it with: multihead serve"
    except Exception as e:
        return f"Error: {e}"


async def _run_status(run_id: str) -> str:
    try:
        result = await _request("GET", f"/runs/{run_id}")
        return json.dumps(result, indent=2)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return f"Error: Run '{run_id}' not found."
        return f"Error: {e}"
    except httpx.ConnectError:
        return "Error: MultiHead server not running. Start it with: multihead serve"
    except Exception as e:
        return f"Error: {e}"


async def _knowledge(query_type: str = "claims", status: str | None = None, limit: int = 10) -> str:
    if query_type not in ("claims", "events"):
        return f"Error: query_type must be 'claims' or 'events', got '{query_type}'"
    try:
        ks = _get_ks()
        if query_type == "claims":
            claims = ks.list_claims(status=status, limit=limit)
            result = [
                {
                    "claim_id": c.claim_id,
                    "claim_type": c.claim_type.value,
                    "claim_status": c.claim_status.value,
                    "statement": c.statement[:200],
                    "confidence": c.confidence,
                    "scope_id": c.scope.scope_id,
                    "claim_key": c.canonical.claim_key,
                    "created_at": c.provenance.created_at.isoformat(),
                }
                for c in claims
            ]
        else:
            events = ks.list_events(status=status, limit=limit)
            result = [
                {
                    "event_id": e.event_id,
                    "event_type": e.event_type.value,
                    "event_status": e.event_status.value,
                    "title": e.title[:200],
                    "summary": e.summary[:200] if e.summary else "",
                    "created_at": e.provenance.created_at.isoformat(),
                }
                for e in events
            ]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {e}"


async def _deposit_claim(
    claim_key: str, statement: str, produced_by: str = "claude_code",
    scope_id: str = "default", claim_type: str = "fact", confidence: float = 0.9,
) -> str:
    try:
        ks = _get_ks()
        now = datetime.now(timezone.utc)
        claim = Claim(
            claim_status=ClaimStatus.ACCEPTED,
            claim_type=ClaimType(claim_type),
            scope=ClaimScope(
                scope_type=ScopeType.PROJECT,
                scope_id=scope_id,
                valid_from=now,
            ),
            canonical=ClaimCanonical(
                claim_key=claim_key,
                subject=EntityRef(
                    entity_type="component",
                    entity_id=claim_key.split(".")[0],
                ),
                predicate="has_state",
                object=ValueObject(value_type="string", value=True),
            ),
            statement=statement,
            confidence=confidence,
            provenance=Provenance(
                produced_by={"kind": "external", "id": produced_by},
            ),
        )
        claim = ks.insert_claim(claim)
        return json.dumps({
            "claim_id": claim.claim_id,
            "claim_key": claim.canonical.claim_key,
            "statement": claim.statement,
            "claim_status": claim.claim_status.value,
        }, indent=2)
    except Exception as e:
        return f"Error: {e}"


async def _deposit_action_claim(
    scope_id: str,
    action_type: str,
    short_id: str,
    statement: str,
    produced_by: str = "claude-multihead-main",
    deadline_hours: int = 48,
) -> str:
    """Deposit an action claim with enforced key convention and valid_to deadline.

    Enforces claim_key = action.{scope_id}.{action_type}.{short_id}
    Sets valid_to = now + deadline_hours.
    """
    from datetime import timedelta

    allowed_types = {"work_order", "consensus", "vote", "progress", "result", "blocker", "proposal", "assignment"}
    if action_type not in allowed_types:
        return f"Error: action_type must be one of {allowed_types}, got '{action_type}'"

    # Map action_type to appropriate claim_type
    _type_map = {
        "work_order": ClaimType.PLAN,
        "consensus": ClaimType.PLAN,
        "proposal": ClaimType.PLAN,
        "assignment": ClaimType.DECISION,
        "vote": ClaimType.DECISION,
        "progress": ClaimType.FACT,
        "result": ClaimType.FACT,
        "blocker": ClaimType.CONSTRAINT,
    }
    claim_key = f"action.{scope_id}.{action_type}.{short_id}"
    try:
        ks = _get_ks()
        now = datetime.now(timezone.utc)
        deadline = now + timedelta(hours=deadline_hours)
        claim = Claim(
            claim_status=ClaimStatus.ACCEPTED,
            claim_type=_type_map.get(action_type, ClaimType.PLAN),
            scope=ClaimScope(
                scope_type=ScopeType.PROJECT,
                scope_id=scope_id,
                valid_from=now,
                valid_to=deadline,
            ),
            canonical=ClaimCanonical(
                claim_key=claim_key,
                subject=EntityRef(entity_type="action", entity_id=short_id),
                predicate=action_type,
                object=ValueObject(value_type="string", value=True),
            ),
            statement=statement,
            confidence=0.9,
            provenance=Provenance(
                produced_by={"kind": "external", "id": produced_by},
            ),
        )
        claim = ks.insert_claim(claim)
        return json.dumps({
            "claim_id": claim.claim_id,
            "claim_key": claim.canonical.claim_key,
            "deadline": deadline.isoformat(),
            "claim_status": claim.claim_status.value,
        }, indent=2)
    except Exception as e:
        return f"Error: {e}"


async def _report_event(
    title: str, summary: str = "", event_type: str = "note",
    produced_by: str = "claude_code", tags: list[str] | None = None,
    metrics: dict | None = None,
) -> str:
    try:
        ks = _get_ks()
        now = datetime.now(timezone.utc)
        event = KnowledgeEvent(
            event_status=EventStatus.CONFIRMED,
            event_type=EventType(event_type),
            title=title,
            summary=summary,
            time=TimeBlock(happened_at=now, time_precision=TimePrecision.SECOND),
            tags=tags or [],
            metrics=metrics or {},
            provenance=Provenance(
                produced_by={"kind": "external", "id": produced_by},
            ),
        )
        event = ks.insert_event(event)
        return json.dumps({
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "title": event.title,
            "event_status": event.event_status.value,
        }, indent=2)
    except Exception as e:
        return f"Error: {e}"


async def _briefing(component: str, scope_id: str = "default") -> str:
    try:
        ks = _get_ks()

        # 1. Claims whose key or statement matches the component
        all_claims = ks.list_claims(scope_id=scope_id, limit=500)
        component_claims = []
        related_claims = []
        for c in all_claims:
            key = c.canonical.claim_key
            if component in key:
                component_claims.append(c)
            elif component.replace("_", " ") in c.statement.lower() or component in c.statement.lower():
                related_claims.append(c)

        component_claims = component_claims[:20]
        related_claims = related_claims[:20]

        # 2. Recent events mentioning this component
        all_events = ks.list_events(limit=200)
        events_out = []
        for e in all_events:
            matched = (
                component in (e.summary or "").lower()
                or component in e.title.lower()
                or any(component in t.lower() for t in e.tags)
                or any(component in ent.entity_id for ent in e.entities)
            )
            if matched:
                events_out.append(e)
            if len(events_out) >= 10:
                break

        result = {
            "component": component,
            "scope_id": scope_id,
            "claims": [
                {
                    "claim_id": c.claim_id,
                    "claim_key": c.canonical.claim_key,
                    "statement": c.statement,
                    "claim_type": c.claim_type.value,
                    "confidence": c.confidence,
                    "status": c.claim_status.value,
                }
                for c in component_claims
            ],
            "related_claims": [
                {
                    "claim_id": c.claim_id,
                    "claim_key": c.canonical.claim_key,
                    "statement": c.statement,
                    "claim_type": c.claim_type.value,
                    "confidence": c.confidence,
                    "status": c.claim_status.value,
                }
                for c in related_claims
            ],
            "recent_events": [
                {
                    "event_id": e.event_id,
                    "title": e.title,
                    "summary": e.summary,
                    "event_type": e.event_type.value,
                    "created_at": e.provenance.created_at.isoformat(),
                }
                for e in events_out
            ],
            "summary": f"{len(component_claims)} direct claims, {len(related_claims)} related, {len(events_out)} events",
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {e}"


async def _file_briefing(file_path: str) -> str:
    """Get knowledge briefing for a file before editing.

    Returns 4 categories:
    - CONSTRAINTS: corroborated facts (independently verified — treat as invariants)
    - WARNINGS: stale claims (things that changed — verify before assuming)
    - SIGNALS: contested claims (disagreements between channels — be careful here)
    - HISTORY: superseded claims (failed approaches — don't repeat these)
    """
    try:
        ks = _get_ks()
        conn = ks._connect()

        # Normalize file path — strip absolute prefixes, match on filename
        import os
        from multihead._paths import normalize_file_path
        basename = os.path.basename(file_path)
        rel_path = normalize_file_path(file_path)

        # Query claims anchored to this file
        rows = conn.execute(
            "SELECT claim_id, claim_key, statement, claim_status, confidence, observation_method "
            "FROM claims WHERE provenance_json LIKE ? OR provenance_json LIKE ? "
            "ORDER BY claim_status, confidence DESC LIMIT 100",
            (f"%{rel_path}%", f"%{basename}%"),
        ).fetchall()

        constraints = []  # corroborated
        warnings = []     # stale
        signals = []      # contested
        history = []      # superseded
        context = []      # proposed/accepted (background)

        for r in rows:
            entry = {
                "claim_key": r["claim_key"],
                "statement": r["statement"][:300],
                "confidence": r["confidence"],
                "channel": r["observation_method"],
            }
            status = r["claim_status"]
            if status == "corroborated":
                constraints.append(entry)
            elif status == "stale":
                warnings.append(entry)
            elif status == "contested":
                signals.append(entry)
            elif status == "superseded":
                history.append(entry)
            else:
                context.append(entry)

        # Also check for improvement_unverified claims mentioning this file
        unverified = conn.execute(
            "SELECT claim_key, statement FROM claims "
            "WHERE contested_reason LIKE '%IMPROVEMENT_UNVERIFIED%' "
            "AND (provenance_json LIKE ? OR provenance_json LIKE ?) LIMIT 10",
            (f"%{rel_path}%", f"%{basename}%"),
        ).fetchall()

        conn.close()

        sections = []
        if constraints:
            sections.append(f"## CONSTRAINTS ({len(constraints)} corroborated — treat as invariants)")
            for c in constraints[:10]:
                sections.append(f"  [{c['channel']}] {c['statement']}")
        if warnings:
            sections.append(f"\n## WARNINGS ({len(warnings)} stale — verify before assuming)")
            for w in warnings[:10]:
                sections.append(f"  {w['statement']}")
        if signals:
            sections.append(f"\n## SIGNALS ({len(signals)} contested — be careful)")
            for s in signals[:10]:
                sections.append(f"  [{s['channel']}] {s['statement']}")
        if history:
            sections.append(f"\n## HISTORY ({len(history)} superseded — don't repeat)")
            for h in history[:10]:
                sections.append(f"  {h['statement']}")
        if unverified:
            sections.append(f"\n## UNVERIFIED FIXES ({len(unverified)} — claimed fixed but not confirmed)")
            for u in unverified:
                sections.append(f"  {u['statement'][:200]}")
        if context:
            sections.append(f"\n## CONTEXT ({len(context)} proposed/accepted claims)")
            for c in context[:5]:
                sections.append(f"  [{c['channel']}] {c['statement']}")

        if not sections:
            return f"No knowledge found for {file_path}. This file has no claims in the knowledge base."

        header = f"# Knowledge Briefing: {rel_path}\n"
        return header + "\n".join(sections)

    except Exception as e:
        return f"Error: {e}"


async def _config(action: str = "show", key: str | None = None, value: str | None = None) -> str:
    try:
        if action == "show":
            result = await _request("GET", "/config")
            return json.dumps(result, indent=2)
        elif action == "set":
            if not key or value is None:
                return "Error: 'set' action requires 'key' and 'value' parameters."
            result = await _request("POST", "/config/set", json={"key": key, "value": value})
            return json.dumps(result, indent=2)
        else:
            return f"Error: action must be 'show' or 'set', got '{action}'"
    except httpx.ConnectError:
        return "Error: MultiHead server not running. Start it with: multihead serve"
    except Exception as e:
        return f"Error: {e}"
