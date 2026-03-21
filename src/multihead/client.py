"""Lightweight HTTP client for external processes to talk to MultiHead.

Usage from any Python process:

    from multihead.client import MultiHeadClient

    mh = MultiHeadClient()  # defaults to localhost:7337
    mh.deposit_claim(
        claim_key="project.auth.status",
        statement="JWT validation middleware deployed with 0 errors",
        produced_by="auth-agent",
    )
    mh.report_event(
        title="Deployment finished",
        summary="Deployed authentication service v2.1",
        event_type="task_completed",
        produced_by="deploy-agent",
    )
    claims = mh.query_claims(scope_id="myproject", limit=10)
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_BASE = "http://localhost:7337"
_TIMEOUT = 10.0


class MultiHeadClient:
    """Thin HTTP client for MultiHead's REST API."""

    def __init__(self, base_url: str = _DEFAULT_BASE, timeout: float = _TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _post(self, path: str, json: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as c:
            r = c.post(self._url(path), json=json)
            r.raise_for_status()
            return r.json()

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        with httpx.Client(timeout=self.timeout) as c:
            r = c.get(self._url(path), params=params)
            r.raise_for_status()
            return r.json()

    # ------------------------------------------------------------------
    # Claims
    # ------------------------------------------------------------------

    def deposit_claim(
        self,
        claim_key: str,
        statement: str,
        *,
        subject_type: str = "component",
        subject_id: str = "",
        predicate: str = "has_state",
        value: Any = True,
        value_type: str = "string",
        claim_type: str = "fact",
        claim_status: str = "accepted",
        scope_type: str = "project",
        scope_id: str = "default",
        confidence: float = 0.9,
        stability: str = "medium",
        importance: float = 0.5,
        rationale: str = "",
        produced_by: str = "external",
    ) -> dict[str, Any]:
        """Deposit a claim into MultiHead's knowledge store."""
        body = {
            "claim_key": claim_key,
            "statement": statement,
            "subject_type": subject_type,
            "subject_id": subject_id or claim_key.split(".")[0],
            "predicate": predicate,
            "value": value,
            "value_type": value_type,
            "claim_type": claim_type,
            "claim_status": claim_status,
            "scope_type": scope_type,
            "scope_id": scope_id,
            "confidence": confidence,
            "stability": stability,
            "importance": importance,
            "rationale": rationale,
            "produced_by": produced_by,
        }
        return self._post("/knowledge/claims", body)

    def query_claims(
        self,
        *,
        status: str | None = None,
        claim_type: str | None = None,
        scope_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query claims from the knowledge store."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        if claim_type:
            params["claim_type"] = claim_type
        if scope_id:
            params["scope_id"] = scope_id
        return self._get("/knowledge/claims", params)

    def get_claim(self, claim_id: str) -> dict[str, Any]:
        """Get a specific claim by ID."""
        return self._get(f"/knowledge/claims/{claim_id}")

    # ------------------------------------------------------------------
    # Briefing — "what should I know?"
    # ------------------------------------------------------------------

    def get_briefing(
        self,
        component: str,
        *,
        scope_id: str = "default",
        include_events: bool = True,
        max_claims: int = 20,
        max_events: int = 10,
    ) -> dict[str, Any]:
        """Get a briefing for a component — everything it needs to know.

        Returns direct claims (key matches component), related claims
        (statement mentions component), and recent events.
        """
        params: dict[str, Any] = {
            "component": component,
            "scope_id": scope_id,
            "include_events": include_events,
            "max_claims": max_claims,
            "max_events": max_events,
        }
        return self._get("/knowledge/briefing", params)

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def report_event(
        self,
        title: str,
        *,
        summary: str = "",
        event_type: str = "note",
        event_status: str = "confirmed",
        tags: list[str] | None = None,
        metrics: dict[str, float] | None = None,
        produced_by: str = "external",
        entities: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Report a knowledge event to MultiHead."""
        body: dict[str, Any] = {
            "title": title,
            "summary": summary,
            "event_type": event_type,
            "event_status": event_status,
            "tags": tags or [],
            "metrics": metrics or {},
            "produced_by": produced_by,
            "entities": entities or [],
        }
        return self._post("/knowledge/events", body)

    def query_events(
        self,
        *,
        status: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query knowledge events."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        if event_type:
            params["event_type"] = event_type
        return self._get("/knowledge/events", params)

    # ------------------------------------------------------------------
    # Chat / Generate (for components that need LLM assistance)
    # ------------------------------------------------------------------

    def chat(self, message: str, *, head_id: str = "") -> dict[str, Any]:
        """Send a chat message to MultiHead's agentic core."""
        body: dict[str, Any] = {"message": message}
        if head_id:
            body["head_id"] = head_id
        return self._post("/chat", body)

    def generate(self, prompt: str, *, head_id: str = "") -> dict[str, Any]:
        """Raw generation from a specific head."""
        body: dict[str, Any] = {"prompt": prompt}
        if head_id:
            body["head_id"] = head_id
        return self._post("/generate", body)

    # ------------------------------------------------------------------
    # Decompose
    # ------------------------------------------------------------------

    def decompose(
        self,
        goal: str,
        *,
        context: str = "",
        head_id: str = "",
        max_depth: int = 4,
    ) -> dict[str, Any]:
        """Decompose a goal into a hierarchical execution plan."""
        body: dict[str, Any] = {
            "goal": goal,
            "context": context,
            "max_depth": max_depth,
        }
        if head_id:
            body["head_id"] = head_id
        return self._post("/decompose", body)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Check if MultiHead server is reachable."""
        try:
            self._get("/heads")
            return True
        except Exception:
            return False
