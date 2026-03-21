"""BotVibes Adapter: Execute tasks on BotVibes marketplace providers.

This adapter allows MultiHead to delegate work to external agents on the
BotVibes marketplace. Two execution modes:

1. **Simple task mode** (default) -- Creates an ACP task, polls for result.
   Good for quick tasks where the provider is already known.

2. **RFQ procurement mode** -- Full marketplace cycle:
   RFQ -> collect bids -> score -> award contract -> escrow -> wait for delivery
   -> verify quality -> release payment.
   Activated when ``budget`` is passed in kwargs.

Privacy enforcement: blocks CONFIDENTIAL/RESTRICTED data from leaving local.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from ...models import DataSensitivity, HeadManifest, PrivacyConstraint
from ..base import HeadAdapter
from ._constants import (
    _ACP_TIMEOUT,
    _MAX_WAIT_TIME_S,
    _TASK_POLL_INTERVAL_S,
    PrivacyViolation,
)
from ._http import HttpMixin
from ._rfq import RfqMixin

logger = logging.getLogger(__name__)


class BotVibesAdapter(RfqMixin, HttpMixin, HeadAdapter):
    """Adapter for delegating tasks to BotVibes marketplace providers.

    Supports two modes:
    - Simple ACP task creation (default)
    - Full RFQ procurement cycle (when budget is provided)

    The manifest should include:
    - endpoint: BotVibes server URL
    - extra.api_key: Authentication token
    - extra.project_id: ACP project ID
    - extra.target_capability: Required capability (e.g., "visual_reasoning")
    - extra.target_agent_id: (optional) Specific agent to target
    - extra.use_rfq: (optional) Force RFQ mode even without budget
    """

    def __init__(
        self,
        manifest: HeadManifest,
        knowledge_store: Any = None,
    ) -> None:
        super().__init__(manifest)

        # Extract ACP configuration from manifest
        self.acp_url = manifest.endpoint or ""
        self.acp_token = manifest.extra.get("api_key", "")
        self.project_id = manifest.extra.get("project_id", "")
        self.target_capability = manifest.extra.get("target_capability", "")
        self.target_agent_id = manifest.extra.get("target_agent_id")
        self._use_rfq = manifest.extra.get("use_rfq", False)
        self._knowledge_store = knowledge_store

        # Cost tracking
        self.total_cost: float = 0.0
        self.call_count: int = 0

        # Token refresh state
        self._login_email = manifest.extra.get("login_email", "")
        self._login_password = manifest.extra.get("login_password", "")

        if not self.acp_url:
            raise ValueError(f"BotVibes adapter {manifest.head_id} missing acp_url (endpoint)")
        if not self.acp_token:
            raise ValueError(f"BotVibes adapter {manifest.head_id} missing acp_token (extra.api_key)")
        if not self.target_capability:
            raise ValueError(f"BotVibes adapter {manifest.head_id} missing target_capability (extra.target_capability)")

        # Normalize URL to end with /api/v1
        self.acp_url = self.acp_url.rstrip("/")
        for suffix in ("/api/v1", "/api/v1/", "/api"):
            if self.acp_url.endswith(suffix):
                self.acp_url = self.acp_url[: -len(suffix)]
                break
        self.acp_url += "/api/v1"

    # ------------------------------------------------------------------
    # Privacy enforcement
    # ------------------------------------------------------------------

    def _enforce_privacy(self, privacy: PrivacyConstraint | None) -> None:
        """Block data that must not leave the local system.

        Defense-in-depth: the router already filters by privacy, but the
        adapter enforces it too in case of misconfiguration.
        """
        if not privacy:
            return
        if privacy.data_sensitivity == DataSensitivity.RESTRICTED:
            raise PrivacyViolation(
                "RESTRICTED data requires human approval and cannot be sent to marketplace"
            )
        if privacy.data_sensitivity == DataSensitivity.CONFIDENTIAL:
            raise PrivacyViolation(
                "CONFIDENTIAL data cannot leave the local system"
            )

    # ------------------------------------------------------------------
    # Lifecycle (no-ops for remote service)
    # ------------------------------------------------------------------

    async def load(self) -> None:
        """BotVibes providers are always 'loaded' (remote service)."""
        logger.debug("BotVibes adapter %s: load() (no-op for remote)", self.manifest.head_id)

    async def unload(self) -> None:
        """BotVibes providers don't need unloading."""
        logger.debug("BotVibes adapter %s: unload() (no-op for remote)", self.manifest.head_id)

    async def sleep(self, level: int = 1) -> None:
        """No-op for remote providers."""

    async def wake(self) -> None:
        """No-op for remote providers."""

    # ------------------------------------------------------------------
    # Main generate: routes to simple or RFQ mode
    # ------------------------------------------------------------------

    async def generate(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Execute task on BotVibes provider and wait for result.

        Routes to RFQ procurement mode when budget is provided or
        ``use_rfq`` is set in the manifest.

        Args:
            prompt: Task prompt/input
            **kwargs: Additional parameters:
                - timeout_s: Max wait time (default 300s)
                - budget: BudgetConstraint triggers RFQ mode
                - privacy: PrivacyConstraint for data protection
                - conversation_id: Thread ACP tasks together
                - bid_wait_s: How long to collect bids (RFQ mode)

        Returns:
            Dict with text, tokens_in, tokens_out, latency_ms, provider_metadata
        """
        # Privacy enforcement (defense-in-depth)
        privacy = kwargs.get("privacy")
        self._enforce_privacy(privacy)

        budget = kwargs.get("budget")
        use_rfq = self._use_rfq or budget is not None

        if use_rfq:
            return await self._generate_rfq(prompt, **kwargs)
        return await self._generate_simple(prompt, **kwargs)

    # ------------------------------------------------------------------
    # Mode 1: Simple ACP task (existing behavior, enhanced with retry)
    # ------------------------------------------------------------------

    async def _generate_simple(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Create an ACP task, poll for result, return it."""
        start_time = time.time()
        timeout_s = kwargs.get("timeout_s", _MAX_WAIT_TIME_S)

        async with httpx.AsyncClient(timeout=_ACP_TIMEOUT) as client:
            # Create task
            task_id = await self._create_task(client, prompt, kwargs)
            logger.info(
                "BotVibes adapter %s: Created task %s (capability=%s)",
                self.manifest.head_id, task_id, self.target_capability,
            )

            # Poll for completion
            result = await self._wait_for_result(client, task_id, timeout_s)

        latency_ms = int((time.time() - start_time) * 1000)
        output_text = result.get("output_ref", "")
        tokens_in = len(prompt) // 4
        tokens_out = len(output_text) // 4

        # Track cost
        cost = self.manifest.capabilities.cost_per_call if self.manifest.capabilities else 0.0
        self.total_cost += cost or 0.0
        self.call_count += 1

        self._record_to_knowledge(True, latency_ms, cost or 0.0)

        return {
            "text": output_text,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": latency_ms,
            "provider_metadata": {
                "task_id": task_id,
                "confidence": result.get("confidence", 0.0),
                "provider_latency_ms": result.get("latency_ms", 0),
                "mode": "simple",
            },
        }

    # ------------------------------------------------------------------
    # Simple task helpers
    # ------------------------------------------------------------------

    async def _create_task(
        self, client: httpx.AsyncClient, prompt: str, kwargs: dict[str, Any]
    ) -> str:
        """Create an ACP task and return its ID."""
        payload: dict[str, Any] = {
            "project_id": self.project_id,
            "required_capability": self.target_capability,
            "payload_ref": prompt,
            "priority": kwargs.get("priority", "normal"),
            "input_schema": "application/json",
            "output_schema": "application/json",
        }

        if self.target_agent_id:
            payload["target_agent_id"] = self.target_agent_id
        if "conversation_id" in kwargs:
            payload["conversation_id"] = kwargs["conversation_id"]

        resp = await self._request_with_retry(
            client, "POST", f"{self.acp_url}/tasks",
            headers=self._auth_headers(), json=payload,
        )
        resp.raise_for_status()
        result = resp.json()
        return result["task_id"]

    async def _wait_for_result(
        self, client: httpx.AsyncClient, task_id: str, timeout_s: float
    ) -> dict[str, Any]:
        """Poll for task completion and return result."""
        import asyncio

        start = time.time()

        while time.time() - start < timeout_s:
            resp = await self._request_with_retry(
                client, "GET", f"{self.acp_url}/tasks/{task_id}",
                headers=self._auth_headers(),
            )

            if resp.status_code != 200:
                logger.warning("Task %s status check failed: %d", task_id, resp.status_code)
                await asyncio.sleep(_TASK_POLL_INTERVAL_S)
                continue

            task = resp.json()
            status = task.get("status", "pending")

            if status == "complete":
                logger.info("Task %s completed successfully", task_id)
                return task
            elif status == "failed":
                error_msg = task.get("message", "Unknown error")
                raise RuntimeError(f"BotVibes task failed: {error_msg}")
            elif status in ("created", "reserved", "dispatched"):
                await asyncio.sleep(_TASK_POLL_INTERVAL_S)
            else:
                logger.warning("Task %s unknown status: %s", task_id, status)
                await asyncio.sleep(_TASK_POLL_INTERVAL_S)

        raise TimeoutError(f"Task {task_id} did not complete within {timeout_s}s")

    # ------------------------------------------------------------------
    # Knowledge store feedback
    # ------------------------------------------------------------------

    def _record_to_knowledge(self, success: bool, latency_ms: int, cost: float) -> None:
        """Record performance to knowledge store for router learning."""
        if not self._knowledge_store:
            return
        try:
            from ...knowledge_models import (
                Claim, ClaimCanonical, ClaimScope, ClaimType,
                EntityRef, Provenance, ScopeType, ValueObject,
            )
            from datetime import datetime, timezone

            status_str = "success" if success else "failure"
            claim = Claim(
                claim_type=ClaimType.FACT,
                scope=ClaimScope(scope_type=ScopeType.PROJECT, scope_id="multihead"),
                canonical=ClaimCanonical(
                    claim_key=f"botvibes.adapter.{self.manifest.head_id}.{int(time.time())}",
                    subject=EntityRef(entity_type="head", entity_id=self.manifest.head_id),
                    predicate=f"marketplace_{status_str}",
                    object=ValueObject(
                        value_type="performance",
                        value=json.dumps({
                            "latency_ms": latency_ms,
                            "cost": cost,
                            "capability": self.target_capability,
                        }),
                    ),
                ),
                statement=f"BotVibes adapter {self.manifest.head_id}: {status_str} ({latency_ms}ms, ${cost:.2f})",
                confidence=1.0,
                importance=0.3,
                provenance=Provenance(
                    produced_by={"id": "botvibes-adapter"},
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                ),
            )
            self._knowledge_store.insert_claim(claim)
        except Exception as e:
            logger.debug("Knowledge store recording failed: %s", e)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def healthcheck(self) -> bool:
        """Check if BotVibes API is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{self.acp_url}/health",
                    headers=self._auth_headers(),
                )
                return resp.status_code == 200
        except Exception as e:
            logger.warning("BotVibes healthcheck failed: %s", e)
            return False
