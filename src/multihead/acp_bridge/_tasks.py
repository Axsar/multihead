"""ACP task polling, execution, and result delivery.

Mixin class providing task lifecycle methods for the ACPBridge.
"""

from __future__ import annotations

import asyncio
import time as _time
import uuid
from typing import Any

import httpx

from ..skill_loader import SKILL_CAPABILITY_PREFIX
from ._constants import ACP_TIMEOUT, TASK_POLL_INTERVAL_S, logger


class TasksMixin:
    """Task polling, execution, result delivery, and Claude task creation."""

    # These attributes are provided by ACPBridge.__init__
    heads: Any
    skills: Any
    acp_url: str | None
    _api_key: str | None
    _agent_id: str
    _claude_agent_id: str
    _project_id: str
    _auto_execute: bool
    _runtime_config: Any
    _agentic_core: Any
    _artifact_store: Any
    _knowledge_store: Any
    _on_task: Any
    _connected: bool
    _poll_now: asyncio.Event
    _auth_headers: Any  # method

    # ------------------------------------------------------------------
    # Task polling (MultiHead picks up work from BotVibes)
    # ------------------------------------------------------------------

    async def _task_poll_loop(self) -> None:
        """Poll BotVibes for tasks targeting MultiHead's capabilities.

        Polls on a regular interval, but also triggers immediately when
        the WebSocket doorbell signals a new task via _poll_now event.
        """
        while self._connected:
            self._poll_now.clear()
            try:
                await self._poll_and_execute()
            except Exception as e:
                logger.warning("ACP task poll failed: %s", e)
            # Wait for either the poll interval or a WebSocket doorbell signal
            try:
                await asyncio.wait_for(self._poll_now.wait(), timeout=TASK_POLL_INTERVAL_S)
                logger.debug("Doorbell triggered immediate poll")
            except asyncio.TimeoutError:
                pass

    async def _poll_and_execute(self) -> None:
        """Check for available tasks and execute them."""
        if not self._agentic_core:
            return  # Not ready yet

        # BotVibes supports prefix matching (LIKE) on /tasks/available
        async with httpx.AsyncClient(timeout=ACP_TIMEOUT) as client:
            # Poll for head-based tasks
            resp = await client.get(
                f"{self.acp_url}/tasks/available",
                headers=self._auth_headers(),
                params={"capability": "com.multihead", "limit": 10},
            )
            tasks = resp.json() if resp.status_code == 200 else []

            # Also poll for skill-based tasks
            if len(self.skills) > 0:
                resp2 = await client.get(
                    f"{self.acp_url}/tasks/available",
                    headers=self._auth_headers(),
                    params={"capability": SKILL_CAPABILITY_PREFIX, "limit": 10},
                )
                if resp2.status_code == 200:
                    tasks.extend(resp2.json())

            seen = set()
            for task in tasks:
                tid = task.get("task_id")
                if tid and tid not in seen:
                    seen.add(tid)
                    await self._execute_task(client, task)

    async def _execute_task(self, client: httpx.AsyncClient, task: dict) -> None:
        """Reserve, execute, and submit result for a single ACP task.

        When ``_on_task`` callback is set (shell mode), the task is
        forwarded to the callback instead of being auto-executed. This
        lets the brain handle marketplace contracts with proper
        decomposition instead of firing a single junk LLM response.
        """
        task_id = task.get("task_id")
        if not task_id:
            return

        # --- Shell/callback mode: route to brain instead of auto-executing ---
        if self._on_task is not None:
            try:
                result = self._on_task(task)
                # Support both sync and async callbacks
                if asyncio.iscoroutine(result):
                    await result
                logger.info(
                    "ACP task %s routed to shell brain", task_id,
                )
            except Exception as e:
                logger.warning(
                    "ACP task %s callback failed: %s", task_id, e,
                )
            return

        # --- Check auto_execute: prefer runtime config (dynamic) over init flag ---
        auto_execute = self._auto_execute
        if self._runtime_config is not None:
            svc = getattr(self._runtime_config, "services", None)
            if svc is not None:
                auto_execute = getattr(svc, "acp_auto_execute", auto_execute)

        if not auto_execute:
            logger.info(
                "ACP task %s detected but auto_execute=False — "
                "use shell brain or /events to handle",
                task_id,
            )
            return

        # --- Headless mode: auto-execute via agentic core ---
        try:
            start = _time.monotonic()

            # Reserve
            resp = await client.post(
                f"{self.acp_url}/tasks/{task_id}/reserve",
                headers=self._auth_headers(),
                json={"agent_id": self._agent_id, "reservation_ttl_ms": 60000},
            )
            if resp.status_code != 200 or not resp.json().get("accepted"):
                return

            # Dispatch
            await client.post(
                f"{self.acp_url}/tasks/{task_id}/dispatch",
                headers=self._auth_headers(),
                json={"agent_id": self._agent_id},
            )

            # Execute — check if this is a skill-based task
            prompt = task.get("payload_ref", "")
            capability = task.get("required_capability", "")
            session = self._agentic_core.sessions.create_session()

            skill = self.skills.find_by_capability_prefix(capability) if capability else None
            if skill:
                logger.info(
                    "ACP task %s matched skill '%s' — executing with skill prompt",
                    task_id, skill.name,
                )
                skill_prompt = skill.build_system_prompt()
                response = await self._agentic_core.chat_with_skill(
                    session.session_id, prompt, skill_prompt,
                )
            else:
                # Default: generic agentic core chat
                response = await self._agentic_core.chat(session.session_id, prompt)

            latency_ms = int((_time.monotonic() - start) * 1000)
            confidence = 0.85 if skill else 0.8

            # --- Save result as artifact and knowledge claim ---
            artifact_id = await self._deliver_result(
                task_id=task_id,
                capability=capability,
                skill_name=skill.name if skill else None,
                response=response,
                confidence=confidence,
                latency_ms=latency_ms,
            )

            # Submit result to BotVibes
            result_payload: dict[str, Any] = {
                "status": "complete",
                "output_ref": response,
                "confidence": confidence,
                "latency_ms": latency_ms,
                "subtasks": [],
            }
            if skill:
                result_payload["skill"] = skill.name
            if artifact_id:
                result_payload["artifact_id"] = artifact_id

            await client.post(
                f"{self.acp_url}/tasks/{task_id}/result",
                headers=self._auth_headers(),
                json=result_payload,
            )
            logger.info(
                "ACP task %s completed successfully%s (artifact=%s)",
                task_id,
                f" (skill: {skill.name})" if skill else "",
                artifact_id or "none",
            )

        except Exception as e:
            logger.error("ACP task %s failed: %s", task_id, e)
            try:
                await client.post(
                    f"{self.acp_url}/tasks/{task_id}/result",
                    headers=self._auth_headers(),
                    json={
                        "status": "failed",
                        "error_code": "EXECUTION_ERROR",
                        "message": str(e),
                    },
                )
            except Exception:
                logger.error("Failed to report task failure for %s", task_id)

    # ------------------------------------------------------------------
    # Result delivery (artifact + knowledge claim)
    # ------------------------------------------------------------------

    async def _deliver_result(
        self,
        task_id: str,
        capability: str,
        skill_name: str | None,
        response: str,
        confidence: float,
        latency_ms: int,
    ) -> str | None:
        """Save task result as an artifact and create a knowledge claim.

        Returns the artifact_id if saved, or None if stores are unavailable.
        """
        artifact_id = None

        # --- Save artifact ---
        if self._artifact_store is not None:
            try:
                ref = self._artifact_store.store(
                    data=response.encode("utf-8") if isinstance(response, str) else response,
                    name=f"task_result_{task_id[:12]}",
                    media_type="text/plain",
                    annotations={
                        "type": "task_result",
                        "task_id": task_id,
                        "capability": capability,
                        "skill": skill_name,
                        "latency_ms": latency_ms,
                    },
                )
                artifact_id = ref.artifact_id
                logger.info("ACP task %s result saved as artifact %s", task_id, artifact_id)
            except Exception as e:
                logger.warning("Failed to save artifact for task %s: %s", task_id, e)

        # --- Create knowledge claim ---
        if self._knowledge_store is not None:
            try:
                from ..knowledge_models import (
                    Claim, ClaimType, ClaimStatus, ClaimScope, ClaimCanonical,
                    ScopeType, EntityRef, ValueObject, Provenance, Stability,
                )

                claim = Claim(
                    claim_id=f"clm_{uuid.uuid4().hex[:24].upper()}",
                    claim_type=ClaimType.FACT,
                    claim_status=ClaimStatus.ACCEPTED,
                    scope=ClaimScope(
                        scope_type=ScopeType.PROJECT,
                        scope_id=self._project_id,
                    ),
                    canonical=ClaimCanonical(
                        claim_key=f"acp.task.{task_id}",
                        subject=EntityRef(
                            entity_type="task",
                            entity_id=task_id,
                            label=f"ACP task {task_id[:8]}",
                        ),
                        predicate="completed_by",
                        object=ValueObject(
                            value_type="json",
                            value={
                                "agent_id": self._agent_id,
                                "skill": skill_name,
                                "capability": capability,
                                "latency_ms": latency_ms,
                                "artifact_id": artifact_id,
                            },
                        ),
                    ),
                    statement=f"ACP task {task_id[:8]} completed"
                              + (f" via skill '{skill_name}'" if skill_name else "")
                              + f" in {latency_ms}ms",
                    confidence=confidence,
                    stability=Stability.HIGH,
                    importance=0.6,
                    provenance=Provenance(
                        produced_by={"kind": "agent", "id": self._agent_id},
                    ),
                )
                self._knowledge_store.insert_claim(claim)
                logger.info("ACP task %s knowledge claim created: %s", task_id, claim.claim_id)
            except Exception as e:
                logger.warning("Failed to create knowledge claim for task %s: %s", task_id, e)

        return artifact_id

    # ------------------------------------------------------------------
    # Create task for Claude Code worker
    # ------------------------------------------------------------------

    async def create_task(
        self,
        capability: str,
        payload_ref: str,
        priority: str = "normal",
        conversation_id: str | None = None,
    ) -> str:
        """Create a generic ACP task for any capability.

        Returns the task_id string.
        """
        if not self.acp_url or not self._api_key:
            raise RuntimeError("ACP not configured")

        payload: dict[str, Any] = {
            "project_id": self._project_id,
            "required_capability": capability,
            "payload_ref": payload_ref,
            "priority": priority,
            "input_schema": "application/json",
            "output_schema": "application/json",
        }
        if conversation_id:
            payload["conversation_id"] = conversation_id

        async with httpx.AsyncClient(timeout=ACP_TIMEOUT) as client:
            resp = await client.post(
                f"{self.acp_url}/tasks",
                headers=self._auth_headers(),
                json=payload,
            )
            resp.raise_for_status()
            result = resp.json()
            task_id = result.get("task_id", "")
            logger.info("Created task (%s): %s", capability, task_id[:8])
            return task_id

    async def poll_for_completion(
        self,
        task_id: str,
        timeout_seconds: float = 300.0,
        poll_interval: float = 5.0,
    ) -> dict[str, Any] | None:
        """Poll an ACP task until it completes or times out."""
        if not self.acp_url or not self._api_key:
            raise RuntimeError("ACP not configured")

        deadline = _time.monotonic() + timeout_seconds
        async with httpx.AsyncClient(timeout=ACP_TIMEOUT) as client:
            while _time.monotonic() < deadline:
                resp = await client.get(
                    f"{self.acp_url}/tasks/{task_id}",
                    headers=self._auth_headers(),
                )
                if resp.status_code == 200:
                    data = resp.json()
                    status = data.get("status", "")
                    if status in ("complete", "completed", "failed", "error"):
                        return data
                await asyncio.sleep(poll_interval)
        logger.warning("Task %s timed out after %.0fs", task_id[:8], timeout_seconds)
        return None

    async def create_claude_task(
        self,
        prompt: str,
        conversation_id: str | None = None,
        priority: str = "normal",
    ) -> dict[str, Any]:
        """Create a task for claude-session-agent via ACP.

        The Claude Worker Daemon picks these up and spawns headless
        ``claude -p`` subprocesses to process them.
        """
        if not self.acp_url or not self._api_key:
            raise RuntimeError("ACP not configured")

        payload: dict[str, Any] = {
            "project_id": self._project_id,
            "required_capability": "com.claude.code",
            "payload_ref": prompt,
            "target_agent_id": self._claude_agent_id,
            "priority": priority,
            "input_schema": "application/json",
            "output_schema": "application/json",
        }
        if conversation_id:
            payload["conversation_id"] = conversation_id

        async with httpx.AsyncClient(timeout=ACP_TIMEOUT) as client:
            resp = await client.post(
                f"{self.acp_url}/tasks",
                headers=self._auth_headers(),
                json=payload,
            )
            resp.raise_for_status()
            result = resp.json()
            logger.info("Created Claude task: %s", result.get("task_id", "?")[:8])
            return result
