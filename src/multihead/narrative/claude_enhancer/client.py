"""ACP/HTTP client for communicating with Claude worker daemons."""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from typing import Any

import httpx

from .constants import _ACP_TIMEOUT, logger


class ACPClient:
    """HTTP client for ACP task management and MultiHead chat API."""

    def __init__(
        self,
        acp_url: str,
        api_key: str,
        acp_project_id: str,
        poll_interval: float = 10.0,
        max_wait: float = 600.0,
    ):
        self.acp_url = acp_url
        self.api_key = api_key
        self.acp_project_id = acp_project_id
        self.poll_interval = poll_interval
        self.max_wait = max_wait

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def create_task(self, prompt: str) -> tuple[str, str]:
        """Call MultiHead's local LLM directly via REST API. Returns (task_id, response_text)."""
        # Call MultiHead serve API at localhost:7337 instead of ACP
        multihead_url = os.environ.get("MULTIHEAD_URL", "http://localhost:7337")
        async with httpx.AsyncClient(timeout=600.0) as client:  # 10min timeout for local LLM
            resp = await client.post(
                f"{multihead_url}/chat",
                json={
                    "message": prompt,
                },
            )
            if resp.status_code >= 400:
                logger.error("Chat request failed %d: %s", resp.status_code, resp.text[:300])
            resp.raise_for_status()
            data = resp.json()
            # Generate fake task_id for compatibility with existing code
            task_id = hashlib.sha256(prompt[:100].encode()).hexdigest()[:16]
            response = data.get("response", "")
            logger.debug("Got response for task %s (%d chars)", task_id[:8], len(response))
            return task_id, response

    async def get_task_status(self, task_id: str) -> dict[str, Any]:
        """Get current status of an ACP task."""
        async with httpx.AsyncClient(timeout=_ACP_TIMEOUT) as client:
            resp = await client.get(
                f"{self.acp_url}/tasks/{task_id}",
                headers=self._headers(),
            )
            resp.raise_for_status()
            return resp.json()

    async def poll_all_tasks(
        self,
        task_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Poll all tasks until complete or timed out.

        Returns {task_id: task_detail} for completed tasks.
        """
        completed: dict[str, dict[str, Any]] = {}
        pending = set(task_ids)
        start = time.monotonic()

        while pending and (time.monotonic() - start) < self.max_wait:
            for task_id in list(pending):
                try:
                    status = await self.get_task_status(task_id)
                    state = status.get("state", "unknown")

                    if state in ("complete", "failed"):
                        completed[task_id] = {
                            "status": state,
                            "output_ref": status.get("output_ref", ""),
                        }
                        pending.discard(task_id)
                        logger.info(
                            "Task %s %s (%d/%d done)",
                            task_id[:8], state,
                            len(completed), len(task_ids),
                        )
                except Exception as e:
                    logger.warning("Poll error for %s: %s", task_id[:8], e)

            if pending:
                elapsed = time.monotonic() - start
                logger.info(
                    "Waiting... %d/%d complete (%.0fs elapsed)",
                    len(completed), len(task_ids), elapsed,
                )
                await asyncio.sleep(self.poll_interval)

        if pending:
            logger.warning(
                "Timed out waiting for %d tasks: %s",
                len(pending),
                [tid[:8] for tid in pending],
            )

        return completed
