"""Shell event watcher — background poller for incoming work.

Polls multiple event sources (ACP/BotVibes tasks, knowledge.db inbox)
and enqueues ShellEvent notifications for the shell REPL to display.
The watcher never executes anything — it only detects and surfaces events.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Suppress noisy httpx request logging during background polling
logging.getLogger("httpx").setLevel(logging.WARNING)

# Hard timeout per source check (never block the loop)
_CHECK_TIMEOUT = 5.0


@dataclass
class ShellEvent:
    """An event surfaced to the shell terminal."""

    source: str          # "acp", "knowledge", "service"
    event_type: str      # "task_available", "collab_request", "service_alert"
    summary: str         # Human-readable one-liner
    detail: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    auto_actionable: bool = False
    event_id: str = ""   # For dedup (task_id or claim_id)


class EventWatcher:
    """Background poller that detects incoming events for the shell.

    Checks ACP/BotVibes for tasks and knowledge.db for collaboration
    requests on a configurable interval. Events are enqueued and
    drained by the shell REPL between prompts.

    Does NOT execute anything — only detects and enqueues.
    """

    def __init__(
        self,
        knowledge_store: Any | None = None,
        session_id: str = "",
        project_id: str = "multihead",
        poll_interval: int = 15,
        watch_acp: bool = True,
        watch_knowledge: bool = True,
        agent_id: str = "",
    ) -> None:
        self._ks = knowledge_store
        self._session_id = session_id
        self._agent_id = agent_id or session_id  # stable identity for dedup
        self._project_id = project_id
        self._poll_interval = poll_interval
        self._watch_acp = watch_acp
        self._watch_knowledge = watch_knowledge

        self._queue: asyncio.Queue[ShellEvent] = asyncio.Queue()
        self._seen: set[str] = set()  # Dedup by event_id
        self._history: list[ShellEvent] = []  # Recent events for /events
        self._max_history = 50
        self._running = False

        # ACP backoff state — avoid spamming logs when local ACP is down
        self._acp_consecutive_failures = 0
        self._acp_skip_until = 0.0  # epoch time; skip checks until this time

    def _seed_seen_from_db(self) -> None:
        """Pre-seed _seen set from claim_interactions table.

        Loads all claim IDs that this agent has already read/responded/dismissed,
        so they won't resurface across shell restarts.
        """
        if not self._ks or not self._agent_id:
            return
        try:
            db_path = getattr(self._ks, "db_path", None)
            if not db_path:
                return
            import sqlite3
            with sqlite3.connect(str(db_path), timeout=5.0) as conn:
                rows = conn.execute(
                    "SELECT claim_id FROM claim_interactions WHERE agent_id = ?",
                    (self._agent_id,),
                ).fetchall()
                for (cid,) in rows:
                    self._seen.add(cid)
            if rows:
                logger.info("EventWatcher seeded %d already-seen claim IDs from DB", len(rows))
        except Exception as e:
            logger.debug("Failed to seed seen set from DB: %s", e)

    async def run(self) -> None:
        """Main poll loop — checks all sources, enqueues events."""
        self._running = True
        self._seed_seen_from_db()
        logger.info(
            "EventWatcher started (interval=%ds, acp=%s, knowledge=%s, seen=%d)",
            self._poll_interval, self._watch_acp, self._watch_knowledge, len(self._seen),
        )
        try:
            while self._running:
                if self._watch_knowledge:
                    await self._check_knowledge_inbox()
                if self._watch_acp:
                    await self._check_acp_tasks()
                await asyncio.sleep(self._poll_interval)
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            logger.info("EventWatcher stopped")

    async def stop(self) -> None:
        """Signal the watcher to stop."""
        self._running = False

    def _enqueue(self, event: ShellEvent) -> bool:
        """Add event to queue if not already seen. Returns True if new."""
        if event.event_id and event.event_id in self._seen:
            return False
        if event.event_id:
            self._seen.add(event.event_id)
        self._queue.put_nowait(event)
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
        return True

    def get_pending(self) -> list[ShellEvent]:
        """Drain queue (non-blocking). Called by shell between prompts."""
        events: list[ShellEvent] = []
        while not self._queue.empty():
            try:
                events.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return events

    @property
    def history(self) -> list[ShellEvent]:
        """Recent event history for /events command."""
        return list(self._history)

    def clear_history(self) -> None:
        """Clear event history."""
        self._history.clear()

    @property
    def pending_count(self) -> int:
        """Number of events waiting in queue."""
        return self._queue.qsize()

    # ------------------------------------------------------------------
    # Knowledge.db inbox
    # ------------------------------------------------------------------

    async def _check_knowledge_inbox(self) -> None:
        """Check knowledge.db for pending collaboration requests.

        Uses stable agent_id for session filtering and skips self-produced claims.
        """
        if not self._ks:
            return
        # Import self-identity filter
        from .shell_pipeline import SELF_IDENTITIES
        try:
            # Use get_unhandled_claims (checks claim_interactions table)
            # instead of get_pending_messages (which ignores read history)
            _fetch = getattr(self._ks, "get_unhandled_claims", None)
            if _fetch:
                pending = await asyncio.wait_for(
                    asyncio.to_thread(
                        _fetch,
                        agent_id=self._agent_id,
                        scope_id=self._project_id,
                        max_age_hours=24,
                        limit=20,
                    ),
                    timeout=_CHECK_TIMEOUT,
                )
            else:
                # Fallback for older knowledge store versions
                pending = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._ks.get_pending_messages,
                        session_id=self._agent_id,
                        scope_id=self._project_id,
                        max_age_hours=24,
                        limit=20,
                    ),
                    timeout=_CHECK_TIMEOUT,
                )
            for claim in pending:
                claim_id = getattr(claim, "claim_id", "")
                requester = "unknown"
                prov = getattr(claim, "provenance", None)
                if prov:
                    produced_by = getattr(prov, "produced_by", {})
                    if isinstance(produced_by, dict):
                        requester = produced_by.get("id", "unknown")
                # Skip self-produced claims
                if requester in SELF_IDENTITIES:
                    continue
                statement = getattr(claim, "statement", "")[:150]
                claim_type = getattr(
                    getattr(claim, "canonical", None), "claim_type", "request",
                )
                # Use claim's original timestamp so stale detection works
                claim_ts = time.time()
                created_at = getattr(claim, "created_at", None)
                if created_at is not None:
                    try:
                        claim_ts = created_at.timestamp()
                    except Exception:
                        pass  # Keep default

                enqueued = self._enqueue(ShellEvent(
                    source="knowledge",
                    event_type="collab_request",
                    summary=f"from {requester}: {statement}",
                    detail={
                        "claim_id": claim_id,
                        "requester": requester,
                        "statement": getattr(claim, "statement", ""),
                        "claim_type": claim_type,
                    },
                    auto_actionable=True,
                    event_id=claim_id,
                    timestamp=claim_ts,
                ))
                # Mark as read in DB so it won't resurface after restart
                if enqueued and claim_id and hasattr(self._ks, "record_interaction"):
                    try:
                        self._ks.record_interaction(
                            claim_id, self._agent_id, "read",
                        )
                    except Exception:
                        pass  # Duplicate is fine
        except asyncio.TimeoutError:
            logger.warning("Knowledge inbox check timed out (%.1fs)", _CHECK_TIMEOUT)
        except Exception as e:
            logger.warning("Knowledge inbox check failed: %s", e)

    # ------------------------------------------------------------------
    # ACP / BotVibes tasks
    # ------------------------------------------------------------------

    async def _check_acp_tasks(self) -> None:
        """Check BotVibes for available tasks targeting our capabilities.

        Uses exponential backoff when the local ACP server is unreachable
        to avoid spamming logs every poll interval.
        """
        # Skip if we're in backoff period
        if time.time() < self._acp_skip_until:
            return

        try:
            import httpx
        except ImportError:
            return

        acp_url = self._get_acp_url()
        api_key = self._get_acp_key()
        if not acp_url or not api_key:
            return

        try:
            async with httpx.AsyncClient(timeout=_CHECK_TIMEOUT) as client:
                resp = await client.get(
                    f"{acp_url}/tasks/available",
                    headers={"Authorization": f"Bearer {api_key}"},
                    params={"capability": "com.multihead", "limit": 10},
                )
                if resp.status_code != 200:
                    return

                # Success — reset backoff
                if self._acp_consecutive_failures > 0:
                    logger.info("ACP connection restored after %d failures", self._acp_consecutive_failures)
                self._acp_consecutive_failures = 0
                self._acp_skip_until = 0.0

                tasks = resp.json()
                for task in tasks:
                    task_id = task.get("task_id", "")
                    payload = task.get("payload_ref", "")[:150]
                    source_agent = task.get("source_agent_id", "unknown")
                    capability = task.get("capability", "")
                    # Use task's original timestamp if available
                    task_ts = time.time()
                    created = task.get("created_at")
                    if created:
                        try:
                            from datetime import datetime, timezone
                            if isinstance(created, str):
                                task_ts = datetime.fromisoformat(created).timestamp()
                            elif isinstance(created, (int, float)):
                                task_ts = float(created)
                        except Exception:
                            pass
                    self._enqueue(ShellEvent(
                        source="acp",
                        event_type="task_available",
                        summary=f"from {source_agent}: {payload}",
                        detail={
                            "task_id": task_id,
                            "payload_ref": task.get("payload_ref", ""),
                            "source_agent": source_agent,
                            "capability": capability,
                            "priority": task.get("priority", "normal"),
                        },
                        auto_actionable=True,
                        event_id=task_id,
                        timestamp=task_ts,
                    ))
        except asyncio.TimeoutError:
            self._acp_backoff("timed out (%.1fs)" % _CHECK_TIMEOUT)
        except Exception as e:
            self._acp_backoff(str(e))

    def _acp_backoff(self, reason: str) -> None:
        """Apply exponential backoff after ACP check failure.

        First 2 failures: log at WARNING (so the user sees them once).
        After that: log at DEBUG and skip checks for increasing intervals
        (30s, 60s, 120s, capped at 5 minutes).
        """
        self._acp_consecutive_failures += 1
        n = self._acp_consecutive_failures

        if n <= 2:
            logger.warning("ACP task check failed: %s", reason)
        else:
            # Exponential backoff: 30s, 60s, 120s, ... capped at 300s
            delay = min(30 * (2 ** (n - 3)), 300)
            self._acp_skip_until = time.time() + delay
            logger.debug(
                "ACP task check failed (%d consecutive), backing off %ds: %s",
                n, delay, reason,
            )

    @staticmethod
    def _get_acp_url() -> str:
        """Read ACP URL from environment."""
        import os
        return os.environ.get("ACP_URL", "")

    @staticmethod
    def _get_acp_key() -> str:
        """Read ACP API key from environment."""
        import os
        return os.environ.get("ACP_SESSION_KEY", "") or os.environ.get("ACP_API_KEY", "")
