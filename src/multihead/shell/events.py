"""Event watcher mixin — drain and display incoming events."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ..event_watcher import ShellEvent
from .prompts import BRAIN_CLAUDE

logger = logging.getLogger(__name__)


class EventsMixin:
    """Mixin providing event-handling methods for the Shell class.

    Expects the host class to provide:
    - self.event_watcher (EventWatcher | None)
    - self.config (RuntimeConfig)
    - self._brain (str)
    - self._shutting_down (bool)
    - self._current_session_id (str)
    - self._tui_print(...)
    - self._display_response(response)
    - self._chat_via_claude(session_id, prompt) -> str
    - self._chat_via_local(session_id, prompt) -> str
    """

    async def _drain_events_loop(self) -> None:
        """Background task: drain events every 1s and display them."""
        while True:
            try:
                await self._drain_events(self._current_session_id)
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.debug("Event drain error: %s", e)
            await asyncio.sleep(1.0)

    async def _drain_events(self, session_id: str) -> None:
        """Pull pending events from watcher and display/handle them.

        Stale events (>60s old) are batch-acknowledged in a single summary
        instead of being routed to the brain one-by-one, which prevents
        backlog floods from overwhelming the TUI.
        """
        if not self.event_watcher or self._shutting_down:
            return

        events = self.event_watcher.get_pending()
        if not events:
            return

        ew_cfg = getattr(getattr(self.config, "pipeline", None), "event_watcher", None)
        auto_handle = ew_cfg and getattr(ew_cfg, "auto_handle", False)

        # Partition into fresh vs stale events
        _STALE_THRESHOLD_S = 60
        now = time.time()
        fresh: list[ShellEvent] = []
        stale: list[ShellEvent] = []
        for event in events:
            if now - event.timestamp > _STALE_THRESHOLD_S:
                stale.append(event)
            else:
                fresh.append(event)

        # Batch-acknowledge stale events in a compact summary
        if stale:
            # Group by source for concise display
            by_source: dict[str, list[ShellEvent]] = {}
            for ev in stale:
                by_source.setdefault(ev.source, []).append(ev)

            parts: list[str] = []
            for src, evts in by_source.items():
                types = {}
                for e in evts:
                    types[e.event_type] = types.get(e.event_type, 0) + 1
                breakdown = ", ".join(f"{n}x {t}" for t, n in types.items())
                parts.append(f"{src}: {breakdown}")

            age_min = min(now - e.timestamp for e in stale)
            age_max = max(now - e.timestamp for e in stale)
            age_range = (
                f"{int(age_min)}s" if age_max - age_min < 5
                else f"{int(age_min)}-{int(age_max)}s"
            )
            self._tui_print(
                f"[dim yellow]  Batch-ack {len(stale)} stale events "
                f"({age_range} old): {'; '.join(parts)}[/dim yellow]"
            )

        # Process fresh events normally
        for event in fresh:
            source_color = {
                "acp": "cyan",
                "knowledge": "magenta",
                "service": "yellow",
            }.get(event.source, "white")
            self._tui_print(
                f"[dim {source_color}]  [{event.source}] "
                f"{event.event_type}: {event.summary}[/dim {source_color}]"
            )

            # Auto-handle if enabled and event supports it
            # Skip monitoring noise (marketplace-activity-loop) — display only
            if auto_handle and event.auto_actionable:
                requester = event.detail.get("requester", "")
                # Skip noise — these don't need LLM processing
                if requester == "marketplace-activity-loop":
                    continue
                # Edit tracking is just a notification for nightshift — don't burn brain cycles
                if event.event_type == "collab_request" and "post-edit-track" in event.summary:
                    self._tui_print(f"[dim]  edit tracked: {event.summary.split('File ')[-1].split(' was')[0] if 'File ' in event.summary else event.summary[:60]}[/dim]")
                    continue
                await self._handle_event_auto(event, session_id)

        if fresh and not auto_handle:
            self._tui_print(
                "[dim]  Type /events to review, or /events handle <N> to act.[/dim]"
            )

    async def _handle_event_auto(self, event: ShellEvent, session_id: str) -> None:
        """Auto-route an event to the brain."""
        if self._shutting_down:
            return
        self._tui_print("[dim yellow]  auto-handling...[/dim yellow]")
        prompt = (
            f"[Incoming {event.source} event: {event.event_type}]\n"
            f"{event.summary}\n\n"
            f"Details: {event.detail}\n\n"
            "Please review and take appropriate action."
        )
        try:
            async with self._brain_lock:
                if self._brain == BRAIN_CLAUDE:
                    response = await self._chat_via_claude(session_id, prompt)
                else:
                    response = await self._chat_via_local(session_id, prompt)
            self._display_response(response)
        except Exception as e:
            self._tui_print(f"[red]  Auto-handle failed: {e}[/red]")
