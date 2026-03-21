"""Events and collaboration command handlers.

/events, /collab, /collab-respond, /collab-ignore
"""

from __future__ import annotations

import time


class EventsCollabMixin:
    """Mixin providing /events and /collab command handlers."""

    # -------------------------------------------------------------------
    # /events
    # -------------------------------------------------------------------

    async def _handle_events(self, args: list[str]) -> str:
        """Show, clear, or handle events from the event watcher."""
        shell = self.shell
        if not shell:
            return "Shell reference not available."

        watcher = getattr(shell, "event_watcher", None)
        if not watcher:
            return "Event watcher not configured."

        if not args or args[0] == "show":
            history = watcher.history
            if not history:
                return "No events recorded. Watcher is listening for incoming work."
            lines = [f"Events ({len(history)} recorded, {watcher.pending_count} pending):"]
            for i, evt in enumerate(history[-15:], 1):
                age = int(time.time() - evt.timestamp)
                if age < 60:
                    age_str = f"{age}s ago"
                elif age < 3600:
                    age_str = f"{age // 60}m ago"
                else:
                    age_str = f"{age // 3600}h ago"
                lines.append(
                    f"  {i}. [{evt.source}] {evt.event_type}: "
                    f"{evt.summary[:80]} ({age_str})"
                )
            return "\n".join(lines)

        if args[0] == "clear":
            watcher.clear_history()
            return "Event history cleared."

        if args[0] == "handle" and len(args) >= 2:
            try:
                idx = int(args[1]) - 1
            except ValueError:
                return "Usage: /events handle <number>"
            history = watcher.history
            if idx < 0 or idx >= len(history[-15:]):
                return f"Invalid event number. Valid range: 1-{min(len(history), 15)}"
            event = history[-15:][idx]
            # Route to brain via shell's auto-handle
            from ..event_watcher import ShellEvent
            session_id = getattr(shell, "_current_session_id", None)
            if not session_id:
                # Fallback — get from slash handler
                session_id = self.session_id or "default"
            await shell._handle_event_auto(event, session_id)
            return ""  # Response already displayed by _handle_event_auto

        return "Usage: /events [show|clear|handle <N>]"

    # -------------------------------------------------------------------
    # /collab commands
    # -------------------------------------------------------------------

    async def _handle_collab(self, args: list[str]) -> str:
        """Handle /collab - list pending requests."""
        if not self.knowledge_store or not self.head_manager or not self.session_id:
            return "Collaboration commands require knowledge store integration."

        from .. import auto_responder

        return await auto_responder.handle_collab_command(
            self.knowledge_store,
            self.session_id,
            self.project_id,
            self.head_manager,
        )

    async def _handle_collab_respond(self, args: list[str]) -> str:
        """Handle /collab-respond <number> - respond to a request."""
        if not args:
            return "Usage: /collab-respond <request_id_prefix>"

        if not self.knowledge_store or not self.head_manager or not self.session_id:
            return "Collaboration commands require knowledge store integration."

        from .. import auto_responder

        request_id_prefix = args[0]
        return await auto_responder.respond_to_request(
            request_id_prefix,
            self.knowledge_store,
            self.head_manager,
            self.session_id,
            self.project_id,
        )

    async def _handle_collab_ignore(self, args: list[str]) -> str:
        """Handle /collab-ignore <number> - ignore a request."""
        if not args:
            return "Usage: /collab-ignore <request_id_prefix>"

        from .. import auto_responder

        request_id_prefix = args[0]
        return auto_responder.ignore_request(request_id_prefix)
