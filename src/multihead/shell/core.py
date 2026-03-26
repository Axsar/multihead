"""Shell core — the Shell class combining all mixins."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.widgets import TextArea
from rich.console import Console

from ..conversation_context import ConversationContext
from ..event_watcher import EventWatcher
from ..service_manager import ServiceManager
from ..shell_pipeline import AGENT_ID, SELF_IDENTITIES, ShellPipeline
from ..subprocess_utils import no_window_flags

from .brain import BrainMixin
from .context import ContextMixin
from .deprecated import DeprecatedMixin
from .display import DisplayMixin
from .events import EventsMixin
from .input import InputMixin
from .prompts import BRAIN_CLAUDE, BRAIN_DUAL, BRAIN_LOCAL
from .tui import _OutputPane, _TUILogHandler

logger = logging.getLogger(__name__)


class Shell(
    BrainMixin,
    DisplayMixin,
    ContextMixin,
    EventsMixin,
    InputMixin,
    DeprecatedMixin,
):
    """Interactive agent terminal with Rich UI and system management.

    Supports two brain modes:
    - **local**: AgenticCore drives the conversation using local GPU models
      (Qwen3-8B etc.) with structured action dispatch
    - **claude**: Claude Agent SDK drives the conversation with native tool
      use via in-process MCP tools — zero IPC overhead

    Switch mid-session with /brain claude or /brain local.

    UI is a full-screen split-pane TUI:
    - Top: scrollable output pane (responses, events, logs)
    - Middle: 1-line status bar (brain mode, multiline hint, /help)
    - Bottom: fixed input area (TextArea with slash-command autocomplete)
    """

    def __init__(
        self,
        agentic_core: Any,
        head_manager: Any,
        knowledge_store: Any,
        session_manager: Any,
        slash_handler: Any,
        runtime_config: Any | None = None,
        console: Console | None = None,
        show_banner: bool = True,
        process_manager: Any | None = None,
        brain: str = BRAIN_LOCAL,
        claude_adapter: Any | None = None,
        pipeline: ShellPipeline | None = None,
        service_manager: ServiceManager | None = None,
        event_watcher: EventWatcher | None = None,
        fast_head: str | None = None,
        debug_enrichment: bool = False,
    ) -> None:
        self.ac = agentic_core
        self.hm = head_manager
        self.ks = knowledge_store
        self.sessions = session_manager
        self.slash = slash_handler
        self.config = runtime_config
        self.console = console or Console()
        self.show_banner = show_banner
        self.process_manager = process_manager
        self.pipeline = pipeline
        self.service_manager = service_manager
        self.event_watcher = event_watcher

        # Conversation context persistence (survives SDK compaction)
        conv_cfg = getattr(getattr(runtime_config, "pipeline", None), "conversation", None)
        self._conversation_ctx = ConversationContext(
            recent_count=getattr(conv_cfg, "recent_count", 6),
            summary_interval=getattr(conv_cfg, "summary_interval", 10),
            max_summary_chars=getattr(conv_cfg, "max_summary_chars", 2000),
            max_recent_chars=getattr(conv_cfg, "max_recent_chars", 4000),
        )

        # Brain mode: "local", "claude", or "dual" (System 1 + System 2)
        self._brain = brain if brain in (BRAIN_LOCAL, BRAIN_CLAUDE, BRAIN_DUAL) else BRAIN_LOCAL
        self._claude_adapter = claude_adapter
        self._claude_conversation_id = "shell-default"
        self._codebase_ctx_cache: str | None = None
        # Dual brain config
        self._fast_head_id: str | None = fast_head  # Ollama head for fast brain
        self._debug_enrichment: bool = debug_enrichment  # print enriched query

        # Verbose output: show model, cost, turns, etc. after each response
        self._verbose = False
        # Last response metadata (populated by _chat_via_claude)
        self._last_meta: dict[str, Any] = {}
        # Event watcher background task
        self._watcher_task: asyncio.Task[None] | None = None

        # Resource tracker for dashboard sparklines
        from ..resource_monitor import ResourceTracker
        self._resource_tracker = ResourceTracker()
        self._resource_tracker.start_background(interval=10.0)

        # TUI state
        self._status_text: str = ""
        self._multiline_mode: bool = False
        self._output_pane: _OutputPane | None = None
        self._output_window: Window | None = None
        self._app: Application | None = None
        self._input_area: TextArea | None = None
        self._event_drain_task: asyncio.Task[None] | None = None
        self._processing_task: asyncio.Task[None] | None = None
        self._brain_lock: asyncio.Lock = asyncio.Lock()
        self._current_session_id: str = ""
        self._tui_log_handler: _TUILogHandler | None = None
        self._shutting_down: bool = False

        # Legacy (kept for deprecated methods / backward compat)
        self._prompt_session: PromptSession[str] | None = None

        # Participant identity (registered on first use)
        self._participant_id: str = ""
        self._register_participant()

    # -------------------------------------------------------------------
    # Participant identity
    # -------------------------------------------------------------------

    @staticmethod
    def _compute_context_hash() -> tuple[str, dict[str, str]]:
        """Compute environment fingerprint for participant identity.

        Returns (context_hash, metadata_dict).
        Hash = sha256(git_remote + hostname + repo_path).
        """
        import hashlib
        import socket
        import subprocess

        meta: dict[str, str] = {}

        try:
            meta["hostname"] = socket.gethostname()
        except Exception:
            meta["hostname"] = "unknown"

        for key, cmd in [
            ("git_remote", ["git", "remote", "get-url", "origin"]),
            ("repo_path", ["git", "rev-parse", "--show-toplevel"]),
            ("branch", ["git", "branch", "--show-current"]),
        ]:
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=5,
                    creationflags=no_window_flags(),
                )
                meta[key] = result.stdout.strip() if result.returncode == 0 else ""
            except Exception:
                meta[key] = ""

        fingerprint = f"{meta.get('git_remote', '')}|{meta['hostname']}|{meta.get('repo_path', '')}"
        context_hash = hashlib.sha256(fingerprint.encode()).hexdigest()
        return context_hash, meta

    def _register_participant(self) -> None:
        """Register this shell as a participant in the knowledge base."""
        if not self.ks or not hasattr(self.ks, "register_participant"):
            return
        try:
            context_hash, meta = self._compute_context_hash()
            participant = self.ks.register_participant(
                name=AGENT_ID,
                context_hash=context_hash,
                agent_type="shell",
                metadata=meta,
            )
            self._participant_id = participant.participant_id
            # Wire to pipeline so claims carry participant_id
            if self.pipeline:
                self.pipeline._participant_id = self._participant_id
        except Exception as e:
            logger.debug("Participant registration failed: %s", e)

    # -------------------------------------------------------------------
    # Main run loop
    # -------------------------------------------------------------------

    async def run(self, session_id: str) -> None:
        """Main entry point — builds TUI and runs until exit."""
        self._current_session_id = session_id

        # Load Claude adapter eagerly if starting in claude mode
        if self._brain == BRAIN_CLAUDE:
            ready = await self._ensure_claude_ready()
            if not ready:
                self._brain = BRAIN_LOCAL

        # Auto-start configured services
        if self.service_manager:
            try:
                messages = await self.service_manager.auto_start_all()
                for msg in messages:
                    logger.info("Service: %s", msg)
            except Exception as e:
                logger.warning("Service auto-start error: %s", e)

        # Start AgenticCore
        if hasattr(self.ac, 'start'):
            await self.ac.start()

        # Start event watcher background task
        if self.event_watcher:
            ew_cfg = getattr(getattr(self.config, "pipeline", None), "event_watcher", None)
            if not ew_cfg or getattr(ew_cfg, "enabled", True):
                self._watcher_task = asyncio.create_task(
                    self.event_watcher.run(), name="event-watcher",
                )

        # Build TUI application (test seam)
        app = self._build_application()

        # Print banner into output pane
        if self.show_banner:
            self._print_banner()

        brain_label = "Claude SDK" if self._brain == BRAIN_CLAUDE else "local"
        self._tui_print(
            f"[dim]Session: {session_id} | Brain: {brain_label} | "
            "Type /help for commands, exit to quit.[/dim]\n"
        )

        # Install TUI log handler and suppress direct-to-terminal handlers
        # so log lines don't bleed through the full-screen layout.
        self._suppressed_handlers: list[tuple[logging.Handler, int]] = []
        if self._output_pane:
            self._tui_log_handler = _TUILogHandler(self._output_pane, app)
            self._tui_log_handler.setLevel(logging.WARNING)
            self._tui_log_handler.setFormatter(
                logging.Formatter("[%(name)s] %(message)s"),
            )
            logging.root.addHandler(self._tui_log_handler)

            # Mute existing StreamHandlers (they write to stderr/stdout)
            for handler in logging.root.handlers:
                if (
                    isinstance(handler, logging.StreamHandler)
                    and not isinstance(handler, logging.FileHandler)
                    and handler is not self._tui_log_handler
                ):
                    self._suppressed_handlers.append((handler, handler.level))
                    handler.setLevel(logging.CRITICAL + 1)

        # Start event drain background loop
        self._event_drain_task = asyncio.create_task(
            self._drain_events_loop(), name="event-drain",
        )

        try:
            await app.run_async()
        except (EOFError, KeyboardInterrupt):
            pass  # Normal exit (Ctrl+D / Ctrl+C)
        finally:
            # Stop all new processing immediately
            self._shutting_down = True
            if self._processing_task and not self._processing_task.done():
                self._processing_task.cancel()
            # Restore suppressed log handlers
            for handler, original_level in self._suppressed_handlers:
                handler.setLevel(original_level)
            self._suppressed_handlers = []
            # Remove TUI log handler
            if self._tui_log_handler:
                logging.root.removeHandler(self._tui_log_handler)
                self._tui_log_handler = None
            # Async cleanup with overall 10s timeout so shell never hangs
            try:
                await asyncio.wait_for(self._async_cleanup(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("Shell cleanup timed out after 10s — forcing exit")
            except Exception as e:
                logger.warning("Shell cleanup error: %s", e)
            print("Session ended.")  # plain print — TUI may be torn down

    async def _async_cleanup(self) -> None:
        """Async cleanup of all background tasks and services."""
        # Cancel event drain loop
        if self._event_drain_task and not self._event_drain_task.done():
            self._event_drain_task.cancel()
            try:
                await self._event_drain_task
            except (asyncio.CancelledError, Exception):
                pass
        # Stop event watcher
        if self._watcher_task and not self._watcher_task.done():
            self._watcher_task.cancel()
            try:
                await self._watcher_task
            except (asyncio.CancelledError, Exception):
                pass
        # Shutdown services (cloud-marketplace, auto-responder, etc.)
        if self.service_manager:
            try:
                await self.service_manager.shutdown_all()
            except Exception as e:
                logger.warning("Service shutdown error: %s", e)
        if self.process_manager:
            try:
                await self.process_manager.cleanup()
            except Exception:
                pass
        if hasattr(self.ac, 'stop'):
            try:
                await self.ac.stop()
            except Exception:
                pass
        if hasattr(self.hm, 'shutdown'):
            try:
                await self.hm.shutdown()
            except Exception:
                pass
