"""Input handling mixin — key bindings, toolbar, TUI app builder."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.widgets import TextArea

from .prompts import BRAIN_CLAUDE, _SLASH_COMMANDS
from .tui import _OutputPane

logger = logging.getLogger(__name__)


class InputMixin:
    """Mixin providing input handling, key bindings, and TUI app building.

    Expects the host class to provide:
    - self.hm (HeadManager)
    - self._brain (str)
    - self._multiline_mode (bool)
    - self._status_text (str)
    - self._output_pane (_OutputPane | None)
    - self._output_window (Window | None)
    - self._app (Application | None)
    - self._input_area (TextArea | None)
    - self._processing_task (asyncio.Task | None)
    - self._current_session_id (str)
    - self._prompt_session (PromptSession | None)
    - self.slash (slash_handler)
    - self.pipeline (ShellPipeline | None)
    - self._tui_print(...)
    - self._set_status(text)
    - self._display_response(response)
    - self._brain_fn_for_pipeline()
    - self._chat_via_claude(session_id, user_input) -> str
    - self._chat_via_local(session_id, user_input) -> str
    """

    # -------------------------------------------------------------------
    # prompt_toolkit helpers (status bar, key bindings)
    # -------------------------------------------------------------------

    def _get_prompt_text(self) -> str:
        """Build prompt string showing active brain/head."""
        if self._brain == BRAIN_CLAUDE:
            label = "[claude-sdk]"
        else:
            active = self.hm.active_head if self.hm else None
            label = f"[{active}]" if active else ""
        return f"{label} you> " if label else "you> "

    def _get_toolbar(self) -> HTML:
        """Bottom toolbar / status bar content."""
        if self._status_text:
            return HTML(f"<b>{self._status_text}</b>")
        brain = "Claude SDK" if self._brain == BRAIN_CLAUDE else "Local GPU"
        if self._multiline_mode:
            mode = "<style bg='ansiyellow' fg='ansiblack'> MULTILINE </style> Alt+Enter to send | Ctrl+E: single-line"
        else:
            mode = "Ctrl+E: multiline"
        scroll_info = ""
        if self._output_pane and self._output_pane.scroll_offset > 0:
            scroll_info = f"  |  <style bg='ansiyellow' fg='ansiblack'> \u2191{self._output_pane.scroll_offset} lines </style> End=bottom"
        return HTML(f"<b>Brain:</b> {brain}  |  {mode}  |  PgUp/Dn Ctrl+\u2191\u2193: scroll{scroll_info}  |  <b>/help</b>")

    def _build_key_bindings(self) -> KeyBindings:
        """Custom key bindings for TUI (multiline toggle, submit, exit)."""
        kb = KeyBindings()
        is_multiline = Condition(lambda: self._multiline_mode)

        @kb.add("c-e")
        def _toggle_multiline(event: Any) -> None:
            self._multiline_mode = not self._multiline_mode
            # Patch the underlying buffer so Enter inserts newline in multiline
            if self._input_area:
                self._input_area.buffer.multiline = Condition(
                    lambda: self._multiline_mode,
                )
            if event.app:
                event.app.invalidate()

        @kb.add("escape", "enter", filter=is_multiline)
        def _submit_multiline(event: Any) -> None:
            event.current_buffer.validate_and_handle()

        @kb.add("c-c")
        def _ctrl_c(event: Any) -> None:
            # Cancel running brain request first
            if self._processing_task and not self._processing_task.done():
                self._processing_task.cancel()
                self._set_status("")
                self._tui_print("[dim]Cancelled.[/dim]")
                return
            buf = event.app.current_buffer
            if buf.text:
                buf.reset()
            else:
                event.app.exit()

        @kb.add("c-d")
        def _ctrl_d(event: Any) -> None:
            event.app.exit()

        # Output pane scrolling — eager=True to preempt TextArea bindings
        @kb.add("pageup", eager=True)
        def _page_up(event: Any) -> None:
            if self._output_pane:
                self._output_pane.scroll_up(20)
                event.app.invalidate()

        @kb.add("pagedown", eager=True)
        def _page_down(event: Any) -> None:
            if self._output_pane:
                self._output_pane.scroll_down(20)
                event.app.invalidate()

        # Ctrl+Up / Ctrl+Down — reliable scroll (never consumed by TextArea)
        @kb.add("c-up")
        def _ctrl_scroll_up(event: Any) -> None:
            if self._output_pane:
                self._output_pane.scroll_up(5)
                event.app.invalidate()

        @kb.add("c-down")
        def _ctrl_scroll_down(event: Any) -> None:
            if self._output_pane:
                self._output_pane.scroll_down(5)
                event.app.invalidate()

        # Shift+Up / Shift+Down for fine scrolling
        @kb.add("s-up")
        def _scroll_up(event: Any) -> None:
            if self._output_pane:
                self._output_pane.scroll_up(3)
                event.app.invalidate()

        @kb.add("s-down")
        def _scroll_down(event: Any) -> None:
            if self._output_pane:
                self._output_pane.scroll_down(3)
                event.app.invalidate()

        # End key — jump to bottom, re-enable follow mode
        @kb.add("end", eager=True)
        def _scroll_to_bottom(event: Any) -> None:
            if self._output_pane:
                self._output_pane.scroll_to_bottom()
                event.app.invalidate()

        return kb

    def _set_status(self, text: str) -> None:
        """Update the status bar text."""
        self._status_text = text
        if self._app:
            try:
                self._app.invalidate()
            except Exception:
                pass
        elif self._prompt_session and self._prompt_session.app:
            try:
                self._prompt_session.app.invalidate()
            except Exception:
                pass

    # -------------------------------------------------------------------
    # TUI Application builder (test seam)
    # -------------------------------------------------------------------

    def _build_application(self) -> Application:
        """Create the full-screen split-pane TUI Application.

        Layout::

            +-- Scrollable Output --+
            |  (all output here)    |
            +-----------------------+
             status bar (1 line)
            +-- Input Area ---------+
            | > type here...        |
            +-----------------------+

        This method is the **test seam**: tests mock it to return a fake
        Application whose ``run_async()`` feeds inputs sequentially.
        """
        # Output pane (scroll is managed internally by _ViewportControl)
        self._output_pane = _OutputPane()
        output_window = Window(
            content=self._output_pane.control,
            wrap_lines=True,
            right_margins=[ScrollbarMargin(display_arrows=True)],
        )
        self._output_window = output_window

        # Status bar
        status_window = Window(
            content=FormattedTextControl(text=self._get_toolbar),
            height=1,
            style="reverse",
        )

        # Input area — multiline=False so TextArea wires accept_handler.
        # Ctrl+E toggles via buffer.multiline at runtime.
        completer = WordCompleter(_SLASH_COMMANDS, sentence=True)
        self._input_area = TextArea(
            height=Dimension(min=1, max=6, preferred=3),
            prompt="\u276f ",   # >
            multiline=False,
            completer=completer,
            accept_handler=self._on_input_accept,
            focusable=True,
            focus_on_click=True,
        )

        # Key bindings
        kb = self._build_key_bindings()

        layout = Layout(
            HSplit([
                output_window,
                status_window,
                self._input_area,
            ]),
            focused_element=self._input_area,
        )

        app: Application = Application(
            layout=layout,
            key_bindings=kb,
            full_screen=True,
            mouse_support=False,
        )
        self._app = app
        return app

    # -------------------------------------------------------------------
    # Input handling
    # -------------------------------------------------------------------

    def _on_input_accept(self, buffer: Any) -> None:
        """Called by TextArea when the user presses Enter (submit).

        Echoes the input to the output pane, then schedules async
        processing via ``_process_input``.
        """
        text = buffer.text.strip() if buffer.text else ""
        if not text:
            return

        # Re-enable follow mode — user wants to see the response
        if self._output_pane:
            self._output_pane.scroll_to_bottom()

        # Echo user input
        prompt_label = self._get_prompt_text()
        self._tui_print(f"[bold]{prompt_label}{text}[/bold]")

        if text.lower() in ("exit", "quit", "q"):
            if self._app:
                self._app.exit()
            return

        # Schedule async processing (track for Ctrl+C cancellation)
        self._processing_task = asyncio.ensure_future(self._process_input(text))

    async def _process_input(self, user_input: str) -> None:
        """Process a single user input (slash command or brain chat).

        This is the core processing extracted from the old REPL loop body.
        It routes slash commands to the handler, and everything else
        through the pipeline or brain.
        """
        session_id = self._current_session_id

        # Slash commands
        if self.slash and self.slash.is_slash_command(user_input):
            result = await self.slash.handle(user_input)
            self._tui_print(f"{result}\n")
            return

        # Route through pipeline or direct to brain
        _pre_k_hits = self.pipeline._stats.get("knowledge_hits", 0) if self.pipeline else 0
        _default_status = (
            "Waiting for Claude..."
            if self._brain == BRAIN_CLAUDE
            else "Thinking..."
        )
        self._set_status(_default_status)

        try:
            async with self._brain_lock:
                if self.pipeline:
                    brain_fn = self._brain_fn_for_pipeline()
                    response = await self.pipeline.process(
                        user_input, brain_fn, session_id,
                        on_status=self._set_status,
                    )
                elif self._brain == BRAIN_CLAUDE:
                    response = await self._chat_via_claude(session_id, user_input)
                else:
                    response = await self._chat_via_local(session_id, user_input)
        except (asyncio.CancelledError, KeyboardInterrupt):
            self._set_status("")
            self._tui_print("[dim]Cancelled.[/dim]")
            return

        self._set_status("")
        self._display_response(response)

        # Knowledge context indicator
        if self.pipeline:
            post_k = self.pipeline._stats.get("knowledge_hits", 0)
            if post_k > _pre_k_hits:
                self._tui_print("[dim][Knowledge context used][/dim]")
