"""TUI helper classes for the full-screen split-pane terminal UI."""

from __future__ import annotations

import logging
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import ANSI


# ---------------------------------------------------------------------------
# TUI helper classes
# ---------------------------------------------------------------------------


class _OutputPane:
    """Manages the scrollable output buffer for the full-screen TUI.

    Stores output as a list of *lines* (each line is a list of
    prompt_toolkit style-tuples).  A ``_ViewportControl`` renders only
    the visible portion based on ``scroll_offset`` (lines from end).

    Scroll is entirely internal — no reliance on ``Window.vertical_scroll``.
    """

    MAX_LINES = 10_000

    def __init__(self) -> None:
        self._lines: list[list[tuple[str, str]]] = []
        self._control = _ViewportControl(pane=self)
        # Follow mode: True = auto-scroll to bottom on new output
        self.follow: bool = True
        # Scroll offset: 0 = bottom (latest), >0 = scrolled up N lines
        self.scroll_offset: int = 0

    # -- Content manipulation ---------------------------------------------

    @staticmethod
    def _split_fragments_to_lines(
        fragments: list[tuple[str, str]],
    ) -> list[list[tuple[str, str]]]:
        """Split a flat fragment list into per-line lists at '\\n' boundaries."""
        lines: list[list[tuple[str, str]]] = [[]]
        for style, text in fragments:
            parts = text.split("\n")
            for i, part in enumerate(parts):
                if i > 0:
                    lines.append([])
                if part:
                    lines[-1].append((style, part))
        return lines

    def append_ansi(self, text: str) -> None:
        """Convert an ANSI-escaped string to lines and append."""
        if not text:
            return
        frags = list(ANSI(text).__pt_formatted_text__())
        new_lines = self._split_fragments_to_lines(frags)
        if not new_lines:
            return
        # Merge first new line into last existing line
        if self._lines:
            self._lines[-1].extend(new_lines[0])
            self._lines.extend(new_lines[1:])
        else:
            self._lines.extend(new_lines)
        self._trim()

    def append_plain(self, text: str, style: str = "") -> None:
        """Append plain text as a new line."""
        self._lines.append([(style, text)])
        self._trim()

    def _trim(self) -> None:
        if len(self._lines) > self.MAX_LINES:
            self._lines = self._lines[-self.MAX_LINES:]

    @property
    def control(self) -> "UIControl":
        return self._control

    # -- Scroll helpers ---------------------------------------------------

    def scroll_up(self, lines: int = 3) -> None:
        """Scroll up (toward older content)."""
        max_offset = max(0, len(self._lines) - 1)
        self.scroll_offset = min(self.scroll_offset + lines, max_offset)
        self.follow = False

    def scroll_down(self, lines: int = 3) -> None:
        """Scroll down (toward newer content)."""
        self.scroll_offset = max(0, self.scroll_offset - lines)
        if self.scroll_offset == 0:
            self.follow = True

    def scroll_to_bottom(self) -> None:
        """Jump to the bottom and re-enable follow mode."""
        self.scroll_offset = 0
        self.follow = True


class _ViewportControl:
    """Custom UIControl that renders a windowed view of _OutputPane lines.

    Implements the prompt_toolkit UIControl protocol directly to bypass
    Window.vertical_scroll (which is unreliable for FormattedTextControl).
    The viewport is controlled entirely via ``_OutputPane.scroll_offset``.
    """

    def __init__(self, pane: _OutputPane) -> None:
        self._pane = pane

    def create_content(self, width: int, height: int) -> Any:
        from prompt_toolkit.layout.controls import UIContent

        lines = self._pane._lines
        total = len(lines)
        offset = self._pane.scroll_offset

        # Visible window: [start, end)
        end = max(0, total - offset)
        start = max(0, end - height)

        visible_count = end - start

        def get_line(i: int) -> list[tuple[str, str]]:
            idx = start + i
            if 0 <= idx < total and idx < end:
                return lines[idx] or [("", " ")]
            return [("", " ")]

        return UIContent(
            get_line=get_line,
            line_count=max(1, visible_count),
        )

    def is_focusable(self) -> bool:
        return True

    def preferred_width(self, max_available_width: int) -> int | None:
        return None

    def preferred_height(
        self,
        width: int,
        max_available_height: int,
        wrap_lines: bool,
        get_line_prefix: Any,
    ) -> int | None:
        return None

    def get_invalidate_events(self) -> list[Any]:
        return []

    def reset(self) -> None:
        pass

    def get_key_bindings(self) -> Any:
        from prompt_toolkit.key_binding import KeyBindings
        return KeyBindings()

    def mouse_handler(self, mouse_event: Any) -> Any:
        return NotImplemented


class _TUILogHandler(logging.Handler):
    """Routes log records into the TUI output pane."""

    def __init__(self, output_pane: _OutputPane, app: Application | None = None) -> None:
        super().__init__()
        self._pane = output_pane
        self._app = app

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._pane.append_plain(msg, "italic")
            if self._app:
                self._app.invalidate()
        except Exception:
            self.handleError(record)
