"""Deprecated shell methods — kept for backward compatibility."""

from __future__ import annotations

import logging
import sys
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.filters import Condition

from .prompts import _SLASH_COMMANDS

logger = logging.getLogger(__name__)


class DeprecatedMixin:
    """Mixin providing deprecated methods for backward compatibility.

    Expects the host class to provide:
    - self._multiline_mode (bool)
    - self._prompt_session (PromptSession | None)
    - self._build_key_bindings() -> KeyBindings
    - self._get_toolbar() -> HTML
    """

    def _build_prompt_session(self, output=None) -> PromptSession[str]:
        """Create a PromptSession with slash-command completion and toolbar.

        Args:
            output: Optional prompt_toolkit output backend. Pass
                ``DummyOutput()`` in tests to avoid Win32 console requirement.

        .. deprecated:: Use ``_build_application()`` instead.
            Kept for backward compatibility with tests that mock this method.
        """
        completer = WordCompleter(_SLASH_COMMANDS, sentence=True)
        return PromptSession(
            completer=completer,
            multiline=Condition(lambda: self._multiline_mode),
            prompt_continuation="  ... ",
            wrap_lines=True,
            key_bindings=self._build_key_bindings(),
            bottom_toolbar=self._get_toolbar,
            complete_while_typing=False,
            output=output,
        )

    def _redirect_logging_to_stdout(self) -> None:
        """Redirect all logging StreamHandlers from stderr to stdout.

        .. deprecated:: TUI mode uses ``_TUILogHandler`` instead.
        """
        self._original_streams: list[tuple[logging.StreamHandler, Any]] = []
        for handler in logging.root.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler
            ):
                self._original_streams.append((handler, handler.stream))
                handler.stream = sys.stdout

    def _restore_logging_streams(self) -> None:
        """Restore logging StreamHandlers to their original streams.

        .. deprecated:: TUI mode uses ``_TUILogHandler`` instead.
        """
        for handler, original_stream in getattr(self, "_original_streams", []):
            handler.stream = original_stream
        self._original_streams = []

    async def _read_multiline(self, first_line: str) -> str | None:
        """Handle multi-line continuation after the first line.

        .. deprecated:: TUI mode handles multi-line via TextArea (Ctrl+E toggle).

        Supports backtick blocks (```), backslash continuation (\\),
        and paste detection (select-based).
        """
        # --- triple-backtick block mode ---
        if first_line.startswith("```"):
            rest = first_line[3:].strip()
            lines: list[str] = []
            if rest:
                lines.append(rest)
            while True:
                try:
                    line = await self._prompt_session.prompt_async("  ... ")
                except (EOFError, KeyboardInterrupt):
                    break
                if line.strip() == "```":
                    break
                lines.append(line)
            return "\n".join(lines) or None

        # --- backslash continuation ---
        if first_line.endswith("\\"):
            lines = [first_line[:-1]]
            while True:
                try:
                    line = await self._prompt_session.prompt_async("  ... ")
                    line = line.strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if line.endswith("\\"):
                    lines.append(line[:-1])
                else:
                    lines.append(line)
                    break
            return "\n".join(lines) or None

        # --- paste detection ---
        pasted = self._collect_paste_lines()
        if pasted:
            return first_line + "\n" + "\n".join(pasted)

        return first_line

    @staticmethod
    def _collect_paste_lines() -> list[str]:
        """Collect any remaining pasted lines from stdin.

        .. deprecated:: TUI mode handles multi-line input natively via TextArea.

        Uses select() to detect data buffered on stdin within a short
        window.  Returns empty list if nothing is available (normal
        typed input).  Only works on Unix-like systems (Linux, macOS).
        """
        import select
        import sys

        lines: list[str] = []
        try:
            # Short timeout — pasted data arrives nearly instantly
            while select.select([sys.stdin], [], [], 0.05)[0]:
                line = sys.stdin.readline()
                if not line:
                    break
                lines.append(line.rstrip("\n"))
        except (OSError, ValueError):
            # select not supported (e.g. Windows without WSL)
            pass
        return lines
