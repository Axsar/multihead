"""Conversation context persistence for shell brain sessions.

Builds rolling summaries and recent-message windows from local
SessionManager history, injected into the system prompt so that
conversation continuity survives Claude SDK context compaction.

All extraction is heuristic — no LLM calls.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_RECENT_COUNT = 6
DEFAULT_SUMMARY_INTERVAL = 10
DEFAULT_MAX_SUMMARY_CHARS = 2000
DEFAULT_MAX_RECENT_CHARS = 4000

# Truncation limits per message in recent window
_MSG_TRUNCATE = 200


class ConversationContext:
    """Tracks conversation state and produces context blocks for injection.

    Heuristic-only — no LLM calls.  Extracts key information from
    the session history stored in SessionManager.

    Usage::

        ctx = ConversationContext()

        # After each turn:
        ctx.on_turn(user_msg, assistant_msg)

        # Before sending to Claude:
        block = ctx.build_context_block(session.messages)
        system_prompt += f"\\n\\n{block}"
    """

    def __init__(
        self,
        recent_count: int = DEFAULT_RECENT_COUNT,
        summary_interval: int = DEFAULT_SUMMARY_INTERVAL,
        max_summary_chars: int = DEFAULT_MAX_SUMMARY_CHARS,
        max_recent_chars: int = DEFAULT_MAX_RECENT_CHARS,
    ) -> None:
        self._recent_count = recent_count
        self._summary_interval = summary_interval
        self._max_summary_chars = max_summary_chars
        self._max_recent_chars = max_recent_chars

        # Rolling state
        self._summary: str = ""
        self._turn_count: int = 0
        self._turns_since_summary: int = 0

    # ------------------------------------------------------------------
    # Turn tracking
    # ------------------------------------------------------------------

    def on_turn(self, user_message: str, assistant_response: str) -> None:
        """Called after each exchange. Updates internal counters."""
        self._turn_count += 1
        self._turns_since_summary += 1

    def needs_summary_refresh(self) -> bool:
        """Whether it's time to rebuild the summary."""
        return self._turns_since_summary >= self._summary_interval

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def summary(self) -> str:
        return self._summary

    # ------------------------------------------------------------------
    # Summary building (extractive, no LLM)
    # ------------------------------------------------------------------

    def build_summary(self, messages: list[Any]) -> str:
        """Build extractive summary from session message history.

        Selection strategy:
        1. First user message (sets the topic)
        2. Every Nth user message (captures topic shifts)
        3. Last 2 user messages (current thread)

        For each selected user message, includes first sentence
        of the next assistant response.

        *messages* should be a list of objects with ``.role`` and
        ``.content`` attributes (e.g. ``session.Message``).
        """
        if not messages:
            self._summary = ""
            return ""

        # Pair user messages with their indices and next assistant response
        pairs: list[tuple[int, str, str]] = []  # (turn_idx, user_short, asst_short)
        turn_idx = 0
        i = 0
        while i < len(messages):
            msg = messages[i]
            role = _get_role(msg)
            content = _get_content(msg)

            if role == "user" and content.strip():
                turn_idx += 1
                user_short = _truncate(content, 120)
                # Find next assistant response
                asst_short = ""
                for j in range(i + 1, len(messages)):
                    if _get_role(messages[j]) == "assistant":
                        asst_short = _extract_first_sentence(
                            _get_content(messages[j])
                        )
                        break
                pairs.append((turn_idx, user_short, asst_short))
            i += 1

        if not pairs:
            self._summary = ""
            return ""

        # Select anchor messages
        selected_indices: set[int] = set()
        # First
        selected_indices.add(0)
        # Every Nth
        n = max(self._summary_interval // 2, 3)
        for idx in range(n, len(pairs), n):
            selected_indices.add(idx)
        # Last 2
        if len(pairs) >= 2:
            selected_indices.add(len(pairs) - 2)
        selected_indices.add(len(pairs) - 1)

        # Build summary lines
        lines: list[str] = []
        total_chars = 0
        for idx in sorted(selected_indices):
            turn_num, user_short, asst_short = pairs[idx]
            if asst_short:
                line = f"Turn {turn_num}: {user_short} -> {asst_short}"
            else:
                line = f"Turn {turn_num}: {user_short}"
            if total_chars + len(line) > self._max_summary_chars:
                break
            lines.append(line)
            total_chars += len(line)

        self._summary = "\n".join(lines)
        self._turns_since_summary = 0
        return self._summary

    # ------------------------------------------------------------------
    # Recent message window
    # ------------------------------------------------------------------

    def build_recent_window(self, messages: list[Any]) -> str:
        """Format the last K messages as a recent conversation block.

        Each message is truncated to keep within budget.
        """
        if not messages:
            return ""

        # Take last recent_count messages (user + assistant)
        recent = messages[-self._recent_count:]

        lines: list[str] = []
        total_chars = 0
        for msg in recent:
            role = _get_role(msg)
            content = _get_content(msg)
            if role not in ("user", "assistant"):
                continue
            label = "User" if role == "user" else "Assistant"
            short = _truncate(content, _MSG_TRUNCATE)
            line = f"{label}: {short}"
            if total_chars + len(line) > self._max_recent_chars:
                break
            lines.append(line)
            total_chars += len(line)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Combined context block
    # ------------------------------------------------------------------

    def build_context_block(self, messages: list[Any]) -> str:
        """Build the full conversation context block for system prompt.

        Combines rolling summary + recent message window.
        Returns empty string if no useful context.
        """
        if not messages:
            return ""

        parts: list[str] = []

        # Summary (if we have one)
        if self._summary:
            parts.append(
                f"[Conversation Summary (turns 1-{self._turn_count})]\n"
                f"{self._summary}"
            )

        # Recent window
        recent = self.build_recent_window(messages)
        if recent:
            parts.append(f"[Recent Conversation]\n{recent}")

        return "\n\n".join(parts)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _get_role(msg: Any) -> str:
    """Extract role from a message (dict or object)."""
    if isinstance(msg, dict):
        return msg.get("role", "")
    return getattr(msg, "role", "")


def _get_content(msg: Any) -> str:
    """Extract content from a message (dict or object)."""
    if isinstance(msg, dict):
        return msg.get("content", "")
    return getattr(msg, "content", "")


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, adding ellipsis if needed."""
    text = text.strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3] + "..."


def _extract_first_sentence(text: str) -> str:
    """Extract first sentence from text (max 100 chars).

    Splits on sentence-ending punctuation or newlines.
    """
    if not text:
        return ""
    text = text.strip()
    # Split on sentence boundaries
    match = re.match(r"^(.+?[.!?])\s", text)
    if match:
        sentence = match.group(1)
    else:
        # No sentence boundary — take first line
        sentence = text.split("\n")[0]
    return _truncate(sentence, 100)
