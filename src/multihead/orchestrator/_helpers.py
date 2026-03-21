"""Shared helper utilities for the orchestrator package."""

from __future__ import annotations


class _SafeFormatDict(dict):
    """Dict that returns the original placeholder for missing keys.

    This allows ``"Hello {name}, keep {unknown}".format_map(...)``
    to substitute ``{name}`` while leaving ``{unknown}`` intact.
    """

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"
