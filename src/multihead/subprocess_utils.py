"""Subprocess utilities for cross-platform window suppression."""
import sys
import subprocess


def no_window_flags() -> int:
    """Return CREATE_NO_WINDOW on Windows, 0 elsewhere."""
    if sys.platform == "win32":
        return getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return 0
