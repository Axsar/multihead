"""MultiHead Agent Terminal — interactive shell for human-agent collaboration.

Usage:
    multihead shell [--head HEAD_ID] [--session SESSION_ID] [--brain local|claude]

The shell provides:
- Dual-brain architecture: local GPU (AgenticCore) or Claude SDK
- Rich status display (heads, VRAM, knowledge, mesh peers)
- Head management (/wake, /sleep, /swap, /status)
- Knowledge-aware RAG context from knowledge.db
- Process orchestration (/spawn, /ps, /kill)
- PLUR safety principles (Peace, Love, Unity, Respect)
- Brain-swap mid-session via /brain command
- Full-screen split-pane TUI (output pane + status bar + input area)

This package was refactored from a single shell.py module.
All public names are re-exported here for backward compatibility.
"""

from .core import Shell
from .prompts import (
    BRAIN_CLAUDE,
    BRAIN_LOCAL,
    SHELL_SYSTEM_PROMPT,
    _SLASH_COMMANDS,
)
from .tui import _OutputPane, _TUILogHandler, _ViewportControl

__all__ = [
    "BRAIN_CLAUDE",
    "BRAIN_LOCAL",
    "SHELL_SYSTEM_PROMPT",
    "Shell",
    "_OutputPane",
    "_SLASH_COMMANDS",
    "_TUILogHandler",
    "_ViewportControl",
]
