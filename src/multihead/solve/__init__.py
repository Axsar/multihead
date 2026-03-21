"""MultiHead Solve: Distributed task solving via multi-agent consensus.

Coordinator implementation for distributing tasks to multiple agents,
collecting proposals, running consensus voting, and orchestrating execution.
"""

# Standard-library modules re-exported so that existing test patches
# targeting ``multihead.solve.signal`` / ``multihead.solve.atexit``
# continue to resolve after the flat module became a package.
import atexit  # noqa: F401
import signal  # noqa: F401

from ..consensus import ConsensusEngine  # noqa: F401  — patched by tests

from .config import SolveConfig, SolveResult, prompt_multi_session
from .coordinator import SolveCoordinator
from .discovery import (
    discover_active_sessions,
    discover_sessions_mdns,
    mark_session_offline,
    write_presence_claim,
)
from .onboarding import (
    _get_state_file,
    _load_seen_sessions,
    _save_seen_sessions,
    _show_onboarding_messages,
)

__all__ = [
    # Config
    "SolveConfig",
    "SolveResult",
    "prompt_multi_session",
    # Coordinator
    "SolveCoordinator",
    # Discovery
    "discover_active_sessions",
    "discover_sessions_mdns",
    "mark_session_offline",
    "write_presence_claim",
    # Onboarding (private but used by tests)
    "_get_state_file",
    "_load_seen_sessions",
    "_save_seen_sessions",
    "_show_onboarding_messages",
]
