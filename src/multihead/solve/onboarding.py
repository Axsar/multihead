"""Onboarding UX helpers for first-run and new-session detection."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_state_file() -> Path:
    """Get path to state file for tracking seen sessions."""
    state_dir = Path.home() / ".multihead"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / "state.json"


def _load_seen_sessions() -> set[str]:
    """Load set of previously seen session IDs."""
    state_file = _get_state_file()
    if not state_file.exists():
        return set()

    try:
        import json
        with open(state_file, 'r') as f:
            state = json.load(f)
        return set(state.get("seen_sessions", []))
    except Exception as e:
        logger.warning("Failed to load state file: %s", e)
        return set()


def _save_seen_sessions(seen: set[str]):
    """Save set of seen session IDs to state file."""
    state_file = _get_state_file()
    try:
        import json
        state = {"seen_sessions": list(seen)}
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.warning("Failed to save state file: %s", e)


def _show_onboarding_messages(
    is_first_run: bool,
    new_sessions: list[str],
    total_sessions: int,
):
    """Display friendly onboarding messages."""
    if is_first_run and total_sessions == 0:
        # First solo run - suggest mesh setup
        print()
        print("💡 Tip: Run 'multihead init --mesh' to enable multi-session collaboration")
        print()
    elif new_sessions:
        # New collaborators detected
        for session_id in new_sessions[:3]:  # Show first 3
            print(f"👋 New session detected: {session_id}!")
        if len(new_sessions) > 3:
            print(f"   ... and {len(new_sessions) - 3} more")
        print()
