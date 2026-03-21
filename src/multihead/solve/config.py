"""Configuration dataclasses and interactive prompts for distributed solve."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..consensus import ConsensusStrategy


@dataclass
class SolveConfig:
    """Configuration for distributed solve session."""

    project_id: str = "multihead"
    session_id: str = "multihead-coordinator"
    proposal_timeout_seconds: float = 300.0  # 5 minutes
    min_proposals: int = 1
    max_proposals: int = 10
    consensus_strategy: ConsensusStrategy = ConsensusStrategy.MAJORITY
    auto_approve: bool = False


@dataclass
class SolveResult:
    """Result of a distributed solve operation."""

    task: str
    request_id: str
    proposals_received: int
    winning_proposal_id: str | None
    assigned_agent: str | None
    execution_started: bool
    result_claim_id: str | None
    success: bool
    error: str | None = None
    decomposition: str | None = None  # Plan text returned to caller in solo mode


def prompt_multi_session(other_sessions: list[dict[str, Any]]) -> bool:
    """Prompt user to wait for multi-session proposals.

    Args:
        other_sessions: List of discovered active sessions

    Returns:
        False (always returns False, meaning user wants to wait for proposals)
        This controls wait_for_proposals behavior, NOT auto_approve.
    """
    import sys
    import select

    names = [s['session_id'] for s in other_sessions[:3]]
    count = len(other_sessions)
    smart_timeout = min(30 * count, 120)  # 30s/session, max 2min

    print(f"\n💡 {count} other session(s) detected: {', '.join(names)}")
    if count > 3:
        print(f"   (and {count - 3} more...)")
    print(f"   Wait for their proposals? [y/N] (auto-no in {smart_timeout}s): ", end="", flush=True)

    # Simple timeout-based input (non-blocking on Unix/Linux)
    if sys.platform != "win32":
        ready, _, _ = select.select([sys.stdin], [], [], smart_timeout)
        if ready:
            response = sys.stdin.readline().strip().lower()
        else:
            print("\n   ⏱️  Timeout - proceeding solo")
            return True  # Timeout = solo mode
    else:
        # Windows: threading.Timer fallback (select doesn't support stdin on Windows)
        import threading

        result: list[str] = []
        input_done = threading.Event()

        def _on_timeout() -> None:
            if not input_done.is_set():
                print("\n   ⏱️  Timeout - proceeding solo")
                result.append("")
                input_done.set()

        timer = threading.Timer(smart_timeout, _on_timeout)
        timer.daemon = True
        timer.start()

        def _read_input() -> None:
            try:
                result.append(input().strip().lower())
            except (EOFError, KeyboardInterrupt):
                print("\n   ⏹️  Interrupted - proceeding solo")
                result.append("")
            finally:
                input_done.set()

        reader = threading.Thread(target=_read_input, daemon=True)
        reader.start()

        input_done.wait()
        timer.cancel()
        response = result[0] if result else ""

    if response in ['y', 'yes']:
        print("   ✓ Waiting for proposals...")
        return False  # Wait for proposals
    else:
        print("   ⏭️  Skipping multi-session - proceeding solo")
        return True  # Solo mode
