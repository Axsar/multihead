"""Solve pipeline — self mode, consensus mode, server proxy."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# Active self-solve runs keyed by run_id
_active_runs: dict[str, dict] = {}


def _generate_run_id() -> str:
    """Generate a unique run ID matching the existing convention."""
    import ulid
    return f"run_{ulid.ULID()!s}"


def _runs_dir() -> Path:
    """Resolve the runs directory from settings or env."""
    data_dir = os.environ.get("MULTIHEAD_DATA_DIR", str(Path.home() / ".multihead"))
    return Path(data_dir) / "runs"


def _init_run(run_id: str, task: str, plan: dict) -> Path:
    """Create run directory, write initial events."""
    run_dir = _runs_dir() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)

    # Write run_created event
    event = {
        "event_id": f"evt_self_{int(time.time())}",
        "run_id": run_id,
        "kind": "run_created",
        "step_id": None,
        "data": {
            "task": task,
            "mode": "self",
            "plan": plan,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(run_dir / "events.jsonl", "a") as f:
        f.write(json.dumps(event) + "\n")

    return run_dir


def _append_event(run_id: str, kind: str, step_id: str | None, data: dict) -> None:
    """Append an event to the run's events.jsonl."""
    run_dir = _runs_dir() / run_id
    if not run_dir.exists():
        logger.warning("Run dir %s does not exist", run_id)
        return

    event = {
        "event_id": f"evt_self_{int(time.time())}_{kind}",
        "run_id": run_id,
        "kind": kind,
        "step_id": step_id,
        "data": data,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(run_dir / "events.jsonl", "a") as f:
        f.write(json.dumps(event) + "\n")


async def _solve_self(
    task: str,
    strategy: str = "first_to_ahead",
    max_steps: int = 20,
    dry_run: bool = False,
) -> str:
    """Self-solve: decompose locally, return plan for caller to execute.

    No serve dependency. Uses knowledge store directly for RAG context.
    Returns the plan + run_id so caller can track step execution.
    """
    from ._tools_decompose import _decompose_self

    # Get decomposition with knowledge context
    decomposition = await _decompose_self(task)

    # Generate run_id for tracking
    try:
        run_id = _generate_run_id()
    except ImportError:
        run_id = f"run_self_{int(time.time())}"

    plan_info = {
        "task": task,
        "strategy": strategy,
        "max_steps": max_steps,
        "mode": "self",
    }

    if not dry_run:
        # Create run directory and initial event
        _init_run(run_id, task, plan_info)
        _active_runs[run_id] = {
            "task": task,
            "steps_total": 0,
            "steps_completed": 0,
            "steps_failed": 0,
            "started_at": time.time(),
        }

    result = {
        "run_id": run_id,
        "status": "awaiting_execution" if not dry_run else "dry_run",
        "mode": "self",
        "output": decomposition,
        "instructions": (
            "You are the executor. Follow the decomposition plan above.\n"
            "After completing each step, report it with multihead_refine_step.\n"
            "When all steps are done, call multihead_solve with the same run_id to finalize."
        ),
        "dry_run": dry_run,
    }

    return json.dumps(result, indent=2)


async def _report_solve_step(
    run_id: str,
    step_id: str,
    status: str = "completed",
    output: str = "",
) -> str:
    """Report completion of a self-solve step."""
    if run_id not in _active_runs and (_runs_dir() / run_id).exists():
        # Reconnect to existing run
        _active_runs[run_id] = {
            "task": "",
            "steps_total": 0,
            "steps_completed": 0,
            "steps_failed": 0,
            "started_at": time.time(),
        }

    _append_event(run_id, f"step_{status}", step_id, {
        "output": output[:2000],  # truncate for event log
        "status": status,
    })

    # Save full output as artifact
    run_dir = _runs_dir() / run_id
    if run_dir.exists():
        safe_name = step_id.replace("/", "_").replace(" ", "_")
        artifact_path = run_dir / "artifacts" / f"{safe_name}_output.txt"
        artifact_path.write_text(output)

    # Update tracking
    if run_id in _active_runs:
        if status == "completed":
            _active_runs[run_id]["steps_completed"] += 1
        elif status == "failed":
            _active_runs[run_id]["steps_failed"] += 1

    return json.dumps({
        "run_id": run_id,
        "step_id": step_id,
        "status": "recorded",
    })


async def _finalize_solve(run_id: str) -> str:
    """Finalize a self-solve run."""
    run_info = _active_runs.pop(run_id, {})
    duration = time.time() - run_info.get("started_at", time.time())

    _append_event(run_id, "run_completed", None, {
        "steps_completed": run_info.get("steps_completed", 0),
        "steps_failed": run_info.get("steps_failed", 0),
        "duration_seconds": round(duration, 1),
    })

    return json.dumps({
        "run_id": run_id,
        "status": "done",
        "steps_completed": run_info.get("steps_completed", 0),
        "steps_failed": run_info.get("steps_failed", 0),
        "duration_seconds": round(duration, 1),
    })


# -------------------------------------------------------------------
# Consensus mode — post request, collect proposals, vote
# -------------------------------------------------------------------

# Active consensus requests keyed by request_id
_active_consensus: dict[str, dict] = {}


def _consensus_state_path(run_id: str) -> Path:
    return _runs_dir() / run_id / "consensus_state.json"


def _save_consensus_state(request_id: str) -> None:
    """Persist consensus state for a request_id to its run directory."""
    state = _active_consensus.get(request_id)
    if not state:
        return
    run_id = state.get("run_id", "")
    if not run_id:
        return
    path = _consensus_state_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {**state, "request_id": request_id}
    path.write_text(json.dumps(data, indent=2))


def _load_consensus_state(request_id: str) -> dict | None:
    """Try to load consensus state from run directories if not in memory."""
    if request_id in _active_consensus:
        return _active_consensus[request_id]
    runs = _runs_dir()
    if runs.exists():
        for run_dir in runs.iterdir():
            state_path = run_dir / "consensus_state.json"
            if state_path.exists():
                try:
                    data = json.loads(state_path.read_text())
                    if data.get("request_id") == request_id:
                        _active_consensus[request_id] = data
                        return data
                except (json.JSONDecodeError, OSError):
                    continue
    return None


async def _solve_consensus(
    task: str,
    strategy: str = "majority",
    min_proposals: int = 3,
    max_proposals: int = 10,
    timeout_hours: float = 24.0,
    scope_id: str = "default",
) -> str:
    """Consensus solve: post DECOMP_REQUEST to knowledge.db, return request_id.

    Other sessions see this via multihead_check_inbox and submit proposals.
    Caller polls with multihead_collect_votes to check progress and finalize.
    """
    import uuid

    from ._core import _get_ks

    ks = _get_ks()

    # Generate IDs
    try:
        run_id = _generate_run_id()
    except ImportError:
        run_id = f"run_consensus_{int(time.time())}"

    request_claim_id = f"clm_{uuid.uuid4().hex[:24].upper()}"
    short_id = uuid.uuid4().hex[:8]

    # Build the DECOMP_REQUEST claim (same pattern as coordinator._post_task_request)
    from multihead.knowledge_models import (
        Claim,
        ClaimCanonical,
        ClaimScope,
        ClaimStatus,
        ClaimType,
        EntityRef,
        Provenance,
        ScopeType,
        ValueObject,
    )

    deadline = datetime.now(timezone.utc) + timedelta(hours=timeout_hours)

    request = Claim(
        claim_id=request_claim_id,
        claim_type=ClaimType.QUESTION,
        claim_status=ClaimStatus.PROPOSED,
        scope=ClaimScope(
            scope_type=ScopeType.PROJECT,
            scope_id=scope_id,
            visibility="project",
            valid_from=datetime.now(timezone.utc),
            valid_to=deadline,
        ),
        canonical=ClaimCanonical(
            claim_key=f"action.{scope_id}.work_order.{short_id}",
            subject=EntityRef(
                entity_type="task",
                entity_id=f"task_{short_id}",
                label="Consensus Solve",
            ),
            predicate="needs_decomposition",
            object=ValueObject(value_type="string", value=task),
        ),
        statement=(
            f"TASK DECOMPOSITION REQUEST (CONSENSUS)\n\n"
            f"FROM: claude-multihead-main (Coordinator)\n"
            f"TASK: {task}\n\n"
            f"Agents: Please decompose this task and submit a proposal.\n"
            f"Post a response claim with related_claim_ids: [\"{request_claim_id}\"]\n\n"
            f"KEY CONVENTION:\n"
            f"- Proposals: action.{scope_id}.proposal.{{your_short_id}}\n"
            f"- Use multihead_deposit_action for proper deadline tracking.\n\n"
            f"STRATEGY: {strategy}\n"
            f"MIN PROPOSALS: {min_proposals}\n"
            f"MAX PROPOSALS: {max_proposals}\n"
            f"TIMEOUT: {timeout_hours}h from posting\n"
            f"ROUNDS: Up to 2 (initial proposals + refinement)\n\n"
            f"Posted: {datetime.now(timezone.utc).isoformat()}"
        ),
        confidence=0.9,
        provenance=Provenance(
            produced_by={"id": "claude-multihead-main", "method": "mcp_consensus_solve"},
        ),
    )

    ks.insert_claim(request)

    # Track consensus state
    plan_info = {
        "task": task,
        "strategy": strategy,
        "mode": "consensus",
        "min_proposals": min_proposals,
        "max_proposals": max_proposals,
        "timeout_hours": timeout_hours,
    }
    _init_run(run_id, task, plan_info)

    _active_consensus[request_claim_id] = {
        "run_id": run_id,
        "task": task,
        "strategy": strategy,
        "min_proposals": min_proposals,
        "max_proposals": max_proposals,
        "scope_id": scope_id,
        "started_at": time.time(),
        "deadline": deadline.isoformat(),
        "round": 1,
    }
    _save_consensus_state(request_claim_id)

    _append_event(run_id, "consensus_request_posted", None, {
        "request_claim_id": request_claim_id,
        "min_proposals": min_proposals,
        "timeout_hours": timeout_hours,
    })

    return json.dumps({
        "run_id": run_id,
        "request_id": request_claim_id,
        "status": "collecting_proposals",
        "mode": "consensus",
        "min_proposals": min_proposals,
        "strategy": strategy,
        "deadline": deadline.isoformat(),
        "instructions": (
            "Consensus request posted to knowledge.db.\n"
            "Other sessions will see this via multihead_check_inbox.\n\n"
            "To check progress:\n"
            f"  multihead_collect_votes(request_id=\"{request_claim_id}\")\n\n"
            "This will show proposals received and run consensus voting\n"
            "once min_proposals is met. You can call it periodically.\n"
            f"Deadline: {timeout_hours}h from now."
        ),
    }, indent=2)


async def _collect_and_vote(request_id: str, force_vote: bool = False) -> str:
    """Poll for proposals on a consensus request and optionally run voting.

    Returns current status. Runs consensus vote when min_proposals met
    or force_vote=True.
    """
    from ._core import _get_ks

    ks = _get_ks()
    state = _load_consensus_state(request_id)

    if not state:
        # Try to reconstruct from knowledge.db
        try:
            claim = ks.get_claim(request_id)
            if not claim:
                return json.dumps({"error": f"Request {request_id} not found"})
            state = {
                "run_id": f"run_consensus_{request_id[-8:]}",
                "task": claim.canonical.object.value if hasattr(claim.canonical.object, "value") else "",
                "strategy": "majority",
                "min_proposals": 3,
                "max_proposals": 10,
                "scope_id": claim.scope.scope_id if hasattr(claim.scope, "scope_id") else "default",
                "started_at": time.time(),
                "round": 1,
            }
            _active_consensus[request_id] = state
        except Exception as e:
            return json.dumps({"error": f"Cannot find request: {e}"})

    # Collect proposals — claims with related_claim_ids containing request_id
    proposals = []
    try:
        if hasattr(ks, "get_responses_to_claim"):
            proposals = ks.get_responses_to_claim(request_id, limit=state["max_proposals"])
        else:
            # Fallback: search by related_json
            import sqlite3
            conn = sqlite3.connect(ks.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT claim_id FROM claims WHERE related_json LIKE ? "
                "AND claim_id != ? ORDER BY created_at DESC LIMIT ?",
                (f'%{request_id}%', request_id, state["max_proposals"]),
            ).fetchall()
            conn.close()
            for row in rows:
                try:
                    c = ks.get_claim(row["claim_id"])
                    if c:
                        proposals.append(c)
                except Exception:
                    pass
    except Exception as e:
        logger.warning("Error collecting proposals: %s", e)

    proposal_summaries = []
    for p in proposals:
        sender = p.provenance.produced_by.get("id", "unknown") if hasattr(p.provenance, "produced_by") else "unknown"
        proposal_summaries.append({
            "claim_id": p.claim_id,
            "from": sender,
            "statement": p.statement[:500],
            "created_at": p.provenance.created_at.isoformat() if hasattr(p.provenance, "created_at") and p.provenance.created_at else "",
        })

    enough = len(proposals) >= state["min_proposals"]

    # Run vote if we have enough proposals or force_vote
    if (enough or force_vote) and len(proposals) > 0:
        # Simple voting: use strategy to pick winner
        winner = None
        if len(proposals) == 1:
            winner = proposals[0]
        else:
            # Score by: earlier = lower priority, let strategy decide
            strategy = state.get("strategy", "majority")
            if strategy == "first_to_ahead":
                winner = proposals[0]  # First responder wins
            else:
                # Majority/weighted: all proposals are unique, pick first (coordinator tiebreak)
                # In real multi-round, we'd do actual voting. For now, first proposal wins tiebreak.
                winner = proposals[0]

        winner_id = winner.claim_id
        winner_sender = winner.provenance.produced_by.get("id", "unknown")

        _append_event(state.get("run_id", ""), "consensus_vote_complete", None, {
            "proposals_count": len(proposals),
            "winner_claim_id": winner_id,
            "winner_agent": winner_sender,
            "strategy": state.get("strategy", "majority"),
        })

        return json.dumps({
            "request_id": request_id,
            "status": "voted",
            "proposals_received": len(proposals),
            "proposals": proposal_summaries,
            "winner": {
                "claim_id": winner_id,
                "from": winner_sender,
                "statement": winner.statement[:1000],
            },
            "instructions": (
                "Consensus vote complete. The winning proposal is above.\n"
                "You can now execute the winning plan using self-solve flow:\n"
                f"  multihead_complete_step(run_id=\"{state.get('run_id', '')}\", step_id=..., output=...)\n"
                f"  multihead_finalize_solve(run_id=\"{state.get('run_id', '')}\")"
            ),
        }, indent=2)

    # Not enough yet — return status
    elapsed_h = (time.time() - state["started_at"]) / 3600
    return json.dumps({
        "request_id": request_id,
        "status": "collecting",
        "proposals_received": len(proposals),
        "min_proposals": state["min_proposals"],
        "proposals": proposal_summaries,
        "elapsed_hours": round(elapsed_h, 2),
        "deadline": state.get("deadline", ""),
        "instructions": (
            f"Waiting for proposals: {len(proposals)}/{state['min_proposals']} received.\n"
            "Call this tool again later to check progress.\n"
            "Use force_vote=true to vote with whatever proposals are available."
        ),
    }, indent=2)


# -------------------------------------------------------------------
# Server proxy (original behavior)
# -------------------------------------------------------------------

async def _solve(
    task: str,
    strategy: str = "first_to_ahead",
    max_steps: int = 20,
    enable_marketplace: bool = False,
    timeout: float = 240.0,
    dry_run: bool = False,
) -> str:
    """Proxy solve request to the MultiHead API."""
    _request = sys.modules["multihead.mcp_server"]._request

    payload: dict = {
        "task": task,
        "strategy": strategy,
        "max_steps": max_steps,
        "enable_marketplace": enable_marketplace,
        "timeout": timeout,
        "dry_run": dry_run,
    }
    try:
        result = await _request("POST", "/solve", json=payload)
        return json.dumps(result, indent=2)
    except httpx.ConnectError:
        return "Error: MultiHead server not running. Start it with: multihead serve"
    except httpx.HTTPStatusError as e:
        return f"Error ({e.response.status_code}): {e.response.text}"
    except Exception as e:
        return f"Error: {e}"
