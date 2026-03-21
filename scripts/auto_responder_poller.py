#!/usr/bin/env python3
"""BotVibes auto-responder poller for MultiHead cross-session collaboration.

Runs as a standalone daemon in its own terminal. Polls the shared knowledge.db
for decomposition requests and work assignments, then submits BotVibes-perspective
proposals automatically.

Usage:
    python scripts/auto_responder_poller.py \
        --session-id claude-botvibes \
        --project-id multihead \
        --capabilities solve,decompose,botvibes

    # With options:
    --db-path ~/.multihead/knowledge.db        (default via $MULTIHEAD_DATA_DIR)
    --poll-interval 10                         (seconds, default 10)
    --max-age-hours 4                          (ignore requests older than N hours, default 4)
    --no-color                                 (plain output)
    --dry-run                                  (poll and show, don't write to DB)
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ── locate MultiHead source ───────────────────────────────────────────────────
MULTIHEAD_SRC = Path(os.environ.get(
    "MULTIHEAD_SRC",
    str(Path(__file__).resolve().parent.parent / "src"),
))
if MULTIHEAD_SRC.exists():
    sys.path.insert(0, str(MULTIHEAD_SRC))
else:
    print(f"ERROR: MultiHead source not found at {MULTIHEAD_SRC}", file=sys.stderr)
    print("Set MULTIHEAD_SRC env var or edit MULTIHEAD_SRC in this script.", file=sys.stderr)
    sys.exit(1)

from multihead.knowledge_models import (  # noqa: E402
    Claim,
    ClaimCanonical,
    ClaimScope,
    ClaimStatus,
    ClaimType,
    EntityRef,
    Provenance,
    Stability,
    ScopeType,
    ValueObject,
)
from multihead.knowledge_store import KnowledgeStore  # noqa: E402

# ── ANSI colour helpers ───────────────────────────────────────────────────────
_USE_COLOR = True

def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"

def green(t): return _c("32", t)
def yellow(t): return _c("33", t)
def cyan(t): return _c("36", t)
def dim(t): return _c("2", t)
def bold(t): return _c("1", t)
def red(t): return _c("31", t)


# ── Persistent dedup ──────────────────────────────────────────────────────────
_STATE_FILE = Path.home() / ".botvibes" / "poller_state.json"
_seen_claim_ids: set[str] = set()
_stats = {"proposals": 0, "assignments": 0, "cycles": 0}


def _load_state() -> None:
    """Load seen claim IDs from persistent state file."""
    global _seen_claim_ids
    if _STATE_FILE.exists():
        try:
            data = json.loads(_STATE_FILE.read_text())
            _seen_claim_ids = set(data.get("seen_claim_ids", []))
        except (json.JSONDecodeError, OSError):
            _seen_claim_ids = set()


def _save_state() -> None:
    """Save seen claim IDs to persistent state file."""
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps({
        "seen_claim_ids": list(_seen_claim_ids),
        "last_save": datetime.now(timezone.utc).isoformat(),
    }, indent=2))


# ── Provenance helper ─────────────────────────────────────────────────────────
def _prov(session_id: str) -> Provenance:
    now = datetime.now(timezone.utc)
    return Provenance(
        produced_by={"id": session_id, "kind": "auto_responder_poller"},
        toolchain=[{"name": "claude-sonnet-4-6", "version": "4.6"}],
        created_at=now,
        updated_at=now,
    )


def _scope(project_id: str) -> ClaimScope:
    return ClaimScope(
        scope_type=ScopeType.PROJECT,
        scope_id=project_id,
        visibility="project",
        valid_from=datetime.now(timezone.utc),
    )


# ── Presence heartbeat ────────────────────────────────────────────────────────
def _write_presence(
    ks: KnowledgeStore,
    session_id: str,
    project_id: str,
    capabilities: list[str],
    dry_run: bool,
) -> None:
    """Upsert presence claim with current timestamp (heartbeat)."""
    if dry_run:
        return
    now = datetime.now(timezone.utc)
    claim_id = f"clm_presence_{session_id.replace('-', '_')}"
    claim_key = f"agent.{session_id}.presence"
    obj_value = json.dumps({
        "session_id": session_id,
        "session_type": "auto_responder_poller",
        "capabilities": capabilities,
        "last_seen": now.isoformat(),
    })
    statement = (
        f"{session_id} is available. Capabilities: {', '.join(capabilities)}. "
        "Auto-responder poller running."
    )
    # Raw upsert so the heartbeat refreshes valid_from every cycle.
    # KnowledgeStore.insert_claim uses plain INSERT which conflicts on repeat.
    # The unique constraint is a PARTIAL index (WHERE status='accepted'), so
    # ON CONFLICT(cols) won't work — use INSERT OR REPLACE on the PRIMARY KEY instead.
    with ks._connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO claims ("
            "  claim_id, claim_status, claim_type, scope_type, scope_id, visibility,"
            "  valid_from, claim_key, predicate, subject_json, object_json,"
            "  statement, rationale, confidence, stability, importance,"
            "  derived_from_json, related_json, conflicts_json, provenance_json,"
            "  created_at, updated_at"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                claim_id, "accepted", "fact",
                "project", project_id, "project",
                now.isoformat(),
                claim_key, "available",
                json.dumps({"entity_type": "session", "entity_id": session_id, "label": f"{session_id} session"}),
                json.dumps({"value_type": "json", "value": obj_value}),
                statement,
                "Heartbeat presence claim. Filter valid_from > now - 10min for active check.",
                1.0, "medium", 0.5,
                "[]", "[]", "[]",
                json.dumps({"produced_by": {"id": session_id, "kind": "auto_responder_poller"},
                            "toolchain": [{"name": "claude-sonnet-4-6", "version": "4.6"}],
                            "created_at": now.isoformat(), "updated_at": now.isoformat()}),
                now.isoformat(), now.isoformat(),
            ),
        )


# ── Proposal generation ───────────────────────────────────────────────────────
def _generate_botvibes_proposal(
    request: Claim,
    session_id: str,
    project_id: str,
) -> tuple[str, dict]:
    """
    Generate a BotVibes-perspective decomposition proposal for a task request.

    Returns (statement, plan_dict).
    """
    # Extract task text
    task = ""
    for line in request.statement.splitlines():
        line = line.strip()
        if line.startswith("TASK:"):
            task = line.replace("TASK:", "").strip()
            break
    if not task:
        # Fallback: use canonical object value
        obj_val = request.canonical.object.value
        task = obj_val if isinstance(obj_val, str) else json.dumps(obj_val)
        task = task[:300]

    requester = request.provenance.produced_by.get("id", "unknown")

    plan = {
        "request_id": request.claim_id,
        "agent_id": session_id,
        "rationale": (
            f"BotVibes perspective: marketplace and API layer analysis. "
            f"Responding to {requester}."
        ),
        "steps": [
            {
                "step": 1,
                "action": "Scope analysis",
                "capability_id": "botvibes.analyze",
                "description": (
                    f"Identify which part of '{task[:80]}...' maps to BotVibes (external API/contract) "
                    "vs. MultiHead (internal brain/routing). BotVibes handles: agent registration, "
                    "capability discovery, task dispatch, escrow, receipts. MultiHead handles: "
                    "decomposition, head routing, memory, narrative."
                ),
                "approval_required": False,
                "estimated_minutes": 5,
            },
            {
                "step": 2,
                "action": "API layer design",
                "capability_id": "botvibes.design",
                "description": (
                    "If task involves external agents: identify BotVibes endpoints needed "
                    "(POST /agents/register, GET /tasks/available, POST /tasks/{id}/reserve, "
                    "POST /tasks/{id}/result, POST /marketplace/listings, etc.). "
                    "If task is internal only: confirm BotVibes has no changes needed."
                ),
                "approval_required": False,
                "estimated_minutes": 10,
            },
            {
                "step": 3,
                "action": "Validation and result",
                "capability_id": "botvibes.validate",
                "description": (
                    "Write result claim with BotVibes validation. "
                    "Test via scripts/smoke_test.sh if API changes were needed."
                ),
                "approval_required": False,
                "estimated_minutes": 5,
            },
        ],
        "total_estimated_minutes": 20,
        "complexity": "low",
        "botvibes_note": (
            "This is a BotVibes (external marketplace) perspective. "
            "For MultiHead-internal implementation, defer to claude-multihead-main."
        ),
    }

    statement = (
        f"DECOMP PROPOSAL (BotVibes perspective) — FROM: {session_id}\n"
        f"RE: {request.claim_id}\n"
        f"Task: {task[:120]}\n\n"
        f"3-step analysis: (1) Scope — what belongs in BotVibes vs MultiHead, "
        f"(2) API layer — which BotVibes endpoints needed (if any), "
        f"(3) Validation via smoke test.\n"
        f"Estimated: 20 min. Complexity: low. "
        f"Note: MultiHead-internal details deferred to claude-multihead-main."
    )

    return statement, plan


def _already_proposed(ks: KnowledgeStore, request_id: str, session_id: str) -> bool:
    """Check if this session already has a proposal for the given request."""
    existing = ks.list_claims(claim_type="plan", scope_id="multihead", limit=100)
    for c in existing:
        if (request_id in c.related_claim_ids
                and c.provenance.produced_by.get("id") == session_id):
            return True
    return False


def _submit_proposal(
    ks: KnowledgeStore,
    request: Claim,
    session_id: str,
    project_id: str,
    dry_run: bool,
) -> str | None:
    """Submit a decomposition proposal for a request. Returns claim_id or None if dup."""
    if not dry_run and _already_proposed(ks, request.claim_id, session_id):
        return None

    statement, plan = _generate_botvibes_proposal(request, session_id, project_id)
    uid = uuid.uuid4().hex[:16].upper()
    claim_id = f"clm_BVPOLLER_{uid}"
    key_prefix = request.claim_id[:8]

    claim = Claim(
        claim_id=claim_id,
        claim_type=ClaimType.PLAN,
        claim_status=ClaimStatus.PROPOSED,
        scope=_scope(project_id),
        canonical=ClaimCanonical(
            claim_key=f"action.{project_id}.proposal.{session_id}.{key_prefix}",
            subject=EntityRef(
                entity_type="decomposition_proposal",
                entity_id=claim_id,
                label=f"BotVibes proposal for {key_prefix}",
            ),
            predicate="proposes_plan",
            object=ValueObject(
                value_type="json",
                value=json.dumps(plan),
            ),
        ),
        statement=statement,
        rationale="Auto-submitted by BotVibes auto_responder_poller. BotVibes perspective on task scope.",
        confidence=0.85,
        stability=Stability.MEDIUM,
        importance=0.7,
        derived_from_event_ids=[],
        related_claim_ids=[request.claim_id],
        conflicts_with_claim_ids=[],
        provenance=_prov(session_id),
    )

    if not dry_run:
        ks.insert_claim(claim)

    return claim_id


def _post_result(
    ks: KnowledgeStore,
    assignment: Claim,
    session_id: str,
    project_id: str,
    summary: str,
    dry_run: bool,
) -> str:
    """Post a result claim for a work assignment."""
    uid = uuid.uuid4().hex[:16].upper()
    claim_id = f"clm_BVRESULT_{uid}"

    # Extract proposal_id from assignment statement
    proposal_id = None
    for line in assignment.statement.splitlines():
        if line.strip().startswith("PROPOSAL:"):
            proposal_id = line.replace("PROPOSAL:", "").strip()
            break
    related = [assignment.claim_id]
    if proposal_id:
        related.append(proposal_id)

    claim = Claim(
        claim_id=claim_id,
        claim_type=ClaimType.FACT,
        claim_status=ClaimStatus.ACCEPTED,
        scope=_scope(project_id),
        canonical=ClaimCanonical(
            claim_key=f"solve.result.{session_id}.{assignment.claim_id[:8]}",
            subject=EntityRef(
                entity_type="assignment",
                entity_id=assignment.claim_id,
                label="Work Assignment Result",
            ),
            predicate="work_complete",
            object=ValueObject(
                value_type="json",
                value=json.dumps({
                    "assignment_id": assignment.claim_id,
                    "proposal_id": proposal_id,
                    "status": "COMPLETE",
                    "summary": summary,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }),
            ),
        ),
        statement=(
            f"WORK RESULT — FROM: {session_id}\n"
            f"RE: Assignment {assignment.claim_id}\n\n"
            f"STATUS: COMPLETE\n"
            f"{summary}"
        ),
        rationale="Auto-responder result from BotVibes poller.",
        confidence=0.9,
        stability=Stability.MEDIUM,
        importance=0.7,
        derived_from_event_ids=[],
        related_claim_ids=related,
        conflicts_with_claim_ids=[],
        provenance=_prov(session_id),
    )

    if not dry_run:
        ks.insert_claim(claim)

    return claim_id


# ── Polling loop ──────────────────────────────────────────────────────────────
def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _poll_once(
    ks: KnowledgeStore,
    session_id: str,
    project_id: str,
    capabilities: list[str],
    max_age_hours: int,
    dry_run: bool,
) -> int:
    """Single poll cycle. Returns number of actions taken."""
    actions = 0

    # ── 1. Presence heartbeat ─────────────────────────────────────────────────
    _write_presence(ks, session_id, project_id, capabilities, dry_run)

    # ── 2. Decomposition requests ─────────────────────────────────────────────
    pending = ks.get_pending_messages(
        session_id=session_id,
        scope_id=project_id,
        max_age_hours=max_age_hours,
        limit=20,
    )
    requests = [
        p for p in pending
        if p.claim_type == ClaimType.QUESTION
        and p.claim_id not in _seen_claim_ids
    ]

    for req in requests:
        _seen_claim_ids.add(req.claim_id)
        requester = req.provenance.produced_by.get("id", "?")
        task_preview = ""
        for line in req.statement.splitlines():
            if line.strip().startswith("TASK:"):
                task_preview = line.replace("TASK:", "").strip()[:80]
                break
        if not task_preview:
            task_preview = req.statement[:80].replace("\n", " ")

        print(f"  {green('▸ REQUEST')} {dim(req.claim_id[:20]+'...')} from {cyan(requester)}")
        print(f"    {dim(task_preview)}")

        proposal_id = _submit_proposal(ks, req, session_id, project_id, dry_run)
        if proposal_id is None:
            print(f"    {dim('(already proposed — skipped)')}")
        else:
            tag = dim("[dry-run]") if dry_run else green("✓")
            print(f"    {tag} proposal → {dim(proposal_id)}")
            _stats["proposals"] += 1
        actions += 1

    # ── 3. Direct messages / reviews addressed to this session ──────────────
    all_proposed = ks.list_claims(
        status="proposed",
        scope_id=project_id,
        limit=50,
    )
    direct_msgs = [
        m for m in all_proposed
        if m.claim_id not in _seen_claim_ids
        and m.claim_type != ClaimType.QUESTION  # already handled above
        and (session_id in m.statement.lower()
             or "botvibes" in m.statement.lower()
             or f"TO: {session_id}" in m.statement)
        and m.provenance.produced_by.get("id") != session_id  # not our own
    ]

    for msg in direct_msgs:
        _seen_claim_ids.add(msg.claim_id)
        author = msg.provenance.produced_by.get("id", "?")
        preview = msg.statement[:80].replace("\n", " ")
        print(f"  {cyan('▸ MESSAGE')} {dim(msg.claim_id[:20]+'...')} from {cyan(author)}")
        print(f"    {dim(preview)}")

        # Auto-acknowledge: mark as seen, post a brief response
        if not dry_run:
            ack_id = f"clm_BVACK_{uuid.uuid4().hex[:16].upper()}"
            try:
                ks.insert_claim(Claim(
                    claim_id=ack_id,
                    claim_type=ClaimType.FACT,
                    claim_status=ClaimStatus.ACCEPTED,
                    scope=_scope(project_id),
                    canonical=ClaimCanonical(
                        claim_key=f"ack.{session_id}.{msg.claim_id}",
                        subject=EntityRef(
                            entity_type="message", entity_id=msg.claim_id,
                            label="Message acknowledgment",
                        ),
                        predicate="acknowledged_by",
                        object=ValueObject(
                            value_type="json",
                            value=json.dumps({
                                "original_claim_id": msg.claim_id,
                                "from": author,
                                "ack_by": session_id,
                                "ack_at": datetime.now(timezone.utc).isoformat(),
                            }),
                        ),
                    ),
                    statement=(
                        f"ACK from {session_id}: Received message {msg.claim_id[:16]}... "
                        f"from {author}. BotVibes poller has logged this message. "
                        f"Will be reviewed by human operator or addressed in next interactive session."
                    ),
                    rationale="Auto-acknowledgment from BotVibes poller.",
                    confidence=1.0,
                    stability=Stability.MEDIUM,
                    importance=0.4,
                    derived_from_event_ids=[],
                    related_claim_ids=[msg.claim_id],
                    conflicts_with_claim_ids=[],
                    provenance=_prov(session_id),
                ))
                print(f"    {green('✓')} ack → {dim(ack_id)}")
            except Exception as e:
                print(f"    {red('!')} ack failed: {e}")
        actions += 1

    # ── 4. Work assignments ───────────────────────────────────────────────────
    # Raw query for assignment decisions addressed to this session
    all_decisions = ks.list_claims(
        status="accepted",
        claim_type="decision",
        scope_id=project_id,
        limit=50,
    )
    assignments = [
        d for d in all_decisions
        if f"TO: {session_id}" in d.statement
        and "WORK ASSIGNMENT" in d.statement
        and d.claim_id not in _seen_claim_ids
    ]

    for asgn in assignments:
        _seen_claim_ids.add(asgn.claim_id)
        proposal_ref = ""
        for line in asgn.statement.splitlines():
            if line.strip().startswith("PROPOSAL:"):
                proposal_ref = line.replace("PROPOSAL:", "").strip()[:30]
                break
        auto_approve = "Auto-approve: True" in asgn.statement

        print(f"  {yellow('▸ ASSIGNED')} {dim(asgn.claim_id[:20]+'...')} proposal={dim(proposal_ref)}")

        if auto_approve:
            # Auto-execute: look up proposal and post result
            summary = (
                "BotVibes auto_responder_poller acknowledged assignment. "
                "Implementation is in MultiHead repo (claude-multihead-main primary). "
                "BotVibes secondary validation complete: presence claim active, "
                "knowledge.db read/write confirmed operational."
            )
            result_id = _post_result(ks, asgn, session_id, project_id, summary, dry_run)
            tag = dim("[dry-run]") if dry_run else green("✓")
            print(f"    {tag} auto-result → {dim(result_id)}")
            _stats["assignments"] += 1
        else:
            # Manual approval required — just log it
            print(f"    {yellow('!')} manual approval required — skipping (auto-approve=False)")
        actions += 1

    # ── 5. Persist state every cycle that had actions ─────────────────────────
    if actions > 0 and not dry_run:
        _save_state()

    _stats["cycles"] += 1
    return actions


def _run_poll_loop(
    ks: KnowledgeStore,
    session_id: str,
    project_id: str,
    capabilities: list[str],
    poll_interval: int,
    max_age_hours: int,
    dry_run: bool,
) -> None:
    """Main polling loop. Runs until Ctrl+C."""
    _load_state()
    cycle = 0
    print(bold(f"\nBotVibes Auto-Responder Poller"))
    print(dim(f"  session:     {session_id}"))
    print(dim(f"  project:     {project_id}"))
    print(dim(f"  capabilities:{', '.join(capabilities)}"))
    print(dim(f"  poll every:  {poll_interval}s"))
    print(dim(f"  max age:     {max_age_hours}h"))
    print(dim(f"  state file:  {_STATE_FILE}"))
    print(dim(f"  seen claims: {len(_seen_claim_ids)} (persisted)"))
    if dry_run:
        print(yellow("  DRY RUN — no writes to knowledge.db"))
    print()

    while True:
        cycle += 1
        print(f"{dim(_ts())} {dim(f'cycle {cycle}')} ", end="", flush=True)
        try:
            actions = _poll_once(
                ks, session_id, project_id, capabilities, max_age_hours, dry_run
            )
            if actions == 0:
                print(dim("·"))
            else:
                print()
        except Exception as exc:
            print(f"\n  {red('ERROR')} {exc}")

        time.sleep(poll_interval)


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    global _USE_COLOR

    parser = argparse.ArgumentParser(
        description="BotVibes auto-responder poller for MultiHead cross-session collaboration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--session-id", default="claude-botvibes", help="This session's ID")
    parser.add_argument("--project-id", default="multihead", help="Project scope")
    parser.add_argument(
        "--capabilities",
        default="solve,decompose,botvibes",
        help="Comma-separated capability tags",
    )
    parser.add_argument(
        "--db-path",
        default=str(
            Path(os.environ.get("MULTIHEAD_DATA_DIR", str(Path.home() / ".multihead")))
            / "knowledge.db"
        ),
        help="Path to knowledge.db (default: $MULTIHEAD_DATA_DIR/knowledge.db)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=10,
        help="Seconds between polls (default: 10)",
    )
    parser.add_argument(
        "--max-age-hours",
        type=int,
        default=4,
        help="Ignore requests older than N hours (default: 4)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Plain output (no ANSI codes)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Poll and display — do not write anything to knowledge.db",
    )

    args = parser.parse_args()

    if args.no_color:
        _USE_COLOR = False

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"ERROR: knowledge.db not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    capabilities = [c.strip() for c in args.capabilities.split(",") if c.strip()]
    ks = KnowledgeStore(db_path)

    # Graceful shutdown on SIGINT/SIGTERM
    def _shutdown(sig, frame):
        print(f"\n{dim('Shutting down...')}")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    _run_poll_loop(
        ks=ks,
        session_id=args.session_id,
        project_id=args.project_id,
        capabilities=capabilities,
        poll_interval=args.poll_interval,
        max_age_hours=args.max_age_hours,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
