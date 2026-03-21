"""Direct solve — no MCP, no serve, no transport layer.

Pure Python module that any Claude Code session can import to use
self-solve and consensus-solve. All operations are SQLite reads/writes
and local file I/O.

Usage (self-solve):
    from multihead.solve_direct import SolveDirect
    s = SolveDirect()
    result = s.start("fix the layout bug")
    # ... execute steps ...
    s.complete_step(result.run_id, "1.1", "Added clamp to line 42")
    s.finalize(result.run_id)

Usage (consensus):
    from multihead.solve_direct import SolveDirect
    s = SolveDirect()
    req = s.request_consensus("Should we refactor auth?", min_proposals=3, timeout_hours=24)
    # ... wait for other agents to propose ...
    status = s.check_proposals(req.request_id)
    # ... when enough proposals ...
    winner = s.vote(req.request_id)

Usage (CLI):
    python -m multihead.solve_direct start "fix the layout bug"
    python -m multihead.solve_direct step <run_id> 1.1 "Added clamp"
    python -m multihead.solve_direct finalize <run_id>
    python -m multihead.solve_direct consensus "Should we refactor auth?"
    python -m multihead.solve_direct proposals <request_id>
    python -m multihead.solve_direct vote <request_id>
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Result types
# -------------------------------------------------------------------

@dataclass
class SolveResult:
    """Returned by start() — contains run_id and knowledge context."""
    run_id: str
    knowledge_context: str
    knowledge_keys: list[str]
    template: str

    def __str__(self) -> str:
        return (
            f"SolveResult(run_id={self.run_id}, "
            f"knowledge_claims={len(self.knowledge_keys)})"
        )


@dataclass
class ConsensusResult:
    """Returned by request_consensus()."""
    run_id: str
    request_id: str
    deadline: str
    min_proposals: int
    strategy: str

    def __str__(self) -> str:
        return (
            f"ConsensusResult(request_id={self.request_id}, "
            f"deadline={self.deadline})"
        )


@dataclass
class ProposalStatus:
    """Returned by check_proposals()."""
    request_id: str
    proposals: list[dict]
    enough: bool
    elapsed_hours: float
    deadline: str


@dataclass
class VoteResult:
    """Returned by vote()."""
    request_id: str
    run_id: str
    winner_claim_id: str
    winner_agent: str
    winner_statement: str
    total_proposals: int


# -------------------------------------------------------------------
# Core class
# -------------------------------------------------------------------

class SolveDirect:
    """Direct solve interface — SQLite + file I/O, no transport layer.

    Wraps KnowledgeStore for RAG context and claim operations,
    plus local file I/O for run tracking. Same logic as MCP tools
    but without the stdio/JSON-RPC wrapper.
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        agent_id: str = "claude-multihead-main",
    ):
        self.data_dir = Path(
            data_dir or os.environ.get("MULTIHEAD_DATA_DIR", str(Path.home() / ".multihead"))
        )
        self.agent_id = agent_id
        self._ks = None
        # In-memory tracking for active consensus requests (backed by file)
        self._consensus_state: dict[str, dict] = {}

    # -------------------------------------------------------------------
    # Consensus state persistence
    # -------------------------------------------------------------------

    def _consensus_state_path(self, run_id: str) -> Path:
        return self.runs_dir / run_id / "consensus_state.json"

    def _save_consensus_state(self, request_id: str) -> None:
        """Persist consensus state for a request_id to its run directory."""
        state = self._consensus_state.get(request_id)
        if not state:
            return
        run_id = state.get("run_id", "")
        if not run_id:
            return
        path = self._consensus_state_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Store request_id inside the state so we can reload it
        data = {**state, "request_id": request_id}
        path.write_text(json.dumps(data, indent=2))

    def _load_consensus_state(self, request_id: str) -> dict | None:
        """Try to load consensus state from any run directory matching request_id."""
        if request_id in self._consensus_state:
            return self._consensus_state[request_id]
        # Scan run directories for consensus_state.json containing this request_id
        if self.runs_dir.exists():
            for run_dir in self.runs_dir.iterdir():
                state_path = run_dir / "consensus_state.json"
                if state_path.exists():
                    try:
                        data = json.loads(state_path.read_text())
                        if data.get("request_id") == request_id:
                            self._consensus_state[request_id] = data
                            return data
                    except (json.JSONDecodeError, OSError):
                        continue
        return None

    @property
    def ks(self):
        """Lazy-init KnowledgeStore."""
        if self._ks is None:
            from multihead.knowledge_store import KnowledgeStore
            self._ks = KnowledgeStore(self.data_dir / "knowledge.db")
        return self._ks

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    # -------------------------------------------------------------------
    # Run ID generation
    # -------------------------------------------------------------------

    @staticmethod
    def _make_run_id(prefix: str = "run") -> str:
        try:
            import ulid
            return f"{prefix}_{ulid.ULID()!s}"
        except ImportError:
            return f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    # -------------------------------------------------------------------
    # Knowledge RAG + Query
    # -------------------------------------------------------------------

    def gather_context(self, goal: str, limit: int = 20) -> tuple[str, list[str]]:
        """Query knowledge.db for claims relevant to the goal using FTS.

        Uses full-text search (FTS5) for relevance-ranked results.
        Falls back to keyword matching if FTS unavailable.

        Returns (formatted_text, list_of_claim_keys).
        """
        try:
            results = self.ks.search_claims_hybrid(goal, limit=limit, min_confidence=0.5)
            if results:
                texts = [f"- [{key}] {stmt[:200]}" for key, stmt, _conf in results]
                keys = [key for key, _stmt, _conf in results]
                return "\n".join(texts), keys
        except Exception:
            pass

        # Fallback: keyword matching on recent claims
        stop_words = {
            "the", "a", "an", "in", "on", "at", "to", "for", "of", "is",
            "and", "or", "with", "from", "by", "it", "this", "that", "be",
            "are", "was", "do", "does", "did", "not", "no", "we", "i", "my",
            "our", "us", "should", "can", "will", "how", "what", "when",
        }
        words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", goal.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        if not keywords:
            return "(no keywords extracted)", []

        try:
            all_claims = self.ks.list_claims(status="accepted", limit=200)
            texts: list[str] = []
            keys: list[str] = []
            for c in all_claims:
                stmt = (c.statement or "").lower()
                if any(kw in stmt for kw in keywords):
                    key = c.canonical.claim_key or ""
                    texts.append(f"- [{key}] {c.statement[:200]}")
                    keys.append(key)
                    if len(texts) >= limit:
                        break
            if not texts:
                return "(no relevant claims found)", []
            return "\n".join(texts), keys
        except Exception as e:
            return f"(knowledge query error: {e})", []

    def query(
        self,
        question: str,
        limit: int = 15,
        scope_id: str | None = None,
        min_confidence: float = 0.5,
    ) -> list[dict]:
        """Ask the knowledge base a question. Returns relevant claims.

        This is the "what do you know about X?" interface.

        Args:
            question: Natural language question or topic
            limit: Max results
            scope_id: Filter by scope (e.g. "myproject", "multihead"). None = all.
            min_confidence: Minimum confidence threshold

        Returns:
            List of dicts with claim_key, statement, confidence, scope_id, claim_type
        """
        # FTS search
        results = []
        try:
            fts_results = self.ks.search_claims_hybrid(
                question, limit=limit, min_confidence=min_confidence,
            )
            if fts_results:
                # Enrich with scope and type info
                for key, stmt, conf in fts_results:
                    results.append({
                        "claim_key": key,
                        "statement": stmt,
                        "confidence": conf,
                    })
        except Exception:
            pass

        # If FTS returned nothing, try broader keyword search
        if not results:
            stop_words = {
                "the", "a", "an", "in", "on", "at", "to", "for", "of", "is",
                "and", "or", "with", "from", "by", "what", "how", "do", "we",
            }
            words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", question.lower())
            keywords = [w for w in words if w not in stop_words and len(w) > 2]
            if keywords:
                try:
                    all_claims = self.ks.list_claims(status="accepted", limit=500)
                    for c in all_claims:
                        stmt = (c.statement or "").lower()
                        if any(kw in stmt for kw in keywords):
                            results.append({
                                "claim_key": c.canonical.claim_key or "",
                                "statement": c.statement,
                                "confidence": c.confidence,
                            })
                            if len(results) >= limit:
                                break
                except Exception:
                    pass

        # Filter by scope if requested
        if scope_id and results:
            # Re-query with scope filter
            filtered = []
            with self.ks._connect() as conn:
                for r in results:
                    row = conn.execute(
                        "SELECT scope_id FROM claims WHERE claim_key = ? LIMIT 1",
                        (r["claim_key"],),
                    ).fetchone()
                    if row and (row["scope_id"] == scope_id or scope_id == "all"):
                        filtered.append(r)
            results = filtered

        return results

    def stats(self) -> dict:
        """Return knowledge base statistics."""
        with self.ks._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
            by_status = dict(conn.execute(
                "SELECT claim_status, COUNT(*) FROM claims GROUP BY claim_status"
            ).fetchall())
            by_scope = dict(conn.execute(
                "SELECT scope_id, COUNT(*) FROM claims GROUP BY scope_id ORDER BY COUNT(*) DESC LIMIT 10"
            ).fetchall())
            conflicts = conn.execute("SELECT COUNT(*) FROM claim_conflicts").fetchone()[0]
        return {
            "total_claims": total,
            "by_status": by_status,
            "top_scopes": by_scope,
            "conflicts": conflicts,
        }

    def _check_for_contradictions(self, claim_keys: list[str]) -> str:
        """Check if any returned claims are contested or have conflicts.

        Returns warning text if contradictions found, empty string otherwise.
        Circuit breaker: surfaces conflicts so the executor can decide.
        """
        if not claim_keys:
            return ""

        warnings = []
        try:
            with self.ks._connect() as conn:
                for key in claim_keys[:20]:
                    row = conn.execute(
                        "SELECT claim_id, claim_status, contested_reason "
                        "FROM claims WHERE claim_key = ? AND claim_status = 'contested' LIMIT 1",
                        (key,),
                    ).fetchone()
                    if row:
                        reason = row["contested_reason"] or "unknown reason"
                        warnings.append(
                            f"- [{key}] is CONTESTED: {reason[:150]}"
                        )

                    # Also check stale
                    stale = conn.execute(
                        "SELECT claim_id FROM claims WHERE claim_key = ? AND claim_status = 'stale' LIMIT 1",
                        (key,),
                    ).fetchone()
                    if stale:
                        warnings.append(
                            f"- [{key}] is STALE — source file may have changed since this was observed"
                        )
        except Exception:
            pass

        if not warnings:
            return ""

        return (
            "The following claims in your context have known issues.\n"
            "DO NOT blindly trust them — verify against actual code before acting.\n\n"
            + "\n".join(warnings)
        )

    # -------------------------------------------------------------------
    # Run tracking (file I/O)
    # -------------------------------------------------------------------

    def _init_run(self, run_id: str, task: str, metadata: dict) -> Path:
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "artifacts").mkdir(exist_ok=True)
        self._append_event(run_id, "run_created", None, {
            "task": task, **metadata,
        })
        return run_dir

    def _append_event(
        self, run_id: str, kind: str, step_id: str | None, data: dict,
    ) -> None:
        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            logger.warning("Run dir %s does not exist", run_id)
            return
        event = {
            "event_id": f"evt_{int(time.time())}_{kind}",
            "run_id": run_id,
            "kind": kind,
            "step_id": step_id,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(run_dir / "events.jsonl", "a") as f:
            f.write(json.dumps(event) + "\n")

    # -------------------------------------------------------------------
    # Self-solve
    # -------------------------------------------------------------------

    def start(self, task: str, context: str = "") -> SolveResult:
        """Start a self-solve: gather RAG context, create run, return template.

        The caller reads code, decomposes, executes, and reports steps.
        Includes contradiction circuit breaker — surfaces conflicts in context.
        """
        knowledge_text, knowledge_keys = self.gather_context(task)
        run_id = self._make_run_id()

        self._init_run(run_id, task, {"mode": "self", "agent_id": self.agent_id})

        # Circuit breaker: check for contradictions in the returned context
        warnings = self._check_for_contradictions(knowledge_keys)

        template = (
            f"## Task\n{task}\n\n"
            f"## Additional Context\n{context or '(none)'}\n\n"
        )

        if warnings:
            template += f"## ⚠ CONTRADICTIONS DETECTED\n{warnings}\n\n"

        template += (
            f"## Codebase Knowledge (from knowledge.db)\n{knowledge_text}\n\n"
            f"## Instructions\n"
            f"You are the executor. Read relevant code, decompose into steps, "
            f"execute each step, then report:\n"
            f"  complete_step(run_id=\"{run_id}\", step_id=\"1.1\", output=\"...\")\n"
            f"  finalize(run_id=\"{run_id}\")\n"
        )

        logger.info("Self-solve started: %s (%d knowledge claims)", run_id, len(knowledge_keys))
        return SolveResult(
            run_id=run_id,
            knowledge_context=knowledge_text,
            knowledge_keys=knowledge_keys,
            template=template,
        )

    def complete_step(
        self, run_id: str, step_id: str, output: str, status: str = "completed",
    ) -> None:
        """Report completion of a solve step."""
        self._append_event(run_id, f"step_{status}", step_id, {
            "output": output[:2000],
            "status": status,
        })
        # Save full output as artifact
        run_dir = self.runs_dir / run_id
        if run_dir.exists():
            safe = step_id.replace("/", "_").replace(" ", "_")
            (run_dir / "artifacts" / f"{safe}_output.txt").write_text(output)
        logger.info("Step %s %s for %s", step_id, status, run_id)

    def finalize(self, run_id: str) -> dict:
        """Finalize a solve run. Returns summary."""
        run_dir = self.runs_dir / run_id
        events_path = run_dir / "events.jsonl"

        # Calculate duration from events
        duration = 0.0
        steps_completed = 0
        steps_failed = 0
        if events_path.exists():
            lines = events_path.read_text().strip().split("\n")
            for line in lines:
                evt = json.loads(line)
                if evt["kind"] == "step_completed":
                    steps_completed += 1
                elif evt["kind"] == "step_failed":
                    steps_failed += 1
            if lines:
                first = json.loads(lines[0])
                first_ts = datetime.fromisoformat(first["timestamp"])
                duration = (datetime.now(timezone.utc) - first_ts).total_seconds()

        self._append_event(run_id, "run_completed", None, {
            "steps_completed": steps_completed,
            "steps_failed": steps_failed,
            "duration_seconds": round(duration, 1),
        })

        summary = {
            "run_id": run_id,
            "status": "done",
            "steps_completed": steps_completed,
            "steps_failed": steps_failed,
            "duration_seconds": round(duration, 1),
        }
        logger.info("Run %s finalized: %d completed, %d failed, %.1fs",
                     run_id, steps_completed, steps_failed, duration)
        return summary

    # -------------------------------------------------------------------
    # Consensus solve
    # -------------------------------------------------------------------

    def request_consensus(
        self,
        task: str,
        min_proposals: int = 3,
        max_proposals: int = 10,
        timeout_hours: float = 24.0,
        strategy: str = "majority",
        scope_id: str = "default",
    ) -> ConsensusResult:
        """Post a DECOMP_REQUEST to knowledge.db for other agents to see.

        Other sessions detect this via check_inbox / collab and submit proposals
        as response claims with related_claim_ids pointing to the request.
        """
        from multihead.knowledge_models import (
            Claim, ClaimCanonical, ClaimScope, ClaimStatus, ClaimType,
            EntityRef, Provenance, ScopeType, ValueObject,
        )

        run_id = self._make_run_id("run_consensus")
        request_id = f"clm_{uuid.uuid4().hex[:24].upper()}"
        short_id = uuid.uuid4().hex[:8]
        now = datetime.now(timezone.utc)
        deadline = now + timedelta(hours=timeout_hours)

        request = Claim(
            claim_id=request_id,
            claim_type=ClaimType.QUESTION,
            claim_status=ClaimStatus.PROPOSED,
            scope=ClaimScope(
                scope_type=ScopeType.PROJECT,
                scope_id=scope_id,
                visibility="project",
                valid_from=now,
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
                f"FROM: {self.agent_id} (Coordinator)\n"
                f"TASK: {task}\n\n"
                f"Agents: Please decompose this task and submit a proposal.\n"
                f"Post a response claim with related_claim_ids: [\"{request_id}\"]\n\n"
                f"STRATEGY: {strategy}\n"
                f"MIN PROPOSALS: {min_proposals}\n"
                f"MAX PROPOSALS: {max_proposals}\n"
                f"TIMEOUT: {timeout_hours}h from posting\n"
                f"ROUNDS: Up to 2 (initial proposals + refinement)\n\n"
                f"Posted: {now.isoformat()}"
            ),
            confidence=0.9,
            provenance=Provenance(
                produced_by={"id": self.agent_id, "method": "solve_direct"},
            ),
        )

        self.ks.insert_claim(request)

        self._init_run(run_id, task, {
            "mode": "consensus",
            "strategy": strategy,
            "min_proposals": min_proposals,
            "max_proposals": max_proposals,
            "timeout_hours": timeout_hours,
            "request_id": request_id,
        })

        self._consensus_state[request_id] = {
            "run_id": run_id,
            "task": task,
            "strategy": strategy,
            "min_proposals": min_proposals,
            "max_proposals": max_proposals,
            "scope_id": scope_id,
            "started_at": time.time(),
            "deadline": deadline.isoformat(),
        }
        self._save_consensus_state(request_id)

        self._append_event(run_id, "consensus_request_posted", None, {
            "request_id": request_id,
            "min_proposals": min_proposals,
            "timeout_hours": timeout_hours,
        })

        logger.info("Consensus request posted: %s (min=%d, timeout=%sh)",
                     request_id, min_proposals, timeout_hours)

        return ConsensusResult(
            run_id=run_id,
            request_id=request_id,
            deadline=deadline.isoformat(),
            min_proposals=min_proposals,
            strategy=strategy,
        )

    def check_proposals(self, request_id: str) -> ProposalStatus:
        """Check how many proposals have been submitted for a consensus request."""
        state = self._load_consensus_state(request_id) or {}

        proposals = self._get_proposals(request_id, state.get("max_proposals", 10))
        summaries = self._summarize_proposals(proposals)
        min_needed = state.get("min_proposals", 3)
        elapsed = (time.time() - state.get("started_at", time.time())) / 3600

        return ProposalStatus(
            request_id=request_id,
            proposals=summaries,
            enough=len(proposals) >= min_needed,
            elapsed_hours=round(elapsed, 2),
            deadline=state.get("deadline", ""),
        )

    def vote(self, request_id: str, force: bool = False) -> VoteResult | None:
        """Run consensus vote on collected proposals.

        Returns the winner, or None if not enough proposals (unless force=True).
        """
        state = self._load_consensus_state(request_id) or {}
        proposals = self._get_proposals(request_id, state.get("max_proposals", 10))
        min_needed = state.get("min_proposals", 3)

        if not proposals:
            return None
        if len(proposals) < min_needed and not force:
            return None

        # Pick winner based on strategy
        strategy = state.get("strategy", "majority")
        if len(proposals) == 1:
            winner = proposals[0]
        elif strategy == "first_to_ahead":
            winner = proposals[0]
        else:
            # All proposals are unique decompositions — first wins tiebreak
            # Future: actual voting with LLM scoring
            winner = proposals[0]

        winner_agent = "unknown"
        if hasattr(winner.provenance, "produced_by") and winner.provenance.produced_by:
            winner_agent = winner.provenance.produced_by.get("id", "unknown")

        run_id = state.get("run_id", "")
        if run_id:
            self._append_event(run_id, "consensus_vote_complete", None, {
                "proposals_count": len(proposals),
                "winner_claim_id": winner.claim_id,
                "winner_agent": winner_agent,
                "strategy": strategy,
            })

        logger.info("Consensus vote: %s wins (%d proposals, strategy=%s)",
                     winner_agent, len(proposals), strategy)

        return VoteResult(
            request_id=request_id,
            run_id=run_id,
            winner_claim_id=winner.claim_id,
            winner_agent=winner_agent,
            winner_statement=winner.statement[:2000],
            total_proposals=len(proposals),
        )

    # -------------------------------------------------------------------
    # Claim deposit (convenience wrapper)
    # -------------------------------------------------------------------

    def deposit_claim(
        self,
        statement: str,
        claim_key: str,
        claim_type: str = "observation",
        scope_id: str = "default",
        confidence: float = 0.8,
        related_claim_ids: list[str] | None = None,
    ) -> str:
        """Deposit a claim to knowledge.db with proper FTS indexing.

        Returns the claim_id.
        """
        from multihead.knowledge_models import (
            Claim, ClaimCanonical, ClaimScope, ClaimStatus, ClaimType as CT,
            EntityRef, Provenance, ScopeType, ValueObject,
        )

        claim_id = f"clm_{uuid.uuid4().hex[:24].upper()}"
        ct_map = {
            "fact": CT.FACT,
            "plan": CT.PLAN,
            "question": CT.QUESTION,
            "decision": CT.DECISION,
            "definition": CT.DEFINITION,
            "constraint": CT.CONSTRAINT,
            "preference": CT.PREFERENCE,
            "assumption": CT.ASSUMPTION,
            "risk": CT.RISK,
        }

        claim = Claim(
            claim_id=claim_id,
            claim_type=ct_map.get(claim_type, CT.FACT),
            claim_status=ClaimStatus.PROPOSED,
            scope=ClaimScope(
                scope_type=ScopeType.PROJECT,
                scope_id=scope_id,
                visibility="project",
                valid_from=datetime.now(timezone.utc),
            ),
            canonical=ClaimCanonical(
                claim_key=claim_key,
                subject=EntityRef(entity_type="claim", entity_id=claim_id),
                predicate="states",
                object=ValueObject(value_type="string", value=statement),
            ),
            statement=statement,
            confidence=confidence,
            provenance=Provenance(
                produced_by={"id": self.agent_id, "method": "solve_direct"},
            ),
            related_claim_ids=related_claim_ids or [],
        )

        self.ks.insert_claim(claim)
        return claim_id

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _get_proposals(self, request_id: str, limit: int = 10) -> list:
        """Get proposal claims that reference request_id."""
        try:
            return self.ks.get_responses_to_claim(request_id, limit=limit)
        except Exception:
            # Fallback: raw SQL
            import sqlite3
            conn = sqlite3.connect(str(self.ks.db_path), timeout=10.0)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT claim_id FROM claims WHERE related_json LIKE ? "
                "AND claim_id != ? ORDER BY created_at DESC LIMIT ?",
                (f"%{request_id}%", request_id, limit),
            ).fetchall()
            conn.close()
            proposals = []
            for row in rows:
                try:
                    c = self.ks.get_claim(row["claim_id"])
                    if c:
                        proposals.append(c)
                except Exception:
                    pass
            return proposals

    @staticmethod
    def _summarize_proposals(proposals: list) -> list[dict]:
        summaries = []
        for p in proposals:
            sender = "unknown"
            if hasattr(p.provenance, "produced_by") and p.provenance.produced_by:
                sender = p.provenance.produced_by.get("id", "unknown")
            summaries.append({
                "claim_id": p.claim_id,
                "from": sender,
                "statement": p.statement[:500],
            })
        return summaries


# -------------------------------------------------------------------
# CLI interface
# -------------------------------------------------------------------

def main():
    """CLI entry point: python -m multihead.solve_direct <command> [args]"""
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    s = SolveDirect()

    if cmd == "start":
        if len(sys.argv) < 3:
            print("Usage: solve_direct start <task>")
            sys.exit(1)
        task = " ".join(sys.argv[2:])
        result = s.start(task)
        print(f"run_id: {result.run_id}")
        print(f"knowledge_claims: {len(result.knowledge_keys)}")
        print(f"\n{result.template}")

    elif cmd == "step":
        if len(sys.argv) < 5:
            print("Usage: solve_direct step <run_id> <step_id> <output>")
            sys.exit(1)
        s.complete_step(sys.argv[2], sys.argv[3], " ".join(sys.argv[4:]))
        print(f"Step {sys.argv[3]} recorded.")

    elif cmd == "finalize":
        if len(sys.argv) < 3:
            print("Usage: solve_direct finalize <run_id>")
            sys.exit(1)
        summary = s.finalize(sys.argv[2])
        print(json.dumps(summary, indent=2))

    elif cmd == "consensus":
        if len(sys.argv) < 3:
            print("Usage: solve_direct consensus <task> [--timeout-hours N] [--min-proposals N]")
            sys.exit(1)
        # Simple arg parsing
        task_parts = []
        timeout_hours = 24.0
        min_proposals = 3
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--timeout-hours" and i + 1 < len(sys.argv):
                timeout_hours = float(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == "--min-proposals" and i + 1 < len(sys.argv):
                min_proposals = int(sys.argv[i + 1])
                i += 2
            else:
                task_parts.append(sys.argv[i])
                i += 1
        task = " ".join(task_parts)
        result = s.request_consensus(task, min_proposals=min_proposals, timeout_hours=timeout_hours)
        print(f"request_id: {result.request_id}")
        print(f"run_id: {result.run_id}")
        print(f"deadline: {result.deadline}")
        print(f"\nOther agents will see this via check_inbox.")
        print(f"Check progress: python -m multihead.solve_direct proposals {result.request_id}")

    elif cmd == "proposals":
        if len(sys.argv) < 3:
            print("Usage: solve_direct proposals <request_id>")
            sys.exit(1)
        status = s.check_proposals(sys.argv[2])
        print(f"Proposals: {len(status.proposals)} (need {3 if not status.enough else 'met'})")
        print(f"Elapsed: {status.elapsed_hours}h")
        for p in status.proposals:
            print(f"  - [{p['from']}] {p['statement'][:100]}")

    elif cmd == "vote":
        if len(sys.argv) < 3:
            print("Usage: solve_direct vote <request_id> [--force]")
            sys.exit(1)
        force = "--force" in sys.argv
        winner = s.vote(sys.argv[2], force=force)
        if not winner:
            print("Not enough proposals yet. Use --force to vote anyway.")
        else:
            print(f"Winner: {winner.winner_agent}")
            print(f"Claim: {winner.winner_claim_id}")
            print(f"Proposals: {winner.total_proposals}")
            print(f"\n{winner.winner_statement[:500]}")

    elif cmd == "query":
        if len(sys.argv) < 3:
            print("Usage: solve_direct query <question> [--scope SCOPE] [--limit N]")
            sys.exit(1)
        query_parts = []
        scope_id = None
        limit = 15
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--scope" and i + 1 < len(sys.argv):
                scope_id = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--limit" and i + 1 < len(sys.argv):
                limit = int(sys.argv[i + 1])
                i += 2
            else:
                query_parts.append(sys.argv[i])
                i += 1
        question = " ".join(query_parts)
        results = s.query(question, limit=limit, scope_id=scope_id)
        print(f"Found {len(results)} relevant claims:\n")
        for r in results:
            conf = r.get("confidence", 0)
            print(f"  [{r['claim_key']}] (conf={conf:.2f})")
            print(f"    {r['statement'][:200]}")
            print()

    elif cmd == "stats":
        st = s.stats()
        print(f"Total claims: {st['total_claims']}")
        print(f"Conflicts: {st['conflicts']}")
        print("\nBy status:")
        for status, count in sorted(st["by_status"].items(), key=lambda x: -x[1]):
            print(f"  {status}: {count}")
        print("\nTop scopes:")
        for scope, count in list(st["top_scopes"].items())[:5]:
            print(f"  {scope}: {count}")

    else:
        print(f"Unknown command: {cmd}")
        print("Commands: start, step, finalize, consensus, proposals, vote, query, stats")
        sys.exit(1)


if __name__ == "__main__":
    main()
