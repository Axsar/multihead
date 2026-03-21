"""SolveCoordinator: distributed task solving via multi-agent consensus.

Coordinates distributing tasks to multiple agents, collecting proposals,
running consensus voting, and orchestrating execution.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import signal
import uuid
from datetime import datetime, timezone
from typing import Any

from ..consensus import ConsensusEngine, ConsensusStrategy
from ..head_manager import HeadManager
from ..knowledge_models import (
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
from ..knowledge_store import KnowledgeStore
from ..orchestrator import Orchestrator

from .config import SolveConfig, SolveResult
from .discovery import (
    discover_active_sessions,
    mark_session_offline,
    write_presence_claim,
)
from .onboarding import (
    _load_seen_sessions,
    _save_seen_sessions,
    _show_onboarding_messages,
)
from .config import prompt_multi_session
from .prompts import load_prompt

logger = logging.getLogger(__name__)


class SolveCoordinator:
    """Coordinates distributed task solving across multiple agents."""

    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        head_manager: HeadManager,
        orchestrator: Orchestrator,
        config: SolveConfig | None = None,
        explicit_auto_approve: bool | None = None,
    ):
        self.knowledge_store = knowledge_store
        self.head_manager = head_manager
        self.orchestrator = orchestrator
        self.config = config or SolveConfig()
        self.explicit_auto_approve = explicit_auto_approve

        # Write presence claim to advertise availability
        write_presence_claim(
            self.knowledge_store,
            self.config.session_id,
            self.config.project_id,
        )

        # Register cleanup handlers for graceful shutdown
        self._register_cleanup_handlers()

    def _cleanup(self):
        """Mark session offline on shutdown."""
        try:
            mark_session_offline(
                self.knowledge_store,
                self.config.session_id,
                self.config.project_id,
            )
        except Exception as e:
            logger.warning("Failed to mark session offline during cleanup: %s", e)

    def _register_cleanup_handlers(self):
        """Register signal handlers and atexit to mark session offline."""
        # Register atexit handler for normal termination
        atexit.register(self._cleanup)

        # Register signal handlers for SIGTERM and SIGINT
        def signal_handler(signum, frame):
            logger.info("Received signal %s, marking session offline", signum)
            self._cleanup()
            # Re-raise the signal to allow normal termination
            signal.signal(signum, signal.SIG_DFL)
            signal.raise_signal(signum)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    async def solve(self, task: str) -> SolveResult:
        """
        Distribute a task to agents, collect proposals, vote, and execute.

        Flow:
        1. Post task decomposition request to knowledge.db
        2. Wait for agent proposals (timeout: proposal_timeout_seconds)
        3. Run consensus voting on proposals
        4. Assign winning proposal to executing agent
        5. Monitor execution (if auto_approve enabled)
        6. Collect and return results

        Args:
            task: Natural language task description

        Returns:
            SolveResult with outcome details
        """
        logger.info("Starting distributed solve for task: %s", task[:60])

        # 0. Smart default: detect active sessions and adapt behavior
        other_sessions = discover_active_sessions(
            self.knowledge_store,
            self.config.project_id,
            self.config.session_id,
        )

        # Onboarding UX: track seen sessions and show friendly messages
        seen_sessions = _load_seen_sessions()
        is_first_run = len(seen_sessions) == 0
        current_session_ids = {s["session_id"] for s in other_sessions}
        new_sessions = [sid for sid in current_session_ids if sid not in seen_sessions]

        # Update seen sessions
        seen_sessions.update(current_session_ids)
        seen_sessions.add(self.config.session_id)  # Remember ourselves too
        _save_seen_sessions(seen_sessions)

        # Show onboarding messages
        _show_onboarding_messages(is_first_run, new_sessions, len(other_sessions))

        # Determine effective auto_approve based on context
        if self.explicit_auto_approve is not None:
            # User passed --auto-approve flag explicitly - honor it always
            effective_auto_approve = self.explicit_auto_approve
            effective_solo_mode = effective_auto_approve
            logger.info("Explicit --auto-approve=%s set, overriding detection", effective_auto_approve)
        elif len(other_sessions) == 0:
            # Solo mode - instant execution, no waiting
            effective_auto_approve = True
            effective_solo_mode = True
            logger.info("Solo mode detected - auto-approve enabled (instant execution)")
        else:
            # Multi-agent mode - prompt user
            logger.info("Multi-agent mode detected: %d other session(s)", len(other_sessions))
            skip_proposals = prompt_multi_session(other_sessions)
            if skip_proposals:
                # User chose to proceed solo
                effective_auto_approve = True
                effective_solo_mode = True
                logger.info("User chose solo mode - proceeding without proposals")
            else:
                # User wants to wait for proposals
                effective_auto_approve = False
                effective_solo_mode = False
                logger.info("User chose multi-session - waiting for proposals")

        # Update config with effective auto_approve
        self.config.auto_approve = effective_auto_approve

        # 1. Post task request (audit trail)
        request_id = await self._post_task_request(task)
        logger.info("Task request posted: %s", request_id)

        # 2. Solo mode: self-decompose and return plan to caller
        if effective_solo_mode:
            logger.info("Solo mode: self-decomposing and returning plan to caller")
            proposals = await self._self_propose(task, request_id)
            if not proposals:
                return SolveResult(
                    task=task, request_id=request_id, proposals_received=0,
                    winning_proposal_id=None, assigned_agent=None,
                    execution_started=False, result_claim_id=None,
                    success=False, error="Self-decomposition failed",
                )
            plan = proposals[0]
            return SolveResult(
                task=task,
                request_id=request_id,
                proposals_received=1,
                winning_proposal_id=plan.claim_id,
                assigned_agent=self.config.session_id,
                execution_started=False,  # Caller is the executor
                result_claim_id=None,
                success=True,
                decomposition=plan.statement,
            )

        # --- Multi-agent mode below ---

        # 2b. Wait for external proposals
        proposals = await self._collect_proposals(request_id)
        logger.info("Collected %d proposal(s)", len(proposals))

        if not proposals:
            return SolveResult(
                task=task,
                request_id=request_id,
                proposals_received=0,
                winning_proposal_id=None,
                assigned_agent=None,
                execution_started=False,
                result_claim_id=None,
                success=False,
                error="No proposals received within timeout",
            )

        # 3. Run consensus voting (if multiple proposals)
        if len(proposals) == 1:
            winning_proposal = proposals[0]
            logger.info("Single proposal, skipping vote: %s", winning_proposal.claim_id)
        else:
            winning_proposal = await self._vote_on_proposals(proposals)
            logger.info("Consensus winner: %s", winning_proposal.claim_id)

        # 4. Assign work to agent
        assigned_agent = winning_proposal.provenance.produced_by.get("id", "unknown")
        assignment_id = await self._assign_work(winning_proposal, assigned_agent)
        logger.info("Work assigned to %s: %s", assigned_agent, assignment_id)

        # 5. Monitor execution (if auto_approve)
        result_claim_id = None
        execution_started = False

        if self.config.auto_approve:
            logger.info("Auto-approve enabled, monitoring execution...")
            execution_started = True
            result_claim_id = await self._monitor_execution(assignment_id, assigned_agent)
        else:
            logger.info("Auto-approve disabled, agent will execute when operator approves")

        # 6. Return result
        return SolveResult(
            task=task,
            request_id=request_id,
            proposals_received=len(proposals),
            winning_proposal_id=winning_proposal.claim_id,
            assigned_agent=assigned_agent,
            execution_started=execution_started,
            result_claim_id=result_claim_id,
            success=True,
        )

    async def _post_task_request(self, task: str) -> str:
        """Post task decomposition request to knowledge.db."""
        request = Claim(
            claim_id=f"clm_{uuid.uuid4().hex[:24].upper()}",
            claim_type=ClaimType.QUESTION,
            claim_status=ClaimStatus.PROPOSED,
            scope=ClaimScope(
                scope_type=ScopeType.PROJECT,
                scope_id=self.config.project_id,
                visibility="project",
                valid_from=datetime.now(timezone.utc),
            ),
            canonical=ClaimCanonical(
                claim_key=f"action.{self.config.project_id}.work_order.{uuid.uuid4().hex[:8]}",
                subject=EntityRef(
                    entity_type="task",
                    entity_id=f"task_{uuid.uuid4().hex[:8]}",
                    label="Solve Task",
                ),
                predicate="needs_decomposition",
                object=ValueObject(value_type="string", value=task),
            ),
            statement=f"""TASK DECOMPOSITION REQUEST

FROM: {self.config.session_id} (Coordinator)
TASK: {task}

Agents: Please decompose this task and submit a proposal.

KEY CONVENTION (REQUIRED):
- Proposals: action.{{scope}}.proposal.{{short_id}}
- Votes: action.{{scope}}.vote.{{work_order_short_id}}.{{your_agent_id}}
- Progress: action.{{scope}}.progress.{{task_short_id}}.phase_{{n}}
- Results: action.{{scope}}.result.{{task_short_id}}
- Blockers: action.{{scope}}.blocker.{{task_short_id}}
Use multihead_deposit_action for proper deadline tracking.

PROPOSAL FORMAT:
- Post a response claim to knowledge.db
- Set related_claim_ids: ["{{}}" ]  # This request's claim_id
- Include your decomposition (steps, approach, estimated complexity)
- Include your agent_id in provenance.produced_by

TIMEOUT: {self.config.proposal_timeout_seconds}s from posting
MIN PROPOSALS: {self.config.min_proposals}
MAX PROPOSALS: {self.config.max_proposals}

If multiple proposals received, coordinator will run consensus voting.
Winning proposal will be assigned back to the proposing agent for execution.

Posted: {datetime.now(timezone.utc).isoformat()}
""",
            rationale="Distributed task decomposition request",
            confidence=0.9,
            provenance=Provenance(
                produced_by={"id": self.config.session_id, "method": "solve_coordinator"}
            ),
        )

        # Fill in the claim_id reference
        request.statement = request.statement.replace("{}", request.claim_id)

        self.knowledge_store.insert_claim(request)
        return request.claim_id

    async def _self_propose(self, task: str, request_id: str) -> list[Claim]:
        """Generate a proposal locally using the head manager (solo mode).

        Instead of polling for external proposals, decompose the task
        ourselves and wrap it as a proposal claim.
        """
        # Solo mode: return task + knowledge context to the calling session.
        # The caller (Claude Code, interactive shell) has codebase context
        # and will do a better decomposition than any local LLM could.
        rag_context = ""
        try:
            # Pull relevant claims from knowledge store for context
            claims = self.knowledge_store.search_claims_fts(task[:200], limit=10)
            if claims:
                rag_context = "\n\nRELEVANT KNOWLEDGE:\n" + "\n".join(
                    f"- {c.statement[:200]}" for c in claims
                )
        except Exception as e:
            logger.debug("RAG context lookup failed: %s", e)

        template = load_prompt("propose")
        if template:
            decomposition = template.format(task=task, rag_context=rag_context)
        else:
            decomposition = (
                f"TASK: {task}\n"
                f"{rag_context}\n\n"
                "YOU are the executor. Decompose this task into concrete steps "
                "using your loaded codebase context, then execute each step.\n\n"
                "SUGGESTED APPROACH:\n"
                "1. Analyze: Read relevant files and understand current state\n"
                "2. Plan: Identify specific changes needed\n"
                "3. Implement: Make the changes\n"
                "4. Verify: Run tests or validate the changes\n"
                "5. Report: Summarize what was done"
            )

        proposal = Claim(
            claim_id=f"clm_{uuid.uuid4().hex[:24].upper()}",
            claim_type=ClaimType.PLAN,
            claim_status=ClaimStatus.PROPOSED,
            scope=ClaimScope(
                scope_type=ScopeType.PROJECT,
                scope_id=self.config.project_id,
                visibility="project",
                valid_from=datetime.now(timezone.utc),
            ),
            canonical=ClaimCanonical(
                claim_key=f"action.{self.config.project_id}.proposal.{uuid.uuid4().hex[:8]}",
                subject=EntityRef(
                    entity_type="proposal",
                    entity_id=f"prop_{uuid.uuid4().hex[:8]}",
                    label="Self Proposal",
                ),
                predicate="proposes_decomposition",
                object=ValueObject(value_type="string", value=decomposition),
            ),
            statement=decomposition,
            confidence=0.9,
            provenance=Provenance(
                produced_by={"id": self.config.session_id, "method": "self_propose"}
            ),
            related_claim_ids=[request_id],
        )

        self.knowledge_store.insert_claim(proposal)
        logger.info("Self-proposal created: %s", proposal.claim_id)
        return [proposal]

    async def _collect_proposals(self, request_id: str) -> list[Claim]:
        """Poll knowledge.db for proposal responses."""
        logger.info("Collecting proposals for %s (timeout: %ss)", request_id, self.config.proposal_timeout_seconds)

        poll_interval = 5.0  # Check every 5 seconds
        elapsed = 0.0
        proposals = []
        min_met_at = None  # Track when we first hit minimum

        while elapsed < self.config.proposal_timeout_seconds:
            try:
                proposals = self.knowledge_store.get_responses_to_claim(
                    request_id, limit=self.config.max_proposals
                )
            except (ValueError, Exception) as e:
                # Handle Stability enum errors and other DB issues
                logger.warning("Error reading proposals (will retry): %s", e)
                # Use direct SQL as fallback
                import sqlite3
                conn = sqlite3.connect(self.knowledge_store.db_path, timeout=10.0)
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT claim_id FROM claims WHERE related_json LIKE ? ORDER BY created_at DESC LIMIT ?",
                    (f'%{request_id}%', self.config.max_proposals)
                ).fetchall()
                conn.close()

                # Get each claim individually with error handling
                proposals = []
                for row in rows:
                    try:
                        claim = self.knowledge_store.get_claim(row['claim_id'])
                        if claim:
                            proposals.append(claim)
                    except:
                        pass  # Skip malformed claims

                logger.info("Recovered %d proposals via fallback", len(proposals))

            # Track when we first hit minimum
            if len(proposals) >= self.config.min_proposals and min_met_at is None:
                min_met_at = elapsed
                logger.info("Minimum proposals reached: %d/%d at %ss",
                          len(proposals), self.config.min_proposals, elapsed)

            # Wait at least 60 seconds after hitting minimum to collect more proposals
            # This gives other agents time to respond
            if min_met_at is not None and (elapsed - min_met_at) >= 60.0:
                logger.info("Waited 60s after minimum, collecting %d total proposals", len(proposals))
                break

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        logger.info("Proposal collection complete: %d received", len(proposals))
        return proposals

    async def _vote_on_proposals(self, proposals: list[Claim]) -> Claim:
        """Run consensus voting on proposals via ConsensusEngine."""
        from ..consensus import ConsensusEngine, VoteResult

        logger.info(
            "Running consensus vote on %d proposals (strategy=%s)",
            len(proposals), self.config.consensus_strategy.value,
        )

        engine = ConsensusEngine(self.head_manager)

        # Convert proposal Claims to VoteResults
        votes: list[VoteResult] = []
        proposal_map: dict[str, Claim] = {}  # head_id -> original Claim

        for proposal in proposals:
            agent_id = proposal.provenance.produced_by.get("id", proposal.claim_id)
            votes.append(VoteResult(
                head_id=agent_id,
                outputs={"text": proposal.statement},
                success=True,
            ))
            proposal_map[agent_id] = proposal

        # Build weight map from proposal confidence scores
        weights = {
            p.provenance.produced_by.get("id", p.claim_id): p.confidence
            for p in proposals
        }

        result = engine.rank_proposals(
            votes=votes,
            strategy=self.config.consensus_strategy,
            weights=weights,
        )

        # Map winning VoteResult back to original Claim
        if result.consensus_outputs:
            winning_text = result.consensus_outputs.get("text", "")
            # Find the proposal whose statement matches
            for proposal in proposals:
                if proposal.statement == winning_text:
                    logger.info(
                        "Consensus winner: %s (agreement=%.0f%%, strategy=%s)",
                        proposal.claim_id,
                        result.agreement_score * 100,
                        result.strategy_used.value,
                    )
                    return proposal

        # Fallback: first proposal (shouldn't happen with valid votes)
        logger.warning("Consensus fallback to first proposal")
        return proposals[0]

    async def _assign_work(self, proposal: Claim, agent_id: str) -> str:
        """Assign winning proposal back to agent for execution."""
        assignment = Claim(
            claim_id=f"clm_{uuid.uuid4().hex[:24].upper()}",
            claim_type=ClaimType.DECISION,
            claim_status=ClaimStatus.ACCEPTED,
            scope=ClaimScope(
                scope_type=ScopeType.PROJECT,
                scope_id=self.config.project_id,
                visibility="project",
                valid_from=datetime.now(timezone.utc),
            ),
            canonical=ClaimCanonical(
                claim_key=f"action.{self.config.project_id}.assignment.{uuid.uuid4().hex[:8]}",
                subject=EntityRef(
                    entity_type="assignment",
                    entity_id=f"assign_{uuid.uuid4().hex[:8]}",
                    label="Work Assignment",
                ),
                predicate="assigned_to",
                object=ValueObject(value_type="string", value=agent_id),
            ),
            statement=f"""WORK ASSIGNMENT

FROM: {self.config.session_id} (Coordinator)
TO: {agent_id}

Your proposal has been selected via consensus voting.
Please execute the work order based on your decomposition.

PROPOSAL: {proposal.claim_id}
ASSIGNED: {datetime.now(timezone.utc).isoformat()}

When complete, post a result claim with:
- related_claim_ids: ["{proposal.claim_id}", "{{}}"]  # Proposal + this assignment
- Statement describing outcome, artifacts, success/failure

Auto-approve: {self.config.auto_approve}
""",
            rationale="Work assignment from consensus voting",
            confidence=0.95,
            provenance=Provenance(
                produced_by={"id": self.config.session_id, "method": "solve_coordinator"}
            ),
            related_claim_ids=[proposal.claim_id],
        )

        # Fill in assignment claim_id
        assignment.statement = assignment.statement.replace("{}", assignment.claim_id)

        self.knowledge_store.insert_claim(assignment)
        return assignment.claim_id

    async def _monitor_execution(self, assignment_id: str, agent_id: str) -> str | None:
        """Monitor for execution result (if auto_approve enabled)."""
        logger.info("Monitoring execution for assignment %s by agent %s", assignment_id, agent_id)

        # Poll for result claim (timeout: 2x proposal timeout)
        monitor_timeout = self.config.proposal_timeout_seconds * 2
        poll_interval = 10.0
        elapsed = 0.0

        while elapsed < monitor_timeout:
            results = self.knowledge_store.get_responses_to_claim(assignment_id, limit=1)

            if results:
                result = results[0]
                logger.info("Execution result received: %s", result.claim_id)
                return result.claim_id

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        logger.warning("Execution monitoring timed out after %ss", monitor_timeout)
        return None
