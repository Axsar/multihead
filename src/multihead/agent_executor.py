"""Agent-side execution for MultiHead Solve.

Agents poll knowledge.db for task requests, decompose tasks,
submit proposals, and execute assigned work.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from .autonomous_executor import AutonomousExecutor, LocalLLMStrategy, ClaudeSessionStrategy
from .decomposer import TaskDecomposer
from .knowledge_models import (
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
from .knowledge_store import KnowledgeStore
from .head_manager import HeadManager

logger = logging.getLogger(__name__)


class AgentExecutor:
    """Agent-side executor for distributed task solving."""

    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        head_manager: HeadManager,
        agent_id: str,
        project_id: str = "multihead",
    ):
        self.knowledge_store = knowledge_store
        self.head_manager = head_manager
        self.agent_id = agent_id
        self.project_id = project_id
        self.decomposer = TaskDecomposer(
            head_manager=head_manager,
            knowledge_store=knowledge_store,
            session_id=agent_id,
            project_id=project_id,
        )

    def poll_for_requests(self, max_age_hours: int = 1) -> list[Claim]:
        """
        Poll knowledge.db for task decomposition requests.

        Args:
            max_age_hours: Only return requests from last N hours

        Returns:
            List of task request claims
        """
        messages = self.knowledge_store.get_pending_messages(
            session_id=self.agent_id,
            scope_id=self.project_id,
            max_age_hours=max_age_hours,
        )

        # Filter for task decomposition requests
        task_requests = [
            m for m in messages
            if m.claim_type == ClaimType.QUESTION
            and "TASK DECOMPOSITION REQUEST" in m.statement
        ]

        logger.info("Found %d task request(s) for agent %s", len(task_requests), self.agent_id)
        return task_requests

    async def respond_to_request(self, request: Claim) -> str:
        """
        Decompose task and submit proposal.

        Args:
            request: Task decomposition request claim

        Returns:
            Proposal claim ID
        """
        # Extract task from request
        task = self._extract_task(request)
        logger.info("Agent %s decomposing task: %s", self.agent_id, task[:60])

        # Decompose using TaskDecomposer
        work_order = await self.decomposer.decompose(task)

        # Submit proposal
        proposal_id = await self._submit_proposal(request.claim_id, task, work_order)
        logger.info("Proposal submitted: %s", proposal_id)

        return proposal_id

    def poll_for_assignments(self, max_age_hours: int = 1) -> list[Claim]:
        """
        Poll knowledge.db for work assignments.

        Returns:
            List of assignment claims for this agent
        """
        # Query for assignments directed at this agent
        messages = self.knowledge_store.get_pending_messages(
            session_id=self.agent_id,
            scope_id=self.project_id,
            max_age_hours=max_age_hours,
        )

        # Filter for work assignments
        assignments = [
            m for m in messages
            if m.claim_type == ClaimType.DECISION
            and "WORK ASSIGNMENT" in m.statement
            and f"TO: {self.agent_id}" in m.statement
        ]

        logger.info("Found %d assignment(s) for agent %s", len(assignments), self.agent_id)
        return assignments

    async def execute_assignment(self, assignment: Claim) -> str:
        """Execute assigned work via AutonomousExecutor and post results.

        Args:
            assignment: Work assignment claim

        Returns:
            Result claim ID
        """
        logger.info("Agent %s executing assignment: %s", self.agent_id, assignment.claim_id)

        # Extract proposal from assignment
        proposal_id = self._extract_proposal_id(assignment)

        if not proposal_id:
            logger.error("Could not extract proposal ID from assignment")
            return await self._post_error_result(
                assignment.claim_id,
                "Could not extract proposal ID from assignment"
            )

        # Get proposal claim
        proposal = self.knowledge_store.get_claim(proposal_id)

        if not proposal:
            logger.error("Could not find proposal: %s", proposal_id)
            return await self._post_error_result(
                assignment.claim_id,
                f"Could not find proposal: {proposal_id}"
            )

        # Extract work order from proposal
        plan_dict = self._extract_work_order(proposal)
        if not plan_dict:
            logger.error("Could not extract work order from proposal %s", proposal_id)
            return await self._post_error_result(
                assignment.claim_id,
                f"Could not extract work order from proposal {proposal_id}"
            )

        task = self._extract_task_from_proposal(proposal)

        # Choose execution strategy: ClaudeSession if available, else LocalLLM
        import shutil
        if shutil.which("claude"):
            strategy = ClaudeSessionStrategy()
        else:
            strategy = LocalLLMStrategy()

        executor = AutonomousExecutor(
            strategy=strategy,
            knowledge_store=self.knowledge_store,
            agent_id=self.agent_id,
            project_id=self.project_id,
        )

        # Execute the plan
        logger.info("Executing work order with %s", type(strategy).__name__)
        report = await executor.execute(
            goal=task,
            plan=plan_dict,
            request_id=assignment.claim_id,
            proposal_id=proposal_id,
        )

        # Post result
        summary = report.summary()
        if report.success:
            result_id = await self._post_success_result(
                assignment.claim_id,
                proposal_id,
                summary,
            )
        else:
            result_id = await self._post_error_result(
                assignment.claim_id,
                f"Execution partial: {summary}",
            )

        logger.info("Result posted: %s (success=%s)", result_id, report.success)
        return result_id

    def _extract_task(self, request: Claim) -> str:
        """Extract task description from request claim."""
        # Look for "TASK: " in statement
        for line in request.statement.split("\n"):
            if line.startswith("TASK:"):
                return line.replace("TASK:", "").strip()

        # Fallback: use canonical object value
        return request.canonical.object.value

    def _extract_work_order(self, proposal: Claim) -> dict | None:
        """Extract serialized work order dict from a proposal claim."""
        obj = proposal.canonical.object.value
        if isinstance(obj, dict) and "work_order" in obj:
            wo = obj["work_order"]
            # Convert flat WorkOrder to plan dict with phases
            if isinstance(wo, dict):
                if "phases" in wo:
                    return wo
                # WorkOrder format: {goal, steps, ...} → wrap as plan
                steps = wo.get("steps", [])
                return {
                    "goal": wo.get("goal", ""),
                    "phases": [{
                        "id": "phase_1",
                        "goal": wo.get("goal", ""),
                        "children": [
                            {
                                "id": s.get("step_id", f"step_{i}"),
                                "goal": s.get("goal", s.get("description", "")),
                                "action_type": s.get("action_type", s.get("required_kind", "")),
                                "target_files": s.get("target_files", []),
                            }
                            for i, s in enumerate(steps)
                        ],
                    }],
                    "complexity": len(steps),
                }
        return None

    def _extract_task_from_proposal(self, proposal: Claim) -> str:
        """Extract task string from a proposal claim."""
        obj = proposal.canonical.object.value
        if isinstance(obj, dict) and "task" in obj:
            return obj["task"]
        # Fallback: parse from statement
        for line in proposal.statement.split("\n"):
            if line.startswith("TASK:"):
                return line.replace("TASK:", "").strip()
        return "Unknown task"

    def _extract_proposal_id(self, assignment: Claim) -> str | None:
        """Extract proposal claim ID from assignment."""
        # Check related_claim_ids first
        if assignment.related_claim_ids:
            return assignment.related_claim_ids[0]

        # Look for "PROPOSAL: " in statement
        for line in assignment.statement.split("\n"):
            if line.startswith("PROPOSAL:"):
                return line.replace("PROPOSAL:", "").strip()

        return None

    async def _submit_proposal(self, request_id: str, task: str, work_order: Any) -> str:
        """Submit decomposition proposal to knowledge.db."""
        # Serialize work order as JSON for downstream execution
        work_order_json = work_order.model_dump_json() if hasattr(work_order, 'model_dump_json') else "{}"

        proposal = Claim(
            claim_id=f"clm_{uuid.uuid4().hex[:24].upper()}",
            claim_type=ClaimType.FACT,
            claim_status=ClaimStatus.ACCEPTED,
            scope=ClaimScope(
                scope_type=ScopeType.PROJECT,
                scope_id=self.project_id,
                visibility="project",
                valid_from=datetime.now(timezone.utc),
            ),
            canonical=ClaimCanonical(
                claim_key=f"action.{self.project_id}.proposal.{uuid.uuid4().hex[:8]}",
                subject=EntityRef(
                    entity_type="proposal",
                    entity_id=f"prop_{uuid.uuid4().hex[:8]}",
                    label="Task Proposal",
                ),
                predicate="proposes_solution",
                object=ValueObject(
                    value_type="json",
                    value={"task": task, "work_order": json.loads(work_order_json)},
                ),
            ),
            statement=f"""TASK PROPOSAL

FROM: {self.agent_id}
RE: {request_id}

TASK: {task}

DECOMPOSITION:
{self._format_work_order(work_order)}

ESTIMATED COMPLEXITY: {len(work_order.steps)} steps
AGENT: {self.agent_id}

Submitted: {datetime.now(timezone.utc).isoformat()}
""",
            rationale="Task decomposition proposal",
            confidence=0.85,
            provenance=Provenance(
                produced_by={"id": self.agent_id, "method": "agent_executor"}
            ),
            related_claim_ids=[request_id],
        )

        self.knowledge_store.insert_claim(proposal)
        return proposal.claim_id

    async def _post_success_result(
        self,
        assignment_id: str,
        proposal_id: str,
        summary: str
    ) -> str:
        """Post successful execution result."""
        result = Claim(
            claim_id=f"clm_{uuid.uuid4().hex[:24].upper()}",
            claim_type=ClaimType.FACT,
            claim_status=ClaimStatus.ACCEPTED,
            scope=ClaimScope(
                scope_type=ScopeType.PROJECT,
                scope_id=self.project_id,
                visibility="project",
                valid_from=datetime.now(timezone.utc),
            ),
            canonical=ClaimCanonical(
                claim_key=f"solve.result.{uuid.uuid4().hex[:8]}",
                subject=EntityRef(
                    entity_type="result",
                    entity_id=f"result_{uuid.uuid4().hex[:8]}",
                    label="Execution Result",
                ),
                predicate="execution_complete",
                object=ValueObject(value_type="string", value="success"),
            ),
            statement=f"""EXECUTION RESULT

FROM: {self.agent_id}
RE: Assignment {assignment_id}
    Proposal {proposal_id}

STATUS: SUCCESS
SUMMARY: {summary}

Completed: {datetime.now(timezone.utc).isoformat()}
""",
            rationale="Execution result",
            confidence=0.9,
            provenance=Provenance(
                produced_by={"id": self.agent_id, "method": "agent_executor"}
            ),
            related_claim_ids=[assignment_id, proposal_id],
        )

        self.knowledge_store.insert_claim(result)
        return result.claim_id

    async def _post_error_result(self, assignment_id: str, error: str) -> str:
        """Post error result."""
        result = Claim(
            claim_id=f"clm_{uuid.uuid4().hex[:24].upper()}",
            claim_type=ClaimType.FACT,
            claim_status=ClaimStatus.ACCEPTED,
            scope=ClaimScope(
                scope_type=ScopeType.PROJECT,
                scope_id=self.project_id,
                visibility="project",
                valid_from=datetime.now(timezone.utc),
            ),
            canonical=ClaimCanonical(
                claim_key=f"solve.result.{uuid.uuid4().hex[:8]}",
                subject=EntityRef(
                    entity_type="result",
                    entity_id=f"result_{uuid.uuid4().hex[:8]}",
                    label="Execution Result",
                ),
                predicate="execution_failed",
                object=ValueObject(value_type="string", value="error"),
            ),
            statement=f"""EXECUTION RESULT

FROM: {self.agent_id}
RE: Assignment {assignment_id}

STATUS: ERROR
ERROR: {error}

Failed: {datetime.now(timezone.utc).isoformat()}
""",
            rationale="Execution error",
            confidence=0.9,
            provenance=Provenance(
                produced_by={"id": self.agent_id, "method": "agent_executor"}
            ),
            related_claim_ids=[assignment_id],
        )

        self.knowledge_store.insert_claim(result)
        return result.claim_id

    def _format_work_order(self, work_order: Any) -> str:
        """Format work order for display."""
        lines = []
        for i, step in enumerate(work_order.steps, 1):
            lines.append(f"  {i}. {step.goal}")
            if step.description:
                lines.append(f"      {step.description}")
        return "\n".join(lines) if lines else "  (No steps)"
