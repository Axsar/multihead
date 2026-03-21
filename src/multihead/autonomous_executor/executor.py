"""AutonomousExecutor — decomposes a goal into a DAG and executes via the chosen strategy."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone

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
from .dag import _flatten_leaves, _infer_layer_dependencies, _summarize_plan, _topological_layers
from .models import ExecutionReport, StepContext, StepExecutionResult
from .strategies import ExecutionStrategy

logger = logging.getLogger(__name__)


class AutonomousExecutor:
    """Decomposes a goal into a DAG and executes via the chosen strategy.

    Flow:
        Request -> Decompose (AutoDecomposer + DAG + research features)
          -> Layer 0: [explore, read]     (parallel claude -p sessions)
          -> Layer 1: [implement]         (with context from layer 0)
          -> Layer 2: [review, test]      (parallel reviewer + tester)
          -> Layer 3: [verify]            (with review + test results)
          -> Post results to knowledge.db
    """

    def __init__(
        self,
        strategy: ExecutionStrategy,
        knowledge_store: KnowledgeStore | None = None,
        agent_id: str = "autonomous-executor",
        project_id: str = "multihead",
        quality_threshold: float = 0.5,
        max_retries: int = 3,
        max_parallel: int = 4,
    ):
        self.strategy = strategy
        self.knowledge_store = knowledge_store
        self.agent_id = agent_id
        self.project_id = project_id
        self.quality_threshold = quality_threshold
        self.max_retries = max_retries
        self.max_parallel = max_parallel

    async def execute(
        self,
        goal: str,
        plan: dict,
        request_id: str = "",
        proposal_id: str = "",
    ) -> ExecutionReport:
        """Execute a decomposed plan.

        Args:
            goal: The overall goal.
            plan: Serialized plan dict with phases/children (from poller).
            request_id: Original request claim ID (for linking results).
            proposal_id: Proposal claim ID (for linking results).

        Returns:
            ExecutionReport with per-step results.
        """
        # 1. Flatten leaves and build dependency graph
        leaves = _flatten_leaves(plan)
        if not leaves:
            return ExecutionReport(
                goal=goal, strategy=type(self.strategy).__name__,
                total_steps=0, completed_steps=0, failed_steps=0,
                skipped_steps=0, total_cost_usd=0, total_duration_secs=0,
            )

        dep_map = _infer_layer_dependencies(leaves)
        layers = _topological_layers(leaves, dep_map)

        # 2. Build context
        plan_summary = _summarize_plan(plan)
        ctx = StepContext(
            goal=goal,
            plan_summary=plan_summary,
            work_dir=getattr(self.strategy, "work_dir", ""),
        )

        # 3. Execute layer by layer
        report = ExecutionReport(
            goal=goal,
            strategy=type(self.strategy).__name__,
            total_steps=len(leaves),
            completed_steps=0,
            failed_steps=0,
            skipped_steps=0,
            total_cost_usd=0.0,
            total_duration_secs=0.0,
            layers=[layer_ids for layer_ids in layers],
        )

        start_time = time.monotonic()

        for layer_idx, layer_step_ids in enumerate(layers):
            layer_steps = [s for s in leaves if s["id"] in layer_step_ids]
            logger.info(
                "Layer %d: executing %d step(s) — %s",
                layer_idx, len(layer_steps),
                [s["id"] for s in layer_steps],
            )

            # Execute steps within layer in parallel (bounded)
            sem = asyncio.Semaphore(self.max_parallel)

            async def _run_step(step: dict) -> StepExecutionResult:
                async with sem:
                    return await self._execute_with_reflection(step, ctx, dep_map)

            results = await asyncio.gather(
                *[_run_step(s) for s in layer_steps],
                return_exceptions=True,
            )

            # Process results
            for step, result in zip(layer_steps, results):
                if isinstance(result, Exception):
                    result = StepExecutionResult(
                        step_id=step["id"],
                        step_goal=step.get("goal", ""),
                        action_type=step.get("action_type", ""),
                        success=False,
                        output="",
                        error=str(result),
                    )

                report.step_results.append(result)
                report.total_cost_usd += result.cost_usd

                if result.success:
                    report.completed_steps += 1
                    # Chain output to context for downstream steps
                    ctx.step_outputs[step["id"]] = result.output
                else:
                    report.failed_steps += 1
                    logger.warning(
                        "Step %s failed: %s", step["id"], result.error or "unknown"
                    )

        report.total_duration_secs = time.monotonic() - start_time

        # 4. Post results to knowledge.db
        if self.knowledge_store and request_id:
            self._post_execution_result(report, request_id, proposal_id)

        return report

    async def _execute_with_reflection(
        self, step: dict, ctx: StepContext, dep_map: dict[str, list[str]],
    ) -> StepExecutionResult:
        """Execute a step with quality-gated retries (reflection loop)."""
        step_id = step["id"]
        action_type = step.get("action_type", "")
        step_goal = step.get("goal", "")
        target_files = step.get("target_files", [])
        dependencies = dep_map.get(step_id, [])

        reflection_feedback = ""

        for attempt in range(1, self.max_retries + 1):
            # Build prompt with context + any reflection feedback
            prompt = ctx.build_prompt(
                step_id, step_goal, action_type, target_files, dependencies,
            )
            if reflection_feedback:
                prompt += (
                    f"\n\n## Feedback from Previous Attempt\n"
                    f"Your previous attempt scored below threshold. Issues:\n"
                    f"{reflection_feedback}\n"
                    f"Please address these issues in this attempt.\n"
                )

            result = await self.strategy.execute_step(
                step_id=step_id,
                prompt=prompt,
                action_type=action_type,
            )
            result.attempt_number = attempt

            # Quality check
            score, feedback = self.strategy.check_quality(result)
            result.quality_score = score
            result.quality_feedback = feedback

            if result.success and score >= self.quality_threshold:
                return result

            if attempt < self.max_retries:
                logger.info(
                    "Step %s attempt %d: score=%.2f (threshold=%.2f), retrying...",
                    step_id, attempt, score, self.quality_threshold,
                )
                reflection_feedback = feedback
                if result.error:
                    reflection_feedback += f"\nError: {result.error}"
            else:
                logger.warning(
                    "Step %s: exhausted %d retries (best score=%.2f)",
                    step_id, self.max_retries, score,
                )

        return result

    def _post_execution_result(
        self, report: ExecutionReport, request_id: str, proposal_id: str,
    ) -> str:
        """Post execution result as a claim to knowledge.db."""
        predicate = "execution_complete" if report.success else "execution_failed"
        status_text = "SUCCESS" if report.success else "PARTIAL"
        value = "success" if report.success else "partial"

        # Build per-step summary
        step_lines = []
        for r in report.step_results:
            icon = "+" if r.success else "-"
            step_lines.append(
                f"  {icon} {r.step_id} ({r.action_type}): "
                f"score={r.quality_score:.2f}, cost=${r.cost_usd:.2f}, "
                f"attempts={r.attempt_number}"
            )
        steps_text = "\n".join(step_lines)

        statement = (
            f"EXECUTION RESULT\n\n"
            f"FROM: {self.agent_id}\n"
            f"RE: Request {request_id}\n"
            f"    Proposal {proposal_id}\n\n"
            f"STATUS: {status_text}\n"
            f"STEPS: {report.completed_steps}/{report.total_steps} completed\n"
            f"COST: ${report.total_cost_usd:.2f}\n"
            f"DURATION: {report.total_duration_secs:.0f}s\n\n"
            f"STEP DETAILS:\n{steps_text}\n\n"
            f"SUMMARY: {report.summary()}\n\n"
            f"Completed: {datetime.now(timezone.utc).isoformat()}\n"
        )

        related = [request_id]
        if proposal_id:
            related.append(proposal_id)

        claim = Claim(
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
                claim_key=f"solve.execution.{uuid.uuid4().hex[:8]}",
                subject=EntityRef(
                    entity_type="result",
                    entity_id=f"exec_{uuid.uuid4().hex[:8]}",
                    label="Autonomous Execution Result",
                ),
                predicate=predicate,
                object=ValueObject(value_type="string", value=value),
            ),
            statement=statement,
            rationale="Autonomous executor result",
            confidence=0.9 if report.success else 0.5,
            provenance=Provenance(
                produced_by={"id": self.agent_id, "method": "autonomous_executor"}
            ),
            related_claim_ids=related,
        )

        self.knowledge_store.insert_claim(claim)
        logger.info("Posted execution result: %s", claim.claim_id[:12])
        return claim.claim_id
