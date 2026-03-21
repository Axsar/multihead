"""Reusable solve pipeline: decompose -> DAG -> execute -> aggregate.

Extracted from ``multihead solve`` CLI to enable reuse in:
- Cloud marketplace contract execution
- Shell /solve command
- Python SDK Engine.solve()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SolveConstraints:
    """SLA constraints for a solve pipeline run."""

    max_steps: int = 20
    max_depth: int = 3
    timeout_seconds: float = 240.0
    consensus_timeout: float = 60.0
    strategy: str = "first_to_ahead"  # ConsensusStrategy value string
    enable_research_features: bool = True
    enable_knowledge_hook: bool = True
    enable_test_generation: bool = False
    test_framework: str = "pytest"  # pytest | unittest
    enable_marketplace_delegation: bool = False


@dataclass
class SolveResult:
    """Aggregated result of a solve pipeline run."""

    run_id: str
    status: str  # "done", "failed", "timed_out"
    output: str
    confidence: float
    steps_total: int = 0
    steps_succeeded: int = 0
    steps_failed: int = 0
    duration_seconds: float = 0.0
    plan_steps: int = 0
    parallel_steps: int = 0
    consensus_meta: dict[str, Any] = field(default_factory=dict)
    test_generation_stats: dict[str, Any] = field(default_factory=dict)
    dry_run: bool = False
    state: Any = None  # RunState if caller needs detail


class SolvePipeline:
    """Reusable multi-step solve pipeline.

    Encapsulates: AutoDecomposer -> DAG WorkOrder -> Orchestrator execution.

    Usage::

        pipeline = SolvePipeline(head_manager=hm, event_store=es,
                                 artifact_store=art, knowledge_store=ks,
                                 runs_dir=runs_dir)
        result = await pipeline.solve("Build a web scraper",
                                      constraints=SolveConstraints(max_steps=10))
    """

    def __init__(
        self,
        head_manager: Any,
        event_store: Any,
        artifact_store: Any,
        knowledge_store: Any = None,
        runs_dir: Any = None,
        metrics: Any = None,
        acp_bridge: Any = None,
    ) -> None:
        self._heads = head_manager
        self._event_store = event_store
        self._artifact_store = artifact_store
        self._knowledge_store = knowledge_store
        self._runs_dir = runs_dir
        self._metrics = metrics
        self._acp_bridge = acp_bridge

    async def solve(
        self,
        task: str,
        *,
        constraints: SolveConstraints | None = None,
        heads: list[str] | None = None,
        context: dict[str, Any] | None = None,
        dry_run: bool = False,
    ) -> SolveResult:
        """Execute the full solve pipeline.

        1. Decompose task with multi-head consensus
        2. Convert to WorkOrder with DAG inference
        3. Execute via Orchestrator (serial or DAG, auto-detected)
        4. Aggregate step results

        If ``dry_run`` is True, stops after phase 2 and returns the plan
        without executing any steps.

        Always returns a ``SolveResult`` — never raises.
        """
        if constraints is None:
            constraints = SolveConstraints()

        t0 = time.time()

        try:
            return await self._run(task, constraints, heads, context, t0, dry_run=dry_run)
        except Exception as e:
            duration = time.time() - t0
            logger.error("SolvePipeline failed: %s", e, exc_info=True)
            return SolveResult(
                run_id="",
                status="failed",
                output=f"Pipeline error: {e}",
                confidence=0.0,
                duration_seconds=duration,
            )

    async def _run(
        self,
        task: str,
        constraints: SolveConstraints,
        heads: list[str] | None,
        context: dict[str, Any] | None,
        t0: float,
        *,
        dry_run: bool = False,
    ) -> SolveResult:
        """Inner pipeline — may raise."""
        import uuid

        from .auto_decomposition import AutoDecomposer
        from .consensus import ConsensusStrategy
        from .knowledge_hook import KnowledgeHook
        from .observability import MetricsCollector
        from .orchestrator import Orchestrator

        strategy_map = {
            "majority": ConsensusStrategy.MAJORITY,
            "weighted": ConsensusStrategy.WEIGHTED,
            "unanimous": ConsensusStrategy.UNANIMOUS,
            "first_to_ahead": ConsensusStrategy.FIRST_TO_AHEAD,
        }
        strategy = strategy_map.get(constraints.strategy, ConsensusStrategy.FIRST_TO_AHEAD)

        # --- Build components ---
        decomposer = AutoDecomposer(
            head_manager=self._heads,
            knowledge_store=self._knowledge_store,
        )

        knowledge_hook = None
        if constraints.enable_knowledge_hook and self._knowledge_store:
            knowledge_hook = KnowledgeHook(
                knowledge_store=self._knowledge_store,
                project_id="multihead",
                min_confidence=0.7,
                capture_successes=True,
                capture_failures=True,
                capture_performance=True,
            )

        # --- Test generation hook (optional) ---
        test_hook = None
        if constraints.enable_test_generation:
            try:
                from .test_generation import TestFramework, TestGenerator
                from .test_generation_hook import TestGenerationHook

                framework_map = {
                    "pytest": TestFramework.PYTEST,
                    "unittest": TestFramework.UNITTEST,
                }
                framework = framework_map.get(
                    constraints.test_framework, TestFramework.PYTEST
                )
                test_gen = TestGenerator(
                    head_manager=self._heads,
                    framework=framework,
                )
                test_hook = TestGenerationHook(test_generator=test_gen)
                logger.info("TestGenerationHook enabled (framework=%s)", constraints.test_framework)
            except Exception as e:
                logger.warning("Could not create TestGenerationHook: %s", e)

        metrics = self._metrics or MetricsCollector()

        orchestrator = Orchestrator(
            event_store=self._event_store,
            artifact_store=self._artifact_store,
            head_manager=self._heads,
            runs_dir=self._runs_dir,
            metrics=metrics,
            knowledge_hook=knowledge_hook,
            test_hook=test_hook,
            acp_bridge=self._acp_bridge,
            enable_marketplace_delegation=constraints.enable_marketplace_delegation,
        )

        # --- Phase 1: Consensus decomposition ---
        logger.info("SolvePipeline phase 1: decomposing '%s'", task[:80])

        plan, consensus_meta = await decomposer.decompose_with_consensus(
            goal=task,
            heads=heads,
            strategy=strategy,
            auto_validate=True,
            enable_research_features=constraints.enable_research_features,
            context=context,
        )

        logger.info(
            "SolvePipeline decomposed: %d steps, complexity=%s, agreement=%.2f",
            plan.total_steps,
            plan.complexity,
            consensus_meta.get("agreement_score", 0.0),
        )

        # --- Phase 2: DAG inference ---
        logger.info("SolvePipeline phase 2: DAG inference")
        work_order = decomposer.to_work_order_with_dag(plan)

        # Propagate remaining time as run timeout
        elapsed = time.time() - t0
        remaining = max(30.0, constraints.timeout_seconds - elapsed)
        work_order.run_timeout_seconds = remaining

        parallel_count = sum(1 for s in work_order.steps if not s.depends_on)

        # --- Dry-run cutoff: return plan without executing ---
        if dry_run:
            plan_lines = []
            for i, step in enumerate(work_order.steps, 1):
                deps = f" (depends on: {', '.join(step.depends_on)})" if step.depends_on else ""
                head = f" [{step.head_id}]" if step.head_id else ""
                plan_lines.append(f"  {i}. {step.name}{head}{deps}")
            plan_output = (
                f"Decomposition Plan (dry-run)\n"
                f"  Steps: {len(work_order.steps)} ({parallel_count} can run in parallel)\n"
                f"  Complexity: {plan.complexity}\n"
                f"  Agreement: {consensus_meta.get('agreement_score', 0.0):.2f}\n\n"
                + "\n".join(plan_lines)
            )
            return SolveResult(
                run_id=f"dry-run-{uuid.uuid4().hex[:8]}",
                status="dry_run",
                output=plan_output,
                confidence=consensus_meta.get("agreement_score", 0.0),
                steps_total=len(work_order.steps),
                plan_steps=plan.total_steps,
                parallel_steps=parallel_count,
                consensus_meta=consensus_meta,
                dry_run=True,
                duration_seconds=time.time() - t0,
            )

        # --- Phase 3: Execute ---
        logger.info(
            "SolvePipeline phase 3: executing %d steps (%d parallel)",
            len(work_order.steps),
            parallel_count,
        )

        state = await orchestrator.create_run(work_order, normalize=True)
        state = await orchestrator.execute_run(state.run_id)

        # --- Phase 4: Aggregate ---
        succeeded = sum(
            1 for r in state.step_results.values()
            if r.status.value == "committed"
        )
        failed = sum(
            1 for r in state.step_results.values()
            if r.status.value == "failed"
        )

        output_parts: list[str] = []
        for step_def in work_order.steps:
            result = state.step_results.get(step_def.step_id) or state.step_results.get(step_def.name)
            if result and result.outputs:
                text = result.outputs.get("text", "")
                if text:
                    output_parts.append(f"## {step_def.name}\n{text}")

        output = "\n\n".join(output_parts) if output_parts else f"Pipeline completed: {state.status.value}"
        confidence = succeeded / max(len(state.step_results), 1)
        duration = time.time() - t0

        logger.info(
            "SolvePipeline done: %s, %d/%d steps, %.1fs",
            state.status.value, succeeded, len(state.step_results), duration,
        )

        # --- Collect test generation stats ---
        test_stats: dict[str, Any] = {}
        if test_hook:
            test_stats = test_hook.get_session_summary()

        return SolveResult(
            run_id=state.run_id,
            status=state.status.value,
            output=output,
            confidence=confidence,
            steps_total=len(state.step_results),
            steps_succeeded=succeeded,
            steps_failed=failed,
            duration_seconds=duration,
            plan_steps=plan.total_steps,
            parallel_steps=parallel_count,
            consensus_meta=consensus_meta,
            test_generation_stats=test_stats,
            state=state,
        )
