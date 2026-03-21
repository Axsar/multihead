"""Solve-pipeline mixin for the MultiHead Engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..head_manager import HeadManager
    from ..knowledge_store import KnowledgeStore
    from ..orchestrator import Orchestrator
    from ..config import Settings


class _SolveMixin:
    """Autonomous solve pipeline capability."""

    # These attributes are provided by the Engine base class.
    _head_manager: HeadManager | None
    _knowledge_store: KnowledgeStore | None
    _orchestrator: Orchestrator | None
    settings: Settings
    _started: bool

    def _ensure_started(self) -> None: ...  # pragma: no cover

    async def solve(
        self,
        task: str,
        *,
        max_steps: int = 20,
        strategy: str = "first_to_ahead",
        timeout: float = 240.0,
        enable_marketplace: bool = False,
        enable_tests: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Run the autonomous solve pipeline.

        Decomposes a task, infers DAG dependencies, routes to heads,
        executes, and aggregates results.

        Args:
            task: High-level task description.
            max_steps: Maximum steps in the plan.
            strategy: Consensus strategy (majority, weighted, first_to_ahead).
            timeout: Maximum seconds for the entire pipeline.
            enable_marketplace: Allow marketplace delegation for unroutable steps.
            enable_tests: Auto-generate tests for code steps.
            dry_run: Stop after decomposition, return plan without executing.

        Returns:
            Dict with run_id, status, output, confidence,
            steps_total, steps_succeeded, steps_failed, duration_seconds.
        """
        self._ensure_started()

        from ..solve_pipeline import SolveConstraints, SolvePipeline

        constraints = SolveConstraints(
            max_steps=max_steps,
            strategy=strategy,
            timeout_seconds=timeout,
            enable_test_generation=enable_tests,
            enable_marketplace_delegation=enable_marketplace,
        )

        pipeline = SolvePipeline(
            head_manager=self._head_manager,
            event_store=self._orchestrator.events,
            artifact_store=self._orchestrator.artifacts,
            knowledge_store=self._knowledge_store,
            runs_dir=self.settings.runs_dir,
        )

        result = await pipeline.solve(task, constraints=constraints, dry_run=dry_run)

        return {
            "run_id": result.run_id,
            "status": result.status,
            "output": result.output,
            "confidence": result.confidence,
            "steps_total": result.steps_total,
            "steps_succeeded": result.steps_succeeded,
            "steps_failed": result.steps_failed,
            "duration_seconds": result.duration_seconds,
            "dry_run": result.dry_run,
        }
