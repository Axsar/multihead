"""Core Orchestrator class: run lifecycle, execution strategies, and run management."""

from __future__ import annotations

import asyncio
import logging
import random
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..artifact_store import ArtifactStore
from ..dag_executor import DAGExecutor
from ..event_store import EventStore
from ..head_manager import HeadManager
from ..knowledge_hook import KnowledgeHook
from ..test_generation_hook import TestGenerationHook
from ..observability import MetricsCollector
from ..plan_normalizer import normalize as normalize_plan
from ..models import (
    EventKind,
    RunEvent,
    RunState,
    RunStatus,
    StageResult,
    StepStatus,
    WorkOrder,
)

from ._delegation import DelegationMixin
from ._execution import ExecutionMixin
from ._prompts import PromptMixin

logger = logging.getLogger(__name__)


class Orchestrator(DelegationMixin, ExecutionMixin, PromptMixin):
    """Executes WorkOrders through the head-swap pipeline.

    Key properties:
    - Durable: event-sourced state survives kill -9
    - Serial: one step at a time, one head at a time
    - Sync checkpointing: artifacts + events persisted before advancing
    """

    def __init__(
        self,
        event_store: EventStore,
        artifact_store: ArtifactStore,
        head_manager: HeadManager,
        runs_dir: Path,
        metrics: MetricsCollector | None = None,
        enable_marketplace_delegation: bool = False,  # Phase 6: RFQ delegation
        knowledge_hook: KnowledgeHook | None = None,  # Real-time learning
        test_hook: TestGenerationHook | None = None,  # Automatic test generation
        acp_bridge: Any | None = None,  # ACP bridge for remote delegation
    ) -> None:
        self.events = event_store
        self.artifacts = artifact_store
        self.heads = head_manager
        self.runs_dir = runs_dir
        self._active_runs: dict[str, RunState] = {}
        self._metrics = metrics or MetricsCollector()
        self.enable_marketplace_delegation = enable_marketplace_delegation
        self.knowledge_hook = knowledge_hook
        self.test_hook = test_hook
        self.acp_bridge = acp_bridge

    async def create_run(self, work_order: WorkOrder, *, normalize: bool = True) -> RunState:
        """Create a new run from a WorkOrder.

        Args:
            normalize: If True (default), validate and enrich the work order
                via plan_normalizer (check head_ids, infer deps, auto-fallback).
        """
        if normalize:
            # Pass knowledge_store to normalize for Router feedback loop
            ks = getattr(self.heads, '_knowledge_store', None)
            if ks is None and self.knowledge_hook:
                ks = getattr(self.knowledge_hook, 'knowledge_store', None)
            work_order = normalize_plan(
                work_order, self.heads,
                metrics=self._metrics,
                knowledge_store=ks,
            )

        run_id = work_order.run_id

        # Emit RUN_CREATED event
        event = RunEvent(
            run_id=run_id,
            kind=EventKind.RUN_CREATED,
            data={"work_order": work_order.model_dump(mode="json")},
        )
        self.events.append(event)

        # Emit STEP_PLANNED for each step
        for step in work_order.steps:
            plan_event = RunEvent(
                run_id=run_id,
                kind=EventKind.STEP_PLANNED,
                step_id=step.step_id,
                data={"step_name": step.name, "head_id": step.head_id},
            )
            self.events.append(plan_event)

        state = RunState(
            run_id=run_id,
            status=RunStatus.QUEUED,
            work_order=work_order,
        )
        self._active_runs[run_id] = state
        self._metrics.inc("runs_created_total")
        return state

    def _should_use_dag(self, work_order: WorkOrder) -> bool:
        """Auto-detect DAG mode: use if any step has depends_on."""
        return any(step.depends_on for step in work_order.steps)

    async def execute_run(self, run_id: str, execution_mode: str = "auto") -> RunState:
        """Execute a run from its current position (supports resume).

        Args:
            execution_mode: "linear" (serial), "dag" (parallel layers), or
                "auto" (dag if any step has depends_on, else linear).
        """
        state = self._active_runs.get(run_id)
        if state is None:
            # Try to replay from events
            state = self.events.replay(run_id)
            self._active_runs[run_id] = state

        if state.work_order is None:
            raise ValueError(f"No work order found for run {run_id}")

        work_order = state.work_order
        steps = work_order.steps

        if state.status in (RunStatus.DONE, RunStatus.CANCELLED):
            logger.info("Run %s already %s, skipping", run_id, state.status.value)
            return state

        # Decide execution mode
        use_dag = (
            execution_mode == "dag"
            or (execution_mode == "auto" and self._should_use_dag(work_order))
        )

        state.status = RunStatus.RUNNING
        self._metrics.inc("runs_started_total")
        t0 = time.perf_counter()

        try:
            timeout = work_order.run_timeout_seconds
            if use_dag:
                result = await asyncio.wait_for(
                    self._execute_run_dag(run_id, work_order, state),
                    timeout=timeout,
                )
            else:
                result = await asyncio.wait_for(
                    self._execute_run_linear(run_id, work_order, state),
                    timeout=timeout,
                )
        except TimeoutError:
            logger.error("Run %s timed out after %.1fs", run_id, work_order.run_timeout_seconds)
            self.events.append(RunEvent(
                run_id=run_id,
                kind=EventKind.RUN_FAILED,
                data={"error": f"Run timed out after {work_order.run_timeout_seconds}s"},
            ))
            state.status = RunStatus.FAILED
            state.ended_at = datetime.now(timezone.utc)
            self._metrics.inc("runs_timed_out_total")
            return state

        elapsed = time.perf_counter() - t0
        self._metrics.observe("run_duration_seconds", elapsed)
        if result.status == RunStatus.DONE:
            self._metrics.inc("runs_completed_total")
        elif result.status == RunStatus.FAILED:
            self._metrics.inc("runs_failed_total")
        return result

    async def _execute_run_dag(
        self, run_id: str, work_order: WorkOrder, state: RunState,
    ) -> RunState:
        """Execute using DAG executor for parallel layer execution."""
        logger.info(
            "Executing run %s in DAG mode: %d steps",
            run_id, len(work_order.steps),
        )

        try:
            dag = DAGExecutor(self)
            state = await dag.execute_dag(run_id, work_order, state)

            # Check for failures
            failed_steps = [
                sid for sid, r in state.step_results.items()
                if r.status == StepStatus.FAILED
            ]
            if failed_steps:
                self.events.append(RunEvent(
                    run_id=run_id,
                    kind=EventKind.RUN_FAILED,
                    data={"failed_steps": failed_steps},
                ))
                state.status = RunStatus.FAILED
            else:
                self.events.append(RunEvent(
                    run_id=run_id,
                    kind=EventKind.RUN_DONE,
                ))
                state.status = RunStatus.DONE

            state.ended_at = datetime.now(timezone.utc)
            logger.info("Run %s completed (DAG mode): %s", run_id, state.status.value)

        except Exception as e:
            logger.error("Run %s failed (DAG mode): %s", run_id, e)
            self.events.append(RunEvent(
                run_id=run_id,
                kind=EventKind.RUN_FAILED,
                data={"error": str(e), "traceback": traceback.format_exc()},
            ))
            state.status = RunStatus.FAILED
            state.ended_at = datetime.now(timezone.utc)

        return state

    async def _execute_run_linear(
        self, run_id: str, work_order: WorkOrder, state: RunState,
    ) -> RunState:
        """Execute steps sequentially (original behavior)."""
        steps = work_order.steps
        logger.info(
            "Executing run %s: %d steps, starting from step %d",
            run_id, len(steps), state.current_step_index,
        )

        # Create run artifacts directory
        run_artifacts_dir = self.runs_dir / run_id / "artifacts"
        run_artifacts_dir.mkdir(parents=True, exist_ok=True)

        try:
            for i in range(state.current_step_index, len(steps)):
                step = steps[i]
                logger.info(
                    "Step %d/%d: %s (head=%s)",
                    i + 1, len(steps), step.name, step.head_id,
                )

                result = await self._execute_step_with_reflection(
                    run_id=run_id,
                    step=step,
                    work_order=work_order,
                    state=state,
                    run_artifacts_dir=run_artifacts_dir,
                )

                if result.status == StepStatus.FAILED:
                    # Check retry policy with exponential backoff
                    max_attempts = step.retry_policy.get("max_attempts", 1)
                    backoff_ms = step.retry_policy.get("backoff_ms", 1000)
                    if max_attempts > 1:
                        for attempt in range(1, max_attempts):
                            # Exponential backoff with jitter
                            delay_ms = backoff_ms * (2 ** (attempt - 1))
                            jitter_ms = random.uniform(0, delay_ms * 0.1)
                            await asyncio.sleep((delay_ms + jitter_ms) / 1000)

                            logger.info("Retrying step %s (attempt %d/%d)", step.name, attempt + 1, max_attempts)
                            self._metrics.inc("steps_retried_total")
                            result = await self._execute_step(
                                run_id=run_id,
                                step=step,
                                work_order=work_order,
                                state=state,
                                run_artifacts_dir=run_artifacts_dir,
                            )
                            if result.status != StepStatus.FAILED:
                                break

                    if result.status == StepStatus.FAILED and step.fallback:
                        # Try fallback heads before giving up
                        for fb_head_id in step.fallback:
                            logger.info(
                                "Trying fallback head %s for step %s",
                                fb_head_id, step.name,
                            )
                            fb_step = step.model_copy(update={"head_id": fb_head_id})
                            result = await self._execute_step(
                                run_id=run_id,
                                step=fb_step,
                                work_order=work_order,
                                state=state,
                                run_artifacts_dir=run_artifacts_dir,
                            )
                            self._metrics.inc("steps_fallback_total")
                            if result.status != StepStatus.FAILED:
                                # Store under original step_id
                                state.step_results[step.step_id] = result
                                break

                    if result.status == StepStatus.FAILED:
                        # Step failed after all retries and fallbacks
                        self.events.append(RunEvent(
                            run_id=run_id,
                            kind=EventKind.RUN_FAILED,
                            data={"failed_step": step.step_id, "error": result.error},
                        ))
                        state.status = RunStatus.FAILED
                        state.ended_at = datetime.now(timezone.utc)
                        return state

                state.current_step_index = i + 1

            # All steps complete
            self.events.append(RunEvent(
                run_id=run_id,
                kind=EventKind.RUN_DONE,
            ))
            state.status = RunStatus.DONE
            state.ended_at = datetime.now(timezone.utc)
            logger.info("Run %s completed successfully", run_id)

        except Exception as e:
            logger.error("Run %s failed: %s", run_id, e)
            self.events.append(RunEvent(
                run_id=run_id,
                kind=EventKind.RUN_FAILED,
                data={"error": str(e), "traceback": traceback.format_exc()},
            ))
            state.status = RunStatus.FAILED
            state.ended_at = datetime.now(timezone.utc)

        return state

    async def cancel_run(self, run_id: str) -> RunState | None:
        """Cancel a running run."""
        state = self._active_runs.get(run_id)
        if state is None:
            return None
        self.events.append(RunEvent(
            run_id=run_id,
            kind=EventKind.RUN_CANCELLED,
        ))
        state.status = RunStatus.CANCELLED
        state.ended_at = datetime.now(timezone.utc)
        return state

    def get_run_state(self, run_id: str) -> RunState | None:
        """Get current run state (from cache or replay)."""
        if run_id in self._active_runs:
            return self._active_runs[run_id]
        state = self.events.replay(run_id)
        if state.work_order is not None:
            self._active_runs[run_id] = state
            return state
        return None

    async def resume_incomplete_runs(self) -> list[str]:
        """On startup: find and resume any incomplete runs."""
        resumed = []
        runs = self.events.list_runs()
        for run_info in runs:
            if run_info["status"] == "running":
                run_id = run_info["run_id"]
                logger.info("Resuming incomplete run: %s", run_id)
                state = self.events.replay(run_id)
                self._active_runs[run_id] = state
                resumed.append(run_id)
        return resumed
