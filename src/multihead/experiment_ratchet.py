"""Experiment Ratchet — AutoResearch-style automated experiment loop.

Inspired by Karpathy's AutoResearch: agent proposes changes, runs them,
keeps improvements (git commit), reverts failures (git reset --hard),
and logs everything to knowledge.db.

Key difference from Karpathy's version:
- Uses knowledge.db instead of results.tsv (structured failure memory)
- Failed experiments are categorized with WHY they failed
- Failures are shared across agents via knowledge store
- Reflection loops learn from failures
- PRM scores each iteration at the step level

Usage:
    ratchet = ExperimentRatchet(
        knowledge_store=ks,
        head_manager=hm,
        experiment_id="balloon-overlap-optimization",
        target_file="src/stage7a/balloonlayout/synthetic_layout.py",
        test_command="python -m pytest tests/test_layout.py -x",
        metric_name="overlap_count",
        metric_goal="minimize",  # or "maximize"
        max_iterations=50,
        time_budget_secs=300,  # 5 min per iteration
    )
    report = await ratchet.run()
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .iteration_tracker import (
    IterationResult,
    IterationTracker,
    _git_commit,
    _git_head_sha,
    _git_reset_hard,
)
from .subprocess_utils import no_window_flags

logger = logging.getLogger(__name__)


@dataclass
class RatchetConfig:
    """Configuration for an experiment ratchet loop."""

    experiment_id: str
    target_files: list[str] = field(default_factory=list)  # files the agent can edit
    test_command: str = ""  # command to run after each change
    metric_name: str = "quality_score"  # metric to optimize
    metric_goal: str = "maximize"  # "maximize" or "minimize"
    max_iterations: int = 50
    time_budget_secs: int = 300  # per iteration
    total_budget_secs: int = 0  # 0 = unlimited
    quality_threshold: float = 0.0  # stop if metric reaches this
    work_dir: str = "."
    head_id: str = ""  # which head to use for proposals
    system_prompt: str = ""  # custom prompt for the agent
    simplicity_bias: bool = True  # prefer simpler changes (Karpathy principle)


@dataclass
class RatchetReport:
    """Summary of a ratchet run."""

    experiment_id: str
    iterations_run: int = 0
    iterations_kept: int = 0
    iterations_reverted: int = 0
    iterations_errored: int = 0
    best_metrics: dict[str, Any] = field(default_factory=dict)
    best_iteration: int = 0
    best_git_sha: str = ""
    total_duration_secs: float = 0.0
    history: list[IterationResult] = field(default_factory=list)
    stopped_reason: str = ""  # "max_iterations", "threshold_reached", "budget_exhausted", "manual_stop"


class ExperimentRatchet:
    """AutoResearch-style experiment loop with knowledge.db backing.

    The ratchet:
    1. Agent proposes a change to target_files
    2. Change is applied
    3. test_command runs (fixed time budget)
    4. If metric improves → git commit (keep)
    5. If metric doesn't improve → git reset --hard (revert)
    6. Result logged to knowledge.db with structured metadata
    7. Previous failures inform next proposal (via knowledge RAG)
    8. Loop continues until max_iterations or threshold reached
    """

    def __init__(
        self,
        knowledge_store: Any,
        head_manager: Any | None = None,
        config: RatchetConfig | None = None,
        propose_fn: Callable[..., Any] | None = None,
        evaluate_fn: Callable[..., dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        self.ks = knowledge_store
        self.hm = head_manager
        self.config = config or RatchetConfig(**kwargs)
        self._propose_fn = propose_fn  # custom proposal function
        self._evaluate_fn = evaluate_fn  # custom evaluation function
        self._stop_requested = False

        self.tracker = IterationTracker(
            knowledge_store=knowledge_store,
            experiment_id=self.config.experiment_id,
            agent_id="experiment-ratchet",
            scope_id="experiments",
            work_dir=self.config.work_dir,
        )

    def stop(self) -> None:
        """Request graceful stop after current iteration."""
        self._stop_requested = True

    async def run(self) -> RatchetReport:
        """Execute the ratchet loop."""
        config = self.config
        report = RatchetReport(experiment_id=config.experiment_id)
        t0 = time.monotonic()

        # Get baseline SHA
        baseline_sha = _git_head_sha(config.work_dir)
        best_metric_value: float | None = None

        logger.info(
            "Experiment ratchet started: %s (max=%d iterations, budget=%ds/iter)",
            config.experiment_id, config.max_iterations, config.time_budget_secs,
        )

        for i in range(config.max_iterations):
            if self._stop_requested:
                report.stopped_reason = "manual_stop"
                break

            # Check total time budget
            elapsed = time.monotonic() - t0
            if config.total_budget_secs and elapsed > config.total_budget_secs:
                report.stopped_reason = "budget_exhausted"
                break

            # Get the current SHA before this iteration
            pre_sha = _git_head_sha(config.work_dir)

            # --- Step 1: Propose a change ---
            iteration_t0 = time.monotonic()
            try:
                proposal = await self._propose_change(i + 1, report)
                if not proposal:
                    logger.info("No proposal for iteration %d, skipping", i + 1)
                    continue
            except Exception as e:
                logger.warning("Proposal failed for iteration %d: %s", i + 1, e)
                result = self.tracker.record_attempt(
                    status="error",
                    error=f"Proposal failed: {e}",
                    duration_secs=time.monotonic() - iteration_t0,
                )
                report.history.append(result)
                report.iterations_errored += 1
                report.iterations_run += 1
                continue

            # --- Step 2: Apply the change ---
            try:
                await self._apply_change(proposal)
            except Exception as e:
                logger.warning("Apply failed for iteration %d: %s", i + 1, e)
                _git_reset_hard(config.work_dir, pre_sha)
                result = self.tracker.record_attempt(
                    status="error",
                    error=f"Apply failed: {e}",
                    params=proposal.get("params", {}),
                    duration_secs=time.monotonic() - iteration_t0,
                )
                report.history.append(result)
                report.iterations_errored += 1
                report.iterations_run += 1
                continue

            # --- Step 3: Evaluate (run tests / measure metric) ---
            try:
                eval_result = await self._evaluate(i + 1)
                metric_value = eval_result.get(config.metric_name)
            except Exception as e:
                logger.warning("Evaluation failed for iteration %d: %s", i + 1, e)
                _git_reset_hard(config.work_dir, pre_sha)
                result = self.tracker.record_attempt(
                    status="error",
                    error=f"Evaluation failed: {e}",
                    params=proposal.get("params", {}),
                    duration_secs=time.monotonic() - iteration_t0,
                )
                report.history.append(result)
                report.iterations_errored += 1
                report.iterations_run += 1
                continue

            # --- Step 4: Keep or revert ---
            improved = self._is_improvement(metric_value, best_metric_value)
            duration = time.monotonic() - iteration_t0

            if improved:
                # Keep: commit the change
                commit_msg = (
                    f"ratchet({config.experiment_id}): iteration {i+1} — "
                    f"{config.metric_name}={metric_value}"
                )
                sha = _git_commit(config.work_dir, commit_msg, config.target_files)
                best_metric_value = metric_value
                report.best_metrics = eval_result
                report.best_iteration = i + 1
                report.best_git_sha = sha
                report.iterations_kept += 1

                result = self.tracker.record_attempt(
                    status="improved",
                    metrics=eval_result,
                    params=proposal.get("params", {}),
                    description=proposal.get("description", ""),
                    git_sha=sha,
                    duration_secs=duration,
                )
                logger.info(
                    "Iteration %d KEPT: %s=%s (sha=%s)",
                    i + 1, config.metric_name, metric_value, sha[:8] if sha else "?",
                )
            else:
                # Revert: reset to pre-change state
                _git_reset_hard(config.work_dir, pre_sha)
                report.iterations_reverted += 1

                result = self.tracker.record_attempt(
                    status="no_improvement",
                    metrics=eval_result,
                    params=proposal.get("params", {}),
                    description=proposal.get("description", ""),
                    duration_secs=duration,
                )
                logger.info(
                    "Iteration %d REVERTED: %s=%s (best=%s)",
                    i + 1, config.metric_name, metric_value, best_metric_value,
                )

            report.history.append(result)
            report.iterations_run += 1

            # Check threshold
            if (
                config.quality_threshold
                and best_metric_value is not None
                and self._meets_threshold(best_metric_value)
            ):
                report.stopped_reason = "threshold_reached"
                logger.info(
                    "Threshold reached: %s=%s >= %s",
                    config.metric_name, best_metric_value, config.quality_threshold,
                )
                break

        if not report.stopped_reason:
            report.stopped_reason = "max_iterations"

        report.total_duration_secs = time.monotonic() - t0
        logger.info(
            "Ratchet complete: %d iterations, %d kept, %d reverted, %d errors. "
            "Best: %s=%s at iteration %d. Duration: %.1fs",
            report.iterations_run, report.iterations_kept,
            report.iterations_reverted, report.iterations_errored,
            config.metric_name, report.best_metrics.get(config.metric_name),
            report.best_iteration, report.total_duration_secs,
        )

        # Deposit final summary to knowledge.db
        self._deposit_summary(report)
        return report

    def _is_improvement(self, current: Any, best: Any) -> bool:
        """Check if current metric is better than best."""
        if current is None:
            return False
        if best is None:
            return True  # first valid result is always an improvement
        try:
            current_f = float(current)
            best_f = float(best)
            if self.config.metric_goal == "minimize":
                return current_f < best_f
            return current_f > best_f
        except (TypeError, ValueError):
            return False

    def _meets_threshold(self, value: Any) -> bool:
        """Check if metric meets the quality threshold."""
        try:
            v = float(value)
            t = self.config.quality_threshold
            if self.config.metric_goal == "minimize":
                return v <= t
            return v >= t
        except (TypeError, ValueError):
            return False

    async def _propose_change(
        self, iteration: int, report: RatchetReport
    ) -> dict[str, Any]:
        """Ask the agent to propose the next change.

        Override _propose_fn for custom proposal logic.
        Default: builds a prompt with failure history and asks the head.
        """
        if self._propose_fn:
            return await self._propose_fn(iteration, report, self.tracker)

        # Default: use head_manager to generate a proposal
        if not self.hm:
            raise RuntimeError("No head_manager and no propose_fn — cannot propose changes")

        # Build context from failure history
        failures = self.tracker.get_failures(limit=10)
        failure_ctx = ""
        if failures:
            failure_ctx = "\n\nPrevious failed attempts (DO NOT repeat these):\n"
            for f in failures:
                failure_ctx += f"- Iteration {f.get('iteration')}: {f.get('description', '')} — {f.get('error', f.get('status', ''))}\n"

        best = self.tracker.get_best()
        best_ctx = ""
        if best:
            best_ctx = f"\n\nCurrent best result: {best.get('metrics', {})} at iteration {best.get('iteration')}"

        prompt = (
            f"You are optimizing '{self.config.experiment_id}'.\n"
            f"Target files: {', '.join(self.config.target_files)}\n"
            f"Metric to optimize: {self.config.metric_name} ({self.config.metric_goal})\n"
            f"Iteration: {iteration}/{self.config.max_iterations}\n"
            f"{best_ctx}{failure_ctx}\n\n"
            f"Propose ONE change. Keep it simple — small targeted edits beat large rewrites.\n"
            f"Return JSON: {{\"description\": \"...\", \"params\": {{}}, \"changes\": [{{\"file\": \"...\", \"old\": \"...\", \"new\": \"...\"}}]}}"
        )

        if self.config.system_prompt:
            prompt = f"{self.config.system_prompt}\n\n{prompt}"

        adapter = self.hm.get_adapter(self.config.head_id)
        result = await adapter.generate(prompt)
        raw = result.get("text", "") if isinstance(result, dict) else str(result)

        # Try to parse JSON from response
        import json
        import re

        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return {"description": raw[:200], "params": {}, "changes": []}

    async def _apply_change(self, proposal: dict[str, Any]) -> None:
        """Apply proposed changes to files."""
        changes = proposal.get("changes", [])
        for change in changes:
            file_path = change.get("file", "")
            if not file_path:
                continue

            full_path = Path(self.config.work_dir) / file_path
            if not full_path.exists():
                logger.warning("Target file does not exist: %s", full_path)
                continue

            old_text = change.get("old", "")
            new_text = change.get("new", "")

            if old_text and new_text:
                content = full_path.read_text()
                if old_text in content:
                    content = content.replace(old_text, new_text, 1)
                    full_path.write_text(content)
                else:
                    logger.warning("Old text not found in %s, skipping", file_path)

    async def _evaluate(self, iteration: int) -> dict[str, Any]:
        """Run test command and extract metrics.

        Override _evaluate_fn for custom evaluation logic.
        """
        if self._evaluate_fn:
            return await self._evaluate_fn(iteration)

        if not self.config.test_command:
            return {}

        # Run test command with time budget
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    self.config.test_command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self.config.time_budget_secs,
                    cwd=self.config.work_dir,
                    creationflags=no_window_flags(),
                ),
            )
            return {
                "exit_code": result.returncode,
                "passed": result.returncode == 0,
                self.config.metric_name: 1.0 if result.returncode == 0 else 0.0,
                "stdout_tail": result.stdout[-500:] if result.stdout else "",
                "stderr_tail": result.stderr[-500:] if result.stderr else "",
            }
        except subprocess.TimeoutExpired:
            return {
                "exit_code": -1,
                "passed": False,
                self.config.metric_name: 0.0,
                "error": f"Timeout after {self.config.time_budget_secs}s",
            }

    def _deposit_summary(self, report: RatchetReport) -> None:
        """Deposit a summary claim for the entire ratchet run."""
        if not self.ks:
            return
        try:
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

            statement = (
                f"Experiment ratchet '{self.config.experiment_id}' completed: "
                f"{report.iterations_run} iterations, {report.iterations_kept} kept, "
                f"{report.iterations_reverted} reverted. "
                f"Best {self.config.metric_name}={report.best_metrics.get(self.config.metric_name)} "
                f"at iteration {report.best_iteration}. "
                f"Stopped: {report.stopped_reason}. "
                f"Duration: {report.total_duration_secs:.1f}s"
            )

            claim = Claim(
                claim_status=ClaimStatus.ACCEPTED,
                claim_type=ClaimType.FACT,
                scope=ClaimScope(
                    scope_type=ScopeType.PROJECT,
                    scope_id="experiments",
                ),
                canonical=ClaimCanonical(
                    claim_key=f"ratchet.summary.{self.config.experiment_id}.{int(time.time())}",
                    subject=EntityRef(
                        entity_type="experiment",
                        entity_id=self.config.experiment_id,
                        label=self.config.experiment_id,
                    ),
                    predicate="completed",
                    object=ValueObject(
                        value_type="json",
                        value={
                            "iterations_run": report.iterations_run,
                            "iterations_kept": report.iterations_kept,
                            "iterations_reverted": report.iterations_reverted,
                            "best_metrics": report.best_metrics,
                            "best_iteration": report.best_iteration,
                            "best_git_sha": report.best_git_sha,
                            "stopped_reason": report.stopped_reason,
                            "total_duration_secs": report.total_duration_secs,
                        },
                    ),
                ),
                statement=statement[:500],
                confidence=0.95,
                provenance=Provenance(
                    produced_by={"kind": "agent", "id": "experiment-ratchet"},
                    observation_method="experiment_summary",
                ),
            )
            self.ks.insert_claim(claim)
        except Exception as e:
            logger.warning("Failed to deposit ratchet summary: %s", e)
