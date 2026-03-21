"""Step execution and reflection logic for the orchestrator."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..consensus import ConsensusEngine
from ..models import (
    EventKind,
    RunEvent,
    RunState,
    StageResult,
    StepStatus,
    WorkOrder,
)
from ..reflection import ReflectionEngine, ReflectionMemory
from ..tot_integration import execute_step_with_tot, get_tot_config, should_use_tot
from ..prm_integration import get_prm_config, should_use_prm
from ..process_reward_models import RubricPRMScorer
from ..validators import ValidationResult

logger = logging.getLogger(__name__)


class ExecutionMixin:
    """Mixin providing step execution and reflection to the Orchestrator."""

    async def _execute_step_with_reflection(
        self,
        run_id: str,
        step: Any,
        work_order: WorkOrder,
        state: RunState,
        run_artifacts_dir: Path,
        reflection_memory: ReflectionMemory | None = None,
    ) -> StageResult:
        """Execute step with optional reflection loop for self-correction.

        If step.enable_reflection is True and step.evaluator is provided:
        1. Execute step
        2. Evaluate output quality
        3. If failed and retry recommended, refine prompt with reflection context
        4. Re-execute up to max_reflection_attempts times
        5. Store reflections in memory for learning

        Args:
            run_id: Run identifier
            step: Step definition
            work_order: Complete work order
            state: Current run state
            run_artifacts_dir: Directory for artifacts
            reflection_memory: Optional shared memory across steps

        Returns:
            Final execution result (success or failure after max attempts)
        """
        # Tree-of-Thoughts: explore alternative reasoning paths before execution
        if should_use_tot(step):
            tot_config = get_tot_config(step)
            logger.info(
                "Step %s using Tree-of-Thoughts (%s, %d alternatives, depth %d)",
                step.step_id,
                tot_config["strategy"].value,
                tot_config["num_alternatives"],
                tot_config["max_depth"],
            )

            # Emit ToT started event
            self.events.append(RunEvent(
                run_id=run_id,
                kind=EventKind.STEP_TOT_STARTED,
                step_id=step.step_id,
                data={
                    "strategy": tot_config["strategy"].value,
                    "num_alternatives": tot_config["num_alternatives"],
                    "max_depth": tot_config["max_depth"],
                },
            ))

            # Wrap _execute_step as the callable for ToT exploration
            async def _tot_execute_func(s):
                return await self._execute_step(
                    run_id, s, work_order, state, run_artifacts_dir
                )

            tot_result = await execute_step_with_tot(
                _tot_execute_func,
                step,
                {"goal": work_order.goal},
                num_alternatives=tot_config["num_alternatives"],
                strategy=tot_config["strategy"],
                max_depth=tot_config["max_depth"],
            )

            # Emit ToT complete event
            self.events.append(RunEvent(
                run_id=run_id,
                kind=EventKind.STEP_TOT_COMPLETE,
                step_id=step.step_id,
                data={
                    "strategy": tot_config["strategy"].value,
                    "result_status": tot_result.status.value if tot_result else "none",
                },
            ))

            return tot_result

        if not step.enable_reflection or step.evaluator is None:
            # No reflection - execute normally
            return await self._execute_step(
                run_id, step, work_order, state, run_artifacts_dir
            )

        # Create reflection engine
        memory = reflection_memory or ReflectionMemory()
        engine = ReflectionEngine(
            evaluator=step.evaluator,
            max_attempts=step.max_reflection_attempts,
            memory=memory,
        )

        # Work on a copy to avoid mutating the original step (critical for parallel DAG)
        original_prompt = step.prompt_template
        current_step = step

        attempt = 1
        while attempt <= step.max_reflection_attempts:
            # Execute step (using copy, never the original)
            result = await self._execute_step(
                run_id, current_step, work_order, state, run_artifacts_dir
            )

            # Check if refinement needed
            context = {
                "step_results": {sid: r.model_dump() for sid, r in state.step_results.items()},
                "run_id": run_id,
            }

            reflection = await engine.should_refine(current_step, result, context, attempt)

            if reflection is None:
                # Passed or max attempts reached - done
                if attempt > 1:
                    logger.info(
                        "Step %s succeeded after %d attempts (reflection enabled)",
                        step.step_id, attempt
                    )
                return result

            # Emit reflection event
            self.events.append(RunEvent(
                run_id=run_id,
                kind=EventKind.STEP_REFLECTION,
                step_id=step.step_id,
                data={
                    "attempt": attempt,
                    "quality_score": reflection.quality_score,
                    "reflection_text": reflection.reflection_text,
                    "suggested_changes": reflection.suggested_changes,
                },
            ))

            logger.info(
                "Step %s reflection (attempt %d/%d): quality=%.2f, will retry",
                step.step_id, attempt, step.max_reflection_attempts,
                reflection.quality_score
            )

            # Refine prompt via model_copy (never mutate the original step)
            refinement_context = engine.get_refinement_context(step.step_id)
            current_step = step.model_copy(
                update={"prompt_template": original_prompt + refinement_context}
            )

            # Emit refinement event
            self.events.append(RunEvent(
                run_id=run_id,
                kind=EventKind.STEP_REFINED,
                step_id=step.step_id,
                data={
                    "attempt": attempt,
                    "refinement_applied": True,
                    "original_prompt_length": len(original_prompt),
                    "refined_prompt_length": len(current_step.prompt_template),
                },
            ))

            attempt += 1

        # Max attempts reached - return last result (original step is unmodified)
        return result

    async def _execute_step(
        self,
        run_id: str,
        step: Any,
        work_order: WorkOrder,
        state: RunState,
        run_artifacts_dir: Path,
    ) -> StageResult:
        """Execute a single pipeline step.

        Phase 6 integration: Before local execution, check if step should be
        delegated to marketplace via RFQ. If yes, delegate instead of running locally.
        """
        # Phase 6: Check marketplace delegation
        if self._should_delegate_to_marketplace(step):
            return await self._delegate_to_marketplace(
                run_id, step, work_order, state, run_artifacts_dir
            )

        # Local execution path
        # Emit STEP_STARTED
        self.events.append(RunEvent(
            run_id=run_id,
            kind=EventKind.STEP_STARTED,
            step_id=step.step_id,
            data={"head_id": step.head_id},
        ))

        try:
            # Validator: Check preconditions BEFORE execution
            if step.validator is not None:
                try:
                    validation_state = {
                        "step_results": {sid: r.model_dump() for sid, r in state.step_results.items()},
                        "run_id": run_id,
                    }
                    validation_inputs = dict(work_order.inputs) if work_order.inputs else {}

                    precondition_result = step.validator.validate_precondition(
                        validation_state, validation_inputs
                    )
                except Exception as e:
                    # Validator exception - treat as validation failure
                    logger.error("Validator exception during precondition check: %s", e)
                    precondition_result = ValidationResult(
                        passed=False,
                        violations=[f"Validator error: {str(e)}"],
                        confidence=0.0
                    )

                if not precondition_result.passed:
                    # Precondition failed - skip execution
                    logger.warning(
                        "Step %s precondition failed: %s",
                        step.step_id,
                        ", ".join(precondition_result.violations)
                    )

                    # Emit validation failure event
                    self.events.append(RunEvent(
                        run_id=run_id,
                        kind=EventKind.STEP_FAILED,
                        step_id=step.step_id,
                        data={
                            "head_id": step.head_id,
                            "error": "Precondition validation failed",
                            "violations": precondition_result.violations,
                            "validation_confidence": precondition_result.confidence,
                        },
                    ))

                    result = StageResult(
                        step_id=step.step_id,
                        head_id=step.head_id,
                        status=StepStatus.FAILED,
                        error=f"Precondition failed: {', '.join(precondition_result.violations)}",
                        ended_at=datetime.now(timezone.utc),
                    )
                    state.step_results[step.step_id] = result
                    return result

            # Build prompt from template + inputs + previous artifacts
            prompt = self._build_prompt(step, work_order, state)

            # Consensus mode: execute across multiple heads
            if step.consensus is not None:
                engine = ConsensusEngine(self.heads, metrics=self._metrics)
                consensus_result = await engine.execute(step.consensus, prompt)

                text_output = consensus_result.consensus_outputs.get("text", "")
                warnings = [f["message"] for f in consensus_result.red_flags]

                if step.consensus.fail_on_disagreement and consensus_result.red_flags:
                    critical = [f for f in consensus_result.red_flags if f["severity"] == "critical"]
                    if critical:
                        raise ValueError(
                            f"Consensus failed with {len(critical)} critical red flag(s): "
                            f"{critical[0]['message']}"
                        )

                response = consensus_result.consensus_outputs
                response.setdefault("tokens_in", 0)
                response.setdefault("tokens_out", 0)
                response["consensus_agreement"] = consensus_result.agreement_score
                response["consensus_red_flags"] = len(consensus_result.red_flags)
            else:
                # Standard single-head execution (through HeadManager for circuit breaker)
                response = await self.heads.generate(step.head_id, prompt)
                warnings = []

            text_output = response.get("text", "")

            # Validator: Check postconditions AFTER execution
            if step.validator is not None:
                try:
                    validation_state = {
                        "step_results": {sid: r.model_dump() for sid, r in state.step_results.items()},
                        "run_id": run_id,
                    }

                    postcondition_result = step.validator.validate_postcondition(
                        validation_state, text_output
                    )
                except Exception as e:
                    # Validator exception - treat as validation failure
                    logger.error("Validator exception during postcondition check: %s", e)
                    postcondition_result = ValidationResult(
                        passed=False,
                        violations=[f"Validator error: {str(e)}"],
                        confidence=0.0
                    )

                if not postcondition_result.passed:
                    # Postcondition failed - discard output
                    logger.warning(
                        "Step %s postcondition failed: %s",
                        step.step_id,
                        ", ".join(postcondition_result.violations)
                    )

                    # Emit validation failure event
                    self.events.append(RunEvent(
                        run_id=run_id,
                        kind=EventKind.STEP_FAILED,
                        step_id=step.step_id,
                        data={
                            "head_id": step.head_id,
                            "error": "Postcondition validation failed",
                            "violations": postcondition_result.violations,
                            "validation_confidence": postcondition_result.confidence,
                            "output_discarded": True,
                        },
                    ))

                    result = StageResult(
                        step_id=step.step_id,
                        head_id=step.head_id,
                        status=StepStatus.FAILED,
                        error=f"Postcondition failed: {', '.join(postcondition_result.violations)}",
                        warnings=[f"Confidence: {postcondition_result.confidence:.2f}"],
                        ended_at=datetime.now(timezone.utc),
                    )
                    state.step_results[step.step_id] = result
                    return result

            # Store output as artifact
            artifact_ref = self.artifacts.store(
                data=text_output.encode("utf-8"),
                name=f"{step.step_id}_output.txt",
                media_type="text/plain",
            )

            # Also write to run artifacts dir for easy inspection
            safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", step.name)[:100]
            output_file = run_artifacts_dir / f"{safe_name}_output.txt"
            output_file.write_text(text_output)

            # Emit STEP_OUTPUT_WRITTEN
            self.events.append(RunEvent(
                run_id=run_id,
                kind=EventKind.STEP_OUTPUT_WRITTEN,
                step_id=step.step_id,
                data={
                    "artifact_id": artifact_ref.artifact_id,
                    "output_file": str(output_file),
                },
            ))

            # Build stage result
            head_label = (
                f"consensus-{step.consensus.strategy.value}"
                if step.consensus else step.head_id
            )
            metrics = {
                "tokens_in": response.get("tokens_in", 0),
                "tokens_out": response.get("tokens_out", 0),
            }
            if step.consensus is not None:
                metrics["consensus_agreement"] = response.get("consensus_agreement", 0)
                metrics["consensus_red_flags"] = response.get("consensus_red_flags", 0)

            result = StageResult(
                step_id=step.step_id,
                head_id=head_label,
                status=StepStatus.COMMITTED,
                outputs={"text": text_output},
                output_artifacts=[artifact_ref],
                metrics=metrics,
                warnings=warnings,
                ended_at=datetime.now(timezone.utc),
            )

            # Emit STEP_COMMITTED (sync checkpoint)
            self.events.append(RunEvent(
                run_id=run_id,
                kind=EventKind.STEP_COMMITTED,
                step_id=step.step_id,
                data={
                    "head_id": step.head_id,
                    "outputs": {"text_length": len(text_output)},
                    "artifact_id": artifact_ref.artifact_id,
                    "metrics": result.metrics,
                },
            ))

            state.step_results[step.step_id] = result

            # Real-time knowledge extraction
            if self.knowledge_hook:
                try:
                    context = {
                        "run_id": run_id,
                        "goal": work_order.goal,
                        "previous_steps": list(state.step_results.values()),
                    }
                    knowledge_meta = await self.knowledge_hook.on_step_complete(
                        step, result, context
                    )
                    logger.debug(
                        "Knowledge extracted: %s",
                        knowledge_meta.get("learning_summary", ""),
                    )
                except Exception as e:
                    logger.warning("Knowledge hook failed: %s", e)

            # Automatic test generation for code steps
            if self.test_hook:
                try:
                    context = {
                        "run_id": run_id,
                        "goal": work_order.goal,
                        "previous_steps": list(state.step_results.values()),
                    }
                    test_meta = await self.test_hook.on_step_complete(
                        step, result, context
                    )
                    if test_meta.get("tests_generated"):
                        logger.info(
                            "Tests generated for %s: %d iterations, converged=%s",
                            step.step_id,
                            test_meta.get("iterations", 0),
                            test_meta.get("converged", False),
                        )
                except Exception as e:
                    logger.warning("Test generation hook failed: %s", e)

            # Process Reward Model: step-level quality scoring
            if should_use_prm(step):
                try:
                    prm_config = get_prm_config(step)
                    # Build a simple rubric scorer from config
                    # (LLM scorer would require head access, rubric is deterministic)
                    def _has_output(r, ctx):
                        return bool(r.outputs and r.outputs.get("text"))

                    def _no_errors(r, ctx):
                        return not r.error

                    def _reasonable_length(r, ctx):
                        text = r.outputs.get("text", "") if r.outputs else ""
                        return len(text) >= 10

                    rubric = [
                        ("Produces output", _has_output, 0.4),
                        ("No errors", _no_errors, 0.4),
                        ("Reasonable length", _reasonable_length, 0.2),
                    ]
                    scorer = RubricPRMScorer(rubric)

                    prm_context = {
                        "run_id": run_id,
                        "goal": work_order.goal,
                        "previous_steps": list(state.step_results.values()),
                    }
                    prm_score = await scorer.score_step(result, prm_context)

                    # Store PRM metrics in result
                    if result.metrics is None:
                        result.metrics = {}
                    result.metrics["prm_score"] = prm_score.score
                    result.metrics["prm_quality"] = prm_score.quality.value
                    result.metrics["prm_confidence"] = prm_score.confidence

                    # Emit PRM scored event
                    self.events.append(RunEvent(
                        run_id=run_id,
                        kind=EventKind.STEP_PRM_SCORED,
                        step_id=step.step_id,
                        data={
                            "prm_score": prm_score.score,
                            "prm_quality": prm_score.quality.value,
                            "prm_feedback": prm_score.feedback,
                            "prm_issues": prm_score.issues,
                            "prm_threshold": prm_config["threshold"],
                        },
                    ))

                    logger.info(
                        "PRM score for %s: %.2f (%s) - %s",
                        step.step_id,
                        prm_score.score,
                        prm_score.quality.value,
                        prm_score.feedback,
                    )

                    # Optionally fail step if score below threshold
                    if prm_config.get("fail_on_low_score") and not prm_score.is_acceptable(
                        prm_config["threshold"]
                    ):
                        logger.warning(
                            "Step %s PRM score %.2f below threshold %.2f — marking as failed",
                            step.step_id,
                            prm_score.score,
                            prm_config["threshold"],
                        )
                        result.status = StepStatus.FAILED
                        result.error = (
                            f"PRM quality gate failed: score {prm_score.score:.2f} "
                            f"< threshold {prm_config['threshold']:.2f}"
                        )
                        result.warnings.extend(prm_score.issues)

                except Exception as e:
                    logger.warning("PRM scoring failed for %s: %s", step.step_id, e)

            return result

        except Exception as e:
            logger.error("Step %s failed: %s", step.step_id, e)

            result = StageResult(
                step_id=step.step_id,
                head_id=step.head_id,
                status=StepStatus.FAILED,
                error=str(e),
                ended_at=datetime.now(timezone.utc),
            )

            self.events.append(RunEvent(
                run_id=run_id,
                kind=EventKind.STEP_FAILED,
                step_id=step.step_id,
                data={"head_id": step.head_id, "error": str(e)},
            ))

            state.step_results[step.step_id] = result

            # Real-time knowledge extraction for failures
            if self.knowledge_hook:
                try:
                    context = {
                        "run_id": run_id,
                        "goal": work_order.goal,
                        "previous_steps": list(state.step_results.values()),
                    }
                    knowledge_meta = await self.knowledge_hook.on_step_complete(
                        step, result, context
                    )
                    logger.debug(
                        "Knowledge extracted from failure: %s",
                        knowledge_meta.get("learning_summary", ""),
                    )
                except Exception as hook_error:
                    logger.warning("Knowledge hook failed: %s", hook_error)

            # Test generation hook (will skip for failed steps)
            if self.test_hook:
                try:
                    context = {
                        "run_id": run_id,
                        "goal": work_order.goal,
                        "previous_steps": list(state.step_results.values()),
                    }
                    test_meta = await self.test_hook.on_step_complete(
                        step, result, context
                    )
                except Exception as hook_error:
                    logger.warning("Test generation hook failed: %s", hook_error)

            return result
