"""Marketplace delegation logic for the orchestrator."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..models import (
    EventKind,
    RunEvent,
    RunState,
    StageResult,
    StepStatus,
    WorkOrder,
)

logger = logging.getLogger(__name__)


class DelegationMixin:
    """Mixin providing marketplace delegation capabilities to the Orchestrator."""

    # ------------------------------------------------------------------
    # Capability mapping: task_types → ACP capability strings
    # ------------------------------------------------------------------

    _TASK_TYPE_TO_CAPABILITY: dict[str, str] = {
        "text_generation": "com.multihead.llm",
        "text_summarization": "com.multihead.llm",
        "code_generation": "com.multihead.llm",
        "visual_reasoning": "com.multihead.vlm",
        "image_understanding": "com.multihead.vlm",
        "image_classification": "com.multihead.vlm",
        "code_editing": "com.claude.code",
        "code_review": "com.claude.code",
        "refactoring": "com.claude.code",
        "testing": "com.claude.code",
        "coordinate_transform": "com.multihead.deterministic",
        "data_preprocessing": "com.multihead.deterministic",
        "object_detection": "com.multihead.vlm",
        "segmentation": "com.multihead.vlm",
        "data_processing": "com.multihead.llm",
    }

    def _map_task_types_to_capability(self, task_types: list[str]) -> str:
        """Map step task_types to the best ACP capability string.

        Returns the most specific capability matching the first recognized task type.
        Falls back to 'com.multihead.llm' if no mapping exists.
        """
        for tt in task_types:
            cap = self._TASK_TYPE_TO_CAPABILITY.get(tt)
            if cap:
                return cap
        return "com.multihead.llm"

    def _should_delegate_to_marketplace(self, step: Any) -> bool:
        """Check if step should be delegated to marketplace instead of running locally.

        Delegation criteria (Phase 6):
        - Marketplace delegation enabled
        - Step has budget constraints
        - Privacy allows delegation (NOT confidential)
        - Step has task_types (for RFQ creation)

        Args:
            step: Step to check

        Returns:
            True if should delegate to marketplace
        """
        if not self.enable_marketplace_delegation:
            return False

        # Must have budget to create RFQ
        if not step.budget:
            return False

        # Must have task types for RFQ
        if not step.task_types:
            return False

        # Privacy check: CONFIDENTIAL data cannot leave local system
        if step.privacy:
            from ..models import DataSensitivity
            if step.privacy.data_sensitivity == DataSensitivity.CONFIDENTIAL:
                logger.debug("Step %s: CONFIDENTIAL data, cannot delegate", step.step_id)
                return False

        logger.info("Step %s eligible for marketplace delegation", step.step_id)
        return True

    async def _delegate_to_marketplace(
        self,
        run_id: str,
        step: Any,
        work_order: WorkOrder,
        state: RunState,
        run_artifacts_dir: Path,
    ) -> StageResult:
        """Delegate step execution to a remote agent via ACP.

        Workflow:
        1. Map task_types to ACP capability string
        2. Build prompt/payload from step context
        3. Create ACP task via acp_bridge
        4. Poll for result until complete/failed/timeout
        5. Extract output and store as artifact
        6. Return StageResult

        Falls back to RFQ-only mode if no acp_bridge is available.

        Args:
            run_id: Run identifier
            step: Step to delegate
            work_order: Full work order
            state: Current run state
            run_artifacts_dir: Artifacts directory

        Returns:
            StageResult with remote execution details
        """
        import httpx

        logger.info("Delegating step %s to marketplace", step.step_id)

        # Emit STEP_STARTED with marketplace flag
        self.events.append(RunEvent(
            run_id=run_id,
            kind=EventKind.STEP_STARTED,
            step_id=step.step_id,
            data={"head_id": "marketplace-acp", "delegation": True},
        ))

        try:
            # If no ACP bridge, fall back to RFQ-only mode
            if not self.acp_bridge or not self.acp_bridge.connected:
                return await self._delegate_rfq_only(
                    run_id, step, work_order, state, run_artifacts_dir,
                )

            # Build prompt from step template + context
            prompt = self._build_prompt(step, work_order, state)

            # Map task_types to ACP capability
            capability = self._map_task_types_to_capability(step.task_types)

            # Determine timeout from budget or default (5 minutes)
            timeout_ms = 300_000  # 5 min default
            if step.budget and step.budget.max_latency_ms:
                timeout_ms = step.budget.max_latency_ms

            # Create ACP task
            task_payload = await self.acp_bridge.request(
                "POST", "/tasks",
                json={
                    "project_id": self.acp_bridge.project_id,
                    "required_capability": capability,
                    "payload_ref": prompt,
                    "priority": "normal",
                    "input_schema": "text/plain",
                    "output_schema": "text/plain",
                    "conversation_id": run_id,
                },
            )
            task_id = task_payload.get("task_id")
            if not task_id:
                raise ValueError("ACP task creation returned no task_id")

            logger.info(
                "Created ACP task %s for step %s (capability=%s)",
                task_id[:8], step.step_id, capability,
            )

            # Poll for result
            poll_interval_s = 2.0
            elapsed_ms = 0
            task_result = None

            while elapsed_ms < timeout_ms:
                await asyncio.sleep(poll_interval_s)
                elapsed_ms += int(poll_interval_s * 1000)

                status_resp = await self.acp_bridge.request(
                    "GET", f"/tasks/{task_id}",
                )
                status = status_resp.get("status", "pending")

                if status == "complete":
                    task_result = status_resp
                    break
                elif status == "failed":
                    error_msg = status_resp.get("message", status_resp.get("error_code", "Unknown error"))
                    raise RuntimeError(f"ACP task {task_id} failed: {error_msg}")

                # Adaptive polling: slow down after first 30s
                if elapsed_ms > 30_000:
                    poll_interval_s = min(poll_interval_s * 1.5, 10.0)

            if task_result is None:
                raise TimeoutError(
                    f"ACP task {task_id} timed out after {timeout_ms}ms"
                )

            # Extract output
            text_output = task_result.get("output_ref", "")
            if isinstance(text_output, dict):
                text_output = json.dumps(text_output, indent=2)

            # Store output as artifact
            artifact_ref = self.artifacts.store(
                data=text_output.encode("utf-8"),
                name=f"{step.step_id}_acp_result.txt",
                media_type="text/plain",
            )

            safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", step.name)[:100]
            output_file = run_artifacts_dir / f"{safe_name}_acp_result.txt"
            output_file.write_text(text_output)

            # Emit STEP_OUTPUT_WRITTEN
            self.events.append(RunEvent(
                run_id=run_id,
                kind=EventKind.STEP_OUTPUT_WRITTEN,
                step_id=step.step_id,
                data={
                    "artifact_id": artifact_ref.artifact_id,
                    "output_file": str(output_file),
                    "acp_task_id": task_id,
                },
            ))

            result = StageResult(
                step_id=step.step_id,
                head_id="marketplace-acp",
                status=StepStatus.COMMITTED,
                outputs={"text": text_output, "acp_task_id": task_id},
                output_artifacts=[artifact_ref],
                metrics={
                    "acp_task_id": task_id,
                    "capability": capability,
                    "delegation": "acp",
                    "latency_ms": elapsed_ms,
                },
                ended_at=datetime.now(timezone.utc),
            )

            # Emit STEP_COMMITTED
            self.events.append(RunEvent(
                run_id=run_id,
                kind=EventKind.STEP_COMMITTED,
                step_id=step.step_id,
                data={
                    "head_id": "marketplace-acp",
                    "acp_task_id": task_id,
                    "artifact_id": artifact_ref.artifact_id,
                    "metrics": result.metrics,
                },
            ))

            state.step_results[step.step_id] = result
            return result

        except Exception as e:
            logger.error("Marketplace delegation failed for step %s: %s", step.step_id, e)

            result = StageResult(
                step_id=step.step_id,
                head_id="marketplace-acp",
                status=StepStatus.FAILED,
                error=f"Marketplace delegation failed: {str(e)}",
                ended_at=datetime.now(timezone.utc),
            )

            self.events.append(RunEvent(
                run_id=run_id,
                kind=EventKind.STEP_FAILED,
                step_id=step.step_id,
                data={"head_id": "marketplace-acp", "error": str(e)},
            ))

            state.step_results[step.step_id] = result
            return result

    async def _delegate_rfq_only(
        self,
        run_id: str,
        step: Any,
        work_order: WorkOrder,
        state: RunState,
        run_artifacts_dir: Path,
    ) -> StageResult:
        """Fallback: create RFQ without submitting to ACP (no bridge available).

        Used when marketplace delegation is enabled but ACP is not connected.
        """
        from ..rfq import RFQManager

        rfq_manager = RFQManager(project_id=run_id)
        rfq = rfq_manager.create_rfq_from_step(step, data_count=1)
        logger.info("Created RFQ %s (no ACP — offline mode)", rfq.rfq_id)

        text_output = json.dumps({
            "status": "rfq_created",
            "rfq_id": rfq.rfq_id,
            "task_type": rfq.task_type,
            "max_budget": rfq.max_total_price,
            "deadline_hours": rfq.deadline_hours,
            "note": "ACP bridge not connected. RFQ created but not submitted.",
        }, indent=2)

        artifact_ref = self.artifacts.store(
            data=text_output.encode("utf-8"),
            name=f"{step.step_id}_rfq.json",
            media_type="application/json",
        )

        safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", step.name)[:100]
        output_file = run_artifacts_dir / f"{safe_name}_rfq.json"
        output_file.write_text(text_output)

        self.events.append(RunEvent(
            run_id=run_id,
            kind=EventKind.STEP_OUTPUT_WRITTEN,
            step_id=step.step_id,
            data={
                "artifact_id": artifact_ref.artifact_id,
                "output_file": str(output_file),
                "rfq_id": rfq.rfq_id,
            },
        ))

        result = StageResult(
            step_id=step.step_id,
            head_id="marketplace-rfq",
            status=StepStatus.COMMITTED,
            outputs={"text": text_output, "rfq_id": rfq.rfq_id},
            output_artifacts=[artifact_ref],
            metrics={
                "rfq_id": rfq.rfq_id,
                "max_budget": rfq.max_total_price or 0.0,
                "delegation": "rfq_only",
            },
            warnings=["ACP bridge not connected — RFQ created but not submitted"],
            ended_at=datetime.now(timezone.utc),
        )

        self.events.append(RunEvent(
            run_id=run_id,
            kind=EventKind.STEP_COMMITTED,
            step_id=step.step_id,
            data={
                "head_id": "marketplace-rfq",
                "rfq_id": rfq.rfq_id,
                "artifact_id": artifact_ref.artifact_id,
                "metrics": result.metrics,
            },
        ))

        state.step_results[step.step_id] = result
        return result
