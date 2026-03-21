"""Vision and pipeline fulfillment handlers."""

from __future__ import annotations

from typing import Any

from ._constants import logger


class VisionFulfillmentMixin:
    """Mixin providing vision and pipeline fulfillment methods."""

    # Attributes defined on the main class.
    _heads: Any
    _agentic_core: Any
    _knowledge_store: Any
    _acp_bridge: Any
    _event_store: Any
    _artifact_store: Any
    _runs_dir: Any

    def _emit(self, event_type: str, message: str) -> None: ...
    def _svc_config(self) -> Any: ...
    def _deposit_claim(self, claim_key: str, statement: str) -> None: ...

    async def _fulfill_vision_task(
        self,
        capability_id: str,
        payload: str,
        vault_inputs: list[dict[str, Any]],
        contract_id: str,
    ) -> tuple[str, float, list[tuple[str, bytes, str]]]:
        """Fulfill a vision task (segmentation/detection) with vault file inputs.

        Routes to the VLM head for processing, returns text output plus
        any binary output files (masks, annotated images) for vault upload.
        """
        import base64 as b64

        # Build a prompt with base64-encoded images for the VLM
        image_parts: list[str] = []
        for vi in vault_inputs:
            fname = vi["filename"]
            data = vi["data"]
            ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else "png"
            mime = {
                "png": "image/png",
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "webp": "image/webp",
            }.get(ext, "image/png")
            encoded = b64.b64encode(data).decode("ascii")
            image_parts.append(
                f"[Image: {fname} ({mime}, {len(data)} bytes)]\n"
                f"data:{mime};base64,{encoded}"
            )

        task_desc = {
            "image.segment.masks.v1": (
                "Segment all objects in this image. "
                "Return mask descriptions, bounding boxes, and object labels."
            ),
            "image.detect.objects.v1": (
                "Detect all objects in this image. "
                "Return bounding boxes with labels and confidence."
            ),
        }.get(capability_id, f"Process this image for {capability_id}")

        full_prompt = (
            f"{task_desc}\n\n"
            f"Input: {payload}\n\n"
            f"Files provided: {len(vault_inputs)} image(s)\n"
            + "\n".join(image_parts)
        )

        # Route to VLM head if available, else agentic core
        output = ""
        confidence = 0.80

        try:
            # Try VLM head first
            from ..models import StepDef
            step = StepDef(
                step_id=f"vault-{contract_id[:8]}",
                prompt=full_prompt,
                required_kind="vlm",
            )
            result = await self._heads.generate(
                step.required_kind or "vlm",
                full_prompt,
            )
            output = result if isinstance(result, str) else str(result)
            confidence = 0.85
        except Exception as vlm_err:
            logger.debug("VLM head unavailable, falling back to core: %s", vlm_err)
            if self._agentic_core:
                output = await self._agentic_core.chat(
                    f"vision-{contract_id[:8]}", full_prompt,
                )
                confidence = 0.75
            else:
                output = f"No VLM head or agentic core for {capability_id}"
                confidence = 0.2

        # For now, return text results only (structured JSON with detections/masks).
        # Binary mask outputs could be added when the actual ML pipeline is wired.
        return output, confidence, []

    # ------------------------------------------------------------------
    # image.describe.v1 — VLM image captioning with Vault I/O
    # ------------------------------------------------------------------

    async def _fulfill_image_describe(
        self,
        payload: str,
        vault_inputs: list[dict[str, Any]],
        contract_id: str,
    ) -> tuple[str, float, list[tuple[str, bytes, str]]]:
        """Describe images uploaded via Vault using the VLM head.

        Accepts images from Vault, runs them through Qwen3-VL (or fallback),
        returns natural-language descriptions.  Binary outputs (annotated
        images) are returned for Vault upload back to the buyer.
        """
        import base64 as b64

        # Build VLM prompt with image data
        image_descs: list[str] = []
        for vi in vault_inputs:
            fname = vi["filename"]
            data = vi["data"]
            ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else "png"
            mime = {
                "png": "image/png", "jpg": "image/jpeg",
                "jpeg": "image/jpeg", "webp": "image/webp",
            }.get(ext, "image/png")
            encoded = b64.b64encode(data).decode("ascii")
            image_descs.append(
                f"[Image: {fname} ({mime}, {len(data)} bytes)]\n"
                f"data:{mime};base64,{encoded}"
            )

        focus = payload.strip() if payload.strip() else "Provide a detailed description"
        full_prompt = (
            f"You are an expert image analyst. Describe the following image(s) "
            f"in rich detail — subjects, composition, colors, text, mood, and "
            f"any notable elements.\n\n"
            f"Focus: {focus}\n\n"
            f"Files ({len(vault_inputs)} image(s)):\n"
            + "\n".join(image_descs)
        )

        output = ""
        confidence = 0.80

        try:
            result = await self._heads.generate("vlm", full_prompt)
            output = result if isinstance(result, str) else str(result)
            confidence = 0.90
        except Exception as vlm_err:
            logger.debug("VLM unavailable for image.describe: %s", vlm_err)
            if self._agentic_core:
                output = await self._agentic_core.chat(
                    f"img-desc-{contract_id[:8]}", full_prompt,
                )
                confidence = 0.75
            else:
                output = "No VLM or agentic core available for image description."
                confidence = 0.2

        logger.info(
            "image.describe.v1: processed %d image(s), %d chars output, conf=%.2f",
            len(vault_inputs), len(output), confidence,
        )

        # No binary outputs for now (pure text description).
        # Could add annotated images in future.
        return output, confidence, []

    # ------------------------------------------------------------------
    # Full solve pipeline for complex tasks
    # ------------------------------------------------------------------

    def _should_use_full_pipeline(self, capability_id: str, payload: str) -> bool:
        """Decide whether to route through the full solve pipeline."""
        # Must have pipeline infrastructure available
        if not (self._event_store and self._artifact_store and self._runs_dir):
            return False

        svc = self._svc_config()
        full_pipeline = getattr(svc, "cloud_full_pipeline", False) if svc else False

        if full_pipeline:
            return True

        # Auto-detect complex tasks even when not explicitly enabled
        threshold = (
            getattr(svc, "cloud_pipeline_complexity_threshold", 0.7) if svc else 0.7
        )
        return self._estimate_complexity(payload) >= threshold

    @staticmethod
    def _estimate_complexity(payload: str) -> float:
        """Heuristic complexity estimate (0.0-1.0)."""
        score = 0.0

        # Length
        if len(payload) > 500:
            score += 0.3
        elif len(payload) > 200:
            score += 0.15

        # Multi-step keywords
        multi_kw = [
            "steps", "phases", "then", "after that", "first",
            "decompose", "plan", "analyze", "implement", "verify",
            "multiple", "several", "pipeline", "workflow",
        ]
        score += min(0.4, sum(1 for kw in multi_kw if kw in payload.lower()) * 0.1)

        # Domain complexity
        domain_kw = [
            "refactor", "architecture", "migration", "integration",
            "optimize", "debug", "investigate", "research",
        ]
        score += min(0.3, sum(1 for kw in domain_kw if kw in payload.lower()) * 0.15)

        return min(1.0, score)

    async def _fulfill_via_pipeline(
        self,
        capability_id: str,
        payload: str,
        contract_id: str,
    ) -> tuple[str, float]:
        """Execute contract through the full solve pipeline."""
        from ..solve_pipeline import SolvePipeline, SolveConstraints

        svc = self._svc_config()
        max_steps = getattr(svc, "cloud_pipeline_max_steps", 15) if svc else 15
        timeout = getattr(svc, "cloud_pipeline_timeout", 180.0) if svc else 180.0

        constraints = SolveConstraints(
            max_steps=max_steps,
            max_depth=3,
            timeout_seconds=timeout,
            consensus_timeout=30.0,
            enable_knowledge_hook=bool(self._knowledge_store),
            enable_marketplace_delegation=bool(self._acp_bridge),
        )

        pipeline = SolvePipeline(
            head_manager=self._heads,
            event_store=self._event_store,
            artifact_store=self._artifact_store,
            knowledge_store=self._knowledge_store,
            runs_dir=self._runs_dir,
            acp_bridge=self._acp_bridge,
        )

        self._emit("pipeline", f"Full solve pipeline for contract {contract_id[:8]}")

        result = await pipeline.solve(
            task=payload,
            constraints=constraints,
            context={"capability_id": capability_id, "contract_id": contract_id},
        )

        logger.info(
            "Pipeline result for contract %s: status=%s, steps=%d/%d, %.1fs",
            contract_id[:8],
            result.status,
            result.steps_succeeded,
            result.steps_total,
            result.duration_seconds,
        )

        self._deposit_claim(
            f"cloud.marketplace.pipeline.{contract_id}",
            f"Contract {contract_id} via full pipeline: "
            f"{result.steps_succeeded}/{result.steps_total} steps, "
            f"conf={result.confidence:.2f}, {result.duration_seconds:.1f}s",
        )

        return result.output, result.confidence
