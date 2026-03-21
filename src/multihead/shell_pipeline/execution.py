"""Stage 3: Execution — task decomposition and VLM routing.

Handles:
- Complex task decomposition via orchestrator
- Image input detection and VLM routing
- Run result formatting
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .constants import BrainFn

logger = logging.getLogger(__name__)


def detect_image_input(user_input: str) -> str | None:
    """Check if user references an image file path."""
    pattern = r"[\w/\\.-]+\.(?:png|jpg|jpeg|gif|bmp|webp|svg)"
    m = re.search(pattern, user_input, re.IGNORECASE)
    return m.group(0) if m else None


async def route_to_vlm(
    user_input: str,
    image_path: str,
    knowledge_ctx: str,
    session_id: str,
    brain_fn: BrainFn,
    router: Any,
    head_manager: Any,
) -> str | None:
    """Route image-containing input to a VLM head.

    Returns the VLM response, or None if no VLM is available
    (falls back to normal brain routing).
    """
    if not router or not head_manager:
        return None

    try:
        vlm_head = router.route(required_kind="vlm")
        if not vlm_head:
            logger.debug("No VLM head available for image routing")
            return None

        # Wake the VLM head if needed
        await head_manager.ensure_active(vlm_head)

        # Get the adapter and generate
        adapter = head_manager.get_adapter(vlm_head)
        if not adapter:
            return None

        prompt = f"{user_input}\n\n[Image: {image_path}]"
        if knowledge_ctx:
            prompt = f"{knowledge_ctx}\n\n{prompt}"

        result = await adapter.generate(prompt, image_path=image_path)
        return result.get("text", "") if isinstance(result, dict) else str(result)

    except Exception as e:
        logger.debug("VLM routing failed, falling back: %s", e)
        return None


async def execute_as_task(
    user_input: str,
    knowledge_ctx: str,
    session_id: str,
    brain_fn: BrainFn,
    config: Any,
    decomposer: Any,
    orchestrator: Any,
    stats: dict[str, int],
) -> str:
    """Decompose and execute a complex task through the orchestrator.

    Falls back to direct brain call if decomposer/orchestrator unavailable.
    """
    if not decomposer or not orchestrator:
        return await brain_fn(session_id, user_input, knowledge_ctx)

    # Resolve which head to decompose with
    decompose_head = (
        config.pipeline.decompose_head
        if config and config.pipeline.decompose_head
        else None
    )

    # If no explicit head configured, fall back to brain (don't auto-wake Qwen)
    if not decompose_head:
        return await brain_fn(session_id, user_input, knowledge_ctx)

    try:
        # 1. Decompose with configured head
        plan = await decomposer.decompose(
            goal=user_input,
            context=knowledge_ctx,
            head_id=decompose_head,
            enable_research_features=True,
        )

        # 2. Convert to WorkOrder with DAG
        work_order = decomposer.to_work_order_with_dag(plan)

        # 3. Execute via orchestrator
        run_state = await orchestrator.create_run(work_order)
        run_state = await orchestrator.execute_run(run_state.run_id)

        stats["tasks_decomposed"] += 1

        # 4. Format results
        return format_run_results(run_state)

    except Exception as e:
        logger.warning("Task decomposition failed, falling back to brain: %s", e)
        return await brain_fn(session_id, user_input, knowledge_ctx)


def format_run_results(run_state: Any) -> str:
    """Format orchestrator run results into a readable response."""
    parts: list[str] = []
    parts.append(f"**Task completed** (run: {run_state.run_id[:12]}...)")
    parts.append(f"Status: {run_state.status}")

    step_results = getattr(run_state, "step_results", {})
    if step_results:
        parts.append("")
        parts.append("**Steps:**")
        for step_id, result in step_results.items():
            status = getattr(result, "status", "unknown")
            output = getattr(result, "output", "")
            if isinstance(output, str) and len(output) > 200:
                output = output[:200] + "..."
            parts.append(f"- {step_id}: {status} — {output}")

    return "\n".join(parts)
