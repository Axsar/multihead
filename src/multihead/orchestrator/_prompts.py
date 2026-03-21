"""Prompt-building logic for the orchestrator."""

from __future__ import annotations

import logging
from typing import Any

from ..models import RunState, WorkOrder
from ._helpers import _SafeFormatDict

logger = logging.getLogger(__name__)


class PromptMixin:
    """Mixin providing prompt-building capabilities to the Orchestrator."""

    def _build_prompt(self, step: Any, work_order: WorkOrder, state: RunState) -> str:
        """Build prompt for a step from template + context.

        Template variables are substituted via str.format_map:
          {goal}                  - work order goal
          {input_path}            - from inputs dict
          {previous_step_output}  - text from last input_ref
          Any key from inputs     - e.g. {input_path}
        Unrecognised braces are left intact (safe for JSON in templates).
        """
        if step.prompt_template:
            prompt = step.prompt_template
        else:
            prompt = f"Task: {step.name}\nGoal: {work_order.goal}"

        # Build name→step_id map so input_refs (step names) resolve
        name_to_id: dict[str, str] = {}
        for s in work_order.steps:
            name_to_id[s.name] = s.step_id

        # Collect previous step outputs text
        prev_outputs: dict[str, str] = {}
        for ref in step.input_refs:
            # ref can be a step name or a step_id
            step_id = name_to_id.get(ref, ref)
            if step_id in state.step_results:
                prev_result = state.step_results[step_id]
                text = prev_result.outputs.get("text", "")
                if text:
                    prev_outputs[ref] = text

        # Build substitution context
        fmt: dict[str, str] = {}
        fmt["goal"] = work_order.goal or ""
        # All user-provided inputs (input_path, etc.)
        if work_order.inputs:
            for k, v in work_order.inputs.items():
                fmt[k] = str(v)
        # previous_step_output = last input_ref's text (most common pattern)
        if prev_outputs:
            last_ref = step.input_refs[-1]
            fmt["previous_step_output"] = prev_outputs.get(last_ref, "")
        # Also expose each ref by name
        for ref_name, ref_text in prev_outputs.items():
            fmt[ref_name] = ref_text

        # Safe format: only substitute keys present, leave unknown braces intact
        try:
            prompt = prompt.format_map(_SafeFormatDict(fmt))
        except Exception:
            # Fallback: don't crash on bad templates, just append context
            logger.warning("Template substitution failed for step %s, using raw template", step.name)

        # If template didn't consume previous outputs, append them
        if prev_outputs and "{previous_step_output}" not in (step.prompt_template or ""):
            for ref, text in prev_outputs.items():
                if f"{{{ref}}}" not in (step.prompt_template or ""):
                    prompt += f"\n\nPrevious step ({ref}) output:\n{text}"

        return prompt
