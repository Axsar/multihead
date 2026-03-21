"""Integration hook for automatic test generation in orchestrator.

Detects code generation steps and automatically generates/validates tests.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict
from typing import Any

from multihead.models import StageResult, StepDef, StepStatus
from multihead.test_generation import TestFramework, TestGenerator

logger = logging.getLogger(__name__)


class TestGenerationHook:
    """Hook for automatic test generation on code steps.

    Integrates with orchestrator to generate and validate tests for
    code generation steps.
    """

    def __init__(
        self,
        test_generator: TestGenerator,
        *,
        enabled: bool = True,
        code_action_types: set[str] | None = None,
    ):
        """Initialize test generation hook.

        Args:
            test_generator: TestGenerator instance
            enabled: Whether to generate tests
            code_action_types: Action types that trigger test generation
                (default: {"create", "edit", "refactor", "implement"})
        """
        self.test_generator = test_generator
        self.enabled = enabled
        self.code_action_types = code_action_types or {
            "create",
            "edit",
            "refactor",
            "implement",
        }

        # Track test results
        self.test_results: dict[str, Any] = {}

    async def on_step_complete(
        self,
        step: StepDef,
        result: StageResult,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Called when a step completes - generate tests if it's a code step.

        Args:
            step: Step definition
            result: Step execution result
            context: Execution context

        Returns:
            Dict with test generation metadata
        """
        if not self.enabled:
            return {"tests_generated": False}

        if result.status != StepStatus.COMMITTED:
            return {"tests_generated": False, "reason": "step_not_committed"}

        # Check if this is a code generation step
        action_type = step.extra.get("action_type", "") if step.extra else ""
        if action_type not in self.code_action_types:
            return {
                "tests_generated": False,
                "reason": f"action_type '{action_type}' not in {self.code_action_types}",
            }

        # Extract code from output
        output_text = result.outputs.get("text", "") if result.outputs else ""
        code = self._extract_code_from_output(output_text)

        if not code:
            return {"tests_generated": False, "reason": "no_code_found"}

        # Detect language
        language = self._detect_language(code, step)
        if language != "python":
            # Only support Python for now
            return {
                "tests_generated": False,
                "reason": f"language '{language}' not supported",
            }

        logger.info("Generating tests for step %s (code: %d chars)", step.step_id, len(code))

        try:
            # Generate tests
            test_iterations = await self.test_generator.generate_tests(
                code=code,
                code_file_path=None,  # Could extract from step if needed
                head_id=result.head_id,
                context=context,
            )

            if not test_iterations:
                return {"tests_generated": False, "reason": "generation_failed"}

            final_iteration = test_iterations[-1]
            test_result = final_iteration.execution_result

            # Store results
            self.test_results[step.step_id] = {
                "iterations": len(test_iterations),
                "converged": final_iteration.converged,
                "test_code": final_iteration.test_code,
                "execution_result": asdict(test_result) if test_result else None,
            }

            logger.info(
                "Test generation complete for %s: %d iterations, converged=%s",
                step.step_id,
                len(test_iterations),
                final_iteration.converged,
            )

            return {
                "tests_generated": True,
                "iterations": len(test_iterations),
                "converged": final_iteration.converged,
                "all_passed": test_result.all_passed if test_result else False,
                "coverage": test_result.coverage_percent if test_result else None,
                "success_rate": test_result.success_rate if test_result else 0.0,
            }

        except Exception as e:
            logger.error("Test generation failed for %s: %s", step.step_id, e)
            return {
                "tests_generated": False,
                "reason": "exception",
                "error": str(e),
            }

    def _extract_code_from_output(self, text: str) -> str | None:
        """Extract code from step output."""
        # Look for code blocks
        code_blocks = re.findall(r"```(?:python|py)?\n(.*?)```", text, re.DOTALL)

        if code_blocks:
            # Return the largest code block
            return max(code_blocks, key=len).strip()

        # Heuristic: if output looks like code (has def/class/import), use it
        if any(keyword in text for keyword in ["def ", "class ", "import ", "from "]):
            return text.strip()

        return None

    def _detect_language(self, code: str, step: StepDef) -> str:
        """Detect programming language from code."""
        # Simple heuristics
        if "def " in code or "import " in code or "class " in code:
            return "python"

        if "function " in code or "const " in code or "let " in code:
            return "javascript"

        if "func " in code or "package " in code:
            return "go"

        # Fallback to checking target files extension
        if step.extra:
            target_files = step.extra.get("target_files", [])
            if target_files:
                first_file = target_files[0]
                if first_file.endswith(".py"):
                    return "python"
                elif first_file.endswith((".js", ".ts", ".jsx", ".tsx")):
                    return "javascript"
                elif first_file.endswith(".go"):
                    return "go"

        return "unknown"

    def get_session_summary(self) -> dict[str, Any]:
        """Get summary of all tests generated this session.

        Returns:
            Dict with test generation statistics
        """
        total_steps = len(self.test_results)
        if total_steps == 0:
            return {
                "total_steps_tested": 0,
                "converged": 0,
                "not_converged": 0,
                "avg_iterations": 0.0,
            }

        converged = sum(
            1 for r in self.test_results.values() if r.get("converged", False)
        )
        total_iterations = sum(
            r.get("iterations", 0) for r in self.test_results.values()
        )
        avg_iterations = total_iterations / total_steps if total_steps else 0.0

        return {
            "total_steps_tested": total_steps,
            "converged": converged,
            "not_converged": total_steps - converged,
            "avg_iterations": avg_iterations,
            "step_details": self.test_results,
        }
