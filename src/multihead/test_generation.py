"""Automatic test generation using kS-LLM iterative refinement pattern.

Implements the universal test generation loop:
1. Generate: LLM produces tests for code
2. Execute: Run tests and capture results
3. Analyze: Reflection on failures
4. Refine: Fix code or tests based on analysis
5. Iterate: Repeat until convergence or max attempts

Based on research findings:
- CoverUp: 50% of successful tests come from multi-turn refinement
- TestPilot: 5x improvement with multi-stage prompts
- TestART: Coverage-guided generation
"""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TestFramework(str, Enum):
    """Supported test frameworks."""
    PYTEST = "pytest"
    UNITTEST = "unittest"
    JEST = "jest"
    MOCHA = "mocha"
    GO_TEST = "go test"


class TestStatus(str, Enum):
    """Test execution status."""
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class TestResult:
    """Result from test execution."""
    framework: TestFramework
    total_tests: int
    passed: int
    failed: int
    errors: int
    skipped: int
    coverage_percent: float | None = None
    failure_messages: list[str] | None = None
    stdout: str = ""
    stderr: str = ""

    @property
    def all_passed(self) -> bool:
        """Check if all tests passed."""
        return self.failed == 0 and self.errors == 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0-1)."""
        if self.total_tests == 0:
            return 0.0
        return self.passed / self.total_tests


@dataclass
class TestGenerationResult:
    """Result from test generation iteration."""
    iteration: int
    test_code: str
    test_file_path: Path | None
    execution_result: TestResult | None
    refinement_needed: bool
    refinement_suggestions: list[str] | None = None
    converged: bool = False


class TestGenerator:
    """Automatic test generator using kS-LLM pattern.

    Generates and iteratively refines tests for code steps.
    """

    def __init__(
        self,
        head_manager: Any,
        test_framework: TestFramework = TestFramework.PYTEST,
        max_iterations: int = 5,
        min_coverage: float = 0.8,
        max_consecutive_no_improvement: int = 3,
    ):
        """Initialize test generator.

        Args:
            head_manager: HeadManager for LLM access
            test_framework: Test framework to use
            max_iterations: Maximum refinement iterations
            min_coverage: Minimum acceptable coverage (0-1)
            max_consecutive_no_improvement: Stop after N iterations without improvement
        """
        self.head_manager = head_manager
        self.test_framework = test_framework
        self.max_iterations = max_iterations
        self.min_coverage = min_coverage
        self.max_consecutive_no_improvement = max_consecutive_no_improvement

    async def generate_tests(
        self,
        code: str,
        code_file_path: Path | None = None,
        head_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[TestGenerationResult]:
        """Generate tests for code using iterative refinement.

        Args:
            code: Code to generate tests for
            code_file_path: Optional path to code file (for imports)
            head_id: Optional head to use for generation
            context: Additional context (requirements, dependencies, etc.)

        Returns:
            List of TestGenerationResult for each iteration
        """
        logger.info("Starting test generation for code (%d chars)", len(code))

        results: list[TestGenerationResult] = []
        current_test_code = ""
        consecutive_no_improvement = 0
        best_coverage = 0.0

        for iteration in range(1, self.max_iterations + 1):
            logger.info("Test generation iteration %d/%d", iteration, self.max_iterations)

            # Phase 1: Generate or refine tests
            if iteration == 1:
                # Initial generation
                test_code = await self._generate_initial_tests(
                    code, code_file_path, head_id, context
                )
            else:
                # Refinement based on previous failures
                prev_result = results[-1]
                test_code = await self._refine_tests(
                    code,
                    current_test_code,
                    prev_result.execution_result,
                    head_id,
                    context,
                )

            current_test_code = test_code

            # Phase 2: Execute tests
            test_file_path = await self._write_test_file(test_code, code_file_path)
            execution_result = await self._execute_tests(test_file_path, code_file_path)

            # Phase 3: Analyze results
            coverage = execution_result.coverage_percent or 0.0
            improved = coverage > best_coverage
            if improved:
                best_coverage = coverage
                consecutive_no_improvement = 0
            else:
                consecutive_no_improvement += 1

            # Check convergence
            converged = (
                execution_result.all_passed
                and (coverage >= self.min_coverage if coverage else True)
            )

            # Phase 4: Determine if refinement needed
            refinement_needed = not converged and iteration < self.max_iterations

            refinement_suggestions = []
            if refinement_needed and execution_result.failure_messages:
                refinement_suggestions = await self._analyze_failures(
                    execution_result.failure_messages,
                    code,
                    test_code,
                    head_id,
                )

            result = TestGenerationResult(
                iteration=iteration,
                test_code=test_code,
                test_file_path=test_file_path,
                execution_result=execution_result,
                refinement_needed=refinement_needed,
                refinement_suggestions=refinement_suggestions,
                converged=converged,
            )

            results.append(result)

            logger.info(
                "Iteration %d: %d/%d passed, coverage=%.1f%%, converged=%s",
                iteration,
                execution_result.passed,
                execution_result.total_tests,
                coverage,
                converged,
            )

            # Check stopping conditions
            if converged:
                logger.info("Tests converged - all tests passing with sufficient coverage")
                break

            if consecutive_no_improvement >= self.max_consecutive_no_improvement:
                logger.info(
                    "Stopping - no improvement for %d iterations",
                    consecutive_no_improvement,
                )
                break

        return results

    async def _generate_initial_tests(
        self,
        code: str,
        code_file_path: Path | None,
        head_id: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate initial test suite."""
        prompt_parts = [
            "Generate comprehensive tests for the following code:\n",
            f"```python\n{code}\n```\n",
            f"\nUse {self.test_framework.value} framework.",
            "\nGenerate tests that cover:",
            "- Normal cases (happy path)",
            "- Edge cases (boundary conditions)",
            "- Error cases (invalid inputs)",
            "- All public functions/methods",
            "\nWrite clear, well-documented tests with descriptive names.",
        ]

        if context:
            if "requirements" in context:
                prompt_parts.append(f"\n\nRequirements:\n{context['requirements']}")
            if "dependencies" in context:
                prompt_parts.append(f"\n\nDependencies:\n{context['dependencies']}")

        prompt = "\n".join(prompt_parts)

        response = await self.head_manager.generate(
            head_id or "qwen-llm",
            prompt,
        )

        test_code = self._extract_code_from_response(response.get("text", ""))
        return test_code

    async def _refine_tests(
        self,
        original_code: str,
        current_tests: str,
        prev_execution: TestResult | None,
        head_id: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Refine tests based on execution results."""
        if not prev_execution or prev_execution.all_passed:
            return current_tests

        prompt_parts = [
            "The following tests have failures. Please fix them.\n",
            f"\nOriginal code:\n```python\n{original_code}\n```\n",
            f"\nCurrent tests:\n```python\n{current_tests}\n```\n",
            f"\nTest results: {prev_execution.passed}/{prev_execution.total_tests} passed\n",
        ]

        if prev_execution.failure_messages:
            prompt_parts.append("\nFailures:")
            for msg in prev_execution.failure_messages[:5]:  # Limit to top 5
                prompt_parts.append(f"  - {msg}")

        if prev_execution.coverage_percent:
            prompt_parts.append(
                f"\nCurrent coverage: {prev_execution.coverage_percent:.1f}%"
            )
            if prev_execution.coverage_percent < self.min_coverage * 100:
                prompt_parts.append(
                    f"Target coverage: {self.min_coverage * 100:.1f}%"
                )
                prompt_parts.append("\nAdd tests to cover missing code paths.")

        prompt_parts.append(
            "\n\nProvide the complete refined test suite addressing these issues."
        )

        prompt = "\n".join(prompt_parts)

        response = await self.head_manager.generate(
            head_id or "qwen-llm",
            prompt,
        )

        test_code = self._extract_code_from_response(response.get("text", ""))
        return test_code

    async def _analyze_failures(
        self,
        failure_messages: list[str],
        code: str,
        tests: str,
        head_id: str | None,
    ) -> list[str]:
        """Analyze test failures using Reflection to suggest fixes."""
        prompt_parts = [
            "Analyze these test failures and suggest specific fixes:\n",
            f"\nCode under test:\n```python\n{code}\n```\n",
            f"\nTest code:\n```python\n{tests}\n```\n",
            "\nFailures:",
        ]

        for msg in failure_messages[:5]:
            prompt_parts.append(f"  - {msg}")

        prompt_parts.append(
            "\n\nFor each failure, provide a specific, actionable fix."
        )

        prompt = "\n".join(prompt_parts)

        response = await self.head_manager.generate(
            head_id or "qwen-llm",
            prompt,
        )

        # Extract suggestions from response
        text = response.get("text", "")
        suggestions = []

        # Simple parsing - look for numbered or bulleted lists
        for line in text.split("\n"):
            line = line.strip()
            if re.match(r"^\d+\.", line) or line.startswith("- "):
                suggestion = re.sub(r"^\d+\.\s*|-\s*", "", line)
                if suggestion:
                    suggestions.append(suggestion)

        return suggestions or [text]  # Fallback to full text if no structure found

    async def _write_test_file(
        self,
        test_code: str,
        code_file_path: Path | None,
    ) -> Path:
        """Write test code to temporary file."""
        import tempfile

        if code_file_path:
            test_dir = code_file_path.parent / "tests"
            test_dir.mkdir(exist_ok=True)
            test_file = test_dir / f"test_{code_file_path.stem}.py"
        else:
            # Use temp file
            fd, path = tempfile.mkstemp(suffix="_test.py", prefix="multihead_")
            test_file = Path(path)

        test_file.write_text(test_code)
        logger.debug("Wrote test file: %s", test_file)

        return test_file

    async def _execute_tests(
        self,
        test_file_path: Path,
        code_file_path: Path | None,
    ) -> TestResult:
        """Execute tests and capture results."""
        if self.test_framework == TestFramework.PYTEST:
            return await self._execute_pytest(test_file_path, code_file_path)
        elif self.test_framework == TestFramework.UNITTEST:
            return await self._execute_unittest(test_file_path)
        else:
            raise NotImplementedError(
                f"Test framework {self.test_framework} not implemented"
            )

    async def _execute_pytest(
        self,
        test_file_path: Path,
        code_file_path: Path | None,
    ) -> TestResult:
        """Execute pytest and parse results."""
        try:
            # Run pytest with coverage if possible
            cmd = [
                "pytest",
                str(test_file_path),
                "-v",
                "--tb=short",
            ]

            # Add coverage if code file provided
            if code_file_path:
                cmd.extend([
                    f"--cov={code_file_path.stem}",
                    "--cov-report=term-missing",
                ])

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(test_file_path.parent),
            )

            stdout_bytes, stderr_bytes = await proc.communicate()
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")

            # Parse pytest output
            passed, failed, errors, skipped = 0, 0, 0, 0
            coverage_percent = None
            failure_messages = []

            for line in stdout.split("\n"):
                # Parse summary line (e.g., "5 passed, 2 failed, 1 error in 0.5s")
                if " passed" in line or " failed" in line:
                    if match := re.search(r"(\d+)\s+passed", line):
                        passed = int(match.group(1))
                    if match := re.search(r"(\d+)\s+failed", line):
                        failed = int(match.group(1))
                    if match := re.search(r"(\d+)\s+error", line):
                        errors = int(match.group(1))
                    if match := re.search(r"(\d+)\s+skipped", line):
                        skipped = int(match.group(1))

                # Parse coverage
                if "TOTAL" in line and "%" in line:
                    if match := re.search(r"(\d+)%", line):
                        coverage_percent = float(match.group(1))

                # Capture failure messages
                if "FAILED" in line or "ERROR" in line:
                    failure_messages.append(line.strip())

            total_tests = passed + failed + errors + skipped

            return TestResult(
                framework=TestFramework.PYTEST,
                total_tests=total_tests,
                passed=passed,
                failed=failed,
                errors=errors,
                skipped=skipped,
                coverage_percent=coverage_percent,
                failure_messages=failure_messages if failure_messages else None,
                stdout=stdout,
                stderr=stderr,
            )

        except Exception as e:
            logger.error("Failed to execute pytest: %s", e)
            return TestResult(
                framework=TestFramework.PYTEST,
                total_tests=0,
                passed=0,
                failed=0,
                errors=1,
                skipped=0,
                failure_messages=[str(e)],
                stderr=str(e),
            )

    async def _execute_unittest(self, test_file_path: Path) -> TestResult:
        """Execute unittest and parse results."""
        # Implementation similar to pytest
        raise NotImplementedError("unittest execution not yet implemented")

    def _extract_code_from_response(self, text: str) -> str:
        """Extract code block from LLM response."""
        # Look for code blocks
        code_blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)

        if code_blocks:
            # Return the largest code block (usually the full test suite)
            return max(code_blocks, key=len).strip()

        # Fallback: return whole text if no code blocks found
        return text.strip()
