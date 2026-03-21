"""Code generation benchmarks (HumanEval-style)."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any, Callable

from multihead.benchmarking.base import Benchmark, BenchmarkResult

logger = logging.getLogger(__name__)


class HumanEvalBenchmark(Benchmark):
    """HumanEval-style code generation benchmark.

    Tests ability to generate correct Python functions from docstrings.
    This is a simplified version with sample problems. For production,
    use the full HumanEval dataset (164 problems).
    """

    # Sample HumanEval-style problems
    SAMPLE_PROBLEMS = [
        {
            "prompt": (
                "def add(a: int, b: int) -> int:\n"
                '    """Return the sum of a and b."""\n'
            ),
            "test_cases": [
                {"input": (1, 2), "expected": 3},
                {"input": (0, 0), "expected": 0},
                {"input": (-1, 1), "expected": 0},
                {"input": (100, 200), "expected": 300},
            ],
            "canonical": "return a + b",
        },
        {
            "prompt": (
                "def max_of_three(a: int, b: int, c: int) -> int:\n"
                '    """Return the maximum of three integers."""\n'
            ),
            "test_cases": [
                {"input": (1, 2, 3), "expected": 3},
                {"input": (3, 2, 1), "expected": 3},
                {"input": (5, 5, 5), "expected": 5},
                {"input": (-1, -2, -3), "expected": -1},
            ],
            "canonical": "return max(a, b, c)",
        },
        {
            "prompt": (
                "def is_palindrome(s: str) -> bool:\n"
                '    """Check if a string is a palindrome."""\n'
            ),
            "test_cases": [
                {"input": ("racecar",), "expected": True},
                {"input": ("hello",), "expected": False},
                {"input": ("",), "expected": True},
                {"input": ("a",), "expected": True},
            ],
            "canonical": "return s == s[::-1]",
        },
        {
            "prompt": (
                "def factorial(n: int) -> int:\n"
                '    """Return the factorial of n. n >= 0."""\n'
            ),
            "test_cases": [
                {"input": (0,), "expected": 1},
                {"input": (1,), "expected": 1},
                {"input": (5,), "expected": 120},
                {"input": (3,), "expected": 6},
            ],
            "canonical": "return 1 if n <= 1 else n * factorial(n - 1)",
        },
        {
            "prompt": (
                "def fibonacci(n: int) -> int:\n"
                '    """Return the nth Fibonacci number (0-indexed)."""\n'
            ),
            "test_cases": [
                {"input": (0,), "expected": 0},
                {"input": (1,), "expected": 1},
                {"input": (5,), "expected": 5},
                {"input": (10,), "expected": 55},
            ],
            "canonical": "a, b = 0, 1\nfor _ in range(n):\n    a, b = b, a + b\nreturn a",
        },
    ]

    def __init__(self):
        """Initialize HumanEval benchmark."""
        super().__init__(
            name="humaneval",
            solver_types=["llm"],
        )

    async def run(
        self,
        solver_id: str,
        generate_func: Callable,
        *,
        sample_limit: int | None = None,
        timeout_seconds: float = 300.0,
    ) -> BenchmarkResult:
        """Run HumanEval benchmark.

        For each problem:
        1. Send the function signature + docstring
        2. Extract the generated code body
        3. Execute against test cases
        4. Score pass@1 (correct on first try)

        Args:
            solver_id: Solver identifier
            generate_func: Function to generate responses
            sample_limit: Maximum problems to test
            timeout_seconds: Total timeout

        Returns:
            BenchmarkResult with pass@1 score
        """
        started_at = datetime.now(timezone.utc)
        problems = self.SAMPLE_PROBLEMS[:sample_limit] if sample_limit else self.SAMPLE_PROBLEMS

        passed = 0
        total = 0
        errors = 0

        try:
            async with asyncio.timeout(timeout_seconds):
                for problem in problems:
                    total += 1
                    try:
                        result = await self._evaluate_problem(problem, generate_func)
                        if result:
                            passed += 1
                    except Exception as e:
                        logger.warning("HumanEval problem failed for %s: %s", solver_id, e)
                        errors += 1

        except asyncio.TimeoutError:
            logger.warning("HumanEval timed out for %s after %ds", solver_id, timeout_seconds)

        score = passed / total if total > 0 else 0.0
        completed_at = datetime.now(timezone.utc)
        runtime = (completed_at - started_at).total_seconds()

        return BenchmarkResult(
            benchmark_name=self.name,
            solver_id=solver_id,
            solver_type="llm",
            score=score,
            metrics={
                "passed": passed,
                "total": total,
                "pass_at_1": score,
            },
            runtime_seconds=runtime,
            sample_count=total,
            error_count=errors,
            started_at=started_at,
            completed_at=completed_at,
        )

    async def _evaluate_problem(
        self,
        problem: dict[str, Any],
        generate_func: Callable,
    ) -> bool:
        """Evaluate a single HumanEval problem.

        Args:
            problem: Problem dict with prompt, test_cases, canonical
            generate_func: Function to generate code

        Returns:
            True if all test cases pass
        """
        prompt = (
            f"Complete the following Python function. Only output the function body, no explanation.\n\n"
            f"{problem['prompt']}"
        )

        response = await generate_func(prompt)
        if isinstance(response, dict):
            code_text = response.get("text", "")
        else:
            code_text = str(response)

        # Extract the function body
        func_body = self._extract_function_body(code_text, problem["prompt"])
        if not func_body:
            return False

        # Build complete function
        func_name = problem["prompt"].split("(")[0].replace("def ", "").strip()
        full_code = problem["prompt"] + "    " + func_body.replace("\n", "\n    ")

        # Execute test cases safely
        return self._run_test_cases(full_code, func_name, problem["test_cases"])

    def _extract_function_body(self, response: str, prompt: str) -> str:
        """Extract function body from LLM response.

        Args:
            response: LLM response text
            prompt: Original prompt (function signature)

        Returns:
            Extracted function body or empty string
        """
        # Try to find code between backticks
        code_match = re.search(r'```(?:python)?\s*\n?(.*?)```', response, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            # If the code includes the function def, extract just the body
            if "def " in code:
                lines = code.split("\n")
                body_lines = []
                in_body = False
                for line in lines:
                    if line.strip().startswith("def "):
                        in_body = True
                        continue
                    if line.strip().startswith('"""') and in_body:
                        # Skip docstring
                        continue
                    if in_body:
                        body_lines.append(line.strip())
                return "\n".join(body_lines)
            return code

        # Try to extract return statement or direct code
        lines = response.strip().split("\n")
        code_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip explanatory text
            if stripped and not stripped.startswith("#") and not stripped.startswith("//"):
                if any(kw in stripped for kw in ("return", "if", "for", "while", "=", "print")):
                    code_lines.append(stripped)

        return "\n".join(code_lines) if code_lines else response.strip()

    def _run_test_cases(
        self,
        full_code: str,
        func_name: str,
        test_cases: list[dict[str, Any]],
    ) -> bool:
        """Execute test cases against generated code.

        Args:
            full_code: Complete function code
            func_name: Function name to call
            test_cases: List of {input, expected} dicts

        Returns:
            True if all test cases pass
        """
        # Execute in isolated namespace
        namespace: dict[str, Any] = {}
        try:
            exec(full_code, namespace)  # noqa: S102
        except Exception:
            return False

        func = namespace.get(func_name)
        if not func or not callable(func):
            return False

        # Run test cases
        for test in test_cases:
            try:
                result = func(*test["input"])
                if result != test["expected"]:
                    return False
            except Exception:
                return False

        return True
