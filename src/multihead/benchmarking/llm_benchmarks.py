"""LLM benchmarks (MMLU, GSM8K, reasoning)."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any, Callable

from multihead.benchmarking.base import Benchmark, BenchmarkResult

logger = logging.getLogger(__name__)


class SimpleReasoningBenchmark(Benchmark):
    """Simple reasoning benchmark with basic logic questions.

    This is a lightweight benchmark for quick evaluation.
    For production, use full MMLU/GSM8K datasets.
    """

    # Sample reasoning questions (answer is in parentheses)
    QUESTIONS = [
        ("If all cats are mammals, and all mammals have hearts, do all cats have hearts?", "yes"),
        ("Is 15 greater than 20?", "no"),
        ("If it's raining, the ground is wet. The ground is wet. Is it raining?", "maybe"),
        ("2 + 2 = ?", "4"),
        ("What is the capital of France?", "paris"),
        ("Is a square a rectangle?", "yes"),
        ("If A > B and B > C, is A > C?", "yes"),
        ("Is water H2O?", "yes"),
        ("10 / 2 = ?", "5"),
        ("Is the sky green?", "no"),
    ]

    def __init__(self):
        """Initialize simple reasoning benchmark."""
        super().__init__(
            name="simple_reasoning",
            solver_types=["llm", "vlm"],
        )

    async def run(
        self,
        solver_id: str,
        generate_func: Callable,
        *,
        sample_limit: int | None = None,
        timeout_seconds: float = 300.0,
    ) -> BenchmarkResult:
        """Run reasoning benchmark.

        Args:
            solver_id: Solver identifier
            generate_func: Function to generate responses
            sample_limit: Maximum questions to test
            timeout_seconds: Total timeout

        Returns:
            BenchmarkResult with accuracy score
        """
        started_at = datetime.now(timezone.utc)
        questions = self.QUESTIONS[:sample_limit] if sample_limit else self.QUESTIONS

        correct = 0
        total = 0
        errors = 0

        try:
            async with asyncio.timeout(timeout_seconds):
                for question, expected_answer in questions:
                    total += 1
                    try:
                        # Generate response
                        response = await generate_func(question)
                        if isinstance(response, dict):
                            response_text = response.get("text", "")
                        else:
                            response_text = str(response)

                        # Check if answer is correct (case-insensitive substring match)
                        response_lower = response_text.lower()
                        expected_lower = expected_answer.lower()

                        if expected_lower in response_lower:
                            correct += 1

                    except Exception as e:
                        logger.warning("Question failed for %s: %s", solver_id, e)
                        errors += 1

        except asyncio.TimeoutError:
            logger.warning("Benchmark timed out for %s after %ds", solver_id, timeout_seconds)

        score = correct / total if total > 0 else 0.0
        completed_at = datetime.now(timezone.utc)
        runtime = (completed_at - started_at).total_seconds()

        return BenchmarkResult(
            benchmark_name=self.name,
            solver_id=solver_id,
            solver_type="llm",
            score=score,
            metrics={
                "correct": correct,
                "total": total,
                "accuracy": score,
            },
            runtime_seconds=runtime,
            sample_count=total,
            error_count=errors,
            started_at=started_at,
            completed_at=completed_at,
        )


class MMLUBenchmark(Benchmark):
    """MMLU (Massive Multitask Language Understanding) benchmark.

    This is a simplified version. For production, use the full MMLU dataset
    with 57 subjects and 14k questions.
    """

    # Sample MMLU-style questions (multiple choice)
    SAMPLE_QUESTIONS = [
        {
            "question": "What is the powerhouse of the cell?",
            "choices": ["Nucleus", "Mitochondria", "Ribosome", "Golgi apparatus"],
            "answer": 1,  # Index of correct answer
        },
        {
            "question": "Which planet is closest to the Sun?",
            "choices": ["Venus", "Earth", "Mercury", "Mars"],
            "answer": 2,
        },
        {
            "question": "What is 7 * 8?",
            "choices": ["54", "56", "58", "60"],
            "answer": 1,
        },
    ]

    def __init__(self):
        """Initialize MMLU benchmark."""
        super().__init__(
            name="mmlu",
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
        """Run MMLU benchmark.

        Args:
            solver_id: Solver identifier
            generate_func: Function to generate responses
            sample_limit: Maximum questions to test
            timeout_seconds: Total timeout

        Returns:
            BenchmarkResult with accuracy score
        """
        started_at = datetime.now(timezone.utc)
        questions = self.SAMPLE_QUESTIONS[:sample_limit] if sample_limit else self.SAMPLE_QUESTIONS

        correct = 0
        total = 0
        errors = 0

        try:
            async with asyncio.timeout(timeout_seconds):
                for q in questions:
                    total += 1
                    try:
                        # Format question with choices
                        prompt = f"{q['question']}\nA. {q['choices'][0]}\nB. {q['choices'][1]}\nC. {q['choices'][2]}\nD. {q['choices'][3]}\nAnswer:"

                        # Generate response
                        response = await generate_func(prompt)
                        if isinstance(response, dict):
                            response_text = response.get("text", "")
                        else:
                            response_text = str(response)

                        # Extract answer (A, B, C, or D)
                        answer_match = re.search(r'\b([ABCD])\b', response_text.upper())
                        if answer_match:
                            answer_letter = answer_match.group(1)
                            answer_index = ord(answer_letter) - ord('A')

                            if answer_index == q['answer']:
                                correct += 1

                    except Exception as e:
                        logger.warning("MMLU question failed for %s: %s", solver_id, e)
                        errors += 1

        except asyncio.TimeoutError:
            logger.warning("MMLU timed out for %s after %ds", solver_id, timeout_seconds)

        score = correct / total if total > 0 else 0.0
        completed_at = datetime.now(timezone.utc)
        runtime = (completed_at - started_at).total_seconds()

        return BenchmarkResult(
            benchmark_name=self.name,
            solver_id=solver_id,
            solver_type="llm",
            score=score,
            metrics={
                "correct": correct,
                "total": total,
                "accuracy": score,
            },
            runtime_seconds=runtime,
            sample_count=total,
            error_count=errors,
            started_at=started_at,
            completed_at=completed_at,
        )


class GSM8KBenchmark(Benchmark):
    """GSM8K (Grade School Math) benchmark.

    Tests mathematical reasoning with word problems.
    This is a simplified version with sample problems.
    """

    # Sample GSM8K-style problems
    SAMPLE_PROBLEMS = [
        {
            "question": "John has 5 apples. He buys 3 more. How many apples does he have?",
            "answer": 8,
        },
        {
            "question": "A rectangle has length 6 and width 4. What is its area?",
            "answer": 24,
        },
        {
            "question": "If a train travels 60 miles in 2 hours, what is its average speed in miles per hour?",
            "answer": 30,
        },
    ]

    def __init__(self):
        """Initialize GSM8K benchmark."""
        super().__init__(
            name="gsm8k",
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
        """Run GSM8K benchmark.

        Args:
            solver_id: Solver identifier
            generate_func: Function to generate responses
            sample_limit: Maximum problems to test
            timeout_seconds: Total timeout

        Returns:
            BenchmarkResult with accuracy score
        """
        started_at = datetime.now(timezone.utc)
        problems = self.SAMPLE_PROBLEMS[:sample_limit] if sample_limit else self.SAMPLE_PROBLEMS

        correct = 0
        total = 0
        errors = 0

        try:
            async with asyncio.timeout(timeout_seconds):
                for problem in problems:
                    total += 1
                    try:
                        # Generate response
                        prompt = f"{problem['question']}\nAnswer:"
                        response = await generate_func(prompt)

                        if isinstance(response, dict):
                            response_text = response.get("text", "")
                        else:
                            response_text = str(response)

                        # Extract number from response
                        numbers = re.findall(r'\d+', response_text)
                        if numbers:
                            # Take the last number mentioned (usually the final answer)
                            answer = int(numbers[-1])
                            if answer == problem['answer']:
                                correct += 1

                    except Exception as e:
                        logger.warning("GSM8K problem failed for %s: %s", solver_id, e)
                        errors += 1

        except asyncio.TimeoutError:
            logger.warning("GSM8K timed out for %s after %ds", solver_id, timeout_seconds)

        score = correct / total if total > 0 else 0.0
        completed_at = datetime.now(timezone.utc)
        runtime = (completed_at - started_at).total_seconds()

        return BenchmarkResult(
            benchmark_name=self.name,
            solver_id=solver_id,
            solver_type="llm",
            score=score,
            metrics={
                "correct": correct,
                "total": total,
                "accuracy": score,
            },
            runtime_seconds=runtime,
            sample_count=total,
            error_count=errors,
            started_at=started_at,
            completed_at=completed_at,
        )
