"""Tests for automatic test generation system."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from multihead.test_generation import (
    TestFramework,
    TestGenerator,
    TestResult,
)


class TestTestGenerator:
    """Test the test generation system."""

    @pytest.fixture
    def mock_head_manager(self):
        """Create mock HeadManager."""
        mgr = MagicMock()
        mgr.generate = AsyncMock(return_value={
            "text": """Here are comprehensive tests:
```python
import pytest

def test_example():
    assert 1 + 1 == 2

def test_edge_case():
    assert 0 + 0 == 0
```
"""
        })
        return mgr

    @pytest.fixture
    def test_generator(self, mock_head_manager):
        """Create TestGenerator."""
        return TestGenerator(
            head_manager=mock_head_manager,
            test_framework=TestFramework.PYTEST,
            max_iterations=3,
            min_coverage=0.8,
        )

    def test_extract_code_from_response(self, test_generator):
        """Should extract code from markdown code blocks."""
        response = """Here are the tests:
```python
def test_foo():
    assert True
```
More text here.
"""
        code = test_generator._extract_code_from_response(response)
        assert "def test_foo():" in code
        assert "assert True" in code

    def test_extract_code_multiple_blocks(self, test_generator):
        """Should return largest code block when multiple present."""
        response = """
```python
short = 1
```

Here's the main code:
```python
def test_longer_function():
    x = 1
    y = 2
    return x + y
```
"""
        code = test_generator._extract_code_from_response(response)
        assert "def test_longer_function" in code
        assert "short = 1" not in code  # Smaller block excluded

    @pytest.mark.asyncio
    async def test_generate_initial_tests(self, test_generator, mock_head_manager):
        """Should generate initial test suite."""
        code = "def add(a, b):\n    return a + b"

        test_code = await test_generator._generate_initial_tests(
            code=code,
            code_file_path=None,
            head_id="mock-llm",
            context=None,
        )

        # Verify LLM was called
        assert mock_head_manager.generate.called
        call_args = mock_head_manager.generate.call_args
        prompt = call_args[0][1]

        # Check prompt contains code
        assert "def add(a, b):" in prompt
        assert "pytest" in prompt.lower()

        # Check test code was extracted
        assert "def test_" in test_code

    @pytest.mark.asyncio
    async def test_refine_tests_with_failures(self, test_generator, mock_head_manager):
        """Should refine tests when given failure information."""
        code = "def divide(a, b):\n    return a / b"
        current_tests = "def test_divide():\n    assert divide(4, 2) == 2"

        # Mock failed execution
        failed_result = TestResult(
            framework=TestFramework.PYTEST,
            total_tests=2,
            passed=1,
            failed=1,
            errors=0,
            skipped=0,
            failure_messages=["ZeroDivisionError: division by zero"],
        )

        refined = await test_generator._refine_tests(
            original_code=code,
            current_tests=current_tests,
            prev_execution=failed_result,
            head_id="mock-llm",
            context=None,
        )

        # Verify refinement was called
        assert mock_head_manager.generate.called
        call_args = mock_head_manager.generate.call_args
        prompt = call_args[0][1]

        # Check prompt contains failure info
        assert "ZeroDivisionError" in prompt
        assert "1/2 passed" in prompt

    @pytest.mark.asyncio
    async def test_analyze_failures(self, test_generator, mock_head_manager):
        """Should analyze failures and provide suggestions."""
        # Mock response with suggestions
        mock_head_manager.generate.return_value = {
            "text": """Analysis of failures:
1. Add test for division by zero case
2. Check for None inputs
3. Validate return type is float
"""
        }

        suggestions = await test_generator._analyze_failures(
            failure_messages=["Test failed: division by zero"],
            code="def divide(a, b):\n    return a / b",
            tests="def test_divide():\n    assert divide(4, 2) == 2",
            head_id="mock-llm",
        )

        assert len(suggestions) > 0
        # Check suggestions were parsed
        assert any("division by zero" in s.lower() for s in suggestions)

    def test_test_result_properties(self):
        """Should correctly compute test result properties."""
        result = TestResult(
            framework=TestFramework.PYTEST,
            total_tests=10,
            passed=8,
            failed=2,
            errors=0,
            skipped=0,
        )

        assert result.all_passed is False
        assert result.success_rate == 0.8

        passing_result = TestResult(
            framework=TestFramework.PYTEST,
            total_tests=5,
            passed=5,
            failed=0,
            errors=0,
            skipped=0,
        )

        assert passing_result.all_passed is True
        assert passing_result.success_rate == 1.0
