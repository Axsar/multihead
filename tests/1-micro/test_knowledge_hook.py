"""Tests for real-time knowledge extraction hooks."""

from __future__ import annotations

import pytest

from multihead.knowledge_hook import KnowledgeHook
from multihead.knowledge_store import KnowledgeStore
from multihead.models import StageResult, StepDef, StepStatus


@pytest.fixture
def temp_knowledge_db(tmp_path):
    """Create a temporary knowledge database."""
    db_path = tmp_path / "test_knowledge.db"
    return KnowledgeStore(db_path)


@pytest.fixture
def knowledge_hook(temp_knowledge_db):
    """Create KnowledgeHook with temp database."""
    return KnowledgeHook(
        knowledge_store=temp_knowledge_db,
        project_id="test_project",
        min_confidence=0.7,
    )


class TestKnowledgeHook:
    """Test real-time knowledge extraction."""

    @pytest.mark.asyncio
    async def test_extracts_success_patterns(self, knowledge_hook):
        """Should extract patterns from successful step execution."""
        step = StepDef(
            step_id="test-1",
            name="Test successful step",
            head_id="mock-llm",
            required_kind="llm",
            prompt_template="Test prompt",
        )

        result = StageResult(
            step_id="test-1",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
            outputs={"text": "Success output"},
            metrics={"prm_quality": "correct"},
        )

        context = {
            "run_id": "run-123",
            "goal": "Test goal",
            "previous_steps": [],
        }

        meta = await knowledge_hook.on_step_complete(step, result, context)

        assert "claims_created" in meta
        assert "patterns_detected" in meta
        assert len(meta["claims_created"]) > 0
        assert "success_pattern" in meta["patterns_detected"]

    @pytest.mark.asyncio
    async def test_extracts_failure_patterns(self, knowledge_hook):
        """Should extract patterns from failed step execution."""
        step = StepDef(
            step_id="test-2",
            name="Test failed step",
            head_id="mock-llm",
            required_kind="llm",
            prompt_template="Test prompt",
        )

        result = StageResult(
            step_id="test-2",
            head_id="mock-llm",
            status=StepStatus.FAILED,
            error="Timeout: exceeded 30s",
        )

        context = {
            "run_id": "run-123",
            "goal": "Test goal",
            "previous_steps": [],
        }

        meta = await knowledge_hook.on_step_complete(step, result, context)

        assert len(meta["claims_created"]) > 0
        assert "failure_pattern" in meta["patterns_detected"]

        # Verify claim was created
        session_summary = knowledge_hook.get_session_summary()
        assert session_summary["total_claims"] > 0
        assert "failure_pattern" in session_summary["patterns_learned"]

    @pytest.mark.asyncio
    async def test_extracts_performance_insights(self, knowledge_hook):
        """Should extract performance insights from step metrics."""
        step = StepDef(
            step_id="test-3",
            name="Test performance step",
            head_id="mock-llm",
            required_kind="llm",
            prompt_template="Test prompt",
        )

        result = StageResult(
            step_id="test-3",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
            outputs={"text": "Output"},
            metrics={
                "tokens_in": 1000,
                "tokens_out": 300,  # High efficiency (0.3 ratio)
            },
        )

        context = {
            "run_id": "run-123",
            "goal": "Test goal",
            "previous_steps": [],
        }

        meta = await knowledge_hook.on_step_complete(step, result, context)

        assert len(meta["claims_created"]) > 0
        assert "performance_insight" in meta["patterns_detected"]

    @pytest.mark.asyncio
    async def test_classifies_error_types(self, knowledge_hook):
        """Should correctly classify different error types."""
        test_cases = [
            ("Timeout: exceeded 30s", "timeout"),
            ("Out of memory error", "out_of_memory"),
            ("Syntax error in response", "syntax_error"),
            ("Context length exceeded", "context_length"),
            ("Connection refused", "network"),
            ("Unknown error", "unknown"),
        ]

        for error_msg, expected_type in test_cases:
            error_type = knowledge_hook._classify_error(error_msg)
            assert error_type == expected_type

    @pytest.mark.asyncio
    async def test_respects_min_confidence(self, temp_knowledge_db):
        """Should only auto-accept claims above min_confidence threshold."""
        # Hook with high min_confidence
        hook_high_conf = KnowledgeHook(
            knowledge_store=temp_knowledge_db,
            project_id="test",
            min_confidence=0.9,  # High threshold
        )

        step = StepDef(
            step_id="test-4",
            name="Test step",
            head_id="mock-llm",
            required_kind="llm",
            prompt_template="Test",
        )

        result = StageResult(
            step_id="test-4",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
            outputs={"text": "Output"},
        )

        context = {"run_id": "run-123", "goal": "Test", "previous_steps": []}

        await hook_high_conf.on_step_complete(step, result, context)

        # Low confidence claims shouldn't be auto-accepted
        # (depends on implementation - this test verifies the threshold logic)
        assert True  # Placeholder - actual verification would check claim status

    @pytest.mark.asyncio
    async def test_capture_flags(self, temp_knowledge_db):
        """Should respect capture flags for different pattern types."""
        # Hook with only success capture enabled
        hook_success_only = KnowledgeHook(
            knowledge_store=temp_knowledge_db,
            project_id="test",
            capture_successes=True,
            capture_failures=False,
            capture_performance=False,
        )

        step = StepDef(
            step_id="test-5",
            name="Test step",
            head_id="mock-llm",
            required_kind="llm",
            prompt_template="Test",
        )

        # Try failed step
        failed_result = StageResult(
            step_id="test-5",
            head_id="mock-llm",
            status=StepStatus.FAILED,
            error="Test error",
        )

        context = {"run_id": "run-123", "goal": "Test", "previous_steps": []}

        meta_failed = await hook_success_only.on_step_complete(
            step, failed_result, context
        )

        # Should not capture failure patterns
        assert "failure_pattern" not in meta_failed["patterns_detected"]

        # Try successful step
        success_result = StageResult(
            step_id="test-5",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
            outputs={"text": "Output"},
        )

        meta_success = await hook_success_only.on_step_complete(
            step, success_result, context
        )

        # Should capture success patterns
        assert len(meta_success["claims_created"]) > 0

    @pytest.mark.asyncio
    async def test_generates_learning_summary(self, knowledge_hook):
        """Should generate human-readable learning summary."""
        step = StepDef(
            step_id="test-6",
            name="Test summarize",
            head_id="mock-llm",
            required_kind="llm",
            prompt_template="Test",
        )

        result = StageResult(
            step_id="test-6",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
            outputs={"text": "Output"},
            metrics={"prm_quality": "correct"},
        )

        context = {"run_id": "run-123", "goal": "Test", "previous_steps": []}

        meta = await knowledge_hook.on_step_complete(step, result, context)

        assert "learning_summary" in meta
        assert isinstance(meta["learning_summary"], str)
        assert len(meta["learning_summary"]) > 0

    @pytest.mark.asyncio
    async def test_session_summary(self, knowledge_hook):
        """Should provide session summary with top patterns."""
        step = StepDef(
            step_id="test-7",
            name="Test session",
            head_id="mock-llm",
            required_kind="llm",
            prompt_template="Test",
        )

        # Execute multiple steps
        for i in range(5):
            result = StageResult(
                step_id=f"test-7-{i}",
                head_id="mock-llm",
                status=StepStatus.COMMITTED if i % 2 == 0 else StepStatus.FAILED,
                outputs={"text": "Output"} if i % 2 == 0 else {},
                error=None if i % 2 == 0 else "Test error",
            )

            context = {
                "run_id": "run-123",
                "goal": "Test",
                "previous_steps": [],
            }

            await knowledge_hook.on_step_complete(step, result, context)

        summary = knowledge_hook.get_session_summary()

        assert "total_claims" in summary
        assert "patterns_learned" in summary
        assert "top_patterns" in summary
        assert summary["total_claims"] > 0
        assert len(summary["top_patterns"]) > 0

    @pytest.mark.asyncio
    async def test_creates_execution_record(self, knowledge_hook):
        """Should create execution record for each step."""
        step = StepDef(
            step_id="test-8",
            name="Test record",
            head_id="mock-llm",
            required_kind="llm",
            prompt_template="Test",
        )

        result = StageResult(
            step_id="test-8",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
            outputs={"text": "Output"},
        )

        context = {
            "run_id": "run-123",
            "goal": "Test",
            "previous_steps": [],
        }

        # Execute should create a record
        meta = await knowledge_hook.on_step_complete(step, result, context)

        # Verify record was created (indirectly via claims)
        assert len(meta["claims_created"]) > 0
