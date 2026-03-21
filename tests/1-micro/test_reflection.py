"""Tests for reflection engine (Actor-Evaluator-Reflect-Memory cycle)."""

from __future__ import annotations

import pytest

from multihead.artifact_store import ArtifactStore
from multihead.event_store import EventStore
from multihead.head_manager import HeadManager
from multihead.models import (
    AdapterKind,
    EventKind,
    HeadManifest,
    StageResult,
    StepDef,
    StepStatus,
    WorkOrder,
)
from multihead.orchestrator import Orchestrator
from multihead.reflection import (
    ConfidenceEvaluator,
    CompositeEvaluator,
    ErrorEvaluator,
    ReflectionEngine,
    ReflectionMemory,
    ReflectionResult,
)


@pytest.fixture
def orchestrator(tmp_path):
    """Create orchestrator with mock head."""
    manifests = {
        "mock-llm": HeadManifest(
            head_id="mock-llm",
            name="Mock LLM",
            adapter=AdapterKind.MOCK,
            model="mock-v1",
            kind="llm",
        ),
    }
    hm = HeadManager(manifests)
    art = ArtifactStore(tmp_path / "artifacts", tmp_path / "artifacts.db")
    es = EventStore(tmp_path / "runs", tmp_path / "state.db")
    return Orchestrator(es, art, hm, tmp_path / "runs")


class TestConfidenceEvaluator:
    """Test confidence-based evaluator."""

    @pytest.mark.asyncio
    async def test_passes_high_confidence(self):
        """High confidence should pass evaluation."""
        evaluator = ConfidenceEvaluator(confidence_threshold=0.7)

        step = StepDef(step_id="s1", name="Test", head_id="mock-llm", prompt_template="Test")
        result = StageResult(
            step_id="s1",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
            outputs={"confidence": 0.85},
        )

        reflection = await evaluator.evaluate(step, result, {})

        assert reflection.passed is True
        assert reflection.quality_score == 0.85
        assert reflection.should_retry is False

    @pytest.mark.asyncio
    async def test_fails_low_confidence(self):
        """Low confidence should fail evaluation and suggest retry."""
        evaluator = ConfidenceEvaluator(confidence_threshold=0.7)

        step = StepDef(step_id="s1", name="Test", head_id="mock-llm", prompt_template="Test")
        result = StageResult(
            step_id="s1",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
            outputs={"confidence": 0.5},
        )

        reflection = await evaluator.evaluate(step, result, {})

        assert reflection.passed is False
        assert reflection.quality_score == 0.5
        assert reflection.should_retry is True
        assert "below threshold" in reflection.reflection_text.lower()
        assert len(reflection.suggested_changes) > 0

    @pytest.mark.asyncio
    async def test_checks_metrics_fallback(self):
        """Should check metrics if confidence not in outputs."""
        evaluator = ConfidenceEvaluator(confidence_threshold=0.7)

        step = StepDef(step_id="s1", name="Test", head_id="mock-llm", prompt_template="Test")
        result = StageResult(
            step_id="s1",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
            outputs={},
            metrics={"confidence": 0.9},
        )

        reflection = await evaluator.evaluate(step, result, {})

        assert reflection.passed is True
        assert reflection.quality_score == 0.9


class TestErrorEvaluator:
    """Test error-based evaluator."""

    @pytest.mark.asyncio
    async def test_passes_successful_step(self):
        """Successful step should pass."""
        evaluator = ErrorEvaluator()

        step = StepDef(step_id="s1", name="Test", head_id="mock-llm", prompt_template="Test")
        result = StageResult(
            step_id="s1",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
        )

        reflection = await evaluator.evaluate(step, result, {})

        assert reflection.passed is True
        assert reflection.quality_score == 1.0
        assert reflection.should_retry is False

    @pytest.mark.asyncio
    async def test_fails_validation_error(self):
        """Validation error should trigger reflection."""
        evaluator = ErrorEvaluator()

        step = StepDef(step_id="s1", name="Test", head_id="mock-llm", prompt_template="Test")
        result = StageResult(
            step_id="s1",
            head_id="mock-llm",
            status=StepStatus.FAILED,
            error="Validation failed: output does not match schema",
        )

        reflection = await evaluator.evaluate(step, result, {})

        assert reflection.passed is False
        assert reflection.quality_score == 0.0
        assert reflection.should_retry is True
        assert "validation" in reflection.reflection_text.lower()
        assert reflection.suggested_changes.get("add_validation_context") is True

    @pytest.mark.asyncio
    async def test_fails_timeout_error(self):
        """Timeout error should suggest simplification."""
        evaluator = ErrorEvaluator()

        step = StepDef(step_id="s1", name="Test", head_id="mock-llm", prompt_template="Test")
        result = StageResult(
            step_id="s1",
            head_id="mock-llm",
            status=StepStatus.FAILED,
            error="Execution timeout after 30s",
        )

        reflection = await evaluator.evaluate(step, result, {})

        assert reflection.passed is False
        assert "timeout" in reflection.reflection_text.lower()
        assert reflection.suggested_changes.get("simplify_prompt") is True

    @pytest.mark.asyncio
    async def test_fails_format_error(self):
        """Format/JSON error should suggest schema."""
        evaluator = ErrorEvaluator()

        step = StepDef(step_id="s1", name="Test", head_id="mock-llm", prompt_template="Test")
        result = StageResult(
            step_id="s1",
            head_id="mock-llm",
            status=StepStatus.FAILED,
            error="Failed to parse JSON output",
        )

        reflection = await evaluator.evaluate(step, result, {})

        assert reflection.passed is False
        assert "format" in reflection.reflection_text.lower()
        assert reflection.suggested_changes.get("add_schema") is True


class TestCompositeEvaluator:
    """Test composite evaluator combining multiple evaluators."""

    @pytest.mark.asyncio
    async def test_all_must_pass(self):
        """Composite passes only if all sub-evaluators pass."""
        evaluator = CompositeEvaluator([
            ConfidenceEvaluator(confidence_threshold=0.7),
            ErrorEvaluator(),
        ])

        step = StepDef(step_id="s1", name="Test", head_id="mock-llm", prompt_template="Test")

        # Case 1: Both pass
        result = StageResult(
            step_id="s1",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
            outputs={"confidence": 0.9},
        )
        reflection = await evaluator.evaluate(step, result, {})
        assert reflection.passed is True

        # Case 2: One fails
        result = StageResult(
            step_id="s1",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
            outputs={"confidence": 0.5},  # Below threshold
        )
        reflection = await evaluator.evaluate(step, result, {})
        assert reflection.passed is False

    @pytest.mark.asyncio
    async def test_combines_reflections(self):
        """Should combine reflection texts from failed evaluators."""
        evaluator = CompositeEvaluator([
            ConfidenceEvaluator(confidence_threshold=0.7),
            ErrorEvaluator(),
        ])

        step = StepDef(step_id="s1", name="Test", head_id="mock-llm", prompt_template="Test")
        result = StageResult(
            step_id="s1",
            head_id="mock-llm",
            status=StepStatus.FAILED,
            error="Validation error",
            outputs={"confidence": 0.5},
        )

        reflection = await evaluator.evaluate(step, result, {})

        assert reflection.passed is False
        # Should contain feedback from both evaluators
        text_lower = reflection.reflection_text.lower()
        assert "confidence" in text_lower or "validation" in text_lower


class TestReflectionMemory:
    """Test reflection memory storage."""

    def test_stores_and_retrieves_reflections(self):
        """Memory should store and retrieve reflections."""
        memory = ReflectionMemory()

        reflection1 = ReflectionResult(
            step_id="s1",
            attempt_number=1,
            quality_score=0.5,
            passed=False,
            reflection_text="Failed",
            suggested_changes={},
            should_retry=True,
            metadata={},
            created_at=None,
        )

        reflection2 = ReflectionResult(
            step_id="s1",
            attempt_number=2,
            quality_score=0.8,
            passed=True,
            reflection_text="Passed",
            suggested_changes={},
            should_retry=False,
            metadata={},
            created_at=None,
        )

        memory.add(reflection1)
        memory.add(reflection2)

        history = memory.get_history("s1")
        assert len(history) == 2
        assert history[0].attempt_number == 1
        assert history[1].attempt_number == 2

    def test_separate_history_per_step(self):
        """Memory should separate history by step_id."""
        memory = ReflectionMemory()

        r1 = ReflectionResult("s1", 1, 0.5, False, "F1", {}, True, {}, None)
        r2 = ReflectionResult("s2", 1, 0.6, False, "F2", {}, True, {}, None)

        memory.add(r1)
        memory.add(r2)

        assert len(memory.get_history("s1")) == 1
        assert len(memory.get_history("s2")) == 1
        assert memory.get_attempt_count("s1") == 1
        assert memory.get_attempt_count("s2") == 1


class TestReflectionEngine:
    """Test reflection engine orchestration."""

    @pytest.mark.asyncio
    async def test_no_refinement_if_passed(self):
        """Engine should return None if evaluation passes."""
        evaluator = ConfidenceEvaluator(confidence_threshold=0.7)
        engine = ReflectionEngine(evaluator, max_attempts=3)

        step = StepDef(step_id="s1", name="Test", head_id="mock-llm", prompt_template="Test")
        result = StageResult(
            step_id="s1",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
            outputs={"confidence": 0.9},
        )

        reflection = await engine.should_refine(step, result, {}, attempt=1)

        assert reflection is None
        assert engine.memory.get_attempt_count("s1") == 1

    @pytest.mark.asyncio
    async def test_refinement_if_failed(self):
        """Engine should return reflection if evaluation fails."""
        evaluator = ConfidenceEvaluator(confidence_threshold=0.7)
        engine = ReflectionEngine(evaluator, max_attempts=3)

        step = StepDef(step_id="s1", name="Test", head_id="mock-llm", prompt_template="Test")
        result = StageResult(
            step_id="s1",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
            outputs={"confidence": 0.5},
        )

        reflection = await engine.should_refine(step, result, {}, attempt=1)

        assert reflection is not None
        assert reflection.should_retry is True
        assert reflection.attempt_number == 1

    @pytest.mark.asyncio
    async def test_stops_at_max_attempts(self):
        """Engine should stop refinement at max attempts."""
        evaluator = ConfidenceEvaluator(confidence_threshold=0.7)
        engine = ReflectionEngine(evaluator, max_attempts=2)

        step = StepDef(step_id="s1", name="Test", head_id="mock-llm", prompt_template="Test")
        result = StageResult(
            step_id="s1",
            head_id="mock-llm",
            status=StepStatus.COMMITTED,
            outputs={"confidence": 0.5},
        )

        # Attempt 1 - should refine
        reflection = await engine.should_refine(step, result, {}, attempt=1)
        assert reflection is not None

        # Attempt 2 (max) - should NOT refine
        reflection = await engine.should_refine(step, result, {}, attempt=2)
        assert reflection is None

    def test_builds_refinement_context(self):
        """Engine should build context from previous reflections."""
        evaluator = ConfidenceEvaluator(confidence_threshold=0.7)
        engine = ReflectionEngine(evaluator, max_attempts=3)

        # Add some reflections to memory
        engine.memory.add(ReflectionResult(
            step_id="s1",
            attempt_number=1,
            quality_score=0.5,
            passed=False,
            reflection_text="First attempt failed",
            suggested_changes={},
            should_retry=True,
            metadata={},
            created_at=None,
        ))

        engine.memory.add(ReflectionResult(
            step_id="s1",
            attempt_number=2,
            quality_score=0.6,
            passed=False,
            reflection_text="Second attempt improved but still failed",
            suggested_changes={},
            should_retry=True,
            metadata={},
            created_at=None,
        ))

        context = engine.get_refinement_context("s1")

        assert "Previous Attempts" in context
        assert "Attempt 1" in context
        assert "Attempt 2" in context
        assert "First attempt failed" in context
        assert "Second attempt improved" in context


class TestReflectionIntegration:
    """Test reflection integrated into orchestrator."""

    @pytest.mark.asyncio
    async def test_step_without_reflection_executes_normally(self, orchestrator):
        """Steps without reflection should execute as before."""
        wo = WorkOrder(
            goal="Test no reflection",
            steps=[
                StepDef(
                    step_id="s1",
                    name="S1",
                    head_id="mock-llm",
                    prompt_template="Hello",
                    enable_reflection=False,
                ),
            ],
        )
        state = await orchestrator.create_run(wo, normalize=False)
        state = await orchestrator.execute_run(state.run_id)

        assert "s1" in state.step_results
        assert state.step_results["s1"].status.value == "committed"

        # Should not have reflection events
        events = orchestrator.events.read_events(state.run_id)
        reflection_kinds = (EventKind.STEP_REFLECTION, EventKind.STEP_REFINED)
        reflection_events = [e for e in events if e.kind in reflection_kinds]
        assert len(reflection_events) == 0

    @pytest.mark.asyncio
    async def test_step_with_passing_evaluator_no_refinement(self, orchestrator):
        """Step with passing evaluator should execute once with no refinement."""
        wo = WorkOrder(
            goal="Test passing evaluator",
            steps=[
                StepDef(
                    step_id="s1",
                    name="S1",
                    head_id="mock-llm",
                    prompt_template="Hello",
                    enable_reflection=True,
                    evaluator=ErrorEvaluator(),  # Will pass since mock doesn't error
                    max_reflection_attempts=3,
                ),
            ],
        )
        state = await orchestrator.create_run(wo, normalize=False)
        state = await orchestrator.execute_run(state.run_id)

        assert "s1" in state.step_results
        assert state.step_results["s1"].status.value == "committed"

        # Should not have reflection events (passed immediately)
        events = orchestrator.events.read_events(state.run_id)
        reflection_kinds = (EventKind.STEP_REFLECTION, EventKind.STEP_REFINED)
        reflection_events = [e for e in events if e.kind in reflection_kinds]
        assert len(reflection_events) == 0

    @pytest.mark.asyncio
    async def test_reflection_events_emitted(self, orchestrator):
        """Reflection loop should emit appropriate events."""
        # Create a custom evaluator that always fails on first attempt
        class FirstAttemptFailsEvaluator:
            def __init__(self):
                self.call_count = 0

            async def evaluate(self, step, result, context):
                from multihead.reflection import ReflectionResult
                self.call_count += 1

                if self.call_count == 1:
                    # Fail first attempt
                    return ReflectionResult(
                        step_id=step.step_id,
                        attempt_number=1,
                        quality_score=0.5,
                        passed=False,
                        reflection_text="First attempt quality too low",
                        suggested_changes={"improve": True},
                        should_retry=True,
                        metadata={},
                        created_at=None,
                    )
                else:
                    # Pass second attempt
                    return ReflectionResult(
                        step_id=step.step_id,
                        attempt_number=self.call_count,
                        quality_score=0.9,
                        passed=True,
                        reflection_text="Passed",
                        suggested_changes={},
                        should_retry=False,
                        metadata={},
                        created_at=None,
                    )

        wo = WorkOrder(
            goal="Test reflection events",
            steps=[
                StepDef(
                    step_id="s1",
                    name="S1",
                    head_id="mock-llm",
                    prompt_template="Hello",
                    enable_reflection=True,
                    evaluator=FirstAttemptFailsEvaluator(),
                    max_reflection_attempts=3,
                ),
            ],
        )
        state = await orchestrator.create_run(wo, normalize=False)
        state = await orchestrator.execute_run(state.run_id)

        # Check for reflection and refinement events
        events = orchestrator.events.read_events(state.run_id)
        reflection_events = [e for e in events if e.kind == EventKind.STEP_REFLECTION]
        refined_events = [e for e in events if e.kind == EventKind.STEP_REFINED]

        # Should have at least one reflection and one refinement
        assert len(reflection_events) >= 1
        assert len(refined_events) >= 1

        # Check reflection event data
        first_reflection = reflection_events[0]
        assert "quality_score" in first_reflection.data
        assert "reflection_text" in first_reflection.data
        assert first_reflection.data["quality_score"] == 0.5
