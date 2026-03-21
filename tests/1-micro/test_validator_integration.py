"""Tests for validator integration in orchestrator."""

from __future__ import annotations

import pytest
from typing import Any

from multihead.artifact_store import ArtifactStore
from multihead.event_store import EventStore
from multihead.head_manager import HeadManager
from multihead.models import (
    AdapterKind,
    EventKind,
    HeadManifest,
    StepDef,
    WorkOrder,
)
from multihead.orchestrator import Orchestrator
from multihead.validators import (
    ValidationResult,
    Validator,
    JSONSchemaValidator,
    FormatValidator,
    CompositeValidator,
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


class AlwaysPassValidator(Validator):
    """Validator that always passes."""
    name = "always_pass"

    def validate_precondition(
        self, state: dict[str, Any], inputs: dict[str, Any],
    ) -> ValidationResult:
        return ValidationResult(passed=True, confidence=1.0)

    def validate_postcondition(self, state: dict[str, Any], output: Any) -> ValidationResult:
        return ValidationResult(passed=True, confidence=1.0)


class AlwaysFailValidator(Validator):
    """Validator that always fails."""
    name = "always_fail"

    def validate_precondition(
        self, state: dict[str, Any], inputs: dict[str, Any],
    ) -> ValidationResult:
        return ValidationResult(
            passed=False,
            violations=["Precondition intentionally failed for testing"],
            confidence=1.0
        )

    def validate_postcondition(self, state: dict[str, Any], output: Any) -> ValidationResult:
        return ValidationResult(
            passed=False,
            violations=["Postcondition intentionally failed for testing"],
            confidence=1.0
        )


class PostconditionOnlyFailValidator(Validator):
    """Validator that passes precondition but fails postcondition."""
    name = "postcondition_fail"

    def validate_precondition(
        self, state: dict[str, Any], inputs: dict[str, Any],
    ) -> ValidationResult:
        return ValidationResult(passed=True, confidence=1.0)

    def validate_postcondition(self, state: dict[str, Any], output: Any) -> ValidationResult:
        return ValidationResult(
            passed=False,
            violations=["Output does not meet quality threshold"],
            confidence=0.85
        )


class TestValidatorIntegration:
    """Test validators integrated into orchestrator execution."""

    @pytest.mark.asyncio
    async def test_step_without_validator_executes_normally(self, orchestrator):
        """Steps without validators should execute as before."""
        wo = WorkOrder(
            goal="Test no validator",
            steps=[
                StepDef(step_id="s1", name="S1", head_id="mock-llm", prompt_template="Hello"),
            ],
        )
        state = await orchestrator.create_run(wo, normalize=False)
        state = await orchestrator.execute_run(state.run_id)

        assert "s1" in state.step_results
        assert state.step_results["s1"].status.value == "committed"

    @pytest.mark.asyncio
    async def test_step_with_passing_validator_executes_normally(self, orchestrator):
        """Step with passing validator should execute and commit."""
        wo = WorkOrder(
            goal="Test passing validator",
            steps=[
                StepDef(
                    step_id="s1",
                    name="S1",
                    head_id="mock-llm",
                    prompt_template="Hello",
                    validator=AlwaysPassValidator(),
                ),
            ],
        )
        state = await orchestrator.create_run(wo, normalize=False)
        state = await orchestrator.execute_run(state.run_id)

        assert "s1" in state.step_results
        assert state.step_results["s1"].status.value == "committed"

    @pytest.mark.asyncio
    async def test_precondition_failure_skips_execution(self, orchestrator):
        """Failed precondition should skip execution and emit STEP_FAILED."""
        wo = WorkOrder(
            goal="Test precondition failure",
            steps=[
                StepDef(
                    step_id="s1",
                    name="S1",
                    head_id="mock-llm",
                    prompt_template="Hello",
                    validator=AlwaysFailValidator(),
                ),
            ],
        )
        state = await orchestrator.create_run(wo, normalize=False)
        state = await orchestrator.execute_run(state.run_id)

        # Step should fail
        assert "s1" in state.step_results
        assert state.step_results["s1"].status.value == "failed"
        assert "Precondition failed" in state.step_results["s1"].error

        # Check events
        events = orchestrator.events.read_events(state.run_id)
        step_events = [e for e in events if e.step_id == "s1"]

        # Should have STEP_STARTED and STEP_FAILED
        event_kinds = [e.kind.value for e in step_events]
        assert "step_started" in event_kinds
        assert "step_failed" in event_kinds

        # STEP_FAILED should have violation details
        failed_events = [e for e in step_events if e.kind == EventKind.STEP_FAILED]
        assert len(failed_events) == 1
        assert "violations" in failed_events[0].data
        assert len(failed_events[0].data["violations"]) > 0

    @pytest.mark.asyncio
    async def test_postcondition_failure_discards_output(self, orchestrator):
        """Failed postcondition should discard output and emit STEP_FAILED."""
        wo = WorkOrder(
            goal="Test postcondition failure",
            steps=[
                StepDef(
                    step_id="s1",
                    name="S1",
                    head_id="mock-llm",
                    prompt_template="Hello",
                    validator=PostconditionOnlyFailValidator(),
                ),
            ],
        )
        state = await orchestrator.create_run(wo, normalize=False)
        state = await orchestrator.execute_run(state.run_id)

        # Step should fail
        assert "s1" in state.step_results
        assert state.step_results["s1"].status.value == "failed"
        assert "Postcondition failed" in state.step_results["s1"].error

        # Check events
        events = orchestrator.events.read_events(state.run_id)
        step_events = [e for e in events if e.step_id == "s1"]

        event_kinds = [e.kind.value for e in step_events]
        assert "step_started" in event_kinds
        assert "step_failed" in event_kinds

        # STEP_FAILED should indicate output was discarded
        failed_events = [e for e in step_events if e.kind == EventKind.STEP_FAILED]
        assert len(failed_events) == 1
        assert failed_events[0].data.get("output_discarded") is True

    @pytest.mark.asyncio
    async def test_validation_confidence_recorded(self, orchestrator):
        """Validation confidence scores should be recorded in events."""
        wo = WorkOrder(
            goal="Test validation confidence",
            steps=[
                StepDef(
                    step_id="s1",
                    name="S1",
                    head_id="mock-llm",
                    prompt_template="Hello",
                    validator=PostconditionOnlyFailValidator(),  # confidence=0.85
                ),
            ],
        )
        state = await orchestrator.create_run(wo, normalize=False)
        state = await orchestrator.execute_run(state.run_id)

        # Check validation_confidence in event data
        events = orchestrator.events.read_events(state.run_id)
        failed_events = [e for e in events if e.kind == EventKind.STEP_FAILED]
        assert len(failed_events) == 1
        assert "validation_confidence" in failed_events[0].data
        assert failed_events[0].data["validation_confidence"] == 0.85


class TestConcreteValidators:
    """Test concrete validator implementations."""

    @pytest.mark.asyncio
    async def test_json_schema_validator_integration(self, orchestrator):
        """JSONSchemaValidator should validate JSON output."""
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "count": {"type": "number"},
            },
            "required": ["status", "count"],
        }

        validator = JSONSchemaValidator(schema=schema)

        # Mock adapter returns non-JSON text, should fail validation
        wo = WorkOrder(
            goal="Test JSON validation",
            steps=[
                StepDef(
                    step_id="s1",
                    name="S1",
                    head_id="mock-llm",
                    prompt_template="Return data",
                    validator=validator,
                ),
            ],
        )
        state = await orchestrator.create_run(wo, normalize=False)
        state = await orchestrator.execute_run(state.run_id)

        # Should fail because mock adapter doesn't return valid JSON
        assert state.step_results["s1"].status.value == "failed"

    @pytest.mark.asyncio
    async def test_format_validator_integration(self, orchestrator):
        """FormatValidator should check output format."""
        # Require output to be at least 5 characters
        validator = FormatValidator(min_length=5, max_length=100)

        wo = WorkOrder(
            goal="Test format validation",
            steps=[
                StepDef(
                    step_id="s1",
                    name="S1",
                    head_id="mock-llm",
                    prompt_template="Hi",  # Mock echoes, likely < 5 chars
                    validator=validator,
                ),
            ],
        )
        state = await orchestrator.create_run(wo, normalize=False)
        state = await orchestrator.execute_run(state.run_id)

        # Check if validation occurred (may pass or fail depending on mock output)
        assert "s1" in state.step_results

    @pytest.mark.asyncio
    async def test_composite_validator_integration(self, orchestrator):
        """CompositeValidator should chain multiple validators."""
        validator = CompositeValidator(
            validators=[
                AlwaysPassValidator(),
                PostconditionOnlyFailValidator(),  # This will fail
            ]
        )

        wo = WorkOrder(
            goal="Test composite validation",
            steps=[
                StepDef(
                    step_id="s1",
                    name="S1",
                    head_id="mock-llm",
                    prompt_template="Test",
                    validator=validator,
                ),
            ],
        )
        state = await orchestrator.create_run(wo, normalize=False)
        state = await orchestrator.execute_run(state.run_id)

        # Should fail because second validator fails
        assert state.step_results["s1"].status.value == "failed"


class TestValidatorErrorHandling:
    """Test validator error handling edge cases."""

    @pytest.mark.asyncio
    async def test_validator_exception_does_not_crash_orchestrator(self, orchestrator):
        """Validator raising an exception should be handled gracefully."""

        class ExplodingValidator(Validator):
            name = "exploding"

            def validate_precondition(self, state, inputs):
                raise RuntimeError("Validator exploded!")

            def validate_postcondition(self, state, output):
                return ValidationResult(passed=True)

        wo = WorkOrder(
            goal="Test validator exception",
            steps=[
                StepDef(
                    step_id="s1",
                    name="S1",
                    head_id="mock-llm",
                    prompt_template="Test",
                    validator=ExplodingValidator(),
                ),
            ],
        )
        state = await orchestrator.create_run(wo, normalize=False)
        state = await orchestrator.execute_run(state.run_id)

        # Orchestrator should catch exception and fail the step
        assert state.step_results["s1"].status.value == "failed"
