"""Tests for Process Reward Model integration with orchestrator execution path.

Verifies that use_prm=True in step.extra triggers PRM scoring via
the orchestrator's _execute_step() method, emits STEP_PRM_SCORED events,
and enforces quality gates when prm_fail_on_low=True.
"""

from __future__ import annotations

import pytest

from multihead.artifact_store import ArtifactStore
from multihead.event_store import EventStore
from multihead.head_manager import HeadManager
from multihead.models import (
    AdapterKind,
    EventKind,
    HeadManifest,
    StepDef,
    StepStatus,
    WorkOrder,
)
from multihead.orchestrator import Orchestrator


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


class TestPRMOrchestratorIntegration:
    """Test that PRM scoring is triggered through the orchestrator."""

    @pytest.mark.asyncio
    async def test_step_without_prm_no_prm_events(self, orchestrator):
        """Steps without use_prm should execute normally — no PRM events."""
        wo = WorkOrder(
            goal="Test no PRM",
            steps=[
                StepDef(
                    step_id="s1",
                    name="Normal step",
                    head_id="mock-llm",
                    prompt_template="Hello",
                ),
            ],
        )
        state = await orchestrator.create_run(wo, normalize=False)
        state = await orchestrator.execute_run(state.run_id)

        assert "s1" in state.step_results
        assert state.step_results["s1"].status == StepStatus.COMMITTED

        # No PRM events
        events = orchestrator.events.read_events(state.run_id)
        prm_events = [e for e in events if e.kind == EventKind.STEP_PRM_SCORED]
        assert len(prm_events) == 0

    @pytest.mark.asyncio
    async def test_step_with_use_prm_emits_prm_event(self, orchestrator):
        """Steps with use_prm=True should trigger PRM scoring and emit events."""
        wo = WorkOrder(
            goal="Test PRM path",
            steps=[
                StepDef(
                    step_id="s1",
                    name="Implementation step",
                    head_id="mock-llm",
                    prompt_template="Implement the feature",
                    extra={"use_prm": True},
                ),
            ],
        )
        state = await orchestrator.create_run(wo, normalize=False)
        state = await orchestrator.execute_run(state.run_id)

        # Step should succeed (mock produces output)
        assert "s1" in state.step_results
        assert state.step_results["s1"].status == StepStatus.COMMITTED

        # Should have PRM scored event
        events = orchestrator.events.read_events(state.run_id)
        prm_events = [e for e in events if e.kind == EventKind.STEP_PRM_SCORED]

        assert len(prm_events) == 1, f"Expected 1 STEP_PRM_SCORED, got {len(prm_events)}"

        # Verify event data
        prm_data = prm_events[0].data
        assert "prm_score" in prm_data
        assert "prm_quality" in prm_data
        assert "prm_feedback" in prm_data
        assert 0.0 <= prm_data["prm_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_prm_score_stored_in_result_metrics(self, orchestrator):
        """PRM scores should be stored in the step result metrics."""
        wo = WorkOrder(
            goal="Test PRM metrics",
            steps=[
                StepDef(
                    step_id="s1",
                    name="Scored step",
                    head_id="mock-llm",
                    prompt_template="Do something",
                    extra={"use_prm": True, "prm_threshold": 0.5},
                ),
            ],
        )
        state = await orchestrator.create_run(wo, normalize=False)
        state = await orchestrator.execute_run(state.run_id)

        result = state.step_results["s1"]
        assert result.metrics is not None
        assert "prm_score" in result.metrics
        assert "prm_quality" in result.metrics
        assert "prm_confidence" in result.metrics

    @pytest.mark.asyncio
    async def test_prm_false_no_prm_events(self, orchestrator):
        """Steps with explicit use_prm=False should NOT trigger PRM."""
        wo = WorkOrder(
            goal="Test explicit no PRM",
            steps=[
                StepDef(
                    step_id="s1",
                    name="No PRM step",
                    head_id="mock-llm",
                    prompt_template="Direct execution",
                    extra={"use_prm": False},
                ),
            ],
        )
        state = await orchestrator.create_run(wo, normalize=False)
        state = await orchestrator.execute_run(state.run_id)

        assert state.step_results["s1"].status == StepStatus.COMMITTED

        events = orchestrator.events.read_events(state.run_id)
        prm_events = [e for e in events if e.kind == EventKind.STEP_PRM_SCORED]
        assert len(prm_events) == 0

    @pytest.mark.asyncio
    async def test_multi_step_only_prm_step_gets_scored(self, orchestrator):
        """In a multi-step WorkOrder, only steps with use_prm get scoring."""
        wo = WorkOrder(
            goal="Mixed steps",
            steps=[
                StepDef(
                    step_id="s1",
                    name="Normal step",
                    head_id="mock-llm",
                    prompt_template="Step 1",
                ),
                StepDef(
                    step_id="s2",
                    name="PRM step",
                    head_id="mock-llm",
                    prompt_template="Step 2",
                    extra={"use_prm": True},
                ),
                StepDef(
                    step_id="s3",
                    name="Another normal",
                    head_id="mock-llm",
                    prompt_template="Step 3",
                ),
            ],
        )
        state = await orchestrator.create_run(wo, normalize=False)
        state = await orchestrator.execute_run(state.run_id)

        # All should succeed
        for sid in ("s1", "s2", "s3"):
            if sid in state.step_results:
                assert state.step_results[sid].status == StepStatus.COMMITTED

        # Only s2 should have PRM events
        events = orchestrator.events.read_events(state.run_id)
        prm_events = [e for e in events if e.kind == EventKind.STEP_PRM_SCORED]

        assert len(prm_events) == 1
        assert prm_events[0].step_id == "s2"

    @pytest.mark.asyncio
    async def test_prm_quality_gate_does_not_fail_by_default(self, orchestrator):
        """By default, low PRM score should NOT fail the step."""
        wo = WorkOrder(
            goal="Test quality gate off",
            steps=[
                StepDef(
                    step_id="s1",
                    name="Low quality step",
                    head_id="mock-llm",
                    prompt_template="x",
                    extra={"use_prm": True, "prm_threshold": 0.99},
                ),
            ],
        )
        state = await orchestrator.create_run(wo, normalize=False)
        state = await orchestrator.execute_run(state.run_id)

        # Should still be COMMITTED (fail_on_low_score defaults to False)
        assert state.step_results["s1"].status == StepStatus.COMMITTED

    @pytest.mark.asyncio
    async def test_prm_and_tot_together(self, orchestrator):
        """Steps with both use_tot and use_prm should get both treatments."""
        wo = WorkOrder(
            goal="Test combined features",
            steps=[
                StepDef(
                    step_id="s1",
                    name="Full feature step",
                    head_id="mock-llm",
                    prompt_template="Explore and score",
                    extra={"use_tot": True, "use_prm": True},
                ),
            ],
        )
        state = await orchestrator.create_run(wo, normalize=False)
        state = await orchestrator.execute_run(state.run_id)

        assert "s1" in state.step_results

        events = orchestrator.events.read_events(state.run_id)
        tot_events = [e for e in events if e.kind == EventKind.STEP_TOT_STARTED]
        prm_events = [e for e in events if e.kind == EventKind.STEP_PRM_SCORED]

        # ToT runs first (in _execute_step_with_reflection), PRM scores the result
        assert len(tot_events) == 1, f"Expected 1 ToT event, got {len(tot_events)}"
        assert len(prm_events) == 1, f"Expected 1 PRM event, got {len(prm_events)}"
