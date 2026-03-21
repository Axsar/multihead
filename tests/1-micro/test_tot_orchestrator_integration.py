"""Tests for Tree-of-Thoughts integration with orchestrator execution path.

Verifies that use_tot=True in step.extra triggers ToT exploration via
the orchestrator's _execute_step_with_reflection() method.
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
    StageResult,
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


class TestTotOrchestratorIntegration:
    """Test that ToT is triggered through the orchestrator execution path."""

    @pytest.mark.asyncio
    async def test_step_without_tot_no_tot_events(self, orchestrator):
        """Steps without use_tot should execute normally — no ToT events."""
        wo = WorkOrder(
            goal="Test no ToT",
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

        # No ToT events
        events = orchestrator.events.read_events(state.run_id)
        tot_events = [
            e for e in events
            if e.kind in (EventKind.STEP_TOT_STARTED, EventKind.STEP_TOT_COMPLETE)
        ]
        assert len(tot_events) == 0

    @pytest.mark.asyncio
    async def test_step_with_use_tot_emits_tot_events(self, orchestrator):
        """Steps with use_tot=True should trigger ToT and emit events."""
        wo = WorkOrder(
            goal="Test ToT path",
            steps=[
                StepDef(
                    step_id="s1",
                    name="Exploratory step",
                    head_id="mock-llm",
                    prompt_template="Explore the problem space",
                    extra={"use_tot": True, "tot_strategy": "bfs"},
                ),
            ],
        )
        state = await orchestrator.create_run(wo, normalize=False)
        state = await orchestrator.execute_run(state.run_id)

        # Step should still produce a result (ToT wraps execution)
        assert "s1" in state.step_results

        # Should have ToT started and complete events
        events = orchestrator.events.read_events(state.run_id)
        tot_started = [e for e in events if e.kind == EventKind.STEP_TOT_STARTED]
        tot_complete = [e for e in events if e.kind == EventKind.STEP_TOT_COMPLETE]

        assert len(tot_started) == 1, f"Expected 1 STEP_TOT_STARTED, got {len(tot_started)}"
        assert len(tot_complete) == 1, f"Expected 1 STEP_TOT_COMPLETE, got {len(tot_complete)}"

        # Verify event data
        assert tot_started[0].data["strategy"] == "bfs"
        assert tot_started[0].data["num_alternatives"] == 3  # default
        assert tot_started[0].data["max_depth"] == 2  # default

    @pytest.mark.asyncio
    async def test_tot_with_beam_strategy(self, orchestrator):
        """ToT should respect beam strategy from step extra."""
        wo = WorkOrder(
            goal="Test beam strategy",
            steps=[
                StepDef(
                    step_id="s1",
                    name="Beam exploration",
                    head_id="mock-llm",
                    prompt_template="Find best approach",
                    extra={
                        "use_tot": True,
                        "tot_strategy": "beam",
                        "tot_alternatives": 5,
                        "tot_max_depth": 3,
                    },
                ),
            ],
        )
        state = await orchestrator.create_run(wo, normalize=False)
        state = await orchestrator.execute_run(state.run_id)

        events = orchestrator.events.read_events(state.run_id)
        tot_started = [e for e in events if e.kind == EventKind.STEP_TOT_STARTED]

        assert len(tot_started) == 1
        assert tot_started[0].data["strategy"] == "beam"
        assert tot_started[0].data["num_alternatives"] == 5
        assert tot_started[0].data["max_depth"] == 3

    @pytest.mark.asyncio
    async def test_tot_false_no_tot_events(self, orchestrator):
        """Steps with explicit use_tot=False should NOT trigger ToT."""
        wo = WorkOrder(
            goal="Test explicit no ToT",
            steps=[
                StepDef(
                    step_id="s1",
                    name="No ToT step",
                    head_id="mock-llm",
                    prompt_template="Direct execution",
                    extra={"use_tot": False},
                ),
            ],
        )
        state = await orchestrator.create_run(wo, normalize=False)
        state = await orchestrator.execute_run(state.run_id)

        assert state.step_results["s1"].status == StepStatus.COMMITTED

        events = orchestrator.events.read_events(state.run_id)
        tot_events = [
            e for e in events
            if e.kind in (EventKind.STEP_TOT_STARTED, EventKind.STEP_TOT_COMPLETE)
        ]
        assert len(tot_events) == 0

    @pytest.mark.asyncio
    async def test_multi_step_only_tot_step_gets_tot(self, orchestrator):
        """In a multi-step WorkOrder, only steps with use_tot get ToT treatment."""
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
                    name="ToT step",
                    head_id="mock-llm",
                    prompt_template="Step 2",
                    extra={"use_tot": True},
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

        # All steps should succeed
        assert all(
            state.step_results[sid].status == StepStatus.COMMITTED
            for sid in ("s1", "s2", "s3")
            if sid in state.step_results
        )

        # Only s2 should have ToT events
        events = orchestrator.events.read_events(state.run_id)
        tot_started = [e for e in events if e.kind == EventKind.STEP_TOT_STARTED]

        assert len(tot_started) == 1
        assert tot_started[0].step_id == "s2"
