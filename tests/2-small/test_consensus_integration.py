"""Integration tests for consensus engine with orchestrator and DAG executor."""

from __future__ import annotations

import pytest

from multihead.artifact_store import ArtifactStore
from multihead.consensus import ConsensusConfig, ConsensusStrategy, HeadTask
from multihead.dag_executor import DAGExecutor
from multihead.event_store import EventStore
from multihead.head_manager import HeadManager
from multihead.models import (
    AdapterKind,
    HeadManifest,
    RunState,
    RunStatus,
    StepDef,
    WorkOrder,
)
from multihead.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def three_head_manifests():
    return {
        "head-a": HeadManifest(
            head_id="head-a", name="Head A", adapter=AdapterKind.MOCK,
            model="mock-v1", kind="llm", gpu_required=False,
        ),
        "head-b": HeadManifest(
            head_id="head-b", name="Head B", adapter=AdapterKind.MOCK,
            model="mock-v1", kind="llm", gpu_required=False,
        ),
        "head-c": HeadManifest(
            head_id="head-c", name="Head C", adapter=AdapterKind.MOCK,
            model="mock-v1", kind="llm", gpu_required=False,
        ),
    }


@pytest.fixture
def orchestrator(tmp_path, three_head_manifests):
    runs_dir = tmp_path / "runs"
    db_path = tmp_path / "test.db"
    artifact_store = ArtifactStore(tmp_path / "artifacts", db_path)
    event_store = EventStore(runs_dir, db_path)
    head_manager = HeadManager(three_head_manifests)
    return Orchestrator(event_store, artifact_store, head_manager, runs_dir)


# ---------------------------------------------------------------------------
# Orchestrator integration
# ---------------------------------------------------------------------------


class TestOrchestratorConsensus:
    @pytest.mark.asyncio
    async def test_consensus_step_in_pipeline(self, orchestrator):
        """A step with consensus config should execute via ConsensusEngine."""
        consensus = ConsensusConfig(
            heads=[
                HeadTask(head_id="head-a"),
                HeadTask(head_id="head-b"),
                HeadTask(head_id="head-c"),
            ],
            strategy=ConsensusStrategy.MAJORITY,
        )
        wo = WorkOrder(
            goal="test consensus pipeline",
            steps=[
                StepDef(
                    name="consensus-step",
                    head_id="head-a",  # Fallback, not used in consensus mode
                    prompt_template="What is 2+2?",
                    consensus=consensus,
                ),
            ],
        )
        state = await orchestrator.create_run(wo)
        state = await orchestrator.execute_run(state.run_id)

        assert state.status == RunStatus.DONE
        result = list(state.step_results.values())[0]
        assert "consensus" in result.head_id
        assert result.outputs.get("text")

    @pytest.mark.asyncio
    async def test_mixed_pipeline(self, orchestrator):
        """Pipeline with regular step followed by consensus step."""
        consensus = ConsensusConfig(
            heads=[
                HeadTask(head_id="head-a"),
                HeadTask(head_id="head-b"),
            ],
        )
        wo = WorkOrder(
            goal="test mixed pipeline",
            steps=[
                StepDef(
                    name="regular",
                    head_id="head-a",
                    prompt_template="Plan something",
                ),
                StepDef(
                    name="verified",
                    head_id="head-a",
                    prompt_template="Verify the plan",
                    input_refs=["regular"],
                    consensus=consensus,
                ),
            ],
        )
        state = await orchestrator.create_run(wo)
        state = await orchestrator.execute_run(state.run_id)

        assert state.status == RunStatus.DONE
        assert len(state.step_results) == 2
        # First step is regular
        regular_result = list(state.step_results.values())[0]
        assert "consensus" not in regular_result.head_id
        # Second step is consensus
        consensus_result = list(state.step_results.values())[1]
        assert "consensus" in consensus_result.head_id

    @pytest.mark.asyncio
    async def test_consensus_metrics_in_result(self, orchestrator):
        """Consensus metrics should be recorded in StageResult."""
        consensus = ConsensusConfig(
            heads=[HeadTask(head_id="head-a"), HeadTask(head_id="head-b")],
        )
        wo = WorkOrder(
            goal="metrics test",
            steps=[
                StepDef(
                    name="step1", head_id="head-a",
                    prompt_template="test", consensus=consensus,
                ),
            ],
        )
        state = await orchestrator.create_run(wo)
        state = await orchestrator.execute_run(state.run_id)

        result = list(state.step_results.values())[0]
        assert "consensus_agreement" in result.metrics

    @pytest.mark.asyncio
    async def test_fail_on_disagreement(self, orchestrator, three_head_manifests):
        """fail_on_disagreement + all heads fail → step failure."""
        # Make all heads fail
        hm = orchestrator.heads
        for hid in ["head-a", "head-b", "head-c"]:
            adapter = hm.get_adapter(hid)

            async def failing_gen(prompt, **kw):
                raise RuntimeError("broken")

            adapter.generate = failing_gen

        consensus = ConsensusConfig(
            heads=[
                HeadTask(head_id="head-a"),
                HeadTask(head_id="head-b"),
                HeadTask(head_id="head-c"),
            ],
            fail_on_disagreement=True,
        )
        wo = WorkOrder(
            goal="fail test",
            steps=[
                StepDef(
                    name="will-fail", head_id="head-a",
                    prompt_template="test", consensus=consensus,
                ),
            ],
        )
        state = await orchestrator.create_run(wo)
        state = await orchestrator.execute_run(state.run_id)
        assert state.status == RunStatus.FAILED


# ---------------------------------------------------------------------------
# DAG executor integration
# ---------------------------------------------------------------------------


class TestDAGConsensus:
    @pytest.mark.asyncio
    async def test_consensus_step_in_dag(self, orchestrator):
        """Consensus step in DAG mode should work."""
        consensus = ConsensusConfig(
            heads=[HeadTask(head_id="head-a"), HeadTask(head_id="head-b")],
        )
        wo = WorkOrder(
            goal="DAG consensus",
            steps=[
                StepDef(
                    name="step1", head_id="head-a",
                    prompt_template="First step",
                ),
                StepDef(
                    name="verified", head_id="head-a",
                    prompt_template="Verify",
                    depends_on=["step1"],
                    consensus=consensus,
                ),
            ],
        )
        # Use step_ids for depends_on
        wo.steps[1].depends_on = [wo.steps[0].step_id]

        state = RunState(run_id=wo.run_id, work_order=wo)
        dag = DAGExecutor(orchestrator)
        state = await dag.execute_dag(wo.run_id, wo, state)

        assert len(state.step_results) == 2
        # Second step should be consensus
        verified_result = state.step_results[wo.steps[1].step_id]
        assert "consensus" in verified_result.head_id

    @pytest.mark.asyncio
    async def test_parallel_consensus_steps(self, orchestrator):
        """Multiple consensus steps in same layer should work."""
        consensus = ConsensusConfig(
            heads=[HeadTask(head_id="head-a"), HeadTask(head_id="head-b")],
        )
        wo = WorkOrder(
            goal="parallel consensus",
            steps=[
                StepDef(
                    name="verify-a", head_id="head-a",
                    prompt_template="Check A", consensus=consensus,
                ),
                StepDef(
                    name="verify-b", head_id="head-a",
                    prompt_template="Check B", consensus=consensus,
                ),
            ],
        )
        state = RunState(run_id=wo.run_id, work_order=wo)
        dag = DAGExecutor(orchestrator)
        state = await dag.execute_dag(wo.run_id, wo, state)

        assert len(state.step_results) == 2


# ---------------------------------------------------------------------------
# Cross-modal integration
# ---------------------------------------------------------------------------


class TestCrossModalIntegration:
    @pytest.mark.asyncio
    async def test_cross_modal_pipeline(self, orchestrator, three_head_manifests):
        """Cross-modal step with different prompts per head."""
        import json

        hm = orchestrator.heads

        # Head A returns object count
        adapter_a = hm.get_adapter("head-a")

        async def detect_gen(prompt, **kw):
            return {"text": json.dumps({"count": 3, "objects": ["face", "face", "face"]}),
                    "tokens_in": 10, "tokens_out": 20}

        adapter_a.generate = detect_gen

        # Head B returns same count from different modality
        adapter_b = hm.get_adapter("head-b")

        async def describe_gen(prompt, **kw):
            return {"text": json.dumps({"count": 3, "description": "Three people in frame"}),
                    "tokens_in": 10, "tokens_out": 20}

        adapter_b.generate = describe_gen

        consensus = ConsensusConfig(
            heads=[
                HeadTask(
                    head_id="head-a",
                    prompt_template="Detect objects",
                    extract_fields=["count"],
                ),
                HeadTask(
                    head_id="head-b",
                    prompt_template="Describe scene",
                    extract_fields=["count"],
                ),
            ],
            cross_modal=True,
        )
        wo = WorkOrder(
            goal="cross-modal test",
            steps=[
                StepDef(
                    name="analyze", head_id="head-a",
                    prompt_template="Analyze image",
                    consensus=consensus,
                ),
            ],
        )
        state = await orchestrator.create_run(wo)
        state = await orchestrator.execute_run(state.run_id)

        assert state.status == RunStatus.DONE
        result = list(state.step_results.values())[0]
        # No warnings because both heads agree on count=3
        cross_modal_warnings = [w for w in result.warnings if "cross_modal" in w.lower()]
        assert len(cross_modal_warnings) == 0
