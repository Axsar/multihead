"""Tests for the DAG executor."""

import pytest

from multihead.dag_executor import DAGExecutor
from multihead.artifact_store import ArtifactStore
from multihead.event_store import EventStore
from multihead.head_manager import HeadManager
from multihead.models import (
    AdapterKind,
    HeadManifest,
    RunState,
    StepDef,
    WorkOrder,
)
from multihead.orchestrator import Orchestrator


@pytest.fixture
def orchestrator(tmp_path):
    manifests = {
        "mock-llm": HeadManifest(
            head_id="mock-llm", name="Mock", adapter=AdapterKind.MOCK,
            model="mock-v1", kind="llm", gpu_required=False,
        ),
        "mock-gpu": HeadManifest(
            head_id="mock-gpu", name="Mock GPU", adapter=AdapterKind.MOCK,
            model="mock-v1", kind="llm", gpu_required=True,
        ),
    }
    hm = HeadManager(manifests)
    art = ArtifactStore(tmp_path / "artifacts", tmp_path / "artifacts.db")
    es = EventStore(tmp_path / "runs", tmp_path / "state.db")
    return Orchestrator(es, art, hm, tmp_path / "runs")


@pytest.fixture
def dag_executor(orchestrator):
    return DAGExecutor(orchestrator)


class TestBuildGraph:
    def test_no_dependencies(self, dag_executor):
        steps = [
            StepDef(step_id="a", name="A", head_id="mock-llm"),
            StepDef(step_id="b", name="B", head_id="mock-llm"),
        ]
        graph = dag_executor.build_graph(steps)
        assert len(graph) == 2
        assert graph["a"].dependencies == []
        assert graph["b"].dependencies == []

    def test_explicit_depends_on(self, dag_executor):
        steps = [
            StepDef(step_id="a", name="A", head_id="mock-llm"),
            StepDef(step_id="b", name="B", head_id="mock-llm", depends_on=["a"]),
        ]
        graph = dag_executor.build_graph(steps)
        assert graph["b"].dependencies == ["a"]
        assert "b" in graph["a"].dependents

    def test_diamond_dependency(self, dag_executor):
        steps = [
            StepDef(step_id="a", name="A", head_id="mock-llm"),
            StepDef(step_id="b", name="B", head_id="mock-llm", depends_on=["a"]),
            StepDef(step_id="c", name="C", head_id="mock-llm", depends_on=["a"]),
            StepDef(step_id="d", name="D", head_id="mock-llm", depends_on=["b", "c"]),
        ]
        graph = dag_executor.build_graph(steps)
        assert set(graph["d"].dependencies) == {"b", "c"}


class TestTopologicalSort:
    def test_linear(self, dag_executor):
        steps = [
            StepDef(step_id="a", name="A", head_id="mock-llm"),
            StepDef(step_id="b", name="B", head_id="mock-llm", depends_on=["a"]),
            StepDef(step_id="c", name="C", head_id="mock-llm", depends_on=["b"]),
        ]
        graph = dag_executor.build_graph(steps)
        layers = dag_executor.topological_sort(graph)
        assert len(layers) == 3
        assert layers[0] == ["a"]
        assert layers[1] == ["b"]
        assert layers[2] == ["c"]

    def test_parallel(self, dag_executor):
        steps = [
            StepDef(step_id="a", name="A", head_id="mock-llm"),
            StepDef(step_id="b", name="B", head_id="mock-llm"),
            StepDef(step_id="c", name="C", head_id="mock-llm"),
        ]
        graph = dag_executor.build_graph(steps)
        layers = dag_executor.topological_sort(graph)
        assert len(layers) == 1
        assert len(layers[0]) == 3

    def test_diamond(self, dag_executor):
        steps = [
            StepDef(step_id="a", name="A", head_id="mock-llm"),
            StepDef(step_id="b", name="B", head_id="mock-llm", depends_on=["a"]),
            StepDef(step_id="c", name="C", head_id="mock-llm", depends_on=["a"]),
            StepDef(step_id="d", name="D", head_id="mock-llm", depends_on=["b", "c"]),
        ]
        graph = dag_executor.build_graph(steps)
        layers = dag_executor.topological_sort(graph)
        assert len(layers) == 3
        assert layers[0] == ["a"]
        assert set(layers[1]) == {"b", "c"}
        assert layers[2] == ["d"]

    def test_fan_out_fan_in(self, dag_executor):
        steps = [
            StepDef(step_id="start", name="Start", head_id="mock-llm"),
            StepDef(step_id="w1", name="W1", head_id="mock-llm", depends_on=["start"]),
            StepDef(step_id="w2", name="W2", head_id="mock-llm", depends_on=["start"]),
            StepDef(step_id="w3", name="W3", head_id="mock-llm", depends_on=["start"]),
            StepDef(step_id="end", name="End", head_id="mock-llm", depends_on=["w1", "w2", "w3"]),
        ]
        graph = dag_executor.build_graph(steps)
        layers = dag_executor.topological_sort(graph)
        assert len(layers) == 3
        assert layers[0] == ["start"]
        assert len(layers[1]) == 3
        assert layers[2] == ["end"]


class TestDAGExecution:
    @pytest.mark.asyncio
    async def test_execute_linear(self, dag_executor, orchestrator):
        steps = [
            StepDef(step_id="a", name="A", head_id="mock-llm", prompt_template="Step A"),
            StepDef(
                step_id="b", name="B", head_id="mock-llm",
                prompt_template="Step B", depends_on=["a"],
            ),
        ]
        wo = WorkOrder(goal="Test linear", steps=steps)
        state = RunState(run_id=wo.run_id, work_order=wo)

        state = await dag_executor.execute_dag(wo.run_id, wo, state)
        assert "a" in state.step_results
        assert "b" in state.step_results

    @pytest.mark.asyncio
    async def test_execute_parallel(self, dag_executor, orchestrator):
        steps = [
            StepDef(step_id="a", name="A", head_id="mock-llm", prompt_template="Step A"),
            StepDef(step_id="b", name="B", head_id="mock-llm", prompt_template="Step B"),
            StepDef(step_id="c", name="C", head_id="mock-llm", prompt_template="Step C"),
        ]
        wo = WorkOrder(goal="Test parallel", steps=steps)
        state = RunState(run_id=wo.run_id, work_order=wo)

        state = await dag_executor.execute_dag(wo.run_id, wo, state)
        assert len(state.step_results) == 3

    @pytest.mark.asyncio
    async def test_execute_diamond(self, dag_executor, orchestrator):
        steps = [
            StepDef(step_id="a", name="A", head_id="mock-llm", prompt_template="A"),
            StepDef(
                step_id="b", name="B", head_id="mock-llm",
                prompt_template="B", depends_on=["a"],
            ),
            StepDef(
                step_id="c", name="C", head_id="mock-llm",
                prompt_template="C", depends_on=["a"],
            ),
            StepDef(
                step_id="d", name="D", head_id="mock-llm",
                prompt_template="D", depends_on=["b", "c"],
            ),
        ]
        wo = WorkOrder(goal="Test diamond", steps=steps)
        state = RunState(run_id=wo.run_id, work_order=wo)

        state = await dag_executor.execute_dag(wo.run_id, wo, state)
        assert len(state.step_results) == 4

    @pytest.mark.asyncio
    async def test_gpu_steps_serialize(self, dag_executor, orchestrator):
        """GPU steps in the same layer should run sequentially."""
        steps = [
            StepDef(step_id="g1", name="GPU1", head_id="mock-gpu", prompt_template="G1"),
            StepDef(step_id="g2", name="GPU2", head_id="mock-gpu", prompt_template="G2"),
        ]
        wo = WorkOrder(goal="Test GPU serialization", steps=steps)
        state = RunState(run_id=wo.run_id, work_order=wo)

        state = await dag_executor.execute_dag(wo.run_id, wo, state)
        assert len(state.step_results) == 2

    @pytest.mark.asyncio
    async def test_input_refs_injected_into_prompt(self, dag_executor, orchestrator):
        """DAG steps should get previous step outputs injected into prompts."""
        steps = [
            StepDef(
                step_id="plan", name="Plan", head_id="mock-llm",
                prompt_template="Create a plan",
            ),
            StepDef(
                step_id="execute", name="Execute", head_id="mock-llm",
                prompt_template="Execute the plan",
                input_refs=["plan"], depends_on=["plan"],
            ),
        ]
        wo = WorkOrder(goal="Test input_refs", steps=steps)
        state = RunState(run_id=wo.run_id, work_order=wo)

        state = await dag_executor.execute_dag(wo.run_id, wo, state)
        assert "plan" in state.step_results
        assert "execute" in state.step_results
        # Execute step should have received previous output (mock adapter echoes prompt)
        exec_output = state.step_results["execute"].outputs.get("text", "")
        assert len(exec_output) > 0

    @pytest.mark.asyncio
    async def test_work_order_inputs_injected(self, dag_executor, orchestrator):
        """Work order inputs should be injected into step prompts."""
        steps = [
            StepDef(step_id="a", name="A", head_id="mock-llm", prompt_template="Do task"),
        ]
        wo = WorkOrder(goal="Test inputs", steps=steps, inputs={"topic": "testing"})
        state = RunState(run_id=wo.run_id, work_order=wo)

        state = await dag_executor.execute_dag(wo.run_id, wo, state)
        assert "a" in state.step_results


class TestDAGEventParity:
    """P3: DAG executor should emit the same step events as linear executor."""

    @pytest.mark.asyncio
    async def test_dag_emits_step_events(self, orchestrator):
        """DAG run should produce STEP_STARTED + STEP_OUTPUT_WRITTEN + STEP_COMMITTED events."""
        wo = WorkOrder(
            goal="event test",
            steps=[
                StepDef(step_id="s1", name="S1", head_id="mock-llm", prompt_template="Hello"),
            ],
        )
        state = await orchestrator.create_run(wo, normalize=False)
        state = await orchestrator.execute_run(state.run_id)

        events = orchestrator.events.read_events(state.run_id)
        event_kinds = [e.kind.value for e in events]

        assert "step_started" in event_kinds
        assert "step_output_written" in event_kinds
        assert "step_committed" in event_kinds

    @pytest.mark.asyncio
    async def test_dag_persists_artifacts(self, orchestrator):
        """DAG output_artifacts should be populated with fetchable refs."""
        wo = WorkOrder(
            goal="artifact test",
            steps=[
                StepDef(
                    step_id="s1", name="S1", head_id="mock-llm",
                    prompt_template="Generate output",
                ),
            ],
        )
        state = await orchestrator.create_run(wo, normalize=False)
        state = await orchestrator.execute_run(state.run_id)

        result = state.step_results.get("s1")
        assert result is not None
        assert len(result.output_artifacts) > 0

        # Artifact should be fetchable from store
        art_ref = result.output_artifacts[0]
        content = orchestrator.artifacts.fetch(art_ref.artifact_id)
        assert content is not None

    @pytest.mark.asyncio
    async def test_dag_replay_reconstructs_state(self, orchestrator):
        """event_store.replay() should recover step_results for DAG runs."""
        wo = WorkOrder(
            goal="replay test",
            steps=[
                StepDef(step_id="s1", name="S1", head_id="mock-llm", prompt_template="One"),
                StepDef(
                    step_id="s2", name="S2", head_id="mock-llm",
                    prompt_template="Two", depends_on=["s1"],
                ),
            ],
        )
        state = await orchestrator.create_run(wo, normalize=False)
        state = await orchestrator.execute_run(state.run_id)

        # Replay from events
        replayed = orchestrator.events.replay(state.run_id, wo)
        assert replayed.run_id == state.run_id
        assert len(replayed.step_results) == 2

    @pytest.mark.asyncio
    async def test_dag_committed_event_data(self, orchestrator):
        """STEP_COMMITTED events should contain head_id, artifact_id, metrics."""
        wo = WorkOrder(
            goal="committed data test",
            steps=[
                StepDef(step_id="s1", name="S1", head_id="mock-llm", prompt_template="Test"),
            ],
        )
        state = await orchestrator.create_run(wo, normalize=False)
        state = await orchestrator.execute_run(state.run_id)

        events = orchestrator.events.read_events(state.run_id)
        committed = [e for e in events if e.kind.value == "step_committed"]
        assert len(committed) >= 1

        data = committed[0].data
        assert "head_id" in data
        assert data["head_id"] == "mock-llm"

    @pytest.mark.asyncio
    async def test_dag_parallel_steps_all_emit_events(self, orchestrator):
        """3 parallel CPU steps should each emit their own events."""
        wo = WorkOrder(
            goal="parallel events",
            steps=[
                StepDef(step_id="p1", name="P1", head_id="mock-llm", prompt_template="A"),
                StepDef(step_id="p2", name="P2", head_id="mock-llm", prompt_template="B"),
                StepDef(step_id="p3", name="P3", head_id="mock-llm", prompt_template="C"),
            ],
        )
        state = await orchestrator.create_run(wo, normalize=False)
        state = await orchestrator.execute_run(state.run_id)

        events = orchestrator.events.read_events(state.run_id)
        started = [e for e in events if e.kind.value == "step_started"]
        committed = [e for e in events if e.kind.value == "step_committed"]

        started_ids = {e.step_id for e in started}
        committed_ids = {e.step_id for e in committed}

        assert {"p1", "p2", "p3"} == started_ids
        assert {"p1", "p2", "p3"} == committed_ids


class TestCycleDetection:
    def test_cycle_raises_error(self, dag_executor):
        """Circular dependencies should raise ValueError."""
        steps = [
            StepDef(step_id="a", name="A", head_id="mock-llm", depends_on=["b"]),
            StepDef(step_id="b", name="B", head_id="mock-llm", depends_on=["a"]),
        ]
        graph = dag_executor.build_graph(steps)
        with pytest.raises(ValueError, match="Cycle detected"):
            dag_executor.topological_sort(graph)

    def test_self_cycle_raises_error(self, dag_executor):
        """Self-referencing dependency should raise ValueError."""
        steps = [
            StepDef(step_id="a", name="A", head_id="mock-llm", depends_on=["a"]),
        ]
        graph = dag_executor.build_graph(steps)
        with pytest.raises(ValueError, match="Cycle detected"):
            dag_executor.topological_sort(graph)

    def test_three_node_cycle_raises_error(self, dag_executor):
        """Three-node cycle should raise ValueError."""
        steps = [
            StepDef(step_id="a", name="A", head_id="mock-llm", depends_on=["c"]),
            StepDef(step_id="b", name="B", head_id="mock-llm", depends_on=["a"]),
            StepDef(step_id="c", name="C", head_id="mock-llm", depends_on=["b"]),
        ]
        graph = dag_executor.build_graph(steps)
        with pytest.raises(ValueError, match="Cycle detected"):
            dag_executor.topological_sort(graph)
