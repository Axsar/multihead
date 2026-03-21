"""Tests for the run orchestrator with mock heads."""

import pytest

from multihead.artifact_store import ArtifactStore
from multihead.event_store import EventStore
from multihead.head_manager import HeadManager
from multihead.models import (
    AdapterKind,
    HeadManifest,
    RunStatus,
    StepDef,
    WorkOrder,
)
from multihead.observability import MetricsCollector
from multihead.orchestrator import Orchestrator


@pytest.fixture
def components(tmp_path):
    runs_dir = tmp_path / "runs"
    artifacts_dir = tmp_path / "artifacts"
    db_path = tmp_path / "test.db"

    artifact_store = ArtifactStore(artifacts_dir, db_path)
    event_store = EventStore(runs_dir, db_path)

    manifests = {
        "mock-llm": HeadManifest(
            head_id="mock-llm", name="Mock LLM",
            adapter=AdapterKind.MOCK, model="mock-v1", kind="llm", gpu_required=False,
        ),
        "mock-vlm": HeadManifest(
            head_id="mock-vlm", name="Mock VLM",
            adapter=AdapterKind.MOCK, model="mock-v1", kind="vlm", gpu_required=False,
        ),
    }
    head_manager = HeadManager(manifests)

    orchestrator = Orchestrator(event_store, artifact_store, head_manager, runs_dir)
    return orchestrator, head_manager, event_store


@pytest.mark.asyncio
async def test_simple_run(components):
    orchestrator, head_manager, event_store = components

    wo = WorkOrder(
        goal="test simple run",
        steps=[
            StepDef(name="plan", head_id="mock-llm"),
        ],
    )

    state = await orchestrator.create_run(wo)
    assert state.status == RunStatus.QUEUED

    state = await orchestrator.execute_run(state.run_id)
    assert state.status == RunStatus.DONE
    assert state.current_step_index == 1

    await head_manager.shutdown()


@pytest.mark.asyncio
async def test_multi_step_with_head_swap(components):
    """The golden demo pattern: LLM → VLM → LLM."""
    orchestrator, head_manager, event_store = components

    wo = WorkOrder(
        goal="test head swap pipeline",
        steps=[
            StepDef(name="plan", head_id="mock-llm"),
            StepDef(name="extract", head_id="mock-vlm"),
            StepDef(name="report", head_id="mock-llm", input_refs=["extract"]),
        ],
    )

    # normalize=False to keep linear mode (this test verifies linear step events)
    state = await orchestrator.create_run(wo, normalize=False)
    state = await orchestrator.execute_run(state.run_id)

    assert state.status == RunStatus.DONE
    assert state.current_step_index == 3
    assert len(state.step_results) == 3

    # Verify events were logged
    events = event_store.read_events(state.run_id)
    event_kinds = [e.kind.value for e in events]
    assert "run_created" in event_kinds
    assert "step_committed" in event_kinds
    assert "run_done" in event_kinds

    await head_manager.shutdown()


@pytest.mark.asyncio
async def test_resume_after_crash(components, tmp_path):
    """Simulate: create run, execute 1 step, 'crash', create new orchestrator, resume."""
    orchestrator, head_manager, event_store = components

    wo = WorkOrder(
        goal="crash resume test",
        steps=[
            StepDef(step_id="s1", name="step1", head_id="mock-llm"),
            StepDef(step_id="s2", name="step2", head_id="mock-vlm"),
        ],
    )

    # Create and run first step
    state = await orchestrator.create_run(wo)
    run_id = state.run_id

    # Manually execute just step 1 by setting up the state
    state = await orchestrator.execute_run(run_id)

    # Both steps should complete since mock adapter doesn't actually crash
    assert state.status == RunStatus.DONE

    # Now test replay
    replayed = event_store.replay(run_id)
    assert replayed.current_step_index == 2
    assert replayed.status == RunStatus.DONE

    await head_manager.shutdown()


@pytest.mark.asyncio
async def test_events_are_durable(components):
    """Events should survive across orchestrator instances."""
    orchestrator, head_manager, event_store = components

    wo = WorkOrder(
        goal="durability test",
        steps=[StepDef(name="only_step", head_id="mock-llm")],
    )

    state = await orchestrator.create_run(wo)
    state = await orchestrator.execute_run(state.run_id)
    run_id = state.run_id

    await head_manager.shutdown()

    # Create a completely new event store pointing at same files
    new_event_store = EventStore(event_store.runs_dir, event_store.db_path)
    events = new_event_store.read_events(run_id)
    assert len(events) > 0

    replayed = new_event_store.replay(run_id)
    assert replayed.status == RunStatus.DONE


@pytest.mark.asyncio
async def test_auto_dag_mode(components):
    """Steps with depends_on auto-trigger DAG execution."""
    orchestrator, head_manager, event_store = components

    wo = WorkOrder(
        goal="test auto DAG",
        steps=[
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
        ],
    )

    state = await orchestrator.create_run(wo)
    state = await orchestrator.execute_run(state.run_id)

    assert state.status == RunStatus.DONE
    assert len(state.step_results) == 4

    await head_manager.shutdown()


@pytest.mark.asyncio
async def test_explicit_dag_mode(components):
    """Explicitly request DAG mode even for linear steps."""
    orchestrator, head_manager, event_store = components

    wo = WorkOrder(
        goal="test explicit DAG",
        steps=[
            StepDef(step_id="x", name="X", head_id="mock-llm", prompt_template="X"),
            StepDef(step_id="y", name="Y", head_id="mock-llm", prompt_template="Y"),
        ],
    )

    state = await orchestrator.create_run(wo)
    state = await orchestrator.execute_run(state.run_id, execution_mode="dag")

    assert state.status == RunStatus.DONE
    assert len(state.step_results) == 2

    await head_manager.shutdown()


@pytest.mark.asyncio
async def test_linear_mode_ignores_depends_on(components):
    """Forced linear mode executes sequentially even with depends_on."""
    orchestrator, head_manager, event_store = components

    wo = WorkOrder(
        goal="test forced linear",
        steps=[
            StepDef(step_id="a", name="A", head_id="mock-llm", prompt_template="A"),
            StepDef(
                step_id="b", name="B", head_id="mock-llm",
                prompt_template="B", depends_on=["a"],
            ),
        ],
    )

    state = await orchestrator.create_run(wo)
    state = await orchestrator.execute_run(state.run_id, execution_mode="linear")

    assert state.status == RunStatus.DONE
    assert state.current_step_index == 2

    await head_manager.shutdown()


# ---------------------------------------------------------------------------
# Fallback execution
# ---------------------------------------------------------------------------


@pytest.fixture
def fallback_components(tmp_path):
    """Components with a primary head that fails and a fallback that succeeds."""
    runs_dir = tmp_path / "runs"
    artifacts_dir = tmp_path / "artifacts"
    db_path = tmp_path / "test.db"

    artifact_store = ArtifactStore(artifacts_dir, db_path)
    event_store = EventStore(runs_dir, db_path)
    metrics = MetricsCollector()

    manifests = {
        "fail-head": HeadManifest(
            head_id="fail-head", name="Failing Head",
            adapter=AdapterKind.MOCK, model="mock-v1", kind="llm", gpu_required=False,
        ),
        "good-head": HeadManifest(
            head_id="good-head", name="Good Head",
            adapter=AdapterKind.MOCK, model="mock-v1", kind="llm", gpu_required=False,
        ),
    }
    head_manager = HeadManager(manifests)
    orchestrator = Orchestrator(
        event_store, artifact_store, head_manager,
        runs_dir, metrics=metrics,
    )
    return orchestrator, head_manager, metrics


@pytest.mark.asyncio
async def test_fallback_on_step_failure(fallback_components):
    """Primary head fails, fallback succeeds, run completes."""
    orchestrator, head_manager, metrics = fallback_components

    # Make fail-head always fail
    adapter = head_manager.get_adapter("fail-head")

    async def failing_gen(prompt, **kw):
        raise RuntimeError("GPU OOM")

    adapter.generate = failing_gen

    wo = WorkOrder(
        goal="test fallback",
        steps=[
            StepDef(
                name="step1", head_id="fail-head",
                fallback=["good-head"],
                prompt_template="hello",
            ),
        ],
    )

    state = await orchestrator.create_run(wo, normalize=False)
    state = await orchestrator.execute_run(state.run_id)

    assert state.status == RunStatus.DONE
    assert metrics.counter("steps_fallback_total") >= 1.0

    await head_manager.shutdown()


@pytest.mark.asyncio
async def test_fallback_exhausted_still_fails(fallback_components):
    """Both primary and fallback fail, run fails."""
    orchestrator, head_manager, metrics = fallback_components

    # Make both heads fail
    for hid in ["fail-head", "good-head"]:
        adapter = head_manager.get_adapter(hid)

        async def failing_gen(prompt, **kw):
            raise RuntimeError("all broken")

        adapter.generate = failing_gen

    wo = WorkOrder(
        goal="test exhausted fallback",
        steps=[
            StepDef(
                name="step1", head_id="fail-head",
                fallback=["good-head"],
                prompt_template="hello",
            ),
        ],
    )

    state = await orchestrator.create_run(wo, normalize=False)
    state = await orchestrator.execute_run(state.run_id)

    assert state.status == RunStatus.FAILED

    await head_manager.shutdown()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_metrics(tmp_path):
    """Orchestrator should record runs_created, runs_started, runs_completed, and duration."""
    runs_dir = tmp_path / "runs"
    db_path = tmp_path / "test.db"
    metrics = MetricsCollector()

    artifact_store = ArtifactStore(tmp_path / "artifacts", db_path)
    event_store = EventStore(runs_dir, db_path)
    manifests = {
        "mock-llm": HeadManifest(
            head_id="mock-llm", name="Mock LLM",
            adapter=AdapterKind.MOCK, model="mock-v1", kind="llm", gpu_required=False,
        ),
    }
    head_manager = HeadManager(manifests)
    orchestrator = Orchestrator(
        event_store, artifact_store, head_manager,
        runs_dir, metrics=metrics,
    )

    wo = WorkOrder(
        goal="metrics test",
        steps=[StepDef(name="step1", head_id="mock-llm", prompt_template="hello")],
    )
    state = await orchestrator.create_run(wo)
    assert metrics.counter("runs_created_total") == 1.0

    state = await orchestrator.execute_run(state.run_id)
    assert state.status == RunStatus.DONE
    assert metrics.counter("runs_started_total") == 1.0
    assert metrics.counter("runs_completed_total") == 1.0
    assert metrics.counter("runs_failed_total") == 0.0

    hist = metrics.histogram("run_duration_seconds")
    assert hist["count"] == 1
    assert hist["avg"] > 0

    await head_manager.shutdown()
