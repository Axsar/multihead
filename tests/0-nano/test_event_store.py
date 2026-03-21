"""Tests for the event-sourced state store."""

import pytest

from multihead.event_store import EventStore
from multihead.models import (
    EventKind,
    RunEvent,
    RunStatus,
    WorkOrder,
    StepDef,
)


@pytest.fixture
def store(tmp_path):
    runs_dir = tmp_path / "runs"
    db_path = tmp_path / "test.db"
    return EventStore(runs_dir, db_path)


def test_append_and_read(store):
    event = RunEvent(run_id="run_test1", kind=EventKind.RUN_CREATED, data={"goal": "test"})
    store.append(event)

    events = store.read_events("run_test1")
    assert len(events) == 1
    assert events[0].kind == EventKind.RUN_CREATED


def test_replay_empty(store):
    state = store.replay("nonexistent")
    assert state.status == RunStatus.QUEUED
    assert state.work_order is None


def test_replay_full_lifecycle(store):
    run_id = "run_lifecycle"
    wo = WorkOrder(
        run_id=run_id,
        goal="test pipeline",
        steps=[
            StepDef(step_id="step_1", name="plan", head_id="mock-llm"),
            StepDef(step_id="step_2", name="extract", head_id="mock-vlm"),
        ],
    )

    # Create
    store.append(RunEvent(run_id=run_id, kind=EventKind.RUN_CREATED,
                          data={"work_order": wo.model_dump(mode="json")}))

    # Step 1
    store.append(RunEvent(run_id=run_id, kind=EventKind.STEP_STARTED, step_id="step_1"))
    store.append(RunEvent(run_id=run_id, kind=EventKind.STEP_COMMITTED, step_id="step_1",
                          data={"head_id": "mock-llm", "outputs": {"text": "plan output"}}))

    # Step 2
    store.append(RunEvent(run_id=run_id, kind=EventKind.STEP_STARTED, step_id="step_2"))
    store.append(RunEvent(run_id=run_id, kind=EventKind.STEP_COMMITTED, step_id="step_2",
                          data={"head_id": "mock-vlm", "outputs": {"text": "vlm output"}}))

    # Done
    store.append(RunEvent(run_id=run_id, kind=EventKind.RUN_DONE))

    state = store.replay(run_id)
    assert state.status == RunStatus.DONE
    assert state.current_step_index == 2
    assert "step_1" in state.step_results
    assert "step_2" in state.step_results
    assert state.work_order is not None
    assert state.work_order.goal == "test pipeline"


def test_replay_resume_from_crash(store):
    """Simulate crash after step 1 committed but before step 2 starts."""
    run_id = "run_crash"
    wo = WorkOrder(
        run_id=run_id,
        goal="crash test",
        steps=[
            StepDef(step_id="s1", name="step1", head_id="mock-llm"),
            StepDef(step_id="s2", name="step2", head_id="mock-vlm"),
            StepDef(step_id="s3", name="step3", head_id="mock-llm"),
        ],
    )

    store.append(RunEvent(run_id=run_id, kind=EventKind.RUN_CREATED,
                          data={"work_order": wo.model_dump(mode="json")}))
    store.append(RunEvent(run_id=run_id, kind=EventKind.STEP_STARTED, step_id="s1"))
    store.append(RunEvent(run_id=run_id, kind=EventKind.STEP_COMMITTED, step_id="s1",
                          data={"head_id": "mock-llm"}))
    # "crash" here - no more events

    state = store.replay(run_id)
    assert state.current_step_index == 1  # Should resume from step 2 (index 1)
    assert state.status == RunStatus.RUNNING  # Was running when crashed
    assert "s1" in state.step_results


def test_list_runs(store):
    store.append(RunEvent(run_id="run_a", kind=EventKind.RUN_CREATED))
    store.append(RunEvent(run_id="run_b", kind=EventKind.RUN_CREATED))
    store.append(RunEvent(run_id="run_a", kind=EventKind.RUN_DONE))

    runs = store.list_runs()
    assert len(runs) == 2
    run_a = next(r for r in runs if r["run_id"] == "run_a")
    assert run_a["status"] == "done"
