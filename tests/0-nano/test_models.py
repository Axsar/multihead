"""Tests for core data models."""

from datetime import datetime

from multihead.models import (
    AdapterKind,
    ArtifactRef,
    CheckpointMode,
    EventKind,
    HeadManifest,
    HeadState,
    RunEvent,
    RunState,
    RunStatus,
    StageResult,
    StepDef,
    StepStatus,
    WorkOrder,
    new_id,
)


class TestNewId:
    def test_generates_string(self):
        uid = new_id()
        assert isinstance(uid, str)
        assert len(uid) > 10

    def test_prefix(self):
        uid = new_id("run_")
        assert uid.startswith("run_")

    def test_unique(self):
        ids = {new_id() for _ in range(100)}
        assert len(ids) == 100


class TestEnums:
    def test_head_state_values(self):
        assert HeadState.OFF.value == "off"
        assert HeadState.ACTIVE.value == "active"
        assert HeadState.ERROR.value == "error"

    def test_run_status_values(self):
        assert RunStatus.QUEUED.value == "queued"
        assert RunStatus.DONE.value == "done"

    def test_event_kind_values(self):
        assert EventKind.RUN_CREATED.value == "run_created"
        assert EventKind.STEP_STARTED.value == "step_started"

    def test_adapter_kind_values(self):
        assert AdapterKind.OLLAMA.value == "ollama"
        assert AdapterKind.MOCK.value == "mock"
        assert AdapterKind.EMBEDDING.value == "embedding"

    def test_step_status_values(self):
        assert StepStatus.COMMITTED.value == "committed"
        assert StepStatus.FAILED.value == "failed"


class TestHeadManifest:
    def test_required_fields(self):
        m = HeadManifest(head_id="h1", name="Test", adapter=AdapterKind.MOCK, model="m1")
        assert m.head_id == "h1"
        assert m.gpu_required is True  # default

    def test_optional_fields(self):
        m = HeadManifest(
            head_id="h2", name="Test", adapter=AdapterKind.OLLAMA,
            model="phi-3", endpoint="http://remote:11434",
            gpu_required=False, vram_hint_mb=4096, quantization="4bit",
        )
        assert m.endpoint == "http://remote:11434"
        assert m.quantization == "4bit"

    def test_serialization_roundtrip(self):
        m = HeadManifest(head_id="h3", name="RT", adapter=AdapterKind.VLLM, model="llama")
        data = m.model_dump(mode="json")
        restored = HeadManifest.model_validate(data)
        assert restored.head_id == "h3"
        assert restored.adapter == AdapterKind.VLLM


class TestStepDef:
    def test_auto_id(self):
        s = StepDef(name="plan", head_id="mock-llm")
        assert s.step_id.startswith("step_")

    def test_explicit_id(self):
        s = StepDef(step_id="my-step", name="plan", head_id="mock-llm")
        assert s.step_id == "my-step"

    def test_defaults(self):
        s = StepDef(name="plan", head_id="mock-llm")
        assert s.depends_on == []
        assert s.input_refs == []
        assert s.checkpoint_mode == CheckpointMode.SYNC
        assert s.retry_policy["max_attempts"] == 1


class TestWorkOrder:
    def test_auto_run_id(self):
        wo = WorkOrder(goal="test")
        assert wo.run_id.startswith("run_")

    def test_explicit_run_id(self):
        wo = WorkOrder(run_id="run_123", goal="test")
        assert wo.run_id == "run_123"

    def test_with_steps(self):
        wo = WorkOrder(
            goal="pipeline",
            steps=[
                StepDef(name="a", head_id="mock-llm"),
                StepDef(name="b", head_id="mock-vlm"),
            ],
        )
        assert len(wo.steps) == 2

    def test_created_at_set(self):
        wo = WorkOrder(goal="test")
        assert isinstance(wo.created_at, datetime)


class TestRunEvent:
    def test_auto_event_id(self):
        e = RunEvent(run_id="run_1", kind=EventKind.RUN_CREATED)
        assert e.event_id.startswith("evt_")

    def test_timestamp_set(self):
        e = RunEvent(run_id="run_1", kind=EventKind.RUN_DONE)
        assert isinstance(e.timestamp, datetime)

    def test_serialization(self):
        e = RunEvent(run_id="run_1", kind=EventKind.STEP_STARTED, step_id="s1", data={"x": 1})
        data = e.model_dump(mode="json")
        assert data["kind"] == "step_started"
        assert data["step_id"] == "s1"


class TestRunState:
    def test_defaults(self):
        s = RunState(run_id="run_1")
        assert s.status == RunStatus.QUEUED
        assert s.current_step_index == 0
        assert s.step_results == {}

    def test_with_work_order(self):
        wo = WorkOrder(goal="test")
        s = RunState(run_id=wo.run_id, work_order=wo)
        assert s.work_order is not None


class TestStageResult:
    def test_success(self):
        r = StageResult(step_id="s1", head_id="h1", status=StepStatus.COMMITTED)
        assert r.error is None
        assert r.outputs == {}

    def test_failure(self):
        r = StageResult(step_id="s1", head_id="h1", status=StepStatus.FAILED, error="boom")
        assert r.error == "boom"


class TestArtifactRef:
    def test_basic(self):
        ref = ArtifactRef(
            artifact_id="art_1", name="output.txt",
            media_type="text/plain", size_bytes=42,
        )
        assert ref.artifact_id == "art_1"
        assert ref.size_bytes == 42
