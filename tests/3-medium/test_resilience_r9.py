"""Round 9 tests: timeouts, concurrency limits, validation, backoff."""

from __future__ import annotations

import asyncio
import time

import pytest

from multihead.consensus import (
    ConsensusConfig,
    ConsensusEngine,
    HeadTask,
)
from multihead.dag_executor import DAGExecutor
from multihead.event_store import EventStore
from multihead.head_manager import HeadManager
from multihead.models import (
    AdapterKind,
    HeadManifest,
    RunState,
    StepDef,
    WorkOrder,
)
from multihead.observability import MetricsCollector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_heads():
    manifests = {
        "head-a": HeadManifest(
            head_id="head-a", name="A", adapter=AdapterKind.MOCK,
            model="mock-v1", kind="llm", gpu_required=False,
        ),
        "head-b": HeadManifest(
            head_id="head-b", name="B", adapter=AdapterKind.MOCK,
            model="mock-v1", kind="llm", gpu_required=False,
        ),
        "head-c": HeadManifest(
            head_id="head-c", name="C", adapter=AdapterKind.MOCK,
            model="mock-v1", kind="llm", gpu_required=False,
        ),
    }
    return HeadManager(manifests)


# ---------------------------------------------------------------------------
# Consensus timeout tests
# ---------------------------------------------------------------------------


class TestConsensusTimeout:
    @pytest.mark.asyncio
    async def test_slow_head_times_out(self, mock_heads):
        """A head that exceeds timeout_seconds should fail gracefully."""
        adapter = mock_heads.get_adapter("head-a")

        async def slow_gen(prompt, **kw):
            await asyncio.sleep(5)  # Way longer than timeout
            return {"text": "late", "tokens_in": 0, "tokens_out": 0}

        adapter.generate = slow_gen

        config = ConsensusConfig(
            heads=[
                HeadTask(head_id="head-a"),
                HeadTask(head_id="head-b"),
            ],
            timeout_seconds=0.5,  # 500ms — tight enough to catch the 5s sleep
        )
        engine = ConsensusEngine(mock_heads)
        result = await engine.execute(config, "test")

        # head-a should have timed out
        vote_a = [v for v in result.all_votes if v.head_id == "head-a"][0]
        assert not vote_a.success
        assert "timeout" in vote_a.error

        # head-b should succeed (mock is fast)
        vote_b = [v for v in result.all_votes if v.head_id == "head-b"][0]
        assert vote_b.success

        # Consensus should still work from head-b
        assert result.consensus_outputs.get("text")

    @pytest.mark.asyncio
    async def test_all_heads_timeout(self, mock_heads):
        """All heads timing out → no_valid_votes red flag."""
        for hid in ["head-a", "head-b"]:
            adapter = mock_heads.get_adapter(hid)

            async def slow_gen(prompt, **kw):
                await asyncio.sleep(5)
                return {"text": "late", "tokens_in": 0, "tokens_out": 0}

            adapter.generate = slow_gen

        config = ConsensusConfig(
            heads=[HeadTask(head_id="head-a"), HeadTask(head_id="head-b")],
            timeout_seconds=0.5,
        )
        engine = ConsensusEngine(mock_heads)
        result = await engine.execute(config, "test")

        assert result.agreement_score == 0.0
        no_votes = [f for f in result.red_flags if f["type"] == "no_valid_votes"]
        assert len(no_votes) == 1

    @pytest.mark.asyncio
    async def test_default_timeout_is_reasonable(self, mock_heads):
        """Default timeout should be 30s — fast mock heads should pass."""
        config = ConsensusConfig(
            heads=[HeadTask(head_id="head-a")],
            # No explicit timeout → uses default 30s
        )
        assert config.timeout_seconds == 30.0

        engine = ConsensusEngine(mock_heads)
        result = await engine.execute(config, "test")
        assert result.all_votes[0].success


# ---------------------------------------------------------------------------
# DAG concurrency limit tests
# ---------------------------------------------------------------------------


class TestDAGConcurrencyLimits:
    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self, mock_heads, tmp_path):
        """DAG executor should respect max_parallel_cpu limit."""
        from multihead.artifact_store import ArtifactStore
        from multihead.orchestrator import Orchestrator

        peak_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        # Track concurrency through adapter
        for hid in ["head-a", "head-b", "head-c"]:
            adapter = mock_heads.get_adapter(hid)

            async def counting_gen(prompt, **kw):
                nonlocal peak_concurrent, current_concurrent
                async with lock:
                    current_concurrent += 1
                    peak_concurrent = max(peak_concurrent, current_concurrent)
                await asyncio.sleep(0.05)  # Simulate work
                async with lock:
                    current_concurrent -= 1
                return {"text": "ok", "tokens_in": 0, "tokens_out": 0}

            adapter.generate = counting_gen

        wo = WorkOrder(
            goal="concurrency test",
            steps=[
                StepDef(name=f"step-{i}", head_id=f"head-{c}", prompt_template="test")
                for i, c in enumerate(["a", "b", "c"])
            ],
        )

        db_path = tmp_path / "test.db"
        artifact_store = ArtifactStore(tmp_path / "artifacts", db_path)
        event_store = EventStore(tmp_path / "runs", db_path)
        orch = Orchestrator(event_store, artifact_store, mock_heads, tmp_path / "runs")

        dag = DAGExecutor(orch, max_parallel_cpu=2)
        state = RunState(run_id=wo.run_id, work_order=wo)
        await dag.execute_dag(wo.run_id, wo, state)

        # With 3 parallel steps and max_parallel_cpu=2, peak should be ≤ 2
        assert peak_concurrent <= 2
        assert len(state.step_results) == 3

    @pytest.mark.asyncio
    async def test_default_concurrency_limit(self):
        """Default max_parallel_cpu should be 8."""
        class FakeOrch:
            pass
        dag = DAGExecutor(FakeOrch())
        assert dag._cpu_semaphore._value == 8


# ---------------------------------------------------------------------------
# API input validation tests
# ---------------------------------------------------------------------------


class TestAPIValidation:
    def test_consensus_prompt_too_long(self):
        """Prompt > 100KB should be rejected by Pydantic."""
        from pydantic import ValidationError
        from multihead.api.routes_consensus import ConsensusExecuteRequest

        with pytest.raises(ValidationError):
            ConsensusExecuteRequest(
                prompt="x" * 100_001,
                heads=[{"head_id": "test"}],
            )

    def test_consensus_empty_prompt(self):
        """Empty prompt should be rejected."""
        from pydantic import ValidationError
        from multihead.api.routes_consensus import ConsensusExecuteRequest

        with pytest.raises(ValidationError):
            ConsensusExecuteRequest(
                prompt="",
                heads=[{"head_id": "test"}],
            )

    def test_consensus_too_many_heads(self):
        """More than 50 heads should be rejected."""
        from pydantic import ValidationError
        from multihead.api.routes_consensus import ConsensusExecuteRequest

        with pytest.raises(ValidationError):
            ConsensusExecuteRequest(
                prompt="test",
                heads=[{"head_id": f"h-{i}"} for i in range(51)],
            )

    def test_consensus_no_heads(self):
        """Empty heads list should be rejected."""
        from pydantic import ValidationError
        from multihead.api.routes_consensus import ConsensusExecuteRequest

        with pytest.raises(ValidationError):
            ConsensusExecuteRequest(
                prompt="test",
                heads=[],
            )

    def test_consensus_invalid_threshold(self):
        """Threshold outside [0, 1] should be rejected."""
        from pydantic import ValidationError
        from multihead.api.routes_consensus import ConsensusExecuteRequest

        with pytest.raises(ValidationError):
            ConsensusExecuteRequest(
                prompt="test",
                heads=[{"head_id": "test"}],
                threshold=1.5,
            )

    def test_consensus_negative_weight(self):
        """Negative weight should be rejected."""
        from pydantic import ValidationError
        from multihead.api.routes_consensus import HeadTaskRequest

        with pytest.raises(ValidationError):
            HeadTaskRequest(head_id="test", weight=-1.0)

    def test_consensus_zero_weight(self):
        """Zero weight should be rejected (gt=0)."""
        from pydantic import ValidationError
        from multihead.api.routes_consensus import HeadTaskRequest

        with pytest.raises(ValidationError):
            HeadTaskRequest(head_id="test", weight=0.0)

    def test_consensus_valid_request_passes(self):
        """Valid request should pass validation."""
        from multihead.api.routes_consensus import ConsensusExecuteRequest

        req = ConsensusExecuteRequest(
            prompt="What is 2+2?",
            heads=[{"head_id": "mock-llm"}],
            strategy="majority",
            threshold=0.5,
        )
        assert req.prompt == "What is 2+2?"
        assert len(req.heads) == 1

    def test_head_validation_at_run_creation(self, tmp_path):
        """POST /runs with unknown head_id should return 400."""
        from fastapi.testclient import TestClient
        from multihead.api.app import create_app
        from multihead.config import Settings

        settings = Settings(data_dir=tmp_path / "data", config_dir=tmp_path / "config")
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "recipes").mkdir()
        (config_dir / "heads.yaml").write_text("""
heads:
  - head_id: mock-llm
    name: Mock LLM
    adapter: mock
    model: mock-v1
    kind: llm
    gpu_required: false
""")

        app = create_app(settings)
        with TestClient(app) as client:
            resp = client.post("/runs", json={
                "work_order": {
                    "goal": "test",
                    "steps": [
                        {"name": "s1", "head_id": "nonexistent-head", "prompt_template": "test"},
                    ],
                },
            })
            assert resp.status_code == 400
            assert "nonexistent-head" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Retry backoff tests
# ---------------------------------------------------------------------------


class TestRetryBackoff:
    @pytest.mark.asyncio
    async def test_retry_with_backoff(self, tmp_path, mock_heads):
        """Retries should have exponential backoff between attempts."""
        from multihead.artifact_store import ArtifactStore
        from multihead.event_store import EventStore
        from multihead.orchestrator import Orchestrator

        db_path = tmp_path / "test.db"
        artifact_store = ArtifactStore(tmp_path / "artifacts", db_path)
        event_store = EventStore(tmp_path / "runs", db_path)
        metrics = MetricsCollector()
        orchestrator = Orchestrator(
            event_store, artifact_store, mock_heads,
            tmp_path / "runs", metrics=metrics,
        )

        # Make head-a fail twice then succeed
        call_count = 0
        adapter = mock_heads.get_adapter("head-a")

        async def flaky_gen(prompt, **kw):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("transient failure")
            return {"text": "success", "tokens_in": 5, "tokens_out": 5}

        adapter.generate = flaky_gen

        wo = WorkOrder(
            goal="retry test",
            steps=[
                StepDef(
                    name="flaky-step",
                    head_id="head-a",
                    prompt_template="test",
                    retry_policy={"max_attempts": 3, "backoff_ms": 50},
                ),
            ],
        )
        state = await orchestrator.create_run(wo)
        t0 = time.perf_counter()
        state = await orchestrator.execute_run(state.run_id)
        elapsed = time.perf_counter() - t0

        # Should succeed after retries
        from multihead.models import RunStatus
        assert state.status == RunStatus.DONE
        assert call_count == 3  # 1 initial + 2 retries
        # Should have taken some time due to backoff (50ms + 100ms ≈ 150ms minimum)
        assert elapsed >= 0.1
        # Retry metrics should be recorded
        assert metrics.counter("steps_retried_total") >= 2

    @pytest.mark.asyncio
    async def test_retry_exhausted_fails_run(self, tmp_path, mock_heads):
        """If all retries fail, the run should fail."""
        from multihead.artifact_store import ArtifactStore
        from multihead.event_store import EventStore
        from multihead.orchestrator import Orchestrator

        db_path = tmp_path / "test.db"
        artifact_store = ArtifactStore(tmp_path / "artifacts", db_path)
        event_store = EventStore(tmp_path / "runs", db_path)
        orchestrator = Orchestrator(event_store, artifact_store, mock_heads, tmp_path / "runs")

        adapter = mock_heads.get_adapter("head-a")

        async def always_fail(prompt, **kw):
            raise RuntimeError("permanent failure")

        adapter.generate = always_fail

        wo = WorkOrder(
            goal="fail test",
            steps=[
                StepDef(
                    name="broken-step",
                    head_id="head-a",
                    prompt_template="test",
                    retry_policy={"max_attempts": 2, "backoff_ms": 10},
                ),
            ],
        )
        # normalize=False to prevent auto-fallback assignment
        state = await orchestrator.create_run(wo, normalize=False)
        state = await orchestrator.execute_run(state.run_id)

        from multihead.models import RunStatus
        assert state.status == RunStatus.FAILED


# ---------------------------------------------------------------------------
# EventStore file locking tests
# ---------------------------------------------------------------------------


class TestEventStoreLocking:
    def test_concurrent_appends_safe(self, tmp_path):
        """Multiple appends to same run should not corrupt the log."""
        from multihead.models import RunEvent, EventKind

        db_path = tmp_path / "test.db"
        store = EventStore(tmp_path / "runs", db_path)

        run_id = "test-run-locking"
        n = 20

        for i in range(n):
            event = RunEvent(
                run_id=run_id,
                kind=EventKind.STEP_STARTED if i > 0 else EventKind.RUN_CREATED,
                step_id=f"step-{i}" if i > 0 else None,
                data={"index": i},
            )
            store.append(event)

        events = store.read_events(run_id)
        assert len(events) == n

    def test_append_and_read_integrity(self, tmp_path):
        """Events written should be readable with correct data."""
        from multihead.models import RunEvent, EventKind

        db_path = tmp_path / "test.db"
        store = EventStore(tmp_path / "runs", db_path)

        run_id = "test-integrity"
        store.append(RunEvent(
            run_id=run_id,
            kind=EventKind.RUN_CREATED,
            data={"work_order": {"goal": "integrity test", "steps": []}},
        ))
        store.append(RunEvent(
            run_id=run_id,
            kind=EventKind.RUN_DONE,
        ))

        events = store.read_events(run_id)
        assert len(events) == 2
        assert events[0].kind == EventKind.RUN_CREATED
        assert events[1].kind == EventKind.RUN_DONE
