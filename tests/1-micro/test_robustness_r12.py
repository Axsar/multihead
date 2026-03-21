"""Round 12 tests: shutdown timeout, nightshift lock, DAG breaker, config validation."""

from __future__ import annotations

import asyncio

import pytest

from multihead.config import Settings
from multihead.head_manager import HeadManager
from multihead.models import (
    AdapterKind,
    HeadManifest,
    HeadState,
    StepDef,
    WorkOrder,
)


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
    }
    return HeadManager(manifests)


# ---------------------------------------------------------------------------
# Shutdown timeout tests
# ---------------------------------------------------------------------------


class TestShutdownTimeout:
    @pytest.mark.asyncio
    async def test_shutdown_completes_normally(self, mock_heads):
        """Normal shutdown should unload all heads."""
        await mock_heads.ensure_active("head-a")
        assert mock_heads.get_state("head-a") == HeadState.ACTIVE

        await mock_heads.shutdown()
        assert mock_heads.get_state("head-a") == HeadState.OFF
        assert mock_heads.active_head is None

    @pytest.mark.asyncio
    async def test_shutdown_survives_hanging_adapter(self, mock_heads):
        """A hanging adapter should not block shutdown."""
        await mock_heads.ensure_active("head-a")
        adapter = mock_heads.get_adapter("head-a")

        original_unload = adapter.unload

        async def hanging_unload():
            await asyncio.sleep(60)  # Way past timeout

        adapter.unload = hanging_unload

        # Shutdown with short timeout should not hang
        await mock_heads.shutdown(timeout_per_head=0.2)
        assert mock_heads.get_state("head-a") == HeadState.OFF
        assert mock_heads.active_head is None

    @pytest.mark.asyncio
    async def test_shutdown_survives_erroring_adapter(self, mock_heads):
        """An adapter that raises during unload should not block shutdown."""
        await mock_heads.ensure_active("head-a")
        adapter = mock_heads.get_adapter("head-a")

        async def erroring_unload():
            raise RuntimeError("GPU driver crashed")

        adapter.unload = erroring_unload

        await mock_heads.shutdown()
        assert mock_heads.get_state("head-a") == HeadState.OFF

    @pytest.mark.asyncio
    async def test_shutdown_default_timeout(self, mock_heads):
        """Default timeout should be 30s."""
        import inspect
        sig = inspect.signature(mock_heads.shutdown)
        assert sig.parameters["timeout_per_head"].default == 30.0


# ---------------------------------------------------------------------------
# Night shift lock tests
# ---------------------------------------------------------------------------


class TestNightShiftLock:
    def test_nightshift_uses_asyncio_lock(self):
        """Night shift routes should use asyncio.Lock for thread safety."""
        import inspect
        from multihead.api import routes_nightshift

        source = inspect.getsource(routes_nightshift)
        assert "_nightshift_lock = asyncio.Lock()" in source
        assert "async with _nightshift_lock:" in source

    def test_nightshift_trigger_sets_running_atomically(self):
        """The running flag should be set inside the lock, before background task."""
        import inspect
        from multihead.api import routes_nightshift

        source = inspect.getsource(routes_nightshift)
        # The running = True should happen inside the lock block (before _run)
        lock_section = source.split("async with _nightshift_lock:")[1]
        # Running should be set before the background task is created
        assert '_nightshift_status["running"] = True' in lock_section.split("async def _run")[0]


# ---------------------------------------------------------------------------
# DAG executor circuit breaker tests
# ---------------------------------------------------------------------------


class TestDAGCircuitBreaker:
    def test_dag_delegates_to_orchestrator(self):
        """DAGExecutor._execute_step should delegate to orchestrator._execute_step."""
        import inspect
        from multihead.dag_executor import DAGExecutor

        source = inspect.getsource(DAGExecutor._execute_step)
        # Should delegate to orchestrator, not call adapter/head_manager directly
        assert "self.orchestrator._execute_step" in source
        assert "adapter.generate" not in source

    @pytest.mark.asyncio
    async def test_dag_step_goes_through_breaker(self, tmp_path, mock_heads):
        """DAG step execution should go through the circuit breaker."""
        from multihead.artifact_store import ArtifactStore
        from multihead.dag_executor import DAGExecutor
        from multihead.event_store import EventStore
        from multihead.models import RunState
        from multihead.orchestrator import Orchestrator

        # Track if generate was called on HeadManager (not adapter)
        generate_calls = []
        original_generate = mock_heads.generate

        async def tracking_generate(head_id, prompt, **kwargs):
            generate_calls.append(head_id)
            return await original_generate(head_id, prompt, **kwargs)

        mock_heads.generate = tracking_generate

        db_path = tmp_path / "test.db"
        artifact_store = ArtifactStore(tmp_path / "artifacts", db_path)
        event_store = EventStore(tmp_path / "runs", db_path)
        orch = Orchestrator(event_store, artifact_store, mock_heads, tmp_path / "runs")

        wo = WorkOrder(
            goal="breaker test",
            steps=[
                StepDef(name="step-a", head_id="head-a", prompt_template="test"),
            ],
        )

        dag = DAGExecutor(orch)
        state = RunState(run_id=wo.run_id, work_order=wo)
        await dag.execute_dag(wo.run_id, wo, state)

        assert len(generate_calls) == 1
        assert generate_calls[0] == "head-a"


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_valid_port(self, tmp_path):
        """Valid port should be accepted."""
        settings = Settings(data_dir=tmp_path, api_port=8080)
        assert settings.api_port == 8080

    def test_invalid_port_zero(self, tmp_path):
        """Port 0 should be rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="api_port"):
            Settings(data_dir=tmp_path, api_port=0)

    def test_invalid_port_negative(self, tmp_path):
        """Negative port should be rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="api_port"):
            Settings(data_dir=tmp_path, api_port=-1)

    def test_invalid_port_too_high(self, tmp_path):
        """Port > 65535 should be rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="api_port"):
            Settings(data_dir=tmp_path, api_port=70000)

    def test_valid_mesh_secret(self, tmp_path):
        """Secret >= 16 chars should be accepted."""
        settings = Settings(data_dir=tmp_path, mesh_secret="a" * 32)
        assert settings.mesh_secret == "a" * 32

    def test_short_mesh_secret(self, tmp_path):
        """Secret < 16 chars should be rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="mesh_secret"):
            Settings(data_dir=tmp_path, mesh_secret="short")

    def test_none_mesh_secret(self, tmp_path):
        """None mesh_secret should be accepted (auth disabled)."""
        settings = Settings(data_dir=tmp_path, mesh_secret=None)
        assert settings.mesh_secret is None

    def test_default_port(self, tmp_path):
        """Default port should be 7337."""
        settings = Settings(data_dir=tmp_path)
        assert settings.api_port == 7337
