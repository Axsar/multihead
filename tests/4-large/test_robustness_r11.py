"""Round 11 tests: unload safety, session limits, health probes, run timeout."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from multihead.api.app import create_app
from multihead.config import Settings
from multihead.head_manager import HeadManager
from multihead.models import (
    AdapterKind,
    HeadManifest,
    HeadState,
    RunStatus,
    StepDef,
    WorkOrder,
)
from multihead.session import SessionManager


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


@pytest.fixture
def client(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        config_dir=tmp_path / "config",
    )
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
  - head_id: mock-vlm
    name: Mock VLM
    adapter: mock
    model: mock-v1
    kind: vlm
    gpu_required: false
""")

    app = create_app(settings)
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# HeadManager public unload tests
# ---------------------------------------------------------------------------


class TestHeadManagerUnload:
    @pytest.mark.asyncio
    async def test_unload_head_public(self, mock_heads):
        """Public unload_head() should safely unload an active head."""
        await mock_heads.ensure_active("head-a")
        assert mock_heads.get_state("head-a") == HeadState.ACTIVE

        await mock_heads.unload_head("head-a")
        assert mock_heads.get_state("head-a") == HeadState.OFF
        assert mock_heads.active_head is None

    @pytest.mark.asyncio
    async def test_unload_head_unknown(self, mock_heads):
        """Unloading unknown head should raise KeyError."""
        with pytest.raises(KeyError, match="Unknown head"):
            await mock_heads.unload_head("nonexistent")

    @pytest.mark.asyncio
    async def test_unload_head_already_off(self, mock_heads):
        """Unloading an already-off head should be a no-op."""
        assert mock_heads.get_state("head-a") == HeadState.OFF
        await mock_heads.unload_head("head-a")  # Should not raise
        assert mock_heads.get_state("head-a") == HeadState.OFF

    @pytest.mark.asyncio
    async def test_unload_via_api(self, client):
        """POST /heads/{id}/unload should use the public method."""
        # First wake the head
        resp = client.post("/heads/mock-llm/wake")
        assert resp.status_code == 200

        # Then unload it
        resp = client.post("/heads/mock-llm/unload")
        assert resp.status_code == 200
        assert resp.json()["state"] == "off"

    @pytest.mark.asyncio
    async def test_unload_unknown_via_api(self, client):
        """POST /heads/unknown/unload should return 404."""
        resp = client.post("/heads/nonexistent/unload")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Session size limits tests
# ---------------------------------------------------------------------------


class TestSessionLimits:
    def test_message_truncation(self, tmp_path):
        """Oversized messages should be truncated to max_message_size."""
        mgr = SessionManager(tmp_path / "sessions", max_message_size=100)
        session = mgr.create_session()
        msg = mgr.add_message(session.session_id, "user", "x" * 200)
        assert len(msg.content) == 100

    def test_message_count_limit(self, tmp_path):
        """Messages beyond max_messages should trigger trimming."""
        mgr = SessionManager(tmp_path / "sessions", max_messages=5)
        session = mgr.create_session()

        for i in range(10):
            mgr.add_message(session.session_id, "user", f"msg-{i}")

        session = mgr.get_session(session.session_id)
        assert len(session.messages) == 5
        # Should keep the most recent
        assert session.messages[-1].content == "msg-9"

    def test_system_messages_preserved(self, tmp_path):
        """System messages should be preserved when trimming."""
        mgr = SessionManager(tmp_path / "sessions", max_messages=5)
        session = mgr.create_session()

        mgr.add_message(session.session_id, "system", "You are a helpful assistant")
        for i in range(10):
            mgr.add_message(session.session_id, "user", f"msg-{i}")

        session = mgr.get_session(session.session_id)
        assert len(session.messages) == 5
        system_msgs = [m for m in session.messages if m.role == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0].content == "You are a helpful assistant"

    def test_default_limits(self, tmp_path):
        """Default limits should be reasonable."""
        mgr = SessionManager(tmp_path / "sessions")
        assert mgr.max_messages == 500
        assert mgr.max_message_size == 50_000

    def test_custom_limits(self, tmp_path):
        """Custom limits should be honored."""
        mgr = SessionManager(tmp_path / "sessions", max_messages=10, max_message_size=1000)
        assert mgr.max_messages == 10
        assert mgr.max_message_size == 1000


# ---------------------------------------------------------------------------
# Health probe tests
# ---------------------------------------------------------------------------


class TestHealthProbes:
    def test_health_endpoint(self, client):
        """GET /health should include ready field."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["ready"] is True

    def test_liveness_probe(self, client):
        """GET /healthz should always return alive."""
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"

    def test_readiness_probe(self, client):
        """GET /readyz should check all components."""
        resp = client.get("/readyz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["checks"]["head_manager"] is True
        assert data["checks"]["event_store"] is True
        assert data["checks"]["orchestrator"] is True


# ---------------------------------------------------------------------------
# Run timeout tests
# ---------------------------------------------------------------------------


class TestRunTimeout:
    @pytest.mark.asyncio
    async def test_run_times_out(self, tmp_path, mock_heads):
        """A run exceeding run_timeout_seconds should fail."""
        from multihead.artifact_store import ArtifactStore
        from multihead.event_store import EventStore
        from multihead.orchestrator import Orchestrator

        db_path = tmp_path / "test.db"
        artifact_store = ArtifactStore(tmp_path / "artifacts", db_path)
        event_store = EventStore(tmp_path / "runs", db_path)
        orchestrator = Orchestrator(event_store, artifact_store, mock_heads, tmp_path / "runs")

        # Make head-a hang
        adapter = mock_heads.get_adapter("head-a")

        async def hanging_gen(prompt, **kw):
            await asyncio.sleep(10)  # Way past timeout
            return {"text": "late", "tokens_in": 0, "tokens_out": 0}

        adapter.generate = hanging_gen

        wo = WorkOrder(
            goal="timeout test",
            run_timeout_seconds=0.5,  # 500ms
            steps=[
                StepDef(name="slow-step", head_id="head-a", prompt_template="test"),
            ],
        )
        state = await orchestrator.create_run(wo)
        state = await orchestrator.execute_run(state.run_id)

        assert state.status == RunStatus.FAILED

    @pytest.mark.asyncio
    async def test_default_timeout(self):
        """Default run_timeout_seconds should be 300s."""
        wo = WorkOrder(goal="test")
        assert wo.run_timeout_seconds == 300.0

    @pytest.mark.asyncio
    async def test_fast_run_no_timeout(self, tmp_path, mock_heads):
        """A fast run should complete normally within timeout."""
        from multihead.artifact_store import ArtifactStore
        from multihead.event_store import EventStore
        from multihead.orchestrator import Orchestrator

        db_path = tmp_path / "test.db"
        artifact_store = ArtifactStore(tmp_path / "artifacts", db_path)
        event_store = EventStore(tmp_path / "runs", db_path)
        orchestrator = Orchestrator(event_store, artifact_store, mock_heads, tmp_path / "runs")

        wo = WorkOrder(
            goal="fast test",
            run_timeout_seconds=30.0,
            steps=[
                StepDef(name="fast-step", head_id="head-a", prompt_template="test"),
            ],
        )
        state = await orchestrator.create_run(wo)
        state = await orchestrator.execute_run(state.run_id)

        assert state.status == RunStatus.DONE


# ---------------------------------------------------------------------------
# WebSocket error logging tests
# ---------------------------------------------------------------------------


class TestWebSocketErrorLogging:
    def test_ws_routes_use_exception_logging(self):
        """WebSocket routes should use logger.exception for full tracebacks."""
        import inspect
        from multihead.api import routes_ws

        source = inspect.getsource(routes_ws)
        # Should use logger.exception (includes traceback) not logger.warning
        assert "logger.exception" in source
        # The old pattern should be gone
        assert 'logger.warning("Error in run events' not in source
        assert 'logger.warning("Error in chat' not in source


# ---------------------------------------------------------------------------
# VRAMManager uses public unload
# ---------------------------------------------------------------------------


class TestVRAMManagerUnload:
    def test_vram_policy_uses_public_method(self):
        """VRAMManager should call unload_head(), not _unload()."""
        import inspect
        from multihead import vram_policy

        source = inspect.getsource(vram_policy)
        assert "._unload(" not in source
        assert ".unload_head(" in source

    @pytest.mark.asyncio
    async def test_vram_prepare_batch_unloads_safely(self, mock_heads):
        """VRAMManager.prepare_for_batch should use public unload."""
        from multihead.vram_policy import VRAMManager, VRAMPolicy

        policy = VRAMPolicy(core_mode="unload_during_batch")
        vram = VRAMManager(mock_heads, policy, core_head_id="head-a")

        await mock_heads.ensure_active("head-a")
        assert mock_heads.get_state("head-a") == HeadState.ACTIVE

        await vram.prepare_for_batch("head-b")
        # head-a should be unloaded, head-b active
        assert mock_heads.get_state("head-a") == HeadState.OFF
        assert mock_heads.get_state("head-b") == HeadState.ACTIVE
