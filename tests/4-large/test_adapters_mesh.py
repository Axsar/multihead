"""Tests for adapter lifecycle, mesh capabilities, and mesh routes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from multihead.adapters.mock import MockAdapter
from multihead.api.app import create_app
from multihead.config import Settings
from multihead.mesh.capability import (
    Capability,
    CapabilityRegistry,
    auto_register_from_heads,
)
from multihead.models import AdapterKind, HeadManifest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_manifest():
    return HeadManifest(
        head_id="mock-llm", name="Mock LLM", adapter=AdapterKind.MOCK,
        model="mock-v1", kind="llm", gpu_required=False,
    )


@pytest.fixture
def vlm_manifest():
    return HeadManifest(
        head_id="mock-vlm", name="Mock VLM", adapter=AdapterKind.MOCK,
        model="mock-v1", kind="vlm", gpu_required=False,
    )


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
# MockAdapter lifecycle tests
# ---------------------------------------------------------------------------


class TestMockAdapter:
    @pytest.mark.asyncio
    async def test_load_unload_cycle(self, mock_manifest):
        adapter = MockAdapter(mock_manifest)
        assert not adapter._loaded

        await adapter.load()
        assert adapter._loaded
        assert await adapter.healthcheck()

        await adapter.unload()
        assert not adapter._loaded
        assert not await adapter.healthcheck()

    @pytest.mark.asyncio
    async def test_generate_requires_load(self, mock_manifest):
        adapter = MockAdapter(mock_manifest)
        with pytest.raises(RuntimeError, match="not loaded"):
            await adapter.generate("test prompt")

    @pytest.mark.asyncio
    async def test_generate_returns_expected_fields(self, mock_manifest):
        adapter = MockAdapter(mock_manifest)
        await adapter.load()
        result = await adapter.generate("Hello world")
        assert "text" in result
        assert "tokens_in" in result
        assert "tokens_out" in result
        assert isinstance(result["text"], str)
        assert len(result["text"]) > 0

    @pytest.mark.asyncio
    async def test_vlm_returns_json(self, vlm_manifest):
        adapter = MockAdapter(vlm_manifest)
        await adapter.load()
        result = await adapter.generate("Describe the image")
        import json
        data = json.loads(result["text"])
        assert "caption" in data
        assert "entities" in data

    @pytest.mark.asyncio
    async def test_call_count_increments(self, mock_manifest):
        adapter = MockAdapter(mock_manifest)
        await adapter.load()
        assert adapter._call_count == 0
        await adapter.generate("test 1")
        assert adapter._call_count == 1
        await adapter.generate("test 2")
        assert adapter._call_count == 2

    @pytest.mark.asyncio
    async def test_generate_stream(self, mock_manifest):
        adapter = MockAdapter(mock_manifest)
        await adapter.load()
        chunks = []
        async for chunk in adapter.generate_stream("Hello"):
            chunks.append(chunk)
        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert len(full_text) > 0

    @pytest.mark.asyncio
    async def test_chat_method(self, mock_manifest):
        adapter = MockAdapter(mock_manifest)
        await adapter.load()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"},
        ]
        result = await adapter.chat(messages)
        assert "text" in result


# ---------------------------------------------------------------------------
# CapabilityRegistry tests
# ---------------------------------------------------------------------------


class TestCapabilityRegistry:
    def test_register_and_list(self):
        registry = CapabilityRegistry()
        cap = Capability(node_id="node-1", name="Test LLM", kind="llm", model="test-v1")
        registry.register(cap)

        caps = registry.list_capabilities()
        assert len(caps) == 1
        assert caps[0].name == "Test LLM"

    def test_filter_by_kind(self):
        registry = CapabilityRegistry()
        registry.register(Capability(node_id="n", name="LLM", kind="llm"))
        registry.register(Capability(node_id="n", name="VLM", kind="vlm"))
        registry.register(Capability(node_id="n", name="Embed", kind="embed"))

        llm_caps = registry.list_capabilities("llm")
        assert len(llm_caps) == 1
        assert llm_caps[0].kind == "llm"

    def test_find_available(self):
        registry = CapabilityRegistry()
        cap1 = Capability(node_id="n", name="A", kind="llm", status="available")
        cap2 = Capability(node_id="n", name="B", kind="llm", status="busy")
        registry.register(cap1)
        registry.register(cap2)

        available = registry.find_available("llm")
        assert len(available) == 1
        assert available[0].status == "available"

    def test_update_status(self):
        registry = CapabilityRegistry()
        cap = Capability(node_id="n", name="A", kind="llm")
        registry.register(cap)

        registry.update_status(cap.capability_id, "busy")
        assert registry.list_capabilities()[0].status == "busy"

        registry.update_status(cap.capability_id, "available")
        assert registry.list_capabilities()[0].status == "available"

    def test_unregister(self):
        registry = CapabilityRegistry()
        cap = Capability(node_id="n", name="A", kind="llm")
        registry.register(cap)
        assert len(registry.list_capabilities()) == 1

        registry.unregister(cap.capability_id)
        assert len(registry.list_capabilities()) == 0

    def test_find_by_model(self):
        registry = CapabilityRegistry()
        registry.register(Capability(node_id="n", name="A", kind="llm", model="qwen3:8b"))
        registry.register(Capability(node_id="n", name="B", kind="llm", model="llama3"))

        found = registry.find_by_model("qwen3:8b")
        assert len(found) == 1
        assert found[0].model == "qwen3:8b"

    def test_auto_register_from_heads(self):
        registry = CapabilityRegistry()
        manifests = {
            "mock-llm": HeadManifest(
                head_id="mock-llm", name="Mock LLM", adapter=AdapterKind.MOCK,
                model="mock-v1", kind="llm", gpu_required=False,
            ),
            "mock-vlm": HeadManifest(
                head_id="mock-vlm", name="Mock VLM", adapter=AdapterKind.MOCK,
                model="mock-v1", kind="vlm", gpu_required=True,
            ),
        }
        registered = auto_register_from_heads(registry, manifests, "node-test")
        assert len(registered) == 2
        assert len(registry.list_capabilities()) == 2

        # Check node_id is set
        for cap in registered:
            assert cap.node_id == "node-test"


# ---------------------------------------------------------------------------
# Mesh route tests (via app)
# ---------------------------------------------------------------------------


class TestMeshRoutes:
    def test_mesh_health(self, client):
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_list_capabilities(self, client):
        resp = client.get("/v1/capabilities")
        assert resp.status_code == 200
        caps = resp.json()
        assert len(caps) >= 2  # mock-llm and mock-vlm

    def test_filter_capabilities_by_kind(self, client):
        resp = client.get("/v1/capabilities?kind=llm")
        assert resp.status_code == 200
        caps = resp.json()
        assert all(c["kind"] == "llm" for c in caps)

    def test_node_info(self, client):
        resp = client.get("/v1/node")
        assert resp.status_code == 200
        data = resp.json()
        assert "node_id" in data
        assert "capabilities" in data
        assert data["capabilities"] >= 2

    def test_submit_task(self, client):
        resp = client.post("/v1/tasks", json={
            "capability_kind": "llm",
            "prompt": "What is 2+2?",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert "result" in data

    def test_submit_task_unknown_kind(self, client):
        resp = client.post("/v1/tasks", json={
            "capability_kind": "nonexistent",
            "prompt": "test",
        })
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# EventStore corruption recovery tests
# ---------------------------------------------------------------------------


class TestEventStoreCorruptionRecovery:
    def test_truncated_jsonl_skipped(self, tmp_path):
        """Corrupted lines in events.jsonl should be skipped, not crash."""
        from multihead.event_store import EventStore
        from multihead.models import EventKind, RunEvent

        db_path = tmp_path / "test.db"
        store = EventStore(tmp_path / "runs", db_path)

        run_id = "test-corruption"
        # Write a valid event
        store.append(RunEvent(
            run_id=run_id,
            kind=EventKind.RUN_CREATED,
            data={"work_order": {"goal": "test", "steps": []}},
        ))

        # Manually append corrupted line
        events_path = tmp_path / "runs" / run_id / "events.jsonl"
        with open(events_path, "a") as f:
            f.write('{"truncated json without closing\n')
            f.write('not json at all\n')

        # Write another valid event
        store.append(RunEvent(
            run_id=run_id,
            kind=EventKind.RUN_DONE,
        ))

        # Should read 2 valid events, skipping 2 corrupted lines
        events = store.read_events(run_id)
        assert len(events) == 2
        assert events[0].kind == EventKind.RUN_CREATED
        assert events[1].kind == EventKind.RUN_DONE

    def test_empty_events_file(self, tmp_path):
        """Empty events file should return empty list."""
        from multihead.event_store import EventStore

        db_path = tmp_path / "test.db"
        store = EventStore(tmp_path / "runs", db_path)

        run_id = "test-empty"
        run_dir = tmp_path / "runs" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "events.jsonl").write_text("")

        events = store.read_events(run_id)
        assert events == []

    def test_replay_survives_corruption(self, tmp_path):
        """Replay should work even if some events are corrupted."""
        from multihead.event_store import EventStore
        from multihead.models import EventKind, RunEvent, RunStatus

        db_path = tmp_path / "test.db"
        store = EventStore(tmp_path / "runs", db_path)

        run_id = "test-replay-corrupt"
        store.append(RunEvent(
            run_id=run_id,
            kind=EventKind.RUN_CREATED,
            data={"work_order": {"goal": "test replay", "steps": []}},
        ))

        # Corrupt a line
        events_path = tmp_path / "runs" / run_id / "events.jsonl"
        with open(events_path, "a") as f:
            f.write("corrupted line here\n")

        store.append(RunEvent(run_id=run_id, kind=EventKind.RUN_DONE))

        state = store.replay(run_id)
        assert state.status == RunStatus.DONE
        assert state.work_order is not None
        assert state.work_order.goal == "test replay"
