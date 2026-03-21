"""Tests for the ACP Bridge (BotVibes integration)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from multihead.acp_bridge import ACPBridge
from multihead.config import Settings
from multihead.head_manager import HeadManager
from multihead.models import AdapterKind, HeadManifest


def _make_heads() -> dict[str, HeadManifest]:
    return {
        "mock-llm": HeadManifest(
            head_id="mock-llm", name="Mock LLM", adapter=AdapterKind.MOCK,
            model="mock-v1", kind="llm", gpu_required=False,
        ),
        "mock-vlm": HeadManifest(
            head_id="mock-vlm", name="Mock VLM", adapter=AdapterKind.MOCK,
            model="mock-v2", kind="vlm", gpu_required=False,
        ),
        "openai-gpt4o": HeadManifest(
            head_id="openai-gpt4o", name="GPT-4o Mini", adapter=AdapterKind.OPENAI,
            model="gpt-4o-mini", kind="llm", gpu_required=False,
            extra={"api_key": "sk-test"},
        ),
    }


def _make_bridge(tmp_path: Path) -> ACPBridge:
    heads = _make_heads()
    hm = HeadManager(heads)
    settings = Settings(data_dir=tmp_path, api_host="127.0.0.1", api_port=7337)
    return ACPBridge(hm, settings)


# ---------------------------------------------------------------------------
# Descriptor building
# ---------------------------------------------------------------------------


class TestDescriptorBuilding:
    def test_builds_capabilities_from_heads(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        desc = bridge._build_multihead_descriptor()
        caps = desc["capabilities"]
        # Should have one capability per head
        assert len(caps) == 3
        assert "com.multihead.llm.mock-llm" in caps
        assert "com.multihead.vlm.mock-vlm" in caps
        assert "com.multihead.llm.openai-gpt4o" in caps

    def test_vlm_adds_image_schemas(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        desc = bridge._build_multihead_descriptor()
        assert "image/jpeg" in desc["input_schema"]
        assert "image/png" in desc["input_schema"]

    def test_max_concurrency_is_one(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        desc = bridge._build_multihead_descriptor()
        assert desc["max_concurrency"] == 1


# ---------------------------------------------------------------------------
# Status building
# ---------------------------------------------------------------------------


class TestStatusBuilding:
    def test_idle_when_no_active_head(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        status = bridge.build_status()
        assert status["status"] == "idle"
        assert status["active_head"] is None
        assert status["active_tasks"] == 0

    @pytest.mark.asyncio
    async def test_busy_when_head_active(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        # Simulate an active head
        await bridge.heads.ensure_active("mock-llm")
        status = bridge.build_status()
        assert status["status"] == "busy"
        assert status["active_head"] == "mock-llm"
        assert status["active_tasks"] == 1
        await bridge.heads.shutdown()


# ---------------------------------------------------------------------------
# File-mode fallback
# ---------------------------------------------------------------------------


class TestFileFallback:
    @pytest.mark.asyncio
    async def test_writes_capability_file(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        await bridge.start()  # No ACP_URL → file mode
        state_file = tmp_path / "acp_state.json"
        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert data["agent_id"] == "multihead-agent"
        assert data["acp_connected"] is False
        assert "mock-llm" in data["heads"]
        assert "endpoint" in data

    @pytest.mark.asyncio
    async def test_refresh_updates_file(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        await bridge.start()
        # Simulate a head becoming active
        await bridge.heads.ensure_active("mock-llm")
        bridge.refresh_capability_file()
        data = json.loads((tmp_path / "acp_state.json").read_text())
        assert data["status"] == "busy"
        assert data["active_head"] == "mock-llm"
        await bridge.heads.shutdown()

    @pytest.mark.asyncio
    async def test_not_connected_without_url(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        await bridge.start()
        assert bridge.connected is False


# ---------------------------------------------------------------------------
# ACP mode (connection failure → graceful fallback)
# ---------------------------------------------------------------------------


class TestACPModeFallback:
    @pytest.mark.asyncio
    async def test_bad_url_falls_back_to_file(self, tmp_path):
        heads = _make_heads()
        hm = HeadManager(heads)
        settings = Settings(data_dir=tmp_path, api_host="127.0.0.1", api_port=7337)
        bridge = ACPBridge(
            hm, settings,
            acp_url="http://localhost:99999/api/v1",
            api_key="fake-key",
            project_id="test-project",
        )
        await bridge.start()
        # Should fall back to file mode
        assert bridge.connected is False
        assert (tmp_path / "acp_state.json").exists()

    @pytest.mark.asyncio
    async def test_stop_is_safe_without_start(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        await bridge.stop()  # Should not raise


# ---------------------------------------------------------------------------
# Task execution routing (on_task callback + auto_execute flag)
# ---------------------------------------------------------------------------


class TestTaskRouting:
    def test_auto_execute_default_true(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        assert bridge._auto_execute is True

    def test_auto_execute_flag(self, tmp_path):
        heads = _make_heads()
        hm = HeadManager(heads)
        settings = Settings(data_dir=tmp_path, api_host="127.0.0.1", api_port=7337)
        bridge = ACPBridge(hm, settings, auto_execute=False)
        assert bridge._auto_execute is False

    def test_on_task_default_none(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        assert bridge._on_task is None

    def test_set_on_task(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        callback = lambda task: None
        bridge.set_on_task(callback)
        assert bridge._on_task is callback

    def test_clear_on_task(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        bridge.set_on_task(lambda t: None)
        bridge.set_on_task(None)
        assert bridge._on_task is None

    @pytest.mark.asyncio
    async def test_on_task_callback_intercepts(self, tmp_path):
        """When on_task is set, tasks go to callback not auto-execute."""
        import httpx

        bridge = _make_bridge(tmp_path)
        received = []
        bridge.set_on_task(lambda task: received.append(task))

        task = {"task_id": "test-123", "payload_ref": "do something", "capability": "com.multihead"}
        client = httpx.AsyncClient()
        await bridge._execute_task(client, task)
        await client.aclose()

        assert len(received) == 1
        assert received[0]["task_id"] == "test-123"

    @pytest.mark.asyncio
    async def test_auto_execute_false_skips(self, tmp_path):
        """When auto_execute=False and no callback, tasks are skipped."""
        import httpx

        heads = _make_heads()
        hm = HeadManager(heads)
        settings = Settings(data_dir=tmp_path, api_host="127.0.0.1", api_port=7337)
        bridge = ACPBridge(hm, settings, auto_execute=False)

        task = {"task_id": "test-456", "payload_ref": "do something"}
        client = httpx.AsyncClient()
        # Should not raise or attempt HTTP calls
        await bridge._execute_task(client, task)
        await client.aclose()
