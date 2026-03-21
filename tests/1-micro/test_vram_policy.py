"""Tests for VRAM policy and manager (standalone, not in test_agentic_core)."""

import pytest

from multihead.head_manager import HeadManager
from multihead.models import AdapterKind, HeadManifest
from multihead.vram_policy import VRAMManager, VRAMPolicy


@pytest.fixture
def heads():
    manifests = {
        "core-llm": HeadManifest(
            head_id="core-llm", name="Core", adapter=AdapterKind.MOCK,
            model="mock-v1", kind="llm", gpu_required=False,
        ),
        "worker-llm": HeadManifest(
            head_id="worker-llm", name="Worker", adapter=AdapterKind.MOCK,
            model="mock-v1", kind="llm", gpu_required=False,
        ),
    }
    return HeadManager(manifests)


class TestVRAMPolicy:
    def test_defaults(self):
        p = VRAMPolicy()
        assert p.core_mode == "keep_loaded"
        assert p.worker_load_policy == "per_stage"

    def test_custom_policy(self):
        p = VRAMPolicy(core_mode="unload_during_batch", worker_load_policy="keep_warm")
        assert p.core_mode == "unload_during_batch"
        assert p.worker_load_policy == "keep_warm"


class TestVRAMManagerKeepLoaded:
    @pytest.mark.asyncio
    async def test_ensure_core(self, heads):
        vm = VRAMManager(heads, VRAMPolicy(), core_head_id="core-llm")
        await vm.ensure_core_available()
        assert heads.active_head == "core-llm"

    @pytest.mark.asyncio
    async def test_prepare_loads_worker(self, heads):
        vm = VRAMManager(heads, VRAMPolicy(), core_head_id="core-llm")
        await vm.ensure_core_available()
        await vm.prepare_for_batch("worker-llm")
        assert heads.active_head == "worker-llm"

    @pytest.mark.asyncio
    async def test_restore_after_batch_noop(self, heads):
        vm = VRAMManager(heads, VRAMPolicy(), core_head_id="core-llm")
        await vm.ensure_core_available()
        await vm.prepare_for_batch("worker-llm")
        await vm.restore_after_batch()
        # keep_loaded mode: restore is a noop, worker stays active
        assert heads.active_head == "worker-llm"


class TestVRAMManagerUnloadDuringBatch:
    @pytest.mark.asyncio
    async def test_unload_and_restore(self, heads):
        policy = VRAMPolicy(core_mode="unload_during_batch")
        vm = VRAMManager(heads, policy, core_head_id="core-llm")

        await vm.ensure_core_available()
        assert heads.active_head == "core-llm"

        await vm.prepare_for_batch("worker-llm")
        assert heads.active_head == "worker-llm"

        await vm.restore_after_batch()
        assert heads.active_head == "core-llm"

    @pytest.mark.asyncio
    async def test_no_restore_if_core_wasnt_active(self, heads):
        policy = VRAMPolicy(core_mode="unload_during_batch")
        vm = VRAMManager(heads, policy, core_head_id="core-llm")

        # Don't call ensure_core first
        await vm.prepare_for_batch("worker-llm")
        assert heads.active_head == "worker-llm"

        await vm.restore_after_batch()
        # Core wasn't active before, so it shouldn't be restored
        assert heads.active_head == "worker-llm"


class TestVRAMStatus:
    @pytest.mark.asyncio
    async def test_get_status(self, heads):
        vm = VRAMManager(heads, VRAMPolicy(), core_head_id="core-llm")
        await vm.ensure_core_available()
        status = vm.get_vram_status()

        assert status["core_head_id"] == "core-llm"
        assert status["active_head"] == "core-llm"
        assert "policy" in status
        assert status["policy"]["core_mode"] == "keep_loaded"
        assert "core-llm" in status["heads"]
        assert "worker-llm" in status["heads"]

    def test_status_before_any_load(self, heads):
        vm = VRAMManager(heads, VRAMPolicy(), core_head_id="core-llm")
        status = vm.get_vram_status()
        assert status["active_head"] is None
