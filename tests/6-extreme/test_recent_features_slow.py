"""Night Shift callback and metrics tests — optimized with shared runs.

TestNightShiftDebugCallback: shares one pipeline run with callback recording.
TestNightShiftMetricsFix: shares one pipeline run for metrics checks.
"""

import json

import pytest

from multihead.artifact_store import ArtifactStore
from multihead.context_packs import PackBuilder
from multihead.head_manager import HeadManager
from multihead.knowledge_models import NightShiftConfig
from multihead.knowledge_store import KnowledgeStore
from multihead.models import AdapterKind, HeadManifest
from multihead.night_shift import NightShift
from multihead.record_store import RecordStore


def _make_stores(base_path):
    ks = KnowledgeStore(base_path / "knowledge.db")
    art = ArtifactStore(base_path / "artifacts", base_path / "artifacts.db")
    rs = RecordStore(ks, art)
    pb = PackBuilder(ks, base_path / "packs")
    return ks, art, rs, pb


def _make_hm():
    manifest = HeadManifest(
        head_id="mock-llm", name="Mock", adapter=AdapterKind.MOCK,
        model="mock-v1", kind="llm", gpu_required=False,
    )
    return HeadManager({"mock-llm": manifest})


# -------------------------------------------------------------------
# Callback tests — single shared run with event capture
# -------------------------------------------------------------------

_CALLBACK_SHARED: dict = {}


@pytest.fixture(scope="module")
def callback_run(tmp_path_factory):
    """Run pipeline ONCE with callback, share events across tests."""
    if "done" in _CALLBACK_SHARED:
        return _CALLBACK_SHARED

    tmp = tmp_path_factory.mktemp("ns_callback")
    ks, art, rs, pb = _make_stores(tmp)
    rs.ingest_text(
        "MultiHead is an orchestration layer for local models.",
        uri="test://1",
    )

    hm = _make_hm()
    config = NightShiftConfig(head_id="mock-llm")
    ns = NightShift(ks, rs, art, pb, hm, config, tmp / "output")

    events: list[dict] = []
    ns.on_progress = lambda evt: events.append(evt)

    import asyncio
    report = asyncio.run(ns.run())

    _CALLBACK_SHARED.update({
        "done": True,
        "events": events,
        "report": report,
    })
    return _CALLBACK_SHARED


class TestNightShiftDebugCallback:
    """Tests for the on_progress callback mechanism."""

    def test_callback_fires_stage_events(self, callback_run):
        events = callback_run["events"]
        starts = [e for e in events if e["event"] == "stage_start"]
        assert len(starts) > 0
        assert starts[0]["index"] == 0
        assert starts[0]["total"] > 0

        dones = [
            e for e in events
            if e["event"] in ("stage_done", "stage_skip")
        ]
        assert len(dones) > 0

        completes = [e for e in events if e["event"] == "complete"]
        assert len(completes) == 1

    def test_llm_call_events_emitted(self, callback_run):
        events = callback_run["events"]
        llm_calls = [e for e in events if e["event"] == "llm_call"]
        if llm_calls:
            assert "elapsed_s" in llm_calls[0]
            assert "prompt_chars" in llm_calls[0]
            assert "tokens" in llm_calls[0]


# -------------------------------------------------------------------
# Metrics tests — single shared run
# -------------------------------------------------------------------

_METRICS_SHARED: dict = {}


@pytest.fixture(scope="module")
def metrics_run(tmp_path_factory):
    """Run pipeline ONCE for metrics checks."""
    if "done" in _METRICS_SHARED:
        return _METRICS_SHARED

    tmp = tmp_path_factory.mktemp("ns_metrics")
    ks, art, rs, pb = _make_stores(tmp)
    for i in range(3):
        rs.ingest_text(
            f"Record {i}: MultiHead uses event sourcing "
            f"for durable execution of work orders.",
            uri=f"test://{i}",
        )

    hm = _make_hm()
    config = NightShiftConfig(head_id="mock-llm")
    output_dir = tmp / "output"
    ns = NightShift(ks, rs, art, pb, hm, config, output_dir)

    events: list[dict] = []
    ns.on_progress = lambda evt: events.append(evt)

    import asyncio
    asyncio.run(ns.run())

    report_path = output_dir / "nightshift_report.json"
    data = (
        json.loads(report_path.read_text())
        if report_path.exists() else {}
    )

    _METRICS_SHARED.update({
        "done": True,
        "report_data": data,
        "output_dir": output_dir,
        "events": events,
    })
    return _METRICS_SHARED


class TestNightShiftMetricsFix:

    def test_chunk_count_in_report(self, metrics_run):
        data = metrics_run["report_data"]
        assert data["chunk_count"] > 0

    def test_all_metrics_accumulated(self, metrics_run):
        data = metrics_run["report_data"]
        assert "stage_metrics" in data
        assert "normalize_chunk" in data["stage_metrics"]
