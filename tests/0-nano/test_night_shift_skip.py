"""Night Shift skip condition tests — fast with mocked stages."""

from unittest.mock import AsyncMock

import pytest

from multihead.artifact_store import ArtifactStore
from multihead.context_packs import PackBuilder
from multihead.head_manager import HeadManager
from multihead.knowledge_models import NightShiftConfig, NightShiftReport
from multihead.knowledge_store import KnowledgeStore
from multihead.models import AdapterKind, HeadManifest
from multihead.night_shift import NightShift
from multihead.record_store import RecordStore

_MOCKABLE_STAGES = [
    "session_harvest", "hot_signals",
    "entity_extraction", "topic_assignment", "event_extraction",
    "claim_extraction", "consistency_check", "conflict_resolution",
    "claim_fusion", "open_loops", "daily_brief", "weekly_rollup",
    "cross_topic_links", "staleness_sweep", "narrative_fusion",
    "solver_discovery", "recipe_learning", "backlog_sweep",
]


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


class TestSkipConditions:
    @pytest.mark.asyncio
    async def test_skip_when_no_records(self, tmp_path):
        """Pipeline skips when no records."""
        ks, art, rs, pb = _make_stores(tmp_path)
        hm = _make_hm()
        config = NightShiftConfig(head_id="mock-llm")
        ns = NightShift(ks, rs, art, pb, hm, config, tmp_path / "out")
        for stage in _MOCKABLE_STAGES:
            setattr(ns, f"_stage_{stage}", AsyncMock(return_value={}))
        report = await ns.run()
        assert isinstance(report, NightShiftReport)
        assert (
            "select_input_window" in report.stages_skipped
            or report.records_processed == 0
        )

    @pytest.mark.asyncio
    async def test_runs_with_records(self, tmp_path):
        """Pipeline processes records."""
        ks, art, rs, pb = _make_stores(tmp_path)
        for i in range(3):
            rs.ingest_text(
                f"Record {i}: MultiHead test entry {i}.",
                uri=f"test://record_{i}",
            )
        hm = _make_hm()
        config = NightShiftConfig(head_id="mock-llm")
        ns = NightShift(ks, rs, art, pb, hm, config, tmp_path / "out")
        for stage in _MOCKABLE_STAGES:
            setattr(ns, f"_stage_{stage}", AsyncMock(return_value={}))
        report = await ns.run()
        assert isinstance(report, NightShiftReport)
        assert report.records_processed >= 3
        assert len(report.stages_completed) > 0


class TestCallbackEdgeCases:
    @pytest.mark.asyncio
    async def test_no_callback_no_crash(self, tmp_path):
        ks, art, rs, pb = _make_stores(tmp_path)
        rs.ingest_text("Test content.", uri="test://1")
        hm = _make_hm()
        config = NightShiftConfig(head_id="mock-llm")
        ns = NightShift(ks, rs, art, pb, hm, config, tmp_path / "out")
        ns.on_progress = None
        for stage in _MOCKABLE_STAGES:
            setattr(ns, f"_stage_{stage}", AsyncMock(return_value={}))
        report = await ns.run()
        assert len(report.stages_completed) > 0

    @pytest.mark.asyncio
    async def test_callback_error_doesnt_break_pipeline(self, tmp_path):
        ks, art, rs, pb = _make_stores(tmp_path)
        rs.ingest_text("Test content.", uri="test://1")
        hm = _make_hm()
        config = NightShiftConfig(head_id="mock-llm")
        ns = NightShift(ks, rs, art, pb, hm, config, tmp_path / "out")

        def bad_callback(evt):
            raise RuntimeError("Callback exploded!")

        ns.on_progress = bad_callback
        for stage in _MOCKABLE_STAGES:
            setattr(ns, f"_stage_{stage}", AsyncMock(return_value={}))
        report = await ns.run()
        assert len(report.stages_completed) > 0
