"""Night Shift full pipeline tests — shares a single run across 6 tests."""

import json
from datetime import datetime, timezone

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


def _seed(rs, count=3):
    for i in range(count):
        rs.ingest_text(
            f"Record {i}: MultiHead uses event sourcing. "
            f"The core LLM runs on CPU. Test entry {i}.",
            uri=f"test://record_{i}",
        )


# -------------------------------------------------------------------
# Full pipeline — single shared run for 6 tests
# -------------------------------------------------------------------

_SHARED: dict = {}


@pytest.fixture(scope="module")
def shared_run(tmp_path_factory):
    """Run the full pipeline ONCE, share results across all tests."""
    if "done" in _SHARED:
        return _SHARED

    tmp = tmp_path_factory.mktemp("ns_shared")
    ks, art, rs, pb = _make_stores(tmp)
    _seed(rs, 5)
    hm = _make_hm()
    config = NightShiftConfig(head_id="mock-llm")
    output_dir = tmp / "output"
    ns = NightShift(ks, rs, art, pb, hm, config, output_dir)

    import asyncio
    report = asyncio.run(ns.run())

    _SHARED.update({
        "done": True,
        "report": report,
        "output_dir": output_dir,
    })
    return _SHARED


class TestFullPipeline:
    def test_all_stages_run(self, shared_run):
        report = shared_run["report"]
        total = (
            len(report.stages_completed)
            + len(report.stages_failed)
            + len(report.stages_skipped)
        )
        assert total >= 10

    def test_report_saved_to_disk(self, shared_run):
        report = shared_run["report"]
        output_dir = shared_run["output_dir"]
        report_path = output_dir / f"{report.report_id}.json"
        assert report_path.exists()
        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert "stages_completed" in data

    def test_daily_brief_written(self, shared_run):
        output_dir = shared_run["output_dir"]
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        daily_path = output_dir / f"daily_{today}.md"
        assert daily_path.exists()

    def test_nightshift_report_markdown(self, shared_run):
        output_dir = shared_run["output_dir"]
        report_md = output_dir / "nightshift_report.md"
        assert report_md.exists()
        content = report_md.read_text(encoding="utf-8")
        assert "Night Shift Report" in content

    def test_packs_built_during_pipeline(self, shared_run):
        report = shared_run["report"]
        assert isinstance(report.packs_built, list)

    def test_keyword_index_built(self, shared_run):
        output_dir = shared_run["output_dir"]
        index_path = output_dir / "keyword_index.json"
        assert index_path.exists()
        data = json.loads(index_path.read_text(encoding="utf-8"))
        assert data["mode"] == "keyword_only"
