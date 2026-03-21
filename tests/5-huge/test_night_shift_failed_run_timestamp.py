"""Test that a failed NightShift run does not update the last-run timestamp."""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

import pytest

from multihead.artifact_store import ArtifactStore
from multihead.context_packs import PackBuilder
from multihead.knowledge_models import NightShiftConfig
from multihead.knowledge_store import KnowledgeStore
from multihead.night_shift import NightShift
from multihead.record_store import RecordStore


@pytest.fixture
def temp_env():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "test.db"
        artifacts_dir = tmp_path / "artifacts"

        ks = KnowledgeStore(db_path)
        artifact_store = ArtifactStore(artifacts_dir, db_path)
        record_store = RecordStore(ks, artifact_store)
        packs_dir = tmp_path / "packs"

        manifest = MagicMock()
        manifest.head_id = "mock-llm"
        manifest.adapter = "mock"
        head_manager = MagicMock()
        head_manager.manifests = {"mock-llm": manifest}
        head_manager.get_manifest.return_value = manifest
        head_manager.get_adapter.return_value = MagicMock()

        config = NightShiftConfig(head_id="mock-llm")
        output_dir = tmp_path / "output"

        yield {
            "ks": ks,
            "as": artifact_store,
            "rs": record_store,
            "pb": PackBuilder(ks, packs_dir),
            "hm": head_manager,
            "config": config,
            "output_dir": output_dir,
            "tmp_path": tmp_path,
        }


@pytest.mark.asyncio
async def test_failed_run_does_not_update_timestamp(temp_env):
    """Failed run does not update the last run timestamp."""
    ns = NightShift(
        temp_env["ks"],
        temp_env["rs"],
        temp_env["as"],
        temp_env["pb"],
        temp_env["hm"],
        temp_env["config"],
        temp_env["output_dir"],
    )

    initial_time = datetime.now(timezone.utc) - timedelta(hours=48)
    ns._update_last_successful_run_time(initial_time)

    # Ingest a record so the pipeline has work to do
    temp_env["rs"].ingest_text("Test content", uri="test://test_record")

    # Force normalize_chunk to fail by mocking it to raise
    ns._stage_normalize_chunk = AsyncMock(side_effect=RuntimeError("forced failure"))

    report = await ns.run()

    assert "normalize_chunk" in report.stages_failed

    # Last run timestamp should NOT be updated
    last_run = ns._get_last_successful_run_time()
    assert last_run is not None

    delta = abs((last_run - initial_time).total_seconds())
    assert delta < 1
