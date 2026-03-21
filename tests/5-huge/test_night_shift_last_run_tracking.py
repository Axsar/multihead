"""Test Night Shift last run timestamp tracking."""

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
    """Create temporary environment for Night Shift."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "test.db"
        artifacts_dir = tmp_path / "artifacts"
        nightshift_dir = tmp_path / "nightshift"

        ks = KnowledgeStore(db_path)
        artifact_store = ArtifactStore(artifacts_dir, db_path)
        record_store = RecordStore(ks, artifact_store)
        packs_dir = tmp_path / "packs"
        pack_builder = PackBuilder(ks, packs_dir)
        head_manager = MagicMock()

        config = NightShiftConfig(
            input_window_hours=24,
            auto_accept_confidence=0.9,
            auto_accept_min_supports=1,
        )

        yield {
            "ks": ks,
            "rs": record_store,
            "as": artifact_store,
            "pb": pack_builder,
            "hm": head_manager,
            "config": config,
            "output_dir": nightshift_dir,
        }


@pytest.mark.asyncio
async def test_first_run_uses_24h_window(temp_env):
    """First run with no previous timestamp uses 24h window."""
    ns = NightShift(
        temp_env["ks"],
        temp_env["rs"],
        temp_env["as"],
        temp_env["pb"],
        temp_env["hm"],
        temp_env["config"],
        temp_env["output_dir"],
    )

    # No last run timestamp yet
    last_run = ns._get_last_successful_run_time()
    assert last_run is None

    # Create a record older than 24h (simulated by backdating in DB)
    old_record = temp_env["rs"].ingest_text("Old record content", uri="test://old_record")
    old_time = datetime.now(timezone.utc) - timedelta(hours=25)
    # Manually backdate the record
    conn = temp_env["ks"]._connect()
    conn.execute(
        "UPDATE records SET created_at = ?, captured_at = ? WHERE record_id = ?",
        (old_time.isoformat(), old_time.isoformat(), old_record.record_id),
    )
    conn.commit()

    # Create a record within 24h
    recent_record = temp_env["rs"].ingest_text("Recent record content", uri="test://recent_record")
    recent_time = datetime.now(timezone.utc) - timedelta(hours=12)
    conn.execute(
        "UPDATE records SET created_at = ?, captured_at = ? WHERE record_id = ?",
        (recent_time.isoformat(), recent_time.isoformat(), recent_record.record_id),
    )
    conn.commit()

    # Run select_input_window
    context = {}
    result = await ns._stage_select_input_window(context)

    # Should only see 1 record (within 24h window)
    assert result is not None
    assert result["record_count"] == 1

    # Since should be approximately 24h ago
    since = result["since"]
    expected = datetime.now(timezone.utc) - timedelta(hours=24)
    delta = abs((since - expected).total_seconds())
    assert delta < 60  # Within 1 minute


@pytest.mark.asyncio
async def test_subsequent_run_uses_last_run_timestamp(temp_env):
    """Subsequent run uses the last successful run timestamp."""
    ns = NightShift(
        temp_env["ks"],
        temp_env["rs"],
        temp_env["as"],
        temp_env["pb"],
        temp_env["hm"],
        temp_env["config"],
        temp_env["output_dir"],
    )

    # Simulate a previous successful run 48h ago
    last_run_time = datetime.now(timezone.utc) - timedelta(hours=48)
    ns._update_last_successful_run_time(last_run_time)

    # Verify it was saved
    retrieved = ns._get_last_successful_run_time()
    assert retrieved is not None
    # Allow small difference due to datetime parsing
    delta = abs((retrieved - last_run_time).total_seconds())
    assert delta < 1

    # Create records at different times
    # 50h ago (before last run)
    r1 = temp_env["rs"].ingest_text("Old record content", uri="test://old_record")
    # 40h ago (after last run, should be picked up)
    r2 = temp_env["rs"].ingest_text("Record 1 content", uri="test://record1")
    # 30h ago (after last run, should be picked up)
    r3 = temp_env["rs"].ingest_text("Record 2 content", uri="test://record2")

    # Backdate records via direct SQL
    t1 = datetime.now(timezone.utc) - timedelta(hours=50)
    t2 = datetime.now(timezone.utc) - timedelta(hours=40)
    t3 = datetime.now(timezone.utc) - timedelta(hours=30)
    conn = temp_env["ks"]._connect()
    conn.execute(
        "UPDATE records SET created_at = ?, captured_at = ? WHERE record_id = ?",
        (t1.isoformat(), t1.isoformat(), r1.record_id),
    )
    conn.execute(
        "UPDATE records SET created_at = ?, captured_at = ? WHERE record_id = ?",
        (t2.isoformat(), t2.isoformat(), r2.record_id),
    )
    conn.execute(
        "UPDATE records SET created_at = ?, captured_at = ? WHERE record_id = ?",
        (t3.isoformat(), t3.isoformat(), r3.record_id),
    )
    conn.commit()
    conn.close()

    # Run select_input_window
    context = {}
    result = await ns._stage_select_input_window(context)

    # Should see 2 records (created after last run)
    assert result is not None
    assert result["record_count"] == 2

    # Since should match the last run time
    since = result["since"]
    delta = abs((since - last_run_time).total_seconds())
    assert delta < 1


@pytest.mark.asyncio
async def test_full_run_updates_timestamp(temp_env):
    """Successful full run updates the last run timestamp."""
    ns = NightShift(
        temp_env["ks"],
        temp_env["rs"],
        temp_env["as"],
        temp_env["pb"],
        temp_env["hm"],
        temp_env["config"],
        temp_env["output_dir"],
    )

    # Create a test record
    record = temp_env["rs"].ingest_text("Test content for Night Shift", uri="test://test_record")

    # Mock LLM stages to avoid actual model calls
    for stage_name in [
        "entity_extraction",
        "topic_assignment",
        "event_extraction",
        "claim_extraction",
        "consistency_check",
        "open_loops",
        "daily_brief",
        "weekly_rollup",
        "cross_topic_links",
    ]:
        setattr(ns, f"_stage_{stage_name}", AsyncMock(return_value={}))

    # Run Night Shift
    before_run = datetime.now(timezone.utc)
    report = await ns.run()
    after_run = datetime.now(timezone.utc)

    # Verify run was successful
    assert report.records_processed > 0
    assert len(report.stages_failed) == 0

    # Check that last run timestamp was updated
    last_run = ns._get_last_successful_run_time()
    assert last_run is not None

    # Should be approximately the report end time
    delta = abs((last_run - report.ended_at).total_seconds())
    assert delta < 1

    # Should be between before and after run
    assert before_run <= last_run <= after_run


