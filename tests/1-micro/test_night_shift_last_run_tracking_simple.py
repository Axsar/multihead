"""Simple tests for Night Shift last run timestamp tracking."""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from multihead.knowledge_store import KnowledgeStore
from multihead.night_shift import NightShift


@pytest.mark.asyncio
async def test_get_and_set_last_run_time():
    """Test getting and setting the last run timestamp."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "test.db"

        ks = KnowledgeStore(db_path)

        # Mock up minimal Night Shift instance just to test the methods
        ns = object.__new__(NightShift)
        ns.knowledge = ks

        # Initially no last run
        assert ns._get_last_successful_run_time() is None

        # Set a timestamp
        test_time = datetime.now(timezone.utc) - timedelta(hours=48)
        ns._update_last_successful_run_time(test_time)

        # Retrieve it
        retrieved = ns._get_last_successful_run_time()
        assert retrieved is not None

        # Should be approximately equal (allow small datetime parsing difference)
        delta = abs((retrieved - test_time).total_seconds())
        assert delta < 1


@pytest.mark.asyncio
async def test_last_run_time_persistence():
    """Test that last run timestamp persists across Night Shift instances."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "test.db"

        ks1 = KnowledgeStore(db_path)

        # First instance sets timestamp
        ns1 = object.__new__(NightShift)
        ns1.knowledge = ks1
        test_time = datetime.now(timezone.utc) - timedelta(hours=24)
        ns1._update_last_successful_run_time(test_time)

        # Second instance reads it
        ks2 = KnowledgeStore(db_path)
        ns2 = object.__new__(NightShift)
        ns2.knowledge = ks2
        retrieved = ns2._get_last_successful_run_time()

        assert retrieved is not None
        delta = abs((retrieved - test_time).total_seconds())
        assert delta < 1


@pytest.mark.asyncio
async def test_multiple_updates_uses_latest():
    """Test that multiple updates use the most recent timestamp."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "test.db"

        ks = KnowledgeStore(db_path)
        ns = object.__new__(NightShift)
        ns.knowledge = ks

        # Set initial timestamp
        time1 = datetime.now(timezone.utc) - timedelta(hours=72)
        ns._update_last_successful_run_time(time1)

        # Update to a newer timestamp
        time2 = datetime.now(timezone.utc) - timedelta(hours=48)
        ns._update_last_successful_run_time(time2)

        # Update to even newer timestamp
        time3 = datetime.now(timezone.utc) - timedelta(hours=24)
        ns._update_last_successful_run_time(time3)

        # Should retrieve the most recent one
        retrieved = ns._get_last_successful_run_time()
        assert retrieved is not None

        # Should match time3
        delta = abs((retrieved - time3).total_seconds())
        assert delta < 1
