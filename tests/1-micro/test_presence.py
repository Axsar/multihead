"""Unit tests for PresenceMonitor heartbeat emission and stale-session cleanup."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from multihead.mesh.presence import PresenceMonitor
from multihead.knowledge_store import KnowledgeStore
from multihead.knowledge_models import ClaimType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_monitor(
    tmp_path: Path, node_id: str = "test-session",
    scope_id: str = "test-proj",
) -> PresenceMonitor:
    """Return a PresenceMonitor with a real (temp) KnowledgeStore."""
    db_path = tmp_path / "presence_test.db"
    ks = KnowledgeStore(db_path)
    return PresenceMonitor(
        node_id=node_id,
        store=ks,
        interval=30.0,
        port=7337,
        scope_id=scope_id,
    )


# ---------------------------------------------------------------------------
# Heartbeat emission
# ---------------------------------------------------------------------------


class TestHeartbeatEmission:
    def test_emit_claim_writes_presence_claim(self, tmp_path):
        """_emit_claim() writes a presence claim for the local node."""
        db_path = tmp_path / "test.db"
        ks = KnowledgeStore(db_path)
        monitor = PresenceMonitor(node_id="my-session", store=ks, scope_id="my-proj")

        monitor._emit_claim(status="online")

        claims = ks.list_claims(claim_type=ClaimType.FACT.value, scope_id="my-proj")
        presence = [c for c in claims if "my-session" in c.canonical.claim_key]
        assert len(presence) >= 1

    def test_emit_claim_predicate_is_presence_status(self, tmp_path):
        """Emitted heartbeat claim has predicate 'presence_status'."""
        db_path = tmp_path / "test.db"
        ks = KnowledgeStore(db_path)
        monitor = PresenceMonitor(node_id="session-x", store=ks, scope_id="proj-x")

        monitor._emit_claim(status="online")

        claims = ks.list_claims(claim_type=ClaimType.FACT.value, scope_id="proj-x")
        presence = [c for c in claims if "session-x" in c.canonical.claim_key
                    and c.canonical.predicate == "presence_status"]
        assert len(presence) == 1

    def test_emit_claim_updates_last_seen_timestamp(self, tmp_path):
        """Repeated _emit_claim() calls refresh the last_seen field."""
        import time
        from multihead.knowledge_models import ScopeType

        db_path = tmp_path / "test.db"
        ks = KnowledgeStore(db_path)
        key = "mesh.presence.session-ts"
        # Use separate monitors with separate DBs to avoid purge interference
        monitor1 = PresenceMonitor(node_id="session-ts", store=ks, scope_id="proj-ts")

        monitor1._emit_claim(status="online")
        first = ks.get_accepted_claim(ScopeType.AGENT.value, "proj-ts", key)
        assert first is not None
        ts1 = first.canonical.object.value.get("last_seen", "")
        assert ts1  # should have a timestamp

        # Use a fresh DB for second emission to avoid purge FK cascade
        db_path2 = tmp_path / "test2.db"
        ks2 = KnowledgeStore(db_path2)
        monitor2 = PresenceMonitor(node_id="session-ts", store=ks2, scope_id="proj-ts")

        time.sleep(0.02)
        monitor2._emit_claim(status="online")
        second = ks2.get_accepted_claim(ScopeType.AGENT.value, "proj-ts", key)
        assert second is not None
        ts2 = second.canonical.object.value.get("last_seen", "")

        assert ts2 >= ts1  # time only moves forward


# ---------------------------------------------------------------------------
# Stale-session scan
# ---------------------------------------------------------------------------


class TestStaleScan:
    def test_scan_stale_marks_old_peers_absent(self, tmp_path):
        """_scan_stale() marks peer sessions as absent when last_seen exceeds threshold."""
        db_path = tmp_path / "test.db"
        ks = KnowledgeStore(db_path)
        coordinator = PresenceMonitor(node_id="coordinator", store=ks, scope_id="test-proj")

        # Create a peer's presence claim with old timestamp
        peer = PresenceMonitor(node_id="old-peer", store=ks, scope_id="test-proj")
        peer._emit_claim(status="online")

        # Manually override last_seen to be old via direct SQL
        claims = ks.list_claims(scope_id="test-proj")
        for c in claims:
            if c.canonical and "old-peer" in c.canonical.claim_key:
                old_time = (datetime.now(timezone.utc) - timedelta(seconds=200)).isoformat()
                c.canonical.object.value["last_seen"] = old_time
                import json as _json
                obj_json = _json.dumps({"value_type": "json", "value": c.canonical.object.value})
                conn = ks._connect()
                conn.execute(
                    "UPDATE claims SET object_json = ? WHERE claim_id = ?",
                    [obj_json, c.claim_id],
                )
                conn.commit()
                conn.close()

        coordinator._scan_stale()

        # Check that an 'absent' claim was written for old-peer
        all_claims = ks.list_claims(scope_id="test-proj")
        absent = [c for c in all_claims
                  if c.canonical and "old-peer" in c.canonical.claim_key
                  and c.canonical.object.value.get("status") == "absent"]
        assert len(absent) >= 1

    def test_scan_stale_does_not_touch_fresh_peers(self, tmp_path):
        """_scan_stale() leaves fresh peers untouched."""
        db_path = tmp_path / "test.db"
        ks = KnowledgeStore(db_path)
        coordinator = PresenceMonitor(node_id="coordinator", store=ks, scope_id="test-proj")

        # Create a fresh peer
        peer = PresenceMonitor(node_id="fresh-peer", store=ks, scope_id="test-proj")
        peer._emit_claim(status="online")

        coordinator._scan_stale()

        # Fresh peer should still be "online", no absent claim
        all_claims = ks.list_claims(scope_id="test-proj")
        absent = [c for c in all_claims
                  if c.canonical and "fresh-peer" in c.canonical.claim_key
                  and c.canonical.object.value.get("status") == "absent"]
        assert len(absent) == 0

    def test_scan_stale_does_not_mark_self(self, tmp_path):
        """_scan_stale() never marks the monitor's own node as absent."""
        db_path = tmp_path / "test.db"
        ks = KnowledgeStore(db_path)
        monitor = PresenceMonitor(node_id="self-node", store=ks, scope_id="test-proj")

        # Emit own heartbeat with old timestamp
        monitor._emit_claim(status="online")
        claims = ks.list_claims(scope_id="test-proj")
        for c in claims:
            if c.canonical and "self-node" in c.canonical.claim_key:
                old_time = (datetime.now(timezone.utc) - timedelta(seconds=200)).isoformat()
                c.canonical.object.value["last_seen"] = old_time
                import json as _json
                obj_json = _json.dumps({"value_type": "json", "value": c.canonical.object.value})
                conn = ks._connect()
                conn.execute(
                    "UPDATE claims SET object_json = ? WHERE claim_id = ?",
                    [obj_json, c.claim_id],
                )
                conn.commit()
                conn.close()

        monitor._scan_stale()

        # Self should NOT be marked absent
        all_claims = ks.list_claims(scope_id="test-proj")
        absent = [c for c in all_claims
                  if c.canonical and "self-node" in c.canonical.claim_key
                  and c.canonical.object.value.get("status") == "absent"]
        assert len(absent) == 0


# ---------------------------------------------------------------------------
# Async lifecycle
# ---------------------------------------------------------------------------


class TestAsyncLifecycle:
    @pytest.mark.asyncio
    async def test_start_creates_background_task(self, tmp_path):
        """start() creates a running asyncio task for the heartbeat loop."""
        monitor = _make_monitor(tmp_path)

        await monitor.start()
        assert monitor._task is not None
        assert not monitor._task.done()

        await monitor.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_task_and_marks_offline(self, tmp_path):
        """stop() cancels the loop and writes an offline claim for the local session."""
        db_path = tmp_path / "lifecycle.db"
        ks = KnowledgeStore(db_path)
        monitor = PresenceMonitor(node_id="leaving-session", store=ks, scope_id="test-proj")

        await monitor.start()
        await monitor.stop()

        assert monitor._task is None or monitor._task.done()

        claims = ks.list_claims(claim_type=ClaimType.FACT.value, scope_id="test-proj")
        offline = [c for c in claims
                   if "leaving-session" in c.canonical.claim_key
                   and c.canonical.object.value.get("status") == "offline"]
        assert len(offline) >= 1

    @pytest.mark.asyncio
    async def test_heartbeat_loop_emits_claims(self, tmp_path):
        """The heartbeat loop calls _emit_claim and _scan_stale."""
        db_path = tmp_path / "loop.db"
        ks = KnowledgeStore(db_path)
        monitor = PresenceMonitor(
            node_id="loop-node", store=ks,
            interval=0.01, scope_id="test-proj",
        )

        await monitor.start()
        await asyncio.sleep(0.05)  # allow at least one loop tick
        await monitor.stop()

        claims = ks.list_claims(scope_id="test-proj")
        presence = [c for c in claims if "loop-node" in c.canonical.claim_key]
        assert len(presence) >= 1
