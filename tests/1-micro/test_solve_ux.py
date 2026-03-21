"""Tests for Solve UX smart-defaults: discover_active_sessions()."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path


from multihead.knowledge_models import (
    Claim,
    ClaimCanonical,
    ClaimScope,
    ClaimStatus,
    ClaimType,
    EntityRef,
    Provenance,
    ScopeType,
    Stability,
    ValueObject,
)
from multihead.knowledge_store import KnowledgeStore
from multihead.solve import discover_active_sessions, write_presence_claim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ks() -> tuple[KnowledgeStore, tempfile.TemporaryDirectory]:
    """Return a fresh in-memory KnowledgeStore (caller owns tmpdir)."""
    tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    db_path = Path(tmpdir.name) / "test.db"
    return KnowledgeStore(db_path), tmpdir


def _write_presence(
    ks: KnowledgeStore,
    session_id: str,
    project_id: str = "proj-test",
    age_minutes: float = 0,
    capabilities: list[str] | None = None,
) -> None:
    """Write a presence claim with an optionally backdated valid_from."""
    valid_from = datetime.now(timezone.utc) - timedelta(minutes=age_minutes)

    claim = Claim(
        claim_type=ClaimType.FACT,
        claim_status=ClaimStatus.ACCEPTED,
        scope=ClaimScope(
            scope_type=ScopeType.PROJECT,
            scope_id=project_id,
            visibility="project",
            valid_from=valid_from,
        ),
        canonical=ClaimCanonical(
            claim_key=f"agent.{session_id}.presence",
            subject=EntityRef(
                entity_type="session",
                entity_id=session_id,
                label=session_id,
            ),
            predicate="available",
            object=ValueObject(
                value_type="json",
                value={
                    "capabilities": capabilities or [],
                    "session_type": "multihead_solve",
                    "last_seen": valid_from.isoformat(),
                },
            ),
        ),
        statement=f"Session {session_id} is active",
        confidence=1.0,
        stability=Stability.VOLATILE,
        provenance=Provenance(produced_by={"kind": "session", "id": session_id}),
    )
    ks.insert_claim(claim)


# ---------------------------------------------------------------------------
# Tests: no sessions
# ---------------------------------------------------------------------------


class TestDiscoverActiveSessionsNoSessions:
    def test_empty_db_returns_empty_list(self):
        """No presence claims → empty list."""
        ks, tmpdir = _ks()
        with tmpdir:
            result = discover_active_sessions(ks, "proj-test", "self-session")
            assert result == []

    def test_only_self_session_returns_empty(self):
        """Own session is excluded; result is empty."""
        ks, tmpdir = _ks()
        with tmpdir:
            write_presence_claim(ks, "self-session", "proj-test")
            result = discover_active_sessions(ks, "proj-test", "self-session")
            assert result == []

    def test_non_presence_claims_ignored(self):
        """Claims with predicate != 'available' are not returned."""
        ks, tmpdir = _ks()
        with tmpdir:
            # Manually insert a claim with a different predicate
            claim = Claim(
                claim_type=ClaimType.FACT,
                claim_status=ClaimStatus.ACCEPTED,
                scope=ClaimScope(
                    scope_type=ScopeType.PROJECT,
                    scope_id="proj-test",
                    visibility="project",
                    valid_from=datetime.now(timezone.utc),
                ),
                canonical=ClaimCanonical(
                    claim_key="agent.other-session.status",
                    subject=EntityRef(entity_type="session", entity_id="other-session"),
                    predicate="offline",  # NOT "available"
                    object=ValueObject(value_type="string", value="gone"),
                ),
                statement="Session other-session is offline",
                confidence=1.0,
                stability=Stability.VOLATILE,
                provenance=Provenance(produced_by={"kind": "session", "id": "other-session"}),
            )
            ks.insert_claim(claim)

            result = discover_active_sessions(ks, "proj-test", "self-session")
            assert result == []


# ---------------------------------------------------------------------------
# Tests: one session (own excluded)
# ---------------------------------------------------------------------------


class TestDiscoverActiveSessionsOneSession:
    def test_one_other_session_returned(self):
        """Single other active session is returned."""
        ks, tmpdir = _ks()
        with tmpdir:
            write_presence_claim(ks, "other-session", "proj-test")
            result = discover_active_sessions(ks, "proj-test", "self-session")

            assert len(result) == 1
            assert result[0]["session_id"] == "other-session"

    def test_returned_session_has_required_fields(self):
        """Returned dict includes session_id, capabilities, last_seen."""
        ks, tmpdir = _ks()
        with tmpdir:
            write_presence_claim(ks, "agent-alpha", "proj-test", ["solve", "research"])
            result = discover_active_sessions(ks, "proj-test", "self-session")

            assert len(result) == 1
            entry = result[0]
            assert "session_id" in entry
            assert "capabilities" in entry
            assert "last_seen" in entry

    def test_capabilities_preserved(self):
        """Capabilities list from presence claim is passed through."""
        ks, tmpdir = _ks()
        with tmpdir:
            caps = ["llm.generate", "reasoning.complex"]
            write_presence_claim(ks, "capable-agent", "proj-test", caps)
            result = discover_active_sessions(ks, "proj-test", "self-session")

            assert result[0]["capabilities"] == caps

    def test_different_project_not_returned(self):
        """Session from a different project is excluded."""
        ks, tmpdir = _ks()
        with tmpdir:
            write_presence_claim(ks, "other-session", "different-project")
            result = discover_active_sessions(ks, "proj-test", "self-session")
            assert result == []

    def test_self_excluded_even_if_both_present(self):
        """Own session is always excluded even when another session exists."""
        ks, tmpdir = _ks()
        with tmpdir:
            write_presence_claim(ks, "self-session", "proj-test")
            write_presence_claim(ks, "other-session", "proj-test")
            result = discover_active_sessions(ks, "proj-test", "self-session")

            ids = [s["session_id"] for s in result]
            assert "self-session" not in ids
            assert "other-session" in ids


# ---------------------------------------------------------------------------
# Tests: multiple sessions
# ---------------------------------------------------------------------------


class TestDiscoverActiveSessionsMultipleSessions:
    def test_multiple_sessions_all_returned(self):
        """All other active sessions are returned."""
        ks, tmpdir = _ks()
        with tmpdir:
            for i in range(3):
                write_presence_claim(ks, f"agent-{i}", "proj-test")

            result = discover_active_sessions(ks, "proj-test", "self-session")
            assert len(result) == 3

    def test_session_ids_correct(self):
        """Returned session IDs match what was written."""
        ks, tmpdir = _ks()
        with tmpdir:
            expected_ids = {"alpha", "beta", "gamma"}
            for sid in expected_ids:
                write_presence_claim(ks, sid, "proj-test")

            result = discover_active_sessions(ks, "proj-test", "self-session")
            returned_ids = {s["session_id"] for s in result}
            assert returned_ids == expected_ids

    def test_self_excluded_from_multiple(self):
        """Own session excluded when multiple sessions are present."""
        ks, tmpdir = _ks()
        with tmpdir:
            write_presence_claim(ks, "self-session", "proj-test")
            write_presence_claim(ks, "agent-1", "proj-test")
            write_presence_claim(ks, "agent-2", "proj-test")

            result = discover_active_sessions(ks, "proj-test", "self-session")
            ids = {s["session_id"] for s in result}
            assert "self-session" not in ids
            assert len(result) == 2


# ---------------------------------------------------------------------------
# Tests: stale session filtered out
# ---------------------------------------------------------------------------


class TestDiscoverActiveSessionsStaleFiltering:
    def test_stale_session_excluded_by_default(self):
        """Session older than max_age_minutes (10) is excluded."""
        ks, tmpdir = _ks()
        with tmpdir:
            _write_presence(ks, "stale-agent", age_minutes=11)
            result = discover_active_sessions(ks, "proj-test", "self-session")
            assert result == []

    def test_fresh_session_included(self):
        """Session within max_age_minutes is included."""
        ks, tmpdir = _ks()
        with tmpdir:
            _write_presence(ks, "fresh-agent", age_minutes=5)
            result = discover_active_sessions(ks, "proj-test", "self-session")
            assert len(result) == 1
            assert result[0]["session_id"] == "fresh-agent"

    def test_exactly_at_cutoff_excluded(self):
        """Session exactly at cutoff (10 min) is excluded (strictly older)."""
        ks, tmpdir = _ks()
        with tmpdir:
            _write_presence(ks, "boundary-agent", age_minutes=10.0)
            result = discover_active_sessions(ks, "proj-test", "self-session")
            assert result == []

    def test_mixed_fresh_and_stale(self):
        """Only fresh sessions returned when mixed with stale ones."""
        ks, tmpdir = _ks()
        with tmpdir:
            _write_presence(ks, "fresh-1", age_minutes=2)
            _write_presence(ks, "fresh-2", age_minutes=8)
            _write_presence(ks, "stale-1", age_minutes=15)
            _write_presence(ks, "stale-2", age_minutes=60)

            result = discover_active_sessions(ks, "proj-test", "self-session")
            ids = {s["session_id"] for s in result}
            assert ids == {"fresh-1", "fresh-2"}
            assert "stale-1" not in ids
            assert "stale-2" not in ids

    def test_custom_max_age_respected(self):
        """Custom max_age_minutes parameter is respected."""
        ks, tmpdir = _ks()
        with tmpdir:
            _write_presence(ks, "agent-5min", age_minutes=5)
            _write_presence(ks, "agent-25min", age_minutes=25)

            # With max_age=30, both should be included
            result = discover_active_sessions(
                ks, "proj-test", "self-session", max_age_minutes=30
            )
            ids = {s["session_id"] for s in result}
            assert "agent-5min" in ids
            assert "agent-25min" in ids

            # With max_age=10, only fresh one should be included
            result = discover_active_sessions(
                ks, "proj-test", "self-session", max_age_minutes=10
            )
            ids = {s["session_id"] for s in result}
            assert "agent-5min" in ids
            assert "agent-25min" not in ids

    def test_stale_self_write_presence_still_excluded_as_self(self):
        """Own session (even if fresh) is never returned."""
        ks, tmpdir = _ks()
        with tmpdir:
            _write_presence(ks, "self-session", age_minutes=1)
            result = discover_active_sessions(ks, "proj-test", "self-session")
            assert result == []
