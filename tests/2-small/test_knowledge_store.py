"""Tests for the knowledge store (SQLite-backed claims, events, records)."""

import sqlite3
import threading
from datetime import datetime, timedelta, timezone

import pytest

from multihead.knowledge_models import (
    Claim,
    ClaimCanonical,
    ClaimScope,
    ClaimStatus,
    ClaimType,
    EntityRef,
    EvidencePointer,
    EventStatus,
    EventType,
    KnowledgeEvent,
    Link,
    Provenance,
    Record,
    ScopeType,
    SpanRef,
    TimeBlock,
    ValueObject,
)
from multihead.knowledge_store import KnowledgeStore


def _prov():
    return Provenance(produced_by={"kind": "extractor", "id": "test"})


def _make_record(uri="file:///test.txt"):
    return Record(uri=uri)


def _make_evidence(record_id: str):
    return EvidencePointer(
        record_id=record_id,
        uri="file:///test.txt",
        span=SpanRef(start=0, end=100),
    )


def _make_event(title="Test event"):
    return KnowledgeEvent(
        event_type=EventType.DECISION,
        title=title,
        time=TimeBlock(happened_at=datetime.now(timezone.utc)),
        provenance=_prov(),
    )


def _make_claim(claim_key="test.key", statement="Test claim."):
    return Claim(
        claim_type=ClaimType.DECISION,
        scope=ClaimScope(scope_type=ScopeType.PROJECT, scope_id="multihead"),
        canonical=ClaimCanonical(
            claim_key=claim_key,
            subject=EntityRef(entity_type="component", entity_id="core"),
            predicate="is",
            object=ValueObject(value_type="string", value="true"),
        ),
        statement=statement,
        provenance=_prov(),
    )


@pytest.fixture
def store(tmp_path):
    return KnowledgeStore(tmp_path / "knowledge.db")


# -------------------------------------------------------------------
# Records
# -------------------------------------------------------------------

class TestRecords:
    def test_insert_and_get(self, store):
        rec = _make_record()
        store.insert_record(rec)
        fetched = store.get_record(rec.record_id)
        assert fetched is not None
        assert fetched.record_id == rec.record_id
        assert fetched.uri == "file:///test.txt"

    def test_list_records(self, store):
        store.insert_record(_make_record("file:///a.txt"))
        store.insert_record(_make_record("file:///b.txt"))
        records = store.list_records()
        assert len(records) == 2

    def test_count_since(self, store):
        store.insert_record(_make_record())
        since = datetime.now(timezone.utc) - timedelta(hours=1)
        assert store.count_records_since(since) == 1
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        assert store.count_records_since(future) == 0

    def test_get_nonexistent(self, store):
        assert store.get_record("rec_nonexistent") is None


# -------------------------------------------------------------------
# Evidence
# -------------------------------------------------------------------

class TestEvidence:
    def test_insert_and_get(self, store):
        rec = _make_record()
        store.insert_record(rec)
        ep = _make_evidence(rec.record_id)
        store.insert_evidence(ep)
        fetched = store.get_evidence(ep.evidence_id)
        assert fetched is not None
        assert fetched.record_id == rec.record_id
        assert fetched.span.start == 0

    def test_get_for_record(self, store):
        rec = _make_record()
        store.insert_record(rec)
        store.insert_evidence(_make_evidence(rec.record_id))
        store.insert_evidence(_make_evidence(rec.record_id))
        evs = store.get_evidence_for_record(rec.record_id)
        assert len(evs) == 2


# -------------------------------------------------------------------
# Events
# -------------------------------------------------------------------

class TestEvents:
    def test_insert_and_get(self, store):
        evt = _make_event()
        store.insert_event(evt)
        fetched = store.get_event(evt.event_id)
        assert fetched is not None
        assert fetched.title == "Test event"
        assert fetched.event_status == EventStatus.DRAFT

    def test_list_by_status(self, store):
        store.insert_event(_make_event("A"))
        store.insert_event(_make_event("B"))
        events = store.list_events(status="draft")
        assert len(events) == 2

    def test_update_status(self, store):
        evt = _make_event()
        store.insert_event(evt)
        # Can't confirm without evidence (trigger), so test draft -> retracted
        store.update_event_status(evt.event_id, EventStatus.RETRACTED)
        fetched = store.get_event(evt.event_id)
        assert fetched.event_status == EventStatus.RETRACTED

    def test_link_evidence(self, store):
        rec = _make_record()
        store.insert_record(rec)
        ep = _make_evidence(rec.record_id)
        store.insert_evidence(ep)
        evt = _make_event()
        store.insert_event(evt)
        store.link_event_evidence(evt.event_id, ep.evidence_id, "supports")
        # Now we can confirm
        store.update_event_status(evt.event_id, EventStatus.CONFIRMED)
        fetched = store.get_event(evt.event_id)
        assert fetched.event_status == EventStatus.CONFIRMED


# -------------------------------------------------------------------
# Trigger enforcement
# -------------------------------------------------------------------

class TestTriggers:
    def test_confirm_without_evidence_fails(self, store):
        evt = _make_event()
        store.insert_event(evt)
        with pytest.raises(sqlite3.IntegrityError, match="supporting evidence"):
            store.update_event_status(evt.event_id, EventStatus.CONFIRMED)

    def test_corrected_without_supersedes_fails(self, store):
        evt = _make_event()
        store.insert_event(evt)
        # Add evidence so it could be confirmed
        rec = _make_record()
        store.insert_record(rec)
        ep = _make_evidence(rec.record_id)
        store.insert_evidence(ep)
        store.link_event_evidence(evt.event_id, ep.evidence_id, "supports")
        # Try to correct without supersedes link
        with pytest.raises(sqlite3.IntegrityError, match="supersede"):
            store.update_event_status(evt.event_id, EventStatus.CORRECTED)

    def test_accept_claim_without_evidence_fails(self, store):
        claim = _make_claim()
        store.insert_claim(claim)
        with pytest.raises(sqlite3.IntegrityError, match="supporting evidence"):
            store.update_claim_status(claim.claim_id, ClaimStatus.ACCEPTED)

    def test_supersede_claim_without_target_fails(self, store):
        claim = _make_claim()
        store.insert_claim(claim)
        with pytest.raises(sqlite3.IntegrityError, match="superseded_by_claim_id"):
            store.update_claim_status(claim.claim_id, ClaimStatus.SUPERSEDED)


# -------------------------------------------------------------------
# Claims
# -------------------------------------------------------------------

class TestClaims:
    def test_insert_and_get(self, store):
        claim = _make_claim()
        store.insert_claim(claim)
        fetched = store.get_claim(claim.claim_id)
        assert fetched is not None
        assert fetched.statement == "Test claim."
        assert fetched.claim_status == ClaimStatus.PROPOSED

    def test_list_by_status(self, store):
        store.insert_claim(_make_claim("k1"))
        store.insert_claim(_make_claim("k2"))
        claims = store.list_claims(status="proposed")
        assert len(claims) == 2

    def test_accept_claim_with_evidence(self, store):
        rec = _make_record()
        store.insert_record(rec)
        ep = _make_evidence(rec.record_id)
        store.insert_evidence(ep)

        claim = _make_claim()
        store.insert_claim(claim)
        store.link_claim_evidence(claim.claim_id, ep.evidence_id, "supports")
        store.accept_claim(claim.claim_id)

        fetched = store.get_claim(claim.claim_id)
        assert fetched.claim_status == ClaimStatus.ACCEPTED

    def test_accept_claim_without_evidence_raises(self, store):
        claim = _make_claim()
        store.insert_claim(claim)
        with pytest.raises(ValueError, match="supporting evidence"):
            store.accept_claim(claim.claim_id)

    def test_atomic_canon_update(self, store):
        """Accepting a new claim with the same key should supersede the old one."""
        rec = _make_record()
        store.insert_record(rec)
        ep1 = _make_evidence(rec.record_id)
        ep2 = _make_evidence(rec.record_id)
        store.insert_evidence(ep1)
        store.insert_evidence(ep2)

        # First claim accepted
        c1 = _make_claim("same.key", "First version")
        store.insert_claim(c1)
        store.link_claim_evidence(c1.claim_id, ep1.evidence_id, "supports")
        store.accept_claim(c1.claim_id)

        # Second claim with same key
        c2 = _make_claim("same.key", "Second version")
        store.insert_claim(c2)
        store.link_claim_evidence(c2.claim_id, ep2.evidence_id, "supports")
        store.accept_claim(c2.claim_id)

        # c1 should be superseded, c2 accepted
        fetched_c1 = store.get_claim(c1.claim_id)
        fetched_c2 = store.get_claim(c2.claim_id)
        assert fetched_c1.claim_status == ClaimStatus.SUPERSEDED
        assert fetched_c1.superseded_by_claim_id == c2.claim_id
        assert fetched_c2.claim_status == ClaimStatus.ACCEPTED

    def test_accept_supersedes_previous(self, store):
        """Accepting a claim with same key supersedes the old one."""
        rec1 = _make_record("file:///test1.txt")
        rec2 = _make_record("file:///test2.txt")
        store.insert_record(rec1)
        store.insert_record(rec2)
        ep1 = _make_evidence(rec1.record_id)
        ep2 = _make_evidence(rec2.record_id)
        store.insert_evidence(ep1)
        store.insert_evidence(ep2)

        c1 = _make_claim("same.key")
        c2 = _make_claim("same.key")
        store.insert_claim(c1)
        store.link_claim_evidence(c1.claim_id, ep1.evidence_id, "supports")
        store.accept_claim(c1.claim_id)

        store.insert_claim(c2, dedup=False)
        store.link_claim_evidence(c2.claim_id, ep2.evidence_id, "supports")
        store.accept_claim(c2.claim_id)

        fetched_c1 = store.get_claim(c1.claim_id)
        fetched_c2 = store.get_claim(c2.claim_id)
        assert fetched_c1.claim_status == ClaimStatus.SUPERSEDED
        assert fetched_c2.claim_status == ClaimStatus.ACCEPTED

    def test_claim_conflict(self, store):
        c1 = _make_claim("k1")
        c2 = _make_claim("k2")
        store.insert_claim(c1)
        store.insert_claim(c2)
        store.add_claim_conflict(c1.claim_id, c2.claim_id, "contradicts")


# -------------------------------------------------------------------
# Dedup (_check_duplicate in insert_claim)
# -------------------------------------------------------------------

class TestCheckDuplicate:
    """Tests for the _check_duplicate pre-insert dedup logic."""

    def test_exact_duplicate_skips(self, store):
        """Same claim_key + same statement => 'skip', claim not inserted twice."""
        c1 = _make_claim("dedup.key", "Exact same statement")
        store.insert_claim(c1)

        c2 = _make_claim("dedup.key", "Exact same statement")
        store.insert_claim(c2)  # should silently skip

        # Only one claim should exist with this key
        claims = store.list_claims(status="proposed")
        matching = [c for c in claims if c.canonical.claim_key == "dedup.key"]
        assert len(matching) == 1
        assert matching[0].claim_id == c1.claim_id

    def test_same_key_different_statement_supersedes(self, store):
        """Same claim_key but different statement => old claim superseded."""
        c1 = _make_claim("dedup.key2", "First version of claim")
        store.insert_claim(c1)

        c2 = _make_claim("dedup.key2", "Updated version of claim")
        store.insert_claim(c2)

        fetched_c1 = store.get_claim(c1.claim_id)
        fetched_c2 = store.get_claim(c2.claim_id)
        assert fetched_c1.claim_status == ClaimStatus.SUPERSEDED
        assert fetched_c1.superseded_by_claim_id == c2.claim_id
        assert fetched_c2.claim_status == ClaimStatus.PROPOSED

    def test_different_key_no_dedup(self, store):
        """Different claim_key => both claims inserted normally."""
        c1 = _make_claim("key.alpha", "Claim alpha")
        c2 = _make_claim("key.beta", "Claim beta")
        store.insert_claim(c1)
        store.insert_claim(c2)

        fetched_c1 = store.get_claim(c1.claim_id)
        fetched_c2 = store.get_claim(c2.claim_id)
        assert fetched_c1 is not None
        assert fetched_c2 is not None
        assert fetched_c1.claim_status == ClaimStatus.PROPOSED
        assert fetched_c2.claim_status == ClaimStatus.PROPOSED

    def test_dedup_false_skips_check(self, store):
        """dedup=False => duplicate check is skipped, both claims inserted."""
        c1 = _make_claim("dedup.off", "Same statement here")
        store.insert_claim(c1)

        c2 = _make_claim("dedup.off", "Same statement here")
        store.insert_claim(c2, dedup=False)

        fetched_c1 = store.get_claim(c1.claim_id)
        fetched_c2 = store.get_claim(c2.claim_id)
        assert fetched_c1 is not None
        assert fetched_c2 is not None
        # Both should still be proposed — no superseding happened
        assert fetched_c1.claim_status == ClaimStatus.PROPOSED
        assert fetched_c2.claim_status == ClaimStatus.PROPOSED


# -------------------------------------------------------------------
# Pack queries
# -------------------------------------------------------------------

class TestPackQueries:
    def test_get_accepted_claims(self, store):
        rec = _make_record()
        store.insert_record(rec)
        ep = _make_evidence(rec.record_id)
        store.insert_evidence(ep)

        c = _make_claim()
        store.insert_claim(c)
        store.link_claim_evidence(c.claim_id, ep.evidence_id, "supports")
        store.accept_claim(c.claim_id)

        claims = store.get_accepted_claims_for_pack()
        assert len(claims) == 1
        assert claims[0].claim_status == ClaimStatus.ACCEPTED

    def test_get_confirmed_events(self, store):
        rec = _make_record()
        store.insert_record(rec)
        ep = _make_evidence(rec.record_id)
        store.insert_evidence(ep)

        evt = _make_event()
        store.insert_event(evt)
        store.link_event_evidence(evt.event_id, ep.evidence_id, "supports")
        store.update_event_status(evt.event_id, EventStatus.CONFIRMED)

        events = store.get_confirmed_events_for_pack()
        assert len(events) == 1

    def test_get_open_loops(self, store):
        evt = KnowledgeEvent(
            event_type=EventType.QUESTION,
            title="Unresolved question",
            time=TimeBlock(happened_at=datetime.now(timezone.utc)),
            provenance=_prov(),
        )
        store.insert_event(evt)
        loops = store.get_open_loops()
        assert len(loops) == 1
        assert loops[0].title == "Unresolved question"


# -------------------------------------------------------------------
# Links
# -------------------------------------------------------------------

class TestLinks:
    def test_insert_and_list(self, store):
        link = Link(
            from_entity=EntityRef(entity_type="project", entity_id="a"),
            to_entity=EntityRef(entity_type="project", entity_id="b"),
            reason_type="shared_constraint",
            score=0.8,
            provenance=_prov(),
        )
        store.insert_link(link)
        links = store.list_links()
        assert len(links) == 1
        assert links[0].score == 0.8

    def test_list_by_entity(self, store):
        link = Link(
            from_entity=EntityRef(entity_type="project", entity_id="a"),
            to_entity=EntityRef(entity_type="project", entity_id="b"),
            reason_type="shared_entity",
            provenance=_prov(),
        )
        store.insert_link(link)
        assert len(store.list_links(entity_id="a")) == 1
        assert len(store.list_links(entity_id="c")) == 0


# -------------------------------------------------------------------
# Concurrent writes
# -------------------------------------------------------------------

class TestConcurrentWrites:
    def test_five_threads_no_data_loss(self, tmp_path):
        """5 threads writing simultaneously must not lose any claims (WAL mode)."""
        from multihead.knowledge_store import KnowledgeStore

        NUM_THREADS = 5
        CLAIMS_PER_THREAD = 10
        errors: list[Exception] = []
        inserted_ids: list[str] = []
        lock = threading.Lock()

        def worker(thread_idx: int) -> None:
            # Each thread opens its own store connection (SQLite WAL allows concurrent writers)
            s = KnowledgeStore(tmp_path / "shared.db")
            for i in range(CLAIMS_PER_THREAD):
                claim = _make_claim(
                    claim_key=f"thread{thread_idx}.item{i}",
                    statement=f"Written by thread {thread_idx}, item {i}",
                )
                try:
                    s.insert_claim(claim)
                    with lock:
                        inserted_ids.append(claim.claim_id)
                except Exception as exc:  # noqa: BLE001
                    with lock:
                        errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(NUM_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Write errors: {errors}"

        # Verify every inserted claim is retrievable — no data loss
        verify_store = KnowledgeStore(tmp_path / "shared.db")
        all_claims = verify_store.list_claims()
        retrieved_ids = {c.claim_id for c in all_claims}

        assert len(inserted_ids) == NUM_THREADS * CLAIMS_PER_THREAD
        missing = [cid for cid in inserted_ids if cid not in retrieved_ids]
        assert missing == [], f"{len(missing)} claims lost under concurrent writes: {missing[:5]}"


# -------------------------------------------------------------------
# FTS5 Full-Text Search
# -------------------------------------------------------------------


class TestFTS5Search:
    def test_fts_table_created(self, store):
        """FTS5 virtual table should be created on init."""
        with store._connect() as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='claims_fts'"
            ).fetchall()
            assert len(tables) == 1

    def test_search_finds_matching_claims(self, store):
        """Insert claims and verify FTS5 search finds them."""
        claim1 = _make_claim(
            "test.fts.router", "Router uses weighted scoring for head selection"
        )
        claim2 = _make_claim(
            "test.fts.consensus", "Consensus strategy uses majority voting"
        )
        store.insert_claim(claim1)
        store.insert_claim(claim2)

        results = store.search_claims_fts("router scoring")
        assert len(results) >= 1
        keys = [r[0] for r in results]
        assert "test.fts.router" in keys

    def test_search_empty_query(self, store):
        results = store.search_claims_fts("")
        assert results == []

    def test_search_no_matches(self, store):
        claim = _make_claim("test.fts.x", "The mesh protocol uses Ed25519")
        store.insert_claim(claim)
        results = store.search_claims_fts("zzznonexistent")
        assert results == []

    def test_search_returns_tuples(self, store):
        claim = _make_claim("test.fts.shape", "Router scoring weights")
        store.insert_claim(claim)
        results = store.search_claims_fts("router")
        assert len(results) >= 1
        assert isinstance(results[0], tuple)
        assert len(results[0]) == 3  # (claim_key, statement, confidence)

    def test_search_respects_limit(self, store):
        for i in range(5):
            store.insert_claim(_make_claim(f"test.fts.bulk.{i}", f"Router feature {i}"))
        results = store.search_claims_fts("router", limit=2)
        assert len(results) <= 2

    def test_fts_synced_on_insert(self, store):
        """New claims should be findable via FTS immediately after insert."""
        claim = _make_claim("test.fts.new", "Newly inserted claim about orchestrator")
        store.insert_claim(claim)
        results = store.search_claims_fts("orchestrator")
        assert len(results) >= 1
        assert any("orchestrator" in r[1].lower() for r in results)

    def test_like_fallback(self, store):
        """_search_claims_like should work as fallback."""
        claim = _make_claim("test.like.x", "Fallback search using LIKE queries")
        store.insert_claim(claim)
        results = store._search_claims_like("fallback LIKE")
        assert len(results) >= 1


# ── Claim Interaction Tracking ─────────────────────────────


def _make_request_claim(claim_key="test.request", statement="Test request", agent_id="other-agent"):
    """Create a question/request claim from a specific agent."""
    return Claim(
        claim_type=ClaimType.QUESTION,
        scope=ClaimScope(scope_type=ScopeType.PROJECT, scope_id="multihead"),
        canonical=ClaimCanonical(
            claim_key=claim_key,
            subject=EntityRef(entity_type="agent", entity_id=agent_id),
            predicate="asks",
            object=ValueObject(value_type="string", value=statement),
        ),
        statement=statement,
        provenance=Provenance(produced_by={"kind": "agent", "id": agent_id}),
    )


class TestClaimInteractions:
    """Tests for the claim_interactions table and agent read-receipt tracking."""

    def test_record_interaction(self, store):
        """Basic interaction recording works."""
        claim = _make_request_claim("test.ci.basic", "Hello?")
        inserted = store.insert_claim(claim)

        result = store.record_interaction(
            claim_id=inserted.claim_id,
            agent_id="test-agent",
            action="read",
        )
        assert result is True

    def test_duplicate_interaction_is_noop(self, store):
        """Recording the same (claim, agent, action) twice is idempotent."""
        claim = _make_request_claim("test.ci.dup", "Duplicate test")
        inserted = store.insert_claim(claim)

        first = store.record_interaction(inserted.claim_id, "test-agent", "read")
        second = store.record_interaction(inserted.claim_id, "test-agent", "read")
        assert first is True
        assert second is False  # no-op

    def test_different_actions_both_recorded(self, store):
        """Same agent can record different actions on the same claim."""
        claim = _make_request_claim("test.ci.multi", "Multi-action")
        inserted = store.insert_claim(claim)

        store.record_interaction(inserted.claim_id, "test-agent", "read")
        store.record_interaction(inserted.claim_id, "test-agent", "responded")

        interactions = store.get_interactions_for_claim(inserted.claim_id)
        assert len(interactions) == 2
        actions = {i["action"] for i in interactions}
        assert actions == {"read", "responded"}

    def test_has_interacted_any(self, store):
        """has_interacted with no action checks for any interaction."""
        claim = _make_request_claim("test.ci.has.any", "Has any?")
        inserted = store.insert_claim(claim)

        assert store.has_interacted(inserted.claim_id, "test-agent") is False
        store.record_interaction(inserted.claim_id, "test-agent", "read")
        assert store.has_interacted(inserted.claim_id, "test-agent") is True

    def test_has_interacted_specific_action(self, store):
        """has_interacted with specific action is precise."""
        claim = _make_request_claim("test.ci.has.specific", "Specific?")
        inserted = store.insert_claim(claim)

        store.record_interaction(inserted.claim_id, "test-agent", "read")
        assert store.has_interacted(inserted.claim_id, "test-agent", "read") is True
        assert store.has_interacted(inserted.claim_id, "test-agent", "responded") is False

    def test_response_claim_id_stored(self, store):
        """response_claim_id is stored when provided."""
        claim = _make_request_claim("test.ci.resp", "With response")
        inserted = store.insert_claim(claim)

        store.record_interaction(
            inserted.claim_id, "test-agent", "responded",
            response_claim_id="clm_fake_response",
            context="Deposited synthesis",
        )

        interactions = store.get_interactions_for_claim(inserted.claim_id)
        assert len(interactions) == 1
        assert interactions[0]["response_claim_id"] == "clm_fake_response"
        assert interactions[0]["context"] == "Deposited synthesis"

    def test_get_unhandled_claims_excludes_handled(self, store):
        """get_unhandled_claims filters out claims we've already interacted with."""
        # Create two request claims from a different agent
        c1 = _make_request_claim("test.ci.unhandled.1", "First request", "agent-x")
        c2 = _make_request_claim("test.ci.unhandled.2", "Second request", "agent-x")
        ins1 = store.insert_claim(c1)
        ins2 = store.insert_claim(c2)

        # Before any interactions, both should appear
        unhandled = store.get_unhandled_claims(
            agent_id="my-agent",
            claim_types=["question"],
            scope_id="multihead",
        )
        unhandled_ids = {c.claim_id for c in unhandled}
        assert ins1.claim_id in unhandled_ids
        assert ins2.claim_id in unhandled_ids

        # Mark first as read
        store.record_interaction(ins1.claim_id, "my-agent", "read")

        # Now only second should appear
        unhandled = store.get_unhandled_claims(
            agent_id="my-agent",
            claim_types=["question"],
            scope_id="multihead",
        )
        unhandled_ids = {c.claim_id for c in unhandled}
        assert ins1.claim_id not in unhandled_ids
        assert ins2.claim_id in unhandled_ids

    def test_get_unhandled_excludes_own_claims(self, store):
        """Claims produced by the querying agent are excluded."""
        claim = _make_request_claim("test.ci.own", "My own question", "my-agent")
        store.insert_claim(claim)

        unhandled = store.get_unhandled_claims(
            agent_id="my-agent",
            claim_types=["question"],
            scope_id="multihead",
        )
        # Should not include own claim
        assert all(
            c.provenance.produced_by.get("id") != "my-agent"
            for c in unhandled
        )

    def test_get_interactions_for_claim_empty(self, store):
        """get_interactions_for_claim returns empty list for untracked claims."""
        claim = _make_request_claim("test.ci.empty", "No interactions")
        inserted = store.insert_claim(claim)

        interactions = store.get_interactions_for_claim(inserted.claim_id)
        assert interactions == []

    def test_get_interactions_multiple_agents(self, store):
        """Multiple agents can independently track interactions."""
        claim = _make_request_claim("test.ci.multi.agent", "Multi-agent")
        inserted = store.insert_claim(claim)

        store.record_interaction(inserted.claim_id, "agent-a", "read")
        store.record_interaction(inserted.claim_id, "agent-b", "read")
        store.record_interaction(inserted.claim_id, "agent-b", "responded")

        interactions = store.get_interactions_for_claim(inserted.claim_id)
        assert len(interactions) == 3

        # Agent A has 1 interaction, Agent B has 2
        agent_a = [i for i in interactions if i["agent_id"] == "agent-a"]
        agent_b = [i for i in interactions if i["agent_id"] == "agent-b"]
        assert len(agent_a) == 1
        assert len(agent_b) == 2

    def test_dismissed_claim_not_in_unhandled(self, store):
        """Dismissed claims don't appear in unhandled."""
        claim = _make_request_claim("test.ci.dismiss", "Dismiss me", "agent-x")
        inserted = store.insert_claim(claim)

        store.record_interaction(inserted.claim_id, "my-agent", "dismissed")

        unhandled = store.get_unhandled_claims(
            agent_id="my-agent",
            claim_types=["question"],
            scope_id="multihead",
        )
        assert inserted.claim_id not in {c.claim_id for c in unhandled}
