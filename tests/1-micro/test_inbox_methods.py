"""Tests for knowledge store inbox methods."""

import pytest
from datetime import datetime, timedelta, timezone

from multihead.knowledge_store import KnowledgeStore
from multihead.knowledge_models import (
    Claim,
    ClaimStatus,
    ClaimType,
    ClaimScope,
    ScopeType,
    ClaimCanonical,
    EntityRef,
    ValueObject,
    Provenance,
)


@pytest.fixture
def knowledge_store(tmp_path):
    """Create a temporary knowledge store."""
    db_path = tmp_path / "test_knowledge.db"
    return KnowledgeStore(db_path)


@pytest.fixture
def sample_question_claim():
    """Create a sample question claim."""
    return Claim(
        claim_id="clm_test_question_001",
        claim_status=ClaimStatus.PROPOSED,
        claim_type=ClaimType.QUESTION,
        scope=ClaimScope(
            scope_type=ScopeType.PROJECT,
            scope_id="multihead",
            visibility="project",
            valid_from=datetime.now(timezone.utc),
        ),
        canonical=ClaimCanonical(
            claim_key="test.question.decomposition",
            subject=EntityRef(entity_type="task", entity_id="task_1", label="Test Task"),
            predicate="needs_decomposition",
            object=ValueObject(value_type="string", value="Build a web scraper"),
        ),
        statement="How should I decompose this task: Build a web scraper?",
        rationale="Need peer review on decomposition approach",
        confidence=0.9,
        provenance=Provenance(
            produced_by={"id": "session-a", "method": "manual"},
        ),
    )


@pytest.fixture
def sample_response_claim(sample_question_claim):
    """Create a sample response claim."""
    return Claim(
        claim_id="clm_test_response_001",
        claim_status=ClaimStatus.ACCEPTED,
        claim_type=ClaimType.FACT,
        scope=ClaimScope(
            scope_type=ScopeType.PROJECT,
            scope_id="multihead",
            visibility="project",
            valid_from=datetime.now(timezone.utc),
        ),
        canonical=ClaimCanonical(
            claim_key="test.response.decomposition",
            subject=EntityRef(entity_type="task", entity_id="task_1", label="Test Task"),
            predicate="decomposition_response",
            object=ValueObject(value_type="string", value="Phase 1: Research..."),
        ),
        statement="RESPONSE: Here's my decomposition approach...",
        rationale="Response to decomposition request",
        confidence=0.88,
        related_claim_ids=[sample_question_claim.claim_id],
        provenance=Provenance(
            produced_by={"id": "session-b", "method": "auto_responder"},
        ),
    )


def test_normalize_claim_id_list_simple_format(knowledge_store):
    """Test normalizing simple list of claim IDs."""
    simple_list = ["clm_123", "clm_456", "clm_789"]
    result = knowledge_store._normalize_claim_id_list(simple_list)
    assert result == simple_list


def test_normalize_claim_id_list_rich_format(knowledge_store):
    """Test normalizing rich format with relations."""
    rich_list = [
        {"claim_id": "clm_123", "rel": "response_to"},
        {"claim_id": "clm_456", "rel": "supports"},
    ]
    result = knowledge_store._normalize_claim_id_list(rich_list)
    assert result == ["clm_123", "clm_456"]


def test_normalize_claim_id_list_mixed_format(knowledge_store):
    """Test normalizing mixed format."""
    mixed_list = [
        "clm_123",
        {"claim_id": "clm_456", "rel": "response_to"},
        "clm_789",
    ]
    result = knowledge_store._normalize_claim_id_list(mixed_list)
    assert result == ["clm_123", "clm_456", "clm_789"]


def test_normalize_claim_id_list_empty(knowledge_store):
    """Test normalizing empty list."""
    result = knowledge_store._normalize_claim_id_list([])
    assert result == []


def test_get_responses_to_claim(knowledge_store, sample_question_claim, sample_response_claim):
    """Test getting responses to a specific claim."""
    # Deposit question and response
    knowledge_store.insert_claim(sample_question_claim)
    knowledge_store.insert_claim(sample_response_claim)

    # Get responses
    responses = knowledge_store.get_responses_to_claim(sample_question_claim.claim_id)

    assert len(responses) == 1
    assert responses[0].claim_id == sample_response_claim.claim_id
    assert sample_question_claim.claim_id in responses[0].related_claim_ids


def test_get_responses_to_claim_no_responses(knowledge_store, sample_question_claim):
    """Test getting responses when none exist."""
    knowledge_store.insert_claim(sample_question_claim)

    responses = knowledge_store.get_responses_to_claim(sample_question_claim.claim_id)
    assert len(responses) == 0


def test_get_pending_messages_filters_own_questions(knowledge_store, sample_question_claim):
    """Test that get_pending_messages filters out own questions."""
    knowledge_store.insert_claim(sample_question_claim)

    # Query as same session that posted the question
    messages = knowledge_store.get_pending_messages(
        session_id="session-a",
        scope_id="multihead",
    )

    # Should not see own question
    assert len(messages) == 0


def test_get_pending_messages_shows_others_questions(knowledge_store, sample_question_claim):
    """Test that get_pending_messages shows questions from other sessions."""
    knowledge_store.insert_claim(sample_question_claim)

    # Query as different session
    messages = knowledge_store.get_pending_messages(
        session_id="session-b",
        scope_id="multihead",
    )

    # Should see the question
    assert len(messages) == 1
    assert messages[0].claim_id == sample_question_claim.claim_id


def test_get_pending_messages_filters_by_scope(knowledge_store):
    """Test scope filtering in get_pending_messages."""
    # Create questions in different scopes
    multihead_question = Claim(
        claim_id="clm_multihead_q",
        claim_status=ClaimStatus.PROPOSED,
        claim_type=ClaimType.QUESTION,
        scope=ClaimScope(
            scope_type=ScopeType.PROJECT,
            scope_id="multihead",
            visibility="project",
            valid_from=datetime.now(timezone.utc),
        ),
        canonical=ClaimCanonical(
            claim_key="test.multihead.question",
            subject=EntityRef(entity_type="task", entity_id="t1", label="Task"),
            predicate="needs_help",
            object=ValueObject(value_type="string", value="help"),
        ),
        statement="MultiHead question",
        provenance=Provenance(
            produced_by={"id": "session-a", "method": "manual"},
        ),
    )

    h2v_question = Claim(
        claim_id="clm_h2v_q",
        claim_status=ClaimStatus.PROPOSED,
        claim_type=ClaimType.QUESTION,
        scope=ClaimScope(
            scope_type=ScopeType.PROJECT,
            scope_id="h2v",
            visibility="project",
            valid_from=datetime.now(timezone.utc),
        ),
        canonical=ClaimCanonical(
            claim_key="test.h2v.question",
            subject=EntityRef(entity_type="task", entity_id="t2", label="Task"),
            predicate="needs_help",
            object=ValueObject(value_type="string", value="help"),
        ),
        statement="Project question",
        provenance=Provenance(
            produced_by={"id": "session-a", "method": "manual"},
        ),
    )

    knowledge_store.insert_claim(multihead_question)
    knowledge_store.insert_claim(h2v_question)

    # Query for multihead scope
    messages = knowledge_store.get_pending_messages(
        session_id="session-b",
        scope_id="multihead",
    )

    # Should only see multihead question
    assert len(messages) == 1
    assert messages[0].claim_id == "clm_multihead_q"


def test_get_pending_messages_respects_max_age(knowledge_store):
    """Test that old questions are filtered out."""
    import sqlite3

    # Create old question (25 hours ago)
    old_time = datetime.now(timezone.utc) - timedelta(hours=25)
    old_question = Claim(
        claim_id="clm_old_q",
        claim_status=ClaimStatus.PROPOSED,
        claim_type=ClaimType.QUESTION,
        scope=ClaimScope(
            scope_type=ScopeType.PROJECT,
            scope_id="multihead",
            visibility="project",
            valid_from=old_time,
        ),
        canonical=ClaimCanonical(
            claim_key="test.old.question",
            subject=EntityRef(entity_type="task", entity_id="t1", label="Task"),
            predicate="needs_help",
            object=ValueObject(value_type="string", value="help"),
        ),
        statement="Old question",
        provenance=Provenance(
            produced_by={"id": "session-a", "method": "manual"},
        ),
    )

    # Insert then backdate via direct SQL
    knowledge_store.insert_claim(old_question)
    conn = sqlite3.connect(knowledge_store.db_path, timeout=10.0)
    conn.execute(
        "UPDATE claims SET created_at = ? WHERE claim_id = ?",
        (old_time.isoformat(), "clm_old_q"),
    )
    conn.commit()
    conn.close()

    # Query with default 24h max age
    messages = knowledge_store.get_pending_messages(
        session_id="session-b",
        scope_id="multihead",
        max_age_hours=24,
    )

    # Should not see old question
    assert len(messages) == 0


def test_get_claims_by_relation(knowledge_store, sample_question_claim, sample_response_claim):
    """Test getting claims by relation."""
    knowledge_store.insert_claim(sample_question_claim)
    knowledge_store.insert_claim(sample_response_claim)

    # Get all related claims
    related = knowledge_store.get_claims_by_relation(sample_question_claim.claim_id)

    assert len(related) == 1
    assert related[0].claim_id == sample_response_claim.claim_id


def test_get_pending_messages_excludes_retracted(knowledge_store):
    """Test that retracted questions are not returned."""
    retracted_question = Claim(
        claim_id="clm_retracted_q",
        claim_status=ClaimStatus.REJECTED,
        claim_type=ClaimType.QUESTION,
        scope=ClaimScope(
            scope_type=ScopeType.PROJECT,
            scope_id="multihead",
            visibility="project",
            valid_from=datetime.now(timezone.utc),
        ),
        canonical=ClaimCanonical(
            claim_key="test.retracted.question",
            subject=EntityRef(entity_type="task", entity_id="t1", label="Task"),
            predicate="needs_help",
            object=ValueObject(value_type="string", value="help"),
        ),
        statement="Retracted question",
        provenance=Provenance(
            produced_by={"id": "session-a", "method": "manual"},
        ),
    )

    knowledge_store.insert_claim(retracted_question)

    messages = knowledge_store.get_pending_messages(
        session_id="session-b",
        scope_id="multihead",
    )

    # Should not see retracted question
    assert len(messages) == 0
