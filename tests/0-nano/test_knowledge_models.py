"""Tests for knowledge layer models."""

from datetime import datetime, timezone

from multihead.knowledge_models import (
    Claim,
    ClaimCanonical,
    ClaimScope,
    ClaimStatus,
    ClaimType,
    ContextPack,
    EntityRef,
    EvidencePointer,
    EventStatus,
    EventType,
    KnowledgeEvent,
    Link,
    NightShiftConfig,
    NightShiftReport,
    PackItem,
    Provenance,
    Record,
    ScopeType,
    SpanRef,
    Stability,
    TimeBlock,
    TimePrecision,
    ValueObject,
)


def _make_provenance():
    return Provenance(produced_by={"kind": "extractor", "id": "test"})


def test_record_auto_id():
    r = Record(uri="file:///test.txt")
    assert r.record_id.startswith("rec_")
    assert r.uri == "file:///test.txt"


def test_evidence_pointer_auto_id():
    ep = EvidencePointer(
        record_id="rec_abc123",
        uri="file:///test.txt",
        span=SpanRef(start=0, end=100),
    )
    assert ep.evidence_id.startswith("evp_")
    assert ep.span.unit == "chars"


def test_knowledge_event_auto_id():
    evt = KnowledgeEvent(
        event_type=EventType.DECISION,
        title="Test decision",
        time=TimeBlock(happened_at=datetime.now(timezone.utc)),
        provenance=_make_provenance(),
    )
    assert evt.event_id.startswith("evt_")
    assert evt.event_status == EventStatus.DRAFT
    assert evt.event_type == EventType.DECISION


def test_claim_auto_id():
    claim = Claim(
        claim_type=ClaimType.DECISION,
        scope=ClaimScope(scope_type=ScopeType.PROJECT, scope_id="multihead"),
        canonical=ClaimCanonical(
            claim_key="project.multihead.core_model",
            subject=EntityRef(entity_type="component", entity_id="core_llm"),
            predicate="runs_on",
            object=ValueObject(value_type="enum", value="cpu"),
        ),
        statement="Core LLM runs on CPU by default.",
        provenance=_make_provenance(),
    )
    assert claim.claim_id.startswith("clm_")
    assert claim.claim_status == ClaimStatus.PROPOSED
    assert claim.confidence == 0.0


def test_link_auto_id():
    link = Link(
        from_entity=EntityRef(entity_type="project", entity_id="multihead"),
        to_entity=EntityRef(entity_type="project", entity_id="botvibes"),
        reason_type="shared_constraint",
        provenance=_make_provenance(),
    )
    assert link.link_id.startswith("lnk_")
    assert link.link_status == "draft"


def test_context_pack_auto_id():
    pack = ContextPack(purpose="test pack")
    assert pack.pack_id.startswith("pack_")
    assert pack.budgets["max_tokens"] == 4000


def test_nightshift_report_auto_id():
    report = NightShiftReport()
    assert report.report_id.startswith("nsr_")
    assert report.records_processed == 0


def test_nightshift_config_defaults():
    config = NightShiftConfig()
    assert config.max_tokens_per_day == 100_000
    assert config.auto_accept_confidence == 0.85
    assert config.auto_accept_min_supports == 2


def test_claim_serialization_roundtrip():
    claim = Claim(
        claim_type=ClaimType.FACT,
        scope=ClaimScope(scope_type=ScopeType.GLOBAL, scope_id="global"),
        canonical=ClaimCanonical(
            claim_key="test.key",
            subject=EntityRef(entity_type="concept", entity_id="test"),
            predicate="is",
            object=ValueObject(value_type="string", value="true"),
        ),
        statement="Test claim.",
        confidence=0.9,
        stability=Stability.STABLE,
        importance=0.8,
        provenance=_make_provenance(),
    )
    data = claim.model_dump(mode="json")
    restored = Claim.model_validate(data)
    assert restored.claim_id == claim.claim_id
    assert restored.confidence == 0.9
    assert restored.canonical.claim_key == "test.key"


def test_knowledge_event_serialization_roundtrip():
    evt = KnowledgeEvent(
        event_type=EventType.TOOL_RUN,
        title="Ran pipeline",
        time=TimeBlock(
            happened_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            time_precision=TimePrecision.SECOND,
        ),
        tags=["pipeline", "test"],
        provenance=_make_provenance(),
    )
    data = evt.model_dump(mode="json")
    restored = KnowledgeEvent.model_validate(data)
    assert restored.event_id == evt.event_id
    assert restored.tags == ["pipeline", "test"]


def test_pack_item_fields():
    item = PackItem(
        type="claim",
        text="Core LLM runs on CPU.",
        priority=0.85,
        evidence_refs=["evt_abc"],
        token_estimate=12,
        why_included="accepted decision",
        source_id="clm_abc",
        status="accepted",
    )
    assert item.priority == 0.85
    assert item.token_estimate == 12
