"""Tests for Context Pack builder."""

import json
import math
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from multihead.context_packs import DEFAULT_WEIGHTS, PackBuilder, cosine_similarity
from multihead.knowledge_models import (
    Claim,
    ClaimCanonical,
    ClaimScope,
    ClaimType,
    ContextPack,
    EntityRef,
    EvidencePointer,
    PackItem,
    Provenance,
    Record,
    ScopeType,
    SpanRef,
    ValueObject,
)
from multihead.knowledge_store import KnowledgeStore


def _prov() -> Provenance:
    return Provenance(produced_by={"kind": "test", "id": "unit"})


@pytest.fixture
def tmp_store(tmp_path):
    return KnowledgeStore(tmp_path / "knowledge.db")


@pytest.fixture
def builder(tmp_store, tmp_path):
    return PackBuilder(tmp_store, tmp_path / "packs")


# -------------------------------------------------------------------
# Scoring
# -------------------------------------------------------------------

class TestScoring:
    def test_trust_score_accepted(self):
        item = PackItem(
            type="claim", text="test", evidence_refs=["x"],
            token_estimate=10, why_included="test",
            source_id="clm_1", status="accepted",
            updated_at=datetime.now(timezone.utc),
        )
        builder = PackBuilder.__new__(PackBuilder)
        assert builder._compute_trust(item) == 1.0

    def test_trust_score_proposed(self):
        item = PackItem(
            type="claim", text="test", evidence_refs=["x"],
            token_estimate=10, why_included="test",
            source_id="clm_1", status="proposed",
            updated_at=datetime.now(timezone.utc),
        )
        builder = PackBuilder.__new__(PackBuilder)
        assert builder._compute_trust(item) == 0.5

    def test_trust_score_contested(self):
        item = PackItem(
            type="claim", text="test", evidence_refs=["x"],
            token_estimate=10, why_included="test",
            source_id="clm_1", status="contested",
            updated_at=datetime.now(timezone.utc),
        )
        builder = PackBuilder.__new__(PackBuilder)
        assert builder._compute_trust(item) == 0.3

    def test_recency_now_is_1(self):
        now = datetime.now(timezone.utc)
        builder = PackBuilder.__new__(PackBuilder)
        score = builder._compute_recency(now)
        assert score > 0.99

    def test_recency_decays(self):
        builder = PackBuilder.__new__(PackBuilder)
        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(days=7)
        score_now = builder._compute_recency(now)
        score_old = builder._compute_recency(week_ago)
        assert score_now > score_old
        # After 1 half-life, should be ~0.5
        assert abs(score_old - 0.5) < 0.05

    def test_score_open_loop_preserved(self, builder):
        item = PackItem(
            type="event", text="[OPEN] test", evidence_refs=["x"],
            token_estimate=10, why_included="open loop",
            source_id="evt_1", status="draft",
            updated_at=datetime.now(timezone.utc),
            priority=999.0,
        )
        score = builder._score_item(item, DEFAULT_WEIGHTS)
        assert score == 999.0

    def test_score_penalty_for_rejected(self, builder):
        item = PackItem(
            type="claim", text="test", evidence_refs=["x"],
            token_estimate=10, why_included="test",
            source_id="clm_1", status="rejected",
            updated_at=datetime.now(timezone.utc),
        )
        score = builder._score_item(item, DEFAULT_WEIGHTS)
        assert score < 0


# -------------------------------------------------------------------
# Budget trimming
# -------------------------------------------------------------------

class TestBudgetTrimming:
    def test_trim_respects_max_items(self, builder):
        items = [
            PackItem(
                type="claim", text=f"item {i}", evidence_refs=[],
                token_estimate=10, why_included="test",
                source_id=f"clm_{i}", status="accepted",
                updated_at=datetime.now(timezone.utc),
                priority=float(10 - i),
            )
            for i in range(10)
        ]
        kept, dropped = builder._trim_to_budget(items, max_tokens=9999, max_items=3)
        assert len(kept) == 3
        assert len(dropped) == 7
        assert all(d["reason"] == "over_max_items" for d in dropped)

    def test_trim_respects_max_tokens(self, builder):
        items = [
            PackItem(
                type="claim", text=f"item {i}", evidence_refs=[],
                token_estimate=100, why_included="test",
                source_id=f"clm_{i}", status="accepted",
                updated_at=datetime.now(timezone.utc),
                priority=float(10 - i),
            )
            for i in range(10)
        ]
        kept, dropped = builder._trim_to_budget(items, max_tokens=250, max_items=99)
        assert len(kept) == 2  # 100 + 100 = 200, 3rd would be 300 > 250
        assert any(d["reason"] == "over_budget" for d in dropped)

    def test_trim_empty(self, builder):
        kept, dropped = builder._trim_to_budget([], max_tokens=100, max_items=10)
        assert kept == []
        assert dropped == []


# -------------------------------------------------------------------
# Token estimation
# -------------------------------------------------------------------

class TestTokenEstimate:
    def test_estimate_basic(self):
        assert PackBuilder._estimate_tokens("hello world") == 2  # 11 // 4

    def test_estimate_minimum_1(self):
        assert PackBuilder._estimate_tokens("") == 1
        assert PackBuilder._estimate_tokens("ab") == 1


# -------------------------------------------------------------------
# Full pack build with real knowledge store
# -------------------------------------------------------------------

class TestBuildPack:
    def test_build_pack_empty_store(self, builder):
        pack = builder.build_pack(purpose="Test Pack")
        assert isinstance(pack, ContextPack)
        assert pack.purpose == "Test Pack"
        assert len(pack.items) == 0

    def test_build_pack_with_claims(self, tmp_store, tmp_path):
        # Insert a record first (needed for FK on evidence)
        rec = Record(uri="test://doc", sha256="abc123", mime="text/plain", provenance=_prov())
        tmp_store.insert_record(rec)

        # Insert a claim with evidence so it becomes accepted
        claim = Claim(
            claim_type=ClaimType.FACT,
            scope=ClaimScope(scope_type=ScopeType.PROJECT, scope_id="proj1"),
            canonical=ClaimCanonical(
                claim_key="test.key",
                subject=EntityRef(entity_type="concept", entity_id="test"),
                predicate="is",
                object=ValueObject(value_type="string", value="true"),
            ),
            statement="Test is true",
            confidence=0.9,
            provenance=_prov(),
        )
        tmp_store.insert_claim(claim)

        # Add evidence
        evp = EvidencePointer(
            record_id=rec.record_id,
            span=SpanRef(start=0, end=10),
            provenance=_prov(),
        )
        tmp_store.insert_evidence(evp)
        tmp_store.link_claim_evidence(claim.claim_id, evp.evidence_id)
        tmp_store.accept_claim(claim.claim_id)

        pb = PackBuilder(tmp_store, tmp_path / "packs2")
        pack = pb.build_pack(purpose="With Claims")
        assert len(pack.items) >= 1
        assert pack.metrics["item_count"] >= 1

    def test_build_writes_files(self, builder):
        pack = builder.build_pack(purpose="File Test")
        md_path = builder.packs_dir / f"{pack.pack_id}.md"
        json_path = builder.packs_dir / f"{pack.pack_id}.pack.json"
        assert md_path.exists()
        assert json_path.exists()
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert data["purpose"] == "File Test"

    def test_build_standard_packs(self, builder):
        packs = builder.build_standard_packs()
        assert len(packs) == 5
        purposes = {p.purpose for p in packs}
        assert "Active Projects" in purposes
        assert "Open Loops" in purposes

    def test_list_packs(self, builder):
        builder.build_pack(purpose="Pack A")
        builder.build_pack(purpose="Pack B")
        listing = builder.list_packs()
        assert len(listing) == 2

    def test_load_pack(self, builder):
        pack = builder.build_pack(purpose="Loadable")
        loaded = builder.load_pack(pack.pack_id)
        assert loaded is not None
        assert loaded.purpose == "Loadable"
        assert loaded.pack_id == pack.pack_id

    def test_load_pack_missing(self, builder):
        assert builder.load_pack("nonexistent") is None


# -------------------------------------------------------------------
# Cosine similarity
# -------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_empty_vectors(self):
        assert cosine_similarity([], []) == 0.0

    def test_mismatched_lengths(self):
        assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0

    def test_zero_vector(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_arbitrary_vectors(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        # Manual: dot=32, |a|=sqrt(14), |b|=sqrt(77), cos=32/sqrt(1078)
        expected = 32.0 / math.sqrt(14 * 77)
        assert cosine_similarity(a, b) == pytest.approx(expected, rel=1e-6)


# -------------------------------------------------------------------
# Embedding-powered relevance
# -------------------------------------------------------------------

class TestEmbeddingRelevance:
    def test_relevance_default_without_embedding(self, builder):
        item = PackItem(
            type="claim", text="test", evidence_refs=["x"],
            token_estimate=10, why_included="test",
            source_id="clm_1", status="accepted",
            updated_at=datetime.now(timezone.utc),
        )
        rel = builder._compute_relevance(item, query_embedding=None)
        assert rel == 0.5

    def test_relevance_default_without_item_embedding(self, builder):
        item = PackItem(
            type="claim", text="test", evidence_refs=["x"],
            token_estimate=10, why_included="test",
            source_id="clm_1", status="accepted",
            updated_at=datetime.now(timezone.utc),
        )
        rel = builder._compute_relevance(item, query_embedding=[1.0, 0.0])
        assert rel == 0.5

    def test_relevance_with_embeddings(self, builder):
        item = PackItem(
            type="claim", text="test", evidence_refs=["x"],
            token_estimate=10, why_included="test",
            source_id="clm_1", status="accepted",
            updated_at=datetime.now(timezone.utc),
            embedding=[1.0, 0.0, 0.0],
        )
        # Same direction → cosine=1.0 → normalized=(1+1)/2=1.0
        rel = builder._compute_relevance(item, query_embedding=[1.0, 0.0, 0.0])
        assert rel == pytest.approx(1.0)

    def test_relevance_orthogonal(self, builder):
        item = PackItem(
            type="claim", text="test", evidence_refs=["x"],
            token_estimate=10, why_included="test",
            source_id="clm_1", status="accepted",
            updated_at=datetime.now(timezone.utc),
            embedding=[1.0, 0.0],
        )
        # Orthogonal → cosine=0.0 → normalized=0.5
        rel = builder._compute_relevance(item, query_embedding=[0.0, 1.0])
        assert rel == pytest.approx(0.5)

    def test_score_with_embedding_boosts_relevant(self, builder):
        """Items with high similarity should score higher than default."""
        item_similar = PackItem(
            type="claim", text="matching topic", evidence_refs=["x"],
            token_estimate=10, why_included="test",
            source_id="clm_1", status="accepted",
            updated_at=datetime.now(timezone.utc),
            embedding=[1.0, 0.0, 0.0],
        )
        item_default = PackItem(
            type="claim", text="unrelated", evidence_refs=["y"],
            token_estimate=10, why_included="test",
            source_id="clm_2", status="accepted",
            updated_at=datetime.now(timezone.utc),
            # No embedding → falls back to 0.5
        )
        query_emb = [1.0, 0.0, 0.0]
        score_similar = builder._score_item(
            item_similar, DEFAULT_WEIGHTS, query_embedding=query_emb,
        )
        score_default = builder._score_item(
            item_default, DEFAULT_WEIGHTS, query_embedding=query_emb,
        )
        assert score_similar > score_default

    @pytest.mark.asyncio
    async def test_build_pack_with_embeddings(self, tmp_store, tmp_path):
        """Async build with mock embedding adapter."""
        mock_adapter = MagicMock()
        mock_adapter.embed = AsyncMock(return_value={"embedding": [1.0, 0.0, 0.0]})
        mock_adapter.embed_batch = AsyncMock(return_value={
            "embeddings": [],
            "count": 0,
        })

        pb = PackBuilder(tmp_store, tmp_path / "packs_emb", embedding_adapter=mock_adapter)
        pack = await pb.build_pack_with_embeddings(
            purpose="Semantic Pack", query="test query",
        )
        assert isinstance(pack, ContextPack)
        mock_adapter.embed.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_build_pack_with_embeddings_fallback(self, tmp_store, tmp_path):
        """Falls back gracefully when embedding fails."""
        mock_adapter = MagicMock()
        mock_adapter.embed = AsyncMock(side_effect=RuntimeError("GPU busy"))

        pb = PackBuilder(tmp_store, tmp_path / "packs_fb", embedding_adapter=mock_adapter)
        pack = await pb.build_pack_with_embeddings(
            purpose="Fallback Pack", query="test",
        )
        assert isinstance(pack, ContextPack)

    @pytest.mark.asyncio
    async def test_build_pack_with_embeddings_no_adapter(self, tmp_store, tmp_path):
        """Without adapter, falls back to sync build_pack."""
        pb = PackBuilder(tmp_store, tmp_path / "packs_no")
        pack = await pb.build_pack_with_embeddings(
            purpose="No Adapter Pack", query="test",
        )
        assert isinstance(pack, ContextPack)
        assert pack.purpose == "No Adapter Pack"
