"""Tests for LLM-powered extractors."""


import pytest

from multihead.adapters.mock import MockAdapter
from multihead.chunker import Chunk
from multihead.extractors.base import BaseExtractor, ExtractorResult
from multihead.extractors.claim_extractor import ClaimExtractor
from multihead.extractors.consistency_checker import ConsistencyChecker
from multihead.extractors.entity_extractor import EntityExtractor
from multihead.extractors.event_extractor import EventExtractor
from multihead.extractors.topic_assigner import TopicAssigner
from multihead.models import AdapterKind, HeadManifest


@pytest.fixture
def mock_adapter():
    manifest = HeadManifest(
        head_id="mock-llm", name="Mock", adapter=AdapterKind.MOCK,
        model="mock-v1", kind="llm", gpu_required=False,
    )
    return MockAdapter(manifest)


@pytest.fixture
def sample_chunks():
    return [
        Chunk(chunk_id="chk_1", record_id="rec_1",
              text="MultiHead is a local orchestration layer. "
                   "It uses event sourcing for durability.",
              span_start=0, span_end=80),
        Chunk(chunk_id="chk_2", record_id="rec_2",
              text="The core LLM runs on CPU by default. VRAM is reserved for worker models.",
              span_start=0, span_end=72),
    ]


# -------------------------------------------------------------------
# Base extractor JSON parsing
# -------------------------------------------------------------------

class TestJsonParsing:
    def test_parse_valid_array(self):
        text = '[{"entity_type": "project", "entity_id": "multihead"}]'
        result = BaseExtractor.parse_json_response(text)
        assert len(result) == 1
        assert result[0]["entity_id"] == "multihead"

    def test_parse_valid_object(self):
        text = '{"entity_type": "project", "entity_id": "multihead"}'
        result = BaseExtractor.parse_json_response(text)
        assert len(result) == 1

    def test_parse_markdown_code_block(self):
        text = 'Here are the entities:\n```json\n[{"id": "test"}]\n```'
        result = BaseExtractor.parse_json_response(text)
        assert len(result) == 1

    def test_parse_embedded_json(self):
        text = 'Found entities: [{"id": "test"}] in the text.'
        result = BaseExtractor.parse_json_response(text)
        assert len(result) == 1

    def test_parse_garbage(self):
        text = "This is not JSON at all, just random text."
        result = BaseExtractor.parse_json_response(text)
        assert result == []

    def test_parse_string_array_returns_empty(self):
        """LLM sometimes returns string arrays instead of object arrays.
        These must be filtered out to prevent 'str' object does not support
        item assignment errors in extractors."""
        text = '["entity1", "entity2", "entity3"]'
        result = BaseExtractor.parse_json_response(text)
        assert result == []

    def test_parse_mixed_array_keeps_dicts(self):
        """Mixed arrays should keep only dict items."""
        text = '[{"id": "good"}, "bad_string", {"id": "also_good"}]'
        result = BaseExtractor.parse_json_response(text)
        assert len(result) == 2
        assert result[0]["id"] == "good"
        assert result[1]["id"] == "also_good"

    def test_parse_partial_json(self):
        text = 'Some text {"key": "value"} more text'
        result = BaseExtractor.parse_json_response(text)
        assert len(result) == 1

    def test_parse_qwen_thinking_blocks(self):
        text = (
            '<think>\nLet me analyze this...\n'
            'I see entities here.\n</think>\n'
            '[{"entity_type":"model","entity_id":"qwen3",'
            '"label":"Qwen3"}]'
        )
        result = BaseExtractor.parse_json_response(text)
        assert len(result) == 1
        assert result[0]["entity_id"] == "qwen3"

    def test_parse_truncated_json_array(self):
        """Qwen hits token limit mid-JSON — repair should recover partial results."""
        text = '[{"key":"a","val":"1"},{"key":"b","val":"2"},{"key":"c","val":"tr'
        result = BaseExtractor.parse_json_response(text)
        assert len(result) >= 2  # Should recover at least the first 2 complete objects

    def test_parse_trailing_comma(self):
        text = '[{"a":1},{"b":2},]'
        result = BaseExtractor.parse_json_response(text)
        assert len(result) == 2

    def test_strip_thinking_empty(self):
        assert BaseExtractor._strip_thinking("no thinking here") == "no thinking here"

    def test_strip_thinking_multiline(self):
        text = "<think>\nlong reasoning\nover lines\n</think>\nactual output"
        assert BaseExtractor._strip_thinking(text) == "actual output"


# -------------------------------------------------------------------
# Entity extractor
# -------------------------------------------------------------------

class TestEntityExtractor:
    @pytest.mark.asyncio
    async def test_extract_with_mock(self, mock_adapter, sample_chunks):
        await mock_adapter.load()
        extractor = EntityExtractor()
        result = await extractor.extract(sample_chunks, mock_adapter)
        assert isinstance(result, ExtractorResult)
        assert "entity_yield_per_1k_tokens" in result.metrics
        assert "alias_conflict_rate" in result.metrics
        assert result.metrics["chunks_processed"] == 2

    def test_canonicalize(self):
        extractor = EntityExtractor()
        entities = [
            {"entity_type": "project", "entity_id": "multihead",
             "label": "MultiHead", "aliases": ["MH"]},
            {"entity_type": "project", "entity_id": "multihead",
             "label": "Multihead", "aliases": ["multi-head"]},
        ]
        deduped = extractor._canonicalize(entities)
        assert len(deduped) == 1
        assert "MH" in deduped[0]["aliases"]
        assert "multi-head" in deduped[0]["aliases"]

    def test_alias_conflict_rate(self):
        extractor = EntityExtractor()
        # No conflicts
        entities = [
            {"entity_id": "a", "aliases": ["x"]},
            {"entity_id": "b", "aliases": ["y"]},
        ]
        assert extractor._compute_alias_conflicts(entities) == 0.0

        # Conflict: alias "x" maps to both "a" and "b"
        entities = [
            {"entity_id": "a", "aliases": ["x"]},
            {"entity_id": "b", "aliases": ["x"]},
        ]
        assert extractor._compute_alias_conflicts(entities) == 1.0


# -------------------------------------------------------------------
# Topic assigner
# -------------------------------------------------------------------

class TestTopicAssigner:
    @pytest.mark.asyncio
    async def test_extract_with_mock(self, mock_adapter, sample_chunks):
        await mock_adapter.load()
        assigner = TopicAssigner()
        result = await assigner.extract(sample_chunks, mock_adapter)
        assert isinstance(result, ExtractorResult)
        assert "unassigned_chunk_rate" in result.metrics
        assert "topic_coherence" in result.metrics

    @pytest.mark.asyncio
    async def test_empty_chunks(self, mock_adapter):
        await mock_adapter.load()
        assigner = TopicAssigner()
        result = await assigner.extract([], mock_adapter)
        assert result.metrics["unassigned_chunk_rate"] == 0.0


# -------------------------------------------------------------------
# Event extractor
# -------------------------------------------------------------------

class TestEventExtractor:
    @pytest.mark.asyncio
    async def test_extract_with_mock(self, mock_adapter, sample_chunks):
        await mock_adapter.load()
        extractor = EventExtractor()
        result = await extractor.extract(sample_chunks, mock_adapter)
        assert isinstance(result, ExtractorResult)
        assert "event_extract_coverage" in result.metrics
        assert result.metrics["chunks_processed"] == 2


# -------------------------------------------------------------------
# Claim extractor
# -------------------------------------------------------------------

class TestClaimExtractor:
    @pytest.mark.asyncio
    async def test_extract_with_mock(self, mock_adapter, sample_chunks):
        await mock_adapter.load()
        extractor = ClaimExtractor()
        result = await extractor.extract(sample_chunks, mock_adapter)
        assert isinstance(result, ExtractorResult)
        assert "claim_count" in result.metrics

    def test_deduplicate(self):
        extractor = ClaimExtractor()
        claims = [
            {"claim_key": "k1", "statement": "A", "confidence": 0.8},
            {"claim_key": "k1", "statement": "A better", "confidence": 0.9},
            {"claim_key": "k2", "statement": "B", "confidence": 0.7},
        ]
        deduped = extractor._deduplicate(claims)
        assert len(deduped) == 2
        k1 = next(c for c in deduped if c["claim_key"] == "k1")
        assert k1["confidence"] == 0.9

    def test_auto_accept_logic(self):
        extractor = ClaimExtractor(
            auto_accept_confidence=0.85,
            auto_accept_min_supports=2,
            auto_accept_types=("definition", "decision"),
        )
        all_claims = [
            {"claim_key": "k1", "claim_type": "decision", "confidence": 0.9,
             "source_record_id": "rec_1"},
            {"claim_key": "k1", "claim_type": "decision", "confidence": 0.9,
             "source_record_id": "rec_2"},
        ]
        assert extractor._should_auto_accept(all_claims[0], all_claims) is True

        # Not enough confidence
        low_conf = {"claim_key": "k2", "claim_type": "decision", "confidence": 0.5,
                     "source_record_id": "rec_1"}
        assert extractor._should_auto_accept(low_conf, [low_conf]) is False

        # Wrong type
        wrong_type = {"claim_key": "k3", "claim_type": "assumption", "confidence": 0.95,
                       "source_record_id": "rec_1"}
        assert extractor._should_auto_accept(wrong_type, [wrong_type]) is False


# -------------------------------------------------------------------
# Consistency checker
# -------------------------------------------------------------------

class TestConsistencyChecker:
    @pytest.mark.asyncio
    async def test_check_with_mock(self, mock_adapter):
        await mock_adapter.load()
        checker = ConsistencyChecker()
        claims = [
            {"claim_key": "k1", "statement": "Core runs on CPU"},
            {"claim_key": "k2", "statement": "Core runs on GPU"},
        ]
        result = await checker.extract([], mock_adapter, claims=claims)
        assert isinstance(result, ExtractorResult)
        assert "contradiction_count" in result.metrics
        assert result.metrics["claims_checked"] == 2

    @pytest.mark.asyncio
    async def test_single_claim(self, mock_adapter):
        await mock_adapter.load()
        checker = ConsistencyChecker()
        result = await checker.extract([], mock_adapter, claims=[{"claim_key": "k1"}])
        assert result.metrics["contradiction_count"] == 0
