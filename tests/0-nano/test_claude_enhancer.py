"""Tests for the Claude-enhanced markdown extraction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from multihead.narrative.claude_enhancer import ClaudeEnhancer
from multihead.narrative.claude_enhancer.parsing import (
    split_sections,
    parse_claude_output,
    convert_to_claims,
    merge_claims,
)
from multihead.knowledge_models import Claim, ClaimType, ClaimScope, ScopeType, Stability
from multihead.narrative.confidence import ConfidenceCalibrator


@pytest.fixture
def enhancer():
    return ClaudeEnhancer(
        acp_url="http://localhost:8000/api/v1",
        api_key="test-token",
        project_id="test_project",
        poll_interval=0.1,  # Fast polling for tests
        max_wait=5.0,
    )


@pytest.fixture
def plan_doc(tmp_path: Path) -> Path:
    """A planning document with multiple H2 sections."""
    doc = tmp_path / "PLAN.md"
    doc.write_text(
        "# Project Expansion Plan\n\n"
        "## Phase 1: Foundation\n\n"
        "- Install dependencies and configure the base environment\n"
        "- Create the database schema for character tracking tables\n"
        "- Set up the VLM inference pipeline for panel detection\n"
        "\n"
        "## Phase 2: Detection Pipeline\n\n"
        "- [x] Implement YOLO-based panel detection model\n"
        "- [ ] Add speech bubble segmentation using SAM\n"
        "- Integrate OCR engine for text extraction from bubbles\n"
        "- Must support both English and Japanese text extraction\n"
        "\n"
        "## Phase 3: Assembly\n\n"
        "- Build narrative assembly pipeline from detected elements\n"
        "- Implement speaker attribution using character embeddings\n"
        "- Create reading order algorithm for panel sequencing\n",
        encoding="utf-8",
    )
    return doc


@pytest.fixture
def short_doc(tmp_path: Path) -> Path:
    """A document with sections too short to extract."""
    doc = tmp_path / "SHORT.md"
    doc.write_text(
        "# Title\n\n"
        "## Section A\n\n"
        "Too short\n"
        "\n"
        "## Section B\n\n"
        "Also very short\n",
        encoding="utf-8",
    )
    return doc


@pytest.fixture
def empty_doc(tmp_path: Path) -> Path:
    doc = tmp_path / "EMPTY.md"
    doc.write_text("", encoding="utf-8")
    return doc


# ------------------------------------------------------------------
# Section splitting tests
# ------------------------------------------------------------------


class TestSectionSplitting:
    """Test H2-level section splitting."""

    def test_splits_h2_sections(self, enhancer: ClaudeEnhancer, plan_doc: Path):
        content = plan_doc.read_text()
        sections = split_sections(content)
        assert len(sections) == 3
        assert sections[0]["heading"] == "Phase 1: Foundation"
        assert sections[1]["heading"] == "Phase 2: Detection Pipeline"
        assert sections[2]["heading"] == "Phase 3: Assembly"

    def test_skips_short_sections(self, enhancer: ClaudeEnhancer, short_doc: Path):
        content = short_doc.read_text()
        sections = split_sections(content)
        assert len(sections) == 0  # Both sections too short (<50 chars)

    def test_empty_content(self, enhancer: ClaudeEnhancer):
        sections = split_sections("")
        assert sections == []

    def test_no_headings(self, enhancer: ClaudeEnhancer):
        content = "Just some text without any headings.\nAnother line."
        sections = split_sections(content)
        assert sections == []

    def test_section_text_includes_content(self, enhancer: ClaudeEnhancer, plan_doc: Path):
        content = plan_doc.read_text()
        sections = split_sections(content)
        assert "Install dependencies" in sections[0]["text"]
        assert "YOLO" in sections[1]["text"]


# ------------------------------------------------------------------
# Prompt building tests
# ------------------------------------------------------------------


class TestPromptBuilding:
    """Test section prompt construction."""

    def test_prompt_includes_section_text(self, enhancer: ClaudeEnhancer):
        prompt = enhancer._build_section_prompt(
            "Phase 1", "- Install deps\n- Build schema", "plan", "PLAN",
        )
        assert "Install deps" in prompt
        assert "Phase 1" in prompt
        assert "plan" in prompt
        assert "PLAN" in prompt

    def test_prompt_caps_section_size(self, enhancer: ClaudeEnhancer):
        long_text = "x" * 10000
        prompt = enhancer._build_section_prompt("Test", long_text, "plan", "DOC")
        # Section text should be capped at 6000 chars
        assert len(prompt) < 10000 + 2000  # Template overhead

    def test_prompt_requests_json_output(self, enhancer: ClaudeEnhancer):
        prompt = enhancer._build_section_prompt("Test", "Some content here", "plan", "DOC")
        assert "```json" in prompt
        assert '"claims"' in prompt


# ------------------------------------------------------------------
# Output parsing tests
# ------------------------------------------------------------------


class TestOutputParsing:
    """Test parsing Claude's structured output."""

    def test_parses_clean_json(self, enhancer: ClaudeEnhancer):
        output = json.dumps({
            "claims": [
                {
                    "text": "The VLM pipeline requires entity UUIDs",
                    "claim_type": "constraint",
                    "predicate": "requires",
                    "confidence": 0.85,
                    "entities": ["vlm_pipeline"],
                    "reasoning": "Explicit requirement",
                }
            ]
        })
        result = parse_claude_output(output)
        assert len(result) == 1
        assert result[0]["text"] == "The VLM pipeline requires entity UUIDs"

    def test_parses_markdown_code_block(self, enhancer: ClaudeEnhancer):
        output = (
            "Here are the extracted claims:\n\n"
            "```json\n"
            '{"claims": [{"text": "OCR supports Japanese", "claim_type": "fact", '
            '"predicate": "supports", "confidence": 0.90, "entities": ["ocr"], '
            '"reasoning": "Stated in doc"}]}\n'
            "```\n"
        )
        result = parse_claude_output(output)
        assert len(result) == 1
        assert "Japanese" in result[0]["text"]

    def test_parses_embedded_json(self, enhancer: ClaudeEnhancer):
        output = (
            "I found the following claims:\n"
            '{"claims": [{"text": "Panel detection uses YOLO", '
            '"claim_type": "fact", "predicate": "uses", '
            '"confidence": 0.9, "entities": [], "reasoning": "fact"}]}'
        )
        result = parse_claude_output(output)
        assert len(result) == 1

    def test_handles_empty_output(self, enhancer: ClaudeEnhancer):
        assert parse_claude_output("") == []

    def test_handles_invalid_json(self, enhancer: ClaudeEnhancer):
        assert parse_claude_output("not json at all") == []

    def test_handles_json_list(self, enhancer: ClaudeEnhancer):
        output = json.dumps([
            {"text": "Claim A", "claim_type": "fact"},
            {"text": "Claim B", "claim_type": "plan"},
        ])
        result = parse_claude_output(output)
        assert len(result) == 2

    def test_parses_empty_claims(self, enhancer: ClaudeEnhancer):
        output = json.dumps({"claims": []})
        result = parse_claude_output(output)
        assert result == []


# ------------------------------------------------------------------
# Claim conversion tests
# ------------------------------------------------------------------


class TestClaimConversion:
    """Test converting parsed JSON to Claim objects."""

    def test_converts_basic_claim(self, enhancer: ClaudeEnhancer):
        parsed = [{
            "text": "The detection pipeline requires CUDA 12 support",
            "claim_type": "constraint",
            "predicate": "requires",
            "confidence": 0.85,
            "entities": ["detection_pipeline"],
            "reasoning": "Explicit constraint",
        }]
        scope = ClaimScope(scope_type=ScopeType.PROJECT, scope_id="test")
        claims = convert_to_claims(
            parsed, "plan", "test", scope, Stability.MEDIUM, "Phase 1", ConfidenceCalibrator(),
        )
        assert len(claims) == 1
        claim = claims[0]
        assert claim.claim_type == ClaimType.CONSTRAINT
        assert claim.canonical.predicate == "requires"
        assert "CUDA 12" in claim.statement

    def test_skips_short_claims(self, enhancer: ClaudeEnhancer):
        parsed = [{"text": "short", "claim_type": "fact"}]
        scope = ClaimScope(scope_type=ScopeType.PROJECT, scope_id="test")
        claims = convert_to_claims(
            parsed, "doc", "test", scope, Stability.MEDIUM, "Section", ConfidenceCalibrator(),
        )
        assert len(claims) == 0

    def test_claim_key_format(self, enhancer: ClaudeEnhancer):
        parsed = [{
            "text": "The baseline detection model is fully trained and validated",
            "claim_type": "fact",
            "predicate": "completed",
            "confidence": 0.90,
        }]
        scope = ClaimScope(scope_type=ScopeType.PROJECT, scope_id="h2v")
        claims = convert_to_claims(
            parsed, "plan", "h2v", scope,
            Stability.MEDIUM, "Phase 1: Setup",
            ConfidenceCalibrator(),
        )
        assert len(claims) == 1
        key = claims[0].canonical.claim_key
        assert key.startswith("claude.h2v.plan.phase_1_setup.")

    def test_confidence_capped_at_llm_inference(self, enhancer: ClaudeEnhancer):
        parsed = [{
            "text": "This claim has unreasonably high confidence value",
            "claim_type": "fact",
            "predicate": "states",
            "confidence": 0.99,
        }]
        scope = ClaimScope(scope_type=ScopeType.PROJECT, scope_id="test")
        claims = convert_to_claims(
            parsed, "doc", "test", scope, Stability.MEDIUM, "Section", ConfidenceCalibrator(),
        )
        # LLM_INFERENCE cap is 0.75
        assert claims[0].confidence <= 0.75

    def test_maps_all_claim_types(self, enhancer: ClaudeEnhancer):
        for ct in ("fact", "plan", "decision", "constraint", "risk", "assumption"):
            parsed = [{
                "text": f"A {ct} claim with enough text to pass the filter",
                "claim_type": ct,
            }]
            scope = ClaimScope(scope_type=ScopeType.PROJECT, scope_id="test")
            claims = convert_to_claims(
                parsed, "doc", "test", scope, Stability.MEDIUM, "Section", ConfidenceCalibrator(),
            )
            assert len(claims) == 1
            assert claims[0].claim_type.value == ct

    def test_unknown_claim_type_defaults_to_fact(self, enhancer: ClaudeEnhancer):
        parsed = [{
            "text": "Some unknown type claim with sufficient length",
            "claim_type": "foobar",
        }]
        scope = ClaimScope(scope_type=ScopeType.PROJECT, scope_id="test")
        claims = convert_to_claims(
            parsed, "doc", "test", scope, Stability.MEDIUM, "Section", ConfidenceCalibrator(),
        )
        assert claims[0].claim_type == ClaimType.FACT


# ------------------------------------------------------------------
# Merge tests
# ------------------------------------------------------------------


class TestMergeClaims:
    """Test merging heuristic and Claude claims."""

    def _make_claim(self, key: str, source: str = "doc") -> Claim:
        from multihead.knowledge_models import (
            ClaimCanonical, ClaimScope, EntityRef, Provenance,
            ScopeType, Stability, ValueObject,
        )
        return Claim(
            claim_type=ClaimType.FACT,
            scope=ClaimScope(scope_type=ScopeType.PROJECT, scope_id="test"),
            canonical=ClaimCanonical(
                claim_key=key,
                subject=EntityRef(entity_type="doc", entity_id="test", label="test"),
                predicate="states",
                object=ValueObject(value_type="string", value="test"),
            ),
            statement="test claim",
            confidence=0.80,
            stability=Stability.MEDIUM,
            provenance=Provenance(
                produced_by={"kind": "extractor", "id": source},
            ),
        )

    def test_no_overlap(self, enhancer: ClaudeEnhancer):
        h = [self._make_claim("heuristic.key1")]
        c = [self._make_claim("claude.key2")]
        merged = merge_claims(h, c)
        assert len(merged) == 2

    def test_heuristic_wins_on_collision(self, enhancer: ClaudeEnhancer):
        h = [self._make_claim("same.key", "heuristic")]
        c = [self._make_claim("same.key", "claude")]
        merged = merge_claims(h, c)
        assert len(merged) == 1
        assert merged[0].provenance.produced_by["id"] == "heuristic"

    def test_empty_inputs(self, enhancer: ClaudeEnhancer):
        assert merge_claims([], []) == []

    def test_only_heuristic(self, enhancer: ClaudeEnhancer):
        h = [self._make_claim("h.1"), self._make_claim("h.2")]
        merged = merge_claims(h, [])
        assert len(merged) == 2

    def test_only_claude(self, enhancer: ClaudeEnhancer):
        c = [self._make_claim("c.1"), self._make_claim("c.2")]
        merged = merge_claims([], c)
        assert len(merged) == 2


# ------------------------------------------------------------------
# Full enhance_document tests (mocked ACP)
# ------------------------------------------------------------------


