"""Tests for the Markdown narrative extractor."""

from __future__ import annotations

from pathlib import Path

import pytest

from multihead.narrative.source_extractors.markdown_extractor import MarkdownExtractor
from multihead.narrative.confidence import SourcePriority


@pytest.fixture
def extractor():
    return MarkdownExtractor(project_id="test_project")


@pytest.fixture
def plan_doc(tmp_path: Path) -> Path:
    """A simple planning document."""
    doc = tmp_path / "PLAN.md"
    doc.write_text(
        "# My Plan\n\n"
        "## Phase 1: Setup\n\n"
        "- Install dependencies and configure environment\n"
        "- Create the database schema for user tables\n"
        "- Short\n"  # Too short, should be skipped
        "\n"
        "## Phase 2: Implementation\n\n"
        "- [x] Implement user authentication flow\n"
        "- [ ] Add password reset functionality\n"
        "- Integrate with OAuth providers for social login\n"
        "\n"
        "## Phase 3: Testing\n\n"
        "1. Write unit tests for all auth endpoints\n"
        "2. Run integration tests against staging\n",
        encoding="utf-8",
    )
    return doc


@pytest.fixture
def status_doc(tmp_path: Path) -> Path:
    """A status document with issues."""
    doc = tmp_path / "STATUS.md"
    doc.write_text(
        "# Project Status\n\n"
        "## Current State\n\n"
        "- The baseline is INVALID due to missing OCR text\n"
        "- Speaker attribution is broken in 14 of 21 cases\n"
        "\n"
        "## What Works\n\n"
        "- Pipeline stages 1-6 are running correctly\n"
        "- [x] Detection model trained and deployed\n",
        encoding="utf-8",
    )
    return doc


@pytest.fixture
def fixes_doc(tmp_path: Path) -> Path:
    """A fixes-required document."""
    doc = tmp_path / "FIXES.md"
    doc.write_text(
        "# Fixes Required\n\n"
        "## Critical Bug: UUID Matching\n\n"
        "- Fix VLM enrichment to include entity UUIDs\n"
        "- Implement backfill script to match by UUID\n"
        "\n"
        "## Minor Issues\n\n"
        "- Update configuration defaults for production\n",
        encoding="utf-8",
    )
    return doc


@pytest.fixture
def empty_doc(tmp_path: Path) -> Path:
    """An empty markdown file."""
    doc = tmp_path / "EMPTY.md"
    doc.write_text("", encoding="utf-8")
    return doc


class TestMarkdownExtractorBasic:
    """Basic extraction from plan documents."""

    def test_extracts_claims_from_plan(self, extractor: MarkdownExtractor, plan_doc: Path):
        artifacts = extractor.extract_from_file(plan_doc, doc_type="plan")
        assert len(artifacts) == 1
        claims = artifacts[0]["claims"]
        # Should extract bullets with len > 10 (skips "Short")
        assert len(claims) >= 6

    def test_returns_record(self, extractor: MarkdownExtractor, plan_doc: Path):
        artifacts = extractor.extract_from_file(plan_doc, doc_type="plan")
        record = artifacts[0]["record"]
        assert record.uri.startswith("markdown://")
        assert record.mime == "text/markdown"
        assert len(record.sha256) == 64

    def test_returns_event(self, extractor: MarkdownExtractor, plan_doc: Path):
        artifacts = extractor.extract_from_file(plan_doc, doc_type="plan")
        event = artifacts[0]["event"]
        assert event is not None
        assert "PLAN.md" in event.title
        assert event.tags  # Should have tags

    def test_returns_evidence(self, extractor: MarkdownExtractor, plan_doc: Path):
        artifacts = extractor.extract_from_file(plan_doc, doc_type="plan")
        evidence = artifacts[0]["evidence"]
        assert len(evidence) >= 1
        assert all(ep.record_id == artifacts[0]["record"].record_id for ep in evidence)

    def test_empty_file_returns_empty(self, extractor: MarkdownExtractor, empty_doc: Path):
        artifacts = extractor.extract_from_file(empty_doc, doc_type="plan")
        assert artifacts == []

    def test_missing_file_returns_empty(self, extractor: MarkdownExtractor, tmp_path: Path):
        artifacts = extractor.extract_from_file(tmp_path / "nonexistent.md")
        assert artifacts == []


class TestClaimKeys:
    """Claim key format: doc.<scope>.<doc_id>.<section>.<hash>."""

    def test_claim_key_format(self, extractor: MarkdownExtractor, plan_doc: Path):
        artifacts = extractor.extract_from_file(plan_doc, doc_type="plan")
        claims = artifacts[0]["claims"]
        for claim in claims:
            key = claim.canonical.claim_key
            assert key.startswith("doc.test_project.plan.")

    def test_custom_source_project(self, extractor: MarkdownExtractor, plan_doc: Path):
        artifacts = extractor.extract_from_file(
            plan_doc, doc_type="plan", source_project="h2v",
        )
        claims = artifacts[0]["claims"]
        for claim in claims:
            assert claim.canonical.claim_key.startswith("doc.h2v.")
            assert claim.scope.scope_id == "h2v"

    def test_claim_keys_are_unique(self, extractor: MarkdownExtractor, plan_doc: Path):
        artifacts = extractor.extract_from_file(plan_doc, doc_type="plan")
        claims = artifacts[0]["claims"]
        keys = [c.canonical.claim_key for c in claims]
        assert len(keys) == len(set(keys))


class TestCheckboxDetection:
    """Checkbox state maps to predicate and claim type."""

    def test_checked_becomes_fact(self, extractor: MarkdownExtractor, plan_doc: Path):
        artifacts = extractor.extract_from_file(plan_doc, doc_type="plan")
        claims = artifacts[0]["claims"]
        checked = [c for c in claims if c.canonical.predicate == "completed_per_doc"]
        assert len(checked) >= 1
        for c in checked:
            assert c.claim_type.value == "fact"

    def test_unchecked_becomes_pending(self, extractor: MarkdownExtractor, plan_doc: Path):
        artifacts = extractor.extract_from_file(plan_doc, doc_type="plan")
        claims = artifacts[0]["claims"]
        unchecked = [c for c in claims if c.canonical.predicate == "planned_pending"]
        assert len(unchecked) >= 1

    def test_regular_bullet_is_planned(self, extractor: MarkdownExtractor, plan_doc: Path):
        artifacts = extractor.extract_from_file(plan_doc, doc_type="plan")
        claims = artifacts[0]["claims"]
        planned = [c for c in claims if c.canonical.predicate == "planned"]
        assert len(planned) >= 1


class TestDocTypes:
    """Different doc types produce different claim types."""

    def test_status_doc_defaults_to_fact(self, extractor: MarkdownExtractor, status_doc: Path):
        artifacts = extractor.extract_from_file(status_doc, doc_type="status")
        claims = artifacts[0]["claims"]
        # Non-checkbox items in status docs should detect issues
        issue_claims = [c for c in claims if c.canonical.predicate == "has_issue"]
        assert len(issue_claims) >= 1

    def test_fixes_doc_defaults_to_fix_required(
        self, extractor: MarkdownExtractor, fixes_doc: Path,
    ):
        artifacts = extractor.extract_from_file(fixes_doc, doc_type="fixes")
        claims = artifacts[0]["claims"]
        fix_claims = [c for c in claims if c.canonical.predicate == "fix_required"]
        assert len(fix_claims) >= 1

    def test_plan_claims_are_plan_type(self, extractor: MarkdownExtractor, plan_doc: Path):
        artifacts = extractor.extract_from_file(plan_doc, doc_type="plan")
        claims = artifacts[0]["claims"]
        plan_claims = [c for c in claims if c.claim_type.value == "plan"]
        assert len(plan_claims) >= 1


class TestConfidence:
    """Confidence calibration uses PLANNING_DOCUMENT source."""

    def test_confidence_within_cap(self, extractor: MarkdownExtractor, plan_doc: Path):
        artifacts = extractor.extract_from_file(plan_doc, doc_type="plan")
        claims = artifacts[0]["claims"]
        for claim in claims:
            assert 0.0 < claim.confidence <= 0.90  # PLANNING_DOCUMENT cap

    def test_checkbox_items_higher_confidence(
        self, extractor: MarkdownExtractor, plan_doc: Path,
    ):
        artifacts = extractor.extract_from_file(plan_doc, doc_type="plan")
        claims = artifacts[0]["claims"]
        checked = [c for c in claims if c.canonical.predicate == "completed_per_doc"]
        regular = [c for c in claims if c.canonical.predicate == "planned"]
        if checked and regular:
            assert checked[0].confidence >= regular[0].confidence


class TestNumberedLists:
    """Numbered list items are extracted."""

    def test_extracts_numbered_items(self, extractor: MarkdownExtractor, plan_doc: Path):
        artifacts = extractor.extract_from_file(plan_doc, doc_type="plan")
        claims = artifacts[0]["claims"]
        statements = [c.statement for c in claims]
        assert any("unit tests" in s.lower() for s in statements)
        assert any("integration tests" in s.lower() for s in statements)


class TestSourcePriority:
    """PLANNING_DOCUMENT priority is properly defined."""

    def test_planning_document_priority_exists(self):
        assert hasattr(SourcePriority, "PLANNING_DOCUMENT")
        assert SourcePriority.PLANNING_DOCUMENT == 18

    def test_planning_document_between_commit_and_blame(self):
        assert (
            SourcePriority.GIT_COMMIT_MESSAGE
            < SourcePriority.PLANNING_DOCUMENT
            < SourcePriority.GIT_BLAME
        )
