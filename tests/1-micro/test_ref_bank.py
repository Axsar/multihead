"""Tests for RefBankBuilder — auto-generated reference files from knowledge.db."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from multihead.ref_bank import RefBankBuilder, REF_FILES, _ClaimProxy, _recency_score


# ---------------------------------------------------------------------------
# _recency_score
# ---------------------------------------------------------------------------


class TestRecencyScore:
    def test_recent_claim_high_score(self):
        now = datetime.now(timezone.utc)
        score = _recency_score(now)
        assert score > 0.9

    def test_old_claim_low_score(self):
        from datetime import timedelta
        old = datetime.now(timezone.utc) - timedelta(days=30)
        score = _recency_score(old)
        assert score < 0.15

    def test_none_returns_default(self):
        assert _recency_score(None) == 0.3

    def test_naive_datetime_handled(self):
        """Naive datetime (no tzinfo) should not crash."""
        naive = datetime(2026, 1, 1)
        score = _recency_score(naive)
        assert 0 <= score <= 1


# ---------------------------------------------------------------------------
# _ClaimProxy
# ---------------------------------------------------------------------------


class TestClaimProxy:
    def test_defaults(self):
        c = _ClaimProxy()
        assert c.key == ""
        assert c.statement == ""
        assert c.confidence == 0.5
        assert c.claim_type == "fact"
        assert c.updated_at is None

    def test_with_values(self):
        now = datetime.now(timezone.utc)
        c = _ClaimProxy(
            key="test.key",
            statement="A test claim",
            confidence=0.95,
            claim_type="decision",
            updated_at=now,
        )
        assert c.key == "test.key"
        assert c.confidence == 0.95


# ---------------------------------------------------------------------------
# REF_FILES config
# ---------------------------------------------------------------------------


class TestRefFilesConfig:
    def test_six_ref_files_defined(self):
        assert len(REF_FILES) == 6

    def test_all_have_required_keys(self):
        required = {"filename", "title", "description", "max_items", "max_lines"}
        for rf in REF_FILES:
            assert required.issubset(rf.keys()), f"{rf['filename']} missing keys"

    def test_filenames_are_md(self):
        for rf in REF_FILES:
            assert rf["filename"].endswith(".md")

    def test_total_line_budget_reasonable(self):
        total = sum(rf["max_lines"] for rf in REF_FILES)
        assert total <= 800, f"Total line budget {total} exceeds 800"


# ---------------------------------------------------------------------------
# RefBankBuilder
# ---------------------------------------------------------------------------


class TestRefBankBuilder:
    @pytest.fixture
    def mock_ks(self):
        ks = MagicMock()
        # FTS5 returns tuples
        ks.search_claims_fts.return_value = [
            ("arch.router", "The router uses weighted scoring for head selection", 0.9),
            ("arch.dag", "DAG executor enables parallel step execution", 0.85),
        ]
        # list_claims returns claim objects
        claim = MagicMock()
        claim.statement = "Decomposition must use worktrees for safety"
        claim.confidence = 0.95
        claim.canonical = MagicMock()
        claim.canonical.claim_key = "constraint.worktree"
        claim.updated_at = datetime.now(timezone.utc)
        ks.list_claims.return_value = [claim]
        return ks

    @pytest.fixture
    def builder(self, mock_ks, tmp_path):
        return RefBankBuilder(mock_ks, tmp_path)

    def test_refresh_all_returns_results(self, builder):
        results = builder.refresh_all()
        assert len(results) == 6
        for r in results:
            assert "filename" in r
            assert "items_count" in r
            assert "lines" in r

    def test_ref_files_created(self, builder, tmp_path):
        builder.refresh_all()
        for rf in REF_FILES:
            path = tmp_path / rf["filename"]
            assert path.exists(), f"{rf['filename']} not created"

    def test_ref_file_has_header(self, builder, tmp_path):
        builder.refresh_all()
        content = (tmp_path / "ref-decisions.md").read_text()
        assert "# Recent Decisions" in content
        assert "Auto-generated" in content

    def test_ref_file_has_claims(self, builder, tmp_path):
        builder.refresh_all()
        content = (tmp_path / "ref-architecture.md").read_text()
        assert "router" in content.lower() or "scoring" in content.lower()

    def test_empty_knowledge_store(self, tmp_path):
        ks = MagicMock()
        ks.search_claims_fts.return_value = []
        ks.list_claims.return_value = []
        builder = RefBankBuilder(ks, tmp_path)
        results = builder.refresh_all()
        # Should still create files (with "no matching claims" message)
        for rf in REF_FILES:
            assert (tmp_path / rf["filename"]).exists()

    def test_fts_exception_handled(self, tmp_path):
        ks = MagicMock()
        ks.search_claims_fts.side_effect = Exception("FTS5 unavailable")
        ks.list_claims.return_value = []
        builder = RefBankBuilder(ks, tmp_path)
        # Should not raise
        results = builder.refresh_all()
        assert len(results) == 6

    def test_deduplication(self, tmp_path):
        ks = MagicMock()
        # Same statement from FTS and list_claims
        ks.search_claims_fts.return_value = [
            ("key1", "Duplicate claim about router", 0.9),
            ("key2", "Duplicate claim about router", 0.85),
        ]
        claim = MagicMock()
        claim.statement = "Duplicate claim about router"
        claim.confidence = 0.9
        claim.canonical = MagicMock()
        claim.canonical.claim_key = "key3"
        claim.updated_at = datetime.now(timezone.utc)
        ks.list_claims.return_value = [claim]
        builder = RefBankBuilder(ks, tmp_path)
        results = builder.refresh_all()
        # The duplicate should be deduplicated
        for r in results:
            if r["filename"] == "ref-architecture.md":
                assert r["items_count"] <= 2  # At most 2 unique

    def test_line_budget_respected(self, tmp_path):
        ks = MagicMock()
        # Return many claims to test truncation
        ks.search_claims_fts.return_value = [
            (f"key{i}", f"Claim number {i} about various topics", 0.8)
            for i in range(200)
        ]
        ks.list_claims.return_value = []
        builder = RefBankBuilder(ks, tmp_path)
        results = builder.refresh_all()
        for r in results:
            rf_def = next(rf for rf in REF_FILES if rf["filename"] == r["filename"])
            # Lines should not exceed max + 1 (truncation notice)
            assert r["lines"] <= rf_def["max_lines"] + 1

    def test_memory_dir_created(self, tmp_path):
        ks = MagicMock()
        ks.search_claims_fts.return_value = []
        ks.list_claims.return_value = []
        new_dir = tmp_path / "new" / "nested" / "dir"
        builder = RefBankBuilder(ks, new_dir)
        builder.refresh_all()
        assert new_dir.exists()

    def test_confidence_shown_in_output(self, builder, tmp_path):
        builder.refresh_all()
        content = (tmp_path / "ref-architecture.md").read_text()
        # Should contain percentage confidence like [90%]
        assert "%" in content


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestScoring:
    def test_high_confidence_high_score(self):
        builder = RefBankBuilder(MagicMock(), Path("/tmp"))
        high = _ClaimProxy(confidence=0.95, updated_at=datetime.now(timezone.utc))
        low = _ClaimProxy(confidence=0.3, updated_at=datetime.now(timezone.utc))
        assert builder._score_claim(high) > builder._score_claim(low)

    def test_recent_beats_old(self):
        from datetime import timedelta
        builder = RefBankBuilder(MagicMock(), Path("/tmp"))
        recent = _ClaimProxy(confidence=0.8, updated_at=datetime.now(timezone.utc))
        old = _ClaimProxy(
            confidence=0.8,
            updated_at=datetime.now(timezone.utc) - timedelta(days=30),
        )
        assert builder._score_claim(recent) > builder._score_claim(old)
