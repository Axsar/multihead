"""Tests for the session harvester — cross-context knowledge aggregator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from multihead.session_harvester import (
    HarvestResult,
    ProjectInfo,
    SessionHarvester,
    classify_claim_type,
    deposit_claims,
    extract_claims,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_knowledge_store():
    ks = MagicMock()
    claim = MagicMock()
    claim.claim_id = "test-claim-id"
    ks.insert_claim.return_value = claim
    return ks


@pytest.fixture
def tmp_claude_home(tmp_path):
    """Create a fake ~/.claude structure with test projects."""
    claude_home = tmp_path / ".claude"
    projects_dir = claude_home / "projects"

    # Project 1: has MEMORY.md
    proj1 = projects_dir / "-mnt-d-DevD-Multihead"
    memory_dir1 = proj1 / "memory"
    memory_dir1.mkdir(parents=True)
    (memory_dir1 / "MEMORY.md").write_text(
        "# MultiHead Project\n\n"
        "## Architecture\n"
        "- Event-sourced durable execution (kill-9 resilient)\n"
        "- Content-addressed artifact store (SHA-256 sharded)\n"
        "- [x] Adapters: Ollama, vLLM, HuggingFace Transformers\n"
        "- [ ] Cloud marketplace integration pending\n"
        "\n"
        "## User Preferences\n"
        "- Always use bun instead of npm\n"
        "- **Repo**: https://github.com/test/multihead.git\n"
    )

    # Project 2: has CLAUDE.md
    proj2 = projects_dir / "-mnt-d-DevD-Vibebots"
    proj2.mkdir(parents=True)
    (proj2 / "CLAUDE.md").write_text(
        "# Vibebots Instructions\n\n"
        "- Use FastAPI for all endpoints\n"
        "- Never commit .env files\n"
        "- **Database**: PostgreSQL 15 with SQLAlchemy\n"
    )

    # Project 3: has both MEMORY.md and CLAUDE.md
    proj3 = projects_dir / "-home-user-projects-myapp"
    memory_dir3 = proj3 / "memory"
    memory_dir3.mkdir(parents=True)
    (memory_dir3 / "MEMORY.md").write_text(
        "# Project Memory\n\n"
        "## Detection\n"
        "- YOLO model achieves 91.85% mAP on balloon detection\n"
    )
    (proj3 / "CLAUDE.md").write_text(
        "# Project Config\n"
        "- tails are always 3 points\n"
    )

    # Project 4: empty directory (no memory files)
    proj4 = projects_dir / "-mnt-d-Empty"
    proj4.mkdir(parents=True)

    return claude_home


@pytest.fixture
def tmp_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def harvester(mock_knowledge_store, tmp_claude_home, tmp_data_dir):
    return SessionHarvester(
        knowledge_store=mock_knowledge_store,
        claude_home=str(tmp_claude_home),
        data_dir=str(tmp_data_dir),
        max_claims_per_project=100,
    )


# ---------------------------------------------------------------------------
# Discovery tests
# ---------------------------------------------------------------------------


class TestScanProjects:
    def test_finds_projects_with_memory_files(self, harvester):
        projects = harvester.scan_projects()
        # Should find 3 projects (not the empty one)
        assert len(projects) == 3

    def test_project_info_fields(self, harvester):
        projects = harvester.scan_projects()
        names = {p.name for p in projects}
        assert "-mnt-d-DevD-Multihead" in names
        assert "-mnt-d-DevD-Vibebots" in names
        assert "-home-user-projects-myapp" in names

    def test_has_memory_flag(self, harvester):
        projects = harvester.scan_projects()
        by_name = {p.name: p for p in projects}
        assert by_name["-mnt-d-DevD-Multihead"].has_memory is True
        assert by_name["-mnt-d-DevD-Vibebots"].has_memory is False
        assert by_name["-mnt-d-DevD-Vibebots"].has_claude_md is True

    def test_has_claude_md_flag(self, harvester):
        projects = harvester.scan_projects()
        by_name = {p.name: p for p in projects}
        assert by_name["-home-user-projects-myapp"].has_memory is True
        assert by_name["-home-user-projects-myapp"].has_claude_md is True

    def test_empty_project_excluded(self, harvester):
        projects = harvester.scan_projects()
        names = {p.name for p in projects}
        assert "-mnt-d-Empty" not in names

    def test_nonexistent_claude_home(self, mock_knowledge_store, tmp_path):
        h = SessionHarvester(
            knowledge_store=mock_knowledge_store,
            claude_home=str(tmp_path / "nonexistent"),
            data_dir=str(tmp_path),
        )
        projects = h.scan_projects()
        assert projects == []


class TestDecodeFolderName:
    def test_standard_path(self):
        result = SessionHarvester._decode_folder_name("-mnt-d-DevD-Multihead")
        assert result == "/mnt/d/DevD/Multihead"

    def test_deep_path(self):
        result = SessionHarvester._decode_folder_name("-home-user-projects-myapp")
        assert result == "/home/user/projects/myapp"

    def test_no_leading_dash(self):
        assert SessionHarvester._decode_folder_name("some-project") == "some-project"


class TestDeriveScopeId:
    def test_multihead(self):
        p = ProjectInfo(name="-mnt-d-DevD-Multihead", path=Path("/tmp"))
        assert SessionHarvester._derive_scope_id(p) == "multihead"

    def test_vibebots(self):
        p = ProjectInfo(name="-mnt-d-DevD-Vibebots", path=Path("/tmp"))
        assert SessionHarvester._derive_scope_id(p) == "vibebots"

    def test_h2v(self):
        p = ProjectInfo(name="-home-user-projects-myapp", path=Path("/tmp"))
        assert SessionHarvester._derive_scope_id(p) == "h2v"

    def test_htovprocess(self):
        p = ProjectInfo(name="-mnt-d-HTOVProcess", path=Path("/tmp"))
        assert SessionHarvester._derive_scope_id(p) == "htovprocess"


# ---------------------------------------------------------------------------
# Claim extraction tests
# ---------------------------------------------------------------------------


class TestExtractClaims:
    def test_extracts_bullets(self):
        content = (
            "# Test Project\n\n"
            "## Architecture\n"
            "- Event-sourced durable execution is the core pattern\n"
            "- Content-addressed artifact store for persistence\n"
        )
        project = ProjectInfo(name="test", path=Path("/tmp"))
        claims = extract_claims(content, "test", project, Path("/tmp/MEMORY.md"), 0.75)
        assert len(claims) == 2
        assert claims[0]["claim_type"] == "fact"
        assert claims[0]["scope_id"] == "test"
        assert "Architecture" in claims[0]["topic"]

    def test_extracts_checkboxes(self):
        content = (
            "## Status\n"
            "- [x] Completed task\n"
            "- [ ] Planned task not done yet\n"
        )
        project = ProjectInfo(name="test", path=Path("/tmp"))
        claims = extract_claims(content, "test", project, Path("/tmp/MEMORY.md"), 0.75)
        assert len(claims) == 2
        preds = {c["predicate"] for c in claims}
        assert "completed" in preds
        assert "planned" in preds

    def test_extracts_key_value(self):
        content = (
            "## Config\n"
            "**Repo**: https://github.com/test/repo.git\n"
            "**Database**: PostgreSQL 15\n"
        )
        project = ProjectInfo(name="test", path=Path("/tmp"))
        claims = extract_claims(content, "test", project, Path("/tmp/MEMORY.md"), 0.75)
        assert len(claims) == 2
        assert claims[0]["predicate"] == "defines"

    def test_skips_short_bullets(self):
        content = "## List\n- short\n- also tiny\n- This is long enough to be a claim\n"
        project = ProjectInfo(name="test", path=Path("/tmp"))
        claims = extract_claims(content, "test", project, Path("/tmp/MEMORY.md"), 0.75)
        # Only the last one should pass (>= 10 chars)
        assert len(claims) == 1

    def test_classifies_constraints(self):
        content = "## Rules\n- Never commit .env files to the repository\n"
        project = ProjectInfo(name="test", path=Path("/tmp"))
        claims = extract_claims(content, "test", project, Path("/tmp/MEMORY.md"), 0.75)
        assert claims[0]["claim_type"] == "constraint"

    def test_classifies_preferences(self):
        content = "## Prefs\n- Prefer using bun over npm for speed\n"
        project = ProjectInfo(name="test", path=Path("/tmp"))
        claims = extract_claims(content, "test", project, Path("/tmp/MEMORY.md"), 0.75)
        assert claims[0]["claim_type"] == "preference"

    def test_classifies_decisions(self):
        content = "## Log\n- Decision: use PostgreSQL for the main database\n"
        project = ProjectInfo(name="test", path=Path("/tmp"))
        claims = extract_claims(content, "test", project, Path("/tmp/MEMORY.md"), 0.75)
        assert claims[0]["claim_type"] == "decision"

    def test_empty_content(self):
        project = ProjectInfo(name="test", path=Path("/tmp"))
        claims = extract_claims("", "test", project, Path("/tmp/MEMORY.md"), 0.75)
        assert claims == []

    def test_stable_claim_keys(self):
        """Same content should produce same claim keys (content-addressed)."""
        content = "## Topic\n- This is a stable claim about something\n"
        project = ProjectInfo(name="test", path=Path("/tmp"))
        claims1 = extract_claims(content, "test", project, Path("/tmp/MEMORY.md"), 0.75)
        claims2 = extract_claims(content, "test", project, Path("/tmp/MEMORY.md"), 0.75)
        assert claims1[0]["claim_key"] == claims2[0]["claim_key"]


# ---------------------------------------------------------------------------
# Manifest tests
# ---------------------------------------------------------------------------


class TestManifest:
    def test_empty_manifest(self, harvester):
        manifest = harvester.get_manifest()
        assert manifest == {"last_full_scan": None, "projects": {}}

    def test_save_and_load(self, harvester):
        manifest = {
            "last_full_scan": "2026-03-03T00:00:00+00:00",
            "projects": {
                "test": {"scope_id": "test", "files": {}, "claim_count": 5},
            },
        }
        harvester._save_manifest(manifest)
        loaded = harvester.get_manifest()
        assert loaded["last_full_scan"] == "2026-03-03T00:00:00+00:00"
        assert loaded["projects"]["test"]["claim_count"] == 5


# ---------------------------------------------------------------------------
# Full harvest tests
# ---------------------------------------------------------------------------


class TestHarvestAll:
    def test_full_harvest_cycle(self, harvester, mock_knowledge_store):
        result = harvester.harvest_all()
        assert isinstance(result, HarvestResult)
        assert result.projects_scanned == 3
        assert result.projects_harvested > 0
        assert result.claims_deposited > 0
        assert result.duration_seconds >= 0
        # Knowledge store should have been called
        assert mock_knowledge_store.insert_claim.call_count > 0

    def test_second_harvest_skips_unchanged(self, harvester, mock_knowledge_store):
        # First harvest
        result1 = harvester.harvest_all()
        count1 = mock_knowledge_store.insert_claim.call_count

        # Second harvest — same files, should skip
        mock_knowledge_store.insert_claim.reset_mock()
        result2 = harvester.harvest_all()
        assert result2.claims_deposited == 0
        assert result2.projects_skipped == result2.projects_scanned
        assert mock_knowledge_store.insert_claim.call_count == 0

    def test_reharvest_after_file_change(self, harvester, mock_knowledge_store, tmp_claude_home):
        # First harvest
        harvester.harvest_all()
        mock_knowledge_store.insert_claim.reset_mock()

        # Modify a file
        proj = "-mnt-d-DevD-Multihead"
        memory_file = tmp_claude_home / "projects" / proj / "memory" / "MEMORY.md"
        content = memory_file.read_text()
        memory_file.write_text(content + "\n- New claim added after first harvest\n")

        # Second harvest should pick up the change
        result2 = harvester.harvest_all()
        assert result2.claims_deposited > 0

    def test_max_claims_per_project(self, mock_knowledge_store, tmp_claude_home, tmp_data_dir):
        h = SessionHarvester(
            knowledge_store=mock_knowledge_store,
            claude_home=str(tmp_claude_home),
            data_dir=str(tmp_data_dir),
            max_claims_per_project=2,
        )
        result = h.harvest_all()
        # Each project capped at 2 claims
        # Project 1 has ~6 extractable items but should cap at 2
        # Total should be <= 2 * 3 projects = 6
        assert result.claims_deposited <= 6

    def test_harvest_deposits_correct_types(self, harvester, mock_knowledge_store):
        harvester.harvest_all()
        calls = mock_knowledge_store.insert_claim.call_args_list
        assert len(calls) > 0
        # Check that claims have proper structure
        for call in calls:
            claim = call[0][0]
            assert hasattr(claim, "statement")
            assert hasattr(claim, "claim_type")
            assert hasattr(claim, "scope")
            assert hasattr(claim, "provenance")

    def test_harvest_error_handling(self, mock_knowledge_store, tmp_claude_home, tmp_data_dir):
        """Errors in one project don't prevent harvesting others."""
        mock_knowledge_store.insert_claim.side_effect = [
            Exception("DB error"),  # First call fails
            MagicMock(claim_id="ok"),  # Rest succeed
        ] + [MagicMock(claim_id=f"ok-{i}") for i in range(50)]

        h = SessionHarvester(
            knowledge_store=mock_knowledge_store,
            claude_home=str(tmp_claude_home),
            data_dir=str(tmp_data_dir),
        )
        result = h.harvest_all()
        # Should still complete without raising
        assert result.projects_scanned == 3


# ---------------------------------------------------------------------------
# File hash tests
# ---------------------------------------------------------------------------


class TestFileHash:
    def test_consistent_hash(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("hello world")
        h1 = SessionHarvester._file_hash(f)
        h2 = SessionHarvester._file_hash(f)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_content_different_hash(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("hello")
        h1 = SessionHarvester._file_hash(f)
        f.write_text("world")
        h2 = SessionHarvester._file_hash(f)
        assert h1 != h2

    def test_nonexistent_file(self):
        h = SessionHarvester._file_hash(Path("/nonexistent/file.md"))
        assert h == ""


# ---------------------------------------------------------------------------
# Status tests
# ---------------------------------------------------------------------------


class TestStatus:
    def test_status_format(self, harvester):
        status = harvester.status()
        assert "projects_found" in status
        assert "last_full_scan" in status
        assert "total_claims" in status
        assert "projects" in status
        assert isinstance(status["projects"], list)

    def test_status_after_harvest(self, harvester):
        harvester.harvest_all()
        status = harvester.status()
        assert status["projects_found"] == 3
        assert status["total_claims"] > 0


# ---------------------------------------------------------------------------
# Deposit tests
# ---------------------------------------------------------------------------


class TestDepositClaims:
    def test_deposits_to_knowledge_store(self, mock_knowledge_store):
        claims = [
            {
                "statement": "Test claim statement for testing",
                "claim_type": "fact",
                "predicate": "states",
                "scope_id": "test",
                "topic": "Testing",
                "source_file": "/tmp/MEMORY.md",
                "project_name": "test-project",
                "confidence": 0.75,
                "claim_key": "session.test.testing.abc123",
            }
        ]
        count = deposit_claims(claims, mock_knowledge_store)
        assert count == 1
        mock_knowledge_store.insert_claim.assert_called_once()

    def test_deposit_failure_handled(self, mock_knowledge_store):
        mock_knowledge_store.insert_claim.side_effect = Exception("DB error")
        claims = [
            {
                "statement": "Will fail to deposit this claim",
                "claim_type": "fact",
                "predicate": "states",
                "scope_id": "test",
                "topic": "",
                "source_file": "/tmp/test.md",
                "project_name": "test",
                "confidence": 0.75,
                "claim_key": "session.test.general.xyz789",
            }
        ]
        count = deposit_claims(claims, mock_knowledge_store)
        assert count == 0


# ---------------------------------------------------------------------------
# Classify claim type tests
# ---------------------------------------------------------------------------


class TestClassifyClaimType:
    def test_fact(self):
        assert classify_claim_type("Some regular fact") == "fact"

    def test_decision(self):
        assert classify_claim_type("Decision: use PostgreSQL") == "decision"

    def test_constraint(self):
        assert classify_claim_type("Never expose API keys") == "constraint"

    def test_preference(self):
        assert classify_claim_type("Prefer TypeScript over JS") == "preference"

    def test_plan(self):
        assert classify_claim_type("Plan: implement auth next") == "plan"
