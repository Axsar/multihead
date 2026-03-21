"""Tests for GitHub Issues integration (gh CLI wrapper)."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from multihead.github_integration import (
    IssueRef,
    parse_issue_ref,
    infer_repo,
    fetch_issue,
    create_issue,
    comment_on_issue,
    update_issue_labels,
    close_issue,
    create_subtask_issues,
    format_solve_results,
    _run_gh,
)


# ---------------------------------------------------------------------------
# parse_issue_ref
# ---------------------------------------------------------------------------


class TestParseIssueRef:
    def test_plain_number(self):
        ref = parse_issue_ref("42")
        assert ref.number == 42
        assert ref.owner is None
        assert ref.repo is None

    def test_hash_number(self):
        ref = parse_issue_ref("#42")
        assert ref.number == 42
        assert ref.owner is None

    def test_owner_repo_hash(self):
        ref = parse_issue_ref("Axsar/multihead#99")
        assert ref.owner == "Axsar"
        assert ref.repo == "multihead"
        assert ref.number == 99

    def test_full_url(self):
        ref = parse_issue_ref("https://github.com/Axsar/multihead/issues/7")
        assert ref.owner == "Axsar"
        assert ref.repo == "multihead"
        assert ref.number == 7

    def test_http_url(self):
        ref = parse_issue_ref("http://github.com/org/repo/issues/123")
        assert ref.owner == "org"
        assert ref.repo == "repo"
        assert ref.number == 123

    def test_whitespace_stripped(self):
        ref = parse_issue_ref("  42  ")
        assert ref.number == 42

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_issue_ref("not-a-ref")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parse_issue_ref("")


class TestIssueRefRepoFlag:
    def test_with_owner_repo(self):
        ref = IssueRef(owner="Axsar", repo="multihead", number=1)
        assert ref.repo_flag == ["--repo", "Axsar/multihead"]

    def test_without_owner_repo(self):
        ref = IssueRef(owner=None, repo=None, number=1)
        assert ref.repo_flag == []


# ---------------------------------------------------------------------------
# infer_repo
# ---------------------------------------------------------------------------


class TestInferRepo:
    @patch("multihead.github_integration.subprocess.run")
    def test_ssh_remote(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="git@github.com:Axsar/multihead.git\n",
        )
        assert infer_repo() == "Axsar/multihead"

    @patch("multihead.github_integration.subprocess.run")
    def test_https_remote(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="https://github.com/Axsar/multihead.git\n",
        )
        assert infer_repo() == "Axsar/multihead"

    @patch("multihead.github_integration.subprocess.run")
    def test_https_no_dotgit(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="https://github.com/Axsar/multihead\n",
        )
        assert infer_repo() == "Axsar/multihead"

    @patch("multihead.github_integration.subprocess.run")
    def test_non_github_returns_none(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="https://gitlab.com/user/repo.git\n",
        )
        assert infer_repo() is None

    @patch("multihead.github_integration.subprocess.run")
    def test_git_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        assert infer_repo() is None

    @patch("multihead.github_integration.subprocess.run")
    def test_git_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=10)
        assert infer_repo() is None

    @patch("multihead.github_integration.subprocess.run")
    def test_git_nonzero_exit(self, mock_run):
        mock_run.return_value = MagicMock(returncode=128, stdout="")
        assert infer_repo() is None


# ---------------------------------------------------------------------------
# _run_gh
# ---------------------------------------------------------------------------


class TestRunGh:
    @patch("multihead.github_integration.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="ok\n", stderr=""
        )
        result = _run_gh(["issue", "list"])
        assert result.stdout == "ok\n"
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "gh"
        assert cmd[1:] == ["issue", "list"]

    @patch("multihead.github_integration.subprocess.run")
    def test_gh_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        with pytest.raises(RuntimeError, match="not found"):
            _run_gh(["issue", "list"])

    @patch("multihead.github_integration.subprocess.run")
    def test_gh_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="gh", timeout=30)
        with pytest.raises(RuntimeError, match="timed out"):
            _run_gh(["issue", "list"])

    @patch("multihead.github_integration.subprocess.run")
    def test_nonzero_exit(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="not found"
        )
        with pytest.raises(RuntimeError, match="not found"):
            _run_gh(["issue", "view", "999"])


# ---------------------------------------------------------------------------
# fetch_issue
# ---------------------------------------------------------------------------


class TestFetchIssue:
    @patch("multihead.github_integration._run_gh")
    def test_fetch_by_number(self, mock_gh):
        issue_data = {
            "number": 42,
            "title": "Fix bug",
            "body": "Something is broken",
            "labels": [{"name": "bug"}],
            "assignees": [],
            "state": "OPEN",
            "url": "https://github.com/Axsar/multihead/issues/42",
        }
        mock_gh.return_value = MagicMock(stdout=json.dumps(issue_data))
        result = fetch_issue("42")
        assert result["number"] == 42
        assert result["title"] == "Fix bug"

    @patch("multihead.github_integration._run_gh")
    def test_fetch_with_repo(self, mock_gh):
        mock_gh.return_value = MagicMock(stdout=json.dumps({"number": 7}))
        fetch_issue("Axsar/multihead#7")
        cmd = mock_gh.call_args[0][0]
        assert "--repo" in cmd
        assert "Axsar/multihead" in cmd

    @patch("multihead.github_integration._run_gh")
    @patch("multihead.github_integration.infer_repo", return_value="Axsar/multihead")
    def test_fetch_infers_repo(self, mock_infer, mock_gh):
        mock_gh.return_value = MagicMock(stdout=json.dumps({"number": 1}))
        fetch_issue("1")
        cmd = mock_gh.call_args[0][0]
        assert "--repo" in cmd
        assert "Axsar/multihead" in cmd


# ---------------------------------------------------------------------------
# create_issue
# ---------------------------------------------------------------------------


class TestCreateIssue:
    @patch("multihead.github_integration.infer_repo", return_value=None)
    @patch("multihead.github_integration._run_gh")
    def test_create_basic(self, mock_gh, mock_infer):
        mock_gh.return_value = MagicMock(
            stdout="https://github.com/Axsar/multihead/issues/50\n"
        )
        result = create_issue("Test title", "Test body")
        assert result["number"] == 50
        assert "issues/50" in result["url"]

    @patch("multihead.github_integration.infer_repo", return_value=None)
    @patch("multihead.github_integration._run_gh")
    def test_create_with_labels(self, mock_gh, mock_infer):
        mock_gh.return_value = MagicMock(
            stdout="https://github.com/Axsar/multihead/issues/51\n"
        )
        create_issue("Title", "Body", labels=["bug", "multihead-solve"])
        cmd = mock_gh.call_args[0][0]
        assert "--label" in cmd
        label_indices = [i for i, v in enumerate(cmd) if v == "--label"]
        labels = [cmd[i + 1] for i in label_indices]
        assert "bug" in labels
        assert "multihead-solve" in labels

    @patch("multihead.github_integration.infer_repo", return_value=None)
    @patch("multihead.github_integration._run_gh")
    def test_create_with_explicit_repo(self, mock_gh, mock_infer):
        mock_gh.return_value = MagicMock(
            stdout="https://github.com/org/repo/issues/1\n"
        )
        create_issue("Title", "Body", repo="org/repo")
        cmd = mock_gh.call_args[0][0]
        assert "--repo" in cmd
        assert "org/repo" in cmd


# ---------------------------------------------------------------------------
# comment_on_issue
# ---------------------------------------------------------------------------


class TestCommentOnIssue:
    @patch("multihead.github_integration._run_gh")
    def test_comment(self, mock_gh):
        mock_gh.return_value = MagicMock(stdout="")
        comment_on_issue("42", "Results here")
        cmd = mock_gh.call_args[0][0]
        assert "comment" in cmd
        assert "42" in cmd
        assert "--body" in cmd

    @patch("multihead.github_integration._run_gh")
    def test_comment_with_repo_ref(self, mock_gh):
        mock_gh.return_value = MagicMock(stdout="")
        comment_on_issue("Axsar/multihead#10", "Done!")
        cmd = mock_gh.call_args[0][0]
        assert "10" in cmd
        assert "--repo" in cmd


# ---------------------------------------------------------------------------
# update_issue_labels
# ---------------------------------------------------------------------------


class TestUpdateIssueLabels:
    @patch("multihead.github_integration._run_gh")
    def test_add_labels(self, mock_gh):
        mock_gh.return_value = MagicMock(stdout="")
        update_issue_labels("42", add=["bug", "priority"])
        cmd = mock_gh.call_args[0][0]
        assert "--add-label" in cmd

    @patch("multihead.github_integration._run_gh")
    def test_remove_labels(self, mock_gh):
        mock_gh.return_value = MagicMock(stdout="")
        update_issue_labels("42", remove=["wontfix"])
        cmd = mock_gh.call_args[0][0]
        assert "--remove-label" in cmd

    @patch("multihead.github_integration._run_gh")
    def test_no_labels_no_call(self, mock_gh):
        """No-op when neither add nor remove specified."""
        update_issue_labels("42")
        mock_gh.assert_not_called()


# ---------------------------------------------------------------------------
# close_issue
# ---------------------------------------------------------------------------


class TestCloseIssue:
    @patch("multihead.github_integration._run_gh")
    def test_close_basic(self, mock_gh):
        mock_gh.return_value = MagicMock(stdout="")
        close_issue("42")
        cmd = mock_gh.call_args[0][0]
        assert "close" in cmd
        assert "42" in cmd

    @patch("multihead.github_integration._run_gh")
    def test_close_with_comment(self, mock_gh):
        mock_gh.return_value = MagicMock(stdout="")
        close_issue("42", comment="Resolved by MultiHead")
        cmd = mock_gh.call_args[0][0]
        assert "--comment" in cmd


# ---------------------------------------------------------------------------
# create_subtask_issues
# ---------------------------------------------------------------------------


class TestCreateSubtaskIssues:
    @patch("multihead.github_integration.create_issue")
    def test_creates_one_per_step(self, mock_create):
        mock_create.return_value = {"number": 100, "url": "http://..."}
        steps = [
            {"name": "Explore codebase", "action_type": "explore"},
            {"name": "Implement feature", "action_type": "edit"},
            {"name": "Run tests", "action_type": "verify"},
        ]
        results = create_subtask_issues("42", steps)
        assert len(results) == 3
        assert mock_create.call_count == 3

    @patch("multihead.github_integration.create_issue")
    def test_title_format(self, mock_create):
        mock_create.return_value = {"number": 100, "url": "http://..."}
        steps = [{"name": "Do thing"}]
        create_subtask_issues("42", steps)
        title = mock_create.call_args_list[0].kwargs.get("title", "")
        assert "Step 1" in title
        assert "Do thing" in title

    @patch("multihead.github_integration.create_issue")
    def test_body_links_parent(self, mock_create):
        mock_create.return_value = {"number": 100, "url": "http://..."}
        steps = [{"name": "Do thing"}]
        create_subtask_issues("42", steps)
        kwargs = mock_create.call_args_list[0].kwargs
        body = kwargs.get("body", "")
        assert "#42" in body

    @patch("multihead.github_integration.create_issue")
    def test_labels_applied(self, mock_create):
        mock_create.return_value = {"number": 100, "url": "http://..."}
        steps = [{"name": "Do thing"}]
        create_subtask_issues("42", steps)
        kwargs = mock_create.call_args_list[0].kwargs
        labels = kwargs.get("labels", [])
        assert "multihead-solve" in labels
        assert "subtask" in labels


# ---------------------------------------------------------------------------
# format_solve_results
# ---------------------------------------------------------------------------


class TestFormatSolveResults:
    def test_basic_format(self):
        result = format_solve_results(
            goal="Fix the bug",
            status="completed",
            duration_seconds=45.2,
            steps=[
                {"name": "Explore", "status": "committed", "head_id": "mock-llm"},
                {"name": "Implement", "status": "committed", "head_id": "mock-llm"},
            ],
        )
        assert "## MultiHead Solve Results" in result
        assert "Fix the bug" in result
        assert "completed" in result
        assert "45.2s" in result
        assert "[x] Step 1: Explore" in result
        assert "[x] Step 2: Implement" in result

    def test_failed_steps(self):
        result = format_solve_results(
            goal="Test",
            status="partial",
            duration_seconds=10.0,
            steps=[
                {"name": "Good step", "status": "committed"},
                {"name": "Bad step", "status": "failed"},
            ],
        )
        assert "[x] Step 1" in result
        assert "[ ] Step 2" in result
        assert "failed" in result

    def test_with_claims(self):
        result = format_solve_results(
            goal="Test",
            status="completed",
            duration_seconds=5.0,
            steps=[],
            claims=[
                {"claim_key": "test.key", "statement": "Something learned"},
            ],
        )
        assert "### Knowledge Claims" in result
        assert "test.key" in result
        assert "Something learned" in result

    def test_claims_capped_at_10(self):
        claims = [
            {"claim_key": f"k{i}", "statement": f"Claim {i}"}
            for i in range(15)
        ]
        result = format_solve_results(
            goal="Test", status="ok", duration_seconds=1.0,
            steps=[], claims=claims,
        )
        assert "and 5 more" in result

    def test_with_run_id(self):
        result = format_solve_results(
            goal="Test", status="ok", duration_seconds=1.0,
            steps=[], run_id="run-abc-123",
        )
        assert "run-abc-123" in result

    def test_duration_minutes(self):
        result = format_solve_results(
            goal="Test", status="ok", duration_seconds=125.3,
            steps=[],
        )
        assert "2m" in result

    def test_no_duration(self):
        result = format_solve_results(
            goal="Test", status="ok", duration_seconds=None, steps=[],
        )
        assert "Duration" not in result

    def test_skipped_step(self):
        result = format_solve_results(
            goal="Test", status="ok", duration_seconds=1.0,
            steps=[{"name": "Skipped", "status": "skipped"}],
        )
        assert "[-]" in result


# ---------------------------------------------------------------------------
# CLI wiring integration (solve --gh-* options)
# ---------------------------------------------------------------------------


class TestSolveCLIGitHubOptions:
    """Test that solve command accepts --gh-* params."""

    def test_gh_options_registered(self):
        """Verify the solve command has --gh-* params."""
        from multihead.cli import solve
        param_names = [p.name for p in solve.params]
        assert "gh_issue" in param_names
        assert "gh_track" in param_names
        assert "gh_subtasks" in param_names
        assert "gh_comment" in param_names
