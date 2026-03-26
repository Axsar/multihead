"""Tests for iteration_tracker.py — Phase 1: recording experiment attempts."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from multihead.iteration_tracker import IterationTracker, IterationResult, _git_head_sha


class TestIterationResult:
    def test_defaults(self):
        r = IterationResult(iteration=1, status="success")
        assert r.iteration == 1
        assert r.status == "success"
        assert r.metrics == {}
        assert r.params == {}
        assert r.git_sha == ""
        assert r.claim_id == ""

    def test_with_values(self):
        r = IterationResult(
            iteration=3,
            status="improved",
            metrics={"accuracy": 0.95},
            params={"lr": 0.001},
            git_sha="abc123",
            duration_secs=12.5,
        )
        assert r.metrics["accuracy"] == 0.95
        assert r.duration_secs == 12.5


class TestIterationTracker:
    def test_init_no_ks(self):
        tracker = IterationTracker(None, "test-exp")
        assert tracker.experiment_id == "test-exp"
        assert tracker.iteration == 0

    def test_record_attempt_no_ks(self):
        tracker = IterationTracker(None, "test-exp")
        result = tracker.record_attempt(
            status="success",
            metrics={"score": 0.9},
            description="test run",
        )
        assert result.iteration == 1
        assert result.status == "success"
        assert result.claim_id == ""  # no knowledge store

    def test_record_increments_iteration(self):
        tracker = IterationTracker(None, "test-exp")
        r1 = tracker.record_attempt(status="success")
        r2 = tracker.record_attempt(status="failed")
        r3 = tracker.record_attempt(status="improved")
        assert r1.iteration == 1
        assert r2.iteration == 2
        assert r3.iteration == 3

    def test_record_with_mock_ks(self):
        ks = MagicMock()
        ks.search_claims.return_value = []
        tracker = IterationTracker(ks, "test-exp", agent_id="test-agent")

        result = tracker.record_attempt(
            status="improved",
            metrics={"overlap_count": 0},
            params={"mode": "bbox"},
            description="Zero overlaps achieved",
            git_sha="c893a88",
        )

        assert result.iteration == 1
        assert result.claim_id != ""  # claim was created
        ks.insert_claim.assert_called_once()

        # Check the claim was well-formed
        claim = ks.insert_claim.call_args[0][0]
        assert "improved" in claim.statement
        assert claim.confidence == 0.95  # success/improved = high confidence

    def test_record_failure_lower_confidence(self):
        ks = MagicMock()
        ks.search_claims.return_value = []
        tracker = IterationTracker(ks, "test-exp")

        result = tracker.record_attempt(
            status="failed",
            error="Overlap regression",
        )

        claim = ks.insert_claim.call_args[0][0]
        assert claim.confidence == 0.7  # failure = lower confidence
        assert "Reverted" in claim.statement

    def test_get_history_no_ks(self):
        tracker = IterationTracker(None, "test-exp")
        assert tracker.get_history() == []

    def test_get_failures_filters(self):
        tracker = IterationTracker(None, "test-exp")
        # Can't test without ks, but verify no crash
        assert tracker.get_failures() == []

    def test_get_best_no_history(self):
        tracker = IterationTracker(None, "test-exp")
        assert tracker.get_best() is None


class TestGitHelpers:
    @patch("multihead.iteration_tracker.subprocess.run")
    def test_git_head_sha(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="abc123def456\n",
        )
        sha = _git_head_sha("/tmp/repo")
        assert sha == "abc123def456"

    @patch("multihead.iteration_tracker.subprocess.run")
    def test_git_head_sha_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        sha = _git_head_sha("/tmp/repo")
        assert sha == ""
