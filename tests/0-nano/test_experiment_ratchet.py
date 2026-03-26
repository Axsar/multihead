"""Tests for experiment_ratchet.py — Phase 2: AutoResearch-style loop."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from multihead.experiment_ratchet import (
    ExperimentRatchet,
    RatchetConfig,
    RatchetReport,
)


class TestRatchetConfig:
    def test_defaults(self):
        c = RatchetConfig(experiment_id="test")
        assert c.experiment_id == "test"
        assert c.max_iterations == 50
        assert c.time_budget_secs == 300
        assert c.metric_goal == "maximize"
        assert c.simplicity_bias is True

    def test_custom(self):
        c = RatchetConfig(
            experiment_id="balloon",
            target_files=["src/layout.py"],
            test_command="pytest -x",
            metric_name="overlap_count",
            metric_goal="minimize",
            max_iterations=10,
        )
        assert c.metric_goal == "minimize"
        assert c.target_files == ["src/layout.py"]


class TestRatchetReport:
    def test_defaults(self):
        r = RatchetReport(experiment_id="test")
        assert r.iterations_run == 0
        assert r.iterations_kept == 0
        assert r.stopped_reason == ""

    def test_summary(self):
        r = RatchetReport(
            experiment_id="test",
            iterations_run=10,
            iterations_kept=3,
            iterations_reverted=6,
            iterations_errored=1,
            best_iteration=7,
        )
        assert r.iterations_kept + r.iterations_reverted + r.iterations_errored == 10


class TestExperimentRatchet:
    def test_init(self):
        ks = MagicMock()
        ks.search_claims.return_value = []
        config = RatchetConfig(experiment_id="test", max_iterations=5)
        ratchet = ExperimentRatchet(knowledge_store=ks, config=config)
        assert ratchet.config.experiment_id == "test"

    def test_is_improvement_maximize(self):
        ks = MagicMock()
        ks.search_claims.return_value = []
        config = RatchetConfig(experiment_id="test", metric_goal="maximize")
        ratchet = ExperimentRatchet(knowledge_store=ks, config=config)

        assert ratchet._is_improvement(0.9, 0.8) is True
        assert ratchet._is_improvement(0.7, 0.8) is False
        assert ratchet._is_improvement(0.5, None) is True  # first result
        assert ratchet._is_improvement(None, 0.5) is False

    def test_is_improvement_minimize(self):
        ks = MagicMock()
        ks.search_claims.return_value = []
        config = RatchetConfig(experiment_id="test", metric_goal="minimize")
        ratchet = ExperimentRatchet(knowledge_store=ks, config=config)

        assert ratchet._is_improvement(0.3, 0.5) is True
        assert ratchet._is_improvement(0.7, 0.5) is False

    def test_meets_threshold_maximize(self):
        ks = MagicMock()
        ks.search_claims.return_value = []
        config = RatchetConfig(
            experiment_id="test",
            metric_goal="maximize",
            quality_threshold=0.9,
        )
        ratchet = ExperimentRatchet(knowledge_store=ks, config=config)

        assert ratchet._meets_threshold(0.95) is True
        assert ratchet._meets_threshold(0.85) is False

    def test_meets_threshold_minimize(self):
        ks = MagicMock()
        ks.search_claims.return_value = []
        config = RatchetConfig(
            experiment_id="test",
            metric_goal="minimize",
            quality_threshold=5.0,
        )
        ratchet = ExperimentRatchet(knowledge_store=ks, config=config)

        assert ratchet._meets_threshold(3.0) is True
        assert ratchet._meets_threshold(7.0) is False

    def test_stop_request(self):
        ks = MagicMock()
        ks.search_claims.return_value = []
        config = RatchetConfig(experiment_id="test", max_iterations=100)
        ratchet = ExperimentRatchet(knowledge_store=ks, config=config)
        ratchet.stop()
        assert ratchet._stop_requested is True

    @pytest.mark.asyncio
    async def test_run_with_custom_fns(self):
        """Test the full loop with custom propose/evaluate functions."""
        ks = MagicMock()
        ks.search_claims.return_value = []
        ks.insert_claim = MagicMock()

        iteration_scores = [0.5, 0.7, 0.6, 0.8, 0.75]  # 3 improvements, 2 reverts

        async def mock_propose(iteration, report, tracker):
            return {
                "description": f"Change {iteration}",
                "params": {"iter": iteration},
                "changes": [],  # no actual file changes
            }

        async def mock_evaluate(iteration):
            idx = min(iteration - 1, len(iteration_scores) - 1)
            return {"quality_score": iteration_scores[idx]}

        config = RatchetConfig(
            experiment_id="test-loop",
            max_iterations=5,
            metric_name="quality_score",
            metric_goal="maximize",
        )

        with patch("multihead.experiment_ratchet._git_head_sha", return_value="abc123"), \
             patch("multihead.experiment_ratchet._git_commit", return_value="def456"), \
             patch("multihead.experiment_ratchet._git_reset_hard", return_value=True):

            ratchet = ExperimentRatchet(
                knowledge_store=ks,
                config=config,
                propose_fn=mock_propose,
                evaluate_fn=mock_evaluate,
            )
            report = await ratchet.run()

        assert report.iterations_run == 5
        assert report.iterations_kept == 3  # 0.5 (first), 0.7, 0.8
        assert report.iterations_reverted == 2  # 0.6, 0.75
        assert report.best_iteration == 4  # score 0.8
        assert report.stopped_reason == "max_iterations"

    @pytest.mark.asyncio
    async def test_run_stops_at_threshold(self):
        ks = MagicMock()
        ks.search_claims.return_value = []
        ks.insert_claim = MagicMock()

        async def mock_propose(iteration, report, tracker):
            return {"description": f"Change {iteration}", "params": {}, "changes": []}

        async def mock_evaluate(iteration):
            return {"score": 0.95}  # immediately exceeds threshold

        config = RatchetConfig(
            experiment_id="test-threshold",
            max_iterations=100,
            metric_name="score",
            metric_goal="maximize",
            quality_threshold=0.9,
        )

        with patch("multihead.experiment_ratchet._git_head_sha", return_value="abc"), \
             patch("multihead.experiment_ratchet._git_commit", return_value="def"), \
             patch("multihead.experiment_ratchet._git_reset_hard", return_value=True):

            ratchet = ExperimentRatchet(
                knowledge_store=ks,
                config=config,
                propose_fn=mock_propose,
                evaluate_fn=mock_evaluate,
            )
            report = await ratchet.run()

        assert report.iterations_run == 1
        assert report.stopped_reason == "threshold_reached"

    @pytest.mark.asyncio
    async def test_run_handles_proposal_error(self):
        ks = MagicMock()
        ks.search_claims.return_value = []
        ks.insert_claim = MagicMock()

        async def mock_propose(iteration, report, tracker):
            raise RuntimeError("Model unavailable")

        config = RatchetConfig(
            experiment_id="test-error",
            max_iterations=2,
        )

        with patch("multihead.experiment_ratchet._git_head_sha", return_value="abc"):
            ratchet = ExperimentRatchet(
                knowledge_store=ks,
                config=config,
                propose_fn=mock_propose,
            )
            report = await ratchet.run()

        assert report.iterations_errored == 2
        assert report.iterations_kept == 0
