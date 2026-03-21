"""Data models for meta-reasoning solver selection."""

from __future__ import annotations

from typing import Any


class SelectionResult:
    """Result of a meta-reasoning solver selection."""

    def __init__(
        self,
        task_type: str,
        selected_solver_id: str,
        reasoning: str,
        confidence_score: float,
        rankings: list[str],
        consensus_votes: dict[str, str],
        benchmark_scores: dict[str, float],
        candidates_evaluated: int,
    ):
        self.task_type = task_type
        self.selected_solver_id = selected_solver_id
        self.reasoning = reasoning
        self.confidence_score = confidence_score
        self.rankings = rankings
        self.consensus_votes = consensus_votes
        self.benchmark_scores = benchmark_scores
        self.candidates_evaluated = candidates_evaluated

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "task_type": self.task_type,
            "selected_solver_id": self.selected_solver_id,
            "reasoning": self.reasoning,
            "confidence_score": self.confidence_score,
            "rankings": self.rankings,
            "consensus_votes": self.consensus_votes,
            "benchmark_scores": self.benchmark_scores,
            "candidates_evaluated": self.candidates_evaluated,
        }
