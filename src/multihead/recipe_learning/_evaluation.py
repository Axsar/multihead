"""Recipe evaluation logic (consensus and benchmark comparison)."""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import TYPE_CHECKING, Any

import yaml

from multihead.consensus import (
    ConsensusConfig,
    ConsensusEngine,
    ConsensusResult,
    ConsensusStrategy,
    HeadTask,
)

from ._parsing import try_parse_vote_json

if TYPE_CHECKING:
    from multihead.head_manager import HeadManager

logger = logging.getLogger(__name__)


def build_evaluation_prompt(
    proposed_recipe: dict[str, Any],
    benchmark_results: dict[str, Any],
    current_recipe: dict[str, Any] | None = None,
    current_benchmark: dict[str, Any] | None = None,
) -> str:
    """Build evaluation prompt for consensus voting."""
    prompt_parts = [
        "Evaluate the following recipe proposal.\n",
        f"\nProposed Recipe:\n{yaml.dump(proposed_recipe, default_flow_style=False)}",
        f"\nBenchmark Results: {json.dumps(benchmark_results, default=str)}",
    ]

    if current_recipe and current_benchmark:
        prompt_parts.extend([
            f"\nCurrent Recipe:\n{yaml.dump(current_recipe, default_flow_style=False)}",
            f"\nCurrent Benchmark: {json.dumps(current_benchmark, default=str)}",
        ])

    prompt_parts.extend([
        "\n\nRespond with JSON:",
        '{"action": "adopt"|"modify"|"reject",',
        ' "rationale": "your reasoning",',
        ' "confidence": 0.0-1.0}',
    ])

    return "\n".join(prompt_parts)


async def evaluate_with_consensus(
    prompt: str,
    consensus_engine: ConsensusEngine,
    head_manager: HeadManager,
) -> dict[str, Any] | None:
    """Evaluate recipe using multi-head consensus.

    Args:
        prompt: Evaluation prompt text
        consensus_engine: ConsensusEngine instance
        head_manager: HeadManager instance

    Returns:
        Decision dict or None if consensus failed.
    """
    # Get available heads for voting
    available = list(head_manager.get_states().keys())
    if len(available) < 1:
        return None

    # Build head tasks (up to 3 heads)
    head_tasks = [
        HeadTask(head_id=hid, weight=1.0, required=(i == 0))
        for i, hid in enumerate(available[:3])
    ]

    config = ConsensusConfig(
        heads=head_tasks,
        strategy=ConsensusStrategy.WEIGHTED,
        timeout_seconds=60.0,
    )

    result: ConsensusResult = await consensus_engine.execute(config, prompt)

    # Parse votes from each head
    votes: dict[str, str] = {}
    actions: list[str] = []
    confidences: list[float] = []
    rationales: list[str] = []

    for vote in result.all_votes:
        if not vote.success:
            continue

        text = vote.outputs.get("text", "")
        parsed = try_parse_vote_json(text)
        votes[vote.head_id] = parsed.get("action", "reject")
        actions.append(parsed.get("action", "reject"))
        confidences.append(parsed.get("confidence", 0.5))
        rationales.append(parsed.get("rationale", ""))

    if not actions:
        return None

    # Majority vote on action
    action_counts = Counter(actions)
    majority_action = action_counts.most_common(1)[0][0]
    avg_confidence = sum(confidences) / len(confidences)

    # Combine rationales
    combined_rationale = "; ".join(r for r in rationales if r)

    return {
        "action": majority_action,
        "rationale": combined_rationale or f"Consensus: {majority_action}",
        "confidence": round(avg_confidence, 3),
        "votes": votes,
        "agreement_score": result.agreement_score,
    }


def evaluate_by_benchmarks(
    benchmark_results: dict[str, Any],
    current_benchmark: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Simple benchmark comparison fallback."""
    decision: dict[str, Any] = {
        "action": "pending",
        "rationale": "Evaluation not yet implemented",
        "confidence": 0.5,
    }

    if not current_benchmark:
        if benchmark_results.get("success_count", 0) > 0:
            decision["action"] = "adopt"
            decision["rationale"] = "No current recipe, proposed shows promise"
            decision["confidence"] = 0.7
    else:
        proposed_success = benchmark_results.get("success_count", 0) / max(
            benchmark_results.get("test_cases_count", 1), 1
        )
        current_success = current_benchmark.get("success_count", 0) / max(
            current_benchmark.get("test_cases_count", 1), 1
        )

        if proposed_success > current_success:
            decision["action"] = "adopt"
            decision["rationale"] = (
                f"Proposed recipe outperforms current "
                f"({proposed_success:.1%} vs {current_success:.1%})"
            )
            decision["confidence"] = 0.8
        else:
            decision["action"] = "reject"
            decision["rationale"] = (
                f"Current recipe performs better "
                f"({current_success:.1%} vs {proposed_success:.1%})"
            )
            decision["confidence"] = 0.7

    logger.info(
        "Recipe evaluation: %s - %s", decision["action"], decision["rationale"]
    )
    return decision
