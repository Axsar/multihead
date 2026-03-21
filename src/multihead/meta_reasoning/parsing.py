"""Prompt formatting and consensus output parsing for meta-reasoning."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from multihead.consensus import ConsensusResult

logger = logging.getLogger(__name__)


def format_candidates_prompt(
    task_type: str,
    candidates: list[dict[str, Any]],
) -> str:
    """Format a ranking prompt with candidate details.

    Args:
        task_type: Task type being evaluated
        candidates: List of candidate solver dicts

    Returns:
        Formatted prompt string
    """
    candidate_lines = []
    for i, c in enumerate(candidates, 1):
        scores = c.get("benchmark_scores", {})
        scores_str = ", ".join(f"{k}={v:.3f}" for k, v in scores.items()) if scores else "none"
        cost = c.get("estimated_cost")
        cost_str = f"${cost:.4f}" if cost is not None else "unknown"
        latency = c.get("estimated_latency_ms")
        latency_str = f"{latency}ms" if latency is not None else "unknown"

        candidate_lines.append(
            f"{i}. **{c['name']}** (id={c['solver_id']})\n"
            f"   - Source: {c['source']}, Type: {c['solver_type']}\n"
            f"   - Benchmarks: {scores_str}\n"
            f"   - Cost: {cost_str}, Latency: {latency_str}\n"
            f"   - License: {c.get('license', 'unknown')}"
        )

    candidates_text = "\n".join(candidate_lines)

    return (
        f"Evaluate these solver candidates for **{task_type}** and rank them "
        f"from BEST to WORST.\n\n"
        f"Candidates:\n{candidates_text}\n\n"
        f"Consider:\n"
        f"1. Benchmark scores (higher is better)\n"
        f"2. Cost per call (lower is better, if quality is similar)\n"
        f"3. Latency (lower is better, if quality is similar)\n"
        f"4. Source preference: local > encrypted > external\n"
        f"5. Specialization for {task_type}\n\n"
        f"Return ONLY valid JSON (no markdown):\n"
        f'{{"rankings": ["best-solver-id", "second-best-id", ...],'
        f' "reasoning": "Why the top solver is best",'
        f' "confidence": 0.85}}'
    )


def try_parse_json(text: str) -> dict[str, Any] | None:
    """Try to parse JSON from text, handling markdown code blocks.

    Args:
        text: Text that may contain JSON

    Returns:
        Parsed dict or None
    """
    if not text:
        return None

    # Try direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting from markdown code block
    match = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except (json.JSONDecodeError, TypeError):
            pass

    # Try finding JSON object in text
    match = re.search(r'\{.*"rankings".*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except (json.JSONDecodeError, TypeError):
            pass

    return None


def parse_consensus_output(
    result: ConsensusResult,
    candidates: list[dict[str, Any]],
) -> tuple[list[str], float, str, dict[str, str]]:
    """Parse rankings from consensus output.

    Args:
        result: ConsensusResult from consensus engine
        candidates: Original candidates (for fallback)

    Returns:
        Tuple of (rankings, confidence, reasoning, votes)
    """
    rankings: list[str] = []
    confidence = 0.5
    reasoning = "Consensus-based selection"
    votes: dict[str, str] = {}

    # Try to parse JSON from consensus outputs
    consensus_text = ""
    if result.consensus_outputs:
        # Get the merged output text
        for key, value in result.consensus_outputs.items():
            if isinstance(value, str):
                consensus_text = value
                break
            elif isinstance(value, dict):
                consensus_text = json.dumps(value)
                break

    # Try parsing individual vote outputs
    for vote in result.all_votes:
        if not vote.success:
            continue

        parsed = try_parse_json(vote.outputs.get("text", ""))
        if parsed:
            vote_rankings = parsed.get("rankings", [])
            if vote_rankings:
                votes[vote.head_id] = vote_rankings[0]  # Top pick

                # Use first successful parse as primary result
                if not rankings:
                    rankings = vote_rankings
                    confidence = parsed.get("confidence", 0.5)
                    reasoning = parsed.get("reasoning", reasoning)

    # If consensus outputs have a merged result, prefer it
    if consensus_text:
        parsed = try_parse_json(consensus_text)
        if parsed and parsed.get("rankings"):
            rankings = parsed["rankings"]
            confidence = parsed.get("confidence", confidence)
            reasoning = parsed.get("reasoning", reasoning)

    # Fallback: if no rankings parsed, use candidate order
    if not rankings:
        rankings = [c["solver_id"] for c in candidates]
        confidence = 0.3
        reasoning = "Fallback to registry order (consensus parsing failed)"

    # Validate rankings contain valid solver IDs
    valid_ids = {c["solver_id"] for c in candidates}
    rankings = [r for r in rankings if r in valid_ids]

    # Add any missing candidates at the end
    for c in candidates:
        if c["solver_id"] not in rankings:
            rankings.append(c["solver_id"])

    # Agreement score affects confidence
    confidence = min(1.0, confidence * result.agreement_score) if result.agreement_score > 0 else confidence * 0.5

    return rankings, confidence, reasoning, votes
