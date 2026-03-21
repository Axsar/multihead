"""First-to-ahead dynamic sampling and result building for ConsensusEngine."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from .models import (
    ConsensusConfig,
    ConsensusResult,
    ConsensusStrategy,
    FirstToAheadConfig,
    VoteResult,
)

logger = logging.getLogger(__name__)


class FirstToAheadMixin:
    """Mixin providing FIRST_TO_AHEAD dynamic sampling and ranking.

    Expects self._execute_head(), self._check_red_flags_pre_vote(),
    self._canonical_hash(), and self._metrics from the engine/strategies.
    """

    async def _resolve_first_to_ahead(
        self,
        config: ConsensusConfig,
        base_prompt: str,
    ) -> ConsensusResult:
        """MAKER-style dynamic sampling with first-to-ahead-by-K voting.

        Samples candidates sequentially through the configured heads (cycling),
        discards red-flagged outputs, clusters remaining into equivalence buckets,
        and stops when one bucket leads by k_margin votes.
        """
        policy = config.first_to_ahead or FirstToAheadConfig()
        all_votes: list[VoteResult] = []
        bucket_counts: Counter[str] = Counter()
        bucket_exemplars: dict[str, VoteResult] = {}  # bucket_id -> first vote in bucket
        valid_count = 0
        discarded_count = 0
        discard_reasons: Counter[str] = Counter()
        escalated = False

        for sample_idx in range(policy.max_samples):
            # Cycle through configured heads
            head_task = config.heads[sample_idx % len(config.heads)]

            # Escalation: lower temperature after stall
            kwargs: dict[str, Any] = {}
            if escalated:
                kwargs["temperature"] = 0.3
            else:
                kwargs["temperature"] = 0.7

            vote = await self._execute_head(
                head_task, base_prompt, config.output_schema,
                config.timeout_seconds, **kwargs,
            )
            all_votes.append(vote)

            # Pre-vote red-flag filter
            is_flagged, reason = self._check_red_flags_pre_vote(
                vote, policy, config.output_schema,
            )
            if is_flagged:
                discarded_count += 1
                discard_reasons[reason.split(":")[0]] += 1
                logger.debug(
                    "FTA sample %d discarded: %s (head=%s)",
                    sample_idx, reason, head_task.head_id,
                )
                continue

            valid_count += 1
            text = vote.outputs.get("text", "")
            bucket_id = self._canonical_hash(text)
            bucket_counts[bucket_id] += 1
            if bucket_id not in bucket_exemplars:
                bucket_exemplars[bucket_id] = vote

            # Check stopping condition
            if valid_count >= policy.min_samples and len(bucket_counts) > 0:
                top2 = bucket_counts.most_common(2)
                leader_id, leader_count = top2[0]
                runner_count = top2[1][1] if len(top2) > 1 else 0

                if leader_count - runner_count >= policy.k_margin:
                    logger.info(
                        "FTA resolved at sample %d: margin=%d (leader=%d, runner=%d, buckets=%d)",
                        sample_idx + 1, leader_count - runner_count,
                        leader_count, runner_count, len(bucket_counts),
                    )
                    winner = bucket_exemplars[leader_id]
                    return self._build_fta_result(
                        winner=winner,
                        all_votes=all_votes,
                        bucket_counts=bucket_counts,
                        valid_count=valid_count,
                        discarded_count=discarded_count,
                        discard_reasons=discard_reasons,
                        exhausted=False,
                        escalated=escalated,
                    )

            # Escalation on stall
            if (sample_idx + 1) >= policy.stall_threshold and not escalated:
                escalated = True
                logger.info(
                    "FTA escalating after %d samples (no k=%d winner yet, buckets=%d)",
                    sample_idx + 1, policy.k_margin, len(bucket_counts),
                )
                if self._metrics:
                    self._metrics.inc("consensus_fta_escalations")

        # Exhausted max_samples -- return argmax
        if not bucket_counts:
            return self._build_fta_result(
                winner=None,
                all_votes=all_votes,
                bucket_counts=bucket_counts,
                valid_count=valid_count,
                discarded_count=discarded_count,
                discard_reasons=discard_reasons,
                exhausted=True,
                escalated=escalated,
            )

        leader_id = bucket_counts.most_common(1)[0][0]
        winner = bucket_exemplars[leader_id]
        logger.warning(
            "FTA exhausted %d samples without k=%d margin; returning argmax (buckets=%d)",
            policy.max_samples, policy.k_margin, len(bucket_counts),
        )
        return self._build_fta_result(
            winner=winner,
            all_votes=all_votes,
            bucket_counts=bucket_counts,
            valid_count=valid_count,
            discarded_count=discarded_count,
            discard_reasons=discard_reasons,
            exhausted=True,
            escalated=escalated,
        )

    def _build_fta_result(
        self,
        winner: VoteResult | None,
        all_votes: list[VoteResult],
        bucket_counts: Counter[str],
        valid_count: int,
        discarded_count: int,
        discard_reasons: Counter[str],
        exhausted: bool,
        escalated: bool,
    ) -> ConsensusResult:
        """Build ConsensusResult for FIRST_TO_AHEAD strategy."""
        if winner is not None:
            consensus_outputs = dict(winner.outputs)
            top2 = bucket_counts.most_common(2)
            leader_count = top2[0][1]
            runner_count = top2[1][1] if len(top2) > 1 else 0
            agreement = leader_count / valid_count if valid_count else 0.0
        else:
            consensus_outputs = {}
            agreement = 0.0
            leader_count = 0
            runner_count = 0

        # Build red flags
        red_flags: list[dict[str, Any]] = []
        if winner is None:
            red_flags.append({
                "type": "no_valid_votes",
                "severity": "critical",
                "message": "No valid candidates after red-flag filtering",
                "details": {"total_samples": len(all_votes), "discarded": discarded_count},
            })
        if exhausted and winner is not None:
            red_flags.append({
                "type": "fta_exhausted",
                "severity": "medium",
                "message": (
                    f"Max samples exhausted without k-margin convergence "
                    f"(leader={leader_count}, runner={runner_count})"
                ),
                "details": {
                    "leader_count": leader_count,
                    "runner_count": runner_count,
                    "unique_buckets": len(bucket_counts),
                },
            })

        metrics = {
            "total_samples": len(all_votes),
            "valid_samples": valid_count,
            "discarded_samples": discarded_count,
            "discard_reasons": dict(discard_reasons),
            "unique_buckets": len(bucket_counts),
            "leader_count": leader_count,
            "runner_count": runner_count,
            "fta_exhausted": exhausted,
            "fta_escalated": escalated,
        }

        # Record observability metrics
        if self._metrics:
            self._metrics.inc("consensus_executions_total")
            self._metrics.inc(
                "consensus_executions_total",
                labels={"strategy": "first_to_ahead"},
            )
            self._metrics.observe("consensus_agreement_score", agreement)
            self._metrics.inc("consensus_fta_samples_total", value=len(all_votes))
            self._metrics.inc("consensus_fta_samples_discarded", value=discarded_count)

        return ConsensusResult(
            consensus_outputs=consensus_outputs,
            all_votes=all_votes,
            agreement_score=agreement,
            red_flags=red_flags,
            strategy_used=ConsensusStrategy.FIRST_TO_AHEAD,
            metrics=metrics,
        )

    def _rank_fta(
        self,
        valid_votes: list[VoteResult],
        all_votes: list[VoteResult],
    ) -> ConsensusResult:
        """FIRST_TO_AHEAD ranking for pre-existing proposals."""
        bucket_counts: Counter[str] = Counter()
        bucket_exemplars: dict[str, VoteResult] = {}

        for vote in valid_votes:
            text = vote.outputs.get("text", "")
            bucket_id = self._canonical_hash(text)
            bucket_counts[bucket_id] += 1
            if bucket_id not in bucket_exemplars:
                bucket_exemplars[bucket_id] = vote

        if not bucket_counts:
            return ConsensusResult(
                all_votes=all_votes,
                strategy_used=ConsensusStrategy.FIRST_TO_AHEAD,
                red_flags=[{
                    "type": "no_valid_votes",
                    "severity": "critical",
                    "message": "No valid proposals after hashing",
                    "details": {},
                }],
            )

        top2 = bucket_counts.most_common(2)
        leader_id, leader_count = top2[0]
        runner_count = top2[1][1] if len(top2) > 1 else 0
        winner = bucket_exemplars[leader_id]
        agreement = leader_count / len(valid_votes)

        return ConsensusResult(
            consensus_outputs=dict(winner.outputs),
            all_votes=all_votes,
            agreement_score=agreement,
            strategy_used=ConsensusStrategy.FIRST_TO_AHEAD,
            metrics={
                "total_proposals": len(all_votes),
                "valid_proposals": len(valid_votes),
                "unique_buckets": len(bucket_counts),
                "leader_count": leader_count,
                "runner_count": runner_count,
                "mode": "rank_proposals",
            },
        )
