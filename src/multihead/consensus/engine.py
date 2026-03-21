"""Consensus engine: orchestration, head execution, and proposal ranking."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from typing import Any

from ..head_manager import HeadManager
from ..observability import MetricsCollector
from ..resilience import CircuitBreakerOpen
from .models import (
    ConsensusConfig,
    ConsensusResult,
    ConsensusStrategy,
    HeadTask,
    VoteResult,
)
from .strategies import ConsensusStrategiesMixin
from ._first_to_ahead import FirstToAheadMixin

logger = logging.getLogger(__name__)


class ConsensusEngine(ConsensusStrategiesMixin, FirstToAheadMixin):
    """Orchestrates multi-head consensus and cross-modal verification.

    Supports two modes:
    - K-Voting: same prompt to K heads, vote on output
    - Cross-Modal: different prompts to specialist heads, verify consistency
    """

    def __init__(self, head_manager: HeadManager, metrics: MetricsCollector | None = None) -> None:
        self.heads = head_manager
        self._metrics = metrics

    async def execute(
        self,
        config: ConsensusConfig,
        base_prompt: str,
    ) -> ConsensusResult:
        """Execute consensus across multiple heads.

        1. Call each head (custom prompt or base_prompt)
        2. Validate outputs against schema
        3. Apply consensus strategy
        4. Detect red flags
        5. Return ConsensusResult
        """
        # Validate all head IDs exist before starting execution
        available = set(self.heads.get_states().keys())
        unknown = [ht.head_id for ht in config.heads if ht.head_id not in available]
        if unknown:
            raise ValueError(f"Unknown head(s): {unknown}. Available: {sorted(available)}")

        # FIRST_TO_AHEAD: dynamic sampling loop (separate path)
        if config.strategy == ConsensusStrategy.FIRST_TO_AHEAD:
            return await self._resolve_first_to_ahead(config, base_prompt)

        votes: list[VoteResult] = []

        for head_task in config.heads:
            vote = await self._execute_head(
                head_task, base_prompt, config.output_schema, config.timeout_seconds,
            )
            votes.append(vote)

        # Filter to successful, schema-valid votes for consensus
        valid_votes = [v for v in votes if v.success and v.schema_valid]

        # Apply consensus strategy
        if valid_votes:
            consensus_outputs, agreement_score = self._apply_strategy(
                valid_votes, config,
            )
        else:
            consensus_outputs = {}
            agreement_score = 0.0

        # Detect red flags
        red_flags = self._detect_red_flags(votes, valid_votes, consensus_outputs, config)

        # Cross-modal verification
        if config.cross_modal and valid_votes:
            cross_flags = self._verify_cross_modal(valid_votes, config.heads)
            red_flags.extend(cross_flags)

        # Build metrics
        metrics = {
            "total_heads": len(config.heads),
            "successful_heads": len([v for v in votes if v.success]),
            "schema_valid_heads": len(valid_votes),
            "total_latency_ms": sum(v.latency_ms for v in votes),
            "avg_latency_ms": (
                sum(v.latency_ms for v in votes) / len(votes) if votes else 0
            ),
        }

        result = ConsensusResult(
            consensus_outputs=consensus_outputs,
            all_votes=votes,
            agreement_score=agreement_score,
            red_flags=red_flags,
            strategy_used=config.strategy,
            metrics=metrics,
        )

        # Record observability metrics
        if self._metrics:
            self._metrics.inc("consensus_executions_total")
            self._metrics.inc(
                "consensus_executions_total",
                labels={"strategy": config.strategy.value},
            )
            self._metrics.observe("consensus_agreement_score", agreement_score)
            self._metrics.inc("consensus_red_flags_total", value=len(red_flags))
            self._metrics.observe(
                "consensus_latency_ms", metrics["total_latency_ms"],
            )

        return result

    async def _execute_head(
        self,
        head_task: HeadTask,
        base_prompt: str,
        output_schema: dict[str, Any],
        timeout_seconds: float = 30.0,
        **kwargs: Any,
    ) -> VoteResult:
        """Execute a single head and validate output."""
        prompt = head_task.prompt_template if head_task.prompt_template else base_prompt

        t0 = time.perf_counter()
        try:
            async with asyncio.timeout(timeout_seconds):
                response = await self.heads.generate(head_task.head_id, prompt, **kwargs)
            latency_ms = (time.perf_counter() - t0) * 1000

            outputs = response
            schema_valid = True
            schema_errors: list[str] = []

            # Validate against schema if provided
            if output_schema:
                schema_valid, schema_errors = self._validate_schema(
                    response.get("text", ""), output_schema,
                )

            return VoteResult(
                head_id=head_task.head_id,
                outputs=outputs,
                success=True,
                schema_valid=schema_valid,
                schema_errors=schema_errors,
                latency_ms=latency_ms,
            )

        except TimeoutError:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.warning(
                "Head %s timed out after %.1fs", head_task.head_id, timeout_seconds,
            )
            return VoteResult(
                head_id=head_task.head_id,
                success=False,
                error=f"timeout: exceeded {timeout_seconds}s",
                latency_ms=latency_ms,
            )
        except CircuitBreakerOpen as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.warning("Head %s circuit breaker open: %s", head_task.head_id, e)
            return VoteResult(
                head_id=head_task.head_id,
                success=False,
                error=f"circuit_breaker_open: {e}",
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.warning("Head %s failed: %s", head_task.head_id, e)
            return VoteResult(
                head_id=head_task.head_id,
                success=False,
                error=str(e),
                latency_ms=latency_ms,
            )

    # ------------------------------------------------------------------
    # Proposal ranking (for SolveCoordinator)
    # ------------------------------------------------------------------

    def rank_proposals(
        self,
        votes: list[VoteResult],
        strategy: ConsensusStrategy = ConsensusStrategy.MAJORITY,
        threshold: float = 0.5,
        weights: dict[str, float] | None = None,
    ) -> ConsensusResult:
        """Rank pre-existing votes (proposals) using the configured strategy.

        Unlike execute(), this does NOT call model heads -- it takes
        already-generated VoteResults and applies the consensus strategy
        to pick a winner.  Used by SolveCoordinator to vote on agent
        proposals that are already collected.

        Args:
            votes: Pre-existing VoteResults (one per proposal).
            strategy: Which consensus strategy to apply.
            threshold: Threshold for THRESHOLD strategy.
            weights: Optional head_id -> weight map for WEIGHTED strategy.

        Returns:
            ConsensusResult with consensus_outputs, agreement_score, etc.
        """
        if not votes:
            return ConsensusResult(
                strategy_used=strategy,
                red_flags=[{
                    "type": "no_valid_votes",
                    "severity": "critical",
                    "message": "No proposals to rank",
                    "details": {"total_votes": 0},
                }],
            )

        valid_votes = [v for v in votes if v.success]

        if not valid_votes:
            return ConsensusResult(
                all_votes=votes,
                strategy_used=strategy,
                red_flags=[{
                    "type": "no_valid_votes",
                    "severity": "critical",
                    "message": "No valid proposals to rank",
                    "details": {"total_votes": len(votes)},
                }],
            )

        # Build a synthetic ConsensusConfig for strategy dispatch
        head_tasks = []
        for v in valid_votes:
            w = (weights or {}).get(v.head_id, 1.0)
            head_tasks.append(HeadTask(head_id=v.head_id, weight=w))

        config = ConsensusConfig(
            heads=head_tasks,
            strategy=strategy,
            threshold=threshold,
        )

        # For FIRST_TO_AHEAD on pre-existing votes, use canonical hashing
        if strategy == ConsensusStrategy.FIRST_TO_AHEAD:
            return self._rank_fta(valid_votes, votes)

        consensus_outputs, agreement_score = self._apply_strategy(
            valid_votes, config,
        )

        red_flags = self._detect_red_flags(votes, valid_votes, consensus_outputs, config)

        metrics = {
            "total_proposals": len(votes),
            "valid_proposals": len(valid_votes),
            "mode": "rank_proposals",
        }

        return ConsensusResult(
            consensus_outputs=consensus_outputs,
            all_votes=votes,
            agreement_score=agreement_score,
            red_flags=red_flags,
            strategy_used=strategy,
            metrics=metrics,
        )
