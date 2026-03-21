"""Meta-Reasoning Solver Selection (Phase 5).

MultiHead uses itself to decide which solver is best for a given task type.
The process:
1. Gather candidates from SolverRegistry
2. Run multi-head consensus to rank them
3. Optionally benchmark top candidates empirically
4. Synthesize final selection
5. Record preference for future routing
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from multihead.consensus import (
    ConsensusConfig,
    ConsensusStrategy,
    HeadTask,
)
from multihead.registry.solver_registry import SolverRegistry

from .models import SelectionResult
from .parsing import format_candidates_prompt, parse_consensus_output

if TYPE_CHECKING:
    from multihead.benchmarking.base import BenchmarkRunner
    from multihead.consensus import ConsensusEngine, ConsensusResult
    from multihead.head_manager import HeadManager

logger = logging.getLogger(__name__)


class MetaReasoningSelector:
    """Uses multi-head consensus to select the best solver for a task type.

    This is the core Phase 5 engine. Given a task_type (e.g., "object_detection"),
    it:
    1. Queries the registry for candidate solvers
    2. Formats a ranking prompt with candidate details
    3. Runs consensus across available heads
    4. Parses rankings from consensus output
    5. Optionally benchmarks top candidates
    6. Records the final preference in the registry
    """

    def __init__(
        self,
        head_manager: HeadManager,
        registry: SolverRegistry,
        benchmark_runner: BenchmarkRunner | None = None,
        *,
        consensus_strategy: ConsensusStrategy = ConsensusStrategy.WEIGHTED,
        consensus_heads: list[str] | None = None,
        top_k: int = 3,
        auto_record: bool = True,
    ):
        """Initialize meta-reasoning selector.

        Args:
            head_manager: HeadManager for running consensus
            registry: SolverRegistry with candidates and preferences
            benchmark_runner: Optional BenchmarkRunner for empirical validation
            consensus_strategy: Strategy for multi-head voting
            consensus_heads: Head IDs to use for consensus (None = all available LLMs)
            top_k: Number of top candidates to benchmark
            auto_record: Automatically record preference after selection
        """
        self.head_manager = head_manager
        self.registry = registry
        self.benchmark_runner = benchmark_runner

        from multihead.consensus import ConsensusEngine
        self.consensus_engine = ConsensusEngine(head_manager)
        self.strategy = consensus_strategy
        self.consensus_heads = consensus_heads
        self.top_k = top_k
        self.auto_record = auto_record

    async def select_best_solver(
        self,
        task_type: str,
        *,
        solver_type: str | None = None,
        min_candidates: int = 2,
        run_benchmarks: bool = False,
    ) -> SelectionResult:
        """Select the best solver for a task type using meta-reasoning.

        Args:
            task_type: Task type to evaluate (e.g., "object_detection")
            solver_type: Optional solver type filter
            min_candidates: Minimum candidates required
            run_benchmarks: Whether to empirically benchmark top candidates

        Returns:
            SelectionResult with selected solver and reasoning

        Raises:
            ValueError: If fewer than min_candidates are found
        """
        logger.info("Starting meta-reasoning selection for task_type=%s", task_type)

        # Step 1: Gather candidates
        candidates = self._gather_candidates(task_type, solver_type=solver_type)
        if len(candidates) < min_candidates:
            raise ValueError(
                f"Only {len(candidates)} candidates for {task_type}, "
                f"need at least {min_candidates}"
            )

        logger.info("Found %d candidates for %s", len(candidates), task_type)

        # Step 2: Run multi-head consensus ranking
        consensus_result = await self._run_consensus_ranking(task_type, candidates)

        # Step 3: Parse rankings from consensus
        rankings, confidence, reasoning, votes = parse_consensus_output(
            consensus_result, candidates,
        )

        # Step 4: Optionally benchmark top candidates
        benchmark_scores: dict[str, float] = {}
        if run_benchmarks and self.benchmark_runner:
            benchmark_scores = await self._benchmark_top_candidates(
                rankings[:self.top_k], solver_type or task_type,
            )
            # Re-rank based on empirical results (60% empirical, 40% consensus)
            rankings = self._rerank_with_benchmarks(rankings, benchmark_scores)

        # Step 5: Build result
        selected = rankings[0] if rankings else candidates[0]["solver_id"]

        result = SelectionResult(
            task_type=task_type,
            selected_solver_id=selected,
            reasoning=reasoning,
            confidence_score=confidence,
            rankings=rankings,
            consensus_votes=votes,
            benchmark_scores=benchmark_scores,
            candidates_evaluated=len(candidates),
        )

        # Step 6: Record preference
        if self.auto_record:
            self._record_preference(result)

        logger.info(
            "Meta-reasoning selected %s for %s (confidence=%.2f, %d candidates)",
            selected, task_type, confidence, len(candidates),
        )

        return result

    def _gather_candidates(
        self,
        task_type: str,
        *,
        solver_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Gather candidate solvers from registry.

        Args:
            task_type: Task type to search for
            solver_type: Optional solver type filter

        Returns:
            List of candidate solver dicts
        """
        # Get adopted and candidate solvers
        candidates = []

        for status in ("adopted", "candidate"):
            solvers = self.registry.list_solvers(
                solver_type=solver_type,
                adoption_status=status,
            )
            candidates.extend(solvers)

        # Filter by task_type if present in solver's task_types
        if task_type:
            filtered = [
                s for s in candidates
                if task_type in s.get("task_types", [])
            ]
            # If filtering removed everything, use all candidates of matching type
            if filtered:
                candidates = filtered

        # Deduplicate
        seen = set()
        unique = []
        for c in candidates:
            if c["solver_id"] not in seen:
                seen.add(c["solver_id"])
                unique.append(c)

        return unique

    async def _run_consensus_ranking(
        self,
        task_type: str,
        candidates: list[dict[str, Any]],
    ) -> ConsensusResult:
        """Run multi-head consensus to rank candidates.

        Args:
            task_type: Task type being evaluated
            candidates: Candidates to rank

        Returns:
            ConsensusResult from the consensus engine
        """
        prompt = format_candidates_prompt(task_type, candidates)

        # Determine heads to use
        head_tasks = self._get_consensus_heads()

        config = ConsensusConfig(
            strategy=self.strategy,
            heads=head_tasks,
            output_schema={
                "type": "object",
                "required": ["rankings", "reasoning", "confidence"],
                "properties": {
                    "rankings": {"type": "array"},
                    "reasoning": {"type": "string"},
                    "confidence": {"type": "number"},
                },
            },
            timeout_seconds=120,
        )

        result = await self.consensus_engine.execute(config, prompt)
        return result

    def _get_consensus_heads(self) -> list[HeadTask]:
        """Get head tasks for consensus.

        Uses configured heads or defaults to all available LLM heads.

        Returns:
            List of HeadTask configurations
        """
        if self.consensus_heads:
            return [
                HeadTask(head_id=hid, weight=1.0, required=False)
                for hid in self.consensus_heads
            ]

        # Default: use all available LLM heads
        states = self.head_manager.get_states()
        llm_heads = [
            hid for hid, state in states.items()
            if state.get("kind") in ("llm", "vlm")
        ]

        if not llm_heads:
            raise ValueError("No LLM/VLM heads available for consensus")

        return [
            HeadTask(head_id=hid, weight=1.0, required=(i == 0))
            for i, hid in enumerate(llm_heads)
        ]

    async def _benchmark_top_candidates(
        self,
        solver_ids: list[str],
        solver_type: str,
    ) -> dict[str, float]:
        """Benchmark top candidates empirically.

        Args:
            solver_ids: Top solver IDs to benchmark
            solver_type: Solver type for benchmark selection

        Returns:
            Dict of solver_id -> aggregate score
        """
        if not self.benchmark_runner:
            return {}

        scores: dict[str, float] = {}

        for solver_id in solver_ids:
            try:
                # Check if we have recent benchmark results
                results = self.registry.get_benchmark_results(solver_id, limit=5)
                if results:
                    # Use existing results
                    avg_score = sum(r["score"] for r in results) / len(results)
                    scores[solver_id] = avg_score
                    logger.debug(
                        "Using cached benchmarks for %s: %.2f (%d results)",
                        solver_id, avg_score, len(results),
                    )
                else:
                    logger.debug("No benchmark results for %s, skipping", solver_id)

            except Exception as e:
                logger.warning("Benchmarking %s failed: %s", solver_id, e)

        return scores

    def _rerank_with_benchmarks(
        self,
        rankings: list[str],
        benchmark_scores: dict[str, float],
    ) -> list[str]:
        """Re-rank based on empirical benchmark results.

        Weight: 60% empirical, 40% consensus position.

        Args:
            rankings: Current rankings from consensus
            benchmark_scores: Empirical scores per solver

        Returns:
            Re-ranked list of solver IDs
        """
        if not benchmark_scores:
            return rankings

        # Compute combined scores
        combined: dict[str, float] = {}
        n = len(rankings)

        for i, solver_id in enumerate(rankings):
            # Consensus score: linearly decreasing from 1.0 to 0.0
            consensus_score = 1.0 - (i / n) if n > 1 else 1.0

            # Benchmark score (0.0 if not benchmarked)
            bench_score = benchmark_scores.get(solver_id, 0.0)

            if solver_id in benchmark_scores:
                # 60% empirical, 40% consensus
                combined[solver_id] = 0.6 * bench_score + 0.4 * consensus_score
            else:
                # No empirical data: 100% consensus, slight penalty
                combined[solver_id] = 0.4 * consensus_score * 0.8

        # Sort by combined score descending
        reranked = sorted(combined.keys(), key=lambda s: combined[s], reverse=True)

        # Add any unranked solvers at the end
        for solver_id in rankings:
            if solver_id not in reranked:
                reranked.append(solver_id)

        return reranked

    def _record_preference(self, result: SelectionResult) -> None:
        """Record selection as preference in registry.

        Args:
            result: SelectionResult to record
        """
        try:
            self.registry.record_selection(
                task_type=result.task_type,
                preferred_solver_id=result.selected_solver_id,
                reasoning=result.reasoning,
                confidence_score=result.confidence_score,
                consensus_votes=result.consensus_votes,
                benchmark_results=result.benchmark_scores,
            )
            logger.info(
                "Recorded preference: %s -> %s (confidence=%.2f)",
                result.task_type, result.selected_solver_id, result.confidence_score,
            )
        except Exception as e:
            logger.error("Failed to record preference: %s", e)

    def get_current_preference(self, task_type: str) -> dict[str, Any] | None:
        """Get the current preference for a task type.

        Args:
            task_type: Task type to query

        Returns:
            Preference dict or None
        """
        return self.registry.get_preference(task_type)

    def list_preferences(self) -> list[dict[str, Any]]:
        """List all recorded preferences.

        Returns:
            List of preference dicts
        """
        return self.registry.list_preferences()

    async def evaluate_all_task_types(
        self,
        task_types: list[str] | None = None,
    ) -> list[SelectionResult]:
        """Run meta-reasoning selection for multiple task types.

        Args:
            task_types: Task types to evaluate (None = discover from registry)

        Returns:
            List of SelectionResult objects
        """
        if task_types is None:
            # Discover task types from registered solvers
            task_types = self._discover_task_types()

        results = []
        for task_type in task_types:
            try:
                result = await self.select_best_solver(
                    task_type,
                    min_candidates=2,
                )
                results.append(result)
            except ValueError as e:
                logger.info("Skipping %s: %s", task_type, e)
            except Exception as e:
                logger.error("Meta-reasoning failed for %s: %s", task_type, e)

        return results

    def _discover_task_types(self) -> list[str]:
        """Discover all task types from registered solvers.

        Returns:
            List of unique task types
        """
        task_types: set[str] = set()

        for status in ("adopted", "candidate"):
            solvers = self.registry.list_solvers(adoption_status=status)
            for solver in solvers:
                for tt in solver.get("task_types", []):
                    task_types.add(tt)

        return sorted(task_types)
