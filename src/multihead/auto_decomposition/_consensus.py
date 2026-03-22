"""Multi-head consensus decomposition and plan voting for AutoDecomposer."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from multihead.consensus import ConsensusConfig, ConsensusEngine, ConsensusStrategy, HeadTask
from multihead.decomposer import DecompositionPlan

from .research import ResearchFeatureIntegrator
from .validators import AtomicityValidator, CompletenessValidator

logger = logging.getLogger(__name__)


class ConsensusDecompositionMixin:
    """Mixin providing multi-head consensus decomposition and plan voting.

    Expects self._base_decomposer from the main AutoDecomposer class.
    """

    async def decompose_with_consensus(
        self,
        goal: str,
        context: dict[str, Any] | None = None,
        heads: list[str] | None = None,
        strategy: ConsensusStrategy = ConsensusStrategy.FIRST_TO_AHEAD,
        weights: dict[str, float] | None = None,
        auto_validate: bool = True,
        enable_research_features: bool = True,
    ) -> tuple[DecompositionPlan, dict[str, Any]]:
        """Decompose with multi-head consensus voting on best plan.

        Args:
            goal: Task to decompose
            context: Additional context for decomposition
            heads: List of head IDs to use for voting (default: all available)
            strategy: Consensus strategy to use
            weights: Optional weights per head for WEIGHTED strategy
            auto_validate: Run atomicity and completeness validation
            enable_research_features: Auto-enable ToT/PRM/Reflection

        Returns:
            (winning_plan, consensus_metadata) tuple where metadata includes:
            - agreement_score: consensus agreement (0-1)
            - votes: list of all plans proposed
            - winner_head: which head's plan won
            - red_flags: any consensus red flags
        """
        from multihead.head_manager import HeadManager

        logger.info(
            "Multi-head consensus decomposition: goal='%s', strategy=%s, heads=%s",
            goal[:60],
            strategy.value,
            heads or "all",
        )

        # Get head manager to execute decomposition on multiple heads
        head_mgr = self._base_decomposer.heads

        # Prepare consensus configuration
        if heads is None:
            # Use only active/loaded LLM heads (skip offline and mock ones)
            from multihead.models import AdapterKind, HeadState
            heads = [h for h, manifest in head_mgr._manifests.items()
                    if manifest.kind == "llm"
                    and manifest.adapter != AdapterKind.MOCK
                    and head_mgr.get_state(h) == HeadState.ACTIVE]
            if not heads:
                # Fallback: first non-mock LLM head (will be loaded on demand)
                heads = [h for h, manifest in head_mgr._manifests.items()
                        if manifest.kind == "llm"
                        and manifest.adapter != AdapterKind.MOCK][:1]

        # Build HeadTask configs
        head_tasks = []
        for head_id in heads:
            weight = weights.get(head_id, 1.0) if weights else 1.0
            head_tasks.append(HeadTask(head_id=head_id, weight=weight))

        consensus_config = ConsensusConfig(
            heads=head_tasks,
            strategy=strategy,
            fail_on_disagreement=False,  # Allow disagreement, pick best
            timeout_seconds=60.0,  # Longer timeout for decomposition
        )

        # Build decomposition prompt
        prompt_parts = [f"Decompose this task into atomic steps:\n\n{goal}"]
        if context:
            prompt_parts.append(f"\nContext:\n{json.dumps(context, indent=2)}")

        prompt = "\n".join(prompt_parts)

        # Execute decomposition on all heads via ConsensusEngine
        consensus_engine = ConsensusEngine(head_mgr, metrics=None)

        # Collect all decomposition plans
        all_plans: list[DecompositionPlan] = []
        plan_votes: list[tuple[str, DecompositionPlan]] = []  # (head_id, plan)

        for head_task in head_tasks:
            try:
                # Run decomposition with this head
                plan = await self._base_decomposer.decompose(
                    goal,
                    context=context,
                    head_id=head_task.head_id,
                )

                # Validate plan if requested
                if auto_validate:
                    # Run atomicity validation
                    issues = []
                    for leaf in plan.all_leaves():
                        is_atomic, reason = AtomicityValidator.is_atomic(leaf)
                        if not is_atomic and reason:
                            issues.append(f"Non-atomic step {leaf.id}: {reason}")

                    # Run completeness validation
                    is_complete, warnings = CompletenessValidator.validate(goal, plan)
                    if not is_complete:
                        issues.extend(warnings)

                    # Store validation issues in plan metadata
                    plan._validation_issues = issues  # type: ignore

                all_plans.append(plan)
                plan_votes.append((head_task.head_id, plan))

                logger.info(
                    "Head %s proposed plan: %d steps, complexity=%s",
                    head_task.head_id,
                    plan.total_steps,
                    plan.complexity,
                )

            except Exception as e:
                logger.warning(
                    "Head %s failed to decompose: %s",
                    head_task.head_id,
                    e,
                )

        if not all_plans:
            raise ValueError("No heads successfully decomposed the task")

        # Vote on best plan
        winner_plan, consensus_meta = self._vote_on_plans(
            plan_votes,
            strategy,
            weights or {},
        )

        # Apply research features to winning plan
        if enable_research_features:
            feature_configs = ResearchFeatureIntegrator.integrate(
                winner_plan,
                enable_auto_features=True,
            )
            winner_plan._feature_configs = feature_configs  # type: ignore

        logger.info(
            "Consensus winner: %d steps, agreement=%.2f, winner_head=%s",
            winner_plan.total_steps,
            consensus_meta["agreement_score"],
            consensus_meta["winner_head"],
        )

        return winner_plan, consensus_meta

    def _vote_on_plans(
        self,
        plan_votes: list[tuple[str, DecompositionPlan]],
        strategy: ConsensusStrategy,
        weights: dict[str, float],
    ) -> tuple[DecompositionPlan, dict[str, Any]]:
        """Vote on best decomposition plan using consensus strategy.

        Args:
            plan_votes: List of (head_id, plan) tuples
            strategy: Consensus strategy to use
            weights: Weight per head for WEIGHTED strategy

        Returns:
            (winning_plan, metadata) tuple
        """
        # Score each plan
        plan_scores: list[tuple[float, str, DecompositionPlan]] = []

        for head_id, plan in plan_votes:
            # Calculate plan quality score
            score = self._score_plan_quality(plan)

            # Apply weight if using WEIGHTED strategy
            if strategy == ConsensusStrategy.WEIGHTED:
                weight = weights.get(head_id, 1.0)
                score *= weight

            plan_scores.append((score, head_id, plan))

        # Sort by score (highest first)
        plan_scores.sort(reverse=True, key=lambda x: x[0])

        # Pick winner based on strategy
        if strategy == ConsensusStrategy.MAJORITY:
            # Majority vote: group similar plans and pick most common
            plan_signatures = {}
            for score, head_id, plan in plan_scores:
                sig = self._plan_signature(plan)
                if sig not in plan_signatures:
                    plan_signatures[sig] = []
                plan_signatures[sig].append((score, head_id, plan))

            # Pick signature with most votes
            winner_sig = max(plan_signatures, key=lambda s: len(plan_signatures[s]))
            winner_score, winner_head, winner_plan = plan_signatures[winner_sig][0]
            agreement_score = len(plan_signatures[winner_sig]) / len(plan_votes)

        elif strategy == ConsensusStrategy.UNANIMOUS:
            # Unanimous: all plans must be identical
            first_sig = self._plan_signature(plan_scores[0][2])
            if all(self._plan_signature(p[2]) == first_sig for p in plan_scores):
                winner_score, winner_head, winner_plan = plan_scores[0]
                agreement_score = 1.0
            else:
                # Fall back to highest quality if not unanimous
                winner_score, winner_head, winner_plan = plan_scores[0]
                agreement_score = 0.0

        else:  # WEIGHTED, FIRST_TO_AHEAD, THRESHOLD all use quality scoring
            winner_score, winner_head, winner_plan = plan_scores[0]
            # Calculate agreement as normalized score difference
            if len(plan_scores) > 1:
                second_score = plan_scores[1][0]
                score_gap = winner_score - second_score
                agreement_score = min(1.0, score_gap / max(winner_score, 0.01))
            else:
                agreement_score = 1.0

        metadata = {
            "agreement_score": agreement_score,
            "winner_head": winner_head,
            "winner_score": winner_score,
            "all_scores": [(head, score) for score, head, _ in plan_scores],
            "num_votes": len(plan_votes),
            "strategy": strategy.value,
            "red_flags": [],
        }

        return winner_plan, metadata

    @staticmethod
    def _score_plan_quality(plan: DecompositionPlan) -> float:
        """Score decomposition plan quality (higher is better).

        Scoring factors:
        - Atomicity: fewer validation issues
        - Completeness: covers all keywords
        - Structure: reasonable step count (not too many/few)
        - Complexity match: complexity assessment matches plan structure
        """
        score = 100.0

        # Penalty for validation issues
        validation_issues = getattr(plan, "_validation_issues", [])
        score -= len(validation_issues) * 5

        # Penalty for extreme step counts
        step_count = plan.total_steps
        if step_count < 3:
            score -= 10  # Too coarse
        elif step_count > 50:
            score -= (step_count - 50) * 2  # Too granular

        # Bonus for balanced depth
        depth = plan.max_depth
        if 2 <= depth <= 4:
            score += 5  # Good decomposition depth

        # Bonus for complexity-appropriate step count
        if plan.complexity == "simple" and 3 <= step_count <= 10:
            score += 10
        elif plan.complexity == "moderate" and 5 <= step_count <= 20:
            score += 10
        elif plan.complexity == "complex" and 10 <= step_count <= 40:
            score += 10

        return max(0.0, score)

    @staticmethod
    def _plan_signature(plan: DecompositionPlan) -> str:
        """Generate signature for plan similarity detection.

        Plans are considered similar if they have same:
        - Number of phases
        - Number of total steps
        - Similar goal decomposition structure
        """
        # Create canonical representation
        sig_data = {
            "phases": len(plan.phases),
            "total_steps": plan.total_steps,
            "complexity": plan.complexity,
            "leaf_actions": sorted([leaf.action_type for leaf in plan.all_leaves()]),
        }

        sig_json = json.dumps(sig_data, sort_keys=True)
        return hashlib.md5(sig_json.encode()).hexdigest()[:8]
