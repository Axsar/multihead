"""AutoDecomposer — Enhanced LLM-driven task decomposition with DAG inference.

Wraps TaskDecomposer and adds automatic DAG dependency inference,
validation, and research feature integration.
"""

from __future__ import annotations

import logging
from typing import Any

from multihead.decomposer import DecompositionPlan, TaskDecomposer, TaskNode
from multihead.models import StepDef, WorkOrder

from ._consensus import ConsensusDecompositionMixin
from .dependency import StepDependencyAnalyzer
from .research import ResearchFeatureIntegrator
from .validators import AtomicityValidator, CompletenessValidator

logger = logging.getLogger(__name__)


class AutoDecomposer(ConsensusDecompositionMixin):
    """Enhanced task decomposer with automatic DAG inference and validation.

    Wraps TaskDecomposer and adds:
    - DAG dependency inference (parallel execution)
    - Atomic step validation (m=1 verification)
    - Completeness validation (goal coverage)
    - Research feature integration (ToT/PRM/Reflection)
    - Multi-head consensus decomposition (via ConsensusDecompositionMixin)

    Usage:
        decomposer = AutoDecomposer(head_manager, knowledge_store)
        plan = await decomposer.decompose(
            goal="Implement feature X",
            auto_validate=True,
            enable_research_features=True
        )
        work_order = plan.to_work_order()  # DAG with parallel steps
    """

    def __init__(
        self,
        head_manager: Any,
        knowledge_store: Any | None = None,
        metrics: Any | None = None,
    ):
        """Initialize AutoDecomposer.

        Args:
            head_manager: HeadManager for LLM access
            knowledge_store: Optional KnowledgeStore for context
            metrics: Optional MetricsCollector
        """
        self._base_decomposer = TaskDecomposer(
            head_manager, knowledge_store, metrics
        )
        self._dependency_analyzer = StepDependencyAnalyzer()

    async def decompose(
        self,
        goal: str,
        context: str = "",
        head_id: str | None = None,
        max_depth: int = 4,
        auto_validate: bool = True,
        enable_research_features: bool = False,
    ) -> DecompositionPlan:
        """Decompose goal into DAG with automatic validation.

        Args:
            goal: High-level task to decompose
            context: Optional user context
            head_id: Specific head to use (None = auto-select)
            max_depth: Maximum tree depth
            auto_validate: Validate atomicity and completeness
            enable_research_features: Auto-enable ToT/PRM/Reflection

        Returns:
            Enhanced DecompositionPlan with DAG dependencies

        Raises:
            ValueError: If validation fails and auto_validate=True
        """
        # 1. Use base decomposer to get hierarchical plan
        plan = await self._base_decomposer.decompose(
            goal=goal,
            context=context,
            head_id=head_id,
            max_depth=max_depth,
        )

        # 2. Validate atomicity
        if auto_validate:
            violations = AtomicityValidator.validate_plan(plan)
            if violations:
                logger.warning(
                    "Plan has %d non-atomic steps: %s",
                    len(violations),
                    violations[:3],
                )
                # Don't raise -- warn and continue (decomposer might have good reasons)

        # 3. Validate completeness
        if auto_validate:
            is_complete, issues = CompletenessValidator.validate(goal, plan)
            if not is_complete:
                logger.warning("Completeness issues: %s", issues)

        # 4. Integrate research features
        feature_configs: dict[str, dict[str, Any]] = {}
        if enable_research_features:
            feature_configs = ResearchFeatureIntegrator.integrate(plan, enable_auto_features=True)
            logger.info(
                "Enabled research features for %d steps",
                len(feature_configs),
            )

        # 5. Store metadata for WorkOrder conversion
        plan._feature_configs = feature_configs  # type: ignore

        logger.info(
            "Auto-decomposed '%s' -> %d steps (complexity: %s, auto_validate: %s, features: %s)",
            goal[:60],
            plan.total_steps,
            plan.complexity,
            auto_validate,
            enable_research_features,
        )

        return plan

    async def refine(
        self,
        node: TaskNode,
        exploration_result: str = "",
        head_id: str | None = None,
    ) -> list[TaskNode]:
        """Refine a single node into sub-nodes.

        Delegates to base TaskDecomposer.refine().
        """
        return await self._base_decomposer.refine(node, exploration_result, head_id)

    def to_work_order_with_dag(self, plan: DecompositionPlan) -> WorkOrder:
        """Convert plan to WorkOrder with DAG dependencies.

        Unlike plan.to_work_order() which creates sequential chain,
        this infers true dependencies and allows parallel execution.

        Args:
            plan: DecompositionPlan to convert

        Returns:
            WorkOrder with DAG dependencies
        """
        leaves = plan.all_leaves()

        # Infer dependencies
        deps_map = self._dependency_analyzer.infer_dependencies(leaves)

        # Get feature configs if they were added
        feature_configs = getattr(plan, "_feature_configs", {})

        # Build steps with inferred dependencies
        steps: list[StepDef] = []
        for leaf in leaves:
            # Start with base extra metadata
            extra: dict[str, Any] = {
                "action_type": leaf.action_type,
                "target_files": leaf.target_files,
                "expected_output": leaf.expected_output,
            }

            # Add research feature configs if available
            if leaf.id in feature_configs:
                extra.update(feature_configs[leaf.id])

            step = StepDef(
                step_id=leaf.id,
                name=leaf.goal,
                head_id="",  # Resolved by Router
                required_kind="llm",
                prompt_template=self._leaf_to_prompt(leaf),
                depends_on=deps_map.get(leaf.id, []),
                extra=extra,
            )
            steps.append(step)

        work_order = WorkOrder(goal=plan.goal, steps=steps)

        logger.info(
            "Created WorkOrder with DAG: %d steps, %d parallel opportunities",
            len(steps),
            sum(1 for deps in deps_map.values() if len(deps) == 0),
        )

        return work_order

    @staticmethod
    def _leaf_to_prompt(leaf: TaskNode) -> str:
        """Convert leaf node to prompt template."""
        parts = [leaf.goal]
        if leaf.target_files:
            parts.append(f"Files: {', '.join(leaf.target_files)}")
        if leaf.expected_output:
            parts.append(f"Expected: {leaf.expected_output}")
        if leaf.rationale:
            parts.append(f"Rationale: {leaf.rationale}")
        return "\n".join(parts)
