"""Research feature integration for decomposed task plans.

Automatically enables Tree-of-Thoughts (ToT), Process Reward Models (PRM),
and Reflection based on step complexity and type.
"""

from __future__ import annotations

from typing import Any

from multihead.decomposer import DecompositionPlan, TaskNode


class ResearchFeatureIntegrator:
    """Integrates ToT/PRM/Reflection into decomposed steps.

    Automatically enables research features based on step complexity:
    - ToT: For creative/exploratory steps (multiple valid approaches)
    - PRM: For critical steps requiring quality validation
    - Reflection: For error-prone steps needing self-correction
    """

    # Step types that benefit from Tree-of-Thoughts
    TOT_ACTION_TYPES = {"explore", "investigate", "design", "plan"}

    # Step types that benefit from Process Reward Models
    PRM_ACTION_TYPES = {"edit", "create", "refactor", "implement"}

    # Step types that benefit from Reflection
    REFLECTION_ACTION_TYPES = {"test", "verify", "debug", "fix"}

    @classmethod
    def integrate(cls, plan: DecompositionPlan, enable_auto_features: bool = True) -> dict[str, dict[str, Any]]:
        """Add research feature configurations to steps.

        Args:
            plan: DecompositionPlan to enhance
            enable_auto_features: If True, auto-enable ToT/PRM/Reflection

        Returns:
            Dict mapping step_id -> extra config (for use during WorkOrder conversion)
        """
        if not enable_auto_features:
            return {}

        configs: dict[str, dict[str, Any]] = {}
        for leaf in plan.all_leaves():
            config = cls._get_feature_config(leaf)
            if config:
                configs[leaf.id] = config

        return configs

    @classmethod
    def _get_feature_config(cls, leaf: TaskNode) -> dict[str, Any]:
        """Get research feature config for a single step."""
        config: dict[str, Any] = {}

        # Enable ToT for exploratory steps
        if leaf.action_type in cls.TOT_ACTION_TYPES:
            config["use_tot"] = True
            config["tot_strategy"] = "bfs"
            config["tot_alternatives"] = 3
            config["tot_max_depth"] = 2

        # Enable PRM for implementation steps
        if leaf.action_type in cls.PRM_ACTION_TYPES:
            config["use_prm"] = True
            config["prm_scorer"] = "rubric"
            config["prm_threshold"] = 0.7

        # Enable Reflection for verification steps
        if leaf.action_type in cls.REFLECTION_ACTION_TYPES:
            config["enable_reflection"] = True
            config["max_reflection_attempts"] = 3

        return config
