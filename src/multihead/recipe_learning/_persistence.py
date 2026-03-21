"""Recipe persistence - save, version tracking, and sharing."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from multihead.acp_bridge import ACPBridge
    from multihead.registry.solver_registry import SolverRegistry

logger = logging.getLogger(__name__)


def save_recipe(
    recipe: dict[str, Any],
    name: str,
    recipes_dir: Path,
    *,
    metadata: dict[str, Any] | None = None,
    task_type: str | None = None,
    benchmark_results: dict[str, Any] | None = None,
    registry: SolverRegistry | None = None,
) -> Path:
    """Save recipe to recipes directory and optionally track in registry.

    Args:
        recipe: Recipe to save
        name: Recipe filename (without .yaml extension)
        recipes_dir: Directory to save recipe into
        metadata: Optional metadata to include in file header
        task_type: Task type for registry tracking
        benchmark_results: Benchmark results for registry tracking
        registry: Optional SolverRegistry for recipe version tracking

    Returns:
        Path to saved recipe
    """
    filepath = recipes_dir / f"{name}.yaml"

    # Add metadata as comments
    content_parts = []
    if metadata:
        content_parts.append("# Recipe Metadata")
        for key, value in metadata.items():
            content_parts.append(f"# {key}: {value}")
        content_parts.append(
            f"# learned_at: {datetime.now(timezone.utc).isoformat()}"
        )
        content_parts.append("")

    # Add recipe YAML
    recipe_yaml = yaml.dump(recipe, default_flow_style=False, sort_keys=False)
    content_parts.append(recipe_yaml)

    # Write to file
    filepath.write_text("\n".join(content_parts), encoding="utf-8")
    logger.info("Saved recipe to %s", filepath)

    # Track in registry if available
    if registry and task_type:
        success_rate = None
        perf_score = None
        avg_latency = None
        avg_cost = None
        test_count = 0

        if benchmark_results:
            tc = max(benchmark_results.get("test_cases_count", 1), 1)
            success_rate = benchmark_results.get("success_count", 0) / tc
            perf_score = success_rate
            avg_latency = benchmark_results.get("avg_latency_ms")
            avg_cost = benchmark_results.get("avg_cost_usd")
            test_count = benchmark_results.get("test_cases_count", 0)

        registry.add_recipe_version(
            recipe_id=name,
            task_type=task_type,
            goal=recipe.get("goal", name),
            source=metadata.get("source", "unknown") if metadata else "unknown",
            recipe_yaml=recipe_yaml,
            performance_score=perf_score,
            success_rate=success_rate,
            avg_latency_ms=avg_latency,
            avg_cost_usd=avg_cost,
            test_cases_count=test_count,
        )

    return filepath


def record_evaluation_votes(
    recipe_id: str,
    version: int,
    votes: dict[str, str],
    registry: SolverRegistry,
    confidences: dict[str, float] | None = None,
    rationales: dict[str, str] | None = None,
) -> None:
    """Record consensus evaluation votes in registry.

    Args:
        recipe_id: Recipe identifier
        version: Recipe version number
        votes: {head_id: action} mapping
        registry: SolverRegistry instance
        confidences: {head_id: confidence} mapping
        rationales: {head_id: rationale} mapping
    """
    confidences = confidences or {}
    rationales = rationales or {}

    for head_id, vote in votes.items():
        registry.add_recipe_evaluation(
            recipe_id=recipe_id,
            version=version,
            head_id=head_id,
            vote=vote,
            confidence=confidences.get(head_id, 0.5),
            rationale=rationales.get(head_id, ""),
        )


async def share_success(
    recipe: dict[str, Any],
    benchmark_results: dict[str, Any],
    acp: ACPBridge,
) -> bool:
    """Share successful recipe back to BotVibes knowledge network.

    Args:
        recipe: Recipe that performed well
        benchmark_results: Benchmark results showing success
        acp: ACPBridge for sharing

    Returns:
        True if shared successfully
    """
    # Build success report
    report = {
        "recipe": recipe,
        "benchmark_results": benchmark_results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": "multihead",
    }

    # Share via ACP task (capability: recipe_feedback)
    try:
        task_id = await acp.create_task(
            capability="recipe_feedback",
            payload_ref=yaml.dump(report),
            priority="batch",  # Low priority feedback
        )

        logger.info("Shared recipe success to knowledge network: %s", task_id)
        return True

    except Exception as e:
        logger.warning("Failed to share recipe success: %s", e)
        return False
