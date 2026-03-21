"""Plan flattening and DAG helpers for the autonomous executor."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _flatten_leaves(plan: dict) -> list[dict]:
    """Extract leaf steps from a serialized plan dict."""
    leaves = []

    def _walk(node: dict):
        children = node.get("children", [])
        if not children:
            leaves.append(node)
        else:
            for child in children:
                _walk(child)

    for phase in plan.get("phases", []):
        _walk(phase)

    return leaves


def _infer_layer_dependencies(leaves: list[dict]) -> dict[str, list[str]]:
    """Infer dependencies between leaf steps based on action type ordering.

    Uses the same heuristic as StepDependencyAnalyzer but on serialized dicts.
    """
    # Action ordering: later types depend on earlier types
    ACTION_ORDER = {
        "test": {"edit", "create", "refactor", "implement", "fix"},
        "verify": {"edit", "create", "test", "implement", "review"},
        "review": {"edit", "create", "refactor", "implement"},
    }

    write_actions = {"edit", "create", "implement", "refactor", "fix"}

    # Build index: step position by ID (for ordering — only depend on EARLIER steps)
    id_to_idx = {leaf["id"]: i for i, leaf in enumerate(leaves)}

    deps: dict[str, list[str]] = {leaf["id"]: [] for leaf in leaves}

    for leaf in leaves:
        sid = leaf["id"]
        action = leaf.get("action_type", "")
        s_idx = id_to_idx[sid]

        # Action ordering: only depend on EARLIER steps with matching action types
        must_follow = ACTION_ORDER.get(action, set())
        for other in leaves:
            oid = other["id"]
            if oid == sid:
                continue
            if id_to_idx[oid] >= s_idx:
                continue  # Only depend on steps that appear earlier
            if other.get("action_type", "") in must_follow:
                if oid not in deps[sid]:
                    deps[sid].append(oid)

        # File dependencies: writer depends on earlier readers of same file
        if action in write_actions:
            for f in leaf.get("target_files", []):
                for other in leaves:
                    oid = other["id"]
                    if oid == sid or id_to_idx[oid] >= s_idx:
                        continue
                    if f in other.get("target_files", []):
                        if oid not in deps[sid]:
                            deps[sid].append(oid)

    return deps


def _topological_layers(
    leaves: list[dict], dep_map: dict[str, list[str]],
) -> list[list[str]]:
    """Topological sort into parallel layers (Kahn's algorithm)."""
    all_ids = {leaf["id"] for leaf in leaves}

    # Filter deps to only include IDs in the current set
    in_degree: dict[str, int] = {sid: 0 for sid in all_ids}
    adj: dict[str, list[str]] = {sid: [] for sid in all_ids}

    for sid, deps in dep_map.items():
        if sid not in all_ids:
            continue
        valid_deps = [d for d in deps if d in all_ids]
        in_degree[sid] = len(valid_deps)
        for d in valid_deps:
            adj[d].append(sid)

    layers: list[list[str]] = []
    remaining = set(all_ids)

    while remaining:
        # Find nodes with in_degree 0
        layer = [sid for sid in remaining if in_degree.get(sid, 0) == 0]

        if not layer:
            # Cycle detected — dump remaining into one layer
            logger.warning(
                "Cycle detected in DAG, forcing remaining %d steps into single layer",
                len(remaining),
            )
            layers.append(sorted(remaining))
            break

        layers.append(sorted(layer))

        for sid in layer:
            remaining.discard(sid)
            for dependent in adj.get(sid, []):
                in_degree[dependent] -= 1

    return layers


def _summarize_plan(plan: dict) -> str:
    """Create a brief text summary of the plan."""
    lines = [f"Goal: {plan.get('goal', 'unknown')}"]
    lines.append(f"Complexity: {plan.get('complexity', 'unknown')}")

    for phase in plan.get("phases", []):
        lines.append(f"- {phase.get('id', '?')}: {phase.get('goal', '')}")
        for child in phase.get("children", []):
            lines.append(f"  - {child.get('id', '?')}: {child.get('goal', '')}")

    return "\n".join(lines)
