"""Plan normalizer: validate and enrich WorkOrders before execution."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .models import StepDef, WorkOrder
from .router import Router

if TYPE_CHECKING:
    from .head_manager import HeadManager
    from .observability import MetricsCollector
    from .resilience import ResourceMonitor

logger = logging.getLogger(__name__)


def normalize(
    work_order: WorkOrder,
    head_manager: HeadManager,
    metrics: MetricsCollector | None = None,
    resource_monitor: ResourceMonitor | None = None,
    knowledge_store: Any = None,
    enable_marketplace_fallback: bool = False,
) -> WorkOrder:
    """Validate and enrich a WorkOrder. Returns a new WorkOrder (never mutates original).

    Steps:
    1. Deep-copy the work order
    2. Route steps with required_kind (resolve empty head_id via Router)
    3. Validate all head_ids (primary + fallback) exist in HeadManager
    4. Infer dependencies from input_refs
    5. Auto-assign fallback heads for steps without explicit fallbacks

    Args:
        enable_marketplace_fallback: If True, steps that can't be routed locally
            will attempt BotVibes marketplace discovery (Phase 3).
    """
    wo = work_order.model_copy(deep=True)
    _route_heads(
        wo.steps, head_manager, metrics, resource_monitor, knowledge_store,
        enable_marketplace_fallback=enable_marketplace_fallback,
    )
    _validate_head_ids(wo.steps, head_manager)
    _infer_dependencies(wo.steps)
    _auto_assign_fallbacks(wo.steps, head_manager)
    return wo


def _route_heads(
    steps: list[StepDef],
    head_manager: HeadManager,
    metrics: MetricsCollector | None = None,
    resource_monitor: ResourceMonitor | None = None,
    knowledge_store: Any = None,
    enable_marketplace_fallback: bool = False,
) -> None:
    """Resolve empty head_id via Router for steps with required_kind or task_types (Phase 1).

    Modifies steps in-place (called on the deep copy).
    Raises ValueError if a step needs routing but no head matches.

    Phase 3: When ``enable_marketplace_fallback`` is True and no local head
    matches, attempts BotVibes marketplace discovery to find an external
    provider and registers it as a dynamic head.
    """
    # Phase 1: Support task_types routing (capability-based)
    needs_routing = [
        s for s in steps
        if not s.head_id and (s.task_types or s.required_kind)
    ]
    if not needs_routing:
        return

    router = Router(head_manager, metrics=metrics, resource_monitor=resource_monitor, knowledge_store=knowledge_store)
    unroutable: list[str] = []

    for step in needs_routing:
        head_id = None

        # Phase 1: Prefer task_types routing (capability-based) with multi-candidate ranking
        if step.task_types:
            ranked = router.rank_by_task(
                task_types=step.task_types,
                privacy=step.privacy,
            )
            if not ranked:
                unroutable.append(f"{step.name} (tasks={step.task_types})")
            else:
                head_id = ranked[0][0]
                step.head_id = head_id
                # Store top-3 alternatives as fallbacks (skip primary)
                if not step.fallback and len(ranked) > 1:
                    step.fallback = [hid for hid, _score in ranked[1:3]]
                logger.info(
                    "Routed step '%s' -> %s (tasks=%s, %d candidates)",
                    step.name, head_id, step.task_types, len(ranked),
                )

        # Fallback: Use required_kind routing (backward compatible) with multi-candidate ranking
        elif step.required_kind:
            ranked = router.rank(step.required_kind)
            if not ranked:
                unroutable.append(f"{step.name} (kind={step.required_kind})")
            else:
                head_id = ranked[0][0]
                step.head_id = head_id
                # Store top-3 alternatives as fallbacks (skip primary)
                if not step.fallback and len(ranked) > 1:
                    step.fallback = [hid for hid, _score in ranked[1:3]]
                logger.info(
                    "Routed step '%s' -> %s (kind=%s, %d candidates)",
                    step.name, head_id, step.required_kind, len(ranked),
                )

    # Phase 3: Marketplace fallback for unroutable steps
    if unroutable and enable_marketplace_fallback:
        import asyncio

        resolved: list[str] = []
        for desc in unroutable:
            # Parse step from description
            step_name = desc.split(" (")[0]
            step = next((s for s in needs_routing if s.name == step_name), None)
            if not step or not step.task_types:
                continue

            # Use the first task_type as marketplace capability query
            capability = step.task_types[0]
            try:
                # Run async discovery in sync context
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Already in async context — can't run_until_complete
                    # Store for deferred resolution
                    logger.info(
                        "Step '%s' unroutable locally; marketplace fallback deferred (async context)",
                        step.name,
                    )
                    continue
                marketplace_head = loop.run_until_complete(
                    router.route_with_marketplace_fallback(
                        capability,
                        task_types=step.task_types,
                        privacy=step.privacy,
                    )
                )
            except RuntimeError:
                # No event loop — create one
                marketplace_head = asyncio.run(
                    router.route_with_marketplace_fallback(
                        capability,
                        task_types=step.task_types,
                        privacy=step.privacy,
                    )
                )

            if marketplace_head:
                step.head_id = marketplace_head
                resolved.append(desc)
                logger.info(
                    "Routed step '%s' -> %s (marketplace fallback)",
                    step.name, marketplace_head,
                )

        # Remove resolved steps from unroutable
        unroutable = [u for u in unroutable if u not in resolved]

    if unroutable:
        raise ValueError(f"No head available for: {', '.join(unroutable)}")


def _validate_head_ids(steps: list[StepDef], head_manager: HeadManager) -> None:
    """Check that every step.head_id and step.fallback entry exists.

    Raises ValueError listing all unknown head_ids.
    """
    unknown: set[str] = set()
    for step in steps:
        if head_manager.get_manifest(step.head_id) is None:
            unknown.add(step.head_id)
        for fb in step.fallback:
            if head_manager.get_manifest(fb) is None:
                unknown.add(fb)
    if unknown:
        raise ValueError(f"Unknown head_id(s): {sorted(unknown)}")


def _infer_dependencies(steps: list[StepDef]) -> None:
    """If step B's input_refs mention step A's name or step_id, add dependency.

    Modifies steps in-place (called on the deep copy).
    """
    name_to_id: dict[str, str] = {}
    valid_ids: set[str] = set()
    for s in steps:
        name_to_id[s.name] = s.step_id
        valid_ids.add(s.step_id)

    for step in steps:
        for ref in step.input_refs:
            dep_id = name_to_id.get(ref, ref)
            if dep_id in valid_ids and dep_id != step.step_id and dep_id not in step.depends_on:
                step.depends_on.append(dep_id)
                logger.debug(
                    "Inferred dependency: %s -> %s (via input_ref '%s')",
                    step.name, dep_id, ref,
                )


def _auto_assign_fallbacks(steps: list[StepDef], head_manager: HeadManager) -> None:
    """For steps without explicit fallback, add other heads of same kind.

    Rules:
    - Only add heads whose kind matches the primary head's kind (llm/vlm/etc.)
    - Don't add the primary head_id itself
    - Sort: non-GPU first (lighter fallbacks), then alphabetical

    Modifies steps in-place (called on the deep copy).
    """
    heads_by_kind: dict[str, list[str]] = {}
    for head_id, info in head_manager.get_states().items():
        kind = info["kind"]
        heads_by_kind.setdefault(kind, []).append(head_id)

    for step in steps:
        if step.fallback:
            continue

        manifest = head_manager.get_manifest(step.head_id)
        if manifest is None:
            continue

        candidates = heads_by_kind.get(manifest.kind, [])
        fallbacks = [h for h in candidates if h != step.head_id]

        def _sort_key(hid: str) -> tuple[int, str]:
            m = head_manager.get_manifest(hid)
            return (1 if m and m.gpu_required else 0, hid)

        fallbacks.sort(key=_sort_key)
        step.fallback = fallbacks
