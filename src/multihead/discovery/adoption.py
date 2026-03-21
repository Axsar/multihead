"""Adoption bridge: register adopted solvers as heads in HeadManager.

Bridges the gap between SolverRegistry (discovery system) and
HeadManager (execution system). When a solver is adopted via
adoption rules, this module creates the corresponding HeadManifest
and registers it for use in the routing/execution pipeline.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from multihead.models import AdapterKind, Capability, HeadManifest

if TYPE_CHECKING:
    from multihead.head_manager import HeadManager
    from multihead.registry.solver_registry import SolverRegistry

logger = logging.getLogger(__name__)

# Map discovery sources to adapter kinds
SOURCE_TO_ADAPTER: dict[str, AdapterKind] = {
    "huggingface": AdapterKind.TRANSFORMERS,
    "ollama": AdapterKind.OLLAMA,
    "botvibes": AdapterKind.BOTVIBES,
}

# Map solver types to head kinds
SOLVER_TYPE_TO_KIND: dict[str, str] = {
    "llm": "llm",
    "vlm": "vlm",
    "object_detection": "vlm",
    "segmentation": "vlm",
    "embedding": "llm",
    "external_service": "llm",
}


def solver_to_manifest(solver: dict[str, Any]) -> HeadManifest | None:
    """Convert a registry solver dict to a HeadManifest.

    Args:
        solver: Solver dict from SolverRegistry.get_solver()

    Returns:
        HeadManifest if convertible, None otherwise
    """
    source = solver.get("source", "")
    adapter = SOURCE_TO_ADAPTER.get(source)

    if not adapter:
        logger.debug("No adapter mapping for source %s", source)
        return None

    model_id = solver.get("model_id", "")
    if not model_id:
        logger.warning("Solver %s has no model_id, skipping", solver["solver_id"])
        return None

    solver_type = solver.get("solver_type", "llm")
    kind = SOLVER_TYPE_TO_KIND.get(solver_type, "llm")

    # Build capability
    capability = Capability(
        solver_type=solver_type,
        input_modalities=solver.get("modalities", ["text"]),
        output_modalities=["text"],
        task_types=solver.get("task_types", []),
        latency_p50_ms=solver.get("estimated_latency_ms"),
        cost_per_call=solver.get("estimated_cost"),
    )

    # Determine GPU requirements
    gpu_required = source in ("huggingface",)  # Ollama manages its own GPU
    is_local = source in ("huggingface", "ollama")

    # Build extra config
    extra: dict[str, Any] = {
        "discovery_source": source,
        "solver_id": solver["solver_id"],
        "adopted_from_registry": True,
    }

    # Source-specific extras
    if source == "ollama":
        gpu_required = False  # Ollama handles this
        extra["ollama_model"] = model_id
    elif source == "botvibes":
        gpu_required = False
        # BotVibes needs endpoint and token
        extra["target_capability"] = solver.get("task_types", [""])[0] if solver.get("task_types") else ""

    # Build endpoint
    endpoint = ""
    if source == "botvibes":
        metadata = solver.get("discovery_metadata", {})
        endpoint = metadata.get("acp_url", "")

    try:
        manifest = HeadManifest(
            head_id=solver["solver_id"],
            name=solver.get("name", solver["solver_id"]),
            adapter=adapter,
            model=model_id,
            kind=kind,
            gpu_required=gpu_required,
            is_local=is_local,
            endpoint=endpoint,
            extra=extra,
            capabilities=capability,
        )
        return manifest
    except Exception as e:
        logger.error("Failed to create manifest for %s: %s", solver["solver_id"], e)
        return None


def register_adopted_solvers(
    registry: SolverRegistry,
    head_manager: HeadManager,
    *,
    dry_run: bool = False,
) -> list[str]:
    """Register all adopted solvers from registry into HeadManager.

    Scans the registry for solvers with adoption_status="adopted",
    converts them to HeadManifest, and registers them with HeadManager.

    Args:
        registry: SolverRegistry with adopted solvers
        head_manager: HeadManager to register manifests in
        dry_run: If True, only log what would be registered

    Returns:
        List of registered solver IDs
    """
    adopted = registry.list_solvers(adoption_status="adopted")
    registered = []

    for solver in adopted:
        solver_id = solver["solver_id"]

        # Skip if already registered
        if head_manager.get_manifest(solver_id) is not None:
            logger.debug("Solver %s already registered in HeadManager", solver_id)
            continue

        manifest = solver_to_manifest(solver)
        if not manifest:
            logger.warning("Could not create manifest for adopted solver %s", solver_id)
            continue

        if dry_run:
            logger.info("[DRY RUN] Would register %s (%s via %s)", solver_id, manifest.kind, manifest.adapter)
            registered.append(solver_id)
            continue

        # Register with HeadManager
        try:
            head_manager._manifests[solver_id] = manifest
            head_manager._states[solver_id] = {
                "head_id": solver_id,
                "kind": manifest.kind,
                "status": "idle",
                "gpu_required": manifest.gpu_required,
                "loaded": False,
                "adopted_from_registry": True,
            }
            registered.append(solver_id)
            logger.info(
                "Registered adopted solver %s (%s via %s)",
                solver_id, manifest.kind, manifest.adapter,
            )
        except Exception as e:
            logger.error("Failed to register %s: %s", solver_id, e)

    if registered:
        logger.info("Registered %d adopted solvers into HeadManager", len(registered))

    return registered


def sync_registry_to_heads(
    registry: SolverRegistry,
    head_manager: HeadManager,
) -> dict[str, Any]:
    """Full sync: register adopted, remove rejected solvers from HeadManager.

    Args:
        registry: SolverRegistry
        head_manager: HeadManager

    Returns:
        Summary dict with counts
    """
    result = {
        "registered": [],
        "removed": [],
        "skipped": [],
    }

    # Register adopted solvers
    result["registered"] = register_adopted_solvers(registry, head_manager)

    # Remove rejected solvers that were previously registered
    rejected = registry.list_solvers(adoption_status="rejected")
    for solver in rejected:
        solver_id = solver["solver_id"]
        state = head_manager._states.get(solver_id, {})
        if state.get("adopted_from_registry"):
            # Only remove registry-adopted heads, not manually configured ones
            if solver_id in head_manager._manifests:
                del head_manager._manifests[solver_id]
            if solver_id in head_manager._states:
                del head_manager._states[solver_id]
            result["removed"].append(solver_id)
            logger.info("Removed rejected solver %s from HeadManager", solver_id)

    return result
