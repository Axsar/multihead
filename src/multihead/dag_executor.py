"""DAG-aware execution engine for work orders with parallel steps."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from multihead.models import RunState, StepDef, StepStatus, WorkOrder

logger = logging.getLogger(__name__)


class NodeStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepNode:
    """A node in the execution DAG."""
    step: StepDef
    dependencies: list[str] = field(default_factory=list)
    dependents: list[str] = field(default_factory=list)
    status: NodeStatus = NodeStatus.PENDING


class DAGExecutor:
    """Execute work order steps as a DAG with parallel layers."""

    def __init__(self, orchestrator: Any, max_parallel_cpu: int = 8) -> None:
        self.orchestrator = orchestrator
        self._cpu_semaphore = asyncio.Semaphore(max_parallel_cpu)

    def build_graph(self, steps: list[StepDef]) -> dict[str, StepNode]:
        """Build a DAG from step definitions using input_refs and depends_on."""
        nodes: dict[str, StepNode] = {}
        step_output_map: dict[str, str] = {}  # output artifact -> step_id

        for step in steps:
            deps = list(getattr(step, "depends_on", []) or [])
            logger.debug("Building graph for step %s (name=%s), initial depends_on=%s, input_refs=%s",
                        step.step_id, step.name, deps, step.input_refs)

            # Infer dependencies from input_refs
            for ref in step.input_refs:
                if ref in step_output_map:
                    dep_id = step_output_map[ref]
                    # Skip self-dependencies to avoid cycles
                    if dep_id != step.step_id and dep_id not in deps:
                        logger.debug("  Adding dependency %s (from input_ref %s)", dep_id, ref)
                        deps.append(dep_id)
                    elif dep_id == step.step_id:
                        logger.debug("  Skipping self-dependency for step %s", step.step_id)
                else:
                    logger.debug("  input_ref %s not found in step_output_map (available: %s)",
                                ref, list(step_output_map.keys()))

            nodes[step.step_id] = StepNode(step=step, dependencies=deps)
            step_output_map[step.name] = step.step_id
            logger.debug("  Final dependencies for %s: %s", step.step_id, deps)

        # Build dependents (reverse edges)
        for node_id, node in nodes.items():
            for dep_id in node.dependencies:
                if dep_id in nodes:
                    nodes[dep_id].dependents.append(node_id)

        return nodes

    def topological_sort(self, graph: dict[str, StepNode]) -> list[list[str]]:
        """Sort into execution layers. Steps in the same layer can run in parallel."""
        in_degree: dict[str, int] = {nid: len(n.dependencies) for nid, n in graph.items()}
        layers: list[list[str]] = []

        remaining = set(graph.keys())
        while remaining:
            # Find nodes with no pending dependencies
            ready = [nid for nid in remaining if in_degree[nid] == 0]
            if not ready:
                cycle_nodes = sorted(remaining)
                raise ValueError(
                    f"Cycle detected in DAG among steps: {cycle_nodes}. "
                    f"Remove circular depends_on / input_refs to fix."
                )

            layers.append(sorted(ready))

            for nid in ready:
                remaining.discard(nid)
                for dependent in graph[nid].dependents:
                    if dependent in in_degree:
                        in_degree[dependent] -= 1

        return layers

    async def execute_dag(
        self,
        run_id: str,
        work_order: WorkOrder,
        state: RunState,
    ) -> RunState:
        """Execute steps in DAG order, parallelizing where possible."""
        graph = self.build_graph(work_order.steps)
        layers = self.topological_sort(graph)

        logger.info("DAG execution plan: %d layers for %d steps", len(layers), len(work_order.steps))

        for layer_idx, layer in enumerate(layers):
            logger.info("Executing layer %d: %s", layer_idx, layer)

            # Separate GPU and CPU steps
            gpu_steps = []
            cpu_steps = []
            for step_id in layer:
                node = graph[step_id]
                head_manager = self.orchestrator.heads
                manifest = head_manager.get_manifest(node.step.head_id)
                # Default to GPU (safe) if manifest is unknown
                if manifest is None or manifest.gpu_required:
                    gpu_steps.append(step_id)
                else:
                    cpu_steps.append(step_id)

            # CPU steps can run in parallel (bounded by semaphore)
            if cpu_steps:
                async def _bounded_execute(sid: str) -> None:
                    async with self._cpu_semaphore:
                        await self._execute_step(run_id, graph[sid].step, state)

                tasks = [_bounded_execute(sid) for sid in cpu_steps]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for sid, result in zip(cpu_steps, results):
                    if isinstance(result, Exception):
                        # Try fallback heads (under semaphore to respect concurrency limit)
                        step = graph[sid].step
                        recovered = False
                        for fb_head_id in step.fallback:
                            try:
                                fb_step = step.model_copy(update={"head_id": fb_head_id})
                                async with self._cpu_semaphore:
                                    await self._execute_step(run_id, fb_step, state)
                                state.step_results[step.step_id] = state.step_results.get(step.step_id, state.step_results[step.step_id])
                                recovered = True
                                logger.info("CPU step %s recovered via fallback %s", sid, fb_head_id)
                                break
                            except Exception:
                                continue
                        graph[sid].status = NodeStatus.DONE if recovered else NodeStatus.FAILED
                        if not recovered:
                            logger.error("Step %s failed (no fallbacks left): %s", sid, result)
                    else:
                        graph[sid].status = NodeStatus.DONE

            # GPU steps must serialize (GPU mutex)
            for sid in gpu_steps:
                try:
                    await self._execute_step(run_id, graph[sid].step, state)
                    graph[sid].status = NodeStatus.DONE
                except Exception as e:
                    # Try fallback heads
                    step = graph[sid].step
                    recovered = False
                    for fb_head_id in step.fallback:
                        try:
                            fb_step = step.model_copy(update={"head_id": fb_head_id})
                            await self._execute_step(run_id, fb_step, state)
                            state.step_results[step.step_id] = state.step_results.get(step.step_id, state.step_results[step.step_id])
                            recovered = True
                            logger.info("GPU step %s recovered via fallback %s", sid, fb_head_id)
                            break
                        except Exception:
                            continue
                    graph[sid].status = NodeStatus.DONE if recovered else NodeStatus.FAILED
                    if not recovered:
                        logger.error("GPU step %s failed (no fallbacks left): %s", sid, e)

        return state

    async def _execute_step(self, run_id: str, step: StepDef, state: RunState) -> None:
        """Execute a single step by delegating to orchestrator (events + artifacts)."""
        # Track step index
        if state.work_order:
            for i, s in enumerate(state.work_order.steps):
                if s.step_id == step.step_id:
                    state.current_step_index = i
                    break

        work_order = state.work_order or WorkOrder(goal="", steps=[])

        # Ensure run artifacts directory exists (same path as linear executor)
        run_artifacts_dir = self.orchestrator.runs_dir / run_id / "artifacts"
        run_artifacts_dir.mkdir(parents=True, exist_ok=True)

        result = await self.orchestrator._execute_step_with_reflection(
            run_id=run_id,
            step=step,
            work_order=work_order,
            state=state,
            run_artifacts_dir=run_artifacts_dir,
        )

        # Re-raise on failure so DAG's fallback logic catches it
        if result.status == StepStatus.FAILED:
            raise RuntimeError(result.error or f"Step {step.step_id} failed")
