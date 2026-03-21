"""Base classes for solver discovery."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class SolverCandidate:
    """Represents a discovered solver candidate.

    This is the intermediate format between discovery sources
    (HuggingFace, Ollama, BotVibes) and the solver registry.
    """

    # Identity
    solver_id: str
    name: str
    source: str  # "huggingface", "ollama", "botvibes"

    # Capabilities
    solver_type: str  # "llm", "vlm", "object_detection", etc.
    task_types: list[str] = field(default_factory=list)
    modalities: list[str] = field(default_factory=list)  # "text", "image", "audio"

    # Performance (if available from source)
    benchmark_scores: dict[str, float] = field(default_factory=dict)  # e.g., {"mmlu": 0.72}
    estimated_latency_ms: int | None = None
    estimated_cost: float | None = None  # Cost per call in USD

    # Metadata
    model_id: str | None = None  # HuggingFace model ID or Ollama tag
    version: str | None = None
    license: str | None = None
    description: str | None = None
    url: str | None = None
    tags: list[str] = field(default_factory=list)

    # Discovery metadata
    discovered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    discovery_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and normalize fields."""
        if not self.solver_id:
            raise ValueError("solver_id is required")
        if not self.name:
            self.name = self.solver_id
        if self.solver_type not in {
            "llm", "vlm", "object_detection", "segmentation",
            "embedding", "deterministic", "validator", "tool", "external_service"
        }:
            # Still allow it, but warn
            pass

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "solver_id": self.solver_id,
            "name": self.name,
            "source": self.source,
            "solver_type": self.solver_type,
            "task_types": self.task_types,
            "modalities": self.modalities,
            "benchmark_scores": self.benchmark_scores,
            "estimated_latency_ms": self.estimated_latency_ms,
            "estimated_cost": self.estimated_cost,
            "model_id": self.model_id,
            "version": self.version,
            "license": self.license,
            "description": self.description,
            "url": self.url,
            "tags": self.tags,
            "discovered_at": self.discovered_at.isoformat(),
            "discovery_metadata": self.discovery_metadata,
        }


class DiscoveryAgent(ABC):
    """Base class for solver discovery agents.

    Each agent knows how to query a specific source (HuggingFace, Ollama, etc.)
    and convert results to SolverCandidate format.
    """

    def __init__(self, source_name: str):
        """Initialize discovery agent.

        Args:
            source_name: Name of the discovery source (e.g., "huggingface")
        """
        self.source_name = source_name

    @abstractmethod
    async def discover_new_solvers(
        self,
        *,
        solver_types: list[str] | None = None,
        min_downloads: int | None = None,
        updated_since: datetime | None = None,
        limit: int = 50,
    ) -> list[SolverCandidate]:
        """Discover new solvers from the source.

        Args:
            solver_types: Filter by solver types (e.g., ["llm", "vlm"])
            min_downloads: Minimum download count (HuggingFace)
            updated_since: Only return models updated after this date
            limit: Maximum number of results

        Returns:
            List of discovered solver candidates
        """
        pass

    @abstractmethod
    async def get_solver_details(self, solver_id: str) -> SolverCandidate | None:
        """Get detailed information about a specific solver.

        Args:
            solver_id: Solver identifier from the source

        Returns:
            Detailed solver candidate or None if not found
        """
        pass

    async def check_solver_updates(
        self,
        current_solvers: list[dict[str, Any]]
    ) -> list[SolverCandidate]:
        """Check if existing solvers have updates.

        Args:
            current_solvers: List of currently registered solvers

        Returns:
            List of updated solver candidates
        """
        # Default implementation: check each solver for updates
        updated = []
        for solver in current_solvers:
            if solver.get("source") != self.source_name:
                continue

            solver_id = solver.get("solver_id", "")
            if not solver_id:
                continue

            details = await self.get_solver_details(solver_id)
            if details and self._has_update(solver, details):
                updated.append(details)

        return updated

    def _has_update(
        self,
        current: dict[str, Any],
        candidate: SolverCandidate
    ) -> bool:
        """Check if candidate represents an update to current solver.

        Args:
            current: Current solver dict
            candidate: Discovered candidate

        Returns:
            True if candidate is newer/different
        """
        # Compare versions if available
        current_version = current.get("version")
        if current_version and candidate.version:
            return candidate.version != current_version

        # Otherwise assume it's updated
        return True
