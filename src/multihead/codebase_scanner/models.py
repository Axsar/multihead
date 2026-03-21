"""Data models and constants for codebase scanning.

Defines DiscoveredCapability, ScanResult dataclasses and module-level
constants used across the scanner package.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Skip these directories during scanning
SKIP_DIRS = frozenset({
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".tox", ".eggs", "dist", "build", ".mypy_cache",
    ".cache", "wandb", "runs", "outputs",
    "site-packages", "egg-info",
})

# Max Python files to analyze per project
MAX_PY_FILES = 300

# Max lines per file for AST analysis
MAX_LINES = 300

# Minimum function body lines to be considered a capability
MIN_FUNCTION_LINES = 8

# Priority directories (scanned first)
PRIORITY_DIRS = {"production", "src", "core", "pipeline", "lib"}

# Commodity capability patterns (everyone has these — skip)
COMMODITY_PATTERNS = re.compile(
    r"(test_|conftest|setup|__init__|__main__|migration|fixture)",
    re.IGNORECASE,
)


@dataclass
class DiscoveredCapability:
    """A capability discovered from codebase analysis."""

    # Identity
    name: str
    capability_id: str  # com.{project}.{category}.{name}
    project: str

    # Source
    source_file: str
    source_type: str  # "code", "model", "pipeline", "memory", "claude_md"

    # Description
    description: str = ""
    category: str = ""  # "detection", "segmentation", "layout", "generation", etc.

    # Quality signals
    has_tests: bool = False
    has_cli: bool = False
    is_production: bool = False  # In production/ dir
    eval_metrics: dict[str, Any] = field(default_factory=dict)

    # Technical details
    requires_gpu: bool = False
    model_path: str = ""
    model_size_mb: float = 0.0
    functions: list[str] = field(default_factory=list)
    input_types: list[str] = field(default_factory=list)
    output_types: list[str] = field(default_factory=list)

    # Maturity
    git_commits: int = 0
    confidence: float = 0.5

    def to_claim(self, scope_id: str = "capabilities") -> dict[str, Any]:
        """Convert to knowledge.db claim format."""
        parts = [f"Capability: {self.name}"]
        if self.description:
            parts.append(self.description[:200])
        parts.append(f"Project: {self.project}")
        parts.append(f"Category: {self.category}")
        if self.eval_metrics:
            parts.append("Metrics: " + ", ".join(
                f"{k}={v}" for k, v in self.eval_metrics.items()
            ))
        if self.requires_gpu:
            parts.append("GPU required")
        if self.is_production:
            parts.append("Production-ready")
        if self.has_tests:
            parts.append("Has tests")

        return {
            "claim_key": f"capability.{self.capability_id}",
            "statement": " | ".join(parts),
            "claim_type": "fact",
            "confidence": self.confidence,
            "scope_id": scope_id,
            "produced_by": "codebase-scanner-v1",
        }

    def to_listing(self) -> dict[str, Any]:
        """Convert to marketplace listing format."""
        price = 0.25
        if self.requires_gpu:
            price = 1.00
        elif self.category == "model":
            price = 0.75
        elif self.category in ("pipeline", "detection", "segmentation"):
            price = 0.50

        return {
            "capability_id": f"com.multihead.{self.capability_id}",
            "name": self.name,
            "description": self.description or f"{self.category}: {self.name}",
            "pricing_model": "per_call",
            "unit_price": price,
            "quality_score": self.confidence,
            "metadata": {
                "category": self.category,
                "project": self.project,
                "requires_gpu": self.requires_gpu,
                "eval_metrics": self.eval_metrics,
                "is_production": self.is_production,
                "has_tests": self.has_tests,
            },
        }


@dataclass
class ScanResult:
    """Result of scanning a single project."""

    project_path: str
    project_name: str
    capabilities: list[DiscoveredCapability] = field(default_factory=list)
    model_checkpoints: list[dict[str, Any]] = field(default_factory=list)
    scan_duration_s: float = 0.0
    files_scanned: int = 0
    errors: list[str] = field(default_factory=list)
