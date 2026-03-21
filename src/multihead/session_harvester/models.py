"""Data models for the session harvester."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Provenance identifier for all harvester claims
PROVENANCE_ID = "session-harvester-v1"


@dataclass
class ProjectInfo:
    """Discovered Claude Code project."""

    name: str                          # folder name, e.g. "-mnt-d-DevD-Multihead"
    path: Path                         # full path to project folder
    decoded_path: str = ""             # human-readable decoded path
    has_memory: bool = False           # MEMORY.md exists
    has_claude_md: bool = False        # CLAUDE.md exists
    memory_files: list[Path] = field(default_factory=list)  # all discoverable files


@dataclass
class EvolutionRecord:
    """Tracks what changed between two harvests of the same file."""

    project_name: str
    file_path: str
    scope_id: str
    added_lines: list[str] = field(default_factory=list)
    removed_lines: list[str] = field(default_factory=list)
    timestamp: str = ""


@dataclass
class HarvestResult:
    """Summary of a harvest run."""

    projects_scanned: int = 0
    projects_harvested: int = 0
    projects_skipped: int = 0          # unchanged since last harvest
    claims_deposited: int = 0
    evolution_events: int = 0          # session_evolution events emitted
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
