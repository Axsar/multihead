"""Session harvester — cross-context knowledge aggregator with history.

Scans all Claude Code project folders (~/.claude/projects/), reads their
MEMORY.md and CLAUDE.md files, extracts structured claims, and deposits
them into knowledge.db. Maintains a manifest to avoid re-harvesting
unchanged files.

**History tracking**: When a file changes between harvests, the harvester
stores a snapshot of the previous version and emits a ``session_evolution``
event capturing what was added, removed, and changed. This preserves the
narrative arc of each session — not just what it knows now, but how its
understanding evolved over time and why decisions were made.

This bridges the knowledge silos between different Claude Code sessions,
giving MultiHead cross-context awareness across all projects.
"""

from .evolution import (
    emit_evolution_event,
    load_snapshot,
    meaningful_lines,
    save_snapshot,
    track_evolution,
)
from .extraction import (
    classify_claim_type,
    deposit_claims,
    extract_claims,
)
from .harvester import SessionHarvester
from .models import (
    PROVENANCE_ID,
    EvolutionRecord,
    HarvestResult,
    ProjectInfo,
)

__all__ = [
    # Core class
    "SessionHarvester",
    # Models
    "EvolutionRecord",
    "HarvestResult",
    "ProjectInfo",
    "PROVENANCE_ID",
    # Extraction
    "classify_claim_type",
    "deposit_claims",
    "extract_claims",
    # Evolution
    "emit_evolution_event",
    "load_snapshot",
    "meaningful_lines",
    "save_snapshot",
    "track_evolution",
]
