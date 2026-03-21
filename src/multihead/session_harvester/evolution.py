"""Evolution tracking — snapshot diffing and event emission."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import PROVENANCE_ID, EvolutionRecord, ProjectInfo

logger = logging.getLogger(__name__)


def save_snapshot(
    snapshots_dir: Path, project_name: str, rel_path: str, content: str,
) -> None:
    """Save a snapshot of a file for future diffing."""
    safe_name = project_name.replace("/", "_")
    safe_rel = rel_path.replace("/", "_").replace("\\", "_")
    snapshot_dir = snapshots_dir / safe_name
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (snapshot_dir / safe_rel).write_text(content, encoding="utf-8")


def load_snapshot(
    snapshots_dir: Path, project_name: str, rel_path: str,
) -> str | None:
    """Load the previous snapshot of a file."""
    safe_name = project_name.replace("/", "_")
    safe_rel = rel_path.replace("/", "_").replace("\\", "_")
    snapshot_file = snapshots_dir / safe_name / safe_rel
    if snapshot_file.exists():
        try:
            return snapshot_file.read_text(encoding="utf-8")
        except OSError:
            return None
    return None


def track_evolution(
    project: ProjectInfo,
    file_path: Path,
    rel_path: str,
    new_content: str,
    scope_id: str,
    snapshots_dir: Path,
    knowledge_store: Any,
) -> EvolutionRecord | None:
    """Diff against previous snapshot and emit a session_evolution event."""
    old_content = load_snapshot(snapshots_dir, project.name, rel_path)
    if old_content is None:
        return None  # no previous snapshot to diff against

    old_lines = set(meaningful_lines(old_content))
    new_lines = set(meaningful_lines(new_content))

    added = sorted(new_lines - old_lines)
    removed = sorted(old_lines - new_lines)

    if not added and not removed:
        return None  # content reshuffled but semantically same

    now = datetime.now(timezone.utc).isoformat()
    evo = EvolutionRecord(
        project_name=project.name,
        file_path=str(file_path),
        scope_id=scope_id,
        added_lines=added[:50],   # cap to avoid huge events
        removed_lines=removed[:50],
        timestamp=now,
    )

    # Emit as a knowledge event
    emit_evolution_event(evo, project, knowledge_store)
    return evo


def emit_evolution_event(
    evo: EvolutionRecord, project: ProjectInfo, knowledge_store: Any,
) -> None:
    """Deposit a session_evolution event into knowledge.db."""
    try:
        from ..knowledge_models import Event, Provenance

        summary_parts: list[str] = []
        if evo.added_lines:
            summary_parts.append(f"+{len(evo.added_lines)} lines")
        if evo.removed_lines:
            summary_parts.append(f"-{len(evo.removed_lines)} lines")
        change_summary = ", ".join(summary_parts)

        # Build a readable diff summary (first few items)
        details: list[str] = []
        for line in evo.added_lines[:5]:
            details.append(f"  + {line[:120]}")
        for line in evo.removed_lines[:5]:
            details.append(f"  - {line[:120]}")
        detail_text = "\n".join(details) if details else "(minor changes)"

        event = Event(
            event_type="session_evolution",
            title=f"Session '{project.decoded_path or project.name}' evolved ({change_summary})",
            summary=f"Memory file changed: {evo.file_path}\n{detail_text}",
            scope_id=evo.scope_id,
            produced_by=PROVENANCE_ID,
            metrics={
                "lines_added": len(evo.added_lines),
                "lines_removed": len(evo.removed_lines),
            },
            provenance=Provenance(
                produced_by={"kind": "harvester", "id": PROVENANCE_ID},
            ),
        )
        knowledge_store.insert_event(event)
    except Exception as e:
        logger.debug("Failed to emit evolution event: %s", e)


def meaningful_lines(content: str) -> list[str]:
    """Extract non-empty, non-heading, non-code-fence lines for diffing."""
    lines = []
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("```"):
            continue
        if stripped.startswith("#"):
            continue  # headings change freely
        lines.append(stripped)
    return lines
