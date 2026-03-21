"""Mid-stage checkpoint support for Night Shift pipeline.

Stores partial results as JSON files so long-running stages can resume
from where they left off after interruption.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StageCheckpoint(BaseModel):
    """Snapshot of a stage's progress through its chunk list."""

    stage_name: str
    processed_count: int = 0
    total_count: int = 0
    partial_results: list[dict[str, Any] | None] = Field(default_factory=list)
    updated_at: str = ""


def _checkpoint_path(output_dir: Path, stage_name: str) -> Path:
    return output_dir / f"{stage_name}_checkpoint.json"


def save_checkpoint(output_dir: Path, checkpoint: StageCheckpoint) -> None:
    """Persist a stage checkpoint to disk."""
    checkpoint.updated_at = datetime.now(timezone.utc).isoformat()
    path = _checkpoint_path(output_dir, checkpoint.stage_name)
    path.write_text(
        json.dumps(checkpoint.model_dump(), indent=2, default=str),
        encoding="utf-8",
    )


def load_checkpoint(output_dir: Path, stage_name: str) -> StageCheckpoint | None:
    """Load a previously saved checkpoint, or None if absent/corrupt."""
    path = _checkpoint_path(output_dir, stage_name)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return StageCheckpoint(**data)
    except Exception as e:
        logger.warning("Corrupt checkpoint %s, ignoring: %s", path, e)
        return None


def clear_checkpoint(output_dir: Path, stage_name: str) -> None:
    """Remove a checkpoint file after successful stage completion."""
    path = _checkpoint_path(output_dir, stage_name)
    try:
        path.unlink(missing_ok=True)
    except Exception as e:
        logger.warning("Failed to clear checkpoint %s: %s", path, e)
