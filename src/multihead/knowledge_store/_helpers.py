"""Shared utility functions for the knowledge store."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _safe_json_loads(raw: str | None, default: Any = None, context: str = "") -> Any:
    """Parse JSON with fallback on corruption."""
    if raw is None:
        return default if default is not None else []
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("Corrupted JSON in %s: %s", context, e)
        return default if default is not None else []


_IMPORTANCE_STRING_MAP: dict[str, float] = {
    "critical": 1.0,
    "high": 0.9,
    "medium": 0.5,
    "low": 0.2,
    "none": 0.0,
}


def _coerce_float(value: Any, default: float = 0.0) -> float:
    """Coerce a DB value to float, handling legacy string labels like 'high'/'medium'/'low'."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in _IMPORTANCE_STRING_MAP:
            return _IMPORTANCE_STRING_MAP[lower]
        try:
            return float(lower)
        except ValueError:
            logger.warning("Cannot coerce importance/confidence value %r to float, using %s", value, default)
            return default
    return default


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_timestamp(ts: str | None) -> str:
    """Normalize timestamp to ISO 8601 with T separator for consistent comparison.

    Handles both formats found in DB:
    - '2026-03-16 17:25:50' (space-separated, from external agents)
    - '2026-03-16T19:24:48.123456+00:00' (ISO, from _now_iso)

    Returns ISO format with T separator so string comparison works correctly.
    """
    if not ts:
        return ""
    # Replace space separator with T if present at position 10
    if len(ts) > 10 and ts[10] == " ":
        ts = ts[:10] + "T" + ts[11:]
    return ts


def _get_producer_id(provenance_json: str | dict | None) -> str:
    """Extract producer ID from provenance, handling both formats.

    Format 1 (our code): {"produced_by": {"kind": "extractor", "id": "nightshift_v1"}}
    Format 2 (external):  {"produced_by": "claude_code_main"}

    Returns the producer ID string, or "" if not found.
    """
    if not provenance_json:
        return ""
    if isinstance(provenance_json, str):
        try:
            prov = json.loads(provenance_json)
        except (json.JSONDecodeError, TypeError):
            return ""
    else:
        prov = provenance_json

    produced_by = prov.get("produced_by", "")
    if isinstance(produced_by, dict):
        return produced_by.get("id", "")
    if isinstance(produced_by, str):
        return produced_by
    return ""
