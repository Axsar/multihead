"""Load prompt templates from config/prompts/ for solve pipeline agents."""

from __future__ import annotations

import functools
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "config" / "prompts"


@functools.lru_cache(maxsize=16)
def load_prompt(name: str) -> str:
    """Load a prompt template by name (without .md extension).

    Returns empty string if the file doesn't exist (non-fatal).
    """
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        logger.debug("Prompt template not found: %s", path)
        return ""
    return path.read_text(encoding="utf-8")
