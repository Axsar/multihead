"""Shared path helpers — project root discovery and path normalization.

Used by claim_fusion, night_shift stages, CLI knowledge commands, and
MCP tools to avoid hardcoding user-specific absolute paths.

Configure via environment variable:
    MULTIHEAD_PROJECT_ROOTS=/path/to/project1/:/path/to/project2/
"""

from __future__ import annotations

import os


def get_known_project_roots() -> list[str]:
    """Project roots for path normalization. Configure via MULTIHEAD_PROJECT_ROOTS env var."""
    env_roots = os.environ.get("MULTIHEAD_PROJECT_ROOTS", "")
    if env_roots:
        return [r.strip() for r in env_roots.split(":") if r.strip()]
    return []  # No default hardcoded paths


def normalize_file_path(path: str) -> str:
    """Strip known project root prefixes from a file path."""
    for prefix in get_known_project_roots():
        if path.startswith(prefix):
            return path[len(prefix):]
    return path
