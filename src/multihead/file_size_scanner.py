"""Scan project directories for oversized files."""

from __future__ import annotations

import os
from pathlib import Path


# Directories to skip
_EXCLUDE_DIRS = {"node_modules", "__pycache__", ".git", "dist", "build"}

# Thresholds
_WARN_LINES = 300
_CRIT_LINES = 500


def _count_lines(filepath: Path) -> int:
    try:
        return sum(1 for _ in filepath.open("r", encoding="utf-8", errors="replace"))
    except OSError:
        return 0


def _should_skip(path: Path) -> bool:
    return any(part in _EXCLUDE_DIRS for part in path.parts)


def scan_file_sizes(directories: list[dict]) -> list[dict]:
    """Scan directories for oversized files.

    directories: [{"path": "src/multihead", "patterns": ["**/*.py"],
                    "project": "multihead"}]

    Each entry may include an optional ``project`` label.  When provided, every
    finding produced from that entry will carry a ``project`` field so callers
    can group results by project.

    Returns: [{"file": "api/app.py", "lines": 450, "severity": "warning",
               "category": "file_size", "project": "multihead",
               "suggestion": "Split into smaller modules (<300 lines each)"}]
    """
    findings: list[dict] = []

    for entry in directories:
        root = Path(entry["path"])
        if not root.is_dir():
            continue

        patterns: list[str] = entry.get("patterns", [])
        project: str | None = entry.get("project")

        for pattern in patterns:
            for filepath in root.glob(pattern):
                if not filepath.is_file() or _should_skip(filepath):
                    continue

                lines = _count_lines(filepath)
                if lines < _WARN_LINES:
                    continue

                severity = "critical" if lines >= _CRIT_LINES else "warning"
                rel = filepath.relative_to(root)

                finding: dict = {
                    "file": str(rel).replace(os.sep, "/"),
                    "lines": lines,
                    "severity": severity,
                    "category": "file_size",
                    "suggestion": (
                        f"Split into smaller modules (<{_WARN_LINES} lines each)"
                        if severity == "critical"
                        else f"Approaching size limit — consider splitting before it exceeds {_CRIT_LINES} lines"
                    ),
                }
                if project is not None:
                    finding["project"] = project

                findings.append(finding)

    # Sort: critical first, then by line count descending
    findings.sort(key=lambda f: (0 if f["severity"] == "critical" else 1, -f["lines"]))
    return findings
