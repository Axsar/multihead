"""Claim corroboration utilities — verify claims against independent sources.

Implements the triangulation pattern for coding agents:
- File existence checks (does the claimed artifact exist on disk?)
- Git SHA staleness (has the source file changed since the claim was derived?)
- Bash output classification (dynamic observation vs inference)
- Path extraction from conversation text

These are cheap, heuristic checks that add corroboration signals
to claims without requiring LLM calls.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from multihead.subprocess_utils import no_window_flags

# Patterns to extract file paths from text
_FILE_PATH_RE = re.compile(
    r'(?:^|[\s\'"`(])(/(?:mnt|home|tmp|usr|opt|var)[/\w._-]+(?:\.\w{1,10})?)',
    re.M,
)
_WINDOWS_PATH_RE = re.compile(
    r'(?:^|[\s\'"`(])([A-Z]:\\[\w._\\-]+(?:\.\w{1,10})?)',
    re.M,
)


def extract_file_paths(text: str) -> list[str]:
    """Extract file paths mentioned in text."""
    paths = set()
    for m in _FILE_PATH_RE.finditer(text):
        p = m.group(1).rstrip(".,;:)]}'\"")
        if len(p) > 5 and not p.endswith("/"):
            paths.add(p)
    for m in _WINDOWS_PATH_RE.finditer(text):
        p = m.group(1).rstrip(".,;:)]}'\"")
        if len(p) > 5:
            # Convert Windows path to WSL path
            drive = p[0].lower()
            rest = p[3:].replace("\\", "/")
            paths.add(f"/mnt/{drive}/{rest}")
    return sorted(paths)


def check_file_exists(path: str) -> dict[str, str | bool]:
    """Check if a file exists on disk. Returns corroboration metadata.

    Returns:
        {"exists": True/False, "size": "123", "mtime": "2026-03-14T..."}
    """
    try:
        p = Path(path)
        if p.exists():
            stat = p.stat()
            return {
                "exists": True,
                "size": str(stat.st_size),
                "is_dir": str(p.is_dir()),
            }
    except (OSError, ValueError):
        pass
    return {"exists": False}


def get_git_head_sha(repo_path: str = ".") -> str | None:
    """Get the current HEAD SHA for a git repository."""
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            creationflags=no_window_flags(),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def check_sha_staleness(
    source_sha: str,
    file_path: str,
    symbol_name: str = "",
    repo_path: str = ".",
) -> dict[str, str | bool | float]:
    """Check if a file has changed since a given git SHA.

    Scales penalty by relevance (expert pushback #1):
    - Symbol appears in diff → heavy penalty (-0.5)
    - Diff touches file but not symbol → medium penalty (-0.15)
    - File unchanged → no penalty (0.0)

    Returns:
        {"stale": True/False, "penalty": float, "symbol_in_diff": True/False}
    """
    if not source_sha or not file_path:
        return {"stale": False, "penalty": 0.0, "reason": "no_sha_or_path"}

    try:
        # Get the actual diff for this file
        result = subprocess.run(
            ["git", "-C", repo_path, "diff", source_sha, "HEAD", "--", file_path],
            capture_output=True, text=True, timeout=10,
            creationflags=no_window_flags(),
        )
        if result.returncode != 0:
            return {"stale": False, "penalty": 0.0, "reason": "git_error"}

        diff_text = result.stdout.strip()
        if not diff_text:
            return {"stale": False, "penalty": 0.0, "file_changed": False}

        # File changed — now check if the symbol is in the diff
        symbol_in_diff = False
        if symbol_name:
            symbol_in_diff = symbol_name in diff_text

        if symbol_in_diff:
            # The specific thing this claim is about was modified
            return {
                "stale": True,
                "penalty": -0.5,
                "file_changed": True,
                "symbol_in_diff": True,
                "source_sha": source_sha,
            }
        else:
            # File changed but not the specific symbol
            return {
                "stale": True,
                "penalty": -0.15,
                "file_changed": True,
                "symbol_in_diff": False,
                "source_sha": source_sha,
            }

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return {"stale": False, "penalty": 0.0, "reason": "git_check_failed"}


def classify_bash_output(text: str) -> str:
    """Classify bash output as a type of dynamic observation.

    Returns: test_pass, test_fail, command_success, command_error, build_output, unknown
    """
    lower = text.lower()

    # Test results
    if re.search(r'\d+\s+passed', lower) or "tests passed" in lower:
        if re.search(r'\d+\s+failed', lower) or "FAILED" in text:
            return "test_fail"
        return "test_pass"

    if "FAILED" in text or "AssertionError" in text or "assert" in lower and "error" in lower:
        return "test_fail"

    # Build output
    if "Successfully installed" in text or "Building wheel" in text:
        return "build_output"

    # Command errors
    if "Error:" in text or "error:" in lower or "Traceback" in text:
        return "command_error"

    # Command success indicators
    if "exit code 0" in lower or text.strip().endswith("OK"):
        return "command_success"

    return "unknown"


def corroborate_claim(
    statement: str,
    observation_method: str = "",
    file_path: str = "",
    source_sha: str = "",
    repo_path: str = ".",
) -> dict:
    """Run all available corroboration checks for a claim.

    Returns a dict with corroboration signals:
    {
        "file_exists": True/False,
        "file_stale": True/False,
        "paths_found": [...],
        "corroboration_level": "strong" | "medium" | "weak" | "none",
        "confidence_modifier": float (-0.3 to +0.2),
    }
    """
    result = {
        "paths_found": [],
        "file_exists": None,
        "file_stale": None,
        "corroboration_level": "none",
        "confidence_modifier": 0.0,
    }

    # Extract paths from statement
    paths = extract_file_paths(statement)
    result["paths_found"] = paths

    # Check specific file_path if provided
    if file_path:
        exists = check_file_exists(file_path)
        result["file_exists"] = exists.get("exists", False)
        if result["file_exists"]:
            result["confidence_modifier"] += 0.1
            result["corroboration_level"] = "medium"

    # Check paths found in statement text
    elif paths:
        any_exists = False
        for p in paths[:5]:  # Check up to 5 paths
            if check_file_exists(p).get("exists"):
                any_exists = True
                break
        result["file_exists"] = any_exists
        if any_exists:
            result["confidence_modifier"] += 0.05
            result["corroboration_level"] = "weak"

    # Git staleness check — scaled by diff proximity (expert pushback #1)
    if source_sha and file_path:
        # Extract symbol from statement for targeted staleness check
        symbol = ""
        if "file_path" in locals():
            # Try to get symbol from kwargs or extract from statement
            pass  # symbol passed separately below
        staleness = check_sha_staleness(source_sha, file_path, symbol_name=symbol, repo_path=repo_path)
        result["file_stale"] = staleness.get("stale", False)
        if result["file_stale"]:
            penalty = staleness.get("penalty", -0.15)
            result["confidence_modifier"] += penalty  # Already negative
            result["symbol_in_diff"] = staleness.get("symbol_in_diff", False)
            result["corroboration_level"] = "stale"

    # Observation method affects baseline
    if observation_method == "bash_output":
        result["confidence_modifier"] += 0.15
        if result["corroboration_level"] == "none":
            result["corroboration_level"] = "medium"
    elif observation_method == "user_statement":
        result["confidence_modifier"] += 0.05
        if result["corroboration_level"] == "none":
            result["corroboration_level"] = "weak"

    return result
