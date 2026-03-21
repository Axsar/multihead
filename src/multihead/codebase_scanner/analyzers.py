"""File analyzers for codebase scanning.

Handles model checkpoint discovery, Python AST analysis, and
conversion of discovered artifacts to DiscoveredCapability objects.
"""

from __future__ import annotations

import ast
import logging
import os
import re
from pathlib import Path
from typing import Any

from .models import (
    COMMODITY_PATTERNS,
    MAX_LINES,
    MAX_PY_FILES,
    MIN_FUNCTION_LINES,
    PRIORITY_DIRS,
    SKIP_DIRS,
    DiscoveredCapability,
)
from .parsers import infer_category, infer_model_type, slug

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Model checkpoint scanning
# ------------------------------------------------------------------


def find_models(project_path: Path) -> list[dict[str, Any]]:
    """Find model checkpoint files, deduplicating training epoch runs."""
    raw_checkpoints: list[dict[str, Any]] = []
    extensions = {".pt", ".pth", ".onnx", ".safetensors"}

    for root, dirs, files in os.walk(project_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for f in files:
            if Path(f).suffix.lower() in extensions:
                fpath = Path(root) / f
                try:
                    size_mb = fpath.stat().st_size / (1024 * 1024)
                except OSError:
                    size_mb = 0

                if size_mb < 1:
                    continue

                rel = str(fpath.relative_to(project_path))
                model_type = infer_model_type(f, rel)

                raw_checkpoints.append({
                    "path": rel,
                    "name": f,
                    "format": Path(f).suffix[1:],
                    "size_mb": round(size_mb, 1),
                    "model_type": model_type,
                    "full_path": str(fpath),
                })

    # Deduplicate: group by (top-level dir, ~size, model_type) to collapse
    # epoch checkpoints from the same training run into one entry.
    # Use top-level subdir (e.g. "Stage2Detector") not immediate parent,
    # so self_training/iteration_N/weights/ all collapse together.
    # Round size to nearest 10MB so slight variations still group.
    groups: dict[tuple, list[dict]] = {}
    for ckpt in raw_checkpoints:
        parts = Path(ckpt["path"]).parts
        # Use first 2 path components as group key (e.g. "Stage2Detector/models")
        # This collapses training_runs/*/weights and self_training/*/weights
        top_dir = str(Path(*parts[:2])) if len(parts) > 2 else str(Path(ckpt["path"]).parent)
        size_bucket = round(ckpt["size_mb"] / 10) * 10
        key = (top_dir, size_bucket, ckpt["model_type"])
        groups.setdefault(key, []).append(ckpt)

    checkpoints = []
    for key, group in groups.items():
        if len(group) <= 3:
            # Few checkpoints — keep all
            checkpoints.extend(group)
        else:
            # Training run with many epochs — keep only "best" or latest
            best = None
            for ckpt in group:
                name_lower = ckpt["name"].lower()
                if "best" in name_lower or "final" in name_lower:
                    best = ckpt
                    break
            if best is None:
                # Keep the last one (highest epoch number usually)
                best = group[-1]
            best["name"] = f"{best['name']} (+{len(group)-1} epoch checkpoints)"
            checkpoints.append(best)

    checkpoints.sort(key=lambda x: x["size_mb"], reverse=True)
    return checkpoints


def model_to_capability(
    ckpt: dict[str, Any], proj_name: str
) -> DiscoveredCapability:
    """Convert a model checkpoint to a capability."""
    name = Path(ckpt["name"]).stem
    model_type = ckpt.get("model_type", "model")
    cap_id = f"{proj_name}.model.{slug(name)}"

    return DiscoveredCapability(
        name=f"Model: {name}",
        capability_id=cap_id,
        project=proj_name,
        source_file=ckpt["path"],
        source_type="model",
        description=f"{ckpt['format']} model ({ckpt['size_mb']}MB)",
        category=model_type,
        requires_gpu=True,
        model_path=ckpt.get("full_path", ""),
        model_size_mb=ckpt["size_mb"],
    )


# ------------------------------------------------------------------
# Python file scanning
# ------------------------------------------------------------------


def scan_python_files(
    project_path: Path, proj_name: str
) -> tuple[list[DiscoveredCapability], int]:
    """Scan Python files for capability-indicating code."""
    caps: list[DiscoveredCapability] = []
    py_files = collect_python_files(project_path)
    scanned = 0

    for py_file in py_files[:MAX_PY_FILES]:
        try:
            file_caps = analyze_file(py_file, project_path, proj_name)
            caps.extend(file_caps)
            scanned += 1
        except Exception as e:
            logger.debug("Error analyzing %s: %s", py_file, e)

    return caps, scanned


def collect_python_files(project_path: Path) -> list[Path]:
    """Collect Python files, prioritizing production directories."""
    priority_files = []
    other_files = []

    for root, dirs, files in os.walk(project_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        root_path = Path(root)
        is_priority = any(
            p in root_path.parts for p in PRIORITY_DIRS
        )

        for f in files:
            if not f.endswith(".py") or f.startswith("__"):
                continue
            if COMMODITY_PATTERNS.search(f):
                continue

            fpath = root_path / f
            if is_priority:
                priority_files.append(fpath)
            else:
                other_files.append(fpath)

    # Priority files first, then others
    return priority_files + other_files


def analyze_file(
    file_path: Path, project_root: Path, proj_name: str
) -> list[DiscoveredCapability]:
    """Analyze a Python file for capabilities via AST."""
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    lines = source.split("\n")
    if len(lines) > MAX_LINES:
        source = "\n".join(lines[:MAX_LINES])

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        return []

    rel_path = str(file_path.relative_to(project_root))
    is_prod = "production" in rel_path.lower() or "src" in rel_path.lower()
    caps: list[DiscoveredCapability] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            cap = _class_to_cap(node, rel_path, proj_name, is_prod)
            if cap:
                caps.append(cap)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if _is_significant_function(node):
                cap = _func_to_cap(node, rel_path, proj_name, is_prod)
                if cap:
                    caps.append(cap)

    # Detect GPU usage
    needs_gpu = bool(re.search(
        r"\.cuda\(\)|\.to\(.device.\)|import\s+torch|from\s+torch|VRAM|\.gpu",
        source, re.IGNORECASE
    ))
    if needs_gpu:
        for cap in caps:
            cap.requires_gpu = True

    return caps


def _class_to_cap(
    node: ast.ClassDef, rel_path: str,
    proj_name: str, is_prod: bool
) -> DiscoveredCapability | None:
    """Convert a class to a capability if it looks significant."""
    name = node.name
    if name.startswith("_") or name.startswith("Test"):
        return None

    # Must contain a capability-indicating suffix
    indicators = {
        "Detector", "Segmenter", "Classifier", "Generator", "Pipeline",
        "Processor", "Analyzer", "Extractor", "Builder", "Engine",
        "Synthesizer", "Transformer", "Layout", "Renderer", "Placer",
        "Evaluator", "Validator", "Trainer", "Model", "Converter",
        "Orchestrator", "Manager", "Cache", "Positioner", "Measurer",
        "Planner", "Splitter", "Fitter", "Cleaner", "Infiller",
    }

    has_indicator = any(ind in name for ind in indicators)
    if not has_indicator:
        return None

    docstring = ast.get_docstring(node) or ""
    methods = [
        item.name for item in node.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not item.name.startswith("_")
    ]

    category = infer_category(name + " " + docstring)
    cap_id = f"{proj_name}.{category}.{slug(name)}"

    return DiscoveredCapability(
        name=name,
        capability_id=cap_id,
        project=proj_name,
        source_file=rel_path,
        source_type="code",
        description=docstring[:200],
        category=category,
        is_production=is_prod,
        functions=methods[:10],
    )


def _func_to_cap(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    rel_path: str, proj_name: str, is_prod: bool
) -> DiscoveredCapability | None:
    """Convert a standalone function to a capability."""
    name = node.name
    docstring = ast.get_docstring(node) or ""
    category = infer_category(name + " " + docstring)
    cap_id = f"{proj_name}.{category}.{slug(name)}"

    return DiscoveredCapability(
        name=name,
        capability_id=cap_id,
        project=proj_name,
        source_file=rel_path,
        source_type="code",
        description=docstring[:200],
        category=category,
        is_production=is_prod,
        functions=[name],
    )


def _is_significant_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check if function is significant enough to be a capability."""
    if node.name.startswith("_") or node.name.startswith("test_"):
        return False

    body_lines = len(node.body)
    if body_lines < MIN_FUNCTION_LINES:
        return False

    action_prefixes = [
        "detect", "segment", "classify", "generate", "process",
        "analyze", "extract", "build", "render", "transform",
        "train", "evaluate", "predict", "export", "convert",
        "validate", "layout", "synthesize", "create", "run",
        "batch", "pipeline", "fit", "clean", "infill",
    ]

    return any(node.name.lower().startswith(p) for p in action_prefixes)
