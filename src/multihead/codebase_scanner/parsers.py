"""Parsing helpers for codebase scanning.

Handles document parsing, metrics extraction, category inference,
slug generation, and capability deduplication/scoring.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .models import COMMODITY_PATTERNS, DiscoveredCapability, ScanResult


def extract_all_metrics(text: str) -> dict[str, float]:
    """Extract all evaluation metrics from text."""
    metrics: dict[str, float] = {}
    patterns = [
        (r"(\d+\.?\d*)%?\s*mAP\d*", "mAP"),
        (r"mAP\d*[:\s]*(\d+\.?\d*)%?", "mAP"),
        (r"(\d+\.?\d*)%?\s*IoU", "IoU"),
        (r"IoU[:\s]*(\d+\.?\d*)%?", "IoU"),
        (r"accuracy[:\s]*(\d+\.?\d*)%?", "accuracy"),
        (r"(\d+)/(\d+)\s*(?:items|objects|regions|success)", "success_rate"),
        (r"(\d+\.?\d*)%\s*success", "success_rate"),
        (r"precision[:\s]*(\d+\.?\d*)", "precision"),
        (r"recall[:\s]*(\d+\.?\d*)", "recall"),
        (r"f1[:\s]*(\d+\.?\d*)", "f1"),
    ]

    for pattern, name in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1))
                if name == "success_rate" and match.lastindex == 2:
                    val = val / float(match.group(2)) * 100
                if val > 1 and name not in ("success_rate",):
                    val = val / 100.0
                # Only set if we don't already have a higher value
                # (prevents later regex backtrack matches from overwriting)
                if val > metrics.get(name, -1):
                    metrics[name] = round(val, 4)
            except (ValueError, IndexError, ZeroDivisionError):
                pass

    return metrics


def metrics_near(
    text: str, target: str,
    all_metrics: dict[str, float], window: int = 500,
) -> dict[str, float]:
    """Get metrics from text near a target string."""
    idx = text.lower().find(target.lower())
    if idx < 0:
        return {}
    start = max(0, idx - window)
    end = min(len(text), idx + len(target) + window)
    return extract_all_metrics(text[start:end])


def infer_category(text: str) -> str:
    """Infer capability category from text."""
    t = text.lower()
    if any(kw in t for kw in ["detect", "yolo", "bbox", "object"]):
        return "detection"
    if any(kw in t for kw in ["segment", "mask", "sam", "unet"]):
        return "segmentation"
    if any(kw in t for kw in ["svg", "vector", "contour"]):
        return "vectorization"
    if any(kw in t for kw in ["layout", "position", "placer"]):
        return "layout"
    if any(kw in t for kw in ["generate", "render", "synthesize"]):
        return "generation"
    if any(kw in t for kw in ["ocr", "text", "recogni"]):
        return "ocr"
    if any(kw in t for kw in ["train", "dataset", "annotation"]):
        return "training"
    if any(kw in t for kw in ["valid", "check", "verify"]):
        return "validation"
    if any(kw in t for kw in ["pipeline", "batch", "orchestrat"]):
        return "pipeline"
    if any(kw in t for kw in ["clean", "infill", "ink"]):
        return "processing"
    if any(kw in t for kw in ["transform", "convert", "export"]):
        return "conversion"
    if any(kw in t for kw in ["cache", "optim"]):
        return "optimization"
    return "tool"


def slug(name: str) -> str:
    """Convert to a clean slug."""
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).lower().strip("_")
    return s[:50]


def project_name(path: Path) -> str:
    """Generate a short project name from path."""
    name = path.name
    # Project name mappings — customize for your setup
    mappings: dict[str, str] = {}
    for key, val in mappings.items():
        if key.lower() in name.lower():
            return val

    return slug(name)


def parse_capability_doc(
    text: str, source_file: str, proj_name: str,
    source_type: str
) -> list[DiscoveredCapability]:
    """Parse a documentation file for capability declarations."""
    caps: list[DiscoveredCapability] = []

    # Extract model performance metrics
    metrics = extract_all_metrics(text)

    # Look for pipeline stage declarations — only match headings/definitions,
    # not every mention of "Stage N" in prose
    stage_pattern = re.compile(
        r"^#{1,4}\s+Stage\s+(\d+)[:\s-]+(\w[\w\s/&]+?)(?:\n|$)"
        r"|"
        r"^\*\*Stage\s+(\d+)[:\s-]+([A-Z][\w\s/&]+?)\*\*",
        re.MULTILINE,
    )
    seen_stages: set[str] = set()
    for m in stage_pattern.finditer(text):
        stage_num = m.group(1) or m.group(3)
        stage_desc = (m.group(2) or m.group(4) or "").strip().rstrip("*#")
        if not stage_num or len(stage_desc) < 5 or len(stage_desc) > 80:
            continue

        cap_name = stage_desc.split("(")[0].strip().split("-")[0].strip()
        # Skip if we already have this stage
        dedup_key = f"stage{stage_num}"
        if dedup_key in seen_stages:
            continue
        seen_stages.add(dedup_key)

        category = infer_category(cap_name + " " + stage_desc)
        cap_id = f"{proj_name}.stage{stage_num}.{slug(cap_name)}"

        cap = DiscoveredCapability(
            name=f"Stage {stage_num}: {cap_name}",
            capability_id=cap_id,
            project=proj_name,
            source_file=source_file,
            source_type=source_type,
            description=stage_desc[:200],
            category=category,
            eval_metrics=metrics_near(text, cap_name, metrics),
        )
        caps.append(cap)

    # Look for tool/script declarations
    script_pattern = re.compile(
        r"\*\*(?:File|Script|Tool)\*\*:\s*`([^`]+)`\s*(?:\(([^)]+)\))?",
    )
    for m in script_pattern.finditer(text):
        script_path = m.group(1)
        script_desc = m.group(2) or ""
        script_name = Path(script_path).stem

        if COMMODITY_PATTERNS.search(script_name):
            continue

        cap_id = f"{proj_name}.tool.{slug(script_name)}"
        caps.append(DiscoveredCapability(
            name=script_name,
            capability_id=cap_id,
            project=proj_name,
            source_file=source_file,
            source_type=source_type,
            description=script_desc[:200],
            category="tool",
            eval_metrics=metrics_near(text, script_name, metrics),
        ))

    # Look for class/component declarations
    class_pattern = re.compile(
        r"\*\*(\w+(?:Fill|Pipeline|Generator|Detector|Segmenter|Validator|"
        r"Positioner|Cache|Editor|Placer|Synthesizer|Measurer|Builder|"
        r"Extractor|Converter|Processor))\*\*"
    )
    for m in class_pattern.finditer(text):
        cls_name = m.group(1)
        # Get context around the match
        start = max(0, m.start() - 100)
        end = min(len(text), m.end() + 200)
        context = text[start:end]

        category = infer_category(cls_name)
        cap_id = f"{proj_name}.{category}.{slug(cls_name)}"

        caps.append(DiscoveredCapability(
            name=cls_name,
            capability_id=cap_id,
            project=proj_name,
            source_file=source_file,
            source_type=source_type,
            description=context[100:300].strip()[:200],
            category=category,
            eval_metrics=metrics_near(text, cls_name, metrics),
        ))

    return caps


def deduplicate(
    caps: list[DiscoveredCapability],
) -> list[DiscoveredCapability]:
    """Deduplicate capabilities by ID, keeping the richest one."""
    by_id: dict[str, DiscoveredCapability] = {}
    for cap in caps:
        existing = by_id.get(cap.capability_id)
        if existing is None:
            by_id[cap.capability_id] = cap
        else:
            # Merge: keep the one with more info, merge metrics
            if len(cap.description) > len(existing.description):
                cap.eval_metrics = {**existing.eval_metrics, **cap.eval_metrics}
                cap.has_tests = cap.has_tests or existing.has_tests
                cap.is_production = cap.is_production or existing.is_production
                by_id[cap.capability_id] = cap
            else:
                existing.eval_metrics.update(cap.eval_metrics)
                existing.has_tests = existing.has_tests or cap.has_tests
                existing.is_production = existing.is_production or cap.is_production
    return list(by_id.values())


def score_all(result: ScanResult) -> None:
    """Score all capabilities by quality signals."""
    for cap in result.capabilities:
        score = 0.3
        if cap.source_type in ("claude_md", "memory"):
            score += 0.2  # Documented = higher confidence
        if cap.is_production:
            score += 0.15
        if cap.has_tests:
            score += 0.1
        if cap.eval_metrics:
            score += 0.15
        if cap.has_cli:
            score += 0.05
        if cap.model_path:
            score += 0.1
        cap.confidence = min(1.0, max(0.1, score))


def infer_model_type(filename: str, rel_path: str) -> str:
    """Infer model type from filename and path."""
    combined = f"{filename} {rel_path}".lower()

    if any(kw in combined for kw in ["yolo", "detect", "object"]):
        return "detection"
    if any(kw in combined for kw in ["sam", "segment", "mask", "unet"]):
        return "segmentation"
    if any(kw in combined for kw in ["ocr", "text", "recognize"]):
        return "ocr"
    if any(kw in combined for kw in ["classify", "class"]):
        return "classification"
    if any(kw in combined for kw in ["embed", "encoder"]):
        return "embedding"
    if any(kw in combined for kw in ["generate", "diffus"]):
        return "generation"
    return "model"
