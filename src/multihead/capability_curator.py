"""Capability curator — auto-classify discovered capabilities as unique vs commodity.

Takes raw DiscoveredCapability objects from codebase_scanner and classifies them
into tiers based on uniqueness signals:

- UNIQUE: Custom-trained model + metrics + domain-specific. Our competitive edge.
- SPECIALIZED: Domain-specific code/pipeline, no custom training. Still valuable.
- COMMODITY: Generic, reproducible with pip install. Skip for marketplace.

Signals used:
- Custom model checkpoint (not a known pretrained base model)
- Training metrics (mAP, IoU, accuracy from our runs)
- Domain-specific keywords (configurable per project)
- Production directory placement
- Multi-stage pipeline membership
- Known pretrained model names (negative signal)
- Cloud API equivalence (negative signal)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .codebase_scanner import DiscoveredCapability, ScanResult


class Tier(str, Enum):
    UNIQUE = "unique"
    SPECIALIZED = "specialized"
    COMMODITY = "commodity"


@dataclass
class CuratedCapability:
    """A capability with curation classification."""

    capability: DiscoveredCapability
    tier: Tier
    reasons: list[str] = field(default_factory=list)
    uniqueness_score: float = 0.0  # 0-1, higher = more unique
    suggested_price: float = 0.0
    marketplace_ready: bool = False


@dataclass
class CurationResult:
    """Result of curating a set of discovered capabilities."""

    unique: list[CuratedCapability] = field(default_factory=list)
    specialized: list[CuratedCapability] = field(default_factory=list)
    commodity: list[CuratedCapability] = field(default_factory=list)

    @property
    def all(self) -> list[CuratedCapability]:
        return self.unique + self.specialized + self.commodity

    @property
    def listable(self) -> list[CuratedCapability]:
        """Capabilities worth listing on marketplace."""
        return [c for c in self.unique + self.specialized if c.marketplace_ready]

    def summary(self) -> dict[str, Any]:
        return {
            "unique": len(self.unique),
            "specialized": len(self.specialized),
            "commodity": len(self.commodity),
            "marketplace_ready": len(self.listable),
        }


# Known pretrained/base model names — NOT custom-trained
_PRETRAINED_PATTERNS = re.compile(
    r"(yolov[5-9][nslmx]|yolov\d+[nslmx]?\.pt$"
    r"|sam2?_hiera|sam_vit"
    r"|resnet\d+|vgg\d+|efficientnet"
    r"|bert[-_]|gpt[-_]|llama[-_]|qwen[-_]|phi[-_]"
    r"|clip[-_]|dino[-_]|mae[-_]"
    r"|detr[-_]|faster_rcnn|mask_rcnn"
    r"|stable[-_]diffusion|sdxl)",
    re.IGNORECASE,
)

# Domain-specific keywords that indicate niche capability.
# Override via MULTIHEAD_DOMAIN_KEYWORDS env var (comma-separated) or pass to CapabilityCurator.
import os as _os
_DOMAIN_KEYWORDS_ENV = _os.environ.get("MULTIHEAD_DOMAIN_KEYWORDS", "")
_DOMAIN_KEYWORDS: frozenset[str] = (
    frozenset(k.strip() for k in _DOMAIN_KEYWORDS_ENV.split(",") if k.strip())
    if _DOMAIN_KEYWORDS_ENV
    else frozenset()
)

# Domain keyword regex for matching in text (built from keywords, empty = never matches)
_DOMAIN_PATTERN = (
    re.compile("|".join(re.escape(k) for k in sorted(_DOMAIN_KEYWORDS)), re.IGNORECASE)
    if _DOMAIN_KEYWORDS
    else re.compile(r"(?!)")  # never-matching pattern
)

# Path-based domain signals (model path or source file path)
_DOMAIN_PATH_PATTERN = re.compile(
    r"Stage\d|Detector|Segmentation|custom|domain|pipeline",
    re.IGNORECASE,
)

# Custom training indicators in model names/paths
_CUSTOM_TRAINING_PATTERN = re.compile(
    r"best_model|final_model|epoch\d+|self_training|training_runs|"
    r"trained_adapter|fine.?tun|custom|checkpoint",
    re.IGNORECASE,
)

# Capabilities available as commodity cloud APIs
_CLOUD_API_EQUIVALENT = re.compile(
    r"(generic.?ocr|text.?extract|summariz|translat|"
    r"sentiment|classif.?(?:text|image)|"
    r"speech.?to.?text|text.?to.?speech|"
    r"face.?detect|object.?detect.?(?:coco|general)|"
    r"image.?caption|embed.?(?:text|sentence))",
    re.IGNORECASE,
)

# Generic utility patterns (not worth listing)
_UTILITY_PATTERNS = re.compile(
    r"(file.?(?:read|writ|copy|mov)|path.?(?:util|help)|"
    r"log.?(?:setup|config|format)|"
    r"config.?(?:load|pars|read)|"
    r"cache.?(?:manag|clear|warm)|"
    r"format.?(?:json|yaml|csv)|"
    r"batch.?(?:process|run)$)",
    re.IGNORECASE,
)

# Pipeline stage indicators (multi-stage = more unique)
_PIPELINE_STAGE_PATTERN = re.compile(
    r"stage\s*\d|step\s*\d|phase\s*\d|pipeline",
    re.IGNORECASE,
)


class CapabilityCurator:
    """Classifies discovered capabilities into uniqueness tiers.

    Usage:
        from multihead.codebase_scanner import CodebaseScanner
        from multihead.capability_curator import CapabilityCurator

        scanner = CodebaseScanner()
        results = scanner.scan_all(paths)

        curator = CapabilityCurator()
        for result in results:
            curated = curator.curate(result.capabilities)
            for cap in curated.listable:
                print(f"{cap.tier}: {cap.capability.name} (${cap.suggested_price})")
    """

    def __init__(
        self,
        domain_keywords: frozenset[str] | None = None,
        min_unique_score: float = 0.6,
        min_specialized_score: float = 0.35,
    ) -> None:
        self._domain_keywords = domain_keywords or _DOMAIN_KEYWORDS
        self._min_unique = min_unique_score
        self._min_specialized = min_specialized_score

    def curate(
        self, capabilities: list[DiscoveredCapability]
    ) -> CurationResult:
        """Classify all capabilities and return curated result."""
        result = CurationResult()

        for cap in capabilities:
            curated = self._classify(cap)

            if curated.tier == Tier.UNIQUE:
                result.unique.append(curated)
            elif curated.tier == Tier.SPECIALIZED:
                result.specialized.append(curated)
            else:
                result.commodity.append(curated)

        # Sort each tier by uniqueness score descending
        result.unique.sort(key=lambda c: -c.uniqueness_score)
        result.specialized.sort(key=lambda c: -c.uniqueness_score)
        result.commodity.sort(key=lambda c: -c.uniqueness_score)

        return result

    def curate_scan_results(
        self, scan_results: list[ScanResult]
    ) -> CurationResult:
        """Curate capabilities from multiple scan results."""
        all_caps = []
        for sr in scan_results:
            all_caps.extend(sr.capabilities)
        return self.curate(all_caps)

    def _classify(self, cap: DiscoveredCapability) -> CuratedCapability:
        """Classify a single capability."""
        score = 0.0
        reasons: list[str] = []

        # --- Positive signals (unique) ---

        # Custom-trained model with non-pretrained name
        if cap.model_path and not _PRETRAINED_PATTERNS.search(cap.model_path):
            score += 0.25
            reasons.append("custom-trained model")
        elif cap.source_type == "model" and not _PRETRAINED_PATTERNS.search(cap.name):
            score += 0.20
            reasons.append("model checkpoint (likely custom)")

        # Has training metrics from our runs
        if cap.eval_metrics:
            metric_count = len(cap.eval_metrics)
            score += min(0.25, 0.1 * metric_count)
            metrics_str = ", ".join(
                f"{k}={v}" for k, v in cap.eval_metrics.items()
            )
            reasons.append(f"training metrics: {metrics_str}")

        # Domain-specific — check text AND paths
        searchable = f"{cap.name} {cap.description} {cap.category}"
        domain_matches = _DOMAIN_PATTERN.findall(searchable)
        if not domain_matches:
            # Check file paths for domain signals
            path_text = f"{cap.source_file} {cap.model_path}"
            path_matches = _DOMAIN_PATH_PATTERN.findall(path_text)
            if path_matches:
                domain_matches = path_matches
        if domain_matches:
            score += 0.20
            reasons.append(f"domain-specific: {', '.join(set(domain_matches[:3]))}")

        # Custom training indicators in name/path (epoch numbers, best_model, etc.)
        trainable = f"{cap.name} {cap.model_path} {cap.source_file}"
        if _CUSTOM_TRAINING_PATTERN.search(trainable) and not _PRETRAINED_PATTERNS.search(trainable):
            score += 0.10
            reasons.append("custom training indicators")

        # Production directory
        if cap.is_production:
            score += 0.10
            reasons.append("production code")

        # Multi-stage pipeline membership
        if _PIPELINE_STAGE_PATTERN.search(f"{cap.name} {cap.capability_id}"):
            score += 0.10
            reasons.append("pipeline stage")

        # Documented capability (from CLAUDE.md or MEMORY.md) — curated by human
        if cap.source_type in ("claude_md", "memory"):
            score += 0.10
            reasons.append("documented in project docs")

        # Has tests (quality signal)
        if cap.has_tests:
            score += 0.05
            reasons.append("tested")

        # Large model (>50MB suggests custom training, not a tiny pretrained)
        if cap.model_size_mb > 50:
            score += 0.05
            reasons.append(f"substantial model ({cap.model_size_mb}MB)")

        # --- Negative signals (commodity) ---

        # Known pretrained base model
        if _PRETRAINED_PATTERNS.search(f"{cap.name} {cap.model_path}"):
            score -= 0.20
            reasons.append("pretrained base model")

        # Available as cloud API
        if _CLOUD_API_EQUIVALENT.search(searchable):
            score -= 0.25
            reasons.append("cloud API equivalent exists")

        # Generic utility code
        if _UTILITY_PATTERNS.search(searchable):
            score -= 0.20
            reasons.append("generic utility")

        # No description = low value
        if not cap.description:
            score -= 0.05
            reasons.append("undocumented")

        # Generic category with no domain signal
        if cap.category in ("tool", "optimization") and not domain_matches:
            score -= 0.10
            reasons.append("generic category")

        # Clamp
        score = max(0.0, min(1.0, score))

        # Determine tier
        if score >= self._min_unique:
            tier = Tier.UNIQUE
        elif score >= self._min_specialized:
            tier = Tier.SPECIALIZED
        else:
            tier = Tier.COMMODITY

        # Pricing based on tier and GPU requirement
        if tier == Tier.UNIQUE:
            price = 2.00 if cap.requires_gpu else 1.00
        elif tier == Tier.SPECIALIZED:
            price = 1.00 if cap.requires_gpu else 0.50
        else:
            price = 0.25

        # Marketplace readiness
        marketplace_ready = (
            tier in (Tier.UNIQUE, Tier.SPECIALIZED)
            and cap.confidence >= 0.4
            and bool(cap.description)
        )

        return CuratedCapability(
            capability=cap,
            tier=tier,
            reasons=reasons,
            uniqueness_score=round(score, 3),
            suggested_price=price,
            marketplace_ready=marketplace_ready,
        )

    def summary_report(self, result: CurationResult) -> str:
        """Human-readable curation report."""
        lines = ["# Capability Curation Report", ""]
        counts = result.summary()
        lines.append(
            f"**{counts['unique']}** unique | "
            f"**{counts['specialized']}** specialized | "
            f"**{counts['commodity']}** commodity | "
            f"**{counts['marketplace_ready']}** marketplace-ready"
        )
        lines.append("")

        if result.unique:
            lines.append("## Unique (list at premium)")
            for cc in result.unique:
                cap = cc.capability
                lines.append(
                    f"- **{cap.name}** [{cap.project}] "
                    f"score={cc.uniqueness_score} ${cc.suggested_price}"
                )
                for r in cc.reasons:
                    lines.append(f"  - {r}")
            lines.append("")

        if result.specialized:
            lines.append("## Specialized (list at standard)")
            for cc in result.specialized:
                cap = cc.capability
                lines.append(
                    f"- **{cap.name}** [{cap.project}] "
                    f"score={cc.uniqueness_score} ${cc.suggested_price}"
                )
                for r in cc.reasons:
                    lines.append(f"  - {r}")
            lines.append("")

        if result.commodity:
            lines.append(f"## Commodity ({len(result.commodity)} skipped)")
            # Show first 10
            for cc in result.commodity[:10]:
                cap = cc.capability
                neg = [r for r in cc.reasons if any(
                    w in r for w in ["pretrained", "cloud", "generic", "undocumented"]
                )]
                why = f" ({'; '.join(neg)})" if neg else ""
                lines.append(f"- {cap.name}{why}")
            if len(result.commodity) > 10:
                lines.append(f"- ... and {len(result.commodity) - 10} more")

        return "\n".join(lines)
