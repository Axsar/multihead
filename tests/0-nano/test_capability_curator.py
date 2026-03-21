"""Tests for capability_curator — auto-classify unique vs commodity."""

from __future__ import annotations

import os

import pytest

from multihead.capability_curator import (
    CapabilityCurator,
    Tier,
)
from multihead.codebase_scanner import DiscoveredCapability


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _set_domain_keywords(monkeypatch):
    """Set domain keywords for tests — curator needs these to detect domain specificity."""
    monkeypatch.setenv(
        "MULTIHEAD_DOMAIN_KEYWORDS",
        "comic,panel,balloon,speech bubble,manga,ink,lettering,tail,narration,sfx,h2v",
    )
    # Force reimport to pick up new env var
    import importlib
    import multihead.capability_curator as mod
    importlib.reload(mod)


@pytest.fixture
def curator():
    from multihead.capability_curator import CapabilityCurator
    return CapabilityCurator()


def _cap(**kwargs) -> DiscoveredCapability:
    """Quick capability builder with sensible defaults."""
    defaults = {
        "name": "TestCap",
        "capability_id": "test.tool.test_cap",
        "project": "test",
        "source_file": "test.py",
        "source_type": "code",
    }
    defaults.update(kwargs)
    return DiscoveredCapability(**defaults)


# ── Tier Classification ───────────────────────────────────────


class TestTierClassification:
    def test_custom_model_with_metrics_is_unique(self, curator):
        cap = _cap(
            name="YOLOComicDetector",
            model_path="/models/best_model_epoch84.pt",
            eval_metrics={"mAP": 0.9185, "IoU": 0.87},
            description="8-class comic panel detection",
            is_production=True,
            requires_gpu=True,
            source_type="model",
        )
        result = curator._classify(cap)
        assert result.tier == Tier.UNIQUE
        assert result.uniqueness_score >= 0.6

    def test_domain_pipeline_stage_is_unique(self, curator):
        cap = _cap(
            name="Stage 2: Panel Detection",
            capability_id="h2v.stage2.panel_detection",
            description="YOLO-based comic panel detection pipeline stage",
            eval_metrics={"mAP": 0.92},
            is_production=True,
            model_path="/m/best_model.pt",
        )
        result = curator._classify(cap)
        assert result.tier == Tier.UNIQUE

    def test_domain_code_no_model_is_specialized(self, curator):
        cap = _cap(
            name="HeuristicPanelDetector",
            description="Connected component analysis for comic panel borders",
            category="detection",
            is_production=True,
            has_tests=True,
        )
        result = curator._classify(cap)
        assert result.tier in (Tier.UNIQUE, Tier.SPECIALIZED)
        assert result.uniqueness_score >= 0.35

    def test_generic_utility_is_commodity(self, curator):
        cap = _cap(
            name="FileReader",
            description="Reads files from disk",
            category="tool",
        )
        result = curator._classify(cap)
        assert result.tier == Tier.COMMODITY

    def test_pretrained_model_is_commodity(self, curator):
        cap = _cap(
            name="Model: yolov8n",
            model_path="/models/yolov8n.pt",
            source_type="model",
            requires_gpu=True,
            description="YOLOv8 nano pretrained",
        )
        result = curator._classify(cap)
        assert result.tier == Tier.COMMODITY
        assert any("pretrained" in r for r in result.reasons)

    def test_cloud_api_equivalent_is_commodity(self, curator):
        cap = _cap(
            name="GenericOCR",
            description="Generic OCR text extraction from images",
            category="ocr",
        )
        result = curator._classify(cap)
        assert result.tier == Tier.COMMODITY
        assert any("cloud API" in r for r in result.reasons)

    def test_config_loader_is_commodity(self, curator):
        cap = _cap(
            name="ConfigParser",
            description="Loads and parses YAML configuration files",
            category="tool",
        )
        result = curator._classify(cap)
        assert result.tier == Tier.COMMODITY

    def test_cache_manager_is_commodity(self, curator):
        cap = _cap(
            name="CacheManager",
            description="Cache management and warming utilities",
            category="optimization",
        )
        result = curator._classify(cap)
        assert result.tier == Tier.COMMODITY


# ── Uniqueness Signals ────────────────────────────────────────


class TestUniquenessSignals:
    def test_custom_model_adds_score(self, curator):
        base = _cap(name="Detector", description="detects things")
        with_model = _cap(
            name="Detector",
            description="detects things",
            model_path="/models/best_model.pt",
        )
        base_result = curator._classify(base)
        model_result = curator._classify(with_model)
        assert model_result.uniqueness_score > base_result.uniqueness_score

    def test_metrics_add_score(self, curator):
        base = _cap(name="Detector", description="comic panel detector")
        with_metrics = _cap(
            name="Detector",
            description="comic panel detector",
            eval_metrics={"mAP": 0.92},
        )
        base_result = curator._classify(base)
        metrics_result = curator._classify(with_metrics)
        assert metrics_result.uniqueness_score > base_result.uniqueness_score

    def test_multiple_metrics_add_more(self, curator):
        one_metric = _cap(
            name="X", description="comic",
            eval_metrics={"mAP": 0.9},
        )
        two_metrics = _cap(
            name="X", description="comic",
            eval_metrics={"mAP": 0.9, "IoU": 0.87},
        )
        r1 = curator._classify(one_metric)
        r2 = curator._classify(two_metrics)
        assert r2.uniqueness_score > r1.uniqueness_score

    def test_domain_keywords_add_score(self, curator):
        generic = _cap(name="Detector", description="detects objects in images")
        domain = _cap(name="Detector", description="detects comic panels and balloons")
        r_gen = curator._classify(generic)
        r_dom = curator._classify(domain)
        assert r_dom.uniqueness_score > r_gen.uniqueness_score

    def test_production_adds_score(self, curator):
        dev = _cap(name="PanelDetector", description="comic panel detection")
        prod = _cap(
            name="PanelDetector",
            description="comic panel detection",
            is_production=True,
        )
        r_dev = curator._classify(dev)
        r_prod = curator._classify(prod)
        assert r_prod.uniqueness_score > r_dev.uniqueness_score

    def test_pipeline_stage_adds_score(self, curator):
        plain = _cap(name="Detector", description="comic detection")
        stage = _cap(
            name="Stage 3: Segmentation",
            capability_id="h2v.stage3.segmentation",
            description="comic segmentation",
        )
        r_plain = curator._classify(plain)
        r_stage = curator._classify(stage)
        assert r_stage.uniqueness_score > r_plain.uniqueness_score

    def test_large_model_adds_score(self, curator):
        small = _cap(name="Model: x", source_type="model", model_size_mb=5)
        large = _cap(name="Model: x", source_type="model", model_size_mb=200)
        r_small = curator._classify(small)
        r_large = curator._classify(large)
        assert r_large.uniqueness_score > r_small.uniqueness_score


# ── Negative Signals ──────────────────────────────────────────


class TestNegativeSignals:
    def test_pretrained_reduces_score(self, curator):
        custom = _cap(name="Model: best_model", model_path="/m/best_model.pt")
        pretrained = _cap(name="Model: resnet50", model_path="/m/resnet50.pt")
        r_custom = curator._classify(custom)
        r_pre = curator._classify(pretrained)
        assert r_custom.uniqueness_score > r_pre.uniqueness_score

    def test_cloud_equivalent_reduces_score(self, curator):
        niche = _cap(name="PanelDetector", description="comic panel detection")
        cloud = _cap(name="SentimentAnalyzer", description="sentiment analysis")
        r_niche = curator._classify(niche)
        r_cloud = curator._classify(cloud)
        assert r_niche.uniqueness_score > r_cloud.uniqueness_score

    def test_undocumented_reduces_score(self, curator):
        # Both need some positive signal so scores aren't both clamped at 0
        documented = _cap(name="X", description="comic panel stuff", is_production=True)
        undocumented = _cap(name="X", description="", is_production=True)
        r_doc = curator._classify(documented)
        r_undoc = curator._classify(undocumented)
        assert r_doc.uniqueness_score > r_undoc.uniqueness_score

    def test_generic_category_reduces_score(self, curator):
        # Non-domain descriptions so the "generic category" penalty fires on tool
        domain_cat = _cap(
            name="X", description="processes data",
            category="detection", is_production=True,
        )
        generic_cat = _cap(
            name="X", description="processes data",
            category="tool", is_production=True,
        )
        r_domain = curator._classify(domain_cat)
        r_generic = curator._classify(generic_cat)
        assert r_domain.uniqueness_score > r_generic.uniqueness_score


# ── Pricing ───────────────────────────────────────────────────


class TestPricing:
    def test_unique_gpu_premium(self, curator):
        cap = _cap(
            name="YOLODetector",
            model_path="/m/best.pt",
            eval_metrics={"mAP": 0.92},
            description="comic panel detection",
            is_production=True,
            requires_gpu=True,
        )
        result = curator._classify(cap)
        assert result.suggested_price == 2.00

    def test_unique_cpu_standard(self, curator):
        cap = _cap(
            name="HeuristicDetector",
            model_path="/m/custom.pt",
            eval_metrics={"mAP": 0.85},
            description="comic panel heuristic",
            is_production=True,
            requires_gpu=False,
        )
        result = curator._classify(cap)
        assert result.suggested_price == 1.00

    def test_commodity_cheap(self, curator):
        cap = _cap(name="FileReader", description="reads files", category="tool")
        result = curator._classify(cap)
        assert result.suggested_price == 0.25


# ── Marketplace Readiness ─────────────────────────────────────


class TestMarketplaceReady:
    def test_unique_with_description_is_ready(self, curator):
        cap = _cap(
            name="ComicDetector",
            model_path="/m/best.pt",
            eval_metrics={"mAP": 0.9},
            description="8-class comic object detection",
            is_production=True,
            confidence=0.8,
        )
        result = curator._classify(cap)
        assert result.marketplace_ready is True

    def test_commodity_not_ready(self, curator):
        cap = _cap(name="FileReader", description="reads files", category="tool")
        result = curator._classify(cap)
        assert result.marketplace_ready is False

    def test_unique_no_description_not_ready(self, curator):
        cap = _cap(
            name="Model: best",
            model_path="/m/best.pt",
            eval_metrics={"mAP": 0.9},
            description="",
            source_type="model",
        )
        result = curator._classify(cap)
        assert result.marketplace_ready is False


# ── Bulk Curation ─────────────────────────────────────────────


class TestBulkCuration:
    def test_curate_sorts_into_tiers(self, curator):
        caps = [
            _cap(
                name="YOLODetector",
                model_path="/m/best.pt",
                eval_metrics={"mAP": 0.92},
                description="comic panel detection",
                is_production=True,
                requires_gpu=True,
            ),
            _cap(
                name="HeuristicPanelDetector",
                description="comic panel connected components",
                category="detection",
                is_production=True,
            ),
            _cap(name="FileReader", description="reads files", category="tool"),
            _cap(name="ConfigParser", description="config loading", category="tool"),
        ]
        result = curator.curate(caps)
        assert len(result.unique) >= 1
        assert len(result.commodity) >= 1
        assert len(result.all) == 4

    def test_listable_excludes_commodity(self, curator):
        caps = [
            _cap(
                name="ComicDetector",
                model_path="/m/best.pt",
                eval_metrics={"mAP": 0.9},
                description="8-class detection",
                is_production=True,
                confidence=0.8,
            ),
            _cap(name="FileReader", description="reads files", category="tool"),
        ]
        result = curator.curate(caps)
        listable = result.listable
        assert all(c.tier != Tier.COMMODITY for c in listable)

    def test_summary_counts(self, curator):
        caps = [
            _cap(
                name="A",
                model_path="/m/best.pt",
                eval_metrics={"mAP": 0.9},
                description="comic detection",
                is_production=True,
            ),
            _cap(name="B", description="reads files", category="tool"),
        ]
        result = curator.curate(caps)
        summary = result.summary()
        assert summary["unique"] + summary["specialized"] + summary["commodity"] == 2

    def test_empty_input(self, curator):
        result = curator.curate([])
        assert result.all == []
        assert result.listable == []

    def test_sorted_by_score(self, curator):
        caps = [
            _cap(name="Low", description="generic thing", category="tool"),
            _cap(
                name="High",
                model_path="/m/best.pt",
                eval_metrics={"mAP": 0.95},
                description="comic panel detection pipeline",
                is_production=True,
            ),
        ]
        result = curator.curate(caps)
        all_scored = result.all
        # Within each tier, higher score first
        for tier_list in [result.unique, result.specialized, result.commodity]:
            if len(tier_list) > 1:
                scores = [c.uniqueness_score for c in tier_list]
                assert scores == sorted(scores, reverse=True)


# ── Report Generation ─────────────────────────────────────────


class TestReport:
    def test_report_contains_tiers(self, curator):
        caps = [
            _cap(
                name="ComicYOLO",
                model_path="/m/best.pt",
                eval_metrics={"mAP": 0.92},
                description="comic detection",
                is_production=True,
            ),
            _cap(name="FileUtil", description="file utility", category="tool"),
        ]
        result = curator.curate(caps)
        report = curator.summary_report(result)
        assert "Unique" in report or "Specialized" in report
        assert "Commodity" in report
        assert "ComicYOLO" in report

    def test_report_empty_input(self, curator):
        result = curator.curate([])
        report = curator.summary_report(result)
        assert "Curation Report" in report


# ── Pretrained Detection ──────────────────────────────────────


class TestPretrainedDetection:
    @pytest.mark.parametrize("name", [
        "yolov8n.pt", "yolov8s", "yolov5m", "sam2_hiera_large",
        "resnet50", "bert-base", "efficientnet", "clip-vit",
        "stable-diffusion", "mask_rcnn",
    ])
    def test_known_pretrained_detected(self, name, curator):
        cap = _cap(name=f"Model: {name}", model_path=f"/m/{name}.pt")
        result = curator._classify(cap)
        assert any("pretrained" in r for r in result.reasons), \
            f"{name} should be detected as pretrained"

    @pytest.mark.parametrize("name", [
        "best_model_epoch84", "final_comic_detector",
        "panel_detector_v3", "h2v_segmenter",
    ])
    def test_custom_names_not_pretrained(self, name, curator):
        cap = _cap(name=f"Model: {name}", model_path=f"/m/{name}.pt")
        result = curator._classify(cap)
        assert not any("pretrained" in r for r in result.reasons), \
            f"{name} should NOT be detected as pretrained"


# ── Domain Keyword Detection ──────────────────────────────────


class TestDomainKeywords:
    @pytest.mark.parametrize("desc", [
        "comic panel detection",
        "manga page segmentation",
        "speech balloon extraction",
        "ink lettering recognition",
        "h2v vectorization pipeline",
    ])
    def test_domain_descriptions_detected(self, desc, curator):
        cap = _cap(name="X", description=desc)
        result = curator._classify(cap)
        assert any("domain" in r for r in result.reasons), \
            f"'{desc}' should be detected as domain-specific"

    @pytest.mark.parametrize("desc", [
        "generic image processing",
        "text summarization",
        "database management",
    ])
    def test_non_domain_not_detected(self, desc, curator):
        cap = _cap(name="X", description=desc)
        result = curator._classify(cap)
        assert not any("domain" in r for r in result.reasons), \
            f"'{desc}' should NOT be detected as domain-specific"
