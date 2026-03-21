"""Tests for codebase_scanner — capability discovery from project directories."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from multihead.codebase_scanner import (
    CodebaseScanner,
    DiscoveredCapability,
    ScanResult,
    deduplicate,
    extract_all_metrics,
    infer_category,
    project_name,
    score_all,
    slug,
)


# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture
def scanner():
    return CodebaseScanner(knowledge_store=None, claude_home="/nonexistent")


@pytest.fixture
def mock_project(tmp_path):
    """Create a mock project with various scannable artifacts."""
    proj = tmp_path / "MyProject"
    proj.mkdir()

    # CLAUDE.md with stage declarations and metrics
    claude_md = proj / "CLAUDE.md"
    claude_md.write_text(textwrap.dedent("""\
        # MyProject

        ## Pipeline Stages

        ### Stage 1: Object Detection
        YOLO-based detection with 87.72% mAP50 across 8 classes.

        ### Stage 2: Segmentation
        SAM2 segmentation achieving 97% IoU on speech balloons.

        **BalloonDetector** — detects speech balloons in comic panels.

        **File**: `scripts/run_pipeline.py` (Full pipeline runner)
    """))

    # Production Python file with capability-indicating classes
    src = proj / "src"
    src.mkdir()
    (src / "__init__.py").touch()
    (src / "detector.py").write_text(textwrap.dedent("""\
        import torch

        class PanelDetector:
            \"\"\"Detect panels in comic page images using YOLO.\"\"\"

            def detect(self, image_path):
                pass

            def batch_detect(self, paths):
                pass

            def validate(self, results):
                pass

            def export_json(self, results, output):
                pass

            def load_model(self):
                pass

            def preprocess(self, image):
                pass

            def postprocess(self, raw):
                pass

            def visualize(self, image, bboxes):
                pass

        class LayoutAnalyzer:
            \"\"\"Analyze page layout for panel ordering.\"\"\"

            def analyze(self, panels):
                pass

            def order_panels(self, panels):
                pass

            def reading_direction(self, panels):
                pass

            def validate_order(self, order):
                pass

            def build_graph(self, panels):
                pass

            def compute_flow(self, graph):
                pass

            def export_layout(self, layout):
                pass

            def detect_double_page(self, panels):
                pass
    """))

    # Non-production file (should still be found but not flagged as prod)
    scripts = proj / "scripts"
    scripts.mkdir()
    (scripts / "train_model.py").write_text(textwrap.dedent("""\
        def train_yolo(dataset_path, epochs=100):
            \"\"\"Train YOLO detection model on custom dataset.\"\"\"
            import ultralytics
            model = ultralytics.YOLO("yolov8m.pt")
            model.train(data=dataset_path, epochs=epochs)
            return model

        def evaluate_model(model_path, test_data):
            pass
    """))

    # Model checkpoint (fake but right size)
    models = proj / "models"
    models.mkdir()
    best_pt = models / "best_model.pt"
    best_pt.write_bytes(b"\x00" * (2 * 1024 * 1024))  # 2MB

    # Epoch checkpoints in same dir, same size (should be deduped)
    for i in range(10):
        (models / f"epoch{i * 10}.pt").write_bytes(b"\x00" * (2 * 1024 * 1024))

    # A different model type in another dir
    seg_models = proj / "segmentation" / "checkpoints"
    seg_models.mkdir(parents=True)
    (seg_models / "sam2_best.pth").write_bytes(b"\x00" * (5 * 1024 * 1024))

    return proj


@pytest.fixture
def mock_project_with_memory(mock_project, tmp_path):
    """Add a Claude session memory directory for the project."""
    claude_home = tmp_path / ".claude"
    # Encode the project path the way Claude does: replace / with -
    encoded_name = str(mock_project).replace("/", "-").lstrip("-")
    projects_dir = claude_home / "projects" / encoded_name / "memory"
    projects_dir.mkdir(parents=True)

    memory_md = projects_dir / "MEMORY.md"
    memory_md.write_text(textwrap.dedent("""\
        # Project Memory

        ## Key Models
        - YOLO detector: 91.85% mAP50, 8 classes
        - SAM2 segmentation: 99.58% IoU

        **TailGenerator** — generates speech balloon tails with 96.5% success rate.

        ### Stage 3: SVG Generation
        Production SVG pipeline with 99.8% success rate.
    """))

    return mock_project, claude_home


# ── CLAUDE.md Parsing ──────────────────────────────────────


class TestClaudeMdParsing:
    def test_extracts_stages(self, scanner, mock_project):
        result = scanner.scan(mock_project)
        stage_caps = [c for c in result.capabilities if "Stage" in c.name]
        assert len(stage_caps) >= 2
        names = [c.name for c in stage_caps]
        assert any("Detection" in n for n in names)
        assert any("Segmentation" in n for n in names)

    def test_extracts_metrics_near_stages(self, scanner, mock_project):
        result = scanner.scan(mock_project)
        detection_caps = [
            c for c in result.capabilities
            if "Detection" in c.name and c.source_type == "claude_md"
        ]
        assert detection_caps
        cap = detection_caps[0]
        assert cap.eval_metrics.get("mAP") or cap.eval_metrics.get("mAP50")

    def test_extracts_bold_components(self, scanner, mock_project):
        result = scanner.scan(mock_project)
        names = [c.name for c in result.capabilities]
        assert "BalloonDetector" in names

    def test_extracts_script_tools(self, scanner, mock_project):
        result = scanner.scan(mock_project)
        names = [c.name for c in result.capabilities]
        assert "run_pipeline" in names

    def test_stage_dedup(self, scanner, mock_project):
        """Same stage number shouldn't appear twice."""
        result = scanner.scan(mock_project)
        stage_nums = [
            c.name.split(":")[0] for c in result.capabilities
            if c.name.startswith("Stage") and c.source_type == "claude_md"
        ]
        assert len(stage_nums) == len(set(stage_nums))


# ── Model Checkpoint Scanning ──────────────────────────────


class TestModelScanning:
    def test_finds_model_files(self, scanner, mock_project):
        result = scanner.scan(mock_project)
        assert len(result.model_checkpoints) > 0

    def test_deduplicates_epoch_checkpoints(self, scanner, mock_project):
        result = scanner.scan(mock_project)
        # 11 checkpoints in models/ (best + epoch0-90) should collapse to 1-2
        models_dir_ckpts = [
            m for m in result.model_checkpoints
            if "models/" in m["path"] or "models\\" in m["path"]
        ]
        assert len(models_dir_ckpts) <= 3, (
            f"Expected epoch dedup, got {len(models_dir_ckpts)}: "
            f"{[m['name'] for m in models_dir_ckpts]}"
        )

    def test_keeps_different_model_types(self, scanner, mock_project):
        result = scanner.scan(mock_project)
        # Should have both the models/ checkpoints and segmentation/ checkpoint
        paths = [m["path"] for m in result.model_checkpoints]
        assert any("segmentation" in p for p in paths)

    def test_skips_tiny_files(self, scanner, mock_project):
        tiny = mock_project / "models" / "tiny.pt"
        tiny.write_bytes(b"\x00" * 100)  # 100 bytes, way under 1MB
        result = scanner.scan(mock_project)
        names = [m["name"] for m in result.model_checkpoints]
        assert "tiny.pt" not in names

    def test_model_to_capability(self, scanner, mock_project):
        result = scanner.scan(mock_project)
        model_caps = [c for c in result.capabilities if c.source_type == "model"]
        assert len(model_caps) > 0
        for cap in model_caps:
            assert cap.requires_gpu
            assert cap.model_path


# ── Python AST Scanning ────────────────────────────────────


class TestAstScanning:
    def test_finds_detector_class(self, scanner, mock_project):
        result = scanner.scan(mock_project)
        names = [c.name for c in result.capabilities if c.source_type == "code"]
        assert "PanelDetector" in names

    def test_finds_analyzer_class(self, scanner, mock_project):
        result = scanner.scan(mock_project)
        names = [c.name for c in result.capabilities if c.source_type == "code"]
        assert "LayoutAnalyzer" in names

    def test_detects_gpu_usage(self, scanner, mock_project):
        result = scanner.scan(mock_project)
        detector = next(
            (c for c in result.capabilities if c.name == "PanelDetector"), None
        )
        assert detector is not None
        assert detector.requires_gpu  # File imports torch

    def test_marks_src_as_production(self, scanner, mock_project):
        result = scanner.scan(mock_project)
        detector = next(
            (c for c in result.capabilities if c.name == "PanelDetector"), None
        )
        assert detector is not None
        assert detector.is_production  # In src/ directory

    def test_extracts_docstring(self, scanner, mock_project):
        result = scanner.scan(mock_project)
        detector = next(
            (c for c in result.capabilities if c.name == "PanelDetector"), None
        )
        assert detector is not None
        assert "YOLO" in detector.description

    def test_skips_test_files(self, scanner, tmp_path):
        proj = tmp_path / "TestProj"
        src = proj / "src"
        src.mkdir(parents=True)
        (src / "test_detector.py").write_text("class TestDetector: pass")
        result = scanner.scan(proj)
        names = [c.name for c in result.capabilities]
        assert "TestDetector" not in names

    def test_skips_private_classes(self, scanner, tmp_path):
        proj = tmp_path / "PrivProj"
        src = proj / "src"
        src.mkdir(parents=True)
        (src / "internal.py").write_text(textwrap.dedent("""\
            class _InternalDetector:
                def detect(self): pass
                def a(self): pass
                def b(self): pass
                def c(self): pass
                def d(self): pass
                def e(self): pass
                def f(self): pass
                def g(self): pass
        """))
        result = scanner.scan(proj)
        names = [c.name for c in result.capabilities]
        assert "_InternalDetector" not in names

    def test_significant_function_detection(self, scanner, tmp_path):
        proj = tmp_path / "FuncProj"
        src = proj / "src"
        src.mkdir(parents=True)
        (src / "tools.py").write_text(textwrap.dedent("""\
            def detect_objects(image, model, config, threshold=0.5):
                \"\"\"Detect objects in image.\"\"\"
                preprocessed = preprocess(image)
                features = extract_features(preprocessed)
                raw_detections = model.forward(features)
                filtered = filter_by_threshold(raw_detections, threshold)
                nms_results = apply_nms(filtered)
                postprocessed = postprocess(nms_results)
                formatted = format_output(postprocessed)
                return formatted

            def helper(x):
                return x + 1
        """))
        result = scanner.scan(proj)
        names = [c.name for c in result.capabilities if c.source_type == "code"]
        assert "detect_objects" in names
        assert "helper" not in names


# ── Session Memory ─────────────────────────────────────────


class TestSessionMemory:
    def test_loads_memory_capabilities(self, mock_project_with_memory):
        proj, claude_home = mock_project_with_memory
        scanner = CodebaseScanner(
            knowledge_store=None, claude_home=str(claude_home)
        )
        # Manually inject the memory text since Claude's path encoding
        # (replace / with -) doesn't round-trip when paths contain hyphens
        # Read from the known encoded directory in the fixture
        encoded_name = str(proj).replace("/", "-").lstrip("-")
        mem_text = (
            claude_home / "projects" / encoded_name / "memory" / "MEMORY.md"
        ).read_text()
        scanner._session_memories[str(proj)] = mem_text
        scanner._loaded_memories = True

        result = scanner.scan(proj)
        memory_caps = [c for c in result.capabilities if c.source_type == "memory"]
        assert len(memory_caps) > 0

    def test_memory_has_metrics(self, mock_project_with_memory):
        proj, claude_home = mock_project_with_memory
        scanner = CodebaseScanner(
            knowledge_store=None, claude_home=str(claude_home)
        )
        encoded_name = str(proj).replace("/", "-").lstrip("-")
        scanner._session_memories[str(proj)] = (
            claude_home / "projects" / encoded_name / "memory" / "MEMORY.md"
        ).read_text()
        scanner._loaded_memories = True

        result = scanner.scan(proj)
        memory_caps = [c for c in result.capabilities if c.source_type == "memory"]
        all_metrics = {}
        for c in memory_caps:
            all_metrics.update(c.eval_metrics)
        # Should find IoU or mAP from the memory text
        assert all_metrics


# ── Deduplication & Scoring ────────────────────────────────


class TestDedup:
    def test_dedup_keeps_richest(self):
        caps = [
            DiscoveredCapability(
                name="Detector", capability_id="proj.detection.detector",
                project="proj", source_file="a.md", source_type="claude_md",
                description="Short",
            ),
            DiscoveredCapability(
                name="Detector", capability_id="proj.detection.detector",
                project="proj", source_file="b.py", source_type="code",
                description="A much longer and more detailed description of the detector",
                is_production=True,
            ),
        ]
        deduped = deduplicate(caps)
        assert len(deduped) == 1
        assert deduped[0].is_production  # Merged from shorter entry

    def test_dedup_merges_metrics(self):
        caps = [
            DiscoveredCapability(
                name="A", capability_id="x.y.a",
                project="x", source_file="a.md", source_type="claude_md",
                eval_metrics={"mAP": 0.87},
            ),
            DiscoveredCapability(
                name="A", capability_id="x.y.a",
                project="x", source_file="a.py", source_type="code",
                eval_metrics={"IoU": 0.95},
            ),
        ]
        deduped = deduplicate(caps)
        assert len(deduped) == 1
        assert "mAP" in deduped[0].eval_metrics
        assert "IoU" in deduped[0].eval_metrics


class TestScoring:
    def test_documented_caps_score_higher(self):
        result = ScanResult(project_path="/x", project_name="x")
        result.capabilities = [
            DiscoveredCapability(
                name="A", capability_id="x.a",
                project="x", source_file="a.md", source_type="claude_md",
            ),
            DiscoveredCapability(
                name="B", capability_id="x.b",
                project="x", source_file="b.py", source_type="code",
            ),
        ]
        score_all(result)
        a = next(c for c in result.capabilities if c.name == "A")
        b = next(c for c in result.capabilities if c.name == "B")
        assert a.confidence > b.confidence

    def test_production_boosts_score(self):
        result = ScanResult(project_path="/x", project_name="x")
        result.capabilities = [
            DiscoveredCapability(
                name="A", capability_id="x.a",
                project="x", source_file="a.py", source_type="code",
                is_production=True,
            ),
            DiscoveredCapability(
                name="B", capability_id="x.b",
                project="x", source_file="b.py", source_type="code",
            ),
        ]
        score_all(result)
        a = next(c for c in result.capabilities if c.name == "A")
        b = next(c for c in result.capabilities if c.name == "B")
        assert a.confidence > b.confidence


# ── Output Formats ─────────────────────────────────────────


class TestOutputFormats:
    def test_to_claim(self):
        cap = DiscoveredCapability(
            name="PanelDetector",
            capability_id="h2v.detection.panel_detector",
            project="h2v",
            source_file="src/detector.py",
            source_type="code",
            description="Detects panels in comics",
            category="detection",
            eval_metrics={"mAP": 0.8772},
            is_production=True,
            requires_gpu=True,
        )
        claim = cap.to_claim()
        assert claim["claim_key"] == "capability.h2v.detection.panel_detector"
        assert claim["claim_type"] == "fact"
        assert "PanelDetector" in claim["statement"]
        assert "mAP" in claim["statement"]
        assert "GPU required" in claim["statement"]
        assert "Production-ready" in claim["statement"]

    def test_to_listing(self):
        cap = DiscoveredCapability(
            name="PanelDetector",
            capability_id="h2v.detection.panel_detector",
            project="h2v",
            source_file="src/detector.py",
            source_type="code",
            category="detection",
            requires_gpu=True,
            confidence=0.85,
        )
        listing = cap.to_listing()
        assert listing["capability_id"] == "com.multihead.h2v.detection.panel_detector"
        assert listing["pricing_model"] == "per_call"
        assert listing["unit_price"] == 1.00  # GPU = $1.00
        assert listing["quality_score"] == 0.85


# ── Category Inference ─────────────────────────────────────


class TestCategoryInference:
    @pytest.mark.parametrize("text,expected", [
        ("YOLODetector", "detection"),
        ("SAM2Segmenter", "segmentation"),
        ("SVGGenerator", "vectorization"),
        ("BalloonPlacer", "layout"),
        ("OCRExtractor", "ocr"),
        ("TrainingPipeline", "training"),
        ("DataValidator", "validation"),
        ("InkCleaner", "processing"),
        ("FormatConverter", "conversion"),
        ("SomethingRandom", "tool"),
    ])
    def test_infer_category(self, text, expected):
        assert infer_category(text) == expected


# ── Metric Extraction ──────────────────────────────────────


class TestMetricExtraction:
    def test_extract_map(self):
        text = "YOLO achieved 87.72% mAP50 on test set"
        metrics = extract_all_metrics(text)
        assert "mAP" in metrics
        assert abs(metrics["mAP"] - 0.8772) < 0.01

    def test_extract_iou(self):
        text = "Segmentation: 97% IoU on balloons"
        metrics = extract_all_metrics(text)
        assert "IoU" in metrics
        assert abs(metrics["IoU"] - 0.97) < 0.01

    def test_extract_success_rate(self):
        text = "Tail generation: 96.5% success rate"
        metrics = extract_all_metrics(text)
        assert "success_rate" in metrics
        assert abs(metrics["success_rate"] - 96.5) < 0.1

    def test_extract_multiple_metrics(self):
        text = "mAP: 0.87, IoU: 0.95, precision: 0.92"
        metrics = extract_all_metrics(text)
        assert len(metrics) >= 3

    def test_no_metrics_returns_empty(self):
        metrics = extract_all_metrics("just some text with no numbers")
        assert metrics == {}


# ── Auto-discover Projects ─────────────────────────────────


class TestAutoDiscover:
    def test_auto_discover_from_claude_sessions(self, tmp_path):
        claude_home = tmp_path / ".claude"
        # Simulate encoded directory names
        projects = claude_home / "projects"
        (projects / "-mnt-d-DevD-Multihead").mkdir(parents=True)

        # The decoded path won't exist in tmp, so it won't be returned
        scanner = CodebaseScanner(claude_home=str(claude_home))
        paths = scanner.auto_discover_projects()
        # Paths that don't exist on disk are filtered out
        assert isinstance(paths, list)


# ── Summary Output ─────────────────────────────────────────


class TestSummary:
    def test_summary_contains_project_names(self, scanner, mock_project):
        result = scanner.scan(mock_project)
        summary = scanner.summary([result])
        assert "Capability Discovery Report" in summary
        assert result.project_name in summary

    def test_summary_empty_results(self, scanner):
        summary = scanner.summary([])
        assert "0 capabilities" in summary or "Total" in summary

    def test_summary_shows_categories(self, scanner, mock_project):
        result = scanner.scan(mock_project)
        summary = scanner.summary([result])
        # Should show at least detection category
        assert "Detection" in summary or "detection" in summary.lower()


# ── Full Integration Scan ──────────────────────────────────


class TestIntegrationScan:
    def test_full_scan_produces_results(self, scanner, mock_project):
        result = scanner.scan(mock_project)
        assert result.project_name
        assert result.capabilities
        assert result.model_checkpoints
        assert result.scan_duration_s >= 0
        assert result.files_scanned > 0

    def test_scan_all(self, scanner, mock_project):
        results = scanner.scan_all([mock_project])
        assert len(results) == 1
        assert results[0].capabilities

    def test_scan_nonexistent_dir(self, scanner):
        result = scanner.scan("/nonexistent/path/that/does/not/exist")
        assert result.errors

    def test_scan_empty_dir(self, scanner, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        result = scanner.scan(empty)
        assert len(result.capabilities) == 0
        assert len(result.model_checkpoints) == 0


# ── Slug & Project Name ────────────────────────────────────


class TestHelpers:
    def test_slug(self):
        assert slug("PanelDetector") == "panel_detector"
        assert slug("SAM2-Model") == "sam2_model"
        assert slug("my cool tool") == "my_cool_tool"

    def test_project_name_mappings(self):
        # Mappings removed (were user-specific). Now returns folder name as-is.
        assert project_name(Path("/tmp/MyProject")) == "myproject"
        assert project_name(Path("/tmp/MyTools")) == "my_tools"
        assert project_name(Path("/tmp/Vibebots")) == "vibebots"
        assert project_name(Path("/tmp/Multihead")) == "multihead"
