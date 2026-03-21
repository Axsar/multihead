"""Tests for bundle export/import."""

import json
import zipfile

import pytest

from multihead.artifact_store import ArtifactStore
from multihead.bundle import BundleExporter, BundleImporter
from multihead.event_store import EventStore


@pytest.fixture
def stores(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    artifact_store = ArtifactStore(tmp_path / "artifacts", tmp_path / "artifacts.db")
    event_store = EventStore(runs_dir, tmp_path / "state.db")
    return event_store, artifact_store, runs_dir


class TestBundleExporter:
    def test_export_run_no_events(self, stores, tmp_path):
        es, art, runs_dir = stores
        exporter = BundleExporter(es, art, runs_dir)
        with pytest.raises(ValueError, match="No events"):
            exporter.export_run("nonexistent", tmp_path / "out")

    def test_export_project(self, stores, tmp_path):
        es, art, runs_dir = stores
        # Store an artifact
        art.store(b"hello world", name="test.txt")

        exporter = BundleExporter(es, art, runs_dir)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        path = exporter.export_project(out_dir)

        assert path.exists()
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            assert "manifest.json" in names
            assert "runs.json" in names


class TestBundleImporter:
    def test_import_project(self, stores, tmp_path):
        es, art, runs_dir = stores

        # Create a minimal project bundle
        zip_path = tmp_path / "test_bundle.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            manifest = {"type": "project", "run_count": 0}
            zf.writestr("manifest.json", json.dumps(manifest))
            zf.writestr("artifacts/sha256:abc", b"test data")

        importer = BundleImporter(art, runs_dir)
        result = importer.import_bundle(zip_path)
        assert result["type"] == "project"

    def test_import_run(self, stores, tmp_path):
        es, art, runs_dir = stores

        zip_path = tmp_path / "run_bundle.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            manifest = {"type": "run", "run_id": "run_test123"}
            zf.writestr("run_test123/manifest.json", json.dumps(manifest))
            zf.writestr("run_test123/events.json", "[]")

        importer = BundleImporter(art, runs_dir)
        result = importer.import_bundle(zip_path)
        assert result["type"] == "run"
        assert result["run_id"] == "run_test123"
        assert result["files_imported"] >= 2

    def test_import_missing_file(self, stores, tmp_path):
        es, art, runs_dir = stores
        importer = BundleImporter(art, runs_dir)
        with pytest.raises(FileNotFoundError):
            importer.import_bundle(tmp_path / "nonexistent.zip")

    def test_import_no_manifest(self, stores, tmp_path):
        es, art, runs_dir = stores
        zip_path = tmp_path / "bad_bundle.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("data.txt", "hello")

        importer = BundleImporter(art, runs_dir)
        with pytest.raises(ValueError, match="No manifest"):
            importer.import_bundle(zip_path)
