"""Export and import bundles for runs and projects."""

from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multihead.artifact_store import ArtifactStore
from multihead.event_store import EventStore


class BundleExporter:
    """Export runs or projects into portable zip bundles."""

    def __init__(self, event_store: EventStore, artifact_store: ArtifactStore, runs_dir: Path) -> None:
        self.events = event_store
        self.artifacts = artifact_store
        self.runs_dir = runs_dir

    def export_run(self, run_id: str, output_path: Path) -> Path:
        """Export a single run with its events and artifacts into a zip."""
        events = self.events.read_events(run_id)
        if not events:
            raise ValueError(f"No events found for run: {run_id}")

        zip_path = output_path / f"{run_id}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Write events
            events_data = [e.model_dump(mode="json") for e in events]
            zf.writestr(f"{run_id}/events.json", json.dumps(events_data, indent=2, default=str))

            # Write run artifacts
            run_dir = self.runs_dir / run_id
            if run_dir.exists():
                for f in run_dir.rglob("*"):
                    if f.is_file():
                        arcname = f"{run_id}/{f.relative_to(run_dir)}"
                        zf.write(f, arcname)

            # Write manifest
            manifest = {
                "type": "run",
                "run_id": run_id,
                "event_count": len(events),
                "exported_at": datetime.now(timezone.utc).isoformat(),
            }
            zf.writestr(f"{run_id}/manifest.json", json.dumps(manifest, indent=2))

        return zip_path

    def export_project(self, output_path: Path, include_artifacts: bool = True) -> Path:
        """Export the entire project (all runs, artifacts, config)."""
        zip_path = output_path / "multihead_project.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # All runs
            runs = self.events.list_runs()
            runs_data = []
            for r in runs:
                rid = r.get("run_id", "")
                events = self.events.read_events(rid)
                runs_data.append({
                    "run_id": rid,
                    "events": [e.model_dump(mode="json") for e in events],
                })
            zf.writestr("runs.json", json.dumps(runs_data, indent=2, default=str))

            # Artifacts
            if include_artifacts:
                all_artifacts = self.artifacts.list_all()
                for art in all_artifacts:
                    data = self.artifacts.fetch(art["artifact_id"])
                    if data:
                        zf.writestr(f"artifacts/{art['artifact_id']}", data)

            # Manifest
            manifest = {
                "type": "project",
                "run_count": len(runs),
                "exported_at": datetime.now(timezone.utc).isoformat(),
            }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        return zip_path


class BundleImporter:
    """Import bundles into the local store."""

    def __init__(self, artifact_store: ArtifactStore, runs_dir: Path) -> None:
        self.artifacts = artifact_store
        self.runs_dir = runs_dir

    def import_bundle(self, zip_path: Path) -> dict[str, Any]:
        """Import a bundle from a zip file."""
        if not zip_path.exists():
            raise FileNotFoundError(f"Bundle not found: {zip_path}")

        with zipfile.ZipFile(zip_path, "r") as zf:
            # Read manifest
            manifest_candidates = [n for n in zf.namelist() if n.endswith("manifest.json")]
            if not manifest_candidates:
                raise ValueError("No manifest.json found in bundle")

            manifest = json.loads(zf.read(manifest_candidates[0]))
            bundle_type = manifest.get("type", "unknown")

            if bundle_type == "run":
                return self._import_run(zf, manifest)
            elif bundle_type == "project":
                return self._import_project(zf, manifest)
            else:
                raise ValueError(f"Unknown bundle type: {bundle_type}")

    def _import_run(self, zf: zipfile.ZipFile, manifest: dict) -> dict[str, Any]:
        """Import a single run bundle."""
        run_id = manifest.get("run_id", "unknown")
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        imported_files = 0
        for name in zf.namelist():
            if name.startswith(f"{run_id}/") and not name.endswith("/"):
                target = self.runs_dir / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(name))
                imported_files += 1

        return {"type": "run", "run_id": run_id, "files_imported": imported_files}

    def _import_project(self, zf: zipfile.ZipFile, manifest: dict) -> dict[str, Any]:
        """Import a project bundle."""
        imported_artifacts = 0
        for name in zf.namelist():
            if name.startswith("artifacts/") and not name.endswith("/"):
                data = zf.read(name)
                artifact_id = name.replace("artifacts/", "")
                self.artifacts.store(data, name=artifact_id)
                imported_artifacts += 1

        return {
            "type": "project",
            "artifacts_imported": imported_artifacts,
            "run_count": manifest.get("run_count", 0),
        }
