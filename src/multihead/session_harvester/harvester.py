"""Core SessionHarvester class — discovery, harvesting, and status reporting."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evolution import save_snapshot, track_evolution
from .extraction import deposit_claims, extract_claims
from .models import HarvestResult, ProjectInfo

logger = logging.getLogger(__name__)


class SessionHarvester:
    """Scans Claude project folders and harvests knowledge into knowledge.db."""

    def __init__(
        self,
        knowledge_store: Any,
        claude_home: str | Path = "~/.claude",
        data_dir: str | Path | None = None,
        max_claims_per_project: int = 100,
    ) -> None:
        self._ks = knowledge_store
        self._claude_home = Path(claude_home).expanduser()
        self._data_dir = Path(data_dir) if data_dir else Path.home() / ".multihead"
        self._max_claims_per_project = max_claims_per_project
        self._manifest_path = self._data_dir / "sessions" / "harvest_manifest.json"
        self._snapshots_dir = self._data_dir / "sessions" / "snapshots"

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def scan_projects(self) -> list[ProjectInfo]:
        """Scan ~/.claude/projects/ for all project folders."""
        projects_dir = self._claude_home / "projects"
        if not projects_dir.exists():
            logger.warning("Claude projects dir not found: %s", projects_dir)
            return []

        projects: list[ProjectInfo] = []
        for entry in sorted(projects_dir.iterdir()):
            if not entry.is_dir():
                continue

            info = ProjectInfo(
                name=entry.name,
                path=entry,
                decoded_path=self._decode_folder_name(entry.name),
            )

            # Check for memory files
            memory_md = entry / "memory" / "MEMORY.md"
            claude_md = entry / "CLAUDE.md"

            if memory_md.exists():
                info.has_memory = True
                info.memory_files.append(memory_md)

            if claude_md.exists():
                info.has_claude_md = True
                info.memory_files.append(claude_md)

            # Also check for additional memory files in memory/ dir
            memory_dir = entry / "memory"
            if memory_dir.exists():
                for md_file in sorted(memory_dir.glob("*.md")):
                    if md_file not in info.memory_files:
                        info.memory_files.append(md_file)

            if info.memory_files:
                projects.append(info)

        return projects

    def get_manifest(self) -> dict[str, Any]:
        """Load the harvest manifest (tracks what's been harvested)."""
        if self._manifest_path.exists():
            try:
                return json.loads(self._manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load manifest: %s", e)
        return {"last_full_scan": None, "projects": {}}

    # ------------------------------------------------------------------
    # Harvesting
    # ------------------------------------------------------------------

    def harvest_all(self, on_progress=None) -> HarvestResult:
        """Full scan + harvest cycle.

        Args:
            on_progress: Optional callback ``(event_dict) -> None`` for progress reporting.
        """
        t0 = time.monotonic()
        result = HarvestResult()
        manifest = self.get_manifest()

        projects = self.scan_projects()
        result.projects_scanned = len(projects)
        total = len(projects)

        for i, project in enumerate(projects):
            if on_progress:
                on_progress({
                    "event": "project_start",
                    "project": project.name,
                    "project_index": i,
                    "project_total": total,
                })
            files_in_project = len(project.memory_files)
            try:
                harvested, evolutions = self._harvest_project(project, manifest)
                if harvested > 0:
                    result.projects_harvested += 1
                    result.claims_deposited += harvested
                    result.evolution_events += evolutions
                else:
                    result.projects_skipped += 1
            except Exception as e:
                msg = f"{project.name}: {e}"
                result.errors.append(msg)
                logger.warning("Harvest error for %s: %s", project.name, e)
            if on_progress:
                on_progress({
                    "event": "project_done",
                    "project": project.name,
                    "project_index": i,
                    "project_total": total,
                    "file_count": files_in_project,
                })

        # Update manifest timestamp
        manifest["last_full_scan"] = datetime.now(timezone.utc).isoformat()
        self._save_manifest(manifest)

        result.duration_seconds = round(time.monotonic() - t0, 2)
        logger.info(
            "Harvest complete: %d scanned, %d harvested, %d claims, %.1fs",
            result.projects_scanned, result.projects_harvested,
            result.claims_deposited, result.duration_seconds,
        )
        return result

    def _harvest_project(
        self, project: ProjectInfo, manifest: dict[str, Any],
    ) -> tuple[int, int]:
        """Harvest a single project. Returns (claims_deposited, evolution_events)."""
        proj_manifest = manifest.setdefault("projects", {}).setdefault(
            project.name, {"scope_id": "", "files": {}, "claim_count": 0},
        )

        scope_id = self._derive_scope_id(project)
        proj_manifest["scope_id"] = scope_id

        total_claims = 0
        evolution_count = 0

        for file_path in project.memory_files:
            rel_path = str(file_path.relative_to(project.path))
            file_hash = self._file_hash(file_path)

            # Check if file changed since last harvest
            prev = proj_manifest["files"].get(rel_path, {})
            if prev.get("sha256") == file_hash:
                continue  # unchanged

            # Read and extract claims
            try:
                content = file_path.read_text(encoding="utf-8")
            except OSError as e:
                logger.warning("Cannot read %s: %s", file_path, e)
                continue

            if not content.strip():
                continue

            # Track evolution: diff against previous snapshot
            had_previous = prev.get("sha256") is not None
            if had_previous:
                evo = track_evolution(
                    project, file_path, rel_path, content,
                    scope_id, self._snapshots_dir, self._ks,
                )
                if evo:
                    evolution_count += 1

            # Save current snapshot for next diff
            save_snapshot(self._snapshots_dir, project.name, rel_path, content)

            # Determine confidence based on file type
            is_memory = "MEMORY" in file_path.name or file_path.parent.name == "memory"
            confidence = 0.75 if is_memory else 0.65

            claims = extract_claims(content, scope_id, project, file_path, confidence)

            # Cap claims per project
            remaining = self._max_claims_per_project - total_claims
            if remaining <= 0:
                break
            claims = claims[:remaining]

            # Deposit
            deposited = deposit_claims(claims, self._ks)
            total_claims += deposited

            # Update manifest for this file
            proj_manifest["files"][rel_path] = {
                "sha256": file_hash,
                "harvested_at": datetime.now(timezone.utc).isoformat(),
                "claims_extracted": deposited,
            }

        proj_manifest["claim_count"] = total_claims
        return total_claims, evolution_count

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_folder_name(name: str) -> str:
        """Convert folder name back to a readable path.

        e.g. '-home-user-projects-myrepo' -> '/home/user/projects/myrepo'
        """
        if name.startswith("-"):
            return "/" + name[1:].replace("-", "/")
        return name

    @staticmethod
    def _derive_scope_id(project: ProjectInfo) -> str:
        """Derive a short scope_id from the project folder name.

        e.g. '-home-user-projects-myapp' -> 'myapp'
             '-workspace-frontend' -> 'frontend'
        """
        name = project.name.lower()
        # Remove common prefixes
        for prefix in ("-mnt-d-devd-", "-mnt-c-dev-", "-mnt-d-", "-mnt-c-"):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break

        # Use last meaningful segment
        parts = [p for p in name.split("-") if p]
        if not parts:
            return "unknown"

        # Special cases
        if "transmediaengine" in parts:
            # Use subdirectory if present
            idx = parts.index("transmediaengine")
            remaining = parts[idx + 1:]
            if remaining:
                return "-".join(remaining[-2:])  # last 2 segments
            return "transmedia"

        return parts[-1] if parts else "unknown"

    @staticmethod
    def _file_hash(path: Path) -> str:
        """SHA-256 hash of a file's contents."""
        h = hashlib.sha256()
        try:
            h.update(path.read_bytes())
        except OSError:
            return ""
        return h.hexdigest()

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        """Persist the harvest manifest."""
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Status / reporting
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Return current harvester status."""
        manifest = self.get_manifest()
        projects = self.scan_projects()

        proj_statuses = []
        for p in projects:
            proj_man = manifest.get("projects", {}).get(p.name, {})
            proj_statuses.append({
                "name": p.name,
                "decoded_path": p.decoded_path,
                "scope_id": proj_man.get("scope_id", self._derive_scope_id(p)),
                "has_memory": p.has_memory,
                "has_claude_md": p.has_claude_md,
                "file_count": len(p.memory_files),
                "claim_count": proj_man.get("claim_count", 0),
                "last_harvested": next(
                    (f.get("harvested_at") for f in proj_man.get("files", {}).values()),
                    None,
                ),
            })

        return {
            "projects_found": len(projects),
            "last_full_scan": manifest.get("last_full_scan"),
            "total_claims": sum(p["claim_count"] for p in proj_statuses),
            "projects": proj_statuses,
        }
