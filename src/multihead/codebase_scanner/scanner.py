"""CodebaseScanner — discover unique capabilities from project directories.

Scans registered project directories to find capabilities that are unique
to THIS MultiHead instance (trained models, domain pipelines, specialized
tools) vs commodity capabilities that ship with every MultiHead install.

Feeds into capability_discovery.py and the marketplace listing pipeline.

Sources scanned (in priority order):
1. CLAUDE.md / MEMORY.md — session-declared capabilities and metrics
2. Model checkpoints — .pt, .pth, .onnx, .safetensors files
3. Production Python code — classes/functions with capability indicators
4. Pipeline configs — YAML/JSON workflow definitions
5. Git history — commit counts for maturity scoring
6. Knowledge.db — cross-reference with existing claims

Design constraints:
- Must handle 35K+ Python files across multiple projects efficiently
- Skips venvs, __pycache__, node_modules, .git
- Limits AST analysis to top ~300 lines per file
- Prioritizes production/ directories over sandbox/experimental
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from .analyzers import find_models, model_to_capability, scan_python_files
from .models import DiscoveredCapability, ScanResult
from .parsers import (
    deduplicate,
    extract_all_metrics,
    parse_capability_doc,
    project_name,
    score_all,
)

logger = logging.getLogger(__name__)


class CodebaseScanner:
    """Scans project directories to discover unique capabilities.

    Usage:
        scanner = CodebaseScanner(knowledge_store=ks)

        # Scan a single project
        result = scanner.scan("/path/to/your/project")

        # Scan all known projects
        results = scanner.scan_all([
            "/path/to/project1",
            "/path/to/project2",
        ])

        # Get marketplace listings
        for r in results:
            for cap in r.capabilities:
                print(cap.to_listing())
    """

    def __init__(
        self,
        knowledge_store: Any = None,
        claude_home: str = "~/.claude",
    ) -> None:
        self._ks = knowledge_store
        self._claude_home = Path(claude_home).expanduser()
        self._session_memories: dict[str, str] = {}
        self._loaded_memories = False

    def scan(self, project_path: str | Path) -> ScanResult:
        """Scan a single project directory for capabilities."""
        start = time.time()
        project_path = Path(project_path)

        if not project_path.is_dir():
            return ScanResult(
                project_path=str(project_path),
                project_name=project_path.name,
                errors=[f"Not a directory: {project_path}"],
            )

        proj_name = project_name(project_path)
        result = ScanResult(
            project_path=str(project_path),
            project_name=proj_name,
        )

        # 1. Parse CLAUDE.md files — highest signal, structured descriptions
        result.capabilities.extend(
            self._scan_claude_md(project_path, proj_name)
        )

        # 2. Load session memory for this project
        memory_caps = self._scan_session_memory(project_path, proj_name)
        result.capabilities.extend(memory_caps)

        # 3. Find model checkpoints
        result.model_checkpoints = find_models(project_path)
        for ckpt in result.model_checkpoints:
            result.capabilities.append(model_to_capability(ckpt, proj_name))

        # 4. Scan Python files (production first, then src)
        py_caps, files_scanned = scan_python_files(project_path, proj_name)
        result.capabilities.extend(py_caps)
        result.files_scanned = files_scanned

        # 5. Enrich from knowledge.db
        if self._ks:
            self._enrich_from_knowledge(result)

        # 6. Deduplicate and score
        result.capabilities = deduplicate(result.capabilities)
        score_all(result)

        result.scan_duration_s = round(time.time() - start, 2)
        return result

    def scan_all(self, project_paths: list[str | Path]) -> list[ScanResult]:
        """Scan multiple project directories."""
        self._load_all_memories()

        results = []
        for path in project_paths:
            try:
                result = self.scan(path)
                if result.capabilities or result.model_checkpoints:
                    results.append(result)
                    logger.info(
                        "Scanned %s: %d capabilities, %d models (%.1fs)",
                        result.project_name,
                        len(result.capabilities),
                        len(result.model_checkpoints),
                        result.scan_duration_s,
                    )
            except Exception as e:
                logger.warning("Failed to scan %s: %s", path, e)
                results.append(ScanResult(
                    project_path=str(path),
                    project_name=Path(path).name,
                    errors=[str(e)],
                ))

        return results

    def auto_discover_projects(self) -> list[Path]:
        """Discover project paths from Claude session directories."""
        projects_dir = self._claude_home / "projects"
        if not projects_dir.is_dir():
            return []

        paths = []
        for d in sorted(projects_dir.iterdir()):
            if not d.is_dir():
                continue
            decoded = "/" + d.name.replace("-", "/").lstrip("/")
            actual = Path(decoded)
            if actual.is_dir():
                paths.append(actual)

        return paths

    # ------------------------------------------------------------------
    # 1. CLAUDE.md scanning
    # ------------------------------------------------------------------

    def _scan_claude_md(
        self, project_path: Path, proj_name: str
    ) -> list[DiscoveredCapability]:
        """Extract capabilities from CLAUDE.md files."""
        caps: list[DiscoveredCapability] = []

        # Find all CLAUDE.md files (including subdirectories)
        claude_files = list(project_path.glob("**/CLAUDE.md"))
        # Limit to first 10 to avoid over-scanning
        for cf in claude_files[:10]:
            try:
                text = cf.read_text(encoding="utf-8", errors="replace")
                caps.extend(parse_capability_doc(
                    text, str(cf.relative_to(project_path)),
                    proj_name, "claude_md"
                ))
            except Exception as e:
                logger.debug("Error reading %s: %s", cf, e)

        return caps

    # ------------------------------------------------------------------
    # 2. Session memory scanning
    # ------------------------------------------------------------------

    def _scan_session_memory(
        self, project_path: Path, proj_name: str
    ) -> list[DiscoveredCapability]:
        """Extract capabilities from session MEMORY.md for this project."""
        self._load_all_memories()

        memory_text = self._session_memories.get(str(project_path), "")
        if not memory_text:
            # Try fuzzy match on project name
            for key, val in self._session_memories.items():
                if proj_name.lower() in key.lower():
                    memory_text = val
                    break

        if not memory_text:
            return []

        return parse_capability_doc(
            memory_text, "MEMORY.md", proj_name, "memory"
        )

    def _load_all_memories(self) -> None:
        """Load all session MEMORY.md files."""
        if self._loaded_memories:
            return
        self._loaded_memories = True

        projects_dir = self._claude_home / "projects"
        if not projects_dir.is_dir():
            return

        for d in projects_dir.iterdir():
            if not d.is_dir():
                continue
            mem_file = d / "memory" / "MEMORY.md"
            if mem_file.exists():
                decoded = "/" + d.name.replace("-", "/").lstrip("/")
                try:
                    self._session_memories[decoded] = mem_file.read_text(
                        encoding="utf-8", errors="replace"
                    )
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Knowledge enrichment
    # ------------------------------------------------------------------

    def _enrich_from_knowledge(self, result: ScanResult) -> None:
        """Cross-reference with knowledge.db claims."""
        if not self._ks:
            return

        for cap in result.capabilities:
            try:
                claims = self._ks.search_claims_fts(cap.name, limit=3)
                for claim in claims:
                    stmt = getattr(claim, "statement", str(claim))
                    metrics = extract_all_metrics(stmt)
                    if metrics:
                        cap.eval_metrics.update(metrics)
                        cap.confidence = min(1.0, cap.confidence + 0.1)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Summary output
    # ------------------------------------------------------------------

    def summary(self, results: list[ScanResult]) -> str:
        """Human-readable summary of all discoveries."""
        lines = ["# Capability Discovery Report", ""]

        total_caps = 0
        total_models = 0

        for result in results:
            if not result.capabilities and not result.model_checkpoints:
                continue

            lines.append(f"## {result.project_name}")
            lines.append(f"Path: `{result.project_path}`")
            lines.append(
                f"Scanned {result.files_scanned} files in "
                f"{result.scan_duration_s}s"
            )

            # Group by category
            by_category: dict[str, list[DiscoveredCapability]] = {}
            for cap in result.capabilities:
                by_category.setdefault(cap.category, []).append(cap)

            for category, caps in sorted(by_category.items()):
                lines.append(f"\n### {category.title()} ({len(caps)})")
                for cap in sorted(caps, key=lambda c: -c.confidence):
                    flags = []
                    if cap.requires_gpu:
                        flags.append("GPU")
                    if cap.is_production:
                        flags.append("PROD")
                    if cap.has_tests:
                        flags.append("TESTED")
                    if cap.eval_metrics:
                        metrics_str = ", ".join(
                            f"{k}={v}" for k, v in cap.eval_metrics.items()
                        )
                        flags.append(metrics_str)

                    flag_str = f" [{', '.join(flags)}]" if flags else ""
                    lines.append(
                        f"- **{cap.name}** (conf={cap.confidence:.2f}){flag_str}"
                    )
                    if cap.description:
                        lines.append(f"  {cap.description[:120]}")

                total_caps += len(caps)

            if result.model_checkpoints:
                lines.append(f"\n### Model Checkpoints ({len(result.model_checkpoints)})")
                # Show top 10 by size
                for ckpt in result.model_checkpoints[:10]:
                    lines.append(
                        f"- {ckpt['name']} ({ckpt['model_type']}, "
                        f"{ckpt['size_mb']}MB)"
                    )
                if len(result.model_checkpoints) > 10:
                    lines.append(
                        f"  ... and {len(result.model_checkpoints) - 10} more"
                    )
                total_models += len(result.model_checkpoints)

            lines.append("")

        unique = sum(
            1 for r in results for c in r.capabilities
            if c.confidence >= 0.5
        )

        lines.append("---")
        lines.append(
            f"**Total**: {total_caps} capabilities discovered, "
            f"{total_models} model checkpoints, "
            f"{unique} marketplace-ready (conf >= 0.5)"
        )

        return "\n".join(lines)
