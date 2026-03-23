"""Context-building mixin — knowledge RAG, codebase context, system prompt."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from multihead.subprocess_utils import no_window_flags
from .prompts import SHELL_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class ContextMixin:
    """Mixin providing context-building methods for the Shell class.

    Expects the host class to provide:
    - self.ks (KnowledgeStore)
    - self._codebase_ctx_cache (str | None)
    """

    # -------------------------------------------------------------------
    # System prompt
    # -------------------------------------------------------------------

    def build_system_prompt(self) -> str:
        """Build PLUR-enhanced system prompt with live stats and dynamic paths."""
        claim_count = self._count_claims()

        # Resolve dynamic paths
        data_dir = str(getattr(self.ks, "db_path", "~/.multihead").parent) if self.ks else "~/.multihead"
        knowledge_db = str(getattr(self.ks, "db_path", "~/.multihead/knowledge.db")) if self.ks else "~/.multihead/knowledge.db"
        config_dir = str(self._config_dir) if hasattr(self, "_config_dir") else "config"

        # GPU info
        gpu_info = "no GPU detected"
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory // (1024**2)
                gpu_info = f"{name}, {vram} MB VRAM"
        except ImportError:
            pass

        return SHELL_SYSTEM_PROMPT.format(
            claim_count=f"{claim_count:,}",
            data_dir=data_dir,
            knowledge_db=knowledge_db,
            config_dir=config_dir,
            gpu_info=gpu_info,
        )

    # -------------------------------------------------------------------
    # Knowledge RAG
    # -------------------------------------------------------------------

    def _build_knowledge_context(self, user_message: str) -> str:
        """Query knowledge store for claims relevant to the user's message.

        Uses FTS5 full-text search for O(log n) indexed lookup with
        relevance ranking, falling back to LIKE-based search if FTS5
        is unavailable.
        """
        if not self.ks:
            return ""

        try:
            # FTS5 search — filter out low-confidence noise and ancient claims
            results = self.ks.search_claims_fts(
                user_message, limit=8, min_confidence=0.3, max_age_days=90,
            )

            if not results:
                return ""

            lines = ["[Knowledge context from past sessions]"]
            for claim_key, statement, confidence in results:
                lines.append(f"- [{confidence:.0%}] {statement[:200]}")
            return "\n".join(lines)

        except Exception as e:
            logger.debug("Knowledge context error: %s", e)
            return ""

    # -------------------------------------------------------------------
    # Codebase context
    # -------------------------------------------------------------------

    def _build_codebase_context(self) -> str:
        """Build a codebase context block for Claude brain sessions.

        Gathers project identity, file tree, recent git commits, and
        project docs (CLAUDE.md, MEMORY_MULTIHEAD.md). Cached for the
        session since the codebase doesn't change mid-conversation.
        """
        if self._codebase_ctx_cache is not None:
            return self._codebase_ctx_cache

        try:
            self._codebase_ctx_cache = self._gather_codebase_context()
        except Exception as e:
            logger.debug("Codebase context error: %s", e)
            self._codebase_ctx_cache = ""
        return self._codebase_ctx_cache

    def _gather_codebase_context(self) -> str:
        """Gather codebase context via subprocess calls."""
        import subprocess

        parts: list[str] = ["\n\n## Codebase Context"]

        # --- Project identity ---
        try:
            repo_root = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, timeout=5,
                creationflags=no_window_flags(),
            ).stdout.strip()
        except Exception:
            repo_root = ""

        if not repo_root:
            return ""  # not in a git repo

        try:
            branch = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True, text=True, timeout=5,
                cwd=repo_root,
                creationflags=no_window_flags(),
            ).stdout.strip()
        except Exception:
            branch = "unknown"

        parts.append(f"\n**Repo**: `{repo_root}` (branch: `{branch}`)")

        # --- Top-level listing ---
        try:
            top_items = subprocess.run(
                ["ls", "-1", repo_root],
                capture_output=True, text=True, timeout=5,
                creationflags=no_window_flags(),
            ).stdout.strip()
            if top_items:
                parts.append(f"\n**Top-level**:\n```\n{top_items}\n```")
        except Exception:
            pass

        # --- Source file tree ---
        src_dir = Path(repo_root) / "src"
        if src_dir.is_dir():
            try:
                result = subprocess.run(
                    ["find", str(src_dir), "-type", "f", "-name", "*.py"],
                    capture_output=True, text=True, timeout=5,
                    creationflags=no_window_flags(),
                )
                files = result.stdout.strip().splitlines()
                # Show relative paths, limit to 40 files
                rel_files = sorted(
                    f.replace(repo_root + "/", "") for f in files
                )[:40]
                if rel_files:
                    tree = "\n".join(rel_files)
                    parts.append(f"\n**Source files** ({len(files)} total):\n```\n{tree}\n```")
            except Exception:
                pass

        # --- Recent git commits ---
        try:
            log = subprocess.run(
                ["git", "log", "--oneline", "-10"],
                capture_output=True, text=True, timeout=5,
                cwd=repo_root,
                creationflags=no_window_flags(),
            ).stdout.strip()
            if log:
                parts.append(f"\n**Recent commits**:\n```\n{log}\n```")
        except Exception:
            pass

        # --- CLAUDE.md ---
        for claude_md_path in [
            Path(repo_root) / "CLAUDE.md",
            Path.home() / ".claude" / "CLAUDE.md",
        ]:
            if claude_md_path.is_file():
                try:
                    text = claude_md_path.read_text(encoding="utf-8")[:2000]
                    parts.append(f"\n**CLAUDE.md** (`{claude_md_path}`):\n{text}")
                except Exception:
                    pass
                break

        # --- MEMORY_MULTIHEAD.md ---
        data_dir = getattr(self.ks, "db_path", None)
        if data_dir:
            data_dir = data_dir.parent
        memory_path = Path(data_dir) / "MEMORY_MULTIHEAD.md" if data_dir else None
        if memory_path and memory_path.is_file():
            try:
                text = memory_path.read_text(encoding="utf-8")[:2000]
                parts.append(f"\n**Shared memory** (`{memory_path}`):\n{text}")
            except Exception:
                pass

        if len(parts) <= 1:
            return ""
        return "\n".join(parts)

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _count_claims(self) -> int:
        """Count claims in knowledge store (uses direct SQL for speed)."""
        if not self.ks:
            return 0
        try:
            # Direct SQL is faster than loading 100k claim objects
            if hasattr(self.ks, '_connect'):
                with self.ks._connect() as conn:
                    row = conn.execute("SELECT COUNT(*) FROM claims").fetchone()
                    count = row[0] if row else 0
                    return count if isinstance(count, int) else 0
            return 0
        except Exception:
            return 0

    def _get_peers(self) -> list[str]:
        """Get list of mesh peer node IDs."""
        if not self.ac:
            return []
        try:
            return self.ac._detect_peers()
        except Exception:
            return []
