"""Shared state and utilities for the MCP server package."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import httpx
from fastmcp import FastMCP

from multihead.knowledge_store import KnowledgeStore

logger = logging.getLogger("multihead.mcp_server")

# Module-level defaults (overridden by run_mcp_server)
_BASE_URL = "http://127.0.0.1:7337"

mcp = FastMCP(
    "multihead",
    instructions="MultiHead: local multimodal task-runner with hot-swappable specialist models",
)

_TIMEOUT = 120.0  # Long timeout for model inference

# Lazy-initialized KnowledgeStore for direct SQLite access (no server needed)
_knowledge_store: KnowledgeStore | None = None


def _get_ks() -> KnowledgeStore:
    """Return a lazily-initialized KnowledgeStore pointing at knowledge.db.

    Checks the package-level ``_knowledge_store`` first so that test fixtures
    which do ``mcp_mod._knowledge_store = ks`` continue to work after the
    module-to-package refactor.
    """
    global _knowledge_store

    # Honour overrides set on the package (e.g. by test fixtures).
    import sys
    pkg = sys.modules.get("multihead.mcp_server")
    if pkg is not None:
        override = pkg.__dict__.get("_knowledge_store")
        if override is not None:
            return override

    if _knowledge_store is None:
        db_path = Path(os.environ.get("MULTIHEAD_DATA_DIR", str(Path.home() / ".multihead"))) / "knowledge.db"
        _knowledge_store = KnowledgeStore(db_path)
    return _knowledge_store


async def _request(method: str, path: str, **kwargs) -> dict:
    """Make an HTTP request to the MultiHead API."""
    async with httpx.AsyncClient(base_url=_BASE_URL, timeout=_TIMEOUT) as client:
        resp = await client.request(method, path, **kwargs)
        resp.raise_for_status()
        return resp.json()
