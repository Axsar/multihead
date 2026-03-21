"""Shared utilities for CLI commands."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from ..agentic_core import AgenticCore
from ..artifact_store import ArtifactStore
from ..auto_decomposition import AutoDecomposer
from ..bundle import BundleExporter, BundleImporter
from ..config import Settings, load_heads, load_recipe, validate_heads
from ..consensus import ConsensusConfig, ConsensusEngine, ConsensusStrategy, HeadTask
from ..context_packs import PackBuilder
from ..diagnostics import Diagnostics
from ..event_store import EventStore
from ..head_manager import HeadManager
from ..init_wizard import InitWizard
from ..knowledge_hook import KnowledgeHook
from ..knowledge_models import NightShiftConfig
from ..knowledge_store import KnowledgeStore
from ..night_shift import NightShift
from ..orchestrator import Orchestrator
from ..record_store import RecordStore
from ..session import SessionManager
from ..tool_registry import ToolRegistry
from ..vram_policy import VRAMManager, VRAMPolicy

console = Console()
logger = logging.getLogger(__name__)


def _get_settings(data_dir: str | None = None, config_dir: str | None = None) -> Settings:
    from ..config import resolve_config_dir

    kwargs = {}
    if data_dir:
        kwargs["data_dir"] = Path(data_dir)
    # Resolve config dir: explicit > ./config > ~/.multihead/config > bundled
    kwargs["config_dir"] = resolve_config_dir(config_dir)
    return Settings(**kwargs)


def _build_orchestrator(settings: Settings) -> tuple[Orchestrator, HeadManager]:
    settings.ensure_dirs()
    artifact_store = ArtifactStore(settings.artifacts_dir, settings.db_path)
    event_store = EventStore(settings.runs_dir, settings.db_path)
    heads = load_heads(settings.config_dir)

    # Build knowledge store for cross-session collaboration
    knowledge_store = KnowledgeStore(settings.knowledge_db_path)

    # Pass knowledge_store to HeadManager
    head_manager = HeadManager(heads, knowledge_store=knowledge_store)
    orchestrator = Orchestrator(event_store, artifact_store, head_manager, settings.runs_dir)
    return orchestrator, head_manager


def _setup_logging(settings: Settings, debug: bool = False) -> None:
    """Configure logging to console + file."""
    import logging.handlers

    level = logging.DEBUG if debug else logging.INFO
    fmt = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

    # Console
    logging.basicConfig(level=level, format=fmt)

    # File (rotated, in data dir)
    log_dir = settings.data_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "multihead.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)  # Always debug to file
    file_handler.setFormatter(logging.Formatter(fmt))
    logging.getLogger().addHandler(file_handler)


def _build_knowledge_deps(settings: Settings):
    """Build knowledge layer dependencies."""
    settings.ensure_dirs()
    artifact_store = ArtifactStore(settings.artifacts_dir, settings.db_path)
    knowledge_store = KnowledgeStore(settings.knowledge_db_path)
    record_store = RecordStore(knowledge_store, artifact_store)
    pack_builder = PackBuilder(knowledge_store, settings.packs_dir)
    return knowledge_store, record_store, artifact_store, pack_builder


def _parse_since(since_str: str | None) -> datetime | None:
    """Parse --since as ISO datetime or relative (e.g. '24h', '7d', '30d')."""
    if not since_str:
        return None
    from datetime import timedelta
    suffix_map = {"h": "hours", "d": "days", "w": "weeks"}
    if since_str[-1] in suffix_map and since_str[:-1].isdigit():
        delta = timedelta(**{suffix_map[since_str[-1]]: int(since_str[:-1])})
        return datetime.now(timezone.utc) - delta
    return datetime.fromisoformat(since_str)
