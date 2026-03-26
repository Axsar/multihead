"""MultiHead CLI: run pipelines, inspect runs, manage heads.

This package re-exports all CLI components so that existing imports like
``from multihead.cli import main`` continue to work unchanged.
"""

from __future__ import annotations

# --- Shared utilities (must be imported first) ---
from ._helpers import (
    console,
    logger,
    _get_settings,
    _build_orchestrator,
    _setup_logging,
    _build_knowledge_deps,
    _parse_since,
)

# --- Core commands: main group, run, status, inspect, heads, info, doctor, export/import ---
from ._core import (
    main,
    run,
    status,
    inspect,
    heads,
    info,
    export_cmd,
    import_cmd,
    doctor,
)

# --- Server commands: serve, daemon, mcp ---
from ._server import serve, daemon, mcp

# --- Solve commands (two 'solve' definitions; the second overwrites the first) ---
from ._solve import solve

# --- Interactive commands: chat, shell ---
from ._interactive import chat, shell

# --- Init command ---
from ._init import (
    init_cmd,
    _detect_peers_from_db,
    _prompt_for_dir,
    _update_env_data_dir,
    _show_init_status,
    _init_mesh_config,
)

# --- Knowledge, packs, nightshift ---
from ._knowledge import (
    knowledge,
    knowledge_claims,
    knowledge_deposit,
    knowledge_events,
    knowledge_refresh_refs,
    knowledge_search,
    kb,
    deposit_shortcut,
    packs,
    packs_list,
    packs_build,
    packs_build_standard,
    packs_show,
    nightshift,
    nightshift_run,
    _search_claims_filtered,
    _run_semantic_search,
    _print_search_results,
)

# --- Skills ---
from ._skills import (
    skills,
    skills_list,
    skills_sync,
    skills_install,
    skills_publish,
)

# --- Narrative ---
from ._narrative import (
    narrative,
    narrative_ingest,
    narrative_fuse,
    narrative_status,
)

# --- Auth ---
from ._auth import auth, auth_status

# --- Consensus ---
from ._consensus import (
    consensus,
    consensus_strategies,
    consensus_test,
)

# --- Discover ---
from ._discover import discover, _register_discovered

# --- Mesh ---
from ._mesh import (
    mesh_group,
    mesh_discover,
    mesh_list_peers,
    mesh_status,
    mesh_peers,
    mesh_sync,
    mesh_keys,
    mesh_trust,
    mesh_join_network,
)

# --- Recipes ---
from ._recipes import recipes, recipes_learn

# --- Ratchet (AutoResearch-style experiment loop) ---
from ._ratchet import ratchet

__all__ = [
    # Core
    "main",
    "run",
    "status",
    "inspect",
    "heads",
    "info",
    "export_cmd",
    "import_cmd",
    "doctor",
    # Server
    "serve",
    "daemon",
    "mcp",
    # Solve
    "solve",
    # Interactive
    "chat",
    "shell",
    # Init
    "init_cmd",
    "_detect_peers_from_db",
    "_prompt_for_dir",
    "_update_env_data_dir",
    "_show_init_status",
    "_init_mesh_config",
    # Knowledge
    "knowledge",
    "knowledge_claims",
    "knowledge_deposit",
    "knowledge_events",
    "knowledge_refresh_refs",
    "knowledge_search",
    "kb",
    "deposit_shortcut",
    "packs",
    "packs_list",
    "packs_build",
    "packs_build_standard",
    "packs_show",
    "nightshift",
    "nightshift_run",
    "_search_claims_filtered",
    "_run_semantic_search",
    "_print_search_results",
    # Skills
    "skills",
    "skills_list",
    "skills_sync",
    "skills_install",
    "skills_publish",
    # Narrative
    "narrative",
    "narrative_ingest",
    "narrative_fuse",
    "narrative_status",
    # Auth
    "auth",
    "auth_status",
    # Consensus
    "consensus",
    "consensus_strategies",
    "consensus_test",
    # Discover
    "discover",
    "_register_discovered",
    # Mesh
    "mesh_group",
    "mesh_discover",
    "mesh_list_peers",
    "mesh_status",
    "mesh_peers",
    "mesh_sync",
    "mesh_keys",
    "mesh_trust",
    "mesh_join_network",
    # Recipes
    "recipes",
    "recipes_learn",
    # Ratchet
    "ratchet",
    # Helpers
    "console",
    "logger",
    "_get_settings",
    "_build_orchestrator",
    "_setup_logging",
    "_build_knowledge_deps",
    "_parse_since",
]
