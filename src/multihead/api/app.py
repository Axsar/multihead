"""FastAPI application with lifespan for MultiHead daemon."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

from fastapi import FastAPI

from ..acp_bridge import ACPBridge
from ..agentic_core import AgenticCore
from ..skill_catalog import SkillCatalog
from ..skill_loader import SkillRegistry
from ..acp_tools import register_acp_tools
from ..runtime_config import RuntimeConfig
from ..solve_tools import register_solve_tools
from ..web_tools import register_web_tools
from ..artifact_store import ArtifactStore
from ..config import Settings, load_heads
from ..context_packs import PackBuilder
from ..event_store import EventStore
from ..head_manager import HeadManager
from ..knowledge_models import NightShiftConfig
from ..knowledge_store import KnowledgeStore
from ..mesh.capability import CapabilityRegistry, auto_register_from_heads
from ..mesh.discovery import MeshDiscovery
from ..mesh.presence import PresenceMonitor
from ..mesh.security import MeshSecurity
from ..night_shift import NightShift
from ..observability import MetricsCollector
from ..orchestrator import Orchestrator
from ..record_store import RecordStore
from ..resilience import ResourceMonitor
from ..session import SessionManager
from ..tool_registry import ToolRegistry
from ..vram_policy import VRAMManager, VRAMPolicy
from .routes_artifacts import router as artifacts_router
from .routes_chat import router as chat_router
from .routes_dashboard import router as dashboard_router
from .routes_heads import router as heads_router
from .routes_knowledge import router as knowledge_router
from .routes_nightshift import router as nightshift_router
from .routes_packs import router as packs_router
from .routes_runs import router as runs_router
from .routes_acp import router as acp_router
from .routes_config import router as config_router
from .routes_consensus import router as consensus_router
from .routes_decompose import router as decompose_router
from .routes_skills import router as skills_router
from .routes_catalog import router as catalog_router
from .routes_solve import router as solve_router
from .routes_file_size import router as file_size_router
from .routes_harvester import router as harvester_router
from .routes_system import router as system_router
from .routes_ws import router as ws_router
from .routes_operations import router as operations_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all components on startup, clean up on shutdown."""
    load_dotenv()  # Load .env into os.environ for ACP_* vars
    settings: Settings = app.state.settings

    settings.ensure_dirs()

    # Initialize stores
    artifact_store = ArtifactStore(settings.artifacts_dir, settings.db_path)
    event_store = EventStore(settings.runs_dir, settings.db_path)

    # v0.2: Knowledge layer
    knowledge_store = KnowledgeStore(settings.knowledge_db_path)
    record_store = RecordStore(knowledge_store, artifact_store)
    pack_builder = PackBuilder(knowledge_store, settings.packs_dir)

    # Load head manifests
    heads = load_heads(settings.config_dir)
    head_manager = HeadManager(heads)

    # Create orchestrator (ACP bridge wired after it's created below)
    orchestrator = Orchestrator(
        event_store=event_store,
        artifact_store=artifact_store,
        head_manager=head_manager,
        runs_dir=settings.runs_dir,
    )
    # Note: acp_bridge is set on orchestrator after ACPBridge is created (below)

    # Night Shift pipeline
    core_head_id = settings.resolve_core_head_id(heads)
    ns_config = NightShiftConfig(head_id=core_head_id)
    night_shift = NightShift(
        knowledge_store, record_store, artifact_store,
        pack_builder, head_manager, ns_config,
        settings.nightshift_output_dir,
    )

    # v0.3: Agentic Core
    sessions_dir = settings.data_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    session_manager = SessionManager(sessions_dir, pack_builder)
    tool_registry = ToolRegistry()

    # Runtime config (mutable, persisted)
    runtime_config_path = settings.data_dir / "runtime_config.json"
    runtime_config = RuntimeConfig.load(runtime_config_path)
    if runtime_config.web_tools_enabled:
        register_web_tools(tool_registry)

    register_solve_tools(
        tool_registry, head_manager, event_store, artifact_store,
        knowledge_store, settings.runs_dir,
    )

    vram_policy = VRAMPolicy(core_mode="keep_loaded")
    vram_manager = VRAMManager(head_manager, vram_policy, core_head_id=core_head_id)

    # Narrative pipeline for live chat ingestion + GPU activity logging
    from multihead.narrative.pipeline import NarrativePipeline
    narrative_pipeline = NarrativePipeline(knowledge_store, project_id="multihead", artifact_store=artifact_store)
    head_manager.set_narrative_pipeline(narrative_pipeline)

    agentic_core = AgenticCore(
        head_manager=head_manager,
        orchestrator=orchestrator,
        tool_registry=tool_registry,
        session_manager=session_manager,
        vram_manager=vram_manager,
        core_head_id=core_head_id,
        strip_thinking=runtime_config.strip_thinking,
        narrative_pipeline=narrative_pipeline,
    )

    # v1.0: Mesh capability registry + security
    # Use dashes (not colons) in node_id — mDNS service names forbid colons
    node_id = f"node-{settings.api_host}-{settings.api_port}"
    capability_registry = CapabilityRegistry()
    auto_register_from_heads(capability_registry, heads, node_id)
    mesh_security = MeshSecurity(settings.mesh_secret) if settings.mesh_secret else None

    # v1.0: Mesh discovery (mDNS) — pass knowledge_store for DB-backed peer scan
    mesh_discovery = MeshDiscovery(
        node_id, port=settings.api_port, knowledge_store=knowledge_store,
    )
    mdns_started = mesh_discovery.start()
    if mdns_started:
        logger.info("mDNS discovery active on port %d", settings.api_port)
    else:
        logger.info("mDNS discovery not available (zeroconf not installed)")

    # v1.0: Presence monitor — heartbeat claims to knowledge.db
    # Skip in test environments to avoid polluting empty-DB assertions.
    presence_monitor = None
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        presence_monitor = PresenceMonitor(
            node_id=node_id,
            store=knowledge_store,
            port=settings.api_port,
        )
        await presence_monitor.start()

    # v1.0: Peer registry, knowledge sync, failover
    from multihead.mesh.peer_registry import PeerRegistry
    from multihead.mesh.knowledge_sync import KnowledgeSync
    from multihead.mesh.failover import MeshFailoverPolicy

    peer_registry = PeerRegistry(
        mesh_discovery=mesh_discovery,
        auth_token=settings.mesh_secret or "",
    )
    await peer_registry.refresh()
    await peer_registry.start_periodic_refresh()

    knowledge_sync = KnowledgeSync(
        knowledge_store=knowledge_store,
        peer_registry=peer_registry,
    )
    await knowledge_sync.start_periodic_sync()

    failover_policy = MeshFailoverPolicy(
        head_manager=head_manager,
        peer_registry=peer_registry,
    )

    # v1.0: Observability + resilience
    metrics = MetricsCollector()
    resource_monitor = ResourceMonitor()

    # ACP Bridge (optional — connects to BotVibes when available)
    # Register ACP tools for the local LLM (if ACP is configured)
    if os.environ.get("ACP_URL"):
        register_acp_tools(tool_registry)

    # Load skills from agentskills.io SKILL.md files
    skill_registry = SkillRegistry()
    skills_dir = settings.data_dir / "skills"
    # Also check project-level skills dir (next to src/)
    project_skills_dir = settings.config_dir.parent / "skills"
    for sd in [skills_dir, project_skills_dir]:
        if sd.is_dir():
            skill_registry.load_directory(sd)
    logger.info("Skills loaded: %d", len(skill_registry))

    # Skill Catalog (SQLite-backed, for browse/install/publish)
    catalog_db_path = settings.data_dir / "skill_catalog.db"
    skill_catalog = SkillCatalog(catalog_db_path)
    await skill_catalog.open()
    logger.info("Skill catalog opened at %s", catalog_db_path)

    # Sync locally-loaded skills into the catalog (only add new ones, don't re-add deleted)
    from ..skill_catalog import CatalogEntry
    synced = 0
    for skill in skill_registry.list_skills():
        existing = await skill_catalog.get(skill.name)
        if existing:
            continue  # Already in catalog
        if await skill_catalog.is_deleted(skill.name):
            continue  # User explicitly deleted this — don't re-add
        entry = CatalogEntry(
            name=skill.name,
            description=skill.description,
            source_url="",
            category=skill.metadata.get("category", "general"),
            license_type=skill.license or "free",
            author=skill.metadata.get("author", "local"),
            version=skill.metadata.get("version", "1.0.0"),
            tags=list(skill.metadata.get("tags", "").split()) if skill.metadata.get("tags") else [],
            compatibility=skill.compatibility,
            instruction_lines=skill.instructions.count("\n") + 1,
            installed=True,
        )
        await skill_catalog.upsert(entry)
        synced += 1
    if synced:
        logger.info("Added %d new skills to catalog", synced)

    acp_auto_execute = getattr(
        getattr(runtime_config, "services", None), "acp_auto_execute", False,
    )
    acp_bridge = ACPBridge(
        head_manager=head_manager,
        settings=settings,
        acp_url=os.environ.get("ACP_URL"),
        api_key=os.environ.get("ACP_API_KEY"),
        project_id=os.environ.get("ACP_PROJECT_ID"),
        agent_id=os.environ.get("ACP_AGENT_ID", "multihead-agent"),
        claude_agent_id=os.environ.get("ACP_CLAUDE_AGENT_ID", "claude-session-agent"),
        auto_execute=acp_auto_execute,
        skill_registry=skill_registry,
        runtime_config=runtime_config,
    )
    acp_bridge.set_agentic_core(agentic_core)
    acp_bridge.set_stores(artifact_store, knowledge_store)
    await acp_bridge.start()

    # Wire ACP bridge to orchestrator for remote delegation
    orchestrator.acp_bridge = acp_bridge
    if acp_bridge.connected:
        orchestrator.enable_marketplace_delegation = True

    # Resume incomplete runs
    resumed = await orchestrator.resume_incomplete_runs()
    if resumed:
        logger.info("Resumed %d incomplete runs", len(resumed))

    # Store refs on app state
    app.state.artifact_store = artifact_store
    app.state.event_store = event_store
    app.state.head_manager = head_manager
    app.state.orchestrator = orchestrator
    app.state.knowledge_store = knowledge_store
    app.state.record_store = record_store
    app.state.pack_builder = pack_builder
    app.state.night_shift = night_shift
    app.state.session_manager = session_manager
    app.state.tool_registry = tool_registry
    app.state.agentic_core = agentic_core
    app.state.capability_registry = capability_registry
    app.state.node_id = node_id
    app.state.metrics = metrics
    app.state.resource_monitor = resource_monitor
    app.state.mesh_security = mesh_security
    app.state.mesh_discovery = mesh_discovery
    app.state.acp_bridge = acp_bridge
    app.state.skill_registry = skill_registry
    app.state.runtime_config = runtime_config
    app.state.skill_catalog = skill_catalog
    app.state.runtime_config_path = runtime_config_path
    app.state.presence_monitor = presence_monitor
    app.state.peer_registry = peer_registry
    app.state.knowledge_sync = knowledge_sync
    app.state.failover_policy = failover_policy

    # MCP-over-HTTP: mount FastMCP on /mcp (Streamable HTTP transport)
    from ..mcp_mount import mcp as mcp_instance, wire_mcp
    wire_mcp(app.state)
    mcp_http = mcp_instance.http_app(transport="streamable-http", stateless_http=True)
    app.mount("/mcp", mcp_http)
    logger.info("MCP-over-HTTP mounted at /mcp (streamable-http)")

    logger.info("MultiHead daemon started on %s:%d", settings.api_host, settings.api_port)
    logger.info("Registered heads: %s", list(heads.keys()))

    # Start MCP session manager lifespan (required for streamable-http)
    async with mcp_http.lifespan(mcp_http):
        yield

    # Shutdown
    logger.info("Shutting down MultiHead daemon...")
    await skill_catalog.close()
    await knowledge_sync.stop()
    await peer_registry.stop()
    await acp_bridge.stop()
    if presence_monitor:
        await presence_monitor.stop()
    mesh_discovery.stop()
    await head_manager.shutdown()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI app."""
    if settings is None:
        settings = Settings()

    app = FastAPI(
        title="MultiHead",
        description="Local multimodal task-runner with hot-swappable specialist models",
        version="1.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings

    # CORS — allow Cortex frontend (and any local dev server) to call the API
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Core routers
    app.include_router(runs_router, prefix="/runs", tags=["Runs"])
    app.include_router(heads_router, prefix="/heads", tags=["Heads"])
    app.include_router(artifacts_router, prefix="/artifacts", tags=["Artifacts"])
    app.include_router(knowledge_router, prefix="/knowledge", tags=["Knowledge"])
    app.include_router(packs_router, prefix="/packs", tags=["Packs"])
    app.include_router(nightshift_router, prefix="/nightshift", tags=["Night Shift"])
    app.include_router(harvester_router, prefix="/nightshift", tags=["Harvester"])
    app.include_router(chat_router, prefix="/chat", tags=["Chat"])
    app.include_router(consensus_router, prefix="/consensus", tags=["Consensus"])
    app.include_router(config_router, prefix="/config", tags=["Config"])
    app.include_router(acp_router, prefix="/acp", tags=["ACP"])
    app.include_router(decompose_router, prefix="/decompose", tags=["Decompose"])
    app.include_router(solve_router, prefix="/solve", tags=["Solve"])
    app.include_router(catalog_router, prefix="/skills", tags=["Catalog"])
    app.include_router(skills_router, prefix="/skills", tags=["Skills"])
    app.include_router(file_size_router, prefix="/file-size", tags=["File Size"])
    app.include_router(system_router, prefix="/system", tags=["System"])
    app.include_router(operations_router, prefix="/operations", tags=["Operations"])

    # v1.0: Mesh, WebSocket, Dashboard
    from ..mesh.mesh_routes import router as mesh_router
    app.include_router(mesh_router, prefix="/v1", tags=["Mesh"])
    app.include_router(ws_router, prefix="/ws", tags=["WebSocket"])
    app.include_router(dashboard_router, tags=["Dashboard"])

    @app.get("/health")
    async def health():
        from .routes_system import get_uptime_seconds

        ready = hasattr(app.state, "head_manager")
        return {
            "status": "ok" if ready else "starting",
            "version": "1.0.0",
            "ready": ready,
            "heads_loaded": len(app.state.head_manager.get_states()) if ready else 0,
            "heads": list(app.state.head_manager.get_states().keys()) if ready else [],
            "uptime_seconds": get_uptime_seconds(),
        }

    @app.get("/healthz")
    async def liveness():
        """Liveness probe: process is alive."""
        return {"status": "alive"}

    @app.get("/readyz")
    async def readiness():
        """Readiness probe: all components initialized and accessible."""
        checks: dict[str, bool] = {}

        # Head manager loaded
        checks["head_manager"] = hasattr(app.state, "head_manager")

        # Event store accessible
        try:
            if hasattr(app.state, "event_store"):
                app.state.event_store.list_runs()
                checks["event_store"] = True
            else:
                checks["event_store"] = False
        except Exception:
            checks["event_store"] = False

        # Orchestrator present
        checks["orchestrator"] = hasattr(app.state, "orchestrator")

        all_ready = all(checks.values())
        if not all_ready:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "checks": checks},
            )
        return {"status": "ready", "checks": checks}

    return app
