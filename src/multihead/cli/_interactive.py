"""Interactive commands: chat and shell."""

from __future__ import annotations

import os

import click

from ._helpers import (
    _build_knowledge_deps,
    asyncio,
    console,
    logger,
    AgenticCore,
    EventStore,
    HeadManager,
    KnowledgeStore,
    Orchestrator,
    SessionManager,
    ToolRegistry,
    VRAMManager,
    VRAMPolicy,
    load_heads,
    validate_heads,
    Path,
)
from ._core import main


@main.command()
@click.option("--head", "head_id", default=None, help="Head ID for the core brain (default: first in config)")
@click.pass_context
def chat(ctx, head_id):
    """Interactive chat with the Agentic Core."""
    from dotenv import load_dotenv
    load_dotenv()
    settings = ctx.obj["settings"]
    settings.ensure_dirs()

    ks, rs, art, pb = _build_knowledge_deps(settings)
    event_store = EventStore(settings.runs_dir, settings.db_path)
    all_heads = load_heads(settings.config_dir)

    # Validate adapter availability
    heads, adapter_warnings = validate_heads(all_heads)
    for w in adapter_warnings:
        console.print(f"[yellow]{w} — disabled[/yellow]")

    if not heads:
        console.print("[red]No working heads found. Run 'multihead init --auto' to configure.[/red]")
        return

    hm = HeadManager(heads, knowledge_store=ks)
    orchestrator = Orchestrator(event_store, art, hm, settings.runs_dir)

    core_head_id = head_id or settings.resolve_core_head_id(heads)
    if core_head_id not in heads:
        console.print(f"[red]Head '{core_head_id}' not found. Available: {list(heads.keys())}[/red]")
        return
    sessions_dir = settings.data_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    sm = SessionManager(sessions_dir, pb)
    tr = ToolRegistry()

    # Runtime config + web tools
    from ..runtime_config import RuntimeConfig
    from ..web_tools import register_web_tools
    from ..slash_commands import SlashCommandHandler

    runtime_config_path = settings.data_dir / "runtime_config.json"
    runtime_config = RuntimeConfig.load(runtime_config_path)
    if runtime_config.web_tools_enabled:
        register_web_tools(tr)

    # Knowledge store tools (always available)
    from ..knowledge_tools import register_knowledge_tools
    register_knowledge_tools(tr, ks)

    # ACP tools (when BotVibes is configured)
    if os.environ.get("ACP_URL"):
        from ..acp_tools import register_acp_tools
        register_acp_tools(tr)

    # Apply disabled tools from config
    for tool_name in runtime_config.disabled_tools:
        spec = tr.get_spec(tool_name)
        if spec:
            spec.enabled = False

    vp = VRAMPolicy(core_mode="keep_loaded")
    vm = VRAMManager(hm, vp, core_head_id=core_head_id)

    # Narrative pipeline for live chat ingestion
    from ..narrative.pipeline import NarrativePipeline
    narrative_pipeline = NarrativePipeline(ks, project_id="multihead", artifact_store=art)

    # Presence heartbeat — lets shared-DB peers discover this session
    from ..mesh.presence import PresenceMonitor
    import socket as _socket
    _node_id = f"node-{_socket.gethostname()}"
    presence_monitor = PresenceMonitor(node_id=_node_id, store=ks)

    ac = AgenticCore(hm, orchestrator, tr, sm, vm, core_head_id,
                     strip_thinking=runtime_config.strip_thinking,
                     narrative_pipeline=narrative_pipeline,
                     knowledge_store=ks,
                     project_id="multihead")

    session = sm.create_session()

    slash = SlashCommandHandler(
        config=runtime_config,
        config_path=runtime_config_path,
        tool_registry=tr,
        head_states_fn=hm.get_states,
        knowledge_store=ks,
        session_id=session.session_id,
        project_id="multihead",
        head_manager=hm,
    )

    console.print(f"[bold]MultiHead Chat[/bold] (session: {session.session_id})")
    console.print("[dim]Type 'exit' or 'quit' to end. Type /help for commands.[/dim]\n")

    async def _chat_loop():
        # AgenticCore has no start() — it's ready after __init__
        try:
            while True:
                try:
                    user_input = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if user_input.lower() in ("exit", "quit", "q"):
                    break
                if not user_input:
                    continue

                # Slash commands handled before reaching the LLM
                if slash.is_slash_command(user_input):
                    result = await slash.handle(user_input)
                    console.print(f"{result}\n")
                    continue

                response = await ac.chat(session.session_id, user_input)
                console.print(f"[green]Assistant:[/green] {response}\n")
        finally:
            await ac.stop()
            await hm.shutdown()
            console.print("[dim]Session ended.[/dim]")

    asyncio.run(_chat_loop())


@main.command()
@click.option("--head", "head_id", default=None, help="Head ID for the core brain")
@click.option("--session", "session_id", default=None, help="Resume a previous session")
@click.option("--no-banner", is_flag=True, default=False, help="Skip the startup banner")
@click.option("--brain", "brain_mode", default="local", type=click.Choice(["local", "claude"]),
              help="Brain mode: local (GPU) or claude (Agent SDK)")
@click.option("--responsive", is_flag=True, default=False,
              help="Enable marketplace + auto-responder for reactive operation")
@click.pass_context
def shell(ctx, head_id, session_id, no_banner, brain_mode, responsive):
    """Interactive agent terminal with Rich UI and system management.

    Enhanced REPL with PLUR safety principles, head management commands,
    knowledge-aware context (RAG), and process orchestration.

    Use --brain claude to use Claude Agent SDK as the conversation brain.
    Use --responsive to enable auto-handling of marketplace and collaboration events.
    """
    from dotenv import load_dotenv
    load_dotenv()
    settings = ctx.obj["settings"]
    settings.ensure_dirs()

    ks, rs, art, pb = _build_knowledge_deps(settings)
    event_store = EventStore(settings.runs_dir, settings.db_path)
    all_heads = load_heads(settings.config_dir)

    # Validate adapter availability — warn and filter broken heads
    heads, adapter_warnings = validate_heads(all_heads)
    for w in adapter_warnings:
        console.print(f"[yellow]{w} — disabled[/yellow]")

    if not heads:
        console.print("[red]No working heads found. Run 'multihead init --auto' to configure.[/red]")
        return

    hm = HeadManager(heads, knowledge_store=ks)
    orchestrator = Orchestrator(event_store, art, hm, settings.runs_dir)

    core_head_id = head_id or settings.resolve_core_head_id(heads)
    if core_head_id not in heads:
        console.print(f"[red]Head '{core_head_id}' not found. Available: {list(heads.keys())}[/red]")
        return

    # First-run hint if using mock brain
    if core_head_id == "mock-llm" and len(heads) <= 2:
        console.print("[dim]Using mock brain. Run 'multihead init --auto' for real models.[/dim]")

    sessions_dir = settings.data_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    sm = SessionManager(sessions_dir, pb)
    tr = ToolRegistry()

    # Runtime config + web tools
    from ..runtime_config import RuntimeConfig
    from ..web_tools import register_web_tools
    from ..slash_commands import SlashCommandHandler

    runtime_config_path = settings.data_dir / "runtime_config.json"
    runtime_config = RuntimeConfig.load(runtime_config_path)
    if runtime_config.web_tools_enabled:
        register_web_tools(tr)

    # Knowledge store tools (always available)
    from ..knowledge_tools import register_knowledge_tools
    register_knowledge_tools(tr, ks)

    # ACP tools (when BotVibes is configured)
    if os.environ.get("ACP_URL"):
        from ..acp_tools import register_acp_tools
        register_acp_tools(tr)

    # Apply disabled tools from config
    for tool_name in runtime_config.disabled_tools:
        spec = tr.get_spec(tool_name)
        if spec:
            spec.enabled = False

    vp = VRAMPolicy(core_mode="keep_loaded")
    vm = VRAMManager(hm, vp, core_head_id=core_head_id)

    # Narrative pipeline for live chat ingestion
    from ..narrative.pipeline import NarrativePipeline
    narrative_pipeline = NarrativePipeline(ks, project_id="multihead", artifact_store=art)

    # Presence heartbeat
    from ..mesh.presence import PresenceMonitor
    import socket as _socket
    _node_id = f"node-{_socket.gethostname()}"
    presence_monitor = PresenceMonitor(node_id=_node_id, store=ks)

    ac = AgenticCore(hm, orchestrator, tr, sm, vm, core_head_id,
                     strip_thinking=runtime_config.strip_thinking,
                     narrative_pipeline=narrative_pipeline,
                     knowledge_store=ks,
                     project_id="multihead")

    # Create or resume session
    if session_id:
        session = sm.get_session(session_id)
        if session is None:
            console.print(f"[red]Session '{session_id}' not found. Creating new session.[/red]")
            session = sm.create_session()
    else:
        session = sm.create_session()

    # --responsive: enable proactive marketplace bidding + contract execution
    if responsive:
        runtime_config.services.auto_responder = True
        runtime_config.services.cloud_marketplace = True
        runtime_config.services.cloud_auto_deliver = True
        runtime_config.pipeline.event_watcher.auto_handle = True
        console.print("[dim]Responsive mode ON — auto-quote, auto-deliver, event auto-handle[/dim]")

    # Setup Claude adapter if brain=claude or claude-sdk head exists
    claude_adapter = None
    if brain_mode == "claude" or "claude-sdk" in heads:
        try:
            from ..adapters.claude_agent_sdk import ClaudeAgentSDKAdapter
            claude_head_id = "claude-sdk"
            if claude_head_id in heads:
                claude_adapter = hm.get_adapter(claude_head_id)
            else:
                # Create adapter from first available head with claude_agent_sdk adapter
                for h in heads.values():
                    if h.adapter.value == "claude_agent_sdk":
                        claude_adapter = hm.get_adapter(h.head_id)
                        break
        except Exception as e:
            if brain_mode == "claude":
                console.print(f"[red]Failed to setup Claude adapter: {e}[/red]")
                console.print("[yellow]Falling back to local brain.[/yellow]")
                brain_mode = "local"

    # Create Service Manager for background services
    from ..service_manager import (
        ServiceManager, auto_responder_service, worker_daemon_service,
        night_shift_service, cloud_marketplace_service,
        session_harvester_service,
    )

    svc_mgr = ServiceManager(runtime_config)
    svc_mgr.register(
        "auto-responder",
        factory=lambda: auto_responder_service(
            ks, hm, session.session_id, "multihead", runtime_config.services,
        ),
        description="Polls knowledge.db for decomposition requests",
    )
    svc_mgr.register(
        "worker-daemon",
        factory=lambda: worker_daemon_service(runtime_config.services),
        description="Listens for BotVibes/ACP tasks via WebSocket",
    )
    svc_mgr.register(
        "night-shift",
        factory=lambda: night_shift_service(
            ks, rs, art, pb, hm,
            settings.nightshift_output_dir,
            runtime_config.services,
        ),
        description="17-stage memory refinery (entity/topic/event/claim extraction)",
    )
    def _marketplace_activity(event_type: str, message: str) -> None:
        """Surface marketplace events in the shell."""
        color = {"quote": "cyan", "contract": "green", "delivered": "bold green",
                 "failed": "red"}.get(event_type, "yellow")
        console.print(f"  [{color}][marketplace] {message}[/{color}]")

    svc_mgr.register(
        "cloud-marketplace",
        factory=lambda: cloud_marketplace_service(
            hm, settings, ac, ks, runtime_config,
            on_activity=_marketplace_activity,
            shared_data=svc_mgr.shared_data,
            event_store=event_store,
            artifact_store=art,
        ),
        description="Cloud BotVibes marketplace: auto-quote RFQs, execute contracts (full pipeline)",
    )
    svc_mgr.register(
        "session-harvester",
        factory=lambda: session_harvester_service(
            ks, runtime_config.services,
            claude_home=str(Path.home() / ".claude"),
            data_dir=str(settings.data_dir),
        ),
        description="Scans Claude project folders, harvests MEMORY.md/CLAUDE.md into knowledge.db",
    )

    # Create Shell Pipeline (dogfooding infrastructure)
    from ..shell_pipeline import ShellPipeline
    from ..router import Router

    router = Router(hm, knowledge_store=ks)

    # AutoDecomposer for complex task routing (optional — graceful if unavailable)
    auto_decomposer = None
    try:
        from ..auto_decomposition import AutoDecomposer
        auto_decomposer = AutoDecomposer(hm, knowledge_store=ks)
    except Exception:
        pass

    pipeline = ShellPipeline(
        knowledge_store=ks,
        head_manager=hm,
        router=router,
        runtime_config=runtime_config,
        orchestrator=orchestrator,
        auto_decomposer=auto_decomposer,
    )

    # Create EventWatcher for background event detection
    from ..event_watcher import EventWatcher

    ew_cfg = getattr(runtime_config.pipeline, "event_watcher", None)
    from ..shell_pipeline import AGENT_ID
    event_watcher = EventWatcher(
        knowledge_store=ks,
        session_id=session.session_id,
        project_id="multihead",
        poll_interval=getattr(ew_cfg, "poll_interval", 15) if ew_cfg else 15,
        watch_acp=getattr(ew_cfg, "watch_acp", True) if ew_cfg else True,
        watch_knowledge=getattr(ew_cfg, "watch_knowledge", True) if ew_cfg else True,
        agent_id=AGENT_ID,
    )

    # Create Shell instance first (slash handler needs reference)
    from ..shell import Shell

    agent_shell = Shell(
        agentic_core=ac,
        head_manager=hm,
        knowledge_store=ks,
        session_manager=sm,
        slash_handler=None,  # set below after slash is created
        runtime_config=runtime_config,
        show_banner=not no_banner,
        brain=brain_mode,
        claude_adapter=claude_adapter,
        pipeline=pipeline,
        service_manager=svc_mgr,
        event_watcher=event_watcher,
    )
    agent_shell._config_dir = settings.config_dir

    slash = SlashCommandHandler(
        config=runtime_config,
        config_path=runtime_config_path,
        tool_registry=tr,
        head_states_fn=hm.get_states,
        knowledge_store=ks,
        session_id=session.session_id,
        project_id="multihead",
        head_manager=hm,
        shell=agent_shell,
        service_manager=svc_mgr,
        event_store=event_store,
        artifact_store=art,
        runs_dir=settings.runs_dir,
    )
    agent_shell.slash = slash
    slash.pipeline = pipeline

    try:
        asyncio.run(agent_shell.run(session.session_id))
    except KeyboardInterrupt:
        console.print("\n[dim]Session ended.[/dim]")
    except Exception as e:
        console.print(f"\n[red]Shell error: {e}[/red]")
        logger.error("Unhandled shell exception", exc_info=True)
