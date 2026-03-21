"""Server commands: serve, daemon, mcp."""

from __future__ import annotations

import logging
import os
import signal
import time

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
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=7337, type=int)
@click.pass_context
def serve(ctx, host, port):
    """Start the MultiHead API server."""
    import uvicorn
    from ..api.app import create_app

    settings = ctx.obj["settings"]
    settings.api_host = host
    settings.api_port = port

    app = create_app(settings)
    uvicorn.run(app, host=host, port=port)


@main.command()
@click.option("--service", "-s", multiple=True,
              help="Services to run (repeatable). Default: cloud-marketplace.")
@click.option("--head", "head_id", default=None,
              help="Core head ID (default: from settings)")
@click.option("--log-level", default="INFO",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
              help="Logging level")
@click.pass_context
def daemon(ctx, service, head_id, log_level):
    """Run background services without the interactive shell.

    Starts selected services (cloud marketplace, auto-responder, worker daemon)
    and blocks until Ctrl+C. Useful for headless marketplace operation.

    Examples:

        multihead daemon

        multihead daemon -s cloud-marketplace -s auto-responder

        multihead daemon --log-level DEBUG
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

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

    from ..runtime_config import RuntimeConfig
    from ..knowledge_tools import register_knowledge_tools

    runtime_config_path = settings.data_dir / "runtime_config.json"
    runtime_config = RuntimeConfig.load(runtime_config_path)
    register_knowledge_tools(tr, ks)

    if os.environ.get("ACP_URL"):
        from ..acp_tools import register_acp_tools
        register_acp_tools(tr)

    vp = VRAMPolicy(core_mode="keep_loaded")
    vm = VRAMManager(hm, vp, core_head_id=core_head_id)

    ac = AgenticCore(hm, orchestrator, tr, sm, vm, core_head_id,
                     knowledge_store=ks, project_id="multihead")

    # Enable responsive mode for daemon (always)
    runtime_config.services.cloud_marketplace = True
    runtime_config.services.cloud_auto_deliver = True
    runtime_config.pipeline.event_watcher.auto_handle = True

    from ..service_manager import (
        ServiceManager, auto_responder_service, worker_daemon_service,
        cloud_marketplace_service,
    )

    svc_mgr = ServiceManager(runtime_config)

    # Determine which services to run
    requested = set(service) if service else {"cloud-marketplace"}

    daemon_session_id = f"daemon-{int(time.time())}"

    if "cloud-marketplace" in requested:
        def _daemon_activity(event_type: str, message: str) -> None:
            logger.info("[marketplace:%s] %s", event_type, message)

        svc_mgr.register(
            "cloud-marketplace",
            factory=lambda: cloud_marketplace_service(
                hm, settings, ac, ks, runtime_config,
                on_activity=_daemon_activity,
                shared_data=svc_mgr.shared_data,
                event_store=event_store,
                artifact_store=art,
            ),
            description="Cloud marketplace: auto-quote + execute contracts (full pipeline)",
            auto_start=True,
        )

    if "auto-responder" in requested:
        svc_mgr.register(
            "auto-responder",
            factory=lambda: auto_responder_service(
                ks, hm, daemon_session_id, "multihead", runtime_config.services,
            ),
            description="Polls knowledge.db for decomposition requests",
            auto_start=True,
        )

    if "worker-daemon" in requested:
        svc_mgr.register(
            "worker-daemon",
            factory=lambda: worker_daemon_service(runtime_config.services),
            description="Listens for BotVibes/ACP tasks via WebSocket",
            auto_start=True,
        )

    if not svc_mgr.registered_names:
        console.print("[red]No valid services specified. Available: "
                      "cloud-marketplace, auto-responder, worker-daemon[/red]")
        return

    console.print(f"[bold]MultiHead Daemon[/bold] — services: {', '.join(svc_mgr.registered_names)}")
    console.print("[dim]Press Ctrl+C to stop.[/dim]")

    async def _run_daemon() -> None:
        # Start all registered services
        messages = await svc_mgr.auto_start_all()
        for msg in messages:
            logger.info(msg)

        # Block until signal
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop_event.set)
            except NotImplementedError:
                pass  # Windows

        # Periodic status log
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=300)
            except asyncio.TimeoutError:
                # Log stats every 5 minutes
                mkt = svc_mgr.shared_data.get("marketplace_stats", {})
                statuses = {s["name"]: s["status"] for s in svc_mgr.status()}
                logger.info(
                    "Daemon alive — services: %s | marketplace: %s",
                    statuses, {k: v for k, v in mkt.items() if v},
                )

        logger.info("Shutdown signal received")
        await svc_mgr.shutdown_all()

        # Cleanup
        try:
            ac.stop()
        except Exception:
            pass
        try:
            await hm.shutdown()
        except Exception:
            pass

    try:
        asyncio.run(_run_daemon())
    except KeyboardInterrupt:
        console.print("\n[dim]Daemon stopped.[/dim]")
    except Exception as e:
        console.print(f"\n[red]Daemon error: {e}[/red]")
        logger.error("Unhandled daemon exception", exc_info=True)


@main.command()
@click.option("--host", default="127.0.0.1", help="MultiHead API host to proxy to")
@click.option("--port", default=7337, type=int, help="MultiHead API port to proxy to")
@click.option("--transport", default="stdio", type=click.Choice(["stdio", "sse", "streamable-http"]),
              help="Transport: stdio (CLI), sse or streamable-http (Cowork/Desktop)")
@click.option("--mcp-port", default=8338, type=int, help="Port for HTTP transports (default 8338)")
def mcp(host, port, transport, mcp_port):
    """Start MCP server for Claude Code / Cowork / Desktop integration."""
    from dotenv import load_dotenv
    load_dotenv()
    from ..mcp_server import run_mcp_server
    run_mcp_server(host=host, port=port, transport=transport, mcp_port=mcp_port)
