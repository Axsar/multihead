"""Service wrapper functions for background services.

Each function is a long-running async coroutine designed to be registered
with ServiceManager as a service factory. They wrap existing subsystem
classes (NightShift, EventWatcher, etc.) into a poll-loop lifecycle.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


async def auto_responder_service(
    knowledge_store: Any,
    head_manager: Any,
    session_id: str,
    project_id: str,
    config: Any,
) -> None:
    """Auto-responder as an async service within shell.

    Polls knowledge.db for decomposition requests and responds
    using the session_poller utilities.
    """
    from ..session_poller import check_for_decomposition_requests

    interval = getattr(config, "responder_interval", 30)
    logger.info("Auto-responder service started (interval=%ds)", interval)

    try:
        while True:
            try:
                requests = check_for_decomposition_requests(
                    knowledge_store, project_id, session_id,
                )
                for req in requests:
                    try:
                        from .. import auto_responder
                        await auto_responder.respond_to_request(
                            req.claim_id,
                            knowledge_store,
                            head_manager,
                            session_id,
                            project_id,
                        )
                    except Exception as e:
                        logger.warning(
                            "Auto-responder failed for %s: %s", req.claim_id[:8], e,
                        )
            except Exception as e:
                logger.warning("Auto-responder poll error: %s", e)

            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        logger.info("Auto-responder service stopped")
        raise


async def night_shift_service(
    knowledge_store: Any,
    record_store: Any,
    artifact_store: Any,
    pack_builder: Any,
    head_manager: Any,
    output_dir: Any,
    config: Any,
) -> None:
    """Night Shift memory refinery as an async service within shell.

    Runs the full 17-stage extraction/consistency pipeline periodically.
    Head selection controlled via config.night_shift_head.
    """
    from pathlib import Path

    interval = getattr(config, "night_shift_interval", 3600)
    head_id = getattr(config, "night_shift_head", "") or "qwen-llm"
    concurrency = getattr(config, "night_shift_concurrency", 1)

    logger.info(
        "Night Shift service started (interval=%ds, head=%s, concurrency=%d)",
        interval, head_id, concurrency,
    )

    try:
        while True:
            try:
                from ..knowledge_models import NightShiftConfig
                from ..night_shift import NightShift

                ns_config = NightShiftConfig(
                    head_id=head_id,
                    concurrency=concurrency,
                )
                ns = NightShift(
                    knowledge_store, record_store, artifact_store,
                    pack_builder, head_manager, ns_config,
                    Path(output_dir) if not isinstance(output_dir, Path) else output_dir,
                )

                # Log progress events
                def _on_progress(evt: dict) -> None:
                    event = evt.get("event", "")
                    if event == "stage_start":
                        logger.info(
                            "Night Shift [%d/%d]: %s",
                            evt.get("index", 0) + 1,
                            evt.get("total", 0),
                            evt.get("stage", ""),
                        )
                    elif event == "stage_fail":
                        logger.warning("Night Shift stage failed: %s", evt.get("stage", ""))
                    elif event == "complete":
                        logger.info("Night Shift run complete")

                ns.on_progress = _on_progress

                report = await ns.run()
                logger.info(
                    "Night Shift: %d records, %d events, %d claims, "
                    "%d completed, %d failed",
                    report.records_processed,
                    report.events_created,
                    report.claims_created,
                    len(report.stages_completed),
                    len(report.stages_failed),
                )
            except Exception as e:
                logger.error("Night Shift run failed: %s", e)

            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        logger.info("Night Shift service stopped")
        raise


async def worker_daemon_service(config: Any) -> None:
    """Claude worker daemon as an async service.

    Reuses the ClaudeWorker class from scripts/claude_worker.py.
    Falls back gracefully if the script isn't available.
    """
    mode = getattr(config, "worker_mode", "sdk")
    logger.info("Worker daemon service starting (mode=%s)", mode)

    try:
        import importlib.util
        import sys
        from pathlib import Path

        # Search for claude_worker.py: repo checkout first, then package
        candidates = [
            Path(__file__).parent.parent.parent.parent / "scripts" / "claude_worker.py",
            Path(__file__).parent.parent / "workers" / "claude_worker.py",
        ]
        worker_path = next((p for p in candidates if p.exists()), None)

        if worker_path is None:
            logger.warning(
                "Claude worker script not found; worker daemon unavailable. "
                "Searched: %s", [str(c) for c in candidates],
            )
            return

        spec = importlib.util.spec_from_file_location("claude_worker", worker_path)
        if spec is None or spec.loader is None:
            logger.error("Could not load claude_worker module")
            return

        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        worker = mod.ClaudeWorker(mode=mode)
        await worker.run()
    except Exception as e:
        logger.error("Worker daemon service failed: %s", e)
        raise


async def cloud_marketplace_service(
    head_manager: Any,
    settings: Any,
    agentic_core: Any,
    knowledge_store: Any,
    runtime_config: Any,
    on_activity: Any = None,
    shared_data: dict[str, Any] | None = None,
    # Pipeline infrastructure (enables full solve for complex contracts)
    event_store: Any = None,
    artifact_store: Any = None,
    acp_bridge: Any = None,
) -> None:
    """Cloud marketplace bridge as an async service.

    Connects to BotVibes cloud marketplace to auto-quote on RFQs,
    execute awarded contracts, and build trust score.
    """
    import os
    from ..cloud_marketplace import CloudMarketplaceBridge

    cloud_url = os.environ.get("ACP_CLOUD_URL")
    cloud_key = os.environ.get("ACP_CLOUD_API_KEY")
    cloud_project = os.environ.get("ACP_CLOUD_PROJECT_ID", "")
    cloud_agent = os.environ.get("ACP_CLOUD_AGENT_ID", "multihead-cloud-agent")
    cloud_email = os.environ.get("ACP_CLOUD_EMAIL", "")
    cloud_password = os.environ.get("ACP_CLOUD_PASSWORD", "")

    if not cloud_url or not cloud_key:
        logger.warning(
            "Cloud marketplace: ACP_CLOUD_URL or ACP_CLOUD_API_KEY not set — "
            "service unavailable. Set these in .env to enable."
        )
        return

    bridge = CloudMarketplaceBridge(
        head_manager=head_manager,
        settings=settings,
        cloud_url=cloud_url,
        cloud_api_key=cloud_key,
        cloud_project_id=cloud_project,
        cloud_agent_id=cloud_agent,
        agentic_core=agentic_core,
        knowledge_store=knowledge_store,
        runtime_config=runtime_config,
        event_store=event_store,
        artifact_store=artifact_store,
        runs_dir=getattr(settings, "runs_dir", None),
        acp_bridge=acp_bridge,
    )
    # Set login credentials for auto re-authentication
    bridge._cloud_email = cloud_email
    bridge._cloud_password = cloud_password
    bridge.on_activity = on_activity

    # Expose stats for /status visibility
    if shared_data is not None:
        shared_data["marketplace_stats"] = bridge._stats

    try:
        await bridge.run()
    except asyncio.CancelledError:
        logger.info("Cloud marketplace service stopped")
        raise
    finally:
        try:
            await bridge.stop()
        except Exception as e:
            logger.warning("Error stopping cloud marketplace bridge: %s", e)


async def event_watcher_service(
    knowledge_store: Any,
    session_id: str,
    project_id: str,
    config: Any,
) -> None:
    """Event watcher as an async service within shell.

    Polls ACP and knowledge.db for incoming events, enqueuing
    them for the shell to display between user prompts.
    """
    from ..event_watcher import EventWatcher

    ew_cfg = getattr(getattr(config, "pipeline", None), "event_watcher", None)
    poll_interval = getattr(ew_cfg, "poll_interval", 15) if ew_cfg else 15
    watch_acp = getattr(ew_cfg, "watch_acp", True) if ew_cfg else True
    watch_knowledge = getattr(ew_cfg, "watch_knowledge", True) if ew_cfg else True

    from ..shell_pipeline import AGENT_ID
    watcher = EventWatcher(
        knowledge_store=knowledge_store,
        session_id=session_id,
        project_id=project_id,
        poll_interval=poll_interval,
        watch_acp=watch_acp,
        watch_knowledge=watch_knowledge,
        agent_id=AGENT_ID,
    )
    await watcher.run()


async def session_harvester_service(
    knowledge_store: Any,
    config: Any,
    claude_home: str = "~/.claude",
    data_dir: str | None = None,
) -> None:
    """Session harvester as an async service within shell.

    Periodically scans all Claude Code project folders, reads MEMORY.md
    and CLAUDE.md files, and deposits extracted claims into knowledge.db.
    """
    from ..session_harvester import SessionHarvester

    interval = getattr(config, "harvester_interval", 300)
    max_claims = getattr(config, "harvester_max_claims", 100)

    harvester = SessionHarvester(
        knowledge_store=knowledge_store,
        claude_home=claude_home,
        data_dir=data_dir,
        max_claims_per_project=max_claims,
    )

    logger.info("Session harvester service started (interval=%ds)", interval)

    try:
        while True:
            try:
                result = harvester.harvest_all()
                if result.claims_deposited > 0:
                    logger.info(
                        "Session harvester: %d projects, %d claims deposited",
                        result.projects_harvested, result.claims_deposited,
                    )
                if result.errors:
                    for err in result.errors:
                        logger.warning("Harvest error: %s", err)
            except Exception as e:
                logger.warning("Session harvester poll error: %s", e)

            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        logger.info("Session harvester service stopped")
        raise
