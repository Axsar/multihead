"""ACP Bridge: connects MultiHead to BotVibes Agent Communication Protocol.

Three modes:
1. ACP mode — when BotVibes server is available, register agents,
   send heartbeats, poll for tasks, and accept work via ACP.
2. File mode — when ACP is offline, write capabilities + state to a JSON
   file that external tools (e.g. Claude Code) can read.
3. Proxy mode — expose a request() method for route handlers to forward
   Claude Code's ACP calls to BotVibes.

Registers two agent identities:
- multihead_orchestrator_v1: GPU inference (LLM/VLM tasks)
- claude_code_proxy: code editing/testing (tasks for Claude Code's inbox)
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

import httpx

from ..config import Settings
from ..head_manager import HeadManager
from ..skill_loader import SkillRegistry
from ._constants import ACP_TIMEOUT, logger
from ._networking import NetworkingMixin
from ._registration import RegistrationMixin
from ._tasks import TasksMixin


class ACPBridge(RegistrationMixin, TasksMixin, NetworkingMixin):
    """Bridges MultiHead to the BotVibes/ACP agent coordination layer.

    Registers MultiHead's heads as ACP capabilities and a proxy identity
    for Claude Code. Sends heartbeats, polls for tasks, and provides a
    request() method for proxy routes.
    """

    def __init__(
        self,
        head_manager: HeadManager,
        settings: Settings,
        acp_url: str | None = None,
        api_key: str | None = None,
        project_id: str | None = None,
        agent_id: str = "multihead-agent",
        claude_agent_id: str = "claude-session-agent",
        auto_execute: bool = True,
        skip_registration: bool = False,
        skill_registry: SkillRegistry | None = None,
        runtime_config: Any = None,
    ) -> None:
        self.heads = head_manager
        self.settings = settings
        self.skills = skill_registry or SkillRegistry()
        self.acp_url = acp_url.rstrip("/") if acp_url else None
        self._api_key = api_key
        self._project_id = project_id or "multihead-default"
        self._agent_id = agent_id
        self._claude_agent_id = claude_agent_id
        self._auto_execute = auto_execute
        self._runtime_config = runtime_config
        self._skip_registration = skip_registration
        self._artifact_store: Any = None  # Set via set_stores()
        self._knowledge_store: Any = None  # Set via set_stores()
        self._connected = False
        self._heartbeat_task: asyncio.Task | None = None
        self._task_poll_task: asyncio.Task | None = None
        self._ws_task: asyncio.Task | None = None
        self._token_refresh_task: asyncio.Task | None = None
        self._poll_now: asyncio.Event = asyncio.Event()
        self._agentic_core: Any = None  # Set after startup via set_agentic_core()
        self._on_task: Any = None  # Callback: intercept tasks instead of auto-executing

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Connect to ACP if available, otherwise fall back to file mode."""
        if self.acp_url and self._api_key:
            try:
                if not self._skip_registration:
                    last_err: Exception | None = None
                    for attempt in range(3):
                        try:
                            await self._register_agents()
                            last_err = None
                            break
                        except Exception as reg_err:
                            last_err = reg_err
                            if attempt < 2:
                                delay = 2 ** attempt  # 1s, 2s
                                logger.warning(
                                    "ACP registration attempt %d/3 failed: %s — retrying in %ds",
                                    attempt + 1, reg_err, delay,
                                )
                                await asyncio.sleep(delay)
                    if last_err is not None:
                        raise last_err
                self._connected = True
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                self._task_poll_task = asyncio.create_task(self._task_poll_loop())
                self._ws_task = asyncio.create_task(self._ws_doorbell_loop())
                self._token_refresh_task = asyncio.create_task(self._token_refresh_loop())
                logger.info(
                    "ACP bridge connected: %s (agents=%s, %s)",
                    self.acp_url, self._agent_id, self._claude_agent_id,
                )
            except Exception as e:
                logger.warning(
                    "ACP server not available at %s: %s — falling back to file mode",
                    self.acp_url, e,
                )
                self._connected = False
                self._write_capability_file()
        else:
            logger.info("ACP bridge in file mode (no ACP_URL or ACP_API_KEY configured)")
            self._write_capability_file()

    async def stop(self) -> None:
        """Stop background loops and clean up."""
        self._connected = False
        for task in (self._heartbeat_task, self._task_poll_task, self._ws_task, self._token_refresh_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._heartbeat_task = None
        self._task_poll_task = None
        self._ws_task = None
        self._token_refresh_task = None
        logger.info("ACP bridge stopped")

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def project_id(self) -> str:
        return self._project_id

    def set_agentic_core(self, core: Any) -> None:
        """Set reference to AgenticCore for task execution."""
        self._agentic_core = core

    def set_stores(self, artifact_store: Any, knowledge_store: Any) -> None:
        """Set references to ArtifactStore and KnowledgeStore for result delivery."""
        self._artifact_store = artifact_store
        self._knowledge_store = knowledge_store

    def set_on_task(self, callback: Any) -> None:
        """Set callback to intercept tasks instead of auto-executing.

        When set, ``_execute_task`` calls ``callback(task_dict)`` instead
        of reserving/executing/completing the task automatically. This
        lets the shell brain handle marketplace contracts properly.

        Set to ``None`` to restore auto-execute (headless) behavior.
        """
        self._on_task = callback

    # ------------------------------------------------------------------
    # Public proxy method (used by routes_acp.py)
    # ------------------------------------------------------------------

    async def request(self, method: str, path: str, **kwargs) -> Any:
        """Make an authenticated request to the BotVibes API.

        Used by ACP proxy routes to forward Claude Code's calls.
        """
        if not self.acp_url:
            raise RuntimeError("ACP not configured")
        async with httpx.AsyncClient(timeout=ACP_TIMEOUT) as client:
            resp = await client.request(
                method,
                f"{self.acp_url}{path}",
                headers=self._auth_headers(),
                **kwargs,
            )
            resp.raise_for_status()
            return resp.json()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def build_status(self) -> dict[str, Any]:
        """Build current status payload."""
        active = self.heads.active_head
        return {
            "agent_id": self._agent_id,
            "claude_agent_id": self._claude_agent_id,
            "status": "busy" if active else "idle",
            "active_head": active,
            "queue_depth": 0,
            "active_tasks": 1 if active else 0,
            "acp_connected": self._connected,
        }

    # ------------------------------------------------------------------
    # File-based fallback
    # ------------------------------------------------------------------

    def _write_capability_file(self) -> None:
        """Write capabilities and status to a JSON file for external tools."""
        state_file = self.settings.data_dir / "acp_state.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)

        heads_info: dict[str, str] = {}
        for head_id, state_info in self.heads.get_states().items():
            heads_info[head_id] = state_info.get("model", "unknown")

        state = {
            "agent_id": self._agent_id,
            "claude_agent_id": self._claude_agent_id,
            "status": "busy" if self.heads.active_head else "idle",
            "active_head": self.heads.active_head,
            "capabilities": self._build_multihead_descriptor(),
            "claude_capabilities": self._build_claude_descriptor(),
            "skills": {s.name: s.to_dict() for s in self.skills.list_skills()},
            "heads": heads_info,
            "endpoint": f"http://{self.settings.api_host}:{self.settings.api_port}",
            "acp_connected": self._connected,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        state_file.write_text(json.dumps(state, indent=2))
        logger.info("ACP state written to %s", state_file)

    def refresh_capability_file(self) -> None:
        """Refresh the capability file with current state. Called externally."""
        self._write_capability_file()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
