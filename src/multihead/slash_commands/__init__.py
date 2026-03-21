"""Slash command handler for interactive chat.

Commands are intercepted in the chat loop before reaching ac.chat().
The LLM never sees slash commands.

This package splits the monolithic SlashCommandHandler into logical
sub-modules (mixins), composed here into the final class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..runtime_config import RuntimeConfig
from ..tool_registry import ToolRegistry

from ._config import ConfigMixin
from ._events_collab import EventsCollabMixin
from ._heads import HeadsMixin
from ._help import HelpMixin
from ._knowledge import KnowledgeMixin
from ._process import ProcessMixin
from ._solve import SolveMixin
from ._status import StatusMixin


class SlashCommandHandler(
    ConfigMixin,
    HeadsMixin,
    StatusMixin,
    KnowledgeMixin,
    ProcessMixin,
    SolveMixin,
    EventsCollabMixin,
    HelpMixin,
):
    """Dispatches /commands typed in the chat REPL."""

    def __init__(
        self,
        config: RuntimeConfig,
        config_path: Path,
        tool_registry: ToolRegistry,
        head_states_fn: Callable[[], dict[str, dict[str, Any]]],
        knowledge_store: Any | None = None,
        session_id: str | None = None,
        project_id: str = "multihead",
        head_manager: Any | None = None,
        process_manager: Any | None = None,
        shell: Any | None = None,
        service_manager: Any | None = None,
        event_store: Any | None = None,
        artifact_store: Any | None = None,
        runs_dir: Any | None = None,
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.tools = tool_registry
        self._head_states_fn = head_states_fn
        self.knowledge_store = knowledge_store
        self.session_id = session_id
        self.project_id = project_id
        self.head_manager = head_manager
        self.process_manager = process_manager
        self.shell = shell  # Shell reference for brain-swap
        self.service_manager = service_manager
        self.event_store = event_store
        self.artifact_store = artifact_store
        self.runs_dir = runs_dir
        self.pipeline = None  # ShellPipeline reference (set externally)

    @staticmethod
    def is_slash_command(text: str) -> bool:
        return text.startswith("/")

    async def handle(self, text: str) -> str:
        """Handle a slash command. Returns output text."""
        parts = text.strip().split()
        cmd = parts[0].lower()
        args = parts[1:]

        match cmd:
            case "/config":
                return await self._handle_config(args)
            case "/tools":
                return self._handle_tools(args)
            case "/heads":
                return self._handle_heads()
            case "/wake":
                return await self._handle_wake(args)
            case "/sleep":
                return await self._handle_sleep(args)
            case "/swap":
                return await self._handle_swap(args)
            case "/status":
                return self._handle_status()
            case "/dashboard" | "/dash":
                await self._handle_dashboard(args)
                return ""
            case "/knowledge":
                return self._handle_knowledge(args)
            case "/session":
                return await self._handle_session(args)
            case "/sessions":
                return self._handle_sessions()
            case "/mesh":
                return self._handle_mesh()
            case "/spawn":
                return await self._handle_spawn(args)
            case "/ps":
                return self._handle_ps()
            case "/output":
                return self._handle_output(args)
            case "/kill":
                return self._handle_kill(args)
            case "/brain":
                return await self._handle_brain(args)
            case "/pipeline":
                return self._handle_pipeline(args)
            case "/services":
                return await self._handle_services(args)
            case "/collab":
                return await self._handle_collab(args)
            case "/collab-respond":
                return await self._handle_collab_respond(args)
            case "/collab-ignore":
                return await self._handle_collab_ignore(args)
            case "/model":
                return self._handle_model(args)
            case "/verbose":
                return self._handle_verbose()
            case "/events":
                return await self._handle_events(args)
            case "/responsive":
                return self._handle_responsive()
            case "/solve":
                return await self._handle_solve(args)
            case "/resolve":
                return self._handle_resolve(args)
            case "/harvest":
                return await self._handle_harvest(args)
            case "/route":
                return await self._handle_route(args)
            case "/discover":
                return await self._handle_discover(args)
            case "/select":
                return await self._handle_select(args)
            case "/learn":
                return await self._handle_learn(args)
            case "/help":
                return self._handle_help()
            case _:
                return f"Unknown command: {cmd}. Type /help for available commands."

    def _save_config(self) -> None:
        """Persist config to disk."""
        try:
            self.config.save(self.config_path)
        except Exception:
            pass  # Non-critical


__all__ = ["SlashCommandHandler"]
