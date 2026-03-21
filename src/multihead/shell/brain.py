"""Brain routing mixin — local GPU and Claude SDK integration."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .prompts import BRAIN_CLAUDE, BRAIN_LOCAL

logger = logging.getLogger(__name__)


class BrainMixin:
    """Mixin providing brain routing methods for the Shell class.

    Expects the host class to provide:
    - self.ac (AgenticCore)
    - self.ks (KnowledgeStore)
    - self.sessions (SessionManager)
    - self.config (RuntimeConfig)
    - self._brain (str)
    - self._claude_adapter
    - self._claude_conversation_id (str)
    - self._conversation_ctx (ConversationContext)
    - self._codebase_ctx_cache (str | None)
    - self._verbose (bool)
    - self._last_meta (dict)
    - self.build_system_prompt() -> str
    - self._build_knowledge_context(msg) -> str
    - self._build_codebase_context() -> str
    - self._tui_print(...)
    """

    def _brain_fn_for_pipeline(self):
        """Return the appropriate brain function for the pipeline.

        Returns an async callable(session_id, user_input, knowledge_ctx) -> str.
        """
        if self._brain == BRAIN_CLAUDE:
            return self._chat_via_claude
        return self._chat_via_local

    async def _chat_via_local(
        self, session_id: str, user_input: str, knowledge_ctx: str = "",
    ) -> str:
        """Route through AgenticCore (local GPU model)."""
        if not knowledge_ctx:
            knowledge_ctx = self._build_knowledge_context(user_input)
        if knowledge_ctx:
            augmented = f"{knowledge_ctx}\nUser: {user_input}"
        else:
            augmented = user_input
        return await self.ac.chat(session_id, augmented)

    async def _chat_via_claude(
        self, session_id: str, user_input: str, knowledge_ctx: str = "",
    ) -> str:
        """Route through Claude Agent SDK with in-process MCP tools."""
        if not self._claude_adapter:
            return "[error] Claude adapter not configured. Use /brain local to switch back."

        try:
            if not knowledge_ctx:
                knowledge_ctx = self._build_knowledge_context(user_input)
            system_prompt = self.build_system_prompt()
            system_prompt += self._build_codebase_context()
            if knowledge_ctx:
                system_prompt += f"\n\n{knowledge_ctx}"

            # Inject conversation context (survives SDK compaction)
            conv_cfg = getattr(getattr(self.config, "pipeline", None), "conversation", None)
            if not conv_cfg or getattr(conv_cfg, "enabled", True):
                session = self.sessions.get_session(session_id) if self.sessions else None
                if session and getattr(session, "messages", None):
                    conv_block = self._conversation_ctx.build_context_block(session.messages)
                    if conv_block:
                        system_prompt += f"\n\n{conv_block}"

            result = await self._claude_adapter.generate(
                user_input,
                system_prompt=system_prompt,
                conversation_id=self._claude_conversation_id,
            )

            text = result.get("text", "")
            cost = result.get("cost_usd", 0)
            turns = result.get("turns", 0)

            # Track session for resume
            sid = result.get("session_id")
            if sid:
                logger.debug("Claude session: %s (cost=$%.4f, turns=%d)", sid, cost, turns)

            # Store metadata for verbose display
            self._last_meta = {
                "model": result.get("model", ""),
                "cost_usd": cost,
                "turns": turns,
                "session_id": sid or "",
                "conv_turns": self._conversation_ctx.turn_count + 1,
                "has_summary": bool(self._conversation_ctx.summary),
            }

            # Record in local session for history
            self.sessions.add_message(session_id, "user", user_input)
            self.sessions.add_message(session_id, "assistant", text)

            # Update conversation context tracker
            self._conversation_ctx.on_turn(user_input, text)
            if self._conversation_ctx.needs_summary_refresh():
                session = self.sessions.get_session(session_id) if self.sessions else None
                if session and getattr(session, "messages", None):
                    self._conversation_ctx.build_summary(session.messages)

            return text
        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info("Claude brain request cancelled by user")
            return "[cancelled]"
        except BaseException as e:
            # Unwrap ExceptionGroup / TaskGroup errors to show root cause
            cause = e
            if hasattr(e, 'exceptions'):
                cause = e.exceptions[0] if e.exceptions else e
            # Treat SIGINT-killed subprocess (exit code -2) as user cancellation
            err_str = str(cause)
            if "exit code -2" in err_str or "exit code: -2" in err_str:
                logger.info("Claude brain request cancelled by user")
                return "[cancelled]"
            logger.error("Claude brain error: %s", cause, exc_info=True)
            return f"[Claude error] {cause}"

    async def _ensure_claude_ready(self) -> bool:
        """Ensure Claude adapter is loaded and has MCP tools.

        Returns True if Claude is ready, False otherwise.
        """
        if not self._claude_adapter:
            self._tui_print("[yellow]No Claude adapter configured.[/yellow]")
            return False

        if not getattr(self._claude_adapter, "_loaded", False):
            try:
                await self._claude_adapter.load()
            except Exception as e:
                self._tui_print(f"[red]Failed to load Claude adapter: {e}[/red]")
                return False

        return True

    def _inject_mcp_tools(self) -> None:
        """Build and inject in-process MCP tools into Claude adapter.

        NOTE: Currently disabled — claude-agent-sdk v0.1.44 has a bug where
        any mcp_servers injection causes ProcessTransport crash. The tools
        are built and tested (see test_sdk_mcp_tools.py) but cannot be
        injected until the SDK fixes this. Claude still works as a brain
        without MCP tools — it just can't call back into MultiHead.
        """
        try:
            from ..sdk_mcp_tools import build_sdk_mcp_server
            server = build_sdk_mcp_server(
                knowledge_store=self.ks,
                head_manager=self.hm,
                process_manager=self.process_manager,
            )
            self._claude_adapter.set_mcp_servers({"multihead": server})
            logger.info("Injected MultiHead MCP tools into Claude adapter")
        except Exception as e:
            logger.warning("Could not inject MCP tools: %s", e)

    @property
    def brain(self) -> str:
        """Current brain mode."""
        return self._brain

    async def switch_brain(self, mode: str) -> str:
        """Switch brain mode. Returns status message."""
        if mode not in (BRAIN_LOCAL, BRAIN_CLAUDE):
            return f"Unknown brain mode: {mode}. Use 'local' or 'claude'."

        if mode == self._brain:
            return f"Already using {mode} brain."

        if mode == BRAIN_CLAUDE:
            if not self._claude_adapter:
                return "Claude adapter not configured. Start shell with --brain claude."
            ready = await self._ensure_claude_ready()
            if not ready:
                return "Failed to switch to Claude brain (adapter not ready)."
            self._brain = BRAIN_CLAUDE
            self._codebase_ctx_cache = None  # rebuild on brain switch
            return "Switched to Claude brain (Claude Agent SDK)."
        else:
            self._brain = BRAIN_LOCAL
            return "Switched to local brain (AgenticCore + local GPU)."
