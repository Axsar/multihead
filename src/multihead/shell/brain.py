"""Brain routing mixin — local GPU and Claude SDK integration."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .prompts import BRAIN_CLAUDE, BRAIN_DUAL, BRAIN_LOCAL

# Fast brain system prompt — must be ≤50 tokens to keep latency low.
# Instructs the local model to give a quick answer AND rewrite the query
# for deeper downstream analysis.
_FAST_BRAIN_SYSTEM_PROMPT = (
    "You are a fast assistant. Respond briefly. "
    "Also rewrite the user's query in one sentence for deeper analysis. "
    "Format exactly:\nRESPONSE: <your brief answer>\nENRICHED: <rewritten query>"
)

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
        if self._brain == BRAIN_DUAL:
            return self._chat_dual_brain
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

    async def _chat_dual_brain(
        self, session_id: str, user_input: str, knowledge_ctx: str = "",
    ) -> str:
        """Dual-brain: System 1 (fast local) enriches, System 2 (slow Claude) answers deep.

        Flow:
        1. Fast head generates a quick response + enriched query (~1-2s)
        2. [quick] response is displayed immediately
        3. Claude receives original + enriched context for deep analysis
        4. [deep] response is returned for display by caller

        Falls back to _chat_via_claude if fast head is unavailable.
        """
        if not self._claude_adapter:
            return await self._chat_via_local(session_id, user_input, knowledge_ctx)

        # --- Step 1: Fast brain call ---
        fast_head_id = getattr(self, "_fast_head_id", None)
        quick_response = ""
        enriched_query = user_input  # fallback: pass original if fast head fails

        if fast_head_id:
            try:
                fast_adapter = self.hm.get_adapter(fast_head_id)
                fast_prompt = (
                    f"{_FAST_BRAIN_SYSTEM_PROMPT}\n\nUser: {user_input}"
                )
                fast_result = await fast_adapter.generate(fast_prompt)
                raw = fast_result.get("text", "") if isinstance(fast_result, dict) else str(fast_result)

                # Parse RESPONSE: ... ENRICHED: ... format
                resp_marker = "RESPONSE:"
                enrich_marker = "ENRICHED:"
                if resp_marker in raw and enrich_marker in raw:
                    resp_start = raw.index(resp_marker) + len(resp_marker)
                    enrich_start = raw.index(enrich_marker)
                    quick_response = raw[resp_start:enrich_start].strip()
                    enriched_query = raw[enrich_start + len(enrich_marker):].strip()
                else:
                    # Unparseable response — use it as-is for quick, keep original query
                    quick_response = raw.strip()
                    enriched_query = user_input

            except Exception as e:
                logger.warning("Fast brain failed, skipping enrichment: %s", e)
                quick_response = ""
                enriched_query = user_input

        # --- Step 2: Display quick response ---
        if quick_response:
            self._tui_print(
                f"[bold yellow]\\[quick][/bold yellow] {quick_response}\n"
            )
            if getattr(self, "_debug_enrichment", False):
                self._tui_print(
                    f"[dim]\\[enriched query] {enriched_query}[/dim]\n"
                )

        # --- Step 3: Slow brain (Claude) with enriched context ---
        # Build enriched input: original question + fast-brain context
        if enriched_query != user_input:
            deep_input = f"{user_input}\n[Fast context: {enriched_query}]"
        else:
            deep_input = user_input

        return await self._chat_via_claude(session_id, deep_input, knowledge_ctx)

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

            import os as _os
            logger.info(
                "Claude brain call: CLAUDECODE=%s, prompt_len=%d, sys_len=%d, conv_id=%s",
                _os.environ.get("CLAUDECODE", "NOT_SET"),
                len(user_input),
                len(system_prompt),
                self._claude_conversation_id,
            )
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
        if mode not in (BRAIN_LOCAL, BRAIN_CLAUDE, BRAIN_DUAL):
            return f"Unknown brain mode: {mode}. Use 'local', 'claude', or 'dual'."

        if mode == self._brain:
            return f"Already using {mode} brain."

        if mode in (BRAIN_CLAUDE, BRAIN_DUAL):
            if not self._claude_adapter:
                return "Claude adapter not configured. Start shell with --brain claude or --brain dual."
            ready = await self._ensure_claude_ready()
            if not ready:
                return "Failed to switch to Claude brain (adapter not ready)."
            self._brain = mode
            self._codebase_ctx_cache = None  # rebuild on brain switch
            if mode == BRAIN_DUAL:
                fast = getattr(self, "_fast_head_id", None) or "not configured"
                return (
                    f"Switched to dual brain (fast={fast} → slow=Claude SDK). "
                    "Use /brain local or /brain claude to exit dual mode."
                )
            return "Switched to Claude brain (Claude Agent SDK)."
        else:
            self._brain = BRAIN_LOCAL
            return "Switched to local brain (AgenticCore + local GPU)."
