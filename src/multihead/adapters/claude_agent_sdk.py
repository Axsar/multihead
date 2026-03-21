"""Claude Agent SDK adapter — runs Claude Code as a native MultiHead head.

Uses the official claude-agent-sdk package to spawn Claude Code sessions
with typed Python messages, async streaming, session resume, and in-process
MCP tool injection. This is the "Claude brain" option for multihead shell.

Requires: pip install claude-agent-sdk
Requires: Claude Code CLI installed (bundled with claude-agent-sdk >= 0.1.0)
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from ..models import HeadManifest
from .base import HeadAdapter

# Strip CLAUDECODE at import time — prevents "nested session" detection
# when multihead shell is launched from a terminal with Claude Code active.
# Must happen before the SDK subprocess transport snapshots os.environ.
os.environ.pop("CLAUDECODE", None)

logger = logging.getLogger(__name__)

# Default timeouts / limits
_DEFAULT_MAX_TURNS = 25
_DEFAULT_MAX_BUDGET_USD = 5.0

# SDK imports — optional dependency, gracefully absent
try:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ClaudeSDKClient,
        ResultMessage,
        TextBlock,
        query,
    )

    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False
    # Placeholders for type checking
    AssistantMessage = None  # type: ignore[assignment,misc]
    ClaudeAgentOptions = None  # type: ignore[assignment,misc]
    ClaudeSDKClient = None  # type: ignore[assignment,misc]
    ResultMessage = None  # type: ignore[assignment,misc]
    TextBlock = None  # type: ignore[assignment,misc]
    query = None  # type: ignore[assignment]


class ClaudeAgentSDKAdapter(HeadAdapter):
    """Adapter wrapping Claude Agent SDK as a MultiHead head.

    Unlike subprocess-based adapters, this uses the official Python SDK
    for typed messages, streaming, session management, and hooks.

    Configuration via manifest.extra:
        model: str          — Claude model (default: from env or "claude-sonnet-4-20250514")
        max_turns: int      — Max agent turns per call (default: 25)
        max_budget_usd: float — Budget cap per call (default: 5.0)
        permission_mode: str — "bypassPermissions" | "acceptEdits" | "default"
        cwd: str            — Working directory for file operations
        system_prompt: str  — Custom system prompt (appended)
        allowed_tools: list — Tools Claude can use (default: all)
        effort: str         — "low" | "medium" | "high" | "max"
    """

    def __init__(self, manifest: HeadManifest) -> None:
        super().__init__(manifest)
        extra = manifest.extra or {}

        self._model = extra.get("model") or manifest.model or os.environ.get(
            "CLAUDE_MODEL", "claude-opus-4-6"
        )
        self._max_turns = extra.get("max_turns", _DEFAULT_MAX_TURNS)
        self._max_budget = extra.get("max_budget_usd", _DEFAULT_MAX_BUDGET_USD)
        self._permission_mode = extra.get("permission_mode", "bypassPermissions")
        self._cwd = extra.get("cwd", os.getcwd())
        self._system_prompt = extra.get("system_prompt")
        self._allowed_tools = extra.get("allowed_tools", [])
        self._effort = extra.get("effort", "high")
        self._thinking = extra.get("thinking", False)
        self._max_thinking_tokens = extra.get("max_thinking_tokens")

        # SDK components (lazy-initialized on load)
        self._sdk_available = False
        self._loaded = False
        self._mcp_servers: dict[str, Any] = {}

        # Session tracking — maps conversation IDs to Claude session IDs
        self._sessions: dict[str, str] = {}

    async def load(self) -> None:
        """Verify SDK is installed and Claude CLI is reachable."""
        if not _SDK_AVAILABLE:
            raise RuntimeError(
                "claude-agent-sdk not installed. Run: pip install claude-agent-sdk"
            )
        self._sdk_available = True

        # Prevent "nested session" detection when launched from Claude Code
        os.environ.pop("CLAUDECODE", None)

        logger.info(
            "Claude Agent SDK adapter ready: model=%s, max_turns=%d, budget=$%.2f",
            self._model, self._max_turns, self._max_budget,
        )
        self._loaded = True

    async def unload(self) -> None:
        """Clean up — no persistent resources to free."""
        self._loaded = False
        self._sessions.clear()
        logger.info("Claude Agent SDK adapter unloaded")

    def set_mcp_servers(self, servers: dict[str, Any]) -> None:
        """Inject in-process MCP servers for Claude to call.

        These are exposed directly to the Claude session with zero IPC overhead.
        Use create_sdk_mcp_server() from claude_agent_sdk to create them.
        """
        self._mcp_servers = servers

    def _build_options(self, **kwargs: Any) -> Any:
        """Build ClaudeAgentOptions from adapter config + per-call overrides."""
        opts = ClaudeAgentOptions(
            model=kwargs.pop("model", self._model),
            max_turns=kwargs.pop("max_turns", self._max_turns),
            max_budget_usd=kwargs.pop("max_budget_usd", self._max_budget),
            permission_mode=kwargs.pop("permission_mode", self._permission_mode),
            cwd=kwargs.pop("cwd", self._cwd),
            effort=kwargs.pop("effort", self._effort),
        )

        # System prompt
        sys_prompt = kwargs.pop("system_prompt", self._system_prompt)
        if sys_prompt:
            opts.system_prompt = sys_prompt

        # Tools
        allowed = kwargs.pop("allowed_tools", self._allowed_tools)
        if allowed:
            opts.allowed_tools = allowed

        # MCP servers (in-process + any per-call)
        mcp = dict(self._mcp_servers)
        mcp.update(kwargs.pop("mcp_servers", {}))
        if mcp:
            opts.mcp_servers = mcp

        # Extended thinking — SDK expects {"type": "enabled", "budget_tokens": N}
        thinking = kwargs.pop("thinking", self._thinking)
        max_thinking = kwargs.pop("max_thinking_tokens", self._max_thinking_tokens)
        if thinking:
            budget = max_thinking or 32_000
            opts.thinking = {"type": "enabled", "budget_tokens": budget}
        elif max_thinking:
            opts.max_thinking_tokens = max_thinking

        # Session resume
        session_id = kwargs.pop("resume", None)
        if session_id:
            opts.resume = session_id

        return opts

    async def generate(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """One-shot generation via Claude Agent SDK query().

        Returns dict with 'text', 'tokens_in', 'tokens_out', 'session_id', etc.
        Automatically resumes existing sessions for the same conversation_id.
        """
        if not self._loaded:
            raise RuntimeError("Claude Agent SDK adapter not loaded")

        # Pop conversation_id before _build_options (same pattern as chat())
        conv_id = kwargs.pop("conversation_id", "default")
        existing_session = self._sessions.get(conv_id)

        opts = self._build_options(**kwargs)

        # Auto-resume existing session
        if existing_session and not opts.resume:
            opts.resume = existing_session

        text_parts: list[str] = []
        session_id = ""
        cost = 0.0
        turns = 0

        async for message in query(prompt=prompt, options=opts):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)
            elif isinstance(message, ResultMessage):
                session_id = message.session_id
                cost = message.total_cost_usd or 0.0
                turns = message.num_turns
                if message.is_error and message.result:
                    text_parts.append(f"[ERROR] {message.result}")

        # Log limit warnings
        if turns >= self._max_turns:
            logger.warning(
                "LIMIT HIT: max_turns reached (%d/%d) — response may be truncated",
                turns, self._max_turns,
            )
        if cost >= self._max_budget:
            logger.warning(
                "LIMIT HIT: max_budget reached ($%.2f/$%.2f) — response may be truncated",
                cost, self._max_budget,
            )
        elif cost >= self._max_budget * 0.8:
            logger.warning(
                "LIMIT WARNING: approaching budget cap ($%.2f/$%.2f, %.0f%%)",
                cost, self._max_budget, (cost / self._max_budget) * 100,
            )
        if turns >= self._max_turns * 0.8 and turns < self._max_turns:
            logger.warning(
                "LIMIT WARNING: approaching turn cap (%d/%d, %.0f%%)",
                turns, self._max_turns, (turns / self._max_turns) * 100,
            )

        # Track session for resume
        if session_id:
            self._sessions[conv_id] = session_id

        result_text = "\n".join(text_parts) if text_parts else ""

        return {
            "text": result_text,
            "session_id": session_id,
            "cost_usd": cost,
            "turns": turns,
            "model": self._model,
            "finish_reason": "stop",
        }

    async def generate_stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Stream text chunks from Claude Agent SDK."""
        if not self._loaded:
            raise RuntimeError("Claude Agent SDK adapter not loaded")

        # Auto-resume existing session (same pattern as generate/chat)
        conv_id = kwargs.pop("conversation_id", "default")
        existing_session = self._sessions.get(conv_id)

        opts = self._build_options(**kwargs)
        if existing_session and not opts.resume:
            opts.resume = existing_session

        async for message in query(prompt=prompt, options=opts):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        yield block.text

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        """Multi-turn chat via ClaudeSDKClient with session continuity.

        If a previous session exists for this conversation, resumes it.
        Otherwise starts a new session.
        """
        if not self._loaded:
            raise RuntimeError("Claude Agent SDK adapter not loaded")

        conv_id = kwargs.pop("conversation_id", "default")
        existing_session = self._sessions.get(conv_id)

        opts = self._build_options(**kwargs)
        if existing_session and not opts.resume:
            opts.resume = existing_session

        # Extract last user message as the prompt
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break

        if not last_user_msg:
            return {"text": "", "session_id": existing_session or ""}

        text_parts: list[str] = []
        session_id = existing_session or ""
        cost = 0.0
        turns = 0

        async with ClaudeSDKClient(options=opts) as client:
            await client.query(last_user_msg)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            text_parts.append(block.text)
                elif isinstance(message, ResultMessage):
                    session_id = message.session_id
                    cost = message.total_cost_usd or 0.0
                    turns = message.num_turns

        if session_id:
            self._sessions[conv_id] = session_id

        return {
            "text": "\n".join(text_parts) if text_parts else "",
            "session_id": session_id,
            "cost_usd": cost,
            "turns": turns,
            "model": self._model,
            "finish_reason": "stop",
        }

    async def healthcheck(self) -> bool:
        """Check if SDK is available and CLI is reachable."""
        if not self._sdk_available:
            return False
        try:
            import shutil
            return shutil.which("claude") is not None
        except Exception:
            return False

    async def sleep(self, level: int = 1) -> None:
        """No-op — no GPU state. Sessions persist for resume."""
        pass

    async def wake(self) -> None:
        """Re-validate SDK availability."""
        await self.load()

    def get_session_id(self, conversation_id: str = "default") -> str | None:
        """Get the Claude session ID for a given conversation."""
        return self._sessions.get(conversation_id)
