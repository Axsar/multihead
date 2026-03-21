"""Session management for the Agentic Core."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field

from multihead.context_packs import PackBuilder
from multihead.models import new_id


class Message(BaseModel):
    """A single message in a session."""
    role: str  # user | assistant | system | tool
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class Session(BaseModel):
    """A conversation session with message history."""
    session_id: str = ""
    messages: list[Message] = Field(default_factory=list)
    context_pack_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def model_post_init(self, __context: Any) -> None:
        if not self.session_id:
            self.session_id = new_id("ses_")


class SessionManager:
    """Manages sessions: create, load, save, assemble context."""

    DEFAULT_MAX_MESSAGES = 500
    DEFAULT_MAX_MESSAGE_SIZE = 50_000  # 50KB per message

    def __init__(
        self,
        sessions_dir: Path,
        pack_builder: PackBuilder | None = None,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        max_message_size: int = DEFAULT_MAX_MESSAGE_SIZE,
    ) -> None:
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.pack_builder = pack_builder
        self.max_messages = max_messages
        self.max_message_size = max_message_size
        self._sessions: dict[str, Session] = {}

    def create_session(self) -> Session:
        """Create a new session."""
        session = Session()
        self._sessions[session.session_id] = session
        self.save_session(session)
        return session

    def add_message(self, session_id: str, role: str, content: str, **metadata: Any) -> Message:
        """Add a message to a session. Enforces size limits."""
        session = self.get_session(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")

        # Truncate oversized messages
        if len(content) > self.max_message_size:
            logger.warning(
                "Truncating message in session %s: %d bytes -> %d bytes",
                session_id, len(content), self.max_message_size,
            )
            content = content[:self.max_message_size]

        msg = Message(role=role, content=content, metadata=metadata)
        session.messages.append(msg)

        # Trim oldest messages if over limit (keep system messages)
        if len(session.messages) > self.max_messages:
            system_msgs = [m for m in session.messages if m.role == "system"]
            non_system = [m for m in session.messages if m.role != "system"]
            keep = max(0, self.max_messages - len(system_msgs))
            session.messages = system_msgs + non_system[-keep:] if keep > 0 else system_msgs

        self.save_session(session)
        return msg

    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID (from memory or disk)."""
        if session_id in self._sessions:
            return self._sessions[session_id]
        return self.load_session(session_id)

    def assemble_context(
        self,
        session_id: str,
        system_prompt: str = "",
        max_tokens: int = 8000,
    ) -> list[dict[str, str]]:
        """Assemble context: system prompt + packs + recent messages.

        Returns a list of messages in chat format.
        """
        session = self.get_session(session_id)
        if session is None:
            return []

        messages: list[dict[str, str]] = []

        # System prompt + pack summaries
        system_parts = [system_prompt] if system_prompt else []

        if self.pack_builder and session.context_pack_ids:
            for pack_id in session.context_pack_ids:
                pack = self.pack_builder.load_pack(pack_id)
                if pack:
                    pack_text = f"\n## {pack.purpose}\n"
                    for item in pack.items[:10]:
                        pack_text += f"- [{item.type}] {item.text}\n"
                    system_parts.append(pack_text)

        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})

        # Add recent messages, trimmed to budget
        token_budget = max_tokens - sum(len(m["content"]) // 4 for m in messages)
        recent = self._trim_messages(session.messages, token_budget)
        for msg in recent:
            messages.append({"role": msg.role, "content": msg.content})

        return messages

    def save_session(self, session: Session) -> None:
        """Save session to disk."""
        path = self.sessions_dir / f"{session.session_id}.json"
        path.write_text(
            json.dumps(session.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )
        self._sessions[session.session_id] = session

    def load_session(self, session_id: str) -> Session | None:
        """Load session from disk."""
        path = self.sessions_dir / f"{session_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        session = Session.model_validate(data)
        self._sessions[session_id] = session
        return session

    def delete_session(self, session_id: str) -> bool:
        """Delete a session from memory and disk."""
        self._sessions.pop(session_id, None)
        path = self.sessions_dir / f"{session_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all saved sessions."""
        sessions = []
        for path in sorted(self.sessions_dir.glob("ses_*.json"), reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                sessions.append({
                    "session_id": data.get("session_id", ""),
                    "message_count": len(data.get("messages", [])),
                    "created_at": data.get("created_at", ""),
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return sessions

    @staticmethod
    def _trim_messages(messages: list[Message], token_budget: int) -> list[Message]:
        """Keep most recent messages within token budget."""
        result: list[Message] = []
        used = 0
        for msg in reversed(messages):
            est = len(msg.content) // 4
            if used + est > token_budget:
                break
            result.insert(0, msg)
            used += est
        return result
