"""Session capture — read Claude SDK JSONL transcripts and export/ingest.

Reads the JSONL files that Claude Code CLI stores in
~/.claude/projects/<project>/<session-id>.jsonl and provides:

1. Export to markdown (full conversation transcript)
2. Ingest into knowledge.db (extract facts/decisions as claims)
3. Session listing and metadata

The JSONL format has these record types:
- queue-operation: enqueue/dequeue markers
- user: user messages (role=user, content=string)
- assistant: assistant responses (role=assistant, content=list of text/tool_use blocks)
- system: system records (subtype: turn_duration, compact_boundary)
- progress: streaming progress events
- file-history-snapshot: file state snapshots
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default Claude Code project sessions directory — auto-detected from cwd
def _detect_sessions_dir() -> Path:
    """Derive Claude project folder from current working directory."""
    cwd = str(Path.cwd())
    folder_name = cwd.replace("/", "-").replace("\\", "-")
    if folder_name.startswith("-"):
        pass  # already starts with dash
    else:
        folder_name = "-" + folder_name
    return Path.home() / ".claude" / "projects" / folder_name

_DEFAULT_SESSIONS_DIR = _detect_sessions_dir()


class SessionRecord:
    """A single record from a JSONL session file."""

    __slots__ = ("type", "subtype", "role", "content", "timestamp", "uuid",
                 "session_id", "tools_used", "model", "raw")

    def __init__(self, data: dict[str, Any]) -> None:
        self.type = data.get("type", "")
        self.subtype = data.get("subtype", "")
        self.session_id = data.get("sessionId", "")
        self.uuid = data.get("uuid", "")
        self.timestamp = data.get("timestamp", "")
        self.raw = data

        msg = data.get("message", {})
        self.role = msg.get("role", "")
        self.model = msg.get("model", "")

        # Extract content
        raw_content = msg.get("content", data.get("content", ""))
        if isinstance(raw_content, str):
            self.content = raw_content
            self.tools_used = []
        elif isinstance(raw_content, list):
            texts = []
            tools = []
            for block in raw_content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        texts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        tools.append(block.get("name", "unknown"))
            self.content = "\n".join(texts)
            self.tools_used = tools
        else:
            self.content = ""
            self.tools_used = []

    @property
    def is_user(self) -> bool:
        return self.type == "user"

    @property
    def is_assistant(self) -> bool:
        return self.type == "assistant"

    @property
    def is_compact_boundary(self) -> bool:
        return self.type == "system" and self.subtype == "compact_boundary"

    @property
    def has_text(self) -> bool:
        return bool(self.content.strip())


class SessionCapture:
    """Reads and processes Claude SDK session JSONL files."""

    def __init__(self, sessions_dir: Path | None = None) -> None:
        self.sessions_dir = sessions_dir or _DEFAULT_SESSIONS_DIR

    def list_sessions(
        self,
        limit: int = 20,
        min_size: int = 1000,
    ) -> list[dict[str, Any]]:
        """List available session files, sorted by modification time (newest first)."""
        if not self.sessions_dir.exists():
            return []

        files = sorted(
            self.sessions_dir.glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        results = []
        for f in files[:limit * 2]:  # Over-fetch then filter
            size = f.stat().st_size
            if size < min_size:
                continue
            mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
            results.append({
                "session_id": f.stem,
                "path": str(f),
                "size_kb": round(size / 1024, 1),
                "modified": mtime.isoformat(),
            })
            if len(results) >= limit:
                break

        return results

    def read_session(self, session_id: str) -> list[SessionRecord]:
        """Read all records from a session JSONL file."""
        path = self.sessions_dir / f"{session_id}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Session not found: {path}")

        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    records.append(SessionRecord(data))
                except json.JSONDecodeError:
                    continue

        return records

    def extract_conversation(
        self,
        session_id: str,
        include_tools: bool = False,
    ) -> list[dict[str, str]]:
        """Extract user/assistant message pairs from a session.

        Returns list of dicts with keys: role, content, timestamp, tools.
        Skips progress events, queue operations, and empty messages.
        """
        records = self.read_session(session_id)
        messages = []

        for rec in records:
            if rec.is_user and rec.has_text:
                messages.append({
                    "role": "user",
                    "content": rec.content,
                    "timestamp": rec.timestamp,
                    "tools": [],
                })
            elif rec.is_assistant and (rec.has_text or (include_tools and rec.tools_used)):
                entry: dict[str, Any] = {
                    "role": "assistant",
                    "content": rec.content,
                    "timestamp": rec.timestamp,
                    "tools": rec.tools_used,
                }
                messages.append(entry)
            elif rec.is_compact_boundary:
                messages.append({
                    "role": "system",
                    "content": "[Context compacted]",
                    "timestamp": rec.timestamp,
                    "tools": [],
                })

        return messages

    def export_markdown(
        self,
        session_id: str,
        output_path: Path | None = None,
        include_tools: bool = True,
    ) -> Path:
        """Export a session as a markdown transcript.

        Returns the path to the written file.
        """
        messages = self.extract_conversation(session_id, include_tools=include_tools)

        if not messages:
            raise ValueError(f"Session {session_id} has no messages")

        lines = [
            f"# Session Transcript: {session_id}",
            f"Exported: {datetime.now(timezone.utc).isoformat()}",
            f"Messages: {len(messages)}",
            "",
            "---",
            "",
        ]

        for msg in messages:
            role = msg["role"].upper()
            ts = msg.get("timestamp", "")
            tools = msg.get("tools", [])

            lines.append(f"### {role}")
            if ts:
                lines.append(f"*{ts}*")
            lines.append("")

            if msg["content"]:
                lines.append(msg["content"])

            if tools:
                lines.append("")
                lines.append(f"Tools: {', '.join(tools)}")

            lines.append("")
            lines.append("---")
            lines.append("")

        content = "\n".join(lines)

        if output_path is None:
            output_dir = self.sessions_dir.parent / "exports"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{session_id}.md"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        logger.info("Exported session %s to %s (%d messages)", session_id, output_path, len(messages))
        return output_path

    def get_session_stats(self, session_id: str) -> dict[str, Any]:
        """Get statistics about a session."""
        records = self.read_session(session_id)

        user_msgs = [r for r in records if r.is_user and r.has_text]
        assistant_msgs = [r for r in records if r.is_assistant and r.has_text]
        tool_calls = [r for r in records if r.is_assistant and r.tools_used]
        compactions = [r for r in records if r.is_compact_boundary]

        # Unique tools used
        all_tools: set[str] = set()
        for r in records:
            if r.tools_used:
                all_tools.update(r.tools_used)

        # Time span
        timestamps = [r.timestamp for r in records if r.timestamp]
        first = timestamps[0] if timestamps else None
        last = timestamps[-1] if timestamps else None

        # Total text length
        total_chars = sum(len(r.content) for r in records if r.has_text)

        return {
            "session_id": session_id,
            "total_records": len(records),
            "user_messages": len(user_msgs),
            "assistant_messages": len(assistant_msgs),
            "tool_calls": len(tool_calls),
            "unique_tools": sorted(all_tools),
            "compactions": len(compactions),
            "first_timestamp": first,
            "last_timestamp": last,
            "total_chars": total_chars,
            "estimated_tokens": total_chars // 4,
        }


def ingest_session_to_knowledge(
    session_id: str,
    knowledge_store: Any,
    sessions_dir: Path | None = None,
    scope_id: str = "multihead",
    max_claims: int = 50,
) -> list[str]:
    """Extract key facts from a session transcript and ingest as claims.

    Extracts user messages that look like decisions, facts, or instructions
    and assistant messages that contain key findings. Returns list of claim IDs.

    This is a lightweight heuristic extractor — for deep extraction,
    use the narrative pipeline (claude_enhancer).
    """
    capture = SessionCapture(sessions_dir)
    messages = capture.extract_conversation(session_id)

    if not messages:
        return []

    claim_ids = []

    # Extract from user messages (decisions, instructions, facts)
    decision_markers = {
        "let's", "we should", "let me", "go ahead", "do it",
        "use ", "always", "never", "make sure", "important",
        "decision:", "plan:", "rule:", "constraint:",
    }

    for msg in messages:
        if len(claim_ids) >= max_claims:
            break

        content = msg["content"].strip()
        if not content or len(content) < 20:
            continue

        # User decisions/instructions
        if msg["role"] == "user":
            content_lower = content.lower()
            if any(marker in content_lower for marker in decision_markers):
                claim_id = _deposit_claim(
                    knowledge_store,
                    statement=content[:500],
                    claim_type="decision" if "decision" in content_lower else "fact",
                    scope_id=scope_id,
                    source=f"session:{session_id}",
                )
                if claim_id:
                    claim_ids.append(claim_id)

        # Assistant key findings (look for structured output)
        elif msg["role"] == "assistant":
            if len(content) > 100:
                # Extract first paragraph as a finding
                first_para = content.split("\n\n")[0].strip()
                if len(first_para) > 50 and len(first_para) < 500:
                    claim_id = _deposit_claim(
                        knowledge_store,
                        statement=first_para,
                        claim_type="fact",
                        scope_id=scope_id,
                        source=f"session:{session_id}",
                    )
                    if claim_id:
                        claim_ids.append(claim_id)

    logger.info("Ingested %d claims from session %s", len(claim_ids), session_id)
    return claim_ids


def _deposit_claim(
    knowledge_store: Any,
    statement: str,
    claim_type: str,
    scope_id: str,
    source: str,
) -> str | None:
    """Deposit a single claim into knowledge.db. Returns claim_id or None."""
    try:
        import time as _time
        from .knowledge_models import (
            Claim, ClaimCanonical, ClaimScope, ClaimStatus, ClaimType,
            EntityRef, Provenance, ScopeType, ValueObject,
        )

        ct = ClaimType(claim_type) if claim_type in ClaimType._value2member_map_ else ClaimType.FACT
        now = datetime.now(timezone.utc)

        claim = Claim(
            claim_status=ClaimStatus.PROPOSED,
            claim_type=ct,
            scope=ClaimScope(
                scope_type=ScopeType.PROJECT,
                scope_id=scope_id,
                valid_from=now,
            ),
            canonical=ClaimCanonical(
                claim_key=f"session.capture.{_time.time():.0f}",
                subject=EntityRef(entity_type="session", entity_id=source),
                predicate="states",
                object=ValueObject(value_type="string", value=True),
            ),
            statement=statement,
            confidence=0.7,
            provenance=Provenance(
                produced_by={"kind": "extractor", "id": f"session_capture:{source}"},
            ),
        )
        claim = knowledge_store.insert_claim(claim)
        return claim.claim_id
    except Exception as e:
        logger.debug("Failed to deposit claim: %s", e)
        return None
