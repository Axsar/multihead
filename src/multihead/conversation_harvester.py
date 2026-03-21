"""Conversation Harvester — extracts knowledge from Claude Code session transcripts.

Reads `.jsonl` session transcripts from `~/.claude/projects/`, filters noise
(tool_use, progress events, system reminders), and feeds clean user/assistant
exchanges into RecordStore for downstream Night Shift processing.

Scale: ~3,000 files, ~485 MB. Processes 50 files per run (most-recent-first),
line-by-line streaming, fast hash manifest to avoid re-processing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# Patterns to strip from kept text
_SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)
_BASE64_RE = re.compile(r"[A-Za-z0-9+/=]{100,}")
_TRACEBACK_RE = re.compile(
    r"Traceback \(most recent call last\):.*?(?=\n\n|\n[A-Z]|\Z)", re.DOTALL
)
_CONTENTS_OF_RE = re.compile(r"Contents of /[^\n]+\n.{1000,}?(?=\n\n|\Z)", re.DOTALL)


@dataclass
class SessionFile:
    """A discovered JSONL session transcript."""
    session_id: str
    path: Path
    project_name: str
    scope_id: str
    size_bytes: int
    modified_at: float


@dataclass
class Exchange:
    """One user→assistant turn extracted from a transcript."""
    user_text: str
    assistant_text: str
    timestamp: str
    session_id: str
    turn_index: int
    # Speaker-aware metadata (new)
    user_valence: str = "neutral"       # frustrated/correction/excited/neutral/low_signal
    user_intent: str = "context"        # directive/question/approval/rejection/context
    user_weight: float = 0.5            # 0.0-1.0 extraction priority
    model: str = ""                     # which model responded
    input_tokens: int = 0
    output_tokens: int = 0
    is_tool_result: bool = False        # True if "user" message was actually a tool result


@dataclass
class ConversationHarvestResult:
    """Summary of a conversation harvest run."""
    files_scanned: int = 0
    files_processed: int = 0
    files_skipped: int = 0
    exchanges_ingested: int = 0
    records_created: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class ConversationHarvester:
    """Harvests Claude Code session transcripts into RecordStore."""

    def __init__(
        self,
        record_store: Any,
        knowledge_store: Any,
        claude_home: str | Path = "~/.claude",
        data_dir: str | Path | None = None,
        max_files_per_run: int = 100000,
        max_exchange_chars: int = 5000,
    ) -> None:
        self._record_store = record_store
        self._knowledge_store = knowledge_store
        self._claude_home = Path(claude_home).expanduser()
        self._data_dir = Path(data_dir) if data_dir else Path.home() / ".multihead"
        self._max_files_per_run = max_files_per_run
        self._max_exchange_chars = max_exchange_chars
        self._manifest_path = self._data_dir / "sessions" / "conversation_manifest.json"

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def scan_sessions(self, project_filter: str | None = None) -> list[SessionFile]:
        """Find all .jsonl session files across all projects."""
        projects_dir = self._claude_home / "projects"
        if not projects_dir.exists():
            logger.warning("Claude projects dir not found: %s", projects_dir)
            return []

        sessions: list[SessionFile] = []
        for entry in sorted(projects_dir.iterdir()):
            if not entry.is_dir():
                continue
            if project_filter and project_filter not in entry.name:
                continue

            scope_id = self._derive_scope_id(entry.name)

            for jsonl in entry.glob("*.jsonl"):
                try:
                    stat = jsonl.stat()
                except OSError:
                    continue
                sessions.append(SessionFile(
                    session_id=jsonl.stem,
                    path=jsonl,
                    project_name=entry.name,
                    scope_id=scope_id,
                    size_bytes=stat.st_size,
                    modified_at=stat.st_mtime,
                ))

        return sessions

    # ------------------------------------------------------------------
    # Harvesting
    # ------------------------------------------------------------------

    def harvest_all(self) -> ConversationHarvestResult:
        """Process up to max_files_per_run session files, most-recent-first."""
        t0 = time.monotonic()
        result = ConversationHarvestResult()
        manifest = self._get_manifest()

        sessions = self.scan_sessions()
        result.files_scanned = len(sessions)

        # Sort: non-primary projects first (under-represented), then by recency.
        # This ensures smaller projects get processed before the largest
        # project's sessions exhaust the per-run limit.
        def _sort_key(s: SessionFile) -> tuple:
            is_multihead = "Multihead" in s.project_name or "multihead" in s.project_name
            return (is_multihead, -s.modified_at)

        sessions.sort(key=_sort_key)

        if len(sessions) > 10000:
            logger.warning("Conversation harvester scanning large session set: %d files", len(sessions))

        processed = 0
        for session in sessions:
            if processed >= self._max_files_per_run:
                break

            file_key = f"{session.project_name}/{session.session_id}"
            file_hash = self._file_hash_fast(session.path)

            # Skip if unchanged
            prev = manifest.get("files", {}).get(file_key, {})
            if prev.get("hash") == file_hash:
                result.files_skipped += 1
                continue

            try:
                exchanges, records = self._harvest_file(session)
                result.exchanges_ingested += exchanges
                result.records_created += records
                result.files_processed += 1
                processed += 1

                # Update manifest
                manifest.setdefault("files", {})[file_key] = {
                    "hash": file_hash,
                    "harvested_at": datetime.now(timezone.utc).isoformat(),
                    "exchanges": exchanges,
                    "records": records,
                    "scope_id": session.scope_id,
                }

            except Exception as e:
                msg = f"{session.session_id}: {e}"
                result.errors.append(msg)
                logger.warning("Harvest error for %s: %s", session.session_id, e)
                processed += 1  # Count toward limit even on error

        manifest["last_harvest"] = datetime.now(timezone.utc).isoformat()
        self._save_manifest(manifest)

        result.duration_seconds = round(time.monotonic() - t0, 2)
        logger.info(
            "Conversation harvest: %d scanned, %d processed, %d exchanges, %d records, %.1fs",
            result.files_scanned, result.files_processed,
            result.exchanges_ingested, result.records_created,
            result.duration_seconds,
        )
        return result

    def _harvest_file(self, session: SessionFile) -> tuple[int, int]:
        """Process one JSONL file. Returns (exchanges_ingested, records_created)."""
        exchanges = 0
        records = 0

        for exchange in self._iter_exchanges(session.path, session.session_id):
            # Build record text
            text = self._format_exchange(exchange)
            if not text or len(text.strip()) < 100:
                continue  # Too short to be useful

            # Truncate if needed
            if len(text) > self._max_exchange_chars:
                text = text[:self._max_exchange_chars] + "\n[truncated]"

            uri = f"conversation://{session.project_name}/{session.session_id}/turn/{exchange.turn_index}"

            try:
                self._record_store.ingest_text(
                    text, uri=uri, mime="text/plain",
                )
                records += 1
            except Exception as e:
                # SHA-dedup will raise on duplicate content
                if "UNIQUE" in str(e) or "duplicate" in str(e).lower():
                    pass  # Expected for unchanged content
                else:
                    logger.debug("Ingest error for %s turn %d: %s",
                                 session.session_id, exchange.turn_index, e)
            exchanges += 1

        return exchanges, records

    # ------------------------------------------------------------------
    # JSONL Streaming & Exchange Extraction
    # ------------------------------------------------------------------

    def _iter_exchanges(self, filepath: Path, session_id: str) -> Iterator[Exchange]:
        """Stream through JSONL, yielding user→assistant exchange pairs.

        Speaker-aware: classifies user messages for valence/intent/weight,
        separates real user input from tool results, preserves model metadata.
        """
        from multihead.user_signals import classify_message

        current_user_text: list[str] = []
        current_assistant_text: list[str] = []
        current_timestamp = ""
        current_model = ""
        current_input_tokens = 0
        current_output_tokens = 0
        current_is_tool_result = False
        turn_index = 0
        in_assistant = False

        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue

                line_type = parsed.get("type", "")

                if self._should_skip_line(line_type):
                    continue

                if line_type == "user":
                    # If we were building an assistant response, yield the exchange
                    if in_assistant and current_user_text and current_assistant_text:
                        user_text = "\n".join(current_user_text)
                        signals = classify_message(user_text, is_user=True)
                        yield Exchange(
                            user_text=user_text,
                            assistant_text="\n".join(current_assistant_text),
                            timestamp=current_timestamp,
                            session_id=session_id,
                            turn_index=turn_index,
                            user_valence=signals["valence"],
                            user_intent=signals["intent"],
                            user_weight=signals["weight"],
                            model=current_model,
                            input_tokens=current_input_tokens,
                            output_tokens=current_output_tokens,
                            is_tool_result=current_is_tool_result,
                        )
                        turn_index += 1
                        current_user_text = []
                        current_assistant_text = []
                        current_model = ""
                        current_input_tokens = 0
                        current_output_tokens = 0
                        current_is_tool_result = False

                    in_assistant = False

                    # Distinguish real user input from tool results
                    is_tool = "toolUseResult" in parsed or "sourceToolAssistantUUID" in parsed
                    if is_tool:
                        current_is_tool_result = True

                    text = self._extract_text_from_message(parsed)
                    if text and not is_tool:
                        # Real user input — primary extraction target
                        current_user_text.append(text)
                    elif text and is_tool and len(text) <= 500:
                        # Short tool result — keep as context, tagged
                        current_user_text.append(f"[Tool result]: {text}")

                    if not current_timestamp:
                        current_timestamp = parsed.get("timestamp", "")

                elif line_type == "assistant":
                    in_assistant = True
                    text = self._extract_text_from_message(parsed)
                    if text:
                        current_assistant_text.append(text)

                    # Extract model and token metadata
                    message = parsed.get("message", {})
                    if message.get("model"):
                        current_model = message["model"]
                    usage = message.get("usage", {})
                    if usage.get("input_tokens"):
                        current_input_tokens += usage["input_tokens"]
                    if usage.get("output_tokens"):
                        current_output_tokens += usage["output_tokens"]

        # Yield final exchange
        if current_user_text and current_assistant_text:
            user_text = "\n".join(current_user_text)
            signals = classify_message(user_text, is_user=True)
            yield Exchange(
                user_text=user_text,
                assistant_text="\n".join(current_assistant_text),
                timestamp=current_timestamp,
                session_id=session_id,
                turn_index=turn_index,
                user_valence=signals["valence"],
                user_intent=signals["intent"],
                user_weight=signals["weight"],
                model=current_model,
                input_tokens=current_input_tokens,
                output_tokens=current_output_tokens,
                is_tool_result=current_is_tool_result,
            )

    @staticmethod
    def _should_skip_line(line_type: str) -> bool:
        """Filter out noise line types."""
        return line_type in (
            "progress", "queue-operation", "file-history-snapshot",
            "system", "last-prompt",
        )

    def _extract_text_from_message(self, parsed: dict) -> str:
        """Extract useful text from a user or assistant message."""
        message = parsed.get("message", {})
        content = message.get("content", "")

        # Simple string content
        if isinstance(content, str):
            return self._clean_text(content)

        # Array of content blocks
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                block_type = block.get("type", "")

                if block_type == "text":
                    text = block.get("text", "")
                    if text:
                        parts.append(text)

                elif block_type == "thinking":
                    text = block.get("thinking", "")
                    if text:
                        parts.append(f"[Thinking]: {text}")

                elif block_type == "tool_result":
                    # Only keep short tool results
                    result_content = block.get("content", "")
                    if isinstance(result_content, str) and len(result_content) <= 500:
                        parts.append(f"[Tool result]: {result_content}")
                    elif isinstance(result_content, list):
                        for sub in result_content:
                            if sub.get("type") == "text":
                                text = sub.get("text", "")
                                if len(text) <= 500:
                                    parts.append(f"[Tool result]: {text}")

                # Skip tool_use blocks entirely

            combined = "\n".join(parts)
            return self._clean_text(combined)

        return ""

    # ------------------------------------------------------------------
    # Text Cleaning
    # ------------------------------------------------------------------

    def _clean_text(self, text: str) -> str:
        """Strip noise from text blocks."""
        if not text:
            return ""

        # Remove system reminders
        text = _SYSTEM_REMINDER_RE.sub("", text)

        # Remove base64 data
        text = _BASE64_RE.sub("[base64-data]", text)

        # Remove long tracebacks (keep first line)
        def _shorten_traceback(m: re.Match) -> str:
            lines = m.group(0).split("\n")
            if len(lines) > 3:
                return lines[0] + "\n  [traceback truncated]\n" + lines[-1]
            return m.group(0)
        text = _TRACEBACK_RE.sub(_shorten_traceback, text)

        # Remove "Contents of /path/..." dumps > 1000 chars
        text = _CONTENTS_OF_RE.sub("[file-contents-removed]", text)

        # Remove large JSON blobs
        text = re.sub(r'\{[^{}]{500,}\}', '[large-json-removed]', text)

        # Collapse multiple blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    # ------------------------------------------------------------------
    # Exchange Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_exchange(exchange: Exchange) -> str:
        """Format an exchange for RecordStore ingestion.

        Preserves speaker identity and signals as structured metadata.
        """
        parts = []
        if exchange.timestamp:
            parts.append(f"[Timestamp: {exchange.timestamp}]")

        # Speaker-aware metadata header
        meta = []
        if exchange.user_valence != "neutral":
            meta.append(f"valence={exchange.user_valence}")
        if exchange.user_intent != "context":
            meta.append(f"intent={exchange.user_intent}")
        meta.append(f"weight={exchange.user_weight}")
        if exchange.model:
            meta.append(f"model={exchange.model}")
        if exchange.is_tool_result:
            meta.append("tool_result=true")
        parts.append(f"[Signals: {', '.join(meta)}]")

        parts.append(f"[User]: {exchange.user_text}")
        parts.append(f"[Assistant]: {exchange.assistant_text}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Manifest (fast hash tracking)
    # ------------------------------------------------------------------

    def _get_manifest(self) -> dict[str, Any]:
        """Load the conversation harvest manifest."""
        if self._manifest_path.exists():
            try:
                return json.loads(self._manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load conversation manifest: %s", e)
        return {"last_harvest": None, "files": {}}

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        """Persist the manifest."""
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8",
        )

    @staticmethod
    def _file_hash_fast(path: Path) -> str:
        """Fast hash based on mtime_ns and size — avoids reading large files."""
        try:
            stat = path.stat()
            return f"{stat.st_mtime_ns}:{stat.st_size}"
        except OSError:
            return ""

    @staticmethod
    def _derive_scope_id(folder_name: str) -> str:
        """Derive scope_id from project folder name."""
        name = folder_name.lower()
        for prefix in ("-mnt-d-devd-", "-mnt-c-dev-", "-mnt-d-", "-mnt-c-"):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        parts = [p for p in name.split("-") if p]
        if not parts:
            return "unknown"
        return parts[-1] if parts else "unknown"

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Return current harvester status."""
        manifest = self._get_manifest()
        sessions = self.scan_sessions()

        total_size = sum(s.size_bytes for s in sessions)
        processed = len(manifest.get("files", {}))
        total_exchanges = sum(
            f.get("exchanges", 0) for f in manifest.get("files", {}).values()
        )

        # Group by project
        by_project: dict[str, int] = {}
        for s in sessions:
            by_project[s.project_name] = by_project.get(s.project_name, 0) + 1

        return {
            "total_files": len(sessions),
            "total_size_mb": round(total_size / 1024 / 1024, 1),
            "files_processed": processed,
            "files_remaining": len(sessions) - processed,
            "total_exchanges": total_exchanges,
            "last_harvest": manifest.get("last_harvest"),
            "projects": len(by_project),
            "project_breakdown": by_project,
        }
