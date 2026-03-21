"""Ingest and manage raw records (evidence sources)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multihead.artifact_store import ArtifactStore
from multihead.knowledge_models import Record
from multihead.knowledge_store import KnowledgeStore


class RecordStore:
    """Ingest raw records into the knowledge store with content in the artifact store."""

    def __init__(self, knowledge_store: KnowledgeStore, artifact_store: ArtifactStore) -> None:
        self.knowledge = knowledge_store
        self.artifacts = artifact_store

    def ingest_file(self, path: Path, mime: str = "") -> Record:
        """Ingest a file as a record. Content stored in artifact store."""
        data = path.read_bytes()
        ref = self.artifacts.store(data, name=path.name, media_type=mime)
        record = Record(
            uri=f"artifact://{ref.artifact_id}",
            sha256=ref.artifact_id.replace("sha256:", ""),
            mime=mime or _guess_mime(path),
        )
        self.knowledge.insert_record(record)
        return record

    def ingest_text(self, text: str, uri: str = "", mime: str = "text/plain") -> Record:
        """Ingest raw text as a record."""
        data = text.encode("utf-8")
        ref = self.artifacts.store(data, name="text_record", media_type=mime)
        record = Record(
            uri=uri or f"artifact://{ref.artifact_id}",
            sha256=ref.artifact_id.replace("sha256:", ""),
            mime=mime,
        )
        self.knowledge.insert_record(record)
        return record

    def ingest_chat_log(self, messages: list[dict[str, Any]], session_id: str = "") -> Record:
        """Ingest a chat session as a JSONL record."""
        lines = [json.dumps(m, default=str) for m in messages]
        text = "\n".join(lines)
        data = text.encode("utf-8")
        ref = self.artifacts.store(data, name=f"chat_{session_id}.jsonl", media_type="application/jsonl")
        record = Record(
            uri=f"artifact://{ref.artifact_id}",
            sha256=ref.artifact_id.replace("sha256:", ""),
            mime="application/jsonl",
        )
        self.knowledge.insert_record(record)
        return record

    def ingest_tool_output(self, tool_name: str, output: dict[str, Any], run_id: str = "") -> Record:
        """Ingest tool/run output as a JSON record."""
        data = json.dumps(output, default=str, indent=2).encode("utf-8")
        name = f"tool_{tool_name}_{run_id}.json" if run_id else f"tool_{tool_name}.json"
        ref = self.artifacts.store(data, name=name, media_type="application/json")
        record = Record(
            uri=f"artifact://{ref.artifact_id}",
            sha256=ref.artifact_id.replace("sha256:", ""),
            mime="application/json",
        )
        self.knowledge.insert_record(record)
        return record

    def get_records_since(self, since: datetime, limit: int = 10_000) -> list[Record]:
        """Get all records ingested since a timestamp."""
        return self.knowledge.list_records(since=since, limit=limit)

    def count_new_records(self, since: datetime) -> int:
        """Count records since a timestamp."""
        return self.knowledge.count_records_since(since)

    def get_content(self, record: Record) -> bytes | None:
        """Retrieve the raw content for a record from the artifact store."""
        if record.sha256:
            return self.artifacts.fetch(f"sha256:{record.sha256}")
        return None


def _guess_mime(path: Path) -> str:
    suffix_map = {
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".json": "application/json",
        ".jsonl": "application/jsonl",
        ".py": "text/x-python",
        ".yaml": "text/yaml",
        ".yml": "text/yaml",
        ".csv": "text/csv",
        ".html": "text/html",
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }
    return suffix_map.get(path.suffix.lower(), "application/octet-stream")
