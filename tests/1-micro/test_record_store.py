"""Tests for the record store."""

from datetime import datetime, timedelta, timezone

import pytest

from multihead.artifact_store import ArtifactStore
from multihead.knowledge_store import KnowledgeStore
from multihead.record_store import RecordStore


@pytest.fixture
def stores(tmp_path):
    knowledge = KnowledgeStore(tmp_path / "knowledge.db")
    artifacts = ArtifactStore(tmp_path / "artifacts", tmp_path / "artifacts.db")
    record_store = RecordStore(knowledge, artifacts)
    return record_store, knowledge, artifacts


def test_ingest_text(stores):
    rs, ks, _ = stores
    rec = rs.ingest_text("Hello world", uri="test://hello")
    assert rec.record_id.startswith("rec_")
    assert rec.mime == "text/plain"
    fetched = ks.get_record(rec.record_id)
    assert fetched is not None


def test_ingest_file(stores, tmp_path):
    rs, ks, _ = stores
    f = tmp_path / "test.txt"
    f.write_text("File content here")
    rec = rs.ingest_file(f)
    assert rec.mime == "text/plain"
    assert rec.sha256 != ""


def test_ingest_chat_log(stores):
    rs, ks, _ = stores
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    rec = rs.ingest_chat_log(messages, session_id="s1")
    assert rec.mime == "application/jsonl"


def test_ingest_tool_output(stores):
    rs, ks, _ = stores
    output = {"text": "result", "tokens": 42}
    rec = rs.ingest_tool_output("llm.generate", output, run_id="run_123")
    assert rec.mime == "application/json"


def test_get_content(stores):
    rs, _, _ = stores
    rec = rs.ingest_text("Content to retrieve")
    content = rs.get_content(rec)
    assert content == b"Content to retrieve"


def test_count_new_records(stores):
    rs, _, _ = stores
    rs.ingest_text("One")
    rs.ingest_text("Two")
    since = datetime.now(timezone.utc) - timedelta(hours=1)
    assert rs.count_new_records(since) == 2


def test_get_records_since(stores):
    rs, _, _ = stores
    rs.ingest_text("Recent")
    since = datetime.now(timezone.utc) - timedelta(hours=1)
    records = rs.get_records_since(since)
    assert len(records) == 1
