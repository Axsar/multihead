"""Tests for narrative extractor content storage via ArtifactStore.

Verifies that extractors store content bytes in the ArtifactStore
so that RecordStore.get_content() can retrieve them later (Night Shift).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from multihead.artifact_store import ArtifactStore
from multihead.knowledge_store import KnowledgeStore
from multihead.narrative.pipeline import NarrativePipeline
from multihead.narrative.source_extractors.agent_extractor import AgentExtractor
from multihead.narrative.source_extractors.chat_extractor import ChatExtractor
from multihead.narrative.source_extractors.git_extractor import GitExtractor
from multihead.record_store import RecordStore


@pytest.fixture
def artifact_store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts", tmp_path / "artifacts.db")


@pytest.fixture
def knowledge_store(tmp_path: Path) -> KnowledgeStore:
    return KnowledgeStore(tmp_path / "knowledge.db")


class TestAgentExtractorContent:
    def test_stores_content_in_artifact_store(self, artifact_store: ArtifactStore):
        ext = AgentExtractor(project_id="test", artifact_store=artifact_store)
        result = {"task": "analyze image", "output": "found 3 panels", "status": "success"}
        artifact = ext.extract_from_result(result, agent_id="qwen-vlm-01", agent_type="vlm")

        record = artifact["record"]
        assert record.sha256 is not None
        content = artifact_store.fetch(f"sha256:{record.sha256}")
        assert content is not None
        assert b"analyze image" in content

    def test_fallback_without_artifact_store(self):
        ext = AgentExtractor(project_id="test", artifact_store=None)
        result = {"task": "test task", "output": "done", "status": "success"}
        artifact = ext.extract_from_result(result, agent_id="test-agent")

        record = artifact["record"]
        assert record.sha256 is not None
        assert len(record.sha256) == 64  # Valid SHA-256 hex


class TestChatExtractorContent:
    def test_extract_from_messages_stores_content(self, artifact_store: ArtifactStore):
        ext = ChatExtractor(project_id="test", artifact_store=artifact_store)
        messages = [
            {"role": "user", "content": "Let's go with the new architecture design pattern"},
            {"role": "assistant", "content": "Agreed to use the event-sourced approach"},
        ]
        artifacts = ext.extract_from_messages(messages, session_id="sess-001")

        # At least the record should have stored content
        # Find a record from the artifacts
        records_seen = set()
        for art in artifacts:
            rec = art.get("record")
            if rec and rec.sha256 and rec.sha256 not in records_seen:
                records_seen.add(rec.sha256)
                content = artifact_store.fetch(f"sha256:{rec.sha256}")
                assert content is not None

        assert len(records_seen) > 0

    def test_extract_from_jsonl_stores_content(self, artifact_store: ArtifactStore, tmp_path: Path):
        ext = ChatExtractor(project_id="test", artifact_store=artifact_store)
        jsonl = tmp_path / "chat.jsonl"
        jsonl.write_text(
            '{"role": "user", "content": "Let\'s go with option A"}\n'
            '{"role": "assistant", "content": "Agreed to proceed with option A"}\n'
        )
        artifacts = ext.extract_from_jsonl(jsonl, session_id="sess-002")

        records_seen = set()
        for art in artifacts:
            rec = art.get("record")
            if rec and rec.sha256 and rec.sha256 not in records_seen:
                records_seen.add(rec.sha256)
                content = artifact_store.fetch(f"sha256:{rec.sha256}")
                assert content is not None

        assert len(records_seen) > 0

    def test_fallback_without_artifact_store(self):
        ext = ChatExtractor(project_id="test", artifact_store=None)
        messages = [{"role": "user", "content": "Let's go with plan B for the design"}]
        artifacts = ext.extract_from_messages(messages, session_id="sess-003")
        for art in artifacts:
            rec = art.get("record")
            if rec:
                assert rec.sha256 is not None
                assert len(rec.sha256) == 64


class TestGitExtractorContent:
    GIT_LOG_OUTPUT = (
        "abc123def456\n"
        "Test Author\n"
        "test@example.com\n"
        "2026-01-15T10:00:00+00:00\n"
        "feat: Add new feature\n"
        "This adds the cool new thing.\n"
        "---END---\n"
    )

    def test_stores_content_in_artifact_store(self, artifact_store: ArtifactStore, tmp_path: Path):
        ext = GitExtractor(tmp_path, project_id="test", artifact_store=artifact_store)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = self.GIT_LOG_OUTPUT
            artifacts = ext.extract_commits(limit=1)

        assert len(artifacts) == 1
        record = artifacts[0]["record"]
        content = artifact_store.fetch(f"sha256:{record.sha256}")
        assert content is not None
        assert b"Add new feature" in content

    def test_fallback_without_artifact_store(self, tmp_path: Path):
        ext = GitExtractor(tmp_path, project_id="test", artifact_store=None)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = self.GIT_LOG_OUTPUT
            artifacts = ext.extract_commits(limit=1)

        assert len(artifacts) == 1
        record = artifacts[0]["record"]
        assert record.sha256 is not None
        assert len(record.sha256) == 64


class TestPipelineWiring:
    def test_pipeline_passes_artifact_store_to_extractors(
        self, knowledge_store: KnowledgeStore, artifact_store: ArtifactStore,
    ):
        pipeline = NarrativePipeline(
            knowledge_store, project_id="test",
            artifact_store=artifact_store,
        )
        assert pipeline.artifact_store is artifact_store
        assert pipeline.chat_extractor.artifact_store is artifact_store
        assert pipeline.agent_extractor.artifact_store is artifact_store

    def test_pipeline_without_artifact_store(self, knowledge_store: KnowledgeStore):
        pipeline = NarrativePipeline(knowledge_store, project_id="test")
        assert pipeline.artifact_store is None
        assert pipeline.chat_extractor.artifact_store is None
        assert pipeline.agent_extractor.artifact_store is None


class TestEndToEndContentRetrieval:
    def test_ingested_content_retrievable_via_record_store(
        self,
        knowledge_store: KnowledgeStore,
        artifact_store: ArtifactStore,
    ):
        """Full pipeline: ingest via NarrativePipeline, retrieve via RecordStore.get_content()."""
        pipeline = NarrativePipeline(
            knowledge_store, project_id="test",
            artifact_store=artifact_store,
        )
        record_store = RecordStore(knowledge_store, artifact_store)

        # Ingest an agent result through the pipeline
        result = {
            "task": "Analyze comic page layout",
            "output": "Found 6 panels in 2x3 grid",
            "status": "success",
            "model": "qwen-vlm-8b",
        }
        pipeline.ingest_agent_result(result, agent_id="test-agent", agent_type="vlm")

        # Retrieve records and verify content is accessible
        from datetime import datetime, timezone, timedelta
        since = datetime.now(timezone.utc) - timedelta(hours=1)
        records = record_store.get_records_since(since)

        assert len(records) > 0
        content = record_store.get_content(records[0])
        assert content is not None
        assert b"Analyze comic page layout" in content
