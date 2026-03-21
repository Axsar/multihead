"""Tests for narrative pipeline integration: context_gen, CLI, Night Shift stage."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from click.testing import CliRunner

from multihead.artifact_store import ArtifactStore
from multihead.context_packs import PackBuilder
from multihead.knowledge_models import (
    Claim,
    ClaimCanonical,
    ClaimScope,
    ClaimStatus,
    ClaimType,
    EntityRef,
    EvidencePointer,
    EventStatus,
    EventType,
    KnowledgeEvent,
    NightShiftConfig,
    Provenance,
    Record,
    ScopeType,
    SpanRef,
    ValueObject,
)
from multihead.knowledge_store import KnowledgeStore
from multihead.models import AdapterKind, HeadManifest, new_id
from multihead.narrative.context_gen import generate_daemon_context


def _prov() -> Provenance:
    return Provenance(produced_by={"kind": "test", "id": "unit"})


def _make_claim(statement: str, confidence: float = 0.8) -> Claim:
    """Create a minimal claim for testing."""
    cid = new_id("clm_")
    rid = new_id("rec_")
    eid = new_id("ev_")
    return Claim(
        claim_id=cid,
        claim_type=ClaimType.FACT,
        claim_status=ClaimStatus.ACCEPTED,
        statement=statement,
        confidence=confidence,
        importance=confidence,
        scope=ClaimScope(scope_type=ScopeType.PROJECT, scope_id="multihead"),
        canonical=ClaimCanonical(
            subject=EntityRef(entity_id="test", entity_type="concept", label="test"),
            predicate="is",
            object=ValueObject(value_type="string", value=statement),
            claim_key=f"test:{statement[:20]}",
        ),
        evidence_supports=[
            EvidencePointer(
                evidence_id=eid,
                record_id=rid,
                span=SpanRef(start=0, end=10, unit="chars"),
                captured_at=datetime.now(timezone.utc),
            ),
        ],
        provenance=_prov(),
    )


def _make_event(title: str) -> KnowledgeEvent:
    """Create a minimal event for testing."""
    from multihead.knowledge_models import TimeBlock
    return KnowledgeEvent(
        event_id=new_id("evt_"),
        event_type=EventType.DECISION,
        event_status=EventStatus.CONFIRMED,
        title=title,
        time=TimeBlock(happened_at=datetime.now(timezone.utc)),
        provenance=_prov(),
    )


@pytest.fixture
def ks(tmp_path):
    return KnowledgeStore(tmp_path / "knowledge.db")


@pytest.fixture
def ks_with_data(ks):
    """KnowledgeStore seeded with claims and events."""
    # Insert records for evidence pointers
    for claim in [
        _make_claim("MultiHead uses event sourcing", 0.9),
        _make_claim("RTX 4090 has 24GB VRAM", 0.95),
        _make_claim("Night Shift runs 15 stages", 0.85),
    ]:
        # Insert the record first so evidence FK is valid
        record = Record(
            record_id=claim.evidence_supports[0].record_id,
            uri="test://record",
            captured_at=datetime.now(timezone.utc),
            provenance=_prov(),
        )
        try:
            ks.insert_record(record)
        except Exception:
            pass
        ks.insert_claim(claim)

    for event in [
        _make_event("Added narrative pipeline module"),
        _make_event("Daemon round-trip confirmed"),
    ]:
        ks.insert_event(event)

    return ks


# -------------------------------------------------------------------
# Context Generation
# -------------------------------------------------------------------

class TestGenerateDaemonContext:
    def test_generates_file(self, ks_with_data, tmp_path):
        output = tmp_path / "context" / "daemon_narrative.md"
        result = generate_daemon_context(ks_with_data, output)
        assert result == output
        assert output.exists()
        content = output.read_text()
        assert "Narrative Context" in content

    def test_includes_claims(self, ks_with_data, tmp_path):
        output = tmp_path / "daemon_narrative.md"
        generate_daemon_context(ks_with_data, output)
        content = output.read_text()
        assert "Key Knowledge Claims" in content
        assert "event sourcing" in content

    def test_includes_events(self, ks_with_data, tmp_path):
        output = tmp_path / "daemon_narrative.md"
        generate_daemon_context(ks_with_data, output)
        content = output.read_text()
        assert "Recent Events" in content
        assert "narrative pipeline" in content

    def test_empty_store(self, ks, tmp_path):
        output = tmp_path / "daemon_narrative.md"
        generate_daemon_context(ks, output)
        content = output.read_text()
        assert "No narrative data yet" in content

    def test_creates_parent_dirs(self, ks, tmp_path):
        output = tmp_path / "deep" / "nested" / "daemon_narrative.md"
        generate_daemon_context(ks, output)
        assert output.exists()

    def test_respects_limits(self, ks_with_data, tmp_path):
        output = tmp_path / "daemon_narrative.md"
        generate_daemon_context(ks_with_data, output, max_claims=1, max_events=1)
        content = output.read_text()
        # Should have at most 1 claim line
        claim_lines = [l for l in content.split("\n") if l.startswith("- [0.")]
        assert len(claim_lines) <= 1


# -------------------------------------------------------------------
# CLI Commands
# -------------------------------------------------------------------

class TestNarrativeCLI:
    def test_narrative_status(self, ks_with_data, tmp_path, monkeypatch):
        from multihead.cli import main

        runner = CliRunner()
        result = runner.invoke(main, [
            "--data-dir", str(tmp_path),
            "narrative", "status",
        ])
        assert result.exit_code == 0
        assert "Narrative Pipeline Status" in result.output

    def test_narrative_ingest_requires_source(self, tmp_path):
        from multihead.cli import main

        runner = CliRunner()
        result = runner.invoke(main, [
            "--data-dir", str(tmp_path),
            "narrative", "ingest",
        ])
        # Should fail — --source is required
        assert result.exit_code != 0

    def test_narrative_group_exists(self):
        from multihead.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["narrative", "--help"])
        assert result.exit_code == 0
        assert "ingest" in result.output
        assert "fuse" in result.output
        assert "status" in result.output


# -------------------------------------------------------------------
# CLI _parse_since helper
# -------------------------------------------------------------------

class TestParseSince:
    def test_none_returns_none(self):
        from multihead.cli import _parse_since
        assert _parse_since(None) is None

    def test_relative_hours(self):
        from multihead.cli import _parse_since
        result = _parse_since("24h")
        assert result is not None
        # Should be approximately 24 hours ago
        diff = datetime.now(timezone.utc) - result
        assert 23 < diff.total_seconds() / 3600 < 25

    def test_relative_days(self):
        from multihead.cli import _parse_since
        result = _parse_since("7d")
        assert result is not None
        diff = datetime.now(timezone.utc) - result
        assert 6 < diff.total_seconds() / 86400 < 8

    def test_iso_format(self):
        from multihead.cli import _parse_since
        result = _parse_since("2026-02-01T00:00:00+00:00")
        assert result is not None
        assert result.year == 2026
        assert result.month == 2


# -------------------------------------------------------------------
# Night Shift Stage
# -------------------------------------------------------------------

class TestNightShiftNarrativeStage:
    @pytest.fixture
    def night_shift(self, tmp_path):
        from multihead.head_manager import HeadManager
        from multihead.night_shift import NightShift
        from multihead.record_store import RecordStore

        ks = KnowledgeStore(tmp_path / "knowledge.db")
        art = ArtifactStore(tmp_path / "artifacts", tmp_path / "artifacts.db")
        rs = RecordStore(ks, art)
        pb = PackBuilder(ks, tmp_path / "packs")
        manifest = HeadManifest(
            head_id="mock-llm", name="Mock", adapter=AdapterKind.MOCK,
            model="mock-v1", kind="llm", gpu_required=False,
        )
        hm = HeadManager({"mock-llm": manifest})
        config = NightShiftConfig(head_id="mock-llm")
        output_dir = tmp_path / "nightshift_output"
        return NightShift(ks, rs, art, pb, hm, config, output_dir)

    def test_has_narrative_pipeline(self, night_shift):
        from multihead.narrative.pipeline import NarrativePipeline
        assert hasattr(night_shift, "narrative_pipeline")
        assert isinstance(night_shift.narrative_pipeline, NarrativePipeline)

    def test_stage_handler_exists(self, night_shift):
        assert hasattr(night_shift, "_stage_narrative_fusion")

    @pytest.mark.asyncio
    async def test_narrative_fusion_no_repo(self, night_shift, tmp_path, monkeypatch):
        """Stage should handle missing git repo gracefully."""
        monkeypatch.setenv("MULTIHEAD_REPO", str(tmp_path / "nonexistent"))
        context = {"since": datetime.now(timezone.utc) - timedelta(hours=24)}
        result = await night_shift._stage_narrative_fusion(context)
        assert result is not None
        assert result["narrative_git_commits"] == 0

    def test_stages_list_includes_narrative(self):
        from multihead.night_shift import STAGES
        names = [s.name for s in STAGES]
        assert "narrative_fusion" in names
        # narrative_fusion should appear before solver_discovery
        nf_idx = names.index("narrative_fusion")
        sd_idx = names.index("solver_discovery")
        assert nf_idx < sd_idx


# -------------------------------------------------------------------
# Live Ingestion Hooks
# -------------------------------------------------------------------

class TestAgenticCoreNarrativeHook:
    def test_accepts_narrative_pipeline_param(self):
        """AgenticCore constructor accepts optional narrative_pipeline."""
        from unittest.mock import MagicMock
        from multihead.agentic_core import AgenticCore

        ac = AgenticCore(
            head_manager=MagicMock(),
            orchestrator=MagicMock(),
            tool_registry=MagicMock(),
            session_manager=MagicMock(),
            vram_manager=MagicMock(),
            narrative_pipeline=None,
        )
        assert ac.narrative_pipeline is None

    def test_ingest_exchange_with_pipeline(self):
        """_ingest_exchange calls pipeline.ingest_chat_messages."""
        from unittest.mock import MagicMock
        from multihead.agentic_core import AgenticCore

        mock_pipeline = MagicMock()
        ac = AgenticCore(
            head_manager=MagicMock(),
            orchestrator=MagicMock(),
            tool_registry=MagicMock(),
            session_manager=MagicMock(),
            vram_manager=MagicMock(),
            narrative_pipeline=mock_pipeline,
        )
        ac._ingest_exchange("ses-1", "hello", "hi there")
        mock_pipeline.ingest_chat_messages.assert_called_once()
        args = mock_pipeline.ingest_chat_messages.call_args
        assert len(args[0][0]) == 2  # 2 messages
        assert args[0][1] == "ses-1"

    def test_ingest_exchange_without_pipeline(self):
        """_ingest_exchange is a no-op when pipeline is None."""
        from unittest.mock import MagicMock
        from multihead.agentic_core import AgenticCore

        ac = AgenticCore(
            head_manager=MagicMock(),
            orchestrator=MagicMock(),
            tool_registry=MagicMock(),
            session_manager=MagicMock(),
            vram_manager=MagicMock(),
            narrative_pipeline=None,
        )
        # Should not raise
        ac._ingest_exchange("ses-1", "hello", "hi there")

    def test_ingest_exchange_handles_errors(self):
        """_ingest_exchange catches exceptions from pipeline."""
        from unittest.mock import MagicMock
        from multihead.agentic_core import AgenticCore

        mock_pipeline = MagicMock()
        mock_pipeline.ingest_chat_messages.side_effect = RuntimeError("boom")
        ac = AgenticCore(
            head_manager=MagicMock(),
            orchestrator=MagicMock(),
            tool_registry=MagicMock(),
            session_manager=MagicMock(),
            vram_manager=MagicMock(),
            narrative_pipeline=mock_pipeline,
        )
        # Should not raise
        ac._ingest_exchange("ses-1", "hello", "hi there")


class TestDaemonNarrativeHook:
    def test_init_narrative_with_db(self, tmp_path):
        """_init_narrative creates pipeline when knowledge.db parent exists."""
        import os
        from scripts.claude_worker import ClaudeWorker

        os.environ["MULTIHEAD_DATA_DIR"] = str(tmp_path)
        worker = ClaudeWorker(mode="headless")
        assert worker.narrative_pipeline is not None
        del os.environ["MULTIHEAD_DATA_DIR"]

    def test_init_narrative_missing_dir(self, monkeypatch):
        """_init_narrative returns None gracefully on missing dir."""
        from scripts.claude_worker import ClaudeWorker

        monkeypatch.setenv("MULTIHEAD_DATA_DIR", "/nonexistent/path")
        worker = ClaudeWorker(mode="headless")
        assert worker.narrative_pipeline is None
