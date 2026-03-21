"""Slow integration tests — tests using full_stack with pipeline runs
(NightShift, orchestrator.execute_run, /chat)."""

from __future__ import annotations

import pytest

from multihead.agentic_core import AgenticCore
from multihead.artifact_store import ArtifactStore
from multihead.chunker import Chunker
from multihead.context_packs import PackBuilder
from multihead.event_store import EventStore
from multihead.head_manager import HeadManager
from multihead.knowledge_models import (
    NightShiftConfig,
    Provenance,
)
from multihead.knowledge_store import KnowledgeStore
from multihead.models import AdapterKind, HeadManifest, StepDef, WorkOrder
from multihead.night_shift import NightShift
from multihead.orchestrator import Orchestrator
from multihead.record_store import RecordStore
from multihead.session import SessionManager
from multihead.tool_registry import ToolRegistry
from multihead.vram_policy import VRAMManager, VRAMPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prov():
    return Provenance(produced_by={"kind": "test", "id": "unit"})


def _seed_records(rs, count=3):
    ids = []
    for i in range(count):
        rec = rs.ingest_text(
            f"Record {i}: MultiHead uses event sourcing. Test entry {i}.",
            uri=f"test://record_{i}",
        )
        ids.append(rec.record_id)
    return ids


# ---------------------------------------------------------------------------
# Fixture: full stack for integration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def full_stack(tmp_path):
    """Build the complete MultiHead stack with mock adapters."""
    manifests = {
        "mock-llm": HeadManifest(
            head_id="mock-llm", name="Mock LLM", adapter=AdapterKind.MOCK,
            model="mock-v1", kind="llm", gpu_required=False,
        ),
    }
    head_manager = HeadManager(manifests)
    artifact_store = ArtifactStore(tmp_path / "artifacts", tmp_path / "artifacts.db")
    event_store = EventStore(tmp_path / "runs", tmp_path / "state.db")
    knowledge_store = KnowledgeStore(tmp_path / "knowledge.db")
    record_store = RecordStore(knowledge_store, artifact_store)
    pack_builder = PackBuilder(knowledge_store, tmp_path / "packs")
    orchestrator = Orchestrator(event_store, artifact_store, head_manager, tmp_path / "runs")
    session_manager = SessionManager(tmp_path / "sessions")
    tool_registry = ToolRegistry()
    vram_policy = VRAMPolicy(core_mode="keep_loaded")
    vram_manager = VRAMManager(head_manager, vram_policy, core_head_id="mock-llm")
    agentic_core = AgenticCore(
        head_manager=head_manager,
        orchestrator=orchestrator,
        tool_registry=tool_registry,
        session_manager=session_manager,
        vram_manager=vram_manager,
        core_head_id="mock-llm",
    )

    return {
        "head_manager": head_manager,
        "artifact_store": artifact_store,
        "event_store": event_store,
        "knowledge_store": knowledge_store,
        "record_store": record_store,
        "pack_builder": pack_builder,
        "orchestrator": orchestrator,
        "session_manager": session_manager,
        "tool_registry": tool_registry,
        "vram_manager": vram_manager,
        "agentic_core": agentic_core,
        "tmp_path": tmp_path,
    }


# ---------------------------------------------------------------------------
# 1. Knowledge Pipeline: Ingest -> Night Shift (slow)
# ---------------------------------------------------------------------------


class TestKnowledgePipeline:
    @pytest.mark.asyncio
    async def test_ingest_to_packs_flow(self, full_stack):
        """Ingest records, run night shift, verify claims and packs are produced."""
        rs = full_stack["record_store"]
        ks = full_stack["knowledge_store"]
        pb = full_stack["pack_builder"]
        hm = full_stack["head_manager"]
        tmp = full_stack["tmp_path"]

        # Step 1: Ingest raw records
        ids = _seed_records(rs, count=5)
        assert len(ids) == 5
        assert len(ks.list_records()) == 5

        # Step 2: Chunk records
        chunker = Chunker(chunk_chars=500, overlap_chars=50)
        records = ks.list_records()
        all_chunks = []
        for rec in records:
            content = f"Record: {rec.uri}"
            chunks = chunker.chunk_text(content, rec.record_id)
            all_chunks.extend(chunks)
        assert len(all_chunks) >= 5

        # Step 3: Run Night Shift pipeline
        config = NightShiftConfig(head_id="mock-llm")
        ns = NightShift(ks, rs, full_stack["artifact_store"], pb, hm, config, tmp / "ns_output")
        report = await ns.run()
        assert report is not None
        assert report.records_processed >= 5

        # Step 4: Build standard packs
        packs = pb.build_standard_packs()
        assert isinstance(packs, list)


# ---------------------------------------------------------------------------
# 2. Orchestrator + HeadManager Full Pipeline
# ---------------------------------------------------------------------------


class TestOrchestratorPipeline:
    @pytest.mark.asyncio
    async def test_create_and_execute_run(self, full_stack):
        """Create a work order, execute it, verify events are recorded."""
        orch = full_stack["orchestrator"]
        es = full_stack["event_store"]

        work_order = WorkOrder(
            goal="Integration test pipeline",
            steps=[
                StepDef(name="step-1", head_id="mock-llm", prompt_template="Analyze this: {input}"),
                StepDef(name="step-2", head_id="mock-llm", prompt_template="Summarize: {input}"),
            ],
            inputs={"input": "test data"},
        )

        state = await orch.create_run(work_order)
        assert state.run_id

        state = await orch.execute_run(state.run_id)
        assert state.status.value in ("done", "failed")

        # Verify events were recorded
        events = es.read_events(state.run_id)
        assert len(events) >= 3  # run_created + step events + run_done

    @pytest.mark.asyncio
    async def test_head_swap_during_pipeline(self, tmp_path):
        """Pipeline with different heads should swap correctly."""
        manifests = {
            "head-a": HeadManifest(
                head_id="head-a", name="Head A", adapter=AdapterKind.MOCK,
                model="mock-a", kind="llm", gpu_required=False,
            ),
            "head-b": HeadManifest(
                head_id="head-b", name="Head B", adapter=AdapterKind.MOCK,
                model="mock-b", kind="llm", gpu_required=False,
            ),
        }
        hm = HeadManager(manifests)
        as_ = ArtifactStore(tmp_path / "art", tmp_path / "art.db")
        es = EventStore(tmp_path / "runs", tmp_path / "state.db")
        orch = Orchestrator(es, as_, hm, tmp_path / "runs")

        wo = WorkOrder(
            goal="Multi-head test",
            steps=[
                StepDef(name="step-a", head_id="head-a", prompt_template="First step"),
                StepDef(name="step-b", head_id="head-b", prompt_template="Second step"),
            ],
        )

        state = await orch.create_run(wo)
        state = await orch.execute_run(state.run_id)
        assert state.status.value == "done"

        # Both heads should have been used
        events = es.read_events(state.run_id)
        step_events = [e for e in events if e.kind.value == "step_started"]
        assert len(step_events) == 2


# ---------------------------------------------------------------------------
# 3. Agentic Core: Chat -> Tool -> Approval -> Resume
# ---------------------------------------------------------------------------


class TestAgenticCoreFlow:
    @pytest.mark.asyncio
    async def test_chat_session_persistence(self, full_stack):
        """Messages persist across multiple chat turns."""
        core = full_stack["agentic_core"]
        sm = full_stack["session_manager"]

        session = sm.create_session()
        await core.chat(session.session_id, "Hello!")
        await core.chat(session.session_id, "How are you?")

        reloaded = sm.get_session(session.session_id)
        # At least 4 messages: user, assistant, user, assistant
        assert len(reloaded.messages) >= 4

    @pytest.mark.asyncio
    async def test_approval_roundtrip(self, full_stack):
        """Full approval flow: request -> approve -> execute -> result."""
        from unittest.mock import patch
        from multihead.action_types import CallToolAction
        from multihead.tool_registry import ToolRegistry

        core = full_stack["agentic_core"]
        sm = full_stack["session_manager"]
        tmp = full_stack["tmp_path"]

        session = sm.create_session()
        target = tmp / "integration_test.txt"

        # Simulate core wanting to write a file
        action = CallToolAction(
            tool="files.write",
            params={"path": str(target), "content": "integration test"},
        )
        response = await core._execute_action(session.session_id, action, depth=0)
        assert "requires approval" in response.lower()

        # User approves (patch _normalize_path to identity for Windows)
        from pathlib import Path as _Path
        with patch.object(ToolRegistry, "_normalize_path", staticmethod(lambda p: _Path(p))):
            response = await core.chat(session.session_id, "yes")

        # File should exist
        assert target.exists()
        assert target.read_text() == "integration test"

    @pytest.mark.asyncio
    async def test_tool_execution_in_session(self, full_stack):
        """Safe tools execute immediately within core loop."""
        from multihead.action_types import CallToolAction

        core = full_stack["agentic_core"]
        sm = full_stack["session_manager"]
        tmp = full_stack["tmp_path"]

        # Create a file to read
        test_file = tmp / "readable.txt"
        test_file.write_text("hello from integration test")

        session = sm.create_session()
        action = CallToolAction(
            tool="files.read",
            params={"path": str(test_file)},
        )
        result = await core._execute_action(session.session_id, action, depth=0)

        # Should have executed (files.read doesn't need approval)
        # and returned the core loop's interpretation
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 5. API End-to-End (slow: chat test)
# ---------------------------------------------------------------------------


class TestAPIEndToEndSlow:
    def test_chat_and_sessions(self, api_client):
        """Chat creates session, sessions are listable."""
        # Create session via chat
        resp = api_client.post("/chat", json={"message": "Hello!"})
        assert resp.status_code == 200
        session_id = resp.json()["session_id"]

        # List sessions
        resp = api_client.get("/chat/sessions")
        assert resp.status_code == 200
        sessions = resp.json()
        assert any(s["session_id"] == session_id for s in sessions)

        # Get session details
        resp = api_client.get(f"/chat/sessions/{session_id}")
        assert resp.status_code == 200
        assert len(resp.json()["messages"]) >= 2
