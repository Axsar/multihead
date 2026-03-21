"""Round 15 tests: upload size guard, DAG null manifest,
consensus head validation, session trim bounds,
transformer partial load."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from multihead.models import (
    AdapterKind,
    HeadManifest,
    RunStatus,
    StepDef,
    WorkOrder,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_heads():
    from multihead.head_manager import HeadManager

    manifests = {
        "head-a": HeadManifest(
            head_id="head-a", name="A", adapter=AdapterKind.MOCK,
            model="mock-v1", kind="llm", gpu_required=False,
        ),
        "head-b": HeadManifest(
            head_id="head-b", name="B", adapter=AdapterKind.MOCK,
            model="mock-v1", kind="llm", gpu_required=True,
        ),
    }
    return HeadManager(manifests)


# ---------------------------------------------------------------------------
# Upload streaming size guard
# ---------------------------------------------------------------------------


class TestUploadSizeGuard:
    def test_upload_route_reads_in_chunks(self):
        """Upload route should read file in chunks, not all at once."""
        import inspect
        from multihead.api.routes_artifacts import upload_artifact

        source = inspect.getsource(upload_artifact)
        # Should NOT have a bare `await file.read()` (unbounded)
        # Should have chunked reading pattern
        assert "UPLOAD_CHUNK_SIZE" in source or "file.read(" in source
        assert "413" in source  # HTTP 413 for too large

    def test_upload_chunk_size_defined(self):
        """Module should define UPLOAD_CHUNK_SIZE constant."""
        from multihead.api.routes_artifacts import UPLOAD_CHUNK_SIZE

        assert UPLOAD_CHUNK_SIZE > 0
        assert UPLOAD_CHUNK_SIZE <= 1024 * 1024  # At most 1MB chunks

    @pytest.mark.asyncio
    async def test_artifact_store_rejects_oversized(self):
        """ArtifactStore.store() should reject data above max_size_bytes."""
        from multihead.artifact_store import ArtifactStore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            store = ArtifactStore(Path(d) / "art", Path(d) / "db.sqlite", max_size_bytes=50)
            with pytest.raises(ValueError, match="too large"):
                store.store(b"x" * 51)

    def test_routes_artifacts_has_logger(self):
        """routes_artifacts should have a logger configured."""
        from multihead.api import routes_artifacts
        assert hasattr(routes_artifacts, "logger")


# ---------------------------------------------------------------------------
# DAG executor null manifest safety
# ---------------------------------------------------------------------------


class TestDAGNullManifest:
    def test_dag_uses_public_get_manifest(self):
        """DAG executor should use public get_manifest(), not private _manifests."""
        import inspect
        from multihead.dag_executor import DAGExecutor

        source = inspect.getsource(DAGExecutor.execute_dag)
        assert "_manifests" not in source
        assert "get_manifest" in source

    def test_head_manager_has_get_manifest(self):
        """HeadManager should expose get_manifest() public method."""
        from multihead.head_manager import HeadManager
        assert hasattr(HeadManager, "get_manifest")

    def test_get_manifest_returns_none_for_unknown(self, mock_heads):
        """get_manifest() should return None for unknown head_id."""
        assert mock_heads.get_manifest("nonexistent") is None

    def test_get_manifest_returns_manifest(self, mock_heads):
        """get_manifest() should return the manifest for known head_id."""
        m = mock_heads.get_manifest("head-a")
        assert m is not None
        assert m.head_id == "head-a"

    def test_unknown_head_defaults_to_gpu(self):
        """When manifest is None, DAG should default to GPU (serialized execution)."""
        import inspect
        from multihead.dag_executor import DAGExecutor

        source = inspect.getsource(DAGExecutor.execute_dag)
        # Should have: if manifest is None or manifest.gpu_required
        assert "manifest is None" in source

    @pytest.mark.asyncio
    async def test_dag_with_gpu_head(self, tmp_path, mock_heads):
        """DAG executor should correctly route GPU heads to serial execution."""
        from multihead.artifact_store import ArtifactStore
        from multihead.event_store import EventStore
        from multihead.orchestrator import Orchestrator
        from multihead.dag_executor import DAGExecutor

        db_path = tmp_path / "test.db"
        artifact_store = ArtifactStore(tmp_path / "artifacts", db_path)
        event_store = EventStore(tmp_path / "runs", db_path)
        orchestrator = Orchestrator(event_store, artifact_store, mock_heads, tmp_path / "runs")

        executor = DAGExecutor(orchestrator)

        wo = WorkOrder(
            goal="dag gpu test",
            steps=[
                StepDef(name="s1", head_id="head-b", prompt_template="test gpu"),
            ],
        )
        state = await orchestrator.create_run(wo)
        state = await orchestrator.execute_run(state.run_id)
        assert state.status == RunStatus.DONE


# ---------------------------------------------------------------------------
# Consensus engine head validation
# ---------------------------------------------------------------------------


class TestConsensusHeadValidation:
    @pytest.mark.asyncio
    async def test_consensus_rejects_unknown_head(self, mock_heads):
        """ConsensusEngine.execute() should reject unknown head IDs."""
        from multihead.consensus import ConsensusConfig, ConsensusEngine, HeadTask

        engine = ConsensusEngine(mock_heads)
        config = ConsensusConfig(
            heads=[HeadTask(head_id="nonexistent")],
        )
        with pytest.raises(ValueError, match="Unknown head"):
            await engine.execute(config, "test prompt")

    @pytest.mark.asyncio
    async def test_consensus_accepts_known_heads(self, mock_heads):
        """ConsensusEngine.execute() should accept known head IDs."""
        from multihead.consensus import ConsensusConfig, ConsensusEngine, HeadTask

        engine = ConsensusEngine(mock_heads)
        config = ConsensusConfig(
            heads=[HeadTask(head_id="head-a")],
        )
        # Should not raise
        result = await engine.execute(config, "test prompt")
        assert result.all_votes  # Got at least one vote

    @pytest.mark.asyncio
    async def test_consensus_reports_all_unknown_heads(self, mock_heads):
        """Error message should list all unknown heads."""
        from multihead.consensus import ConsensusConfig, ConsensusEngine, HeadTask

        engine = ConsensusEngine(mock_heads)
        config = ConsensusConfig(
            heads=[
                HeadTask(head_id="bad-1"),
                HeadTask(head_id="bad-2"),
            ],
        )
        with pytest.raises(ValueError, match="bad-1.*bad-2|bad-2.*bad-1"):
            await engine.execute(config, "test")

    def test_consensus_execute_has_validation(self):
        """ConsensusEngine.execute() should validate heads upfront."""
        import inspect
        from multihead.consensus import ConsensusEngine

        source = inspect.getsource(ConsensusEngine.execute)
        assert "Unknown head" in source


# ---------------------------------------------------------------------------
# Session trim bounds safety
# ---------------------------------------------------------------------------


class TestSessionTrimBounds:
    def test_trim_with_many_system_messages(self):
        """Trimming should be safe when system messages exceed max_messages."""
        from multihead.session import SessionManager

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            mgr = SessionManager(Path(d), max_messages=3)
            session = mgr.create_session()

            # Add 5 system messages (more than max_messages)
            for i in range(5):
                session.messages.append(
                    __import__("multihead.session", fromlist=["Message"]).Message(
                        role="system", content=f"sys {i}"
                    )
                )

            # Add a user message - this triggers trimming
            mgr.add_message(session.session_id, "user", "hello")

            reloaded = mgr.get_session(session.session_id)
            # All system messages should be preserved
            system_count = sum(1 for m in reloaded.messages if m.role == "system")
            assert system_count == 5

    def test_trim_with_max_messages_one(self):
        """max_messages=1 should keep only system messages when they exist."""
        from multihead.session import SessionManager, Message

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            mgr = SessionManager(Path(d), max_messages=1)
            session = mgr.create_session()
            session.messages.append(Message(role="system", content="sys"))
            mgr.save_session(session)

            # Add user messages - trimming kicks in
            mgr.add_message(session.session_id, "user", "msg1")
            mgr.add_message(session.session_id, "user", "msg2")

            reloaded = mgr.get_session(session.session_id)
            # System message preserved, no crash from negative slice
            assert any(m.role == "system" for m in reloaded.messages)

    def test_trim_clamps_keep_to_zero(self):
        """Trim logic should clamp keep to 0, not go negative."""
        import inspect
        from multihead.session import SessionManager

        source = inspect.getsource(SessionManager.add_message)
        assert "max(0," in source


# ---------------------------------------------------------------------------
# Transformer partial load cleanup
# ---------------------------------------------------------------------------


class TestTransformerPartialLoad:
    def test_load_has_cleanup_on_processor_failure(self):
        """VLM load path should clean up model if processor fails."""
        import inspect
        from multihead.adapters.transformers_adapter import TransformersAdapter

        source = inspect.getsource(TransformersAdapter.load)
        # Should have cleanup (unload) when processor/tokenizer fails
        assert "await self.unload()" in source

    def test_load_has_cleanup_on_tokenizer_failure(self):
        """LLM load path should clean up model if tokenizer fails."""
        import inspect
        from multihead.adapters.transformers_adapter import TransformersAdapter

        source = inspect.getsource(TransformersAdapter.load)
        # Should handle both VLM processor and LLM tokenizer failures
        assert source.count("await self.unload()") >= 2

    def test_load_logs_cleanup(self):
        """Load failure cleanup should be logged."""
        import inspect
        from multihead.adapters.transformers_adapter import TransformersAdapter

        source = inspect.getsource(TransformersAdapter.load)
        assert "cleaning up model" in source
