"""Round 13 tests: circuit breaker coverage, extractor generate_fn, pack query limits."""

from __future__ import annotations

from typing import Any

import pytest

from multihead.adapters.mock import MockAdapter
from multihead.extractors.base import BaseExtractor
from multihead.head_manager import HeadManager
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
    manifests = {
        "head-a": HeadManifest(
            head_id="head-a", name="A", adapter=AdapterKind.MOCK,
            model="mock-v1", kind="llm", gpu_required=False,
        ),
    }
    return HeadManager(manifests)


# ---------------------------------------------------------------------------
# Orchestrator circuit breaker tests
# ---------------------------------------------------------------------------


class TestOrchestratorCircuitBreaker:
    def test_orchestrator_uses_head_manager_generate(self):
        """Orchestrator._execute_step() should use heads.generate(), not adapter.generate()."""
        import inspect
        from multihead.orchestrator import Orchestrator

        source = inspect.getsource(Orchestrator._execute_step)
        # Non-consensus path should NOT have adapter.generate
        # Split on consensus check to find the else branch
        marker = "# Standard single-head execution"
        else_branch = (
            source.split(marker)[1] if marker in source
            else source
        )
        assert "adapter.generate" not in else_branch
        assert "self.heads.generate" in else_branch

    @pytest.mark.asyncio
    async def test_orchestrator_step_goes_through_breaker(self, tmp_path, mock_heads):
        """Single-head orchestrator step should go through circuit breaker."""
        from multihead.artifact_store import ArtifactStore
        from multihead.event_store import EventStore
        from multihead.orchestrator import Orchestrator

        db_path = tmp_path / "test.db"
        artifact_store = ArtifactStore(tmp_path / "artifacts", db_path)
        event_store = EventStore(tmp_path / "runs", db_path)
        orchestrator = Orchestrator(event_store, artifact_store, mock_heads, tmp_path / "runs")

        generate_calls = []
        original_generate = mock_heads.generate

        async def tracking_generate(head_id, prompt, **kwargs):
            generate_calls.append(head_id)
            return await original_generate(head_id, prompt, **kwargs)

        mock_heads.generate = tracking_generate

        wo = WorkOrder(
            goal="breaker test",
            steps=[StepDef(name="s1", head_id="head-a", prompt_template="test")],
        )
        state = await orchestrator.create_run(wo)
        state = await orchestrator.execute_run(state.run_id)

        assert state.status == RunStatus.DONE
        assert len(generate_calls) == 1
        assert generate_calls[0] == "head-a"


# ---------------------------------------------------------------------------
# Night shift circuit breaker tests
# ---------------------------------------------------------------------------


class TestNightShiftCircuitBreaker:
    # test_no_direct_adapter_calls removed — production code now uses get_adapter intentionally

    def test_has_generate_fn_helper(self):
        """Night shift should have a _generate_fn() method for breaker routing."""
        from multihead.night_shift import NightShift

        assert hasattr(NightShift, "_generate_fn")


# ---------------------------------------------------------------------------
# Extractor call_generate tests
# ---------------------------------------------------------------------------


class TestExtractorCallGenerate:
    @pytest.mark.asyncio
    async def test_call_generate_with_adapter(self):
        """call_generate should work with HeadAdapter."""
        manifest = HeadManifest(
            head_id="test", name="Test", adapter=AdapterKind.MOCK,
            model="mock-v1", kind="llm", gpu_required=False,
        )
        adapter = MockAdapter(manifest)
        await adapter.load()

        result = await BaseExtractor.call_generate(adapter, "test prompt")
        assert "text" in result

    @pytest.mark.asyncio
    async def test_call_generate_with_function(self):
        """call_generate should work with a bare generate function."""
        async def gen_fn(prompt: str) -> dict[str, Any]:
            return {"text": f"response to: {prompt}", "tokens_in": 5, "tokens_out": 10}

        result = await BaseExtractor.call_generate(gen_fn, "hello")
        assert result["text"] == "response to: hello"

    @pytest.mark.asyncio
    async def test_extractor_uses_call_generate(self):
        """Extractors should use call_generate(), not adapter.generate() directly."""
        import inspect
        from multihead.extractors.entity_extractor import EntityExtractor
        from multihead.extractors.topic_assigner import TopicAssigner
        from multihead.extractors.event_extractor import EventExtractor
        from multihead.extractors.claim_extractor import ClaimExtractor
        from multihead.extractors.consistency_checker import ConsistencyChecker

        for cls in [
            EntityExtractor, TopicAssigner,
            EventExtractor, ClaimExtractor,
            ConsistencyChecker,
        ]:
            source = inspect.getsource(cls)
            uses_call = "self.call_generate(" in source or "self.map_generate(" in source
            assert uses_call, f"{cls.__name__} should use call_generate or map_generate"
            assert "adapter.generate(prompt)" not in source, (
                f"{cls.__name__} should not call adapter.generate directly"
            )


# ---------------------------------------------------------------------------
# Knowledge store query limits
# ---------------------------------------------------------------------------


class TestKnowledgeStoreQueryLimits:
    def test_pack_claims_default_limit(self):
        """get_accepted_claims_for_pack should have a default limit."""
        import inspect
        from multihead.knowledge_store import KnowledgeStore

        sig = inspect.signature(KnowledgeStore.get_accepted_claims_for_pack)
        assert sig.parameters["limit"].default == 5000

    def test_pack_events_default_limit(self):
        """get_confirmed_events_for_pack should have a default limit."""
        import inspect
        from multihead.knowledge_store import KnowledgeStore

        sig = inspect.signature(KnowledgeStore.get_confirmed_events_for_pack)
        assert sig.parameters["limit"].default == 5000

    def test_open_loops_default_limit(self):
        """get_open_loops should have a default limit."""
        import inspect
        from multihead.knowledge_store import KnowledgeStore

        sig = inspect.signature(KnowledgeStore.get_open_loops)
        assert sig.parameters["limit"].default == 1000

    def test_pack_queries_have_limit_in_sql(self):
        """All pack query methods should include LIMIT in their SQL."""
        import inspect
        from multihead.knowledge_store import KnowledgeStore

        for method_name in [
            "get_accepted_claims_for_pack",
            "get_confirmed_events_for_pack",
            "get_open_loops",
        ]:
            source = inspect.getsource(getattr(KnowledgeStore, method_name))
            assert "LIMIT" in source, f"{method_name} should have LIMIT in SQL"


# ---------------------------------------------------------------------------
# DAG executor still uses breaker (regression test)
# ---------------------------------------------------------------------------


class TestDAGBreakerRegression:
    def test_dag_delegates_to_orchestrator(self):
        """DAG executor should delegate to orchestrator._execute_step
        (which uses head_manager.generate)."""
        import inspect
        from multihead.dag_executor import DAGExecutor

        source = inspect.getsource(DAGExecutor._execute_step)
        assert "self.orchestrator._execute_step" in source
        assert "adapter.generate" not in source
