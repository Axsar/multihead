"""Tests for recent features: Claude adapter, parallel extraction,
knowledge tools, and orchestrator _build_prompt (fast tests only)."""

import asyncio
import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from multihead.adapters.base import HeadAdapter
from multihead.artifact_store import ArtifactStore
from multihead.chunker import Chunk
from multihead.extractors.base import BaseExtractor
from multihead.extractors.entity_extractor import EntityExtractor
from multihead.head_manager import HeadManager
from multihead.knowledge_models import (
    Claim,
    ClaimCanonical,
    ClaimScope,
    ClaimType,
    EntityRef,
    EventType,
    KnowledgeEvent,
    Provenance,
    ScopeType,
    TimeBlock,
    ValueObject,
)
from multihead.knowledge_store import KnowledgeStore
from multihead.knowledge_tools import register_knowledge_tools
from multihead.models import AdapterKind, HeadManifest
from multihead.tool_registry import ToolRegistry


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _prov() -> Provenance:
    return Provenance(produced_by={"kind": "test", "id": "unit"})


@pytest.fixture
def sample_chunks():
    return [
        Chunk(chunk_id="chk_1", record_id="rec_1",
              text="MultiHead is a local orchestration layer.",
              span_start=0, span_end=42),
        Chunk(chunk_id="chk_2", record_id="rec_2",
              text="The core LLM runs on CPU by default.",
              span_start=0, span_end=36),
        Chunk(chunk_id="chk_3", record_id="rec_3",
              text="Night Shift processes records overnight.",
              span_start=0, span_end=40),
    ]


class FakeAdapter(HeadAdapter):
    """Adapter that returns canned JSON responses for testing."""

    def __init__(self, responses: list[str] | None = None):
        manifest = HeadManifest(
            head_id="fake", name="Fake", adapter=AdapterKind.MOCK,
            model="fake-v1", kind="llm", gpu_required=False,
        )
        super().__init__(manifest)
        self._responses = responses or []
        self._call_count = 0
        self.prompts: list[str] = []

    async def load(self) -> None:
        pass

    async def unload(self) -> None:
        pass

    async def generate(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        self.prompts.append(prompt)
        idx = self._call_count % max(len(self._responses), 1)
        text = self._responses[idx] if self._responses else "[]"
        self._call_count += 1
        return {"text": text, "tokens_in": 10, "tokens_out": 20}

    async def healthcheck(self) -> bool:
        return True

    async def sleep(self, level: int = 1) -> None:
        pass

    async def wake(self) -> None:
        pass


class SlowFakeAdapter(FakeAdapter):
    """Adapter with a configurable delay per call, for concurrency testing."""

    def __init__(self, responses: list[str] | None = None, delay: float = 0.05):
        super().__init__(responses)
        self.delay = delay

    async def generate(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(self.delay)
        return await super().generate(prompt, **kwargs)


# ===================================================================
# 1. Claude Adapter Tests
# ===================================================================

class TestClaudeAdapter:
    """Tests for the Claude CLI adapter."""

    def test_import(self):
        from multihead.adapters.claude_adapter import ClaudeAdapter
        assert ClaudeAdapter is not None

    def test_init_defaults(self):
        from multihead.adapters.claude_adapter import ClaudeAdapter
        manifest = HeadManifest(
            head_id="claude-sonnet", name="Claude", adapter=AdapterKind.CLAUDE,
            model="sonnet", kind="llm", gpu_required=False,
        )
        adapter = ClaudeAdapter(manifest)
        assert adapter.model_name == "sonnet"
        assert adapter.max_tokens == 16384

    def test_init_custom_model(self):
        from multihead.adapters.claude_adapter import ClaudeAdapter
        manifest = HeadManifest(
            head_id="claude-opus", name="Claude Opus", adapter=AdapterKind.CLAUDE,
            model="opus", kind="llm", gpu_required=False,
        )
        adapter = ClaudeAdapter(manifest)
        assert adapter.model_name == "opus"

    @pytest.mark.asyncio
    async def test_load_finds_claude(self):
        from multihead.adapters.claude_adapter import ClaudeAdapter
        manifest = HeadManifest(
            head_id="claude-test", name="Test", adapter=AdapterKind.CLAUDE,
            model="sonnet", kind="llm", gpu_required=False,
        )
        adapter = ClaudeAdapter(manifest)
        with patch("shutil.which", return_value="/usr/bin/claude"):
            await adapter.load()
            assert adapter._claude_path == "/usr/bin/claude"

    @pytest.mark.asyncio
    async def test_load_raises_if_missing(self):
        from multihead.adapters.claude_adapter import ClaudeAdapter
        manifest = HeadManifest(
            head_id="claude-test", name="Test", adapter=AdapterKind.CLAUDE,
            model="sonnet", kind="llm", gpu_required=False,
        )
        adapter = ClaudeAdapter(manifest)
        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="claude CLI not found"):
                await adapter.load()

    @pytest.mark.asyncio
    async def test_generate_parses_json_array(self):
        """Test that generate correctly parses the [init, assistant, result] array."""
        from multihead.adapters.claude_adapter import ClaudeAdapter
        manifest = HeadManifest(
            head_id="claude-test", name="Test", adapter=AdapterKind.CLAUDE,
            model="sonnet", kind="llm", gpu_required=False,
        )
        adapter = ClaudeAdapter(manifest)
        adapter._claude_path = "/usr/bin/claude"

        # Simulate claude -p JSON output
        json_output = json.dumps([
            {"type": "system", "content": "init"},
            {"type": "assistant", "content": "thinking..."},
            {
                "type": "result",
                "result": "Hello world!",
                "total_cost_usd": 0.013,
                "usage": {"input_tokens": 100, "cache_read_input_tokens": 50, "output_tokens": 25},
            },
        ])

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(json_output.encode(), b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await adapter.generate("test prompt")

        assert result["text"] == "Hello world!"
        assert result["tokens_in"] == 150  # 100 + 50 cache
        assert result["tokens_out"] == 25
        assert result["cost_usd"] == 0.013
        assert result["model"] == "sonnet"

    @pytest.mark.asyncio
    async def test_generate_strips_claudecode_env(self):
        """Verify CLAUDECODE is removed from subprocess env."""
        from multihead.adapters.claude_adapter import ClaudeAdapter
        manifest = HeadManifest(
            head_id="claude-test", name="Test", adapter=AdapterKind.CLAUDE,
            model="sonnet", kind="llm", gpu_required=False,
        )
        adapter = ClaudeAdapter(manifest)
        adapter._claude_path = "/usr/bin/claude"

        captured_env = {}
        json_output = json.dumps([{"type": "result", "result": "ok"}])
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(json_output.encode(), b""))
        mock_proc.returncode = 0

        async def capture_exec(*args, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=capture_exec):
            with patch.dict("os.environ", {"CLAUDECODE": "1", "PATH": "/usr/bin"}):
                await adapter.generate("test")

        assert "CLAUDECODE" not in captured_env

    @pytest.mark.asyncio
    async def test_generate_handles_cli_error(self):
        from multihead.adapters.claude_adapter import ClaudeAdapter
        manifest = HeadManifest(
            head_id="claude-test", name="Test", adapter=AdapterKind.CLAUDE,
            model="sonnet", kind="llm", gpu_required=False,
        )
        adapter = ClaudeAdapter(manifest)
        adapter._claude_path = "/usr/bin/claude"

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"Error: rate limited"))
        mock_proc.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="Claude CLI exited 1"):
                await adapter.generate("test")

    @pytest.mark.asyncio
    async def test_healthcheck(self):
        from multihead.adapters.claude_adapter import ClaudeAdapter
        manifest = HeadManifest(
            head_id="claude-test", name="Test", adapter=AdapterKind.CLAUDE,
            model="sonnet", kind="llm", gpu_required=False,
        )
        adapter = ClaudeAdapter(manifest)
        with patch("shutil.which", return_value="/usr/bin/claude"):
            assert await adapter.healthcheck() is True
        with patch("shutil.which", return_value=None):
            assert await adapter.healthcheck() is False

    def test_factory_creates_claude_adapter(self):
        """_create_adapter factory should create ClaudeAdapter for adapter=claude."""
        from multihead.adapters.claude_adapter import ClaudeAdapter
        from multihead.head_manager import _create_adapter
        manifest = HeadManifest(
            head_id="claude-sonnet", name="Claude", adapter=AdapterKind.CLAUDE,
            model="sonnet", kind="llm", gpu_required=False,
        )
        adapter = _create_adapter(manifest)
        assert isinstance(adapter, ClaudeAdapter)


# ===================================================================
# 2. map_generate Parallel Extraction Tests
# ===================================================================

class TestMapGenerate:
    """Tests for BaseExtractor.map_generate with concurrency control."""

    @pytest.mark.asyncio
    async def test_sequential_returns_results_in_order(self):
        responses = ['[{"id": "a"}]', '[{"id": "b"}]', '[{"id": "c"}]']
        adapter = FakeAdapter(responses)
        prompts = ["p1", "p2", "p3"]

        results = await BaseExtractor.map_generate(adapter, prompts, concurrency=1)

        assert len(results) == 3
        assert all(not isinstance(r, Exception) for r in results)
        assert results[0]["text"] == '[{"id": "a"}]'
        assert results[1]["text"] == '[{"id": "b"}]'
        assert results[2]["text"] == '[{"id": "c"}]'

    @pytest.mark.asyncio
    async def test_parallel_returns_results_in_order(self):
        responses = ['[{"id": "a"}]', '[{"id": "b"}]', '[{"id": "c"}]']
        adapter = SlowFakeAdapter(responses, delay=0.01)
        prompts = ["p1", "p2", "p3"]

        results = await BaseExtractor.map_generate(adapter, prompts, concurrency=3)

        assert len(results) == 3
        assert all(not isinstance(r, Exception) for r in results)
        # Order preserved even with parallel execution
        assert results[0]["text"] == '[{"id": "a"}]'
        assert results[1]["text"] == '[{"id": "b"}]'
        assert results[2]["text"] == '[{"id": "c"}]'

    @pytest.mark.asyncio
    async def test_parallel_is_faster_than_sequential(self):
        """With concurrency > 1, parallel calls should complete faster."""
        import time
        adapter = SlowFakeAdapter(["[]"] * 6, delay=0.05)
        prompts = ["p"] * 6

        t0 = time.monotonic()
        await BaseExtractor.map_generate(adapter, prompts, concurrency=1)
        seq_time = time.monotonic() - t0

        adapter2 = SlowFakeAdapter(["[]"] * 6, delay=0.05)
        t0 = time.monotonic()
        await BaseExtractor.map_generate(adapter2, prompts, concurrency=6)
        par_time = time.monotonic() - t0

        # Parallel should be significantly faster (at least 2x)
        assert par_time < seq_time * 0.7

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        """Concurrency=2 should limit to 2 simultaneous calls."""
        active = 0
        max_active = 0

        class TrackedAdapter(SlowFakeAdapter):
            async def generate(self, prompt, **kwargs):
                nonlocal active, max_active
                active += 1
                max_active = max(max_active, active)
                result = await super().generate(prompt, **kwargs)
                active -= 1
                return result

        adapter = TrackedAdapter(["[]"] * 8, delay=0.05)
        prompts = ["p"] * 8
        await BaseExtractor.map_generate(adapter, prompts, concurrency=2)
        assert max_active <= 2

    @pytest.mark.asyncio
    async def test_handles_exceptions_gracefully(self):
        """Failed calls return Exception objects, don't crash the batch."""
        call_count = 0

        class FailingAdapter(FakeAdapter):
            async def generate(self, prompt, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise RuntimeError("LLM exploded")
                return await super().generate(prompt, **kwargs)

        adapter = FailingAdapter(["[]"] * 3)
        results = await BaseExtractor.map_generate(adapter, ["a", "b", "c"], concurrency=1)

        assert len(results) == 3
        assert not isinstance(results[0], Exception)
        assert isinstance(results[1], Exception)
        assert not isinstance(results[2], Exception)

    @pytest.mark.asyncio
    async def test_parallel_handles_exceptions(self):
        """Same as above but with parallel execution."""
        call_count = 0

        class FailingAdapter(SlowFakeAdapter):
            async def generate(self, prompt, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise RuntimeError("LLM exploded")
                return await super().generate(prompt, **kwargs)

        adapter = FailingAdapter(["[]"] * 3, delay=0.01)
        results = await BaseExtractor.map_generate(adapter, ["a", "b", "c"], concurrency=3)

        assert len(results) == 3
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 1

    @pytest.mark.asyncio
    async def test_empty_prompts(self):
        adapter = FakeAdapter(["[]"])
        results = await BaseExtractor.map_generate(adapter, [], concurrency=4)
        assert results == []

    @pytest.mark.asyncio
    async def test_extractors_pass_concurrency_kwarg(self, sample_chunks):
        """Verify extractors pass concurrency through to map_generate."""
        entity_resp = json.dumps([
            {"entity_type": "project", "entity_id": "test",
             "label": "Test", "aliases": []},
        ])
        adapter = FakeAdapter([entity_resp] * 3)

        extractor = EntityExtractor()
        result = await extractor.extract(sample_chunks, adapter, concurrency=2)
        assert result.items  # Should have extracted entities
        assert len(adapter.prompts) == 3  # One per chunk


# ===================================================================
# 3. Knowledge Tools Tests
# ===================================================================

class TestKnowledgeTools:
    """Tests for knowledge.claims, knowledge.events, knowledge.stats tools."""

    @pytest.fixture
    def populated_store(self, tmp_path):
        """Create a KnowledgeStore with sample data."""
        ks = KnowledgeStore(tmp_path / "knowledge.db")
        # Insert sample claims
        for i in range(5):
            claim = Claim(
                claim_type=ClaimType.FACT,
                scope=ClaimScope(scope_type=ScopeType.PROJECT, scope_id="test"),
                canonical=ClaimCanonical(
                    claim_key=f"test.claim.{i}",
                    subject=EntityRef(entity_type="component", entity_id=f"comp-{i}"),
                    predicate="uses",
                    object=ValueObject(value_type="string", value=f"value-{i}"),
                ),
                statement=(
                    f"Component {i} uses claude adapter for inference"
                    if i < 2 else f"Component {i} runs locally"
                ),
                confidence=0.9,
                provenance=_prov(),
            )
            ks.insert_claim(claim)

        # Insert sample events
        for i in range(3):
            event = KnowledgeEvent(
                event_type=EventType.COMMIT,
                title=f"feat: Add feature {i}",
                summary=f"Added feature {i} to the system",
                time=TimeBlock(happened_at=datetime.now(timezone.utc)),
                provenance=_prov(),
            )
            ks.insert_event(event)

        event = KnowledgeEvent(
            event_type=EventType.DECISION,
            title="decision: Use Claude for Night Shift",
            summary="Decided to use Claude Sonnet as Night Shift LLM backend",
            time=TimeBlock(happened_at=datetime.now(timezone.utc)),
            provenance=_prov(),
        )
        ks.insert_event(event)

        return ks

    @pytest.fixture
    def registry_with_tools(self, populated_store):
        tr = ToolRegistry()
        register_knowledge_tools(tr, populated_store)
        return tr

    def test_tools_registered(self, registry_with_tools):
        names = [t.name for t in registry_with_tools.list_tools()]
        assert "knowledge.claims" in names
        assert "knowledge.events" in names
        assert "knowledge.stats" in names

    @pytest.mark.asyncio
    async def test_query_claims_all(self, registry_with_tools):
        result = await registry_with_tools.execute("knowledge.claims", {})
        assert result.success
        items = json.loads(result.output)
        assert len(items) == 5

    @pytest.mark.asyncio
    async def test_query_claims_with_search(self, registry_with_tools):
        result = await registry_with_tools.execute("knowledge.claims", {"search": "claude"})
        assert result.success
        items = json.loads(result.output)
        assert len(items) == 2  # Only 2 claims mention "claude"

    @pytest.mark.asyncio
    async def test_query_claims_with_limit(self, registry_with_tools):
        result = await registry_with_tools.execute("knowledge.claims", {"limit": 2})
        assert result.success
        items = json.loads(result.output)
        assert len(items) == 2

    @pytest.mark.asyncio
    async def test_query_claims_with_type_filter(self, registry_with_tools):
        result = await registry_with_tools.execute("knowledge.claims", {"claim_type": "fact"})
        assert result.success
        items = json.loads(result.output)
        assert len(items) == 5
        assert all(item["type"] == "fact" for item in items)

    @pytest.mark.asyncio
    async def test_query_events_all(self, registry_with_tools):
        result = await registry_with_tools.execute("knowledge.events", {})
        assert result.success
        items = json.loads(result.output)
        assert len(items) == 4

    @pytest.mark.asyncio
    async def test_query_events_by_type(self, registry_with_tools):
        result = await registry_with_tools.execute("knowledge.events", {"event_type": "commit"})
        assert result.success
        items = json.loads(result.output)
        assert len(items) == 3

    @pytest.mark.asyncio
    async def test_query_events_with_search(self, registry_with_tools):
        result = await registry_with_tools.execute("knowledge.events", {"search": "claude"})
        assert result.success
        items = json.loads(result.output)
        assert len(items) == 1
        assert "Claude" in items[0]["title"]

    @pytest.mark.asyncio
    async def test_stats(self, registry_with_tools):
        result = await registry_with_tools.execute("knowledge.stats", {})
        assert result.success
        stats = json.loads(result.output)
        assert stats["total_claims"] == 5
        assert stats["total_events"] == 4
        assert "fact" in stats["claim_types"]
        assert "commit" in stats["event_types"]

    @pytest.mark.asyncio
    async def test_empty_store(self, tmp_path):
        ks = KnowledgeStore(tmp_path / "empty.db")
        tr = ToolRegistry()
        register_knowledge_tools(tr, ks)

        result = await tr.execute("knowledge.stats", {})
        assert result.success
        stats = json.loads(result.output)
        assert stats["total_claims"] == 0
        assert stats["total_events"] == 0


# ===========================================================================
# 6) Orchestrator _build_prompt template substitution
# ===========================================================================

class TestOrchestratorBuildPrompt:
    """Tests for _build_prompt template variable substitution."""

    def _make_orchestrator(self, tmp_path):
        from multihead.orchestrator import Orchestrator
        from multihead.event_store import EventStore
        events = EventStore(tmp_path / "runs", tmp_path / "events.db")
        artifacts = ArtifactStore(tmp_path / "artifacts", tmp_path / "artifacts.db")
        heads = MagicMock(spec=HeadManager)
        return Orchestrator(events, artifacts, heads, tmp_path / "runs")

    def _make_work_order(self, steps_data, inputs=None):
        from multihead.models import WorkOrder, StepDef
        steps = []
        for sd in steps_data:
            steps.append(StepDef(**sd))
        return WorkOrder(goal="Test goal", steps=steps, inputs=inputs or {})

    def _make_state(self, work_order, step_results=None):
        from multihead.models import RunState, RunStatus
        state = RunState(
            run_id=work_order.run_id,
            status=RunStatus.RUNNING,
            work_order=work_order,
        )
        if step_results:
            state.step_results = step_results
        return state

    def test_goal_substitution(self, tmp_path):
        """Template {goal} is replaced with work order goal."""
        orch = self._make_orchestrator(tmp_path)
        wo = self._make_work_order([
            {"name": "step1", "head_id": "mock-llm",
             "prompt_template": "Analyze: {goal}"},
        ])
        state = self._make_state(wo)
        prompt = orch._build_prompt(wo.steps[0], wo, state)
        assert "Test goal" in prompt
        assert "{goal}" not in prompt

    def test_input_path_substitution(self, tmp_path):
        """Template {input_path} is replaced from inputs dict."""
        orch = self._make_orchestrator(tmp_path)
        wo = self._make_work_order(
            [{"name": "step1", "head_id": "mock-llm",
              "prompt_template": "Read from: {input_path}"}],
            inputs={"input_path": "/my/files"},
        )
        state = self._make_state(wo)
        prompt = orch._build_prompt(wo.steps[0], wo, state)
        assert "/my/files" in prompt
        assert "{input_path}" not in prompt

    def test_previous_step_output_substitution(self, tmp_path):
        """Template {previous_step_output} gets text from input_refs."""
        from multihead.models import StageResult, StepStatus
        orch = self._make_orchestrator(tmp_path)
        wo = self._make_work_order([
            {"name": "step1", "head_id": "mock-llm",
             "prompt_template": "First step"},
            {"name": "step2", "head_id": "mock-llm",
             "prompt_template": "Previous: {previous_step_output}",
             "input_refs": ["step1"]},
        ])
        # Simulate step1 completed
        step1_id = wo.steps[0].step_id
        state = self._make_state(wo, step_results={
            step1_id: StageResult(
                step_id=step1_id, head_id="mock-llm",
                status=StepStatus.COMMITTED,
                outputs={"text": "Step 1 output here"},
            ),
        })
        prompt = orch._build_prompt(wo.steps[1], wo, state)
        assert "Step 1 output here" in prompt
        assert "{previous_step_output}" not in prompt

    def test_json_braces_preserved(self, tmp_path):
        """Double braces in JSON templates are preserved."""
        orch = self._make_orchestrator(tmp_path)
        wo = self._make_work_order([
            {"name": "step1", "head_id": "mock-llm",
             "prompt_template": 'Return JSON: {{"key": "value"}}. Goal: {goal}'},
        ])
        state = self._make_state(wo)
        prompt = orch._build_prompt(wo.steps[0], wo, state)
        assert "Test goal" in prompt
        assert '{"key": "value"}' in prompt

    def test_unknown_placeholders_left_intact(self, tmp_path):
        """Unknown {placeholders} are left as-is (not crash)."""
        orch = self._make_orchestrator(tmp_path)
        wo = self._make_work_order([
            {"name": "step1", "head_id": "mock-llm",
             "prompt_template": "Hello {unknown_var}, goal: {goal}"},
        ])
        state = self._make_state(wo)
        prompt = orch._build_prompt(wo.steps[0], wo, state)
        assert "Test goal" in prompt
        assert "{unknown_var}" in prompt

    def test_input_refs_resolve_by_name(self, tmp_path):
        """input_refs use step names, step_results keyed by step_id."""
        from multihead.models import StageResult, StepStatus
        orch = self._make_orchestrator(tmp_path)
        wo = self._make_work_order([
            {"name": "summarize", "head_id": "mock-llm",
             "prompt_template": "Summarize"},
            {"name": "extract", "head_id": "mock-llm",
             "prompt_template": "Extract from: {previous_step_output}",
             "input_refs": ["summarize"]},
        ])
        # step_results keyed by auto-generated step_id
        sum_id = wo.steps[0].step_id
        state = self._make_state(wo, step_results={
            sum_id: StageResult(
                step_id=sum_id, head_id="mock-llm",
                status=StepStatus.COMMITTED,
                outputs={"text": "Summary content"},
            ),
        })
        prompt = orch._build_prompt(wo.steps[1], wo, state)
        assert "Summary content" in prompt
        assert "{previous_step_output}" not in prompt
