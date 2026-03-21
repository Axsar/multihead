"""Tests for the MultiHead Engine (Layer 1 Python SDK)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from multihead.engine import Engine
from multihead.models import AdapterKind, HeadManifest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_heads():
    return {
        "mock-llm": HeadManifest(
            head_id="mock-llm", name="Mock LLM",
            adapter=AdapterKind.MOCK, model="test", kind="llm",
        ),
        "mock-vlm": HeadManifest(
            head_id="mock-vlm", name="Mock VLM",
            adapter=AdapterKind.MOCK, model="test", kind="vlm",
        ),
    }


@pytest.fixture
def engine(tmp_path, mock_heads):
    """Create an Engine with mocked internals."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    # Write a minimal heads.yaml (list format with dashes)
    heads_yaml = config_dir / "heads.yaml"
    heads_yaml.write_text(
        "heads:\n"
        "  - head_id: mock-llm\n"
        "    name: Mock LLM\n"
        "    adapter: mock\n"
        "    model: test\n"
        "    kind: llm\n"
        "    gpu_required: false\n"
        "  - head_id: mock-vlm\n"
        "    name: Mock VLM\n"
        "    adapter: mock\n"
        "    model: test\n"
        "    kind: vlm\n"
        "    gpu_required: false\n"
    )
    return Engine(config_dir=str(config_dir), data_dir=str(tmp_path / "data"))


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestEngineInit:
    def test_creates_engine(self, engine):
        assert engine is not None
        assert engine._started is False

    def test_not_started_raises(self, engine):
        with pytest.raises(RuntimeError, match="not started"):
            engine.head_manager

    def test_not_started_generate_raises(self, engine):
        with pytest.raises(RuntimeError, match="not started"):
            import asyncio
            asyncio.get_event_loop().run_until_complete(engine.generate("test"))


# ---------------------------------------------------------------------------
# Start / Stop lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    async def test_start_initializes_components(self, engine):
        await engine.start()
        assert engine._started is True
        assert engine._head_manager is not None
        assert engine._router is not None
        assert engine._knowledge_store is not None
        assert engine._orchestrator is not None
        await engine.stop()

    async def test_stop_shuts_down(self, engine):
        await engine.start()
        await engine.stop()
        assert engine._started is False

    async def test_double_start_idempotent(self, engine):
        await engine.start()
        await engine.start()  # should not raise
        assert engine._started is True
        await engine.stop()

    async def test_stop_without_start(self, engine):
        await engine.stop()  # should not raise
        assert engine._started is False

    async def test_heads_loaded(self, engine):
        await engine.start()
        assert "mock-llm" in engine.heads
        assert "mock-vlm" in engine.heads
        await engine.stop()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    async def test_head_manager(self, engine):
        await engine.start()
        hm = engine.head_manager
        assert hm is not None
        states = hm.get_states()
        assert "mock-llm" in states
        await engine.stop()

    async def test_knowledge_store(self, engine):
        await engine.start()
        ks = engine.knowledge
        assert ks is not None
        await engine.stop()

    async def test_router(self, engine):
        await engine.start()
        r = engine.router
        assert r is not None
        await engine.stop()

    async def test_orchestrator(self, engine):
        await engine.start()
        o = engine.orchestrator
        assert o is not None
        await engine.stop()


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


class TestRouting:
    async def test_route_llm(self, engine):
        await engine.start()
        hid = engine.route("llm")
        assert hid == "mock-llm"
        await engine.stop()

    async def test_route_vlm(self, engine):
        await engine.start()
        hid = engine.route("vlm")
        assert hid == "mock-vlm"
        await engine.stop()

    async def test_route_unknown_kind(self, engine):
        await engine.start()
        hid = engine.route("nonexistent")
        assert hid is None
        await engine.stop()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


class TestGeneration:
    async def test_generate_by_kind(self, engine):
        await engine.start()
        result = await engine.generate("Hello", kind="llm")
        assert "text" in result
        await engine.stop()

    async def test_generate_by_head_id(self, engine):
        await engine.start()
        result = await engine.generate("Hello", head_id="mock-llm")
        assert "text" in result
        await engine.stop()

    async def test_generate_no_head_raises(self, engine):
        await engine.start()
        with pytest.raises(RuntimeError, match="No head available"):
            await engine.generate("Hello", kind="nonexistent")
        await engine.stop()

    async def test_chat_multi_turn(self, engine):
        await engine.start()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        result = await engine.chat(messages, kind="llm")
        assert "text" in result
        await engine.stop()


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------


class TestSolve:
    async def test_solve_not_started_raises(self, engine):
        with pytest.raises(RuntimeError, match="not started"):
            await engine.solve("test task")

    async def test_solve_returns_dict(self, engine):
        await engine.start()

        from multihead.solve_pipeline import SolveResult

        mock_result = SolveResult(
            run_id="run-1", status="done", output="ok",
            confidence=0.9, steps_total=2, steps_succeeded=2,
            steps_failed=0, duration_seconds=3.0,
        )

        with patch("multihead.solve_pipeline.SolvePipeline") as MockPipeline:
            MockPipeline.return_value.solve = AsyncMock(return_value=mock_result)
            result = await engine.solve("build something")

        assert isinstance(result, dict)
        assert result["run_id"] == "run-1"
        assert result["status"] == "done"
        assert result["confidence"] == 0.9
        assert result["steps_total"] == 2
        assert result["steps_succeeded"] == 2
        assert result["steps_failed"] == 0
        assert result["duration_seconds"] == 3.0
        assert result["output"] == "ok"
        await engine.stop()

    async def test_solve_passes_constraints(self, engine):
        await engine.start()

        from multihead.solve_pipeline import SolveResult

        mock_result = SolveResult(
            run_id="r2", status="done", output="ok", confidence=0.8,
        )

        with patch("multihead.solve_pipeline.SolvePipeline") as MockPipeline:
            MockPipeline.return_value.solve = AsyncMock(return_value=mock_result)
            await engine.solve(
                "task", max_steps=5, strategy="majority", timeout=60.0,
            )

            # Verify SolveConstraints were built correctly
            call_kwargs = MockPipeline.return_value.solve.call_args
            constraints = call_kwargs.kwargs.get("constraints") or call_kwargs[1].get("constraints")
            assert constraints.max_steps == 5
            assert constraints.strategy == "majority"
            assert constraints.timeout_seconds == 60.0

        await engine.stop()

    async def test_solve_pipeline_error_propagates(self, engine):
        """SolvePipeline always returns SolveResult (never raises),
        so engine.solve should return a dict even on failure."""
        await engine.start()

        from multihead.solve_pipeline import SolveResult

        failed_result = SolveResult(
            run_id="", status="failed",
            output="Pipeline error: boom", confidence=0.0,
        )

        with patch("multihead.solve_pipeline.SolvePipeline") as MockPipeline:
            MockPipeline.return_value.solve = AsyncMock(return_value=failed_result)
            result = await engine.solve("bad task")

        assert result["status"] == "failed"
        assert "boom" in result["output"]
        await engine.stop()

    async def test_solve_dry_run(self, engine):
        """engine.solve(dry_run=True) should pass through to pipeline."""
        await engine.start()

        from multihead.solve_pipeline import SolveResult

        mock_result = SolveResult(
            run_id="dry-run-1", status="dry_run", output="plan",
            confidence=0.9, dry_run=True,
        )

        with patch("multihead.solve_pipeline.SolvePipeline") as MockPipeline:
            MockPipeline.return_value.solve = AsyncMock(return_value=mock_result)
            result = await engine.solve("task", dry_run=True)

        assert result["status"] == "dry_run"
        assert result["dry_run"] is True
        call_kwargs = MockPipeline.return_value.solve.call_args
        assert call_kwargs.kwargs.get("dry_run") is True
        await engine.stop()

    async def test_solve_enable_tests(self, engine):
        """engine.solve(enable_tests=True) should set constraint."""
        await engine.start()

        from multihead.solve_pipeline import SolveResult

        mock_result = SolveResult(
            run_id="r-tests", status="done", output="ok", confidence=0.9,
        )

        with patch("multihead.solve_pipeline.SolvePipeline") as MockPipeline:
            MockPipeline.return_value.solve = AsyncMock(return_value=mock_result)
            await engine.solve("task", enable_tests=True)

        call_kwargs = MockPipeline.return_value.solve.call_args
        constraints = call_kwargs.kwargs.get("constraints")
        assert constraints.enable_test_generation is True
        await engine.stop()


# ---------------------------------------------------------------------------
# Head management
# ---------------------------------------------------------------------------


class TestHeadManagement:
    async def test_get_states(self, engine):
        await engine.start()
        states = engine.get_states()
        assert "mock-llm" in states
        assert "mock-vlm" in states
        await engine.stop()

    async def test_wake_head(self, engine):
        await engine.start()
        await engine.wake("mock-llm")
        states = engine.get_states()
        assert states["mock-llm"]["state"] in ("active", "loading")
        await engine.stop()

    async def test_swap_head(self, engine):
        await engine.start()
        await engine.swap("mock-llm")
        # Should be active now
        states = engine.get_states()
        assert states["mock-llm"]["state"] == "active"
        await engine.stop()


# ---------------------------------------------------------------------------
# Package imports
# ---------------------------------------------------------------------------


class TestPackageExports:
    def test_import_engine(self):
        from multihead import Engine
        assert Engine is not None

    def test_import_head_manager(self):
        from multihead import HeadManager
        assert HeadManager is not None

    def test_import_knowledge_store(self):
        from multihead import KnowledgeStore
        assert KnowledgeStore is not None

    def test_import_router(self):
        from multihead import Router
        assert Router is not None

    def test_import_orchestrator(self):
        from multihead import Orchestrator
        assert Orchestrator is not None

    def test_import_settings(self):
        from multihead import Settings, load_heads
        assert Settings is not None
        assert load_heads is not None

    def test_import_models(self):
        from multihead import AdapterKind, HeadManifest, StepDef, WorkOrder
        assert AdapterKind is not None
        assert HeadManifest is not None
        assert StepDef is not None
        assert WorkOrder is not None

    def test_version(self):
        import multihead
        assert multihead.__version__ == "1.3.52"

    def test_import_rfq_manager(self):
        from multihead import RFQManager
        assert RFQManager is not None


# ---------------------------------------------------------------------------
# Marketplace
# ---------------------------------------------------------------------------


class TestMarketplaceProcure:
    async def test_procure_not_started_raises(self, engine):
        with pytest.raises(RuntimeError, match="not started"):
            await engine.marketplace_procure("text_generation", "test task")

    async def test_procure_missing_env_raises(self, engine):
        await engine.start()
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="not configured"):
                await engine.marketplace_procure("text_generation", "test task")
        await engine.stop()

    async def test_procure_calls_rfq_workflow(self, engine):
        await engine.start()
        mock_quote = MagicMock()
        mock_quote.price = 0.35
        mock_quote.estimated_latency_ms = 4000

        mock_rfq_mgr = AsyncMock()
        mock_rfq_mgr.rfq_workflow.return_value = {
            "contract_id": "c-123",
            "task_id": "t-456",
            "provider_id": "agent-789",
            "selected_quote": mock_quote,
        }

        env = {
            "ACP_CLOUD_URL": "https://cloud.example.com/api/v1",
            "ACP_CLOUD_API_KEY": "test-key",
            "ACP_CLOUD_PROJECT_ID": "test-proj",
        }

        with patch.dict("os.environ", env):
            with patch("multihead.engine._marketplace.RFQManager", return_value=mock_rfq_mgr):
                result = await engine.marketplace_procure(
                    "text_generation", "Summarize this doc",
                    max_price=0.50,
                )

        mock_rfq_mgr.rfq_workflow.assert_called_once_with(
            "text_generation", "Summarize this doc",
            max_price=0.50, max_latency_ms=None, min_quality=None,
            quote_timeout=30.0,
        )
        assert result["contract_id"] == "c-123"
        assert result["provider_id"] == "agent-789"
        assert result["quote_price"] == 0.35
        await engine.stop()

    async def test_procure_deposits_claim(self, engine):
        await engine.start()
        mock_quote = MagicMock()
        mock_quote.price = 0.20
        mock_quote.estimated_latency_ms = 3000

        mock_rfq_mgr = AsyncMock()
        mock_rfq_mgr.rfq_workflow.return_value = {
            "contract_id": "c-abc",
            "task_id": "t-def",
            "provider_id": "agent-ghi",
            "selected_quote": mock_quote,
        }

        env = {
            "ACP_CLOUD_URL": "https://cloud.example.com/api/v1",
            "ACP_CLOUD_API_KEY": "test-key",
        }

        with patch.dict("os.environ", env):
            with patch("multihead.engine._marketplace.RFQManager", return_value=mock_rfq_mgr):
                await engine.marketplace_procure("text_generation", "test")

        # Knowledge store should have a marketplace claim
        all_claims = engine.knowledge.list_claims(limit=50)
        marketplace_claims = [
            c for c in all_claims
            if c.canonical.claim_key.startswith("marketplace.procure.")
        ]
        assert len(marketplace_claims) >= 1
        assert "agent-ghi" in marketplace_claims[0].statement
        await engine.stop()

    async def test_procure_returns_result_shape(self, engine):
        await engine.start()
        mock_quote = MagicMock()
        mock_quote.price = 0.10
        mock_quote.estimated_latency_ms = 2000

        mock_rfq_mgr = AsyncMock()
        mock_rfq_mgr.rfq_workflow.return_value = {
            "contract_id": "c-x",
            "task_id": "t-y",
            "provider_id": "p-z",
            "selected_quote": mock_quote,
        }

        env = {
            "ACP_CLOUD_URL": "https://cloud.example.com/api/v1",
            "ACP_CLOUD_API_KEY": "k",
        }

        with patch.dict("os.environ", env):
            with patch("multihead.engine._marketplace.RFQManager", return_value=mock_rfq_mgr):
                result = await engine.marketplace_procure("code_generation", "fix bug")

        assert set(result.keys()) == {
            "contract_id", "task_id", "provider_id",
            "quote_price", "quote_latency_ms",
        }
        await engine.stop()


class TestMarketplaceSearch:
    async def test_search_not_started_raises(self, engine):
        with pytest.raises(RuntimeError, match="not started"):
            await engine.marketplace_search("text_generation")

    async def test_search_missing_env_raises(self, engine):
        await engine.start()
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="not configured"):
                await engine.marketplace_search("text_generation")
        await engine.stop()

    async def test_search_returns_listings(self, engine):
        await engine.start()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [
                {
                    "listing": {
                        "listing_id": "lst-1",
                        "agent_id": "agent-a",
                        "capability_id": "text_generation",
                        "name": "GPT-4 Expert",
                        "unit_price": 0.50,
                    },
                    "stats": {"quality_score": 0.92},
                },
            ],
        }
        mock_resp.raise_for_status = MagicMock()

        env = {
            "ACP_CLOUD_URL": "https://cloud.example.com/api/v1",
            "ACP_CLOUD_API_KEY": "test-key",
        }

        with patch.dict("os.environ", env):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get.return_value = mock_resp
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client

                results = await engine.marketplace_search("text_generation")

        assert len(results) == 1
        assert results[0]["listing_id"] == "lst-1"
        assert results[0]["quality_score"] == 0.92
        await engine.stop()

    async def test_search_empty_results(self, engine):
        await engine.start()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"results": []}
        mock_resp.raise_for_status = MagicMock()

        env = {
            "ACP_CLOUD_URL": "https://cloud.example.com/api/v1",
            "ACP_CLOUD_API_KEY": "test-key",
        }

        with patch.dict("os.environ", env):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get.return_value = mock_resp
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client

                results = await engine.marketplace_search("quantum_computing")

        assert results == []
        await engine.stop()
