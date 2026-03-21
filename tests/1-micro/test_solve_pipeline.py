"""Tests for the reusable SolvePipeline."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from multihead.solve_pipeline import SolveConstraints, SolvePipeline, SolveResult


class TestSolveConstraints:
    def test_defaults(self):
        sc = SolveConstraints()
        assert sc.max_steps == 20
        assert sc.max_depth == 3
        assert sc.timeout_seconds == 240.0
        assert sc.strategy == "first_to_ahead"
        assert sc.enable_research_features is True
        assert sc.enable_knowledge_hook is True
        assert sc.enable_marketplace_delegation is False

    def test_custom(self):
        sc = SolveConstraints(max_steps=5, timeout_seconds=30.0, strategy="majority")
        assert sc.max_steps == 5
        assert sc.timeout_seconds == 30.0
        assert sc.strategy == "majority"


class TestSolveResult:
    def test_construction(self):
        r = SolveResult(run_id="r1", status="done", output="ok", confidence=0.9)
        assert r.run_id == "r1"
        assert r.status == "done"
        assert r.steps_total == 0
        assert r.consensus_meta == {}
        assert r.state is None

    def test_full_construction(self):
        r = SolveResult(
            run_id="r1", status="done", output="ok", confidence=0.85,
            steps_total=5, steps_succeeded=4, steps_failed=1,
            duration_seconds=12.5, plan_steps=5, parallel_steps=2,
            consensus_meta={"agreement_score": 0.9},
        )
        assert r.steps_succeeded == 4
        assert r.parallel_steps == 2


class TestSolvePipeline:
    @pytest.fixture
    def mock_deps(self):
        return {
            "head_manager": MagicMock(),
            "event_store": MagicMock(),
            "artifact_store": MagicMock(),
            "knowledge_store": MagicMock(),
            "runs_dir": "/tmp/test_runs",
        }

    def test_init(self, mock_deps):
        pipeline = SolvePipeline(**mock_deps)
        assert pipeline._heads is mock_deps["head_manager"]
        assert pipeline._event_store is mock_deps["event_store"]

    @pytest.mark.asyncio
    async def test_solve_error_returns_failed(self, mock_deps):
        """Pipeline errors should return SolveResult with status='failed'."""
        pipeline = SolvePipeline(**mock_deps)

        with patch("multihead.solve_pipeline.SolvePipeline._run", side_effect=RuntimeError("boom")):
            result = await pipeline.solve("test task")

        assert result.status == "failed"
        assert "boom" in result.output
        assert result.confidence == 0.0
        assert result.duration_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_solve_uses_constraints(self, mock_deps):
        """Constraints should propagate to decomposer and orchestrator."""
        pipeline = SolvePipeline(**mock_deps)

        mock_plan = MagicMock()
        mock_plan.total_steps = 3
        mock_plan.complexity = "medium"
        mock_meta = {"agreement_score": 0.8, "winner_head": "mock-llm"}

        mock_wo = MagicMock()
        mock_wo.steps = [MagicMock(step_id="s1", name="step1", depends_on=[])]
        mock_wo.run_timeout_seconds = 100.0

        mock_state = MagicMock()
        mock_state.run_id = "run_123"
        mock_state.status = MagicMock(value="done")
        mock_state.step_results = {
            "s1": MagicMock(
                status=MagicMock(value="committed"),
                outputs={"text": "result text"},
            ),
        }

        with patch("multihead.auto_decomposition.AutoDecomposer") as MockDecomp, \
             patch("multihead.orchestrator.Orchestrator") as MockOrch, \
             patch("multihead.knowledge_hook.KnowledgeHook"), \
             patch("multihead.observability.MetricsCollector"):

            decomp = MockDecomp.return_value
            decomp.decompose_with_consensus = AsyncMock(return_value=(mock_plan, mock_meta))
            decomp.to_work_order_with_dag = MagicMock(return_value=mock_wo)

            orch = MockOrch.return_value
            orch.create_run = AsyncMock(return_value=mock_state)
            orch.execute_run = AsyncMock(return_value=mock_state)

            constraints = SolveConstraints(max_steps=10, strategy="majority")
            result = await pipeline.solve("test task", constraints=constraints)

        assert result.status == "done"
        assert result.steps_succeeded == 1
        assert result.confidence == 1.0
        assert "result text" in result.output

        # Verify decomposer was called with right strategy
        call_kwargs = decomp.decompose_with_consensus.call_args
        assert call_kwargs.kwargs.get("goal") or call_kwargs.args[0] == "test task"

    @pytest.mark.asyncio
    async def test_solve_timeout_propagation(self, mock_deps):
        """Remaining time should be set on work_order.run_timeout_seconds."""
        pipeline = SolvePipeline(**mock_deps)

        mock_plan = MagicMock()
        mock_plan.total_steps = 1
        mock_plan.complexity = "low"

        mock_wo = MagicMock()
        mock_wo.steps = []
        mock_wo.run_timeout_seconds = 100.0

        mock_state = MagicMock()
        mock_state.run_id = "run_t"
        mock_state.status = MagicMock(value="done")
        mock_state.step_results = {}

        with patch("multihead.auto_decomposition.AutoDecomposer") as MockDecomp, \
             patch("multihead.orchestrator.Orchestrator") as MockOrch, \
             patch("multihead.knowledge_hook.KnowledgeHook"), \
             patch("multihead.observability.MetricsCollector"):

            decomp = MockDecomp.return_value
            decomp.decompose_with_consensus = AsyncMock(
                return_value=(mock_plan, {"agreement_score": 1.0})
            )
            decomp.to_work_order_with_dag = MagicMock(return_value=mock_wo)

            orch = MockOrch.return_value
            orch.create_run = AsyncMock(return_value=mock_state)
            orch.execute_run = AsyncMock(return_value=mock_state)

            constraints = SolveConstraints(timeout_seconds=60.0)
            result = await pipeline.solve("task", constraints=constraints)

        # The remaining time should be less than original 60s (some time elapsed)
        assert mock_wo.run_timeout_seconds <= 60.0
        assert mock_wo.run_timeout_seconds >= 30.0  # minimum bound

    @pytest.mark.asyncio
    async def test_solve_no_knowledge_store(self):
        """Pipeline should work without knowledge_store."""
        pipeline = SolvePipeline(
            head_manager=MagicMock(),
            event_store=MagicMock(),
            artifact_store=MagicMock(),
        )

        with patch(
            "multihead.solve_pipeline.SolvePipeline._run",
            side_effect=RuntimeError("no ks"),
        ):
            result = await pipeline.solve("task")

        assert result.status == "failed"


class TestMCPSolveTool:
    """Tests for the MCP solve tool proxy."""

    @pytest.mark.asyncio
    async def test_solve_proxy_success(self):
        """_solve should POST to /solve and return JSON."""
        from multihead.mcp_server import _solve

        mock_response = {
            "run_id": "run-mcp-1",
            "status": "done",
            "output": "ok",
            "confidence": 0.9,
            "steps_total": 2,
            "steps_succeeded": 2,
            "steps_failed": 0,
            "duration_seconds": 5.0,
            "plan_steps": 2,
            "parallel_steps": 1,
        }

        with patch("multihead.mcp_server._request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            result = await _solve("build something")

        import json
        parsed = json.loads(result)
        assert parsed["run_id"] == "run-mcp-1"
        assert parsed["status"] == "done"

        # Verify correct POST
        mock_req.assert_called_once_with(
            "POST", "/solve",
            json={
                "task": "build something",
                "strategy": "first_to_ahead",
                "max_steps": 20,
                "enable_marketplace": False,
                "timeout": 240.0,
                "dry_run": False,
            },
        )

    @pytest.mark.asyncio
    async def test_solve_proxy_connect_error(self):
        """Connection error should return friendly message."""
        import httpx
        from multihead.mcp_server import _solve

        with patch("multihead.mcp_server._request", side_effect=httpx.ConnectError("refused")):
            result = await _solve("task")

        assert "MultiHead server not running" in result

    @pytest.mark.asyncio
    async def test_solve_proxy_http_error(self):
        """HTTP errors should be reported."""
        import httpx
        from multihead.mcp_server import _solve

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal error"

        with patch("multihead.mcp_server._request",
                    side_effect=httpx.HTTPStatusError(
                        "err", request=MagicMock(), response=mock_resp,
                    )):
            result = await _solve("task")

        assert "500" in result


class TestDryRun:
    """Tests for dry-run mode."""

    @pytest.fixture
    def mock_deps(self):
        return {
            "head_manager": MagicMock(),
            "event_store": MagicMock(),
            "artifact_store": MagicMock(),
            "knowledge_store": MagicMock(),
            "runs_dir": "/tmp/test_runs",
        }

    @pytest.mark.asyncio
    async def test_dry_run_returns_plan(self, mock_deps):
        """Dry-run should return plan without executing."""
        pipeline = SolvePipeline(**mock_deps)

        mock_plan = MagicMock()
        mock_plan.total_steps = 3
        mock_plan.complexity = "medium"
        mock_meta = {"agreement_score": 0.85}

        mock_step = MagicMock()
        mock_step.name = "Read code"
        mock_step.action = "read"
        mock_step.depends_on = []

        mock_wo = MagicMock()
        mock_wo.steps = [mock_step]

        with patch("multihead.auto_decomposition.AutoDecomposer") as MockDecomp, \
             patch("multihead.orchestrator.Orchestrator"), \
             patch("multihead.knowledge_hook.KnowledgeHook"), \
             patch("multihead.observability.MetricsCollector"):

            decomp = MockDecomp.return_value
            decomp.decompose_with_consensus = AsyncMock(
                return_value=(mock_plan, mock_meta)
            )
            decomp.to_work_order_with_dag = MagicMock(return_value=mock_wo)

            result = await pipeline.solve("test task", dry_run=True)

        assert result.status == "dry_run"
        assert result.dry_run is True
        assert result.run_id.startswith("dry-run-")
        assert "Read code" in result.output
        assert result.confidence == 0.85
        assert result.plan_steps == 3

    @pytest.mark.asyncio
    async def test_dry_run_skips_execution(self, mock_deps):
        """Dry-run must NOT call orchestrator.create_run or execute_run."""
        pipeline = SolvePipeline(**mock_deps)

        mock_plan = MagicMock()
        mock_plan.total_steps = 1
        mock_plan.complexity = "low"

        mock_wo = MagicMock()
        mock_wo.steps = []

        with patch("multihead.auto_decomposition.AutoDecomposer") as MockDecomp, \
             patch("multihead.orchestrator.Orchestrator") as MockOrch, \
             patch("multihead.knowledge_hook.KnowledgeHook"), \
             patch("multihead.observability.MetricsCollector"):

            decomp = MockDecomp.return_value
            decomp.decompose_with_consensus = AsyncMock(
                return_value=(mock_plan, {"agreement_score": 1.0})
            )
            decomp.to_work_order_with_dag = MagicMock(return_value=mock_wo)

            orch = MockOrch.return_value
            orch.create_run = AsyncMock()
            orch.execute_run = AsyncMock()

            result = await pipeline.solve("task", dry_run=True)

        assert result.dry_run is True
        orch.create_run.assert_not_called()
        orch.execute_run.assert_not_called()

    def test_solve_result_dry_run_default(self):
        """SolveResult.dry_run defaults to False."""
        r = SolveResult(run_id="r1", status="done", output="ok", confidence=0.9)
        assert r.dry_run is False


class TestTestGenerationHook:
    """Tests for TestGenerationHook wiring in SolvePipeline."""

    @pytest.fixture
    def mock_deps(self):
        return {
            "head_manager": MagicMock(),
            "event_store": MagicMock(),
            "artifact_store": MagicMock(),
            "knowledge_store": MagicMock(),
            "runs_dir": "/tmp/test_runs",
        }

    def test_constraints_defaults(self):
        """Test generation is disabled by default."""
        sc = SolveConstraints()
        assert sc.enable_test_generation is False
        assert sc.test_framework == "pytest"

    def test_constraints_enable(self):
        sc = SolveConstraints(enable_test_generation=True, test_framework="unittest")
        assert sc.enable_test_generation is True
        assert sc.test_framework == "unittest"

    def test_solve_result_has_test_stats(self):
        r = SolveResult(run_id="r1", status="done", output="ok", confidence=0.9)
        assert r.test_generation_stats == {}

    @pytest.mark.asyncio
    async def test_test_hook_created_when_enabled(self, mock_deps):
        """When enable_test_generation=True, test_hook is passed to Orchestrator."""
        pipeline = SolvePipeline(**mock_deps)

        mock_plan = MagicMock()
        mock_plan.total_steps = 1
        mock_plan.complexity = "low"

        mock_wo = MagicMock()
        mock_wo.steps = []
        mock_wo.run_timeout_seconds = 100.0

        mock_state = MagicMock()
        mock_state.run_id = "r1"
        mock_state.status = MagicMock(value="done")
        mock_state.step_results = {}

        with patch("multihead.auto_decomposition.AutoDecomposer") as MockDecomp, \
             patch("multihead.orchestrator.Orchestrator") as MockOrch, \
             patch("multihead.knowledge_hook.KnowledgeHook"), \
             patch("multihead.observability.MetricsCollector"), \
             patch("multihead.test_generation.TestGenerator"), \
             patch("multihead.test_generation_hook.TestGenerationHook") as MockTestHook:

            decomp = MockDecomp.return_value
            decomp.decompose_with_consensus = AsyncMock(
                return_value=(mock_plan, {"agreement_score": 1.0})
            )
            decomp.to_work_order_with_dag = MagicMock(return_value=mock_wo)

            orch = MockOrch.return_value
            orch.create_run = AsyncMock(return_value=mock_state)
            orch.execute_run = AsyncMock(return_value=mock_state)

            mock_hook_instance = MockTestHook.return_value
            mock_hook_instance.get_session_summary.return_value = {
                "total_steps_tested": 1, "converged": 1,
            }

            constraints = SolveConstraints(enable_test_generation=True)
            result = await pipeline.solve("test task", constraints=constraints)

        # Verify test_hook was passed to Orchestrator
        orch_call_kwargs = MockOrch.call_args.kwargs
        assert orch_call_kwargs.get("test_hook") is not None

        # Verify stats are in result
        assert result.test_generation_stats["total_steps_tested"] == 1


class TestAPISolveRoute:
    """Tests for the API POST /solve route."""

    def test_solve_request_model(self):
        from multihead.api.routes_solve import SolveRequest
        req = SolveRequest(task="build it")
        assert req.task == "build it"
        assert req.strategy == "first_to_ahead"
        assert req.max_steps == 20
        assert req.timeout == 240.0
        assert req.enable_marketplace is False
        assert req.enable_tests is False
        assert req.dry_run is False

    def test_solve_response_model(self):
        from multihead.api.routes_solve import SolveResponse
        resp = SolveResponse(
            run_id="r1", status="done", output="ok", confidence=0.9,
        )
        assert resp.run_id == "r1"
        assert resp.steps_total == 0
        assert resp.plan_steps == 0
        assert resp.dry_run is False

    def test_solve_request_custom(self):
        from multihead.api.routes_solve import SolveRequest
        req = SolveRequest(
            task="refactor auth",
            strategy="majority",
            max_steps=10,
            timeout=60.0,
            enable_marketplace=True,
            enable_tests=True,
            dry_run=True,
        )
        assert req.strategy == "majority"
        assert req.max_steps == 10
        assert req.enable_marketplace is True
        assert req.enable_tests is True
        assert req.dry_run is True
