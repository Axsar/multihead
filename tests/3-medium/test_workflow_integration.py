"""Tests for the 10-step workflow integration gaps.

ACP delegation, multi-candidate routing, knowledge feedback.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from multihead.artifact_store import ArtifactStore
from multihead.event_store import EventStore
from multihead.head_manager import HeadManager
from multihead.models import (
    AdapterKind,
    BudgetConstraint,
    Capability,
    DataSensitivity,
    HeadManifest,
    PrivacyConstraint,
    StepDef,
    WorkOrder,
)
from multihead.observability import MetricsCollector
from multihead.orchestrator import Orchestrator
from multihead.plan_normalizer import normalize
from multihead.router import Router


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def manifests():
    """Create a set of head manifests for testing."""
    return {
        "mock-llm": HeadManifest(
            head_id="mock-llm",
            name="Mock LLM",
            adapter=AdapterKind.MOCK,
            model="mock",
            kind="llm",
            capabilities=Capability(
                solver_type="llm",
                task_types=["text_generation", "code_generation", "data_processing"],
                input_modalities=["text"],
                output_modalities=["text"],
            ),
        ),
        "mock-vlm": HeadManifest(
            head_id="mock-vlm",
            name="Mock VLM",
            adapter=AdapterKind.MOCK,
            model="mock-v",
            kind="vlm",
            capabilities=Capability(
                solver_type="vlm",
                task_types=["visual_reasoning", "image_classification", "object_detection"],
                input_modalities=["text", "image"],
                output_modalities=["text"],
            ),
        ),
        "mock-llm-2": HeadManifest(
            head_id="mock-llm-2",
            name="Mock LLM 2",
            adapter=AdapterKind.MOCK,
            model="mock-2",
            kind="llm",
            capabilities=Capability(
                solver_type="llm",
                task_types=["text_generation", "data_processing"],
                input_modalities=["text"],
                output_modalities=["text"],
            ),
        ),
    }


@pytest.fixture
def head_manager(manifests):
    return HeadManager(manifests)


@pytest.fixture
def mock_acp_bridge():
    """Create a mock ACP bridge that simulates task creation and polling."""
    bridge = MagicMock()
    bridge.connected = True
    bridge.project_id = "test-project"

    # Track created tasks
    _tasks = {}
    _task_counter = [0]

    async def mock_request(method, path, **kwargs):
        if method == "POST" and path == "/tasks":
            _task_counter[0] += 1
            task_id = f"acp-task-{_task_counter[0]}"
            _tasks[task_id] = {
                "task_id": task_id,
                "status": "pending",
                "payload_ref": kwargs.get("json", {}).get("payload_ref", ""),
            }
            return {"task_id": task_id}

        if method == "GET" and path.startswith("/tasks/"):
            task_id = path.split("/tasks/")[1]
            task = _tasks.get(task_id)
            if task:
                # Simulate task completing on second poll
                if task["status"] == "pending":
                    task["status"] = "complete"
                    task["output_ref"] = f"Result for task {task_id}: done"
                return task
            return {"status": "failed", "message": "Task not found"}

        return {}

    bridge.request = AsyncMock(side_effect=mock_request)

    async def mock_stop():
        pass

    bridge.stop = AsyncMock(side_effect=mock_stop)
    return bridge


@pytest.fixture
def orchestrator_with_acp(tmp_path, head_manager, mock_acp_bridge):
    """Create orchestrator with ACP bridge wired."""
    runs_dir = tmp_path / "runs"
    artifacts_dir = tmp_path / "artifacts"
    db_path = tmp_path / "store.db"

    runs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    event_store = EventStore(runs_dir, db_path)
    artifact_store = ArtifactStore(artifacts_dir, db_path)
    metrics = MetricsCollector()

    return Orchestrator(
        event_store=event_store,
        artifact_store=artifact_store,
        head_manager=head_manager,
        runs_dir=runs_dir,
        metrics=metrics,
        enable_marketplace_delegation=True,
        acp_bridge=mock_acp_bridge,
    )


@pytest.fixture
def orchestrator_no_acp(tmp_path, head_manager):
    """Create orchestrator without ACP bridge."""
    runs_dir = tmp_path / "runs"
    artifacts_dir = tmp_path / "artifacts"
    db_path = tmp_path / "store.db"

    runs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    event_store = EventStore(runs_dir, db_path)
    artifact_store = ArtifactStore(artifacts_dir, db_path)
    metrics = MetricsCollector()

    return Orchestrator(
        event_store=event_store,
        artifact_store=artifact_store,
        head_manager=head_manager,
        runs_dir=runs_dir,
        metrics=metrics,
        enable_marketplace_delegation=False,
    )


# ---------------------------------------------------------------------------
# Test: Capability mapping (Step 3 in plan)
# ---------------------------------------------------------------------------

class TestCapabilityMapping:
    """Test _map_task_types_to_capability helper."""

    def test_text_generation_maps_to_llm(self, orchestrator_no_acp):
        cap = orchestrator_no_acp._map_task_types_to_capability
        assert cap(["text_generation"]) == "com.multihead.llm"

    def test_visual_reasoning_maps_to_vlm(self, orchestrator_no_acp):
        cap = orchestrator_no_acp._map_task_types_to_capability
        assert cap(["visual_reasoning"]) == "com.multihead.vlm"

    def test_code_editing_maps_to_claude(self, orchestrator_no_acp):
        cap = orchestrator_no_acp._map_task_types_to_capability
        assert cap(["code_editing"]) == "com.claude.code"

    def test_coordinate_transform_maps_to_deterministic(
        self, orchestrator_no_acp,
    ):
        cap = orchestrator_no_acp._map_task_types_to_capability
        assert cap(["coordinate_transform"]) == (
            "com.multihead.deterministic"
        )

    def test_object_detection_maps_to_vlm(self, orchestrator_no_acp):
        cap = orchestrator_no_acp._map_task_types_to_capability
        assert cap(["object_detection"]) == "com.multihead.vlm"

    def test_unknown_task_type_defaults_to_llm(self, orchestrator_no_acp):
        cap = orchestrator_no_acp._map_task_types_to_capability
        assert cap(["unknown_type"]) == "com.multihead.llm"

    def test_first_recognized_type_wins(self, orchestrator_no_acp):
        result = orchestrator_no_acp._map_task_types_to_capability(
            ["unknown", "visual_reasoning", "text_generation"]
        )
        assert result == "com.multihead.vlm"


# ---------------------------------------------------------------------------
# Test: ACP delegation (Step 2 in plan)
# ---------------------------------------------------------------------------

class TestACPDelegation:
    """Test _delegate_to_marketplace with real ACP task submission."""

    @pytest.mark.asyncio
    async def test_creates_acp_task_and_polls_result(self, orchestrator_with_acp, tmp_path):
        """Verify delegation creates an ACP task, polls for result, and stores artifact."""
        work_order = WorkOrder(
            run_id="test-acp-run",
            goal="Test ACP delegation",
            steps=[
                StepDef(
                    step_id="acp-step",
                    name="Delegate to ACP",
                    task_types=["text_generation"],
                    budget=BudgetConstraint(max_total_cost=100.0),
                    prompt_template="Generate something useful",
                ),
            ],
        )

        state = await orchestrator_with_acp.create_run(work_order, normalize=False)
        run_artifacts_dir = tmp_path / "runs" / state.run_id / "artifacts"
        run_artifacts_dir.mkdir(parents=True, exist_ok=True)

        result = await orchestrator_with_acp._delegate_to_marketplace(
            run_id=state.run_id,
            step=work_order.steps[0],
            work_order=work_order,
            state=state,
            run_artifacts_dir=run_artifacts_dir,
        )

        # Verify ACP task was created
        assert result.status.value == "committed"
        assert result.head_id == "marketplace-acp"
        assert "acp_task_id" in result.outputs
        assert result.metrics["delegation"] == "acp"
        assert result.metrics["capability"] == "com.multihead.llm"

        # Verify the ACP bridge was called correctly
        bridge = orchestrator_with_acp.acp_bridge
        calls = bridge.request.call_args_list
        # First call: POST /tasks
        assert calls[0][0] == ("POST", "/tasks")
        task_payload = calls[0][1]["json"]
        assert task_payload["required_capability"] == "com.multihead.llm"
        assert task_payload["conversation_id"] == "test-acp-run"

    @pytest.mark.asyncio
    async def test_falls_back_to_rfq_when_no_bridge(self, orchestrator_no_acp, tmp_path):
        """Verify fallback to RFQ-only when ACP bridge is not available."""
        orchestrator_no_acp.enable_marketplace_delegation = True
        work_order = WorkOrder(
            run_id="test-rfq-run",
            goal="Test RFQ fallback",
            steps=[
                StepDef(
                    step_id="rfq-step",
                    name="Detect objects",
                    task_types=["object_detection"],
                    budget=BudgetConstraint(max_total_cost=500.0),
                ),
            ],
        )

        state = await orchestrator_no_acp.create_run(work_order, normalize=False)
        run_artifacts_dir = tmp_path / "runs" / state.run_id / "artifacts"
        run_artifacts_dir.mkdir(parents=True, exist_ok=True)

        result = await orchestrator_no_acp._delegate_to_marketplace(
            run_id=state.run_id,
            step=work_order.steps[0],
            work_order=work_order,
            state=state,
            run_artifacts_dir=run_artifacts_dir,
        )

        # Should fall back to rfq_only
        assert result.status.value == "committed"
        assert result.head_id == "marketplace-rfq"
        assert result.metrics["delegation"] == "rfq_only"

    @pytest.mark.asyncio
    async def test_acp_task_failure_returns_failed_result(self, orchestrator_with_acp, tmp_path):
        """Verify that ACP task failure is properly propagated."""
        # Override mock to simulate failure
        async def failing_request(method, path, **kwargs):
            if method == "POST" and path == "/tasks":
                return {"task_id": "fail-task"}
            if method == "GET":
                return {"status": "failed", "message": "Agent crashed"}
            return {}

        orchestrator_with_acp.acp_bridge.request = AsyncMock(side_effect=failing_request)

        work_order = WorkOrder(
            run_id="test-fail-run",
            goal="Test failure",
            steps=[
                StepDef(
                    step_id="fail-step",
                    name="Will fail",
                    task_types=["text_generation"],
                    budget=BudgetConstraint(max_total_cost=100.0),
                ),
            ],
        )

        state = await orchestrator_with_acp.create_run(work_order, normalize=False)
        run_artifacts_dir = tmp_path / "runs" / state.run_id / "artifacts"
        run_artifacts_dir.mkdir(parents=True, exist_ok=True)

        result = await orchestrator_with_acp._delegate_to_marketplace(
            run_id=state.run_id,
            step=work_order.steps[0],
            work_order=work_order,
            state=state,
            run_artifacts_dir=run_artifacts_dir,
        )

        assert result.status.value == "failed"
        assert "Agent crashed" in result.error

    @pytest.mark.asyncio
    async def test_full_run_with_acp_delegation(self, orchestrator_with_acp):
        """Test a full run where eligible steps delegate to ACP."""
        work_order = WorkOrder(
            run_id="test-full-run",
            goal="Full run with ACP delegation",
            steps=[
                # Step 1: local (no budget)
                StepDef(
                    step_id="local-step",
                    name="Local step",
                    head_id="mock-llm",
                ),
                # Step 2: delegate to ACP (has budget + task_types)
                StepDef(
                    step_id="acp-step",
                    name="ACP step",
                    task_types=["visual_reasoning"],
                    budget=BudgetConstraint(max_total_cost=50.0),
                ),
            ],
        )

        state = await orchestrator_with_acp.create_run(work_order, normalize=False)
        state = await orchestrator_with_acp.execute_run(state.run_id)

        # Step 1 ran locally
        assert state.step_results["local-step"].head_id == "mock-llm"
        # Step 2 delegated to ACP
        assert state.step_results["acp-step"].head_id == "marketplace-acp"
        assert state.step_results["acp-step"].metrics["delegation"] == "acp"


# ---------------------------------------------------------------------------
# Test: Multi-candidate routing (Step 4 in plan)
# ---------------------------------------------------------------------------

class TestMultiCandidateRouting:
    """Test that plan normalizer populates fallback lists from ranked candidates."""

    def test_rank_by_task_returns_multiple_candidates(self, head_manager):
        """Router.rank_by_task returns all matching candidates sorted."""
        router = Router(head_manager)
        ranked = router.rank_by_task(task_types=["text_generation"])

        # Both mock-llm and mock-llm-2 have text_generation
        assert len(ranked) >= 2
        # Results are sorted by score (descending)
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_by_task_empty_for_no_match(self, head_manager):
        """Router.rank_by_task returns empty for unmatched tasks."""
        router = Router(head_manager)
        ranked = router.rank_by_task(task_types=["quantum_computing"])
        assert ranked == []

    def test_normalize_populates_fallbacks_from_ranking(self, head_manager):
        """Plan normalizer fills step.fallback from multi-candidate ranking."""
        wo = WorkOrder(
            run_id="test-routing",
            goal="Test multi-candidate routing",
            steps=[
                StepDef(
                    step_id="step-1",
                    name="Generate text",
                    # No head_id set — needs routing
                    task_types=["text_generation"],
                ),
            ],
        )

        result = normalize(wo, head_manager)
        step = result.steps[0]

        # Should have been routed
        assert step.head_id != ""
        # Should have fallback(s) populated
        assert len(step.fallback) >= 1
        # Fallback should not include the primary
        assert step.head_id not in step.fallback

    def test_normalize_preserves_explicit_fallbacks(self, head_manager):
        """Explicit fallbacks are not overwritten by routing."""
        wo = WorkOrder(
            run_id="test-explicit-fb",
            goal="Test explicit fallback preservation",
            steps=[
                StepDef(
                    step_id="step-1",
                    name="Generate text",
                    task_types=["text_generation"],
                    fallback=["mock-llm-2"],
                ),
            ],
        )

        result = normalize(wo, head_manager)
        step = result.steps[0]

        # Original fallback should be preserved
        assert step.fallback == ["mock-llm-2"]

    def test_normalize_kind_routing_populates_fallbacks(self, head_manager):
        """required_kind routing also fills fallbacks."""
        wo = WorkOrder(
            run_id="test-kind-fb",
            goal="Test kind-based fallback population",
            steps=[
                StepDef(
                    step_id="step-1",
                    name="LLM task",
                    required_kind="llm",
                ),
            ],
        )

        result = normalize(wo, head_manager)
        step = result.steps[0]

        assert step.head_id != ""
        # Both mock-llm and mock-llm-2 are LLM kind, so one is primary, other is fallback
        assert len(step.fallback) >= 1


# ---------------------------------------------------------------------------
# Test: Router knowledge feedback (Step 6 in plan)
# ---------------------------------------------------------------------------

class TestRouterKnowledgeFeedback:
    """Test that Router reads knowledge claims and adjusts head scores."""

    def _make_mock_knowledge_store(self, claims):
        """Create a mock knowledge store with predefined claims."""
        store = MagicMock()
        store.list_claims.return_value = claims
        return store

    def _make_claim(self, key, confidence=0.9, status="accepted"):
        """Create a mock claim object."""
        claim = MagicMock()
        claim.claim_key = key
        claim.confidence = confidence
        claim.claim_status = status
        return claim

    def test_success_claim_boosts_score(self, head_manager):
        """Head with success claims gets a score boost."""
        claims = [self._make_claim("head.mock-llm.success", confidence=0.9)]
        ks = self._make_mock_knowledge_store(claims)

        router = Router(head_manager, knowledge_store=ks)

        # Get scores for LLM heads
        score_with_success = router._score("mock-llm")

        # Create router without knowledge
        router_no_ks = Router(head_manager)
        score_without = router_no_ks._score("mock-llm")

        # Score should be boosted
        assert score_with_success > score_without

    def test_failure_claim_penalizes_score(self, head_manager):
        """Head with failure claims gets a score penalty."""
        claims = [self._make_claim("head.mock-llm.failure", confidence=0.9)]
        ks = self._make_mock_knowledge_store(claims)

        router = Router(head_manager, knowledge_store=ks)

        score_with_failure = router._score("mock-llm")

        router_no_ks = Router(head_manager)
        score_without = router_no_ks._score("mock-llm")

        # Score should be penalized
        assert score_with_failure < score_without

    def test_no_knowledge_store_no_change(self, head_manager):
        """Without knowledge_store, no boost/penalty applied."""
        router = Router(head_manager)
        boost = router._get_knowledge_boost("mock-llm")
        assert boost == 0.0

    def test_knowledge_boost_clamped(self, head_manager):
        """Knowledge boost is clamped to [-10, +10]."""
        # Create many success claims to test clamping
        claims = [
            self._make_claim(f"head.mock-llm.success", confidence=0.9)
            for _ in range(10)
        ]
        ks = self._make_mock_knowledge_store(claims)

        router = Router(head_manager, knowledge_store=ks)
        boost = router._get_knowledge_boost("mock-llm")
        assert -10.0 <= boost <= 10.0

    def test_knowledge_cache_built_once(self, head_manager):
        """Knowledge cache is built once and reused."""
        claims = [self._make_claim("head.mock-llm.success")]
        ks = self._make_mock_knowledge_store(claims)

        router = Router(head_manager, knowledge_store=ks)

        # First call builds cache
        router._get_knowledge_boost("mock-llm")
        # Second call uses cache
        router._get_knowledge_boost("mock-llm")

        # list_claims should only be called once
        ks.list_claims.assert_called_once()

    def test_knowledge_feedback_in_capability_scoring(self, head_manager):
        """Knowledge feedback also applies in _score_with_capability."""
        claims = [self._make_claim("head.mock-llm.success", confidence=0.9)]
        ks = self._make_mock_knowledge_store(claims)

        router_with_ks = Router(head_manager, knowledge_store=ks)
        router_without = Router(head_manager)

        score_with = router_with_ks._score_with_capability("mock-llm", ["text_generation"])
        score_without = router_without._score_with_capability("mock-llm", ["text_generation"])

        # Score with knowledge boost should be higher
        assert score_with > score_without

    def test_non_head_claims_ignored(self, head_manager):
        """Claims with keys not matching head.* pattern are ignored."""
        claims = [
            self._make_claim("project.status.active"),
            self._make_claim("mesh.presence.node-1"),
        ]
        ks = self._make_mock_knowledge_store(claims)

        router = Router(head_manager, knowledge_store=ks)
        boost = router._get_knowledge_boost("mock-llm")
        assert boost == 0.0


# ---------------------------------------------------------------------------
# Test: Orchestrator ACP bridge wiring (Step 1 in plan)
# ---------------------------------------------------------------------------

class TestOrchestratorACPWiring:
    """Test that Orchestrator accepts and stores acp_bridge."""

    def test_acp_bridge_stored(self, orchestrator_with_acp, mock_acp_bridge):
        """acp_bridge is stored on orchestrator."""
        assert orchestrator_with_acp.acp_bridge is mock_acp_bridge

    def test_acp_bridge_defaults_to_none(self, orchestrator_no_acp):
        """acp_bridge defaults to None when not provided."""
        assert orchestrator_no_acp.acp_bridge is None

    def test_marketplace_delegation_enabled_with_bridge(self, orchestrator_with_acp):
        """enable_marketplace_delegation is True when bridge provided."""
        assert orchestrator_with_acp.enable_marketplace_delegation is True

    def test_marketplace_delegation_disabled_without_bridge(self, orchestrator_no_acp):
        """enable_marketplace_delegation is False when no bridge."""
        assert orchestrator_no_acp.enable_marketplace_delegation is False


# ---------------------------------------------------------------------------
# Test: End-to-end integration (solve → ACP → result)
# ---------------------------------------------------------------------------

class TestEndToEndIntegration:
    """Integration tests for the full workflow."""

    @pytest.mark.asyncio
    async def test_local_execution_unchanged_without_acp(self, orchestrator_no_acp):
        """Verify local execution works identically when no ACP configured."""
        wo = WorkOrder(
            run_id="test-local",
            goal="Local only task",
            steps=[
                StepDef(
                    step_id="step1",
                    name="Local step",
                    head_id="mock-llm",
                    prompt_template="Do something locally",
                ),
            ],
        )

        state = await orchestrator_no_acp.create_run(wo, normalize=False)
        state = await orchestrator_no_acp.execute_run(state.run_id)

        assert state.status.value == "done"
        assert state.step_results["step1"].head_id == "mock-llm"
        assert state.step_results["step1"].status.value == "committed"

    @pytest.mark.asyncio
    async def test_mixed_local_and_acp_execution(self, orchestrator_with_acp):
        """Test workflow mixing local and ACP-delegated steps."""
        wo = WorkOrder(
            run_id="test-mixed",
            goal="Mixed local and ACP",
            steps=[
                # Step 1: local (no budget)
                StepDef(
                    step_id="step1-local",
                    name="Preprocess",
                    head_id="mock-llm",
                ),
                # Step 2: ACP (has budget + task_types)
                StepDef(
                    step_id="step2-acp",
                    name="Process remotely",
                    task_types=["text_generation"],
                    budget=BudgetConstraint(max_total_cost=100.0),
                    prompt_template="Process this data",
                ),
                # Step 3: local (confidential — never delegates)
                StepDef(
                    step_id="step3-local",
                    name="Confidential processing",
                    head_id="mock-llm",
                    task_types=["data_processing"],
                    budget=BudgetConstraint(max_total_cost=100.0),
                    privacy=PrivacyConstraint(
                        data_sensitivity=DataSensitivity.CONFIDENTIAL,
                    ),
                ),
            ],
        )

        state = await orchestrator_with_acp.create_run(wo, normalize=False)
        state = await orchestrator_with_acp.execute_run(state.run_id)

        assert state.status.value == "done"

        # Step 1: local
        assert state.step_results["step1-local"].head_id == "mock-llm"

        # Step 2: ACP delegation
        assert state.step_results["step2-acp"].head_id == "marketplace-acp"
        assert state.step_results["step2-acp"].metrics["delegation"] == "acp"

        # Step 3: local (confidential)
        assert state.step_results["step3-local"].head_id == "mock-llm"
