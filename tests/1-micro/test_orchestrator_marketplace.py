"""Tests for Orchestrator marketplace delegation integration (Phase 6)."""

from __future__ import annotations

import json

import pytest

from multihead.artifact_store import ArtifactStore
from multihead.event_store import EventStore
from multihead.head_manager import HeadManager
from multihead.models import (
    AdapterKind,
    BudgetConstraint,
    DataSensitivity,
    HeadManifest,
    PrivacyConstraint,
    StepDef,
    WorkOrder,
)
from multihead.observability import MetricsCollector
from multihead.orchestrator import Orchestrator


@pytest.fixture
def simple_head_manager():
    """Create minimal head manager for testing."""
    manifests = {
        "mock-llm": HeadManifest(
            head_id="mock-llm",
            name="Mock LLM",
            adapter=AdapterKind.MOCK,
            model="mock",
            kind="llm",
        ),
    }
    return HeadManager(manifests)


@pytest.fixture
def orchestrator(tmp_path, simple_head_manager):
    """Create orchestrator with marketplace delegation enabled."""
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
        head_manager=simple_head_manager,
        runs_dir=runs_dir,
        metrics=metrics,
        enable_marketplace_delegation=True,  # Enable Phase 6
    )


@pytest.fixture
def orchestrator_without_marketplace(tmp_path, simple_head_manager):
    """Create orchestrator with marketplace delegation disabled."""
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
        head_manager=simple_head_manager,
        runs_dir=runs_dir,
        metrics=metrics,
        enable_marketplace_delegation=False,  # Disabled
    )


class TestMarketplaceDelegationDecision:
    """Test _should_delegate_to_marketplace logic."""

    def test_delegates_when_enabled_with_budget(self, orchestrator):
        """Test delegation when conditions are met."""
        step = StepDef(
            step_id="test-step",
            name="Test Step",
            task_types=["object_detection"],
            budget=BudgetConstraint(
                max_cost_per_step=0.05,
                max_total_cost=500.0,
            ),
            privacy=PrivacyConstraint(
                data_sensitivity=DataSensitivity.INTERNAL,
            ),
        )

        should_delegate = orchestrator._should_delegate_to_marketplace(step)
        assert should_delegate is True

    def test_no_delegation_when_disabled(self, orchestrator_without_marketplace):
        """Test no delegation when marketplace is disabled."""
        step = StepDef(
            step_id="test-step",
            name="Test Step",
            task_types=["object_detection"],
            budget=BudgetConstraint(max_total_cost=500.0),
        )

        should_delegate = orchestrator_without_marketplace._should_delegate_to_marketplace(step)
        assert should_delegate is False

    def test_no_delegation_without_budget(self, orchestrator):
        """Test no delegation when step has no budget."""
        step = StepDef(
            step_id="test-step",
            name="Test Step",
            task_types=["object_detection"],
            # No budget
        )

        should_delegate = orchestrator._should_delegate_to_marketplace(step)
        assert should_delegate is False

    def test_no_delegation_without_task_types(self, orchestrator):
        """Test no delegation when step has no task types."""
        step = StepDef(
            step_id="test-step",
            name="Test Step",
            # No task_types
            budget=BudgetConstraint(max_total_cost=500.0),
        )

        should_delegate = orchestrator._should_delegate_to_marketplace(step)
        assert should_delegate is False

    def test_no_delegation_for_confidential_data(self, orchestrator):
        """Test CONFIDENTIAL data is never delegated."""
        step = StepDef(
            step_id="test-step",
            name="Test Step",
            task_types=["object_detection"],
            budget=BudgetConstraint(max_total_cost=500.0),
            privacy=PrivacyConstraint(
                data_sensitivity=DataSensitivity.CONFIDENTIAL,  # Cannot delegate!
            ),
        )

        should_delegate = orchestrator._should_delegate_to_marketplace(step)
        assert should_delegate is False

    def test_delegates_internal_data(self, orchestrator):
        """Test INTERNAL data can be delegated (with encryption)."""
        step = StepDef(
            step_id="test-step",
            name="Test Step",
            task_types=["object_detection"],
            budget=BudgetConstraint(max_total_cost=500.0),
            privacy=PrivacyConstraint(
                data_sensitivity=DataSensitivity.INTERNAL,
                require_encryption=True,
            ),
        )

        should_delegate = orchestrator._should_delegate_to_marketplace(step)
        assert should_delegate is True


class TestMarketplaceDelegation:
    """Test marketplace delegation execution."""

    @pytest.mark.asyncio
    async def test_delegate_to_marketplace_creates_rfq(self, orchestrator, tmp_path):
        """Test delegation creates RFQ successfully."""
        work_order = WorkOrder(
            run_id="test-run",
            goal="Test marketplace delegation",
            steps=[
                StepDef(
                    step_id="delegate-step",
                    name="Detect objects",
                    task_types=["object_detection"],
                    budget=BudgetConstraint(
                        max_cost_per_step=0.05,
                        max_total_cost=500.0,
                        max_total_time_s=43200,  # 12 hours
                    ),
                    privacy=PrivacyConstraint(
                        data_sensitivity=DataSensitivity.INTERNAL,
                        require_encryption=True,
                    ),
                )
            ],
        )

        state = await orchestrator.create_run(work_order, normalize=False)
        run_artifacts_dir = tmp_path / "runs" / state.run_id / "artifacts"
        run_artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Execute step (should delegate)
        result = await orchestrator._delegate_to_marketplace(
            run_id=state.run_id,
            step=work_order.steps[0],
            work_order=work_order,
            state=state,
            run_artifacts_dir=run_artifacts_dir,
        )

        # Verify RFQ was created (no ACP bridge = rfq_only fallback)
        assert result.status.value == "committed"
        assert result.head_id == "marketplace-rfq"
        assert "rfq_id" in result.outputs
        assert result.metrics["delegation"] == "rfq_only"

        # Verify output contains RFQ details
        output = json.loads(result.outputs["text"])
        assert output["status"] == "rfq_created"
        assert output["task_type"] == "object_detection"
        assert output["max_budget"] == 500.0
        assert output["deadline_hours"] == 12.0

    @pytest.mark.asyncio
    async def test_orchestrator_delegates_eligible_steps(self, orchestrator):
        """Test orchestrator automatically delegates eligible steps."""
        work_order = WorkOrder(
            run_id="test-run",
            goal="Process images with marketplace",
            steps=[
                StepDef(
                    step_id="marketplace-step",
                    name="Detect objects via marketplace",
                    task_types=["object_detection"],
                    budget=BudgetConstraint(
                        max_cost_per_step=0.05,
                        max_total_cost=500.0,
                    ),
                    privacy=PrivacyConstraint(
                        data_sensitivity=DataSensitivity.PUBLIC,
                    ),
                )
            ],
        )

        # Create and execute run (should delegate to marketplace)
        state = await orchestrator.create_run(work_order, normalize=False)
        state = await orchestrator.execute_run(state.run_id)

        # Verify step was delegated (not run locally)
        result = state.step_results["marketplace-step"]
        assert result.head_id == "marketplace-rfq"
        assert "rfq_id" in result.outputs

    @pytest.mark.asyncio
    async def test_orchestrator_runs_locally_when_no_budget(self, orchestrator):
        """Test orchestrator runs locally when step has no budget."""
        work_order = WorkOrder(
            run_id="test-run",
            goal="Process locally (no budget)",
            steps=[
                StepDef(
                    step_id="local-step",
                    name="Run locally",
                    head_id="mock-llm",
                    task_types=["text_generation"],
                    # No budget = run locally
                )
            ],
        )

        state = await orchestrator.create_run(work_order, normalize=False)
        state = await orchestrator.execute_run(state.run_id)

        # Verify step ran locally (not marketplace)
        result = state.step_results["local-step"]
        assert result.head_id == "mock-llm"  # Local execution
        assert "rfq_id" not in result.outputs

    @pytest.mark.asyncio
    async def test_orchestrator_respects_confidential_data(self, orchestrator):
        """Test CONFIDENTIAL data never delegated even with budget."""
        work_order = WorkOrder(
            run_id="test-run",
            goal="Process confidential data locally",
            steps=[
                StepDef(
                    step_id="confidential-step",
                    name="Process confidential data",
                    head_id="mock-llm",
                    task_types=["data_processing"],
                    budget=BudgetConstraint(
                        max_total_cost=1000.0,  # Has budget but...
                    ),
                    privacy=PrivacyConstraint(
                        data_sensitivity=DataSensitivity.CONFIDENTIAL,  # Cannot delegate!
                    ),
                )
            ],
        )

        state = await orchestrator.create_run(work_order, normalize=False)
        state = await orchestrator.execute_run(state.run_id)

        # Verify step ran locally (CONFIDENTIAL data)
        result = state.step_results["confidential-step"]
        assert result.head_id == "mock-llm"  # Local execution
        assert "rfq_id" not in result.outputs


class TestMarketplaceDelegationIntegration:
    """Test end-to-end marketplace delegation workflow."""

    @pytest.mark.asyncio
    async def test_mixed_local_and_marketplace_execution(self, orchestrator):
        """Test workflow with both local and marketplace steps."""
        work_order = WorkOrder(
            run_id="test-run",
            goal="Mixed execution: local + marketplace",
            steps=[
                # Step 1: Local (no budget)
                StepDef(
                    step_id="step1-local",
                    name="Preprocess data",
                    head_id="mock-llm",
                    task_types=["data_preprocessing"],
                ),
                # Step 2: Marketplace (has budget)
                StepDef(
                    step_id="step2-marketplace",
                    name="Detect objects",
                    task_types=["object_detection"],
                    budget=BudgetConstraint(
                        max_total_cost=500.0,
                    ),
                ),
                # Step 3: Local (confidential)
                StepDef(
                    step_id="step3-local",
                    name="Process results",
                    head_id="mock-llm",
                    task_types=["data_processing"],
                    privacy=PrivacyConstraint(
                        data_sensitivity=DataSensitivity.CONFIDENTIAL,
                    ),
                ),
            ],
        )

        state = await orchestrator.create_run(work_order, normalize=False)
        state = await orchestrator.execute_run(state.run_id)

        # Verify step 1 ran locally
        assert state.step_results["step1-local"].head_id == "mock-llm"

        # Verify step 2 delegated to marketplace
        assert state.step_results["step2-marketplace"].head_id == "marketplace-rfq"
        assert "rfq_id" in state.step_results["step2-marketplace"].outputs

        # Verify step 3 ran locally (CONFIDENTIAL)
        assert state.step_results["step3-local"].head_id == "mock-llm"
