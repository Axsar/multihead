"""Tests for Auto-Decomposition module."""

from __future__ import annotations

import json
import pytest

from multihead.auto_decomposition import (
    AtomicityValidator,
    AutoDecomposer,
    CompletenessValidator,
    ResearchFeatureIntegrator,
    StepDependencyAnalyzer,
)
from multihead.decomposer import DecompositionPlan, TaskNode, parse_plan
from multihead.models import WorkOrder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_leaves():
    """Sample leaf nodes for dependency testing."""
    return [
        TaskNode(
            id="1.1",
            goal="Read config file",
            action_type="read",
            target_files=["config.py"],
        ),
        TaskNode(
            id="1.2",
            goal="Read utils file",
            action_type="read",
            target_files=["utils.py"],
        ),
        TaskNode(
            id="2.1",
            goal="Edit config file",
            action_type="edit",
            target_files=["config.py"],
        ),
        TaskNode(
            id="2.2",
            goal="Create new test",
            action_type="create",
            target_files=["test_feature.py"],
        ),
        TaskNode(
            id="3.1",
            goal="Run tests",
            action_type="test",
        ),
        TaskNode(
            id="3.2",
            goal="Verify results",
            action_type="verify",
        ),
    ]


@pytest.fixture
def parallel_leaves():
    """Leaf nodes that can execute in parallel."""
    return [
        TaskNode(id="1.1", goal="Read file A", action_type="read", target_files=["a.py"]),
        TaskNode(id="1.2", goal="Read file B", action_type="read", target_files=["b.py"]),
        TaskNode(id="1.3", goal="Read file C", action_type="read", target_files=["c.py"]),
    ]


@pytest.fixture
def sequential_leaves():
    """Leaf nodes that must execute sequentially."""
    return [
        TaskNode(id="1", goal="Read file", action_type="read", target_files=["main.py"]),
        TaskNode(id="2", goal="Edit file", action_type="edit", target_files=["main.py"]),
        TaskNode(id="3", goal="Test changes", action_type="test"),
        TaskNode(id="4", goal="Verify correctness", action_type="verify"),
    ]


@pytest.fixture
def sample_plan_json():
    """Sample plan JSON for parsing."""
    return json.dumps({
        "complexity": "moderate",
        "phases": [
            {
                "id": "1",
                "goal": "Understand codebase",
                "children": [
                    {
                        "id": "1.1",
                        "goal": "Read main module",
                        "action_type": "read",
                        "target_files": ["main.py"],
                        "children": [],
                    },
                    {
                        "id": "1.2",
                        "goal": "Read tests",
                        "action_type": "read",
                        "target_files": ["test_main.py"],
                        "children": [],
                    },
                ],
            },
            {
                "id": "2",
                "goal": "Implement feature",
                "children": [
                    {
                        "id": "2.1",
                        "goal": "Add new function",
                        "action_type": "edit",
                        "target_files": ["main.py"],
                        "children": [],
                    },
                    {
                        "id": "2.2",
                        "goal": "Add test case",
                        "action_type": "create",
                        "target_files": ["test_main.py"],
                        "children": [],
                    },
                ],
            },
            {
                "id": "3",
                "goal": "Validate",
                "children": [
                    {
                        "id": "3.1",
                        "goal": "Run tests",
                        "action_type": "test",
                        "children": [],
                    },
                ],
            },
        ],
    })


class FakeHeadManager:
    """Mock HeadManager for testing."""

    def __init__(self, response: str):
        self._response = response
        self.last_prompt = ""

    async def generate(self, head_id: str, prompt: str = "", **kwargs):
        self.last_prompt = prompt
        return {"text": self._response}

    def get_states(self):
        from multihead.models import HeadState
        return {"mock-llm": HeadState.ACTIVE}

    def get_manifest(self, head_id: str):
        return {"kind": "llm"}


# ---------------------------------------------------------------------------
# StepDependencyAnalyzer tests
# ---------------------------------------------------------------------------


class TestStepDependencyAnalyzer:
    """Test dependency inference."""

    def test_parallel_reads_no_deps(self, parallel_leaves):
        """Parallel reads of different files should have no dependencies."""
        analyzer = StepDependencyAnalyzer()
        deps = analyzer.infer_dependencies(parallel_leaves)

        assert deps["1.1"] == []
        assert deps["1.2"] == []
        assert deps["1.3"] == []

    def test_read_then_edit_creates_dep(self):
        """Edit after read of same file should create dependency."""
        leaves = [
            TaskNode(id="1", goal="Read file", action_type="read", target_files=["a.py"]),
            TaskNode(id="2", goal="Edit file", action_type="edit", target_files=["a.py"]),
        ]
        analyzer = StepDependencyAnalyzer()
        deps = analyzer.infer_dependencies(leaves)

        assert deps["1"] == []
        assert deps["2"] == ["1"]  # Edit depends on read

    def test_multiple_reads_then_edit(self):
        """Edit should depend on all prior reads of same file."""
        leaves = [
            TaskNode(id="1", goal="Read A", action_type="read", target_files=["a.py"]),
            TaskNode(id="2", goal="Read A again", action_type="read", target_files=["a.py"]),
            TaskNode(id="3", goal="Edit A", action_type="edit", target_files=["a.py"]),
        ]
        analyzer = StepDependencyAnalyzer()
        deps = analyzer.infer_dependencies(leaves)

        assert "1" in deps["3"]
        assert "2" in deps["3"]

    def test_test_depends_on_edit(self):
        """Test action should depend on most recent edit."""
        leaves = [
            TaskNode(id="1", goal="Edit code", action_type="edit", target_files=["a.py"]),
            TaskNode(id="2", goal="Read other", action_type="read", target_files=["b.py"]),
            TaskNode(id="3", goal="Run tests", action_type="test"),
        ]
        analyzer = StepDependencyAnalyzer()
        deps = analyzer.infer_dependencies(leaves)

        assert "1" in deps["3"]  # Test depends on edit
        assert "2" not in deps["3"]  # But not on unrelated read

    def test_verify_depends_on_test(self):
        """Verify action should depend on most recent test."""
        leaves = [
            TaskNode(id="1", goal="Run tests", action_type="test"),
            TaskNode(id="2", goal="Verify results", action_type="verify"),
        ]
        analyzer = StepDependencyAnalyzer()
        deps = analyzer.infer_dependencies(leaves)

        assert "1" in deps["2"]

    def test_complex_dag_inference(self, sample_leaves):
        """Complex scenario with mixed dependencies."""
        analyzer = StepDependencyAnalyzer()
        deps = analyzer.infer_dependencies(sample_leaves)

        # 1.1 and 1.2 are parallel reads
        assert deps["1.1"] == []
        assert deps["1.2"] == []

        # 2.1 edits config.py, depends on 1.1 (read config.py)
        assert "1.1" in deps["2.1"]

        # 2.2 creates new file, no file deps
        assert deps["2.2"] == []

        # 3.1 tests, depends on edits (2.1, 2.2)
        assert "2.1" in deps["3.1"] or "2.2" in deps["3.1"]

        # 3.2 verifies, depends on test
        assert "3.1" in deps["3.2"]

    def test_artifact_dependency_from_description(self):
        """Infer dependencies from artifact mentions in descriptions."""
        leaves = [
            TaskNode(id="1", goal="Generate config", expected_output="Produces config_dict"),
            TaskNode(id="2", goal="Use config_dict for setup", action_type="edit"),
        ]
        analyzer = StepDependencyAnalyzer()
        deps = analyzer.infer_dependencies(leaves)

        # Step 2 should depend on step 1 (artifact flow)
        assert "1" in deps["2"]


# ---------------------------------------------------------------------------
# AtomicityValidator tests
# ---------------------------------------------------------------------------


class TestAtomicityValidator:
    """Test atomic step validation."""

    def test_single_action_is_atomic(self):
        """Single action step should be atomic."""
        node = TaskNode(
            id="1",
            goal="Read the config file",
            action_type="read",
            target_files=["config.py"],
        )
        is_atomic, reason = AtomicityValidator.is_atomic(node)
        assert is_atomic
        assert reason is None

    def test_multiple_files_not_atomic(self):
        """Step targeting multiple files is not atomic."""
        node = TaskNode(
            id="1",
            goal="Read files",
            action_type="read",
            target_files=["a.py", "b.py"],
        )
        is_atomic, reason = AtomicityValidator.is_atomic(node)
        assert not is_atomic
        assert "2 files" in reason

    def test_and_keyword_not_atomic(self):
        """Step with 'and' keyword is likely not atomic."""
        node = TaskNode(id="1", goal="Read file and edit it", action_type="read")
        is_atomic, reason = AtomicityValidator.is_atomic(node)
        assert not is_atomic
        assert "and" in reason.lower()

    def test_then_keyword_not_atomic(self):
        """Step with 'then' keyword is sequential, not atomic."""
        node = TaskNode(id="1", goal="Edit code then run tests", action_type="edit")
        is_atomic, reason = AtomicityValidator.is_atomic(node)
        assert not is_atomic
        assert "then" in reason.lower()

    def test_multiple_verbs_not_atomic(self):
        """Step with multiple action verbs is not atomic."""
        node = TaskNode(id="1", goal="Read file, edit function, save changes", action_type="read")
        is_atomic, reason = AtomicityValidator.is_atomic(node)
        assert not is_atomic
        assert "multiple actions" in reason.lower()

    def test_read_and_understand_is_atomic(self):
        """Common phrase 'read and understand' is still atomic."""
        # Note: "understand" is not in SINGLE_ACTION_VERBS,
        # so this won't trigger the multi-action check
        # The phrase "read and understand" only finds "read" as a verb
        node = TaskNode(id="1", goal="Read and understand the layout logic", action_type="read")
        is_atomic, reason = AtomicityValidator.is_atomic(node)
        assert is_atomic  # Only finds "read" verb, so it's atomic

    def test_read_test_cases_is_atomic(self):
        """Reading test cases is a single atomic action."""
        node = TaskNode(id="1", goal="Read test cases", action_type="read")
        is_atomic, reason = AtomicityValidator.is_atomic(node)
        assert is_atomic  # "read test" is an allowed combination

    def test_long_goal_not_atomic(self):
        """Very long goal description likely indicates composite action."""
        long_goal = "a" * 150  # > 100 chars
        node = TaskNode(id="1", goal=long_goal, action_type="edit")
        is_atomic, reason = AtomicityValidator.is_atomic(node)
        assert not is_atomic
        assert "too long" in reason.lower()

    def test_validate_plan(self, sample_plan_json):
        """Validate entire plan for atomicity."""
        plan = parse_plan(sample_plan_json, "Implement feature")
        violations = AtomicityValidator.validate_plan(plan)
        # Sample plan should be atomic
        assert len(violations) == 0

    def test_validate_plan_with_violations(self):
        """Detect non-atomic steps in plan."""
        plan_json = json.dumps({
            "complexity": "simple",
            "phases": [{
                "id": "1",
                "goal": "Phase",
                "children": [
                    {
                        "id": "1.1", "goal": "Read A and B",
                        "target_files": ["a.py", "b.py"],
                        "children": [],
                    },
                    {"id": "1.2", "goal": "Edit then test", "action_type": "edit", "children": []},
                ],
            }],
        })
        plan = parse_plan(plan_json, "Test")
        violations = AtomicityValidator.validate_plan(plan)
        assert len(violations) == 2
        assert any("1.1" in v[0] for v in violations)
        assert any("1.2" in v[0] for v in violations)


# ---------------------------------------------------------------------------
# CompletenessValidator tests
# ---------------------------------------------------------------------------


class TestCompletenessValidator:
    """Test completeness validation."""

    def test_complete_plan_passes(self, sample_plan_json):
        """Well-structured plan should pass validation."""
        plan = parse_plan(sample_plan_json, "Implement feature")
        is_complete, issues = CompletenessValidator.validate("Implement feature", plan)
        # May have minor issues, but should be mostly complete
        assert len(issues) <= 1  # Allow minor keyword issues

    def test_empty_plan_fails(self):
        """Empty plan should fail validation."""
        plan = DecompositionPlan(goal="Something", phases=[])
        is_complete, issues = CompletenessValidator.validate("Do something", plan)
        assert not is_complete
        assert any("no steps" in issue.lower() for issue in issues)

    def test_missing_implementation_phase(self):
        """Plan without implementation phase should be flagged."""
        plan_json = json.dumps({
            "complexity": "moderate",
            "phases": [{
                "id": "1",
                "goal": "Explore",
                "children": [
                    {"id": "1.1", "goal": "Read files", "action_type": "read", "children": []},
                ],
            }],
        })
        plan = parse_plan(plan_json, "Add feature")
        is_complete, issues = CompletenessValidator.validate("Add feature", plan)
        assert any("implementation" in issue.lower() for issue in issues)

    def test_missing_verification_phase(self):
        """Plan without verification should be flagged."""
        plan_json = json.dumps({
            "complexity": "moderate",
            "phases": [{
                "id": "1",
                "goal": "Implement",
                "children": [
                    {"id": "1.1", "goal": "Edit code", "action_type": "edit", "children": []},
                ],
            }],
        })
        plan = parse_plan(plan_json, "Fix bug")
        is_complete, issues = CompletenessValidator.validate("Fix bug", plan)
        assert any("verification" in issue.lower() for issue in issues)

    def test_single_step_for_complex_goal(self):
        """Single step for complex goal should be flagged."""
        plan = DecompositionPlan(
            goal="Refactor entire system",
            complexity="complex",
            phases=[TaskNode(id="1", goal="Phase", children=[
                TaskNode(id="1.1", goal="Do everything", action_type="edit"),
            ])],
        )
        is_complete, issues = CompletenessValidator.validate("Refactor entire system", plan)
        assert any("1 step" in issue.lower() for issue in issues)

    def test_missing_goal_keywords(self):
        """Plan missing key goal terms should be flagged."""
        plan_json = json.dumps({
            "complexity": "simple",
            "phases": [{
                "id": "1",
                "goal": "Phase",
                "children": [
                    {
                        "id": "1.1",
                        "goal": "Do something unrelated",
                        "action_type": "edit",
                        "children": [],
                    },
                ],
            }],
        })
        plan = parse_plan(plan_json, "Fix authentication bug")
        is_complete, issues = CompletenessValidator.validate("Fix authentication bug", plan)
        # Should flag missing "authentication" or "bug" in plan
        assert len(issues) > 0


# ---------------------------------------------------------------------------
# ResearchFeatureIntegrator tests
# ---------------------------------------------------------------------------


class TestResearchFeatureIntegrator:
    """Test research feature integration."""

    def test_tot_enabled_for_exploratory_steps(self):
        """ToT should be enabled for explore/investigate steps."""
        node = TaskNode(id="1", goal="Explore codebase", action_type="explore")
        config = ResearchFeatureIntegrator._get_feature_config(node)

        assert config.get("use_tot") is True
        assert config.get("tot_strategy") == "bfs"
        assert config.get("tot_alternatives") == 3

    def test_prm_enabled_for_implementation_steps(self):
        """PRM should be enabled for edit/create steps."""
        node = TaskNode(id="1", goal="Edit function", action_type="edit")
        config = ResearchFeatureIntegrator._get_feature_config(node)

        assert config.get("use_prm") is True
        assert config.get("prm_scorer") == "rubric"
        assert config.get("prm_threshold") == 0.7

    def test_reflection_enabled_for_verification_steps(self):
        """Reflection should be enabled for test/verify steps."""
        node = TaskNode(id="1", goal="Run tests", action_type="test")
        config = ResearchFeatureIntegrator._get_feature_config(node)

        assert config.get("enable_reflection") is True
        assert config.get("max_reflection_attempts") == 3

    def test_no_features_for_read_steps(self):
        """Read steps shouldn't get research features by default."""
        node = TaskNode(id="1", goal="Read file", action_type="read")
        config = ResearchFeatureIntegrator._get_feature_config(node)

        assert "use_tot" not in config
        assert "use_prm" not in config
        assert "enable_reflection" not in config

    def test_integrate_plan(self, sample_plan_json):
        """Test integration across entire plan."""
        plan = parse_plan(sample_plan_json, "Implement feature")
        configs = ResearchFeatureIntegrator.integrate(plan, enable_auto_features=True)

        assert isinstance(configs, dict)
        assert len(configs) > 0

        # Check that implementation steps got PRM configs
        impl_step_ids = [
            l.id for l in plan.all_leaves()
            if l.action_type in ("edit", "create")
        ]
        for step_id in impl_step_ids:
            if step_id in configs:
                assert configs[step_id].get("use_prm") is True

        # Check that test steps got reflection configs
        test_step_ids = [
            l.id for l in plan.all_leaves()
            if l.action_type == "test"
        ]
        for step_id in test_step_ids:
            if step_id in configs:
                assert configs[step_id].get("enable_reflection") is True

    def test_disabled_integration(self, sample_plan_json):
        """When disabled, no features should be added."""
        plan = parse_plan(sample_plan_json, "Implement feature")
        configs = ResearchFeatureIntegrator.integrate(plan, enable_auto_features=False)

        # Should return empty dict
        assert configs == {}


# ---------------------------------------------------------------------------
# AutoDecomposer tests
# ---------------------------------------------------------------------------


class TestAutoDecomposer:
    """Test AutoDecomposer main class."""

    @pytest.mark.asyncio
    async def test_decompose_basic(self, sample_plan_json):
        """Basic decomposition should work like TaskDecomposer."""
        hm = FakeHeadManager(sample_plan_json)
        decomposer = AutoDecomposer(hm)

        plan = await decomposer.decompose(
            "Implement feature",
            head_id="mock-llm",
            auto_validate=False,
            enable_research_features=False,
        )

        assert plan.goal == "Implement feature"
        assert plan.total_steps == 5

    @pytest.mark.asyncio
    async def test_decompose_with_validation(self, sample_plan_json):
        """Decomposition with validation enabled."""
        hm = FakeHeadManager(sample_plan_json)
        decomposer = AutoDecomposer(hm)

        plan = await decomposer.decompose(
            "Implement feature",
            head_id="mock-llm",
            auto_validate=True,  # Enable validation
        )

        # Should complete without raising (plan is well-formed)
        assert plan.total_steps > 0

    @pytest.mark.asyncio
    async def test_decompose_with_research_features(self, sample_plan_json):
        """Decomposition with research features enabled."""
        hm = FakeHeadManager(sample_plan_json)
        decomposer = AutoDecomposer(hm)

        plan = await decomposer.decompose(
            "Implement feature",
            head_id="mock-llm",
            enable_research_features=True,
        )

        # Check that feature configs were stored
        assert hasattr(plan, "_feature_configs")
        assert len(plan._feature_configs) > 0

    @pytest.mark.asyncio
    async def test_to_work_order_with_dag(self, sample_plan_json):
        """Convert plan to WorkOrder with DAG dependencies."""
        hm = FakeHeadManager(sample_plan_json)
        decomposer = AutoDecomposer(hm)

        plan = await decomposer.decompose("Implement feature", head_id="mock-llm")
        work_order = decomposer.to_work_order_with_dag(plan)

        assert isinstance(work_order, WorkOrder)
        assert work_order.goal == "Implement feature"
        assert len(work_order.steps) == 5

        # Check dependencies are inferred (not just sequential chain)
        # First two reads should have no deps (parallel)
        assert work_order.steps[0].depends_on == []
        assert work_order.steps[1].depends_on == []

        # Edits should depend on reads
        step_21 = next(s for s in work_order.steps if s.step_id == "2.1")
        assert len(step_21.depends_on) > 0

        # Test should depend on implementation
        step_31 = next(s for s in work_order.steps if s.step_id == "3.1")
        assert len(step_31.depends_on) > 0

    @pytest.mark.asyncio
    async def test_parallel_steps_have_no_deps(self):
        """Parallel-safe steps should have no dependencies."""
        parallel_plan_json = json.dumps({
            "complexity": "simple",
            "phases": [{
                "id": "1",
                "goal": "Read all files",
                "children": [
                    {
                        "id": "1.1", "goal": "Read A",
                        "action_type": "read",
                        "target_files": ["a.py"],
                        "children": [],
                    },
                    {
                        "id": "1.2", "goal": "Read B",
                        "action_type": "read",
                        "target_files": ["b.py"],
                        "children": [],
                    },
                    {
                        "id": "1.3", "goal": "Read C",
                        "action_type": "read",
                        "target_files": ["c.py"],
                        "children": [],
                    },
                ],
            }],
        })
        hm = FakeHeadManager(parallel_plan_json)
        decomposer = AutoDecomposer(hm)

        plan = await decomposer.decompose("Read files", head_id="mock-llm")
        work_order = decomposer.to_work_order_with_dag(plan)

        # All three reads should be parallel (no dependencies)
        for step in work_order.steps:
            assert step.depends_on == []

    @pytest.mark.asyncio
    async def test_sequential_steps_have_deps(self, sequential_leaves):
        """Sequential steps should have correct dependency chain."""
        seq_plan_json = json.dumps({
            "complexity": "simple",
            "phases": [{
                "id": "1",
                "goal": "Sequential workflow",
                "children": [
                    {
                        "id": "1", "goal": "Read file",
                        "action_type": "read",
                        "target_files": ["main.py"],
                        "children": [],
                    },
                    {
                        "id": "2", "goal": "Edit file",
                        "action_type": "edit",
                        "target_files": ["main.py"],
                        "children": [],
                    },
                    {
                        "id": "3", "goal": "Test changes",
                        "action_type": "test",
                        "children": [],
                    },
                    {
                        "id": "4", "goal": "Verify correctness",
                        "action_type": "verify",
                        "children": [],
                    },
                ],
            }],
        })
        hm = FakeHeadManager(seq_plan_json)
        decomposer = AutoDecomposer(hm)

        plan = await decomposer.decompose("Fix bug", head_id="mock-llm")
        work_order = decomposer.to_work_order_with_dag(plan)

        # Read has no deps
        assert work_order.steps[0].depends_on == []
        # Edit depends on read
        assert work_order.steps[1].depends_on == ["1"]
        # Test depends on edit
        assert work_order.steps[2].depends_on == ["2"]
        # Verify depends on test
        assert work_order.steps[3].depends_on == ["3"]

    @pytest.mark.asyncio
    async def test_refine_delegates_to_base(self):
        """Refine should delegate to base TaskDecomposer."""
        refine_response = json.dumps([
            {"id": "1.1", "goal": "Substep A", "action_type": "read", "children": []},
            {"id": "1.2", "goal": "Substep B", "action_type": "edit", "children": []},
        ])
        hm = FakeHeadManager(refine_response)
        decomposer = AutoDecomposer(hm)

        node = TaskNode(id="1", goal="Parent step", action_type="explore")
        children = await decomposer.refine(node, head_id="mock-llm")

        assert len(children) == 2
        assert children[0].id == "1.1"
        assert children[1].id == "1.2"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Test full decomposition workflow."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, sample_plan_json):
        """Test complete auto-decomposition workflow."""
        hm = FakeHeadManager(sample_plan_json)
        decomposer = AutoDecomposer(hm)

        # 1. Decompose with all features enabled
        plan = await decomposer.decompose(
            goal="Implement user authentication",
            context="Add login/logout functionality",
            head_id="mock-llm",
            auto_validate=True,
            enable_research_features=True,
        )

        # 2. Verify plan structure
        assert plan.total_steps > 0
        assert plan.complexity in ("simple", "moderate", "complex")

        # 3. Convert to WorkOrder with DAG
        work_order = decomposer.to_work_order_with_dag(plan)

        # 4. Verify WorkOrder
        assert isinstance(work_order, WorkOrder)
        assert work_order.goal == "Implement user authentication"
        assert len(work_order.steps) > 0

        # 5. Verify some steps have research features in extra
        steps_with_research_features = sum(
            1 for s in work_order.steps
            if s.extra and (
                "use_tot" in s.extra
                or "use_prm" in s.extra
                or "enable_reflection" in s.extra
            )
        )
        assert steps_with_research_features > 0

        # 6. Verify dependencies are not just sequential
        # (at least one step should have 0 or >1 dependencies, not strictly sequential)
        parallel_steps = sum(1 for s in work_order.steps if len(s.depends_on) == 0)
        assert parallel_steps >= 1  # At least the first step
