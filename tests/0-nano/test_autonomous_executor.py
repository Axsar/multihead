"""Tests for autonomous_executor.py — execution strategies, DAG layers, reflection."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from multihead.autonomous_executor import (
    ROLE_PROMPTS,
    ROLE_TOOL_MAP,
    DEFAULT_TOOLS,
    AutonomousExecutor,
    ClaudeSessionStrategy,
    ExecutionReport,
    ExecutionStrategy,
    LocalLLMStrategy,
    StepContext,
    StepExecutionResult,
    _flatten_leaves,
    _infer_layer_dependencies,
    _normalize_claude_output,
    _summarize_plan,
    _topological_layers,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_PLAN = {
    "goal": "Add logging to the API",
    "complexity": "moderate",
    "phases": [
        {
            "id": "1",
            "goal": "Explore existing code",
            "action_type": "explore",
            "target_files": ["src/api.py"],
            "children": [
                {
                    "id": "1.1",
                    "goal": "Read API handler",
                    "action_type": "read",
                    "target_files": ["src/api.py"],
                },
                {
                    "id": "1.2",
                    "goal": "Read config",
                    "action_type": "read",
                    "target_files": ["config.yaml"],
                },
            ],
        },
        {
            "id": "2",
            "goal": "Implement changes",
            "action_type": "implement",
            "target_files": ["src/api.py"],
            "children": [
                {
                    "id": "2.1",
                    "goal": "Add logging imports and setup",
                    "action_type": "edit",
                    "target_files": ["src/api.py"],
                },
            ],
        },
        {
            "id": "3",
            "goal": "Test and verify",
            "action_type": "test",
            "target_files": [],
            "children": [
                {
                    "id": "3.1",
                    "goal": "Run test suite",
                    "action_type": "test",
                    "target_files": ["tests/test_api.py"],
                },
                {
                    "id": "3.2",
                    "goal": "Review changes",
                    "action_type": "review",
                    "target_files": ["src/api.py"],
                },
            ],
        },
        {
            "id": "4",
            "goal": "Final verification",
            "action_type": "verify",
            "target_files": [],
            "children": [
                {
                    "id": "4.1",
                    "goal": "Verify acceptance criteria",
                    "action_type": "verify",
                    "target_files": [],
                },
            ],
        },
    ],
}


def make_result(step_id="1", success=True, output="done", **kw) -> StepExecutionResult:
    return StepExecutionResult(
        step_id=step_id,
        step_goal="test goal",
        action_type=kw.get("action_type", "explore"),
        success=success,
        output=output,
        **{k: v for k, v in kw.items() if k != "action_type"},
    )


# ===========================================================================
# StepContext tests
# ===========================================================================


class TestStepContext:
    def test_build_prompt_basic(self):
        ctx = StepContext(goal="Fix the bug", plan_summary="Phase 1: explore")
        prompt = ctx.build_prompt("1.1", "Read main.py", "read", ["main.py"], [])
        assert "Fix the bug" in prompt
        assert "Read main.py" in prompt
        assert "main.py" in prompt
        assert "READER" in prompt

    def test_build_prompt_with_dependencies(self):
        ctx = StepContext(
            goal="Fix bug",
            plan_summary="",
            step_outputs={"1.1": "Found the issue in line 42"},
        )
        prompt = ctx.build_prompt("2.1", "Fix it", "edit", [], ["1.1"])
        assert "Found the issue in line 42" in prompt
        assert "Step 1.1 output" in prompt

    def test_dependency_context_truncation(self):
        ctx = StepContext(
            goal="Test",
            plan_summary="",
            step_outputs={"1.1": "x" * 5000},
            max_context_chars=100,
        )
        dep_ctx = ctx._dependency_context(["1.1"])
        assert len(dep_ctx) < 5000
        assert "truncated" in dep_ctx

    def test_dependency_context_empty_deps(self):
        ctx = StepContext(goal="Test", plan_summary="")
        assert ctx._dependency_context([]) == ""

    def test_dependency_context_missing_step(self):
        ctx = StepContext(
            goal="Test", plan_summary="",
            step_outputs={"1.1": "some output"},
        )
        # Requesting dep "2.1" which doesn't exist in outputs
        dep_ctx = ctx._dependency_context(["2.1"])
        assert dep_ctx == ""

    def test_build_prompt_includes_knowledge(self):
        ctx = StepContext(
            goal="Upgrade",
            plan_summary="",
            knowledge_claims=["API uses REST", "Auth is JWT-based"],
        )
        prompt = ctx.build_prompt("1", "Check auth", "explore", [], [])
        assert "API uses REST" in prompt
        assert "Auth is JWT-based" in prompt


# ===========================================================================
# ROLE_PROMPTS tests
# ===========================================================================


class TestRolePrompts:
    def test_all_action_types_have_prompts(self):
        for action_type in ROLE_TOOL_MAP:
            assert action_type in ROLE_PROMPTS, f"Missing prompt for {action_type}"

    def test_explore_prompt_is_read_only(self):
        assert "DO NOT modify" in ROLE_PROMPTS["explore"]

    def test_review_prompt_is_read_only(self):
        assert "DO NOT modify" in ROLE_PROMPTS["review"]

    def test_verify_prompt_is_read_only(self):
        assert "DO NOT modify" in ROLE_PROMPTS["verify"]

    def test_implement_prompt_allows_writes(self):
        prompt = ROLE_PROMPTS["implement"].lower()
        assert "write" in prompt or "modify" in prompt

    def test_test_prompt_mentions_tests(self):
        assert "test" in ROLE_PROMPTS["test"].lower()

    def test_tool_map_explore_no_edit(self):
        tools = ROLE_TOOL_MAP["explore"]
        assert "Edit" not in tools
        assert "Write" not in tools

    def test_tool_map_implement_has_edit(self):
        tools = ROLE_TOOL_MAP["implement"]
        assert "Edit" in tools
        assert "Write" in tools

    def test_default_tools_read_only(self):
        assert "Edit" not in DEFAULT_TOOLS
        assert "Read" in DEFAULT_TOOLS


# ===========================================================================
# LocalLLMStrategy tests
# ===========================================================================


class TestLocalLLMStrategy:
    @pytest.mark.asyncio
    async def test_returns_plan_only_output(self):
        strategy = LocalLLMStrategy()
        result = await strategy.execute_step("1", "Do something", "explore")
        assert result.success
        assert "[plan-only]" in result.output

    @pytest.mark.asyncio
    async def test_quality_always_perfect(self):
        strategy = LocalLLMStrategy()
        result = await strategy.execute_step("1", "Do something", "explore")
        score, feedback = strategy.check_quality(result)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_preserves_step_metadata(self):
        strategy = LocalLLMStrategy()
        result = await strategy.execute_step("step-42", "My task", "implement")
        assert result.step_id == "step-42"
        assert result.action_type == "implement"


# ===========================================================================
# ClaudeSessionStrategy tests
# ===========================================================================


class TestClaudeSessionStrategy:
    def test_init_defaults(self):
        s = ClaudeSessionStrategy()
        assert s.model == "claude-sonnet-4-6"
        assert s.max_budget_usd == 1.0

    def test_init_custom(self):
        s = ClaudeSessionStrategy(model="claude-haiku-4-5-20251001", max_budget_usd=0.5)
        assert s.model == "claude-haiku-4-5-20251001"
        assert s.max_budget_usd == 0.5

    @pytest.mark.asyncio
    async def test_no_claude_binary(self):
        s = ClaudeSessionStrategy()
        with patch("shutil.which", return_value=None):
            result = await s.execute_step("1", "test", "explore")
        assert not result.success
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_subprocess_args(self):
        s = ClaudeSessionStrategy(model="claude-sonnet-4-6", max_budget_usd=2.0)
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = json.dumps([
            {"type": "system", "subtype": "init", "session_id": "sess-123"},
            {"type": "result", "result": "All done", "cost_usd": 0.5, "is_error": False},
        ])
        mock_proc.stderr = ""

        with patch("shutil.which", return_value="/usr/bin/claude"), \
             patch("subprocess.run", return_value=mock_proc) as mock_run:
            result = await s.execute_step("1", "explore code", "explore")

            # Verify subprocess.run was called
            call_args = mock_run.call_args
            cmd = call_args[0][0]
            assert cmd[0] == "/usr/bin/claude"
            assert "-p" in cmd
            assert "--model" in cmd
            assert "claude-sonnet-4-6" in cmd
            assert "--max-budget-usd" in cmd
            assert "2.0" in cmd
            assert "--allowedTools" in cmd
            # explore role should get read-only tools
            tools_idx = cmd.index("--allowedTools") + 1
            assert "Edit" not in cmd[tools_idx]

        assert result.success
        assert result.output == "All done"
        assert result.cost_usd == 0.5
        assert result.session_id == "sess-123"

    @pytest.mark.asyncio
    async def test_subprocess_error_exit(self):
        s = ClaudeSessionStrategy()
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = ""
        mock_proc.stderr = "Something went wrong"

        with patch("shutil.which", return_value="/usr/bin/claude"), \
             patch("subprocess.run", return_value=mock_proc):
            result = await s.execute_step("1", "test", "explore")

        assert not result.success
        assert "Exit 1" in result.error

    @pytest.mark.asyncio
    async def test_subprocess_timeout(self):
        import subprocess as sp

        s = ClaudeSessionStrategy(subprocess_timeout=5)

        with patch("shutil.which", return_value="/usr/bin/claude"), \
             patch("subprocess.run", side_effect=sp.TimeoutExpired(["claude"], 5)):
            result = await s.execute_step("1", "test", "explore", timeout=5)

        assert not result.success
        assert "timed out" in result.error

    def test_quality_check_good_output(self):
        s = ClaudeSessionStrategy()
        result = make_result(
            output="Successfully edited the file and added logging. All tests pass."
        )
        score, _ = s.check_quality(result)
        assert score >= 0.8

    def test_quality_check_empty_output(self):
        s = ClaudeSessionStrategy()
        result = make_result(output="")
        score, feedback = s.check_quality(result)
        assert score < 0.6
        assert "short" in feedback.lower() or "empty" in feedback.lower()

    def test_quality_check_error_in_output(self):
        s = ClaudeSessionStrategy()
        result = make_result(output="An error occurred while processing the request")
        score, feedback = s.check_quality(result)
        assert score < 1.0
        assert "error" in feedback.lower()

    def test_quality_check_test_without_pass(self):
        s = ClaudeSessionStrategy()
        result = make_result(
            output="Ran the test suite, found some issues",
            action_type="test",
        )
        score, _ = s.check_quality(result)
        assert score < 1.0

    def test_quality_check_test_with_pass(self):
        s = ClaudeSessionStrategy()
        result = make_result(
            output="All 15 tests pass successfully",
            action_type="test",
        )
        score, _ = s.check_quality(result)
        assert score >= 0.8


# ===========================================================================
# Plan flattening tests
# ===========================================================================


class TestFlattenLeaves:
    def test_extracts_all_leaves(self):
        leaves = _flatten_leaves(SAMPLE_PLAN)
        ids = [l["id"] for l in leaves]
        assert ids == ["1.1", "1.2", "2.1", "3.1", "3.2", "4.1"]

    def test_empty_plan(self):
        assert _flatten_leaves({"phases": []}) == []

    def test_flat_phases_are_leaves(self):
        plan = {"phases": [
            {"id": "1", "goal": "Do it", "action_type": "edit", "target_files": []},
        ]}
        leaves = _flatten_leaves(plan)
        assert len(leaves) == 1
        assert leaves[0]["id"] == "1"


# ===========================================================================
# Dependency inference tests
# ===========================================================================


class TestDependencyInference:
    def test_action_ordering(self):
        leaves = _flatten_leaves(SAMPLE_PLAN)
        deps = _infer_layer_dependencies(leaves)
        # test (3.1) should depend on edit (2.1)
        assert "2.1" in deps["3.1"]
        # verify (4.1) should depend on test (3.1)
        assert "3.1" in deps["4.1"]
        # review (3.2) should depend on edit (2.1)
        assert "2.1" in deps["3.2"]

    def test_read_steps_independent(self):
        leaves = _flatten_leaves(SAMPLE_PLAN)
        deps = _infer_layer_dependencies(leaves)
        # read steps (1.1, 1.2) should have no deps on each other
        assert "1.2" not in deps["1.1"]
        assert "1.1" not in deps["1.2"]

    def test_file_dependency(self):
        leaves = [
            {"id": "a", "action_type": "read", "target_files": ["x.py"], "goal": ""},
            {"id": "b", "action_type": "edit", "target_files": ["x.py"], "goal": ""},
        ]
        deps = _infer_layer_dependencies(leaves)
        # editor depends on reader of same file
        assert "a" in deps["b"]


# ===========================================================================
# Topological layers tests
# ===========================================================================


class TestTopologicalLayers:
    def test_basic_layers(self):
        leaves = _flatten_leaves(SAMPLE_PLAN)
        deps = _infer_layer_dependencies(leaves)
        layers = _topological_layers(leaves, deps)
        # Should have at least 2 layers
        assert len(layers) >= 2
        # First layer should be read steps (no deps)
        assert "1.1" in layers[0] or "1.2" in layers[0]

    def test_no_deps_single_layer(self):
        leaves = [
            {"id": "a", "action_type": "read", "target_files": [], "goal": ""},
            {"id": "b", "action_type": "read", "target_files": [], "goal": ""},
        ]
        deps = _infer_layer_dependencies(leaves)
        layers = _topological_layers(leaves, deps)
        assert len(layers) == 1
        assert set(layers[0]) == {"a", "b"}

    def test_linear_chain(self):
        leaves = [
            {"id": "a", "action_type": "read", "target_files": ["x.py"], "goal": ""},
            {"id": "b", "action_type": "edit", "target_files": ["x.py"], "goal": ""},
            {"id": "c", "action_type": "test", "target_files": [], "goal": ""},
        ]
        deps = _infer_layer_dependencies(leaves)
        layers = _topological_layers(leaves, deps)
        assert len(layers) >= 2
        # read before edit
        a_layer = next(i for i, l in enumerate(layers) if "a" in l)
        b_layer = next(i for i, l in enumerate(layers) if "b" in l)
        c_layer = next(i for i, l in enumerate(layers) if "c" in l)
        assert a_layer < b_layer
        assert b_layer < c_layer

    def test_cycle_handling(self):
        """Cycles should be forced into a single layer, not crash."""
        leaves = [
            {"id": "a", "action_type": "read", "target_files": [], "goal": ""},
            {"id": "b", "action_type": "read", "target_files": [], "goal": ""},
        ]
        # Force a cycle: a depends on b, b depends on a
        deps = {"a": ["b"], "b": ["a"]}
        layers = _topological_layers(leaves, deps)
        # Should not crash, all IDs present
        all_ids = {sid for layer in layers for sid in layer}
        assert all_ids == {"a", "b"}

    def test_empty(self):
        assert _topological_layers([], {}) == []


# ===========================================================================
# AutonomousExecutor tests
# ===========================================================================


class TestAutonomousExecutor:
    @pytest.mark.asyncio
    async def test_execute_empty_plan(self):
        strategy = LocalLLMStrategy()
        executor = AutonomousExecutor(strategy=strategy)
        report = await executor.execute("do stuff", {"phases": []})
        assert report.total_steps == 0
        assert report.success

    @pytest.mark.asyncio
    async def test_execute_all_succeed(self):
        strategy = AsyncMock(spec=ExecutionStrategy)
        strategy.execute_step = AsyncMock(
            side_effect=lambda **kw: make_result(
                step_id=kw["step_id"], success=True,
                output="Done successfully with details",
                action_type=kw["action_type"],
            )
        )
        strategy.check_quality = MagicMock(return_value=(0.9, "Good"))

        executor = AutonomousExecutor(strategy=strategy, quality_threshold=0.5)
        report = await executor.execute("Add logging", SAMPLE_PLAN)

        assert report.total_steps == 6
        assert report.completed_steps == 6
        assert report.failed_steps == 0
        assert report.success

    @pytest.mark.asyncio
    async def test_execute_step_failure(self):
        call_count = 0

        async def _execute(**kw):
            nonlocal call_count
            call_count += 1
            if kw["step_id"] == "2.1":
                return make_result(
                    step_id="2.1", success=False, output="",
                    error="Compilation failed", action_type="edit",
                )
            return make_result(
                step_id=kw["step_id"], success=True,
                output="Done with plenty of details here",
                action_type=kw["action_type"],
            )

        strategy = AsyncMock(spec=ExecutionStrategy)
        strategy.execute_step = AsyncMock(side_effect=_execute)
        strategy.check_quality = MagicMock(return_value=(0.9, "Good"))

        executor = AutonomousExecutor(
            strategy=strategy, quality_threshold=0.5, max_retries=1,
        )
        report = await executor.execute("Add logging", SAMPLE_PLAN)

        assert report.failed_steps >= 1
        assert not report.success

    @pytest.mark.asyncio
    async def test_context_chaining(self):
        """Output from layer N should appear in layer N+1 prompts."""
        prompts_seen = []

        async def _capture_execute(**kw):
            prompts_seen.append(kw["prompt"])
            return make_result(
                step_id=kw["step_id"], success=True,
                output=f"Output from {kw['step_id']}",
                action_type=kw["action_type"],
            )

        strategy = AsyncMock(spec=ExecutionStrategy)
        strategy.execute_step = AsyncMock(side_effect=_capture_execute)
        strategy.check_quality = MagicMock(return_value=(0.9, "Good"))

        executor = AutonomousExecutor(strategy=strategy, quality_threshold=0.5)
        await executor.execute("Test", SAMPLE_PLAN)

        # Later steps should see outputs from earlier steps in their prompts
        # 2.1 (edit) depends on 1.1/1.2 (read), so its prompt should have their output
        # Find the prompt for step 2.1 — it has EDITOR role prompt
        edit_prompts = [p for p in prompts_seen if "EDITOR" in p]
        assert len(edit_prompts) >= 1
        # The dependency context should include output from read steps
        # (1.1 reads src/api.py, 2.1 edits src/api.py => file dep)
        has_dep_output = any(
            "Output from 1.1" in p or "Output from 1.2" in p
            for p in edit_prompts
        )
        assert has_dep_output, (
            f"Edit step prompt should contain read step output. "
            f"Edit prompts: {[p[:200] for p in edit_prompts]}"
        )

    @pytest.mark.asyncio
    async def test_reflection_retry(self):
        """Steps below quality threshold should be retried."""
        attempt_counts: dict[str, int] = {}

        async def _execute(**kw):
            sid = kw["step_id"]
            attempt_counts[sid] = attempt_counts.get(sid, 0) + 1
            if sid == "2.1" and attempt_counts[sid] < 3:
                return make_result(
                    step_id=sid, success=True, output="Bad output",
                    action_type=kw["action_type"],
                )
            return make_result(
                step_id=sid, success=True,
                output="Good output with enough detail here",
                action_type=kw["action_type"],
            )

        def _quality(result):
            if result.step_id == "2.1" and result.output == "Bad output":
                return (0.2, "Output is garbage")
            return (0.9, "Good")

        strategy = AsyncMock(spec=ExecutionStrategy)
        strategy.execute_step = AsyncMock(side_effect=_execute)
        strategy.check_quality = MagicMock(side_effect=_quality)

        executor = AutonomousExecutor(
            strategy=strategy, quality_threshold=0.5, max_retries=3,
        )
        report = await executor.execute("Test", SAMPLE_PLAN)

        # Step 2.1 should have been retried
        assert attempt_counts.get("2.1", 0) == 3

    @pytest.mark.asyncio
    async def test_reflection_feedback_in_prompt(self):
        """Reflection feedback should appear in retry prompts."""
        prompts_for_step = []

        async def _execute(**kw):
            if kw["step_id"] == "1.1":
                prompts_for_step.append(kw["prompt"])
                return make_result(
                    step_id="1.1", success=True, output="meh",
                    action_type="read",
                )
            return make_result(
                step_id=kw["step_id"], success=True,
                output="Good output here with details",
                action_type=kw["action_type"],
            )

        call_count = [0]

        def _quality(result):
            if result.step_id == "1.1":
                call_count[0] += 1
                if call_count[0] < 2:
                    return (0.3, "Output too vague, needs specific findings")
                return (0.9, "Good")
            return (0.9, "Good")

        strategy = AsyncMock(spec=ExecutionStrategy)
        strategy.execute_step = AsyncMock(side_effect=_execute)
        strategy.check_quality = MagicMock(side_effect=_quality)

        executor = AutonomousExecutor(
            strategy=strategy, quality_threshold=0.5, max_retries=3,
        )
        await executor.execute("Test", SAMPLE_PLAN)

        # Second prompt for step 1.1 should include reflection feedback
        assert len(prompts_for_step) >= 2
        assert "Previous Attempt" in prompts_for_step[1]
        assert "too vague" in prompts_for_step[1]

    @pytest.mark.asyncio
    async def test_posts_result_to_knowledge(self):
        strategy = LocalLLMStrategy()
        mock_ks = MagicMock()

        executor = AutonomousExecutor(
            strategy=strategy,
            knowledge_store=mock_ks,
            agent_id="test-agent",
        )
        report = await executor.execute(
            "Test", SAMPLE_PLAN,
            request_id="req-123", proposal_id="prop-456",
        )

        # Should have called insert_claim
        assert mock_ks.insert_claim.called
        claim = mock_ks.insert_claim.call_args[0][0]
        assert "EXECUTION RESULT" in claim.statement
        assert "test-agent" in claim.statement
        assert "req-123" in claim.related_claim_ids

    @pytest.mark.asyncio
    async def test_no_post_without_request_id(self):
        strategy = LocalLLMStrategy()
        mock_ks = MagicMock()

        executor = AutonomousExecutor(strategy=strategy, knowledge_store=mock_ks)
        await executor.execute("Test", SAMPLE_PLAN)
        assert not mock_ks.insert_claim.called

    @pytest.mark.asyncio
    async def test_layers_in_report(self):
        strategy = LocalLLMStrategy()
        executor = AutonomousExecutor(strategy=strategy)
        report = await executor.execute("Test", SAMPLE_PLAN)
        assert len(report.layers) >= 2


# ===========================================================================
# ExecutionReport tests
# ===========================================================================


class TestExecutionReport:
    def test_success_when_all_complete(self):
        report = ExecutionReport(
            goal="Test", strategy="Local", total_steps=3,
            completed_steps=3, failed_steps=0, skipped_steps=0,
            total_cost_usd=0.5, total_duration_secs=10.0,
        )
        assert report.success

    def test_failure_when_steps_failed(self):
        report = ExecutionReport(
            goal="Test", strategy="Local", total_steps=3,
            completed_steps=2, failed_steps=1, skipped_steps=0,
            total_cost_usd=0.5, total_duration_secs=10.0,
        )
        assert not report.success

    def test_summary_text(self):
        report = ExecutionReport(
            goal="Test", strategy="Local", total_steps=5,
            completed_steps=5, failed_steps=0, skipped_steps=0,
            total_cost_usd=1.23, total_duration_secs=45.0,
        )
        summary = report.summary()
        assert "5/5" in summary
        assert "$1.23" in summary


# ===========================================================================
# Output normalizer tests
# ===========================================================================


class TestNormalizeOutput:
    def test_extracts_session_and_result(self):
        messages = [
            {"type": "system", "subtype": "init", "session_id": "abc"},
            {"type": "result", "result": "Hello world", "cost_usd": 0.01, "is_error": False},
        ]
        out = _normalize_claude_output(messages)
        assert out["session_id"] == "abc"
        assert out["result"] == "Hello world"
        assert out["cost_usd"] == 0.01

    def test_fallback_concatenation(self):
        messages = [
            {"type": "assistant", "message": {"content": [
                {"type": "text", "text": "line1"},
                {"type": "text", "text": "line2"},
            ]}},
        ]
        out = _normalize_claude_output(messages)
        assert "line1" in out["result"]
        assert "line2" in out["result"]


# ===========================================================================
# Plan summary tests
# ===========================================================================


class TestSummarizePlan:
    def test_basic_summary(self):
        summary = _summarize_plan(SAMPLE_PLAN)
        assert "Add logging" in summary
        assert "moderate" in summary

    def test_empty_plan(self):
        summary = _summarize_plan({"goal": "nothing", "phases": []})
        assert "nothing" in summary


# ===========================================================================
# Integration: poller wiring
# ===========================================================================


class TestPollerIntegration:
    def test_import_from_poller_script(self):
        """Verify the poller script can import the new classes."""
        from multihead.autonomous_executor import (
            AutonomousExecutor,
            ClaudeSessionStrategy,
            ExecutionStrategy,
            LocalLLMStrategy,
        )
        assert AutonomousExecutor is not None
        assert ClaudeSessionStrategy is not None

    def test_local_strategy_is_plan_only(self):
        """LocalLLMStrategy should not trigger executor creation."""
        strategy = LocalLLMStrategy()
        # The poller checks isinstance(strategy, LocalLLMStrategy) to skip executor
        assert isinstance(strategy, ExecutionStrategy)
        assert isinstance(strategy, LocalLLMStrategy)
