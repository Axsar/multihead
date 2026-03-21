"""Tests for intent classification + VLM routing + task execution edge cases.

Sprint 2: deeper testing of ShellPipeline's classification, decomposition
routing, VLM auto-routing, and run result formatting.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from multihead.runtime_config import RuntimeConfig
from multihead.shell_pipeline import ShellPipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    return RuntimeConfig()


@pytest.fixture
def pipeline(config):
    return ShellPipeline(runtime_config=config)


@pytest.fixture
def vlm_pipeline(config):
    """Pipeline with VLM routing infrastructure."""
    hm = MagicMock()
    hm.ensure_active = AsyncMock()
    adapter = MagicMock()
    adapter.generate = AsyncMock(return_value={"text": "I see a cat in the image."})
    hm.get_adapter.return_value = adapter

    router = MagicMock()
    router.route.return_value = "qwen-vlm"

    config.pipeline.vlm_auto_route = True
    return ShellPipeline(
        head_manager=hm,
        router=router,
        runtime_config=config,
    )


@pytest.fixture
def brain_fn():
    return AsyncMock(return_value="Brain response here.")


# ---------------------------------------------------------------------------
# Intent Classification — Chat Cases
# ---------------------------------------------------------------------------


class TestChatClassification:
    def test_greeting(self, pipeline):
        assert pipeline._classify_intent("hi") == "chat"

    def test_short_question(self, pipeline):
        assert pipeline._classify_intent("what time?") == "chat"

    def test_one_word(self, pipeline):
        assert pipeline._classify_intent("hello") == "chat"

    def test_simple_how_question(self, pipeline):
        assert pipeline._classify_intent("How does the router work?") == "chat"

    def test_opinion_question(self, pipeline):
        assert pipeline._classify_intent(
            "What do you think about the current architecture?"
        ) == "chat"

    def test_explanation_request(self, pipeline):
        assert pipeline._classify_intent("Explain the consensus mechanism") == "chat"

    def test_status_question(self, pipeline):
        assert pipeline._classify_intent("What is the current status of the project?") == "chat"

    def test_yes_no(self, pipeline):
        assert pipeline._classify_intent("yes") == "chat"

    def test_thanks(self, pipeline):
        assert pipeline._classify_intent("thanks") == "chat"

    def test_four_word_sentence(self, pipeline):
        assert pipeline._classify_intent("tell me about this") == "chat"


# ---------------------------------------------------------------------------
# Intent Classification — Task Cases
# ---------------------------------------------------------------------------


class TestTaskClassification:
    def test_fix_with_file_path(self, pipeline):
        assert pipeline._classify_intent(
            "Fix the bug in src/multihead/router.py that causes incorrect scoring"
        ) == "task"

    def test_implement_with_details(self, pipeline):
        assert pipeline._classify_intent(
            "Implement a retry decorator for the API client in utils.py with exponential backoff"
        ) == "task"

    def test_multi_step_first_then(self, pipeline):
        assert pipeline._classify_intent(
            "First refactor the knowledge store, then add FTS5 support for search"
        ) == "task"

    def test_create_with_file(self, pipeline):
        assert pipeline._classify_intent(
            "Create a new test file tests/test_vlm.py with coverage for all VLM adapters"
        ) == "task"

    def test_refactor_long(self, pipeline):
        assert pipeline._classify_intent(
            "Refactor the orchestrator in src/multihead/orchestrator.py to support "
            "parallel execution with DAG-based dependency resolution using the "
            "existing auto-decomposition infrastructure"
        ) == "task"

    def test_build_with_steps(self, pipeline):
        assert pipeline._classify_intent(
            "Build a caching layer. Step 1: add Redis connection. Step 2: implement cache decorator"
        ) == "task"

    def test_update_with_path(self, pipeline):
        assert pipeline._classify_intent(
            "Update the config/solvers.yaml to add a new deterministic solver for URL parsing"
        ) == "task"

    def test_deploy_multi_step(self, pipeline):
        assert pipeline._classify_intent(
            "Deploy the application: first run tests,"
            " then build the docker image,"
            " after that push to registry"
        ) == "task"


# ---------------------------------------------------------------------------
# Intent Classification — Edge Cases
# ---------------------------------------------------------------------------


class TestIntentEdgeCases:
    def test_question_with_action_verb_no_file(self, pipeline):
        """Questions with action verbs but no files should be chat."""
        assert pipeline._classify_intent("How do I fix this error?") == "chat"

    def test_single_action_verb_short(self, pipeline):
        """Single action verb in short sentence should be chat."""
        assert pipeline._classify_intent("fix the bug") == "chat"

    def test_long_question_no_action(self, pipeline):
        """Long question without action verbs should be chat."""
        assert pipeline._classify_intent(
            "What are the different consensus strategies available in the system "
            "and how do they compare in terms of accuracy and performance?"
        ) == "chat"

    def test_empty_input(self, pipeline):
        assert pipeline._classify_intent("") == "chat"


# ---------------------------------------------------------------------------
# Action Verb Detection
# ---------------------------------------------------------------------------


class TestActionVerbs:
    def test_common_verbs(self, pipeline):
        for verb in ["build", "fix", "implement", "create", "refactor",
                      "add", "remove", "delete", "update", "modify"]:
            assert pipeline._has_action_verbs(f"please {verb} this thing"), f"Failed for {verb}"

    def test_no_action_verbs(self, pipeline):
        assert not pipeline._has_action_verbs("the sky is blue today")

    def test_action_verb_as_substring(self, pipeline):
        """'created' contains 'create' but split-based check should not match."""
        assert not pipeline._has_action_verbs("I already created it")


# ---------------------------------------------------------------------------
# File Detection
# ---------------------------------------------------------------------------


class TestFileMentions:
    def test_python_file(self, pipeline):
        assert pipeline._mentions_files("check router.py")

    def test_absolute_path(self, pipeline):
        assert pipeline._mentions_files("look at /home/user/projects/file.txt")

    def test_src_prefix(self, pipeline):
        assert pipeline._mentions_files("update src/multihead/core.py")

    def test_tests_prefix(self, pipeline):
        assert pipeline._mentions_files("run tests/test_router.py")

    def test_config_prefix(self, pipeline):
        assert pipeline._mentions_files("edit config/heads.yaml")

    def test_no_files(self, pipeline):
        assert not pipeline._mentions_files("hello world")


# ---------------------------------------------------------------------------
# Multi-Step Language
# ---------------------------------------------------------------------------


class TestMultiStepLanguage:
    def test_first_then(self, pipeline):
        assert pipeline._has_multi_step_language("first do this then do that")

    def test_step_numbered(self, pipeline):
        assert pipeline._has_multi_step_language("step 1: setup the environment")

    def test_numbered_list(self, pipeline):
        assert pipeline._has_multi_step_language("1. install deps 2. run tests")

    def test_and_then(self, pipeline):
        assert pipeline._has_multi_step_language("run the build and then deploy")

    def test_after_that(self, pipeline):
        assert pipeline._has_multi_step_language("compile the code after that run tests")

    def test_finally(self, pipeline):
        assert pipeline._has_multi_step_language("finally verify everything works")

    def test_no_multi_step(self, pipeline):
        assert not pipeline._has_multi_step_language("just run the tests")


# ---------------------------------------------------------------------------
# VLM Auto-Routing
# ---------------------------------------------------------------------------


class TestVLMAutoRouting:
    async def test_routes_image_to_vlm(self, vlm_pipeline, brain_fn):
        result = await vlm_pipeline.process(
            "Analyze this image /tmp/screenshot.png and describe what you see",
            brain_fn, "ses_1",
        )
        assert "cat" in result
        assert vlm_pipeline._stats["vlm_routes"] == 1
        brain_fn.assert_not_called()

    async def test_no_vlm_route_when_disabled(self, vlm_pipeline, brain_fn):
        vlm_pipeline._config.pipeline.vlm_auto_route = False
        result = await vlm_pipeline.process(
            "Analyze this image /tmp/screenshot.png",
            brain_fn, "ses_1",
        )
        assert result == brain_fn.return_value
        assert vlm_pipeline._stats["vlm_routes"] == 0

    async def test_fallback_when_no_vlm_head(self, config, brain_fn):
        router = MagicMock()
        router.route.return_value = None  # No VLM available
        config.pipeline.vlm_auto_route = True
        p = ShellPipeline(router=router, head_manager=MagicMock(), runtime_config=config)
        result = await p.process(
            "Analyze this image /tmp/screenshot.png and tell me what you see in it",
            brain_fn, "ses_1",
        )
        assert result == brain_fn.return_value

    async def test_fallback_on_vlm_error(self, vlm_pipeline, brain_fn):
        vlm_pipeline._hm.get_adapter.return_value.generate = AsyncMock(
            side_effect=RuntimeError("VLM crashed")
        )
        result = await vlm_pipeline.process(
            "Analyze this image /tmp/screenshot.png and describe what you see in it",
            brain_fn, "ses_1",
        )
        assert result == brain_fn.return_value

    async def test_no_vlm_route_without_image(self, vlm_pipeline, brain_fn):
        result = await vlm_pipeline.process(
            "Tell me about the current model architecture",
            brain_fn, "ses_1",
        )
        assert result == brain_fn.return_value
        assert vlm_pipeline._stats["vlm_routes"] == 0

    async def test_vlm_wakes_head(self, vlm_pipeline, brain_fn):
        await vlm_pipeline.process(
            "Analyze this image /tmp/screenshot.png and describe what you see",
            brain_fn, "ses_1",
        )
        vlm_pipeline._hm.ensure_active.assert_called_once_with("qwen-vlm")


# ---------------------------------------------------------------------------
# Task Execution Edge Cases
# ---------------------------------------------------------------------------


class TestTaskExecutionEdges:
    async def test_task_with_knowledge_context(self):
        decomp = MagicMock()
        plan = MagicMock()
        work_order = MagicMock()
        decomp.decompose = AsyncMock(return_value=plan)
        decomp.to_work_order_with_dag = MagicMock(return_value=work_order)

        orch = MagicMock()
        run_state = MagicMock()
        run_state.run_id = "run_123"
        run_state.status = "completed"
        run_state.step_results = {}
        orch.create_run = AsyncMock(return_value=run_state)
        orch.execute_run = AsyncMock(return_value=run_state)

        rc = RuntimeConfig()
        rc.pipeline.decompose_head = "mock-llm"
        p = ShellPipeline(
            orchestrator=orch,
            auto_decomposer=decomp,
            runtime_config=rc,
        )

        brain_fn = AsyncMock(return_value="fallback")
        result = await p._execute_as_task("do task", "context here", "ses_1", brain_fn)

        # Should pass context to decomposer
        call_kwargs = decomp.decompose.call_args[1]
        assert call_kwargs["context"] == "context here"
        assert "run_123" in result

    async def test_stats_tracked_on_decompose(self):
        decomp = MagicMock()
        plan = MagicMock()
        work_order = MagicMock()
        decomp.decompose = AsyncMock(return_value=plan)
        decomp.to_work_order_with_dag = MagicMock(return_value=work_order)

        orch = MagicMock()
        run_state = MagicMock()
        run_state.run_id = "run_456"
        run_state.status = "completed"
        run_state.step_results = {}
        orch.create_run = AsyncMock(return_value=run_state)
        orch.execute_run = AsyncMock(return_value=run_state)

        rc = RuntimeConfig()
        rc.pipeline.decompose_head = "mock-llm"
        p = ShellPipeline(
            orchestrator=orch,
            auto_decomposer=decomp,
            runtime_config=rc,
        )

        brain_fn = AsyncMock()
        await p._execute_as_task("do task", "", "ses_1", brain_fn)
        assert p._stats["tasks_decomposed"] == 1

    def test_format_results_no_steps(self):
        p = ShellPipeline()
        run_state = MagicMock()
        run_state.run_id = "run_000"
        run_state.status = "completed"
        run_state.step_results = {}
        text = p._format_run_results(run_state)
        assert "completed" in text

    def test_format_results_truncates_long_output(self):
        p = ShellPipeline()
        run_state = MagicMock()
        run_state.run_id = "run_000"
        run_state.status = "completed"
        step_result = MagicMock()
        step_result.status = "completed"
        step_result.output = "x" * 500
        run_state.step_results = {"step_1": step_result}
        text = p._format_run_results(run_state)
        assert "..." in text
