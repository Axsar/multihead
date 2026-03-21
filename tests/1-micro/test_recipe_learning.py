"""Tests for BotVibes recipe learning workflow."""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from multihead.recipe_learning import RecipeLearner, learn_recipe_workflow
from multihead.recipe_learning._parsing import extract_yaml, parse_recipe_from_response
from multihead.acp_bridge import ACPBridge


@pytest.fixture
def mock_acp_bridge():
    """Create mock ACP bridge."""
    bridge = MagicMock(spec=ACPBridge)
    bridge.create_task = AsyncMock()
    bridge.poll_for_completion = AsyncMock()
    return bridge


@pytest.fixture
def recipe_learner(mock_acp_bridge, tmp_path):
    """Create RecipeLearner with mocks."""
    recipes_dir = tmp_path / "recipes"
    test_data_dir = tmp_path / "test_data"
    return RecipeLearner(
        acp_bridge=mock_acp_bridge,
        recipes_dir=recipes_dir,
        test_data_dir=test_data_dir,
    )


class TestRecipeLearnerInit:
    """Test RecipeLearner initialization."""

    def test_creates_recipes_dir(self, mock_acp_bridge, tmp_path):
        """Should create recipes directory if it doesn't exist."""
        recipes_dir = tmp_path / "new_recipes"
        assert not recipes_dir.exists()

        learner = RecipeLearner(
            acp_bridge=mock_acp_bridge,
            recipes_dir=recipes_dir,
        )

        assert recipes_dir.exists()
        assert learner.recipes_dir == recipes_dir

    def test_stores_test_data_dir(self, mock_acp_bridge, tmp_path):
        """Should store test data directory."""
        test_data_dir = tmp_path / "test_data"
        learner = RecipeLearner(
            acp_bridge=mock_acp_bridge,
            recipes_dir=tmp_path / "recipes",
            test_data_dir=test_data_dir,
        )

        assert learner.test_data_dir == test_data_dir


class TestBuildRecipeDesignPrompt:
    """Test prompt building for recipe design."""

    def test_builds_prompt_with_goal_and_requirements(self, recipe_learner):
        """Should build prompt with goal and requirements."""
        goal = "Extract entities from documents"
        requirements = {
            "input_format": "markdown",
            "output_format": "json",
            "max_steps": 5,
        }

        prompt = recipe_learner._build_recipe_design_prompt(goal, requirements)

        assert "Extract entities from documents" in prompt
        assert "input_format: markdown" in prompt
        assert "output_format: json" in prompt
        assert "max_steps: 5" in prompt
        assert "atomic, composable steps" in prompt
        assert "dependencies clearly" in prompt

    def test_includes_privacy_guidance(self, recipe_learner):
        """Should include privacy and validation guidance."""
        prompt = recipe_learner._build_recipe_design_prompt(
            "Test goal",
            {"requirement": "value"},
        )

        assert "privacy constraints" in prompt
        assert "consensus where appropriate" in prompt


class TestExtractYAML:
    """Test YAML extraction from responses."""

    def test_extracts_from_yaml_code_block(self, recipe_learner):
        """Should extract YAML from markdown code block."""
        response = """
Here's the recipe:

```yaml
goal: Test recipe
steps:
  - step_id: step1
    prompt_template: "Do something"
```

This should work well.
"""
        yaml_text = extract_yaml(response)

        assert yaml_text is not None
        assert "goal: Test recipe" in yaml_text
        assert "step_id: step1" in yaml_text

    def test_extracts_from_yml_code_block(self, recipe_learner):
        """Should extract YAML from yml code block."""
        response = """```yml
goal: Test
steps: []
```"""
        yaml_text = extract_yaml(response)

        assert yaml_text is not None
        assert "goal: Test" in yaml_text

    def test_extracts_plain_yaml(self, recipe_learner):
        """Should extract plain YAML without code blocks."""
        response = """goal: Plain recipe
steps:
  - step_id: test
    prompt_template: "Test"
"""
        yaml_text = extract_yaml(response)

        assert yaml_text is not None
        assert "goal: Plain recipe" in yaml_text

    def test_extracts_yaml_after_text(self, recipe_learner):
        """Should extract YAML that appears after explanatory text."""
        response = """Here's what I designed:

goal: Embedded recipe
steps:
  - step_id: s1
    prompt_template: "Do it"
"""
        yaml_text = extract_yaml(response)

        assert yaml_text is not None
        assert "goal: Embedded recipe" in yaml_text

    def test_returns_none_for_no_yaml(self, recipe_learner):
        """Should return None if no YAML found."""
        response = "Just some text without any YAML content"
        yaml_text = extract_yaml(response)

        assert yaml_text is None


class TestParseRecipeFromResponse:
    """Test recipe parsing."""

    def test_parses_valid_recipe(self, recipe_learner):
        """Should parse valid recipe YAML."""
        response = """```yaml
goal: Test recipe
steps:
  - step_id: step1
    prompt_template: "Do something"
    head_id: mock-llm
```"""
        recipe = parse_recipe_from_response(response)

        assert recipe is not None
        assert recipe["goal"] == "Test recipe"
        assert len(recipe["steps"]) == 1
        assert recipe["steps"][0]["step_id"] == "step1"

    def test_adds_default_goal_if_missing(self, recipe_learner):
        """Should add default goal if missing."""
        response = """steps:
  - step_id: step1
    prompt_template: "Test"
"""
        recipe = parse_recipe_from_response(response)

        assert recipe is not None
        assert recipe["goal"] == "Expert-designed recipe"

    def test_returns_none_for_missing_steps(self, recipe_learner):
        """Should return None if steps missing."""
        response = "goal: No steps recipe"
        recipe = parse_recipe_from_response(response)

        assert recipe is None

    def test_returns_none_for_invalid_yaml(self, recipe_learner):
        """Should return None for invalid YAML."""
        response = """```yaml
goal: Broken
  steps:
    this is not valid yaml: [[[
```"""
        recipe = parse_recipe_from_response(response)

        assert recipe is None

    def test_returns_none_for_non_dict(self, recipe_learner):
        """Should return None if parsed YAML is not a dict."""
        response = "```yaml\n- item1\n- item2\n```"
        recipe = parse_recipe_from_response(response)

        assert recipe is None


class TestQueryExpertRecipe:
    """Test querying BotVibes experts."""

    @pytest.mark.asyncio
    async def test_creates_acp_task(self, recipe_learner, mock_acp_bridge):
        """Should create ACP task with recipe_design capability."""
        mock_acp_bridge.create_task.return_value = "task-123"
        mock_acp_bridge.poll_for_completion.return_value = {
            "status": "complete",
            "output_ref": "goal: Test\nsteps:\n  - step_id: s1\n    prompt_template: 'Test'",
        }

        goal = "Extract entities"
        requirements = {"format": "json"}

        recipe = await recipe_learner.query_expert_recipe(goal, requirements)

        # Verify ACP task creation
        mock_acp_bridge.create_task.assert_called_once()
        call_args = mock_acp_bridge.create_task.call_args
        assert call_args.kwargs["capability"] == "recipe_design"
        assert "Extract entities" in call_args.kwargs["payload_ref"]
        assert "format: json" in call_args.kwargs["payload_ref"]

    @pytest.mark.asyncio
    async def test_polls_for_completion(self, recipe_learner, mock_acp_bridge):
        """Should poll for task completion."""
        mock_acp_bridge.create_task.return_value = "task-456"
        mock_acp_bridge.poll_for_completion.return_value = {
            "status": "complete",
            "output_ref": "goal: Test\nsteps:\n  - step_id: s1\n    prompt_template: 'Test'",
        }

        await recipe_learner.query_expert_recipe("Test goal", {})

        mock_acp_bridge.poll_for_completion.assert_called_once_with(
            "task-456",
            timeout_seconds=300.0,
            poll_interval=5.0,
        )

    @pytest.mark.asyncio
    async def test_parses_expert_response(self, recipe_learner, mock_acp_bridge):
        """Should parse recipe from expert response."""
        mock_acp_bridge.create_task.return_value = "task-789"
        mock_acp_bridge.poll_for_completion.return_value = {
            "status": "complete",
            "output_ref": """```yaml
goal: Expert recipe
steps:
  - step_id: extract
    prompt_template: "Extract entities from {input}"
    head_id: qwen-llm
```""",
        }

        recipe = await recipe_learner.query_expert_recipe("Extract entities", {})

        assert recipe is not None
        assert recipe["goal"] == "Expert recipe"
        assert len(recipe["steps"]) == 1
        assert recipe["steps"][0]["step_id"] == "extract"

    @pytest.mark.asyncio
    async def test_returns_none_on_timeout(self, recipe_learner, mock_acp_bridge):
        """Should return None if task times out."""
        mock_acp_bridge.create_task.return_value = "task-timeout"
        mock_acp_bridge.poll_for_completion.return_value = {
            "status": "pending",
        }

        recipe = await recipe_learner.query_expert_recipe("Test", {})

        assert recipe is None

    @pytest.mark.asyncio
    async def test_returns_none_on_error(self, recipe_learner, mock_acp_bridge):
        """Should return None if query fails."""
        mock_acp_bridge.create_task.side_effect = Exception("Network error")

        recipe = await recipe_learner.query_expert_recipe("Test", {})

        assert recipe is None

    @pytest.mark.asyncio
    async def test_supports_conversation_threading(self, recipe_learner, mock_acp_bridge):
        """Should pass conversation_id for multi-turn refinement."""
        mock_acp_bridge.create_task.return_value = "task-conv"
        mock_acp_bridge.poll_for_completion.return_value = {
            "status": "complete",
            "output_ref": "goal: Test\nsteps:\n  - step_id: s1\n    prompt_template: 'Test'",
        }

        await recipe_learner.query_expert_recipe(
            "Test",
            {},
            conversation_id="conv-123",
        )

        call_args = mock_acp_bridge.create_task.call_args
        assert call_args.kwargs["conversation_id"] == "conv-123"


class TestBenchmarkRecipe:
    """Test recipe benchmarking."""

    @pytest.mark.asyncio
    async def test_returns_benchmark_structure(self, recipe_learner):
        """Should return benchmark results structure."""
        recipe = {
            "goal": "Test recipe",
            "steps": [{"step_id": "s1", "prompt_template": "Test"}],
        }
        test_cases = [
            {"input": "test1", "expected": "output1"},
            {"input": "test2", "expected": "output2"},
        ]

        results = await recipe_learner.benchmark_recipe(recipe, test_cases)

        assert results["recipe_goal"] == "Test recipe"
        assert results["test_cases_count"] == 2
        assert "success_count" in results
        assert "failure_count" in results
        assert "avg_latency_ms" in results
        assert "avg_cost_usd" in results
        assert "test_results" in results


class TestEvaluateRecipe:
    """Test recipe evaluation."""

    @pytest.mark.asyncio
    async def test_adopts_when_no_current_recipe(self, recipe_learner):
        """Should adopt if no current recipe and shows promise."""
        proposed = {"goal": "New recipe", "steps": []}
        benchmark = {"success_count": 5, "test_cases_count": 10}

        decision = await recipe_learner.evaluate_recipe(proposed, benchmark)

        assert decision["action"] == "adopt"
        assert "No current recipe" in decision["rationale"]

    @pytest.mark.asyncio
    async def test_adopts_when_better_than_current(self, recipe_learner):
        """Should adopt if proposed outperforms current."""
        proposed = {"goal": "Better recipe", "steps": []}
        proposed_bench = {"success_count": 9, "test_cases_count": 10}
        current = {"goal": "Old recipe", "steps": []}
        current_bench = {"success_count": 6, "test_cases_count": 10}

        decision = await recipe_learner.evaluate_recipe(
            proposed, proposed_bench, current, current_bench
        )

        assert decision["action"] == "adopt"
        assert "outperforms current" in decision["rationale"]

    @pytest.mark.asyncio
    async def test_rejects_when_worse_than_current(self, recipe_learner):
        """Should reject if current performs better."""
        proposed = {"goal": "Worse recipe", "steps": []}
        proposed_bench = {"success_count": 5, "test_cases_count": 10}
        current = {"goal": "Good recipe", "steps": []}
        current_bench = {"success_count": 9, "test_cases_count": 10}

        decision = await recipe_learner.evaluate_recipe(
            proposed, proposed_bench, current, current_bench
        )

        assert decision["action"] == "reject"
        assert "performs better" in decision["rationale"]


class TestSaveRecipe:
    """Test recipe saving."""

    def test_saves_recipe_to_yaml(self, recipe_learner):
        """Should save recipe as YAML file."""
        recipe = {
            "goal": "Test recipe",
            "steps": [
                {"step_id": "s1", "prompt_template": "Do something"},
            ],
        }

        path = recipe_learner.save_recipe(recipe, "test-recipe")

        assert path.exists()
        assert path.name == "test-recipe.yaml"
        content = path.read_text()
        assert "goal: Test recipe" in content
        assert "step_id: s1" in content

    def test_adds_metadata_as_comments(self, recipe_learner):
        """Should add metadata as YAML comments."""
        recipe = {"goal": "Test", "steps": []}
        metadata = {
            "source": "botvibes_expert",
            "confidence": 0.85,
        }

        path = recipe_learner.save_recipe(recipe, "with-metadata", metadata=metadata)

        content = path.read_text()
        assert "# source: botvibes_expert" in content
        assert "# confidence: 0.85" in content
        assert "# learned_at:" in content


class TestShareSuccess:
    """Test sharing successful recipes."""

    @pytest.mark.asyncio
    async def test_creates_feedback_task(self, recipe_learner, mock_acp_bridge):
        """Should create ACP task with recipe_feedback capability."""
        mock_acp_bridge.create_task.return_value = "feedback-task"

        recipe = {"goal": "Successful recipe", "steps": []}
        benchmark = {"success_count": 10, "test_cases_count": 10}

        success = await recipe_learner.share_success(recipe, benchmark)

        assert success is True
        mock_acp_bridge.create_task.assert_called_once()
        call_args = mock_acp_bridge.create_task.call_args
        assert call_args.kwargs["capability"] == "recipe_feedback"
        assert call_args.kwargs["priority"] == "batch"

    @pytest.mark.asyncio
    async def test_returns_false_on_error(self, recipe_learner, mock_acp_bridge):
        """Should return False if sharing fails."""
        mock_acp_bridge.create_task.side_effect = Exception("Network error")

        recipe = {"goal": "Test", "steps": []}
        benchmark = {"success_count": 5, "test_cases_count": 10}

        success = await recipe_learner.share_success(recipe, benchmark)

        assert success is False


class TestLearnRecipeWorkflow:
    """Test complete learning workflow."""

    @pytest.mark.asyncio
    async def test_complete_workflow_success(self, recipe_learner, mock_acp_bridge):
        """Should complete full workflow and save recipe."""
        # Mock expert query
        mock_acp_bridge.create_task.return_value = "task-123"
        mock_acp_bridge.poll_for_completion.return_value = {
            "status": "complete",
            "output_ref": """goal: Learned recipe
steps:
  - step_id: extract
    prompt_template: "Extract"
""",
        }

        goal = "Extract entities"
        requirements = {"format": "json"}
        test_cases = [{"input": "test", "expected": "output"}]

        result = await learn_recipe_workflow(
            goal, requirements, test_cases, recipe_learner, save_name="learned-test"
        )

        assert result["success"] is True
        assert result["proposed_recipe"]["goal"] == "Learned recipe"
        assert result["evaluation"]["action"] == "adopt"
        assert result["saved_path"] is not None

        # Verify recipe was saved
        saved_path = Path(result["saved_path"])
        assert saved_path.exists()

    @pytest.mark.asyncio
    async def test_workflow_fails_if_no_expert_response(self, recipe_learner, mock_acp_bridge):
        """Should fail if expert doesn't respond."""
        mock_acp_bridge.create_task.side_effect = Exception("No expert available")

        result = await learn_recipe_workflow(
            "Test", {}, [], recipe_learner
        )

        assert result["success"] is False
        assert "error" in result
