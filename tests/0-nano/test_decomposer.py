"""Tests for the TaskDecomposer module."""

from __future__ import annotations

import json
import pytest

from multihead.decomposer import (
    DecompositionPlan,
    TaskDecomposer,
    TaskNode,
    _extract_keywords,
    _strip_llm_wrapper,
    parse_json_array,
    parse_json_object,
    parse_node,
    parse_plan,
)
from multihead.models import WorkOrder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_LLM_RESPONSE = json.dumps({
    "complexity": "moderate",
    "phases": [
        {
            "id": "1",
            "goal": "Understand the problem",
            "rationale": "Need to read relevant code first",
            "children": [
                {
                    "id": "1.1",
                    "goal": "Read the layout loop",
                    "action_type": "read",
                    "target_files": ["auth_handler.py"],
                    "expected_output": "Understanding of bottom-up layout",
                    "rationale": "",
                    "children": [],
                },
                {
                    "id": "1.2",
                    "goal": "Read test cases",
                    "action_type": "read",
                    "target_files": ["test_layout.py"],
                    "expected_output": "List of tested edge cases",
                    "rationale": "",
                    "children": [],
                },
            ],
        },
        {
            "id": "2",
            "goal": "Fix the overflow",
            "rationale": "Apply the fix based on diagnosis",
            "children": [
                {
                    "id": "2.1",
                    "goal": "Modify freeze threshold",
                    "action_type": "edit",
                    "target_files": ["auth_handler.py"],
                    "expected_output": "Threshold changed from 50% to 40%",
                    "rationale": "",
                    "children": [],
                },
                {
                    "id": "2.2",
                    "goal": "Run tests",
                    "action_type": "test",
                    "target_files": [],
                    "expected_output": "All tests pass",
                    "rationale": "",
                    "children": [],
                },
            ],
        },
    ],
})

NESTED_RESPONSE = json.dumps({
    "complexity": "complex",
    "phases": [
        {
            "id": "1",
            "goal": "Phase one",
            "children": [
                {
                    "id": "1.1",
                    "goal": "Step with substeps",
                    "children": [
                        {
                            "id": "1.1.1",
                            "goal": "Deep leaf",
                            "action_type": "read",
                            "children": [],
                        },
                        {
                            "id": "1.1.2",
                            "goal": "Another deep leaf",
                            "action_type": "edit",
                            "children": [],
                        },
                    ],
                },
            ],
        },
    ],
})


class FakeHeadManager:
    """Mock HeadManager that returns canned responses."""

    def __init__(self, response: str = SAMPLE_LLM_RESPONSE):
        self._response = response
        self.last_prompt = ""
        self.generate_count = 0

    async def generate(self, head_id: str, prompt: str = "", **kwargs):
        self.last_prompt = prompt
        self.generate_count += 1
        return {"text": self._response}

    def get_states(self):
        return {
            "mock-llm": {
                "head_id": "mock-llm",
                "state": "active",
                "kind": "llm",
                "is_active": True,
            }
        }

    def get_manifest(self, head_id: str):
        return {"kind": "llm"}


class FakeKnowledgeStore:
    """Mock KnowledgeStore that returns canned claims."""

    def __init__(self, claims=None):
        self._claims = claims or []

    def list_claims(self, status=None, limit=100):
        return self._claims


class FakeClaim:
    def __init__(self, key: str, statement: str):
        self.statement = statement
        self.canonical = type("C", (), {"claim_key": key})()


# ---------------------------------------------------------------------------
# TaskNode tests
# ---------------------------------------------------------------------------

class TestTaskNode:
    def test_leaf_node(self):
        node = TaskNode(id="1", goal="Read file")
        assert node.is_leaf
        assert node.depth == 0
        assert node.leaf_count() == 1
        assert node.leaves() == [node]

    def test_parent_node(self):
        child1 = TaskNode(id="1.1", goal="Step A")
        child2 = TaskNode(id="1.2", goal="Step B")
        parent = TaskNode(id="1", goal="Phase", children=[child1, child2])
        assert not parent.is_leaf
        assert parent.depth == 1
        assert parent.leaf_count() == 2
        assert parent.leaves() == [child1, child2]

    def test_nested_depth(self):
        deep = TaskNode(id="1.1.1", goal="Deep")
        mid = TaskNode(id="1.1", goal="Mid", children=[deep])
        top = TaskNode(id="1", goal="Top", children=[mid])
        assert top.depth == 2
        assert top.leaf_count() == 1

    def test_find(self):
        leaf = TaskNode(id="1.1.2", goal="Target")
        mid = TaskNode(id="1.1", goal="Mid", children=[
            TaskNode(id="1.1.1", goal="Other"),
            leaf,
        ])
        top = TaskNode(id="1", goal="Top", children=[mid])
        assert top.find("1.1.2") is leaf
        assert top.find("1.1") is mid
        assert top.find("1") is top
        assert top.find("9.9") is None


# ---------------------------------------------------------------------------
# DecompositionPlan tests
# ---------------------------------------------------------------------------

class TestDecompositionPlan:
    def _sample_plan(self):
        return parse_plan(SAMPLE_LLM_RESPONSE, "Fix overflow")

    def test_total_steps(self):
        plan = self._sample_plan()
        assert plan.total_steps == 4  # 1.1, 1.2, 2.1, 2.2

    def test_max_depth(self):
        plan = self._sample_plan()
        assert plan.max_depth == 2  # phase -> leaf

    def test_complexity(self):
        plan = self._sample_plan()
        assert plan.complexity == "moderate"

    def test_all_leaves(self):
        plan = self._sample_plan()
        leaves = plan.all_leaves()
        assert len(leaves) == 4
        assert leaves[0].id == "1.1"
        assert leaves[3].id == "2.2"

    def test_find(self):
        plan = self._sample_plan()
        assert plan.find("2.1") is not None
        assert plan.find("2.1").goal == "Modify freeze threshold"
        assert plan.find("99") is None

    def test_to_work_order(self):
        plan = self._sample_plan()
        wo = plan.to_work_order()
        assert isinstance(wo, WorkOrder)
        assert wo.goal == "Fix overflow"
        assert len(wo.steps) == 4
        # First step has no deps
        assert wo.steps[0].depends_on == []
        # Second step depends on first
        assert wo.steps[1].depends_on == ["1.1"]
        # Last step depends on previous
        assert wo.steps[3].depends_on == ["2.1"]

    def test_work_order_step_metadata(self):
        plan = self._sample_plan()
        wo = plan.to_work_order()
        step = wo.steps[0]
        assert step.name == "Read the layout loop"
        assert step.extra["action_type"] == "read"
        assert "auth_handler.py" in step.extra["target_files"]

    def test_render_tree(self):
        plan = self._sample_plan()
        tree = plan.render_tree()
        assert "Fix overflow" in tree
        assert "moderate" in tree.lower()
        assert "1.1." in tree
        assert "2.2." in tree
        assert "Total leaf steps: 4" in tree

    def test_nested_plan(self):
        plan = parse_plan(NESTED_RESPONSE, "Complex task")
        assert plan.total_steps == 2  # 1.1.1, 1.1.2
        assert plan.max_depth == 3
        assert plan.complexity == "complex"


# ---------------------------------------------------------------------------
# Parsing tests
# ---------------------------------------------------------------------------

class TestParsing:
    def test_parse_json_object(self):
        result = parse_json_object('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_object_with_markdown(self):
        raw = '```json\n{"key": "value"}\n```'
        result = parse_json_object(raw)
        assert result == {"key": "value"}

    def test_parse_json_object_with_thinking(self):
        raw = '<think>Let me think about this...</think>\n{"key": "value"}'
        result = parse_json_object(raw)
        assert result == {"key": "value"}

    def test_parse_json_object_with_preamble(self):
        raw = 'Here is the plan:\n{"complexity": "simple", "phases": []}'
        result = parse_json_object(raw)
        assert result["complexity"] == "simple"

    def test_parse_json_object_no_json(self):
        with pytest.raises(ValueError, match="No JSON object"):
            parse_json_object("no json here")

    def test_parse_json_array(self):
        result = parse_json_array('[{"id": "1"}, {"id": "2"}]')
        assert len(result) == 2

    def test_parse_json_array_with_fences(self):
        raw = '```\n[{"id": "1"}]\n```'
        result = parse_json_array(raw)
        assert len(result) == 1

    def test_parse_node_basic(self):
        data = {"id": "1", "goal": "Read file", "action_type": "read", "children": []}
        node = parse_node(data)
        assert node.id == "1"
        assert node.goal == "Read file"
        assert node.action_type == "read"
        assert node.is_leaf

    def test_parse_node_with_steps_key(self):
        """LLM might use 'steps' instead of 'children'."""
        data = {
            "id": "1",
            "goal": "Phase",
            "steps": [
                {"id": "1.1", "goal": "Step A", "children": []},
            ],
        }
        node = parse_node(data)
        assert len(node.children) == 1
        assert node.children[0].id == "1.1"

    def test_parse_node_with_name_key(self):
        """LLM might use 'name' instead of 'goal'."""
        data = {"id": "1", "name": "Read the code"}
        node = parse_node(data)
        assert node.goal == "Read the code"

    def test_parse_plan(self):
        plan = parse_plan(SAMPLE_LLM_RESPONSE, "My goal", ["key1"])
        assert plan.goal == "My goal"
        assert plan.context_used == ["key1"]
        assert len(plan.phases) == 2

    def test_strip_llm_wrapper_clean(self):
        assert _strip_llm_wrapper('{"a": 1}') == '{"a": 1}'

    def test_strip_llm_wrapper_fences(self):
        raw = "```json\n{\"a\": 1}\n```"
        assert _strip_llm_wrapper(raw) == '{"a": 1}'

    def test_strip_llm_wrapper_thinking(self):
        raw = "<think>hmm</think>\n{\"a\": 1}"
        assert '{"a": 1}' in _strip_llm_wrapper(raw)

    def test_extract_keywords(self):
        kws = _extract_keywords("Fix the memory leak in auth handler")
        assert "fix" in kws
        assert "text" in kws
        assert "overflow" in kws
        assert "balloon" in kws
        assert "layout" in kws
        # Stop words excluded
        assert "the" not in kws
        assert "in" not in kws


# ---------------------------------------------------------------------------
# TaskDecomposer tests
# ---------------------------------------------------------------------------

class TestTaskDecomposer:
    @pytest.mark.asyncio
    async def test_decompose_basic(self):
        hm = FakeHeadManager()
        td = TaskDecomposer(hm)
        plan = await td.decompose("Fix text overflow", head_id="mock-llm")
        assert plan.goal == "Fix text overflow"
        assert plan.total_steps == 4
        assert plan.complexity == "moderate"
        assert hm.generate_count == 1

    @pytest.mark.asyncio
    async def test_decompose_with_context(self):
        hm = FakeHeadManager()
        td = TaskDecomposer(hm)
        plan = await td.decompose(
            "Fix text overflow",
            context="This is in the balloon layout module",
            head_id="mock-llm",
        )
        assert "balloon layout" in hm.last_prompt

    @pytest.mark.asyncio
    async def test_decompose_with_knowledge(self):
        hm = FakeHeadManager()
        ks = FakeKnowledgeStore([
            FakeClaim("h2v.bubblefill", "BubbleFill is a 9-step bottom-up balloon text layout"),
            FakeClaim("h2v.stage7a", "Stage 7a does export with balloon layout"),
            FakeClaim("unrelated", "Something about databases"),
        ])
        td = TaskDecomposer(hm, knowledge_store=ks)
        plan = await td.decompose("Fix memory leak in auth handler", head_id="mock-llm")
        # Should have found relevant claims
        assert "BubbleFill" in hm.last_prompt or "balloon" in hm.last_prompt.lower()

    @pytest.mark.asyncio
    async def test_decompose_resolves_head(self):
        hm = FakeHeadManager()
        td = TaskDecomposer(hm)
        # No explicit head_id — should resolve via get_states
        plan = await td.decompose("Fix something")
        assert hm.generate_count == 1

    @pytest.mark.asyncio
    async def test_decompose_enforces_depth(self):
        hm = FakeHeadManager(NESTED_RESPONSE)
        td = TaskDecomposer(hm)
        plan = await td.decompose("Complex task", head_id="mock-llm", max_depth=2)
        # At max_depth=2, depth-3 nodes should be trimmed
        assert plan.max_depth <= 2

    @pytest.mark.asyncio
    async def test_refine(self):
        refine_response = json.dumps([
            {
                "id": "2.1.1", "goal": "Read the threshold constant",
                "action_type": "read", "children": [],
            },
            {
                "id": "2.1.2", "goal": "Change value from 50 to 40",
                "action_type": "edit", "children": [],
            },
        ])
        hm = FakeHeadManager(refine_response)
        td = TaskDecomposer(hm)
        node = TaskNode(id="2.1", goal="Modify freeze threshold", action_type="edit")
        children = await td.refine(node, head_id="mock-llm")
        assert len(children) == 2
        assert children[0].id == "2.1.1"
        assert children[1].action_type == "edit"

    @pytest.mark.asyncio
    async def test_refine_with_exploration(self):
        refine_response = json.dumps([
            {"id": "1.1", "goal": "Substep", "children": []},
        ])
        hm = FakeHeadManager(refine_response)
        td = TaskDecomposer(hm)
        node = TaskNode(id="1", goal="Explore", action_type="explore")
        children = await td.refine(
            node,
            exploration_result="Found 3 relevant files",
            head_id="mock-llm",
        )
        assert "Found 3 relevant files" in hm.last_prompt

    @pytest.mark.asyncio
    async def test_decompose_bad_json_raises(self):
        hm = FakeHeadManager("This is not JSON at all")
        td = TaskDecomposer(hm)
        with pytest.raises(ValueError, match="No JSON"):
            await td.decompose("Something", head_id="mock-llm")

    @pytest.mark.asyncio
    async def test_no_head_available_raises(self):
        class EmptyHeadManager:
            def get_states(self):
                return {}
            def get_manifest(self, hid):
                return {}

        td = TaskDecomposer(EmptyHeadManager())
        with pytest.raises(RuntimeError, match="No LLM head"):
            await td.decompose("Something")

    @pytest.mark.asyncio
    async def test_knowledge_query_failure_handled(self):
        """Knowledge store errors don't crash decomposition."""
        class BrokenKS:
            def list_claims(self, **kwargs):
                raise RuntimeError("DB gone")

        hm = FakeHeadManager()
        td = TaskDecomposer(hm, knowledge_store=BrokenKS())
        plan = await td.decompose("Fix something", head_id="mock-llm")
        assert plan.total_steps == 4  # Still works, just no context


# ---------------------------------------------------------------------------
# Integration: round-trip plan -> work_order -> verify
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_plan_to_work_order_to_steps(self):
        plan = parse_plan(SAMPLE_LLM_RESPONSE, "Fix overflow")
        wo = plan.to_work_order()

        # All leaf steps present
        assert len(wo.steps) == 4

        # Dependencies form a chain
        for i in range(1, len(wo.steps)):
            assert wo.steps[i].depends_on == [wo.steps[i - 1].step_id]

        # All steps have required_kind for routing
        for step in wo.steps:
            assert step.required_kind == "llm"

    def test_empty_plan_to_work_order(self):
        plan = DecompositionPlan(goal="Nothing")
        wo = plan.to_work_order()
        assert len(wo.steps) == 0
        assert wo.goal == "Nothing"
