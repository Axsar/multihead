"""Tests for the plan normalizer module."""

from __future__ import annotations

import pytest

from multihead.head_manager import HeadManager
from multihead.models import AdapterKind, HeadManifest, StepDef, WorkOrder
from multihead.plan_normalizer import normalize


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def head_manager():
    """HeadManager with llm + vlm heads (mix of GPU and non-GPU)."""
    manifests = {
        "llm-a": HeadManifest(
            head_id="llm-a", name="LLM A", adapter=AdapterKind.MOCK,
            model="mock-v1", kind="llm", gpu_required=True,
        ),
        "llm-b": HeadManifest(
            head_id="llm-b", name="LLM B", adapter=AdapterKind.MOCK,
            model="mock-v1", kind="llm", gpu_required=False,
        ),
        "vlm-a": HeadManifest(
            head_id="vlm-a", name="VLM A", adapter=AdapterKind.MOCK,
            model="mock-v1", kind="vlm", gpu_required=True,
        ),
    }
    return HeadManager(manifests)


# ---------------------------------------------------------------------------
# Validate head IDs
# ---------------------------------------------------------------------------


class TestValidateHeadIds:
    def test_catches_unknown_primary_head(self, head_manager):
        wo = WorkOrder(
            goal="test",
            steps=[StepDef(name="s1", head_id="nonexistent")],
        )
        with pytest.raises(ValueError, match="Unknown head_id"):
            normalize(wo, head_manager)

    def test_catches_unknown_fallback_head(self, head_manager):
        wo = WorkOrder(
            goal="test",
            steps=[StepDef(name="s1", head_id="llm-a", fallback=["ghost"])],
        )
        with pytest.raises(ValueError, match="Unknown head_id"):
            normalize(wo, head_manager)

    def test_passes_for_valid_heads(self, head_manager):
        wo = WorkOrder(
            goal="test",
            steps=[StepDef(name="s1", head_id="llm-a")],
        )
        result = normalize(wo, head_manager)
        assert result.steps[0].head_id == "llm-a"


# ---------------------------------------------------------------------------
# Infer dependencies
# ---------------------------------------------------------------------------


class TestInferDependencies:
    def test_from_input_refs_by_name(self, head_manager):
        s1 = StepDef(name="extract", head_id="llm-a")
        s2 = StepDef(name="summarize", head_id="llm-a", input_refs=["extract"])
        wo = WorkOrder(goal="test", steps=[s1, s2])
        result = normalize(wo, head_manager)
        # s2 should now depend on s1's step_id
        assert result.steps[0].step_id in result.steps[1].depends_on

    def test_no_duplicates(self, head_manager):
        s1 = StepDef(name="extract", head_id="llm-a")
        s2 = StepDef(
            name="summarize", head_id="llm-a",
            input_refs=["extract"],
            depends_on=[s1.step_id],  # already explicit
        )
        wo = WorkOrder(goal="test", steps=[s1, s2])
        result = normalize(wo, head_manager)
        # Should not duplicate the dependency
        deps = result.steps[1].depends_on
        assert deps.count(result.steps[0].step_id) == 1

    def test_no_self_dependency(self, head_manager):
        s1 = StepDef(name="s1", head_id="llm-a", input_refs=["s1"])
        wo = WorkOrder(goal="test", steps=[s1])
        result = normalize(wo, head_manager)
        assert result.steps[0].step_id not in result.steps[0].depends_on


# ---------------------------------------------------------------------------
# Auto-assign fallbacks
# ---------------------------------------------------------------------------


class TestAutoAssignFallbacks:
    def test_assigns_same_kind(self, head_manager):
        wo = WorkOrder(
            goal="test",
            steps=[StepDef(name="s1", head_id="llm-a")],
        )
        result = normalize(wo, head_manager)
        # llm-a should get llm-b as fallback (same kind)
        assert "llm-b" in result.steps[0].fallback
        # vlm-a should NOT be in fallback (different kind)
        assert "vlm-a" not in result.steps[0].fallback

    def test_skips_when_explicit(self, head_manager):
        wo = WorkOrder(
            goal="test",
            steps=[StepDef(name="s1", head_id="llm-a", fallback=["llm-b"])],
        )
        result = normalize(wo, head_manager)
        # Should keep the explicit fallback, not add more
        assert result.steps[0].fallback == ["llm-b"]

    def test_prefers_non_gpu(self, head_manager):
        wo = WorkOrder(
            goal="test",
            steps=[StepDef(name="s1", head_id="vlm-a")],
        )
        result = normalize(wo, head_manager)
        # vlm-a is the only vlm, so no fallbacks available
        assert result.steps[0].fallback == []


# ---------------------------------------------------------------------------
# Deep copy guarantee
# ---------------------------------------------------------------------------


class TestRouteHeads:
    def test_routes_step_with_required_kind(self, head_manager):
        """Step with required_kind and empty head_id should get resolved."""
        wo = WorkOrder(
            goal="routing test",
            steps=[StepDef(name="s1", required_kind="llm")],
        )
        result = normalize(wo, head_manager)
        assert result.steps[0].head_id in ("llm-a", "llm-b")

    def test_raises_for_unroutable_kind(self, head_manager):
        """No heads of kind should raise ValueError."""
        wo = WorkOrder(
            goal="test",
            steps=[StepDef(name="s1", required_kind="embed")],
        )
        with pytest.raises(ValueError, match="No head available"):
            normalize(wo, head_manager)

    def test_explicit_head_id_not_overridden(self, head_manager):
        """Step with head_id set + required_kind should keep head_id."""
        wo = WorkOrder(
            goal="test",
            steps=[StepDef(name="s1", head_id="llm-a", required_kind="llm")],
        )
        result = normalize(wo, head_manager)
        assert result.steps[0].head_id == "llm-a"


class TestNormalizeReturnsCopy:
    def test_returns_copy(self, head_manager):
        s1 = StepDef(name="s1", head_id="llm-a")
        wo = WorkOrder(goal="test", steps=[s1])
        result = normalize(wo, head_manager)
        # Original should be unmodified
        assert wo.steps[0].fallback == []
        # Result should have fallback assigned
        assert len(result.steps[0].fallback) > 0
        # Different objects
        assert result is not wo
        assert result.steps[0] is not wo.steps[0]
