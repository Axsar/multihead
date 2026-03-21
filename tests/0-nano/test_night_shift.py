"""Tests for the Night Shift 15-stage pipeline (fast unit tests only)."""

from multihead.knowledge_models import (
    NightShiftConfig,
    Provenance,
)
from multihead.night_shift import NightShift, StageGate


def _prov() -> Provenance:
    return Provenance(produced_by={"kind": "test", "id": "unit"})


# -------------------------------------------------------------------
# Gate evaluation
# -------------------------------------------------------------------

class TestGateEvaluation:
    def test_no_conditions_is_accept(self):
        gate = StageGate()
        result = NightShift._evaluate_gate(gate, {})
        assert result == "accept"

    def test_condition_met_gte(self):
        gate = StageGate(accept_if=[{"metric": "score", "op": ">=", "value": 0.9}])
        result = NightShift._evaluate_gate(gate, {"score": 0.95})
        assert result == "accept"

    def test_condition_not_met_triggers_retry(self):
        gate = StageGate(
            accept_if=[{"metric": "score", "op": ">=", "value": 0.9}],
            retry={"max_attempts": 3},
        )
        result = NightShift._evaluate_gate(gate, {"score": 0.5})
        assert result == "retry"

    def test_condition_not_met_triggers_fallback(self):
        gate = StageGate(
            accept_if=[{"metric": "score", "op": ">=", "value": 0.9}],
            retry={"max_attempts": 1},
            fallback={"mode": "keyword_only"},
        )
        result = NightShift._evaluate_gate(gate, {"score": 0.5})
        assert result == "fallback"

    def test_condition_not_met_fail(self):
        gate = StageGate(
            accept_if=[{"metric": "score", "op": ">=", "value": 0.9}],
        )
        result = NightShift._evaluate_gate(gate, {"score": 0.5})
        assert result == "fail"

    def test_lte_operator(self):
        gate = StageGate(accept_if=[{"metric": "error_rate", "op": "<=", "value": 0.03}])
        assert NightShift._evaluate_gate(gate, {"error_rate": 0.01}) == "accept"
        gate2 = StageGate(accept_if=[{"metric": "error_rate", "op": "<=", "value": 0.03}])
        result = NightShift._evaluate_gate(gate2, {"error_rate": 0.1})
        assert result in ("retry", "fallback", "fail")

    def test_eq_operator(self):
        gate = StageGate(accept_if=[{"metric": "count", "op": "==", "value": 5}])
        assert NightShift._evaluate_gate(gate, {"count": 5}) == "accept"

    def test_missing_metric_defaults_to_zero(self):
        gate = StageGate(accept_if=[{"metric": "nonexistent", "op": ">=", "value": 1}])
        result = NightShift._evaluate_gate(gate, {})
        assert result in ("retry", "fallback", "fail")


# -------------------------------------------------------------------
# Stage definitions
# -------------------------------------------------------------------

class TestStageDefinitions:
    def test_stages_defined(self):
        from multihead.night_shift import STAGES
        assert len(STAGES) == 26  # 0-25 including file_path_anchoring stage

    def test_stage_ids_sequential(self):
        from multihead.night_shift import STAGES
        for i, stage in enumerate(STAGES):
            assert stage.stage_id == i

    def test_llm_stages_marked(self):
        from multihead.night_shift import STAGES
        llm_stages = [s for s in STAGES if s.requires_llm]
        assert len(llm_stages) >= 6  # stages 3-8, 10-12


# -------------------------------------------------------------------
# NightShiftConfig defaults
# -------------------------------------------------------------------

class TestNightShiftConfig:
    def test_defaults(self):
        config = NightShiftConfig(head_id="test")
        assert config.input_window_hours == 24
        assert config.auto_accept_confidence == 0.85
        assert config.auto_accept_min_supports == 2

    def test_custom_config(self):
        config = NightShiftConfig(
            head_id="custom",
            input_window_hours=48,
            auto_accept_confidence=0.9,
        )
        assert config.input_window_hours == 48
        assert config.auto_accept_confidence == 0.9
