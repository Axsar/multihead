"""Tests for the multi-head consensus engine."""

from __future__ import annotations

import pytest

from multihead.consensus import (
    ConsensusConfig,
    ConsensusEngine,
    ConsensusResult,
    ConsensusStrategy,
    FirstToAheadConfig,
    HeadTask,
    VoteResult,
)
from multihead.head_manager import HeadManager
from multihead.models import AdapterKind, HeadManifest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def three_heads():
    """Three mock heads for voting tests."""
    manifests = {
        "head-a": HeadManifest(
            head_id="head-a", name="Head A", adapter=AdapterKind.MOCK,
            model="mock-v1", kind="llm", gpu_required=False,
        ),
        "head-b": HeadManifest(
            head_id="head-b", name="Head B", adapter=AdapterKind.MOCK,
            model="mock-v1", kind="llm", gpu_required=False,
        ),
        "head-c": HeadManifest(
            head_id="head-c", name="Head C", adapter=AdapterKind.MOCK,
            model="mock-v1", kind="llm", gpu_required=False,
        ),
    }
    return HeadManager(manifests)


@pytest.fixture
def engine(three_heads):
    return ConsensusEngine(three_heads)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestConsensusModels:
    def test_head_task_defaults(self):
        ht = HeadTask(head_id="test")
        assert ht.weight == 1.0
        assert ht.required is True
        assert ht.prompt_template == ""
        assert ht.extract_fields == []

    def test_consensus_config_defaults(self):
        cfg = ConsensusConfig(heads=[HeadTask(head_id="a")])
        assert cfg.strategy == ConsensusStrategy.MAJORITY
        assert cfg.threshold == 0.5
        assert cfg.cross_modal is False
        assert cfg.fail_on_disagreement is False

    def test_vote_result_success(self):
        vr = VoteResult(head_id="a", outputs={"text": "hello"})
        assert vr.success is True
        assert vr.schema_valid is True

    def test_vote_result_failure(self):
        vr = VoteResult(head_id="a", success=False, error="boom")
        assert vr.success is False

    def test_consensus_result_empty(self):
        cr = ConsensusResult()
        assert cr.agreement_score == 0.0
        assert cr.all_votes == []


# ---------------------------------------------------------------------------
# Majority vote
# ---------------------------------------------------------------------------


class TestMajorityVote:
    @pytest.mark.asyncio
    async def test_all_agree(self, engine, three_heads):
        """When all heads produce same output, agreement = 1.0."""
        # Mock all heads to return same text
        for hid in ["head-a", "head-b", "head-c"]:
            adapter = three_heads.get_adapter(hid)
            original = adapter.generate

            async def same_gen(prompt, _orig=original, **kw):
                return {"text": "consensus answer", "tokens_in": 10, "tokens_out": 5}

            adapter.generate = same_gen

        config = ConsensusConfig(
            heads=[
                HeadTask(head_id="head-a"),
                HeadTask(head_id="head-b"),
                HeadTask(head_id="head-c"),
            ],
            strategy=ConsensusStrategy.MAJORITY,
        )
        result = await engine.execute(config, "test prompt")
        assert result.agreement_score == 1.0
        assert result.consensus_outputs["text"] == "consensus answer"
        assert len(result.all_votes) == 3
        assert all(v.success for v in result.all_votes)

    @pytest.mark.asyncio
    async def test_two_vs_one(self, engine, three_heads):
        """2 out of 3 agree → winner is majority, agreement ≈ 0.67."""
        for hid in ["head-a", "head-b"]:
            adapter = three_heads.get_adapter(hid)

            async def agree_gen(prompt, **kw):
                return {"text": "agree", "tokens_in": 10, "tokens_out": 5}

            adapter.generate = agree_gen

        adapter_c = three_heads.get_adapter("head-c")

        async def disagree_gen(prompt, **kw):
            return {"text": "disagree", "tokens_in": 10, "tokens_out": 5}

        adapter_c.generate = disagree_gen

        config = ConsensusConfig(
            heads=[
                HeadTask(head_id="head-a"),
                HeadTask(head_id="head-b"),
                HeadTask(head_id="head-c"),
            ],
            strategy=ConsensusStrategy.MAJORITY,
        )
        result = await engine.execute(config, "test prompt")
        assert result.consensus_outputs["text"] == "agree"
        assert abs(result.agreement_score - 2 / 3) < 0.01

    @pytest.mark.asyncio
    async def test_default_mock_voting(self, engine):
        """With default mock adapter, each head returns unique output (call count differs)."""
        config = ConsensusConfig(
            heads=[
                HeadTask(head_id="head-a"),
                HeadTask(head_id="head-b"),
                HeadTask(head_id="head-c"),
            ],
            strategy=ConsensusStrategy.MAJORITY,
        )
        result = await engine.execute(config, "test prompt")
        # Mock adapter returns unique call # text, so no majority
        assert result.agreement_score > 0
        assert len(result.all_votes) == 3
        assert result.metrics["successful_heads"] == 3


# ---------------------------------------------------------------------------
# Weighted vote
# ---------------------------------------------------------------------------


class TestWeightedVote:
    @pytest.mark.asyncio
    async def test_high_weight_wins(self, engine, three_heads):
        """High-weight head breaks a tie."""
        # A and B disagree, C agrees with A but A has high weight
        for hid, text in [("head-a", "answer_a"), ("head-b", "answer_b"), ("head-c", "answer_b")]:
            adapter = three_heads.get_adapter(hid)
            _text = text

            async def gen(prompt, _t=_text, **kw):
                return {"text": _t, "tokens_in": 10, "tokens_out": 5}

            adapter.generate = gen

        config = ConsensusConfig(
            heads=[
                HeadTask(head_id="head-a", weight=5.0),  # Heavy weight
                HeadTask(head_id="head-b", weight=1.0),
                HeadTask(head_id="head-c", weight=1.0),
            ],
            strategy=ConsensusStrategy.WEIGHTED,
        )
        result = await engine.execute(config, "test")
        # A has weight 5, B+C have weight 2 total → A wins
        assert result.consensus_outputs["text"] == "answer_a"
        assert result.agreement_score > 0.5

    @pytest.mark.asyncio
    async def test_equal_weights_like_majority(self, engine, three_heads):
        """Equal weights should behave like majority vote."""
        for hid, text in [("head-a", "same"), ("head-b", "same"), ("head-c", "diff")]:
            adapter = three_heads.get_adapter(hid)
            _text = text

            async def gen(prompt, _t=_text, **kw):
                return {"text": _t, "tokens_in": 10, "tokens_out": 5}

            adapter.generate = gen

        config = ConsensusConfig(
            heads=[
                HeadTask(head_id="head-a", weight=1.0),
                HeadTask(head_id="head-b", weight=1.0),
                HeadTask(head_id="head-c", weight=1.0),
            ],
            strategy=ConsensusStrategy.WEIGHTED,
        )
        result = await engine.execute(config, "test")
        assert result.consensus_outputs["text"] == "same"


# ---------------------------------------------------------------------------
# Unanimous
# ---------------------------------------------------------------------------


class TestUnanimous:
    @pytest.mark.asyncio
    async def test_all_agree_perfect(self, engine, three_heads):
        """All heads agree → score = 1.0."""
        for hid in ["head-a", "head-b", "head-c"]:
            adapter = three_heads.get_adapter(hid)

            async def gen(prompt, **kw):
                return {"text": "unanimous", "tokens_in": 10, "tokens_out": 5}

            adapter.generate = gen

        config = ConsensusConfig(
            heads=[
                HeadTask(head_id="head-a"),
                HeadTask(head_id="head-b"),
                HeadTask(head_id="head-c"),
            ],
            strategy=ConsensusStrategy.UNANIMOUS,
        )
        result = await engine.execute(config, "test")
        assert result.agreement_score == 1.0
        assert result.consensus_outputs["text"] == "unanimous"

    @pytest.mark.asyncio
    async def test_one_disagrees(self, engine, three_heads):
        """One head disagrees → score < 1.0."""
        for hid, text in [("head-a", "yes"), ("head-b", "yes"), ("head-c", "no")]:
            adapter = three_heads.get_adapter(hid)
            _text = text

            async def gen(prompt, _t=_text, **kw):
                return {"text": _t, "tokens_in": 10, "tokens_out": 5}

            adapter.generate = gen

        config = ConsensusConfig(
            heads=[
                HeadTask(head_id="head-a"),
                HeadTask(head_id="head-b"),
                HeadTask(head_id="head-c"),
            ],
            strategy=ConsensusStrategy.UNANIMOUS,
        )
        result = await engine.execute(config, "test")
        assert result.agreement_score < 1.0
        # Still returns majority as best guess
        assert result.consensus_outputs["text"] == "yes"


# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------


class TestThreshold:
    @pytest.mark.asyncio
    async def test_above_threshold(self, engine, three_heads):
        """Agreement above threshold → no red flag."""
        for hid, text in [("head-a", "yes"), ("head-b", "yes"), ("head-c", "no")]:
            adapter = three_heads.get_adapter(hid)
            _text = text

            async def gen(prompt, _t=_text, **kw):
                return {"text": _t, "tokens_in": 10, "tokens_out": 5}

            adapter.generate = gen

        config = ConsensusConfig(
            heads=[
                HeadTask(head_id="head-a"),
                HeadTask(head_id="head-b"),
                HeadTask(head_id="head-c"),
            ],
            strategy=ConsensusStrategy.THRESHOLD,
            threshold=0.6,
        )
        result = await engine.execute(config, "test")
        assert result.agreement_score >= 0.6
        low_agreement_flags = [f for f in result.red_flags if f["type"] == "low_agreement"]
        assert len(low_agreement_flags) == 0

    @pytest.mark.asyncio
    async def test_below_threshold(self, engine, three_heads):
        """Agreement below threshold → red flag."""
        for hid, text in [("head-a", "a"), ("head-b", "b"), ("head-c", "c")]:
            adapter = three_heads.get_adapter(hid)
            _text = text

            async def gen(prompt, _t=_text, **kw):
                return {"text": _t, "tokens_in": 10, "tokens_out": 5}

            adapter.generate = gen

        config = ConsensusConfig(
            heads=[
                HeadTask(head_id="head-a"),
                HeadTask(head_id="head-b"),
                HeadTask(head_id="head-c"),
            ],
            strategy=ConsensusStrategy.THRESHOLD,
            threshold=0.8,
        )
        result = await engine.execute(config, "test")
        assert result.agreement_score < 0.8
        low_flags = [f for f in result.red_flags if f["type"] == "low_agreement"]
        assert len(low_flags) == 1
        assert low_flags[0]["severity"] == "high"


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    @pytest.mark.asyncio
    async def test_valid_json_object(self, engine, three_heads):
        """Valid JSON output matching schema should pass."""
        import json

        for hid in ["head-a", "head-b", "head-c"]:
            adapter = three_heads.get_adapter(hid)

            async def gen(prompt, **kw):
                return {
                    "text": json.dumps({"sentiment": "positive"}),
                    "tokens_in": 10, "tokens_out": 5,
                }

            adapter.generate = gen

        config = ConsensusConfig(
            heads=[
                HeadTask(head_id="head-a"),
                HeadTask(head_id="head-b"),
                HeadTask(head_id="head-c"),
            ],
            output_schema={"type": "object", "required": ["sentiment"]},
        )
        result = await engine.execute(config, "test")
        assert all(v.schema_valid for v in result.all_votes)
        schema_flags = [f for f in result.red_flags if f["type"] == "schema_violation"]
        assert len(schema_flags) == 0

    @pytest.mark.asyncio
    async def test_invalid_json_flagged(self, engine, three_heads):
        """Non-JSON output when schema expects object → schema violation flag."""
        for hid in ["head-a", "head-b", "head-c"]:
            adapter = three_heads.get_adapter(hid)

            async def gen(prompt, **kw):
                return {"text": "not json at all", "tokens_in": 10, "tokens_out": 5}

            adapter.generate = gen

        config = ConsensusConfig(
            heads=[
                HeadTask(head_id="head-a"),
                HeadTask(head_id="head-b"),
                HeadTask(head_id="head-c"),
            ],
            output_schema={"type": "object"},
        )
        result = await engine.execute(config, "test")
        assert not all(v.schema_valid for v in result.all_votes)
        schema_flags = [f for f in result.red_flags if f["type"] == "schema_violation"]
        assert len(schema_flags) > 0

    @pytest.mark.asyncio
    async def test_missing_required_field(self, engine, three_heads):
        """JSON missing required field → schema violation."""
        import json

        for hid in ["head-a", "head-b", "head-c"]:
            adapter = three_heads.get_adapter(hid)

            async def gen(prompt, **kw):
                return {"text": json.dumps({"other": "field"}), "tokens_in": 10, "tokens_out": 5}

            adapter.generate = gen

        config = ConsensusConfig(
            heads=[
                HeadTask(head_id="head-a"),
                HeadTask(head_id="head-b"),
                HeadTask(head_id="head-c"),
            ],
            output_schema={"type": "object", "required": ["sentiment"]},
        )
        result = await engine.execute(config, "test")
        assert any(not v.schema_valid for v in result.all_votes)


# ---------------------------------------------------------------------------
# Head failures and red flags
# ---------------------------------------------------------------------------


class TestRedFlags:
    @pytest.mark.asyncio
    async def test_required_head_failure(self, engine, three_heads):
        """Required head failing → critical red flag."""
        adapter = three_heads.get_adapter("head-a")

        async def failing_gen(prompt, **kw):
            raise RuntimeError("GPU OOM")

        adapter.generate = failing_gen

        config = ConsensusConfig(
            heads=[
                HeadTask(head_id="head-a", required=True),
                HeadTask(head_id="head-b"),
            ],
        )
        result = await engine.execute(config, "test")
        failure_flags = [f for f in result.red_flags if f["type"] == "head_failure"]
        assert len(failure_flags) == 1
        assert failure_flags[0]["severity"] == "critical"

    @pytest.mark.asyncio
    async def test_optional_head_failure(self, engine, three_heads):
        """Optional head failing → medium severity flag."""
        adapter = three_heads.get_adapter("head-a")

        async def failing_gen(prompt, **kw):
            raise RuntimeError("timeout")

        adapter.generate = failing_gen

        config = ConsensusConfig(
            heads=[
                HeadTask(head_id="head-a", required=False),
                HeadTask(head_id="head-b"),
                HeadTask(head_id="head-c"),
            ],
        )
        result = await engine.execute(config, "test")
        failure_flags = [f for f in result.red_flags if f["type"] == "head_failure"]
        assert len(failure_flags) == 1
        assert failure_flags[0]["severity"] == "medium"
        # Consensus should still work with remaining 2 heads
        assert result.metrics["successful_heads"] == 2

    @pytest.mark.asyncio
    async def test_all_heads_fail(self, engine, three_heads):
        """All heads failing → critical no_valid_votes flag."""
        for hid in ["head-a", "head-b", "head-c"]:
            adapter = three_heads.get_adapter(hid)

            async def failing_gen(prompt, **kw):
                raise RuntimeError("all broken")

            adapter.generate = failing_gen

        config = ConsensusConfig(
            heads=[
                HeadTask(head_id="head-a"),
                HeadTask(head_id="head-b"),
                HeadTask(head_id="head-c"),
            ],
        )
        result = await engine.execute(config, "test")
        assert result.agreement_score == 0.0
        no_votes_flags = [f for f in result.red_flags if f["type"] == "no_valid_votes"]
        assert len(no_votes_flags) == 1


# ---------------------------------------------------------------------------
# Cross-modal verification
# ---------------------------------------------------------------------------


class TestCrossModal:
    @pytest.mark.asyncio
    async def test_matching_fields_no_conflict(self, engine, three_heads):
        """Cross-modal heads that agree on extracted fields → no conflict."""
        import json

        adapter_a = three_heads.get_adapter("head-a")

        async def gen_a(prompt, **kw):
            return {"text": json.dumps({"count": 3}), "tokens_in": 10, "tokens_out": 5}

        adapter_a.generate = gen_a

        adapter_b = three_heads.get_adapter("head-b")

        async def gen_b(prompt, **kw):
            return {"text": json.dumps({"count": 3}), "tokens_in": 10, "tokens_out": 5}

        adapter_b.generate = gen_b

        config = ConsensusConfig(
            heads=[
                HeadTask(
                    head_id="head-a",
                    prompt_template="Detect objects",
                    extract_fields=["count"],
                ),
                HeadTask(
                    head_id="head-b",
                    prompt_template="Describe scene",
                    extract_fields=["count"],
                ),
            ],
            cross_modal=True,
        )
        result = await engine.execute(config, "test")
        cross_flags = [f for f in result.red_flags if f["type"] == "cross_modal_conflict"]
        assert len(cross_flags) == 0

    @pytest.mark.asyncio
    async def test_mismatching_fields_conflict(self, engine, three_heads):
        """Cross-modal heads that disagree → cross_modal_conflict flag."""
        import json

        adapter_a = three_heads.get_adapter("head-a")

        async def gen_a(prompt, **kw):
            return {"text": json.dumps({"count": 3}), "tokens_in": 10, "tokens_out": 5}

        adapter_a.generate = gen_a

        adapter_b = three_heads.get_adapter("head-b")

        async def gen_b(prompt, **kw):
            return {"text": json.dumps({"count": 5}), "tokens_in": 10, "tokens_out": 5}

        adapter_b.generate = gen_b

        config = ConsensusConfig(
            heads=[
                HeadTask(head_id="head-a", prompt_template="YOLO detect", extract_fields=["count"]),
                HeadTask(
                    head_id="head-b",
                    prompt_template="VLM describe",
                    extract_fields=["count"],
                ),
            ],
            cross_modal=True,
        )
        result = await engine.execute(config, "test")
        cross_flags = [f for f in result.red_flags if f["type"] == "cross_modal_conflict"]
        assert len(cross_flags) == 1
        assert cross_flags[0]["severity"] == "high"
        assert "count" in cross_flags[0]["message"]

    @pytest.mark.asyncio
    async def test_cross_modal_custom_prompts(self, engine, three_heads):
        """Each head gets its own prompt_template in cross-modal mode."""
        prompts_received = []

        for hid in ["head-a", "head-b"]:
            adapter = three_heads.get_adapter(hid)

            async def gen(prompt, _hid=hid, **kw):
                prompts_received.append((hid, prompt))
                return {"text": "ok", "tokens_in": 10, "tokens_out": 5}

            adapter.generate = gen

        config = ConsensusConfig(
            heads=[
                HeadTask(head_id="head-a", prompt_template="Detect objects in image"),
                HeadTask(head_id="head-b", prompt_template="Describe the scene"),
            ],
            cross_modal=True,
        )
        await engine.execute(config, "base prompt ignored")
        # Each head should get its custom prompt, not the base prompt
        assert any("Detect objects" in p for _, p in prompts_received)
        assert any("Describe the scene" in p for _, p in prompts_received)

    @pytest.mark.asyncio
    async def test_cross_modal_non_overlapping_fields(self, engine, three_heads):
        """Non-overlapping extract_fields shouldn't conflict."""
        import json

        adapter_a = three_heads.get_adapter("head-a")

        async def gen_a(prompt, **kw):
            return {"text": json.dumps({"objects": ["face"]}), "tokens_in": 10, "tokens_out": 5}

        adapter_a.generate = gen_a

        adapter_b = three_heads.get_adapter("head-b")

        async def gen_b(prompt, **kw):
            return {"text": json.dumps({"text_content": "hello"}), "tokens_in": 10, "tokens_out": 5}

        adapter_b.generate = gen_b

        config = ConsensusConfig(
            heads=[
                HeadTask(head_id="head-a", extract_fields=["objects"]),
                HeadTask(head_id="head-b", extract_fields=["text_content"]),
            ],
            cross_modal=True,
        )
        result = await engine.execute(config, "test")
        cross_flags = [f for f in result.red_flags if f["type"] == "cross_modal_conflict"]
        assert len(cross_flags) == 0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestConsensusMetrics:
    @pytest.mark.asyncio
    async def test_metrics_populated(self, engine):
        config = ConsensusConfig(
            heads=[HeadTask(head_id="head-a"), HeadTask(head_id="head-b")],
        )
        result = await engine.execute(config, "test")
        assert result.metrics["total_heads"] == 2
        assert result.metrics["successful_heads"] == 2
        assert result.metrics["total_latency_ms"] > 0
        assert result.metrics["avg_latency_ms"] > 0

    @pytest.mark.asyncio
    async def test_strategy_recorded(self, engine):
        config = ConsensusConfig(
            heads=[HeadTask(head_id="head-a")],
            strategy=ConsensusStrategy.WEIGHTED,
        )
        result = await engine.execute(config, "test")
        assert result.strategy_used == ConsensusStrategy.WEIGHTED

    @pytest.mark.asyncio
    async def test_observability_metrics_recorded(self, three_heads):
        """ConsensusEngine with MetricsCollector tracks execution metrics."""
        from multihead.observability import MetricsCollector

        metrics = MetricsCollector()
        engine = ConsensusEngine(three_heads, metrics=metrics)

        config = ConsensusConfig(
            heads=[HeadTask(head_id="head-a"), HeadTask(head_id="head-b")],
            strategy=ConsensusStrategy.MAJORITY,
        )
        await engine.execute(config, "test")

        assert metrics.counter("consensus_executions_total") >= 1
        assert metrics.counter(
            "consensus_executions_total", labels={"strategy": "majority"},
        ) >= 1
        hist = metrics.histogram("consensus_agreement_score")
        assert hist["count"] >= 1
        latency_hist = metrics.histogram("consensus_latency_ms")
        assert latency_hist["count"] >= 1

    @pytest.mark.asyncio
    async def test_no_metrics_without_collector(self, three_heads):
        """ConsensusEngine without MetricsCollector doesn't crash."""
        engine = ConsensusEngine(three_heads, metrics=None)
        config = ConsensusConfig(
            heads=[HeadTask(head_id="head-a")],
        )
        result = await engine.execute(config, "test")
        assert result.agreement_score > 0


# ---------------------------------------------------------------------------
# FIRST_TO_AHEAD voting
# ---------------------------------------------------------------------------


class TestFirstToAhead:
    @pytest.mark.asyncio
    async def test_basic_convergence(self, engine, three_heads):
        """All heads agree → resolves in min_samples (3)."""
        for hid in ["head-a", "head-b", "head-c"]:
            adapter = three_heads.get_adapter(hid)

            async def same_gen(prompt, **kw):
                return {"text": "consensus answer", "tokens_in": 10, "tokens_out": 5}

            adapter.generate = same_gen

        config = ConsensusConfig(
            heads=[
                HeadTask(head_id="head-a"),
                HeadTask(head_id="head-b"),
                HeadTask(head_id="head-c"),
            ],
            strategy=ConsensusStrategy.FIRST_TO_AHEAD,
            first_to_ahead=FirstToAheadConfig(k_margin=3, min_samples=3, max_samples=25),
        )
        result = await engine.execute(config, "test prompt")
        assert result.consensus_outputs["text"] == "consensus answer"
        assert result.strategy_used == ConsensusStrategy.FIRST_TO_AHEAD
        # All agreed, so leader count should be >= k_margin
        assert result.metrics["leader_count"] >= 3
        assert result.metrics["fta_exhausted"] is False

    @pytest.mark.asyncio
    async def test_k_margin_needed(self, engine, three_heads):
        """2 heads agree, 1 disagrees → takes more than 3 samples."""
        call_count = 0

        for hid in ["head-a", "head-b"]:
            adapter = three_heads.get_adapter(hid)

            async def agree_gen(prompt, **kw):
                return {"text": "agree", "tokens_in": 10, "tokens_out": 5}

            adapter.generate = agree_gen

        adapter_c = three_heads.get_adapter("head-c")

        async def disagree_gen(prompt, **kw):
            return {"text": "disagree", "tokens_in": 10, "tokens_out": 5}

        adapter_c.generate = disagree_gen

        config = ConsensusConfig(
            heads=[
                HeadTask(head_id="head-a"),
                HeadTask(head_id="head-b"),
                HeadTask(head_id="head-c"),
            ],
            strategy=ConsensusStrategy.FIRST_TO_AHEAD,
            first_to_ahead=FirstToAheadConfig(k_margin=3, min_samples=3, max_samples=25),
        )
        result = await engine.execute(config, "test")
        assert result.consensus_outputs["text"] == "agree"
        # Should have sampled more than 3 times
        assert result.metrics["total_samples"] > 3
        assert result.metrics["unique_buckets"] == 2

    @pytest.mark.asyncio
    async def test_red_flag_discard(self, engine, three_heads):
        """Long outputs are discarded and don't count as votes."""
        for hid in ["head-a", "head-b"]:
            adapter = three_heads.get_adapter(hid)

            async def short_gen(prompt, **kw):
                return {"text": "short answer", "tokens_in": 10, "tokens_out": 5}

            adapter.generate = short_gen

        # head-c returns very long output
        adapter_c = three_heads.get_adapter("head-c")

        async def long_gen(prompt, **kw):
            return {"text": " ".join(["word"] * 2000), "tokens_in": 10, "tokens_out": 2000}

        adapter_c.generate = long_gen

        config = ConsensusConfig(
            heads=[
                HeadTask(head_id="head-a"),
                HeadTask(head_id="head-b"),
                HeadTask(head_id="head-c"),
            ],
            strategy=ConsensusStrategy.FIRST_TO_AHEAD,
            first_to_ahead=FirstToAheadConfig(
                k_margin=3, min_samples=3, max_samples=25,
                red_flag_max_tokens=700,
            ),
        )
        result = await engine.execute(config, "test")
        assert result.consensus_outputs["text"] == "short answer"
        assert result.metrics["discarded_samples"] > 0
        assert "too_long" in result.metrics["discard_reasons"]

    @pytest.mark.asyncio
    async def test_max_samples_exhaustion(self, engine, three_heads):
        """No convergence → returns argmax + fta_exhausted flag."""
        # Each head returns different answer
        for hid, text in [("head-a", "a"), ("head-b", "b"), ("head-c", "c")]:
            adapter = three_heads.get_adapter(hid)
            _text = text

            async def gen(prompt, _t=_text, **kw):
                return {"text": _t, "tokens_in": 10, "tokens_out": 5}

            adapter.generate = gen

        config = ConsensusConfig(
            heads=[
                HeadTask(head_id="head-a"),
                HeadTask(head_id="head-b"),
                HeadTask(head_id="head-c"),
            ],
            strategy=ConsensusStrategy.FIRST_TO_AHEAD,
            first_to_ahead=FirstToAheadConfig(k_margin=3, min_samples=3, max_samples=9),
        )
        result = await engine.execute(config, "test")
        assert result.metrics["fta_exhausted"] is True
        # Should have fta_exhausted red flag
        exhausted_flags = [f for f in result.red_flags if f["type"] == "fta_exhausted"]
        assert len(exhausted_flags) == 1
        assert exhausted_flags[0]["severity"] == "medium"

    @pytest.mark.asyncio
    async def test_stall_escalation(self, engine, three_heads):
        """Escalation triggers after stall_threshold."""
        temps_used = []

        for hid in ["head-a", "head-b", "head-c"]:
            adapter = three_heads.get_adapter(hid)

            async def gen(prompt, _hid=hid, **kw):
                temps_used.append(kw.get("temperature", None))
                return {"text": _hid, "tokens_in": 10, "tokens_out": 5}

            adapter.generate = gen

        config = ConsensusConfig(
            heads=[
                HeadTask(head_id="head-a"),
                HeadTask(head_id="head-b"),
                HeadTask(head_id="head-c"),
            ],
            strategy=ConsensusStrategy.FIRST_TO_AHEAD,
            first_to_ahead=FirstToAheadConfig(
                k_margin=10, min_samples=3, max_samples=15,
                stall_threshold=5,
            ),
        )
        result = await engine.execute(config, "test")
        # Should have escalated (some temps at 0.3)
        assert result.metrics["fta_escalated"] is True
        assert 0.3 in temps_used

    def test_canonical_hash_json(self):
        """Different key order in JSON → same hash."""
        h1 = ConsensusEngine._canonical_hash('{"a":1,"b":2}')
        h2 = ConsensusEngine._canonical_hash('{"b":2,"a":1}')
        assert h1 == h2

    def test_canonical_hash_text(self):
        """Whitespace/case normalized for non-JSON."""
        h1 = ConsensusEngine._canonical_hash("  Hello World  ")
        h2 = ConsensusEngine._canonical_hash("hello world")
        assert h1 == h2


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


class TestExports:
    def test_init_exports(self):
        """Consensus classes should be importable from multihead package."""
        from multihead import (
            ConsensusConfig,
            ConsensusEngine,
            ConsensusResult,
            ConsensusStrategy,
            FirstToAheadConfig,
            HeadTask,
            VoteResult,
        )
        assert ConsensusStrategy.MAJORITY.value == "majority"
        assert ConsensusStrategy.FIRST_TO_AHEAD.value == "first_to_ahead"
        assert ConsensusConfig is not None
        assert ConsensusEngine is not None
        assert ConsensusResult is not None
        assert FirstToAheadConfig is not None
        assert HeadTask is not None
        assert VoteResult is not None
