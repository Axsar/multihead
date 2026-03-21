"""Tests for Phase 3: BotVibes Marketplace Integration.

Covers:
- BotVibesAdapter RFQ procurement cycle
- Privacy enforcement (defense-in-depth)
- Retry logic with exponential backoff
- Cost tracking
- Knowledge store feedback
- Provider discovery cache in Router
- Kind inference from capabilities
- Env var resolution in config
- Plan normalizer marketplace fallback
"""

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from multihead.adapters.botvibes_adapter import (
    BotVibesAdapter,
    PrivacyViolation,
)
from multihead.models import (
    AdapterKind,
    BudgetConstraint,
    Capability,
    DataSensitivity,
    HeadManifest,
    PrivacyConstraint,
)
from multihead.router import Router


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def botvibes_manifest():
    """Standard BotVibes provider manifest."""
    return HeadManifest(
        head_id="botvibes-test-provider",
        name="Test BotVibes Provider",
        adapter=AdapterKind.BOTVIBES,
        model="test-model-v1",
        kind="llm",
        endpoint="http://localhost:8000/api/v1",
        gpu_required=False,
        is_local=False,
        privacy_level="encrypted",
        extra={
            "api_key": "test-token-12345",
            "project_id": "test-project-123",
            "target_capability": "ai.text.generate.v1",
            "target_agent_id": "test-agent-456",
        },
        capabilities=Capability(
            solver_type="llm",
            input_modalities=["text"],
            output_modalities=["text"],
            task_types=["text_generation"],
            latency_p50_ms=1000,
            cost_per_call=0.50,
        ),
    )


@pytest.fixture
def rfq_manifest():
    """BotVibes manifest with RFQ mode enabled."""
    return HeadManifest(
        head_id="botvibes-rfq-provider",
        name="RFQ Provider",
        adapter=AdapterKind.BOTVIBES,
        model="rfq-model-v1",
        kind="llm",
        endpoint="http://localhost:8000/api/v1",
        gpu_required=False,
        is_local=False,
        extra={
            "api_key": "test-token-rfq",
            "project_id": "proj-rfq",
            "target_capability": "code.review.v1",
            "use_rfq": True,
        },
        capabilities=Capability(
            solver_type="llm",
            input_modalities=["text", "code"],
            output_modalities=["text"],
            task_types=["code_review"],
            cost_per_call=3.00,
        ),
    )


def _mock_response(status_code=200, json_data=None):
    """Create a mock httpx.Response."""
    resp = Mock()
    resp.status_code = status_code
    resp.json = Mock(return_value=json_data or {})
    resp.raise_for_status = Mock()
    resp.text = str(json_data)
    if status_code >= 400:
        import httpx
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}", request=Mock(), response=resp
        )
    return resp


# ── BotVibesAdapter: Init & Privacy ──────────────────────────


class TestBotVibesAdapterInit:
    """Test adapter initialization and config."""

    def test_init_valid(self, botvibes_manifest):
        adapter = BotVibesAdapter(botvibes_manifest)
        assert adapter.acp_url == "http://localhost:8000/api/v1"
        assert adapter.acp_token == "test-token-12345"
        assert adapter.target_capability == "ai.text.generate.v1"
        assert adapter.total_cost == 0.0
        assert adapter.call_count == 0

    def test_init_missing_url(self):
        m = HeadManifest(
            head_id="t", name="T", adapter=AdapterKind.BOTVIBES, model="t",
            extra={"api_key": "k", "target_capability": "c"},
        )
        with pytest.raises(ValueError, match="missing acp_url"):
            BotVibesAdapter(m)

    def test_init_missing_token(self):
        m = HeadManifest(
            head_id="t", name="T", adapter=AdapterKind.BOTVIBES, model="t",
            endpoint="http://x", extra={"target_capability": "c"},
        )
        with pytest.raises(ValueError, match="missing acp_token"):
            BotVibesAdapter(m)

    def test_init_missing_capability(self):
        m = HeadManifest(
            head_id="t", name="T", adapter=AdapterKind.BOTVIBES, model="t",
            endpoint="http://x", extra={"api_key": "k"},
        )
        with pytest.raises(ValueError, match="missing target_capability"):
            BotVibesAdapter(m)

    def test_url_normalization(self):
        """URL with trailing /api/v1 or bare host gets normalized."""
        for url in ["http://x/api/v1", "http://x/api/v1/", "http://x"]:
            m = HeadManifest(
                head_id="t", name="T", adapter=AdapterKind.BOTVIBES, model="t",
                endpoint=url, extra={"api_key": "k", "target_capability": "c"},
            )
            adapter = BotVibesAdapter(m)
            assert adapter.acp_url == "http://x/api/v1"

    def test_knowledge_store_passthrough(self, botvibes_manifest):
        ks = Mock()
        adapter = BotVibesAdapter(botvibes_manifest, knowledge_store=ks)
        assert adapter._knowledge_store is ks


class TestPrivacyEnforcement:
    """Test defense-in-depth privacy blocking."""

    def test_restricted_blocked(self, botvibes_manifest):
        adapter = BotVibesAdapter(botvibes_manifest)
        privacy = PrivacyConstraint(data_sensitivity=DataSensitivity.RESTRICTED)
        with pytest.raises(PrivacyViolation, match="RESTRICTED"):
            adapter._enforce_privacy(privacy)

    def test_confidential_blocked(self, botvibes_manifest):
        adapter = BotVibesAdapter(botvibes_manifest)
        privacy = PrivacyConstraint(data_sensitivity=DataSensitivity.CONFIDENTIAL)
        with pytest.raises(PrivacyViolation, match="CONFIDENTIAL"):
            adapter._enforce_privacy(privacy)

    def test_internal_allowed(self, botvibes_manifest):
        adapter = BotVibesAdapter(botvibes_manifest)
        privacy = PrivacyConstraint(data_sensitivity=DataSensitivity.INTERNAL)
        adapter._enforce_privacy(privacy)  # Should not raise

    def test_public_allowed(self, botvibes_manifest):
        adapter = BotVibesAdapter(botvibes_manifest)
        privacy = PrivacyConstraint(data_sensitivity=DataSensitivity.PUBLIC)
        adapter._enforce_privacy(privacy)  # Should not raise

    def test_none_allowed(self, botvibes_manifest):
        adapter = BotVibesAdapter(botvibes_manifest)
        adapter._enforce_privacy(None)  # Should not raise

    @pytest.mark.asyncio
    async def test_generate_blocks_confidential(self, botvibes_manifest):
        adapter = BotVibesAdapter(botvibes_manifest)
        privacy = PrivacyConstraint(data_sensitivity=DataSensitivity.CONFIDENTIAL)
        with pytest.raises(PrivacyViolation):
            await adapter.generate("test", privacy=privacy)


# ── BotVibesAdapter: Simple Mode ─────────────────────────────


class TestSimpleMode:
    """Test simple ACP task mode (default)."""

    @pytest.mark.asyncio
    async def test_generate_simple_success(self, botvibes_manifest):
        adapter = BotVibesAdapter(botvibes_manifest)

        create_resp = _mock_response(200, {"task_id": "task-001"})
        poll_resp = _mock_response(200, {
            "status": "complete",
            "output_ref": "Generated text result",
            "confidence": 0.95,
        })

        with patch("httpx.AsyncClient") as mock_cls:
            client = AsyncMock()
            client.request = AsyncMock(side_effect=[create_resp, poll_resp])
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = client

            result = await adapter.generate("Write a haiku")

        assert result["text"] == "Generated text result"
        assert result["provider_metadata"]["mode"] == "simple"
        assert adapter.call_count == 1
        assert adapter.total_cost == 0.50  # from capabilities.cost_per_call

    @pytest.mark.asyncio
    async def test_generate_simple_task_failure(self, botvibes_manifest):
        adapter = BotVibesAdapter(botvibes_manifest)

        create_resp = _mock_response(200, {"task_id": "task-fail"})
        poll_resp = _mock_response(200, {
            "status": "failed",
            "message": "Provider OOM",
        })

        with patch("httpx.AsyncClient") as mock_cls:
            client = AsyncMock()
            client.request = AsyncMock(side_effect=[create_resp, poll_resp])
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = client

            with pytest.raises(RuntimeError, match="BotVibes task failed"):
                await adapter.generate("bad prompt")

    @pytest.mark.asyncio
    async def test_load_unload_noops(self, botvibes_manifest):
        adapter = BotVibesAdapter(botvibes_manifest)
        await adapter.load()
        await adapter.unload()
        await adapter.sleep()
        await adapter.wake()


# ── BotVibesAdapter: RFQ Mode ────────────────────────────────


class TestRFQMode:
    """Test RFQ procurement cycle."""

    @pytest.mark.asyncio
    async def test_rfq_triggered_by_budget(self, botvibes_manifest):
        """Budget kwarg triggers RFQ mode."""
        adapter = BotVibesAdapter(botvibes_manifest)
        budget = BudgetConstraint(max_cost_per_step=5.0)

        # Mock the full RFQ cycle — use a list that returns appropriate
        # responses based on URL pattern
        rfq_resp = _mock_response(200, {"rfq_id": "rfq-001"})
        bids_resp = _mock_response(200, {"quotes": [
            {"quote_id": "q1", "agent_id": "provider-a", "unit_price": 2.0,
             "estimated_confidence": 0.9, "estimated_latency_ms": 5000},
        ]})
        award_resp = _mock_response(200, {"contract_id": "contract-001"})
        contract_resp = _mock_response(200, {
            "status": "delivered", "delivery_notes": "Review complete",
        })
        vault_entries_resp = _mock_response(200, {"entries": []})
        accept_resp = _mock_response(200, {})

        async def mock_request(method, url, **kwargs):
            if "rfqs" in url and method == "POST" and "accept" not in url and "cancel" not in url:
                return rfq_resp
            if "quotes" in url:
                return bids_resp
            if "accept" in url and "rfqs" in url:
                return award_resp
            if "contracts" in url and "accept" in url:
                return accept_resp
            if "contracts" in url and "entries" not in url:
                return contract_resp
            if "entries" in url:
                return vault_entries_resp
            return _mock_response(200, {})

        with patch("httpx.AsyncClient") as mock_cls:
            client = AsyncMock()
            client.request = AsyncMock(side_effect=mock_request)
            client.get = AsyncMock(return_value=_mock_response(404))
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = client

            result = await adapter.generate(
                "Review my code", budget=budget, bid_wait_s=0.5,
            )

        assert result["provider_metadata"]["mode"] == "rfq"
        assert result["provider_metadata"]["rfq_id"] == "rfq-001"
        assert result["provider_metadata"]["contract_id"] == "contract-001"
        assert result["provider_metadata"]["bid_price"] == 2.0
        assert adapter.total_cost == 2.0

    @pytest.mark.asyncio
    async def test_rfq_triggered_by_use_rfq_flag(self, rfq_manifest):
        """use_rfq in manifest.extra triggers RFQ mode."""
        adapter = BotVibesAdapter(rfq_manifest)
        assert adapter._use_rfq is True

    @pytest.mark.asyncio
    async def test_rfq_no_bids_falls_back_to_simple(self, botvibes_manifest):
        """When RFQ gets no bids, fall back to simple task mode."""
        adapter = BotVibesAdapter(botvibes_manifest)
        budget = BudgetConstraint(max_cost_per_step=5.0)

        rfq_resp = _mock_response(200, {"rfq_id": "rfq-empty"})
        no_bids_resp = _mock_response(200, {"quotes": []})
        cancel_resp = _mock_response(200, {})
        create_resp = _mock_response(200, {"task_id": "task-fallback"})
        complete_resp = _mock_response(200, {
            "status": "complete", "output_ref": "Fallback result",
        })

        async def mock_request(method, url, **kwargs):
            if "rfqs" in url and method == "POST" and "cancel" not in url and "accept" not in url:
                return rfq_resp
            if "quotes" in url:
                return no_bids_resp
            if "cancel" in url:
                return cancel_resp
            if "tasks" in url and method == "POST":
                return create_resp
            if "tasks" in url and method == "GET":
                return complete_resp
            return _mock_response(200, {})

        with patch("httpx.AsyncClient") as mock_cls:
            client = AsyncMock()
            client.request = AsyncMock(side_effect=mock_request)
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = client

            result = await adapter.generate(
                "Review code", budget=budget, bid_wait_s=0.5,
            )

        assert result["text"] == "Fallback result"
        assert result["provider_metadata"]["mode"] == "simple"


class TestBidScoring:
    """Test bid scoring logic."""

    def test_single_bid_returned(self, botvibes_manifest):
        adapter = BotVibesAdapter(botvibes_manifest)
        bids = [{"quote_id": "q1", "unit_price": 1.0, "estimated_confidence": 0.9}]
        assert adapter._score_bids(bids) == bids[0]

    def test_cheaper_bid_preferred(self, botvibes_manifest):
        adapter = BotVibesAdapter(botvibes_manifest)
        bids = [
            {"quote_id": "q1", "unit_price": 5.0,
             "estimated_confidence": 0.9,
             "estimated_latency_ms": 5000},
            {"quote_id": "q2", "unit_price": 1.0,
             "estimated_confidence": 0.9,
             "estimated_latency_ms": 5000},
        ]
        best = adapter._score_bids(bids)
        assert best["quote_id"] == "q2"  # Cheaper wins

    def test_higher_confidence_can_win(self, botvibes_manifest):
        adapter = BotVibesAdapter(botvibes_manifest)
        bids = [
            {"quote_id": "q1", "unit_price": 2.0,
             "estimated_confidence": 0.99,
             "estimated_latency_ms": 1000},
            {"quote_id": "q2", "unit_price": 1.9,
             "estimated_confidence": 0.50,
             "estimated_latency_ms": 50000},
        ]
        best = adapter._score_bids(bids)
        assert best["quote_id"] == "q1"  # Higher confidence + lower latency wins


# ── BotVibesAdapter: Retry & Token Refresh ───────────────────


class TestRetryLogic:
    """Test HTTP retry with backoff."""

    @pytest.mark.asyncio
    async def test_retry_on_503(self, botvibes_manifest):
        adapter = BotVibesAdapter(botvibes_manifest)

        resp_503 = _mock_response(503)
        resp_200 = _mock_response(200, {"ok": True})

        client = AsyncMock()
        client.request = AsyncMock(side_effect=[resp_503, resp_200])

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter._request_with_retry(
                client, "GET", "http://test/api", max_retries=2,
            )

        assert result.status_code == 200
        assert client.request.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_exhausted(self, botvibes_manifest):
        adapter = BotVibesAdapter(botvibes_manifest)

        resp_503 = _mock_response(503)
        client = AsyncMock()
        client.request = AsyncMock(return_value=resp_503)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter._request_with_retry(
                client, "GET", "http://test/api", max_retries=2,
            )

        # Returns the 503 after retries exhausted
        assert result.status_code == 503
        assert client.request.call_count == 3  # 1 + 2 retries

    @pytest.mark.asyncio
    async def test_token_refresh_on_401(self, botvibes_manifest):
        adapter = BotVibesAdapter(botvibes_manifest)

        resp_401 = _mock_response(401)
        resp_200 = _mock_response(200, {"ok": True})
        refresh_resp = _mock_response(200, {"access_token": "new-token"})

        client = AsyncMock()
        # First call returns 401, refresh succeeds, retry succeeds
        client.request = AsyncMock(side_effect=[resp_401, resp_200])
        client.post = AsyncMock(return_value=refresh_resp)

        result = await adapter._request_with_retry(
            client, "GET", "http://test/api", headers=adapter._auth_headers(),
        )

        assert result.status_code == 200
        assert adapter.acp_token == "new-token"


# ── BotVibesAdapter: Knowledge Store Feedback ────────────────


class TestKnowledgeFeedback:

    def test_record_success(self, botvibes_manifest):
        ks = Mock()
        ks.insert_claim = Mock()
        adapter = BotVibesAdapter(botvibes_manifest, knowledge_store=ks)
        adapter._record_to_knowledge(True, 1500, 0.50)
        ks.insert_claim.assert_called_once()

    def test_record_no_knowledge_store(self, botvibes_manifest):
        adapter = BotVibesAdapter(botvibes_manifest)
        # Should not raise
        adapter._record_to_knowledge(True, 1500, 0.50)

    def test_record_failure_logged(self, botvibes_manifest):
        ks = Mock()
        ks.insert_claim = Mock(side_effect=Exception("DB error"))
        adapter = BotVibesAdapter(botvibes_manifest, knowledge_store=ks)
        # Should not raise even on error
        adapter._record_to_knowledge(False, 0, 0.0)


# ── Router: Provider Cache ───────────────────────────────────


class TestProviderCache:

    def _make_router(self):
        hm = Mock()
        hm.get_states.return_value = {}
        hm.manifests = {}
        return Router(hm)

    def test_cache_miss(self):
        router = self._make_router()
        assert router._get_cached_providers("visual_reasoning") is None

    def test_cache_hit(self):
        router = self._make_router()
        providers = [{"provider_id": "a"}]
        router._set_cached_providers("visual_reasoning", providers)
        assert router._get_cached_providers("visual_reasoning") == providers

    def test_cache_expired(self):
        router = self._make_router()
        router._provider_cache_ttl_s = 0.0  # Immediate expiry
        router._set_cached_providers("visual_reasoning", [{"provider_id": "a"}])
        # Force expiry
        import time
        time.sleep(0.01)
        assert router._get_cached_providers("visual_reasoning") is None

    def test_invalidate_single(self):
        router = self._make_router()
        router._set_cached_providers("a", [{"id": "1"}])
        router._set_cached_providers("b", [{"id": "2"}])
        router.invalidate_provider_cache("a")
        assert router._get_cached_providers("a") is None
        assert router._get_cached_providers("b") is not None

    def test_invalidate_all(self):
        router = self._make_router()
        router._set_cached_providers("a", [{"id": "1"}])
        router._set_cached_providers("b", [{"id": "2"}])
        router.invalidate_provider_cache()
        assert router._get_cached_providers("a") is None
        assert router._get_cached_providers("b") is None


# ── Router: Kind Inference ───────────────────────────────────


class TestKindInference:

    def test_visual_capabilities(self):
        assert Router._infer_kind_from_capabilities(["visual_reasoning"]) == "vlm"
        assert Router._infer_kind_from_capabilities(["image_classification"]) == "vlm"
        assert Router._infer_kind_from_capabilities(["image_description"]) == "vlm"

    def test_detection_capabilities(self):
        assert Router._infer_kind_from_capabilities(["object_detection"]) == "tool"
        assert Router._infer_kind_from_capabilities(["image.detect.objects.v1"]) == "tool"

    def test_segmentation_capabilities(self):
        assert Router._infer_kind_from_capabilities(["segmentation"]) == "tool"
        assert Router._infer_kind_from_capabilities(["image.segment.masks.v1"]) == "tool"

    def test_code_capabilities(self):
        assert Router._infer_kind_from_capabilities(["code.review.v1"]) == "llm"
        assert Router._infer_kind_from_capabilities(["code.generation.v1"]) == "llm"

    def test_fuzzy_visual_match(self):
        assert Router._infer_kind_from_capabilities(["my.image.analyzer"]) == "vlm"
        assert Router._infer_kind_from_capabilities(["vision_expert"]) == "vlm"

    def test_default_llm(self):
        assert Router._infer_kind_from_capabilities(["text_generation"]) == "llm"
        assert Router._infer_kind_from_capabilities(["unknown_cap"]) == "llm"
        assert Router._infer_kind_from_capabilities([]) == "llm"


# ── Config: Env Var Resolution ───────────────────────────────


class TestEnvVarResolution:

    def test_resolve_simple_var(self):
        from multihead.config import _resolve_env_vars
        with patch.dict(os.environ, {"MY_VAR": "hello"}):
            assert _resolve_env_vars("${MY_VAR}") == "hello"

    def test_resolve_in_dict(self):
        from multihead.config import _resolve_env_vars
        with patch.dict(os.environ, {"KEY": "value123"}):
            result = _resolve_env_vars({"api_key": "${KEY}", "other": "static"})
            assert result == {"api_key": "value123", "other": "static"}

    def test_resolve_in_list(self):
        from multihead.config import _resolve_env_vars
        with patch.dict(os.environ, {"A": "1"}):
            assert _resolve_env_vars(["${A}", "literal"]) == ["1", "literal"]

    def test_unresolvable_left_as_is(self):
        from multihead.config import _resolve_env_vars
        result = _resolve_env_vars("${NONEXISTENT_VAR_12345}")
        assert result == "${NONEXISTENT_VAR_12345}"

    def test_non_string_passthrough(self):
        from multihead.config import _resolve_env_vars
        assert _resolve_env_vars(42) == 42
        assert _resolve_env_vars(None) is None
        assert _resolve_env_vars(True) is True


# ── Plan Normalizer: Marketplace Fallback ────────────────────


class TestNormalizerMarketplaceFallback:

    def test_normalize_without_marketplace_works(self):
        """Normal normalize without marketplace flag still works."""
        from multihead.plan_normalizer import normalize
        from multihead.models import WorkOrder, StepDef

        hm = Mock()
        hm.get_states.return_value = {
            "mock-llm": {"kind": "llm", "head_id": "mock-llm", "name": "Mock",
                         "adapter": "mock", "model": "mock", "state": "off",
                         "is_active": False, "circuit_breaker": "closed"},
        }
        hm.get_manifest.return_value = HeadManifest(
            head_id="mock-llm", name="Mock", adapter=AdapterKind.MOCK,
            model="mock", kind="llm",
        )
        hm.manifests = {"mock-llm": hm.get_manifest.return_value}

        wo = WorkOrder(
            goal="test",
            steps=[StepDef(name="step1", head_id="mock-llm")],
        )
        result = normalize(wo, hm)
        assert result.steps[0].head_id == "mock-llm"

    def test_normalize_with_marketplace_flag_accepted(self):
        """enable_marketplace_fallback parameter is accepted."""
        from multihead.plan_normalizer import normalize
        from multihead.models import WorkOrder, StepDef

        hm = Mock()
        hm.get_states.return_value = {
            "mock-llm": {"kind": "llm", "head_id": "mock-llm", "name": "Mock",
                         "adapter": "mock", "model": "mock", "state": "off",
                         "is_active": False, "circuit_breaker": "closed"},
        }
        hm.get_manifest.return_value = HeadManifest(
            head_id="mock-llm", name="Mock", adapter=AdapterKind.MOCK,
            model="mock", kind="llm",
        )
        hm.manifests = {"mock-llm": hm.get_manifest.return_value}

        wo = WorkOrder(
            goal="test",
            steps=[StepDef(name="step1", head_id="mock-llm")],
        )
        # Should not raise with the new parameter
        result = normalize(wo, hm, enable_marketplace_fallback=True)
        assert result.steps[0].head_id == "mock-llm"


# ── Head Manager: Knowledge Store Wiring ─────────────────────


class TestHeadManagerBotVibesWiring:

    def test_botvibes_adapter_gets_knowledge_store(self):
        """BotVibesAdapter receives knowledge_store from HeadManager factory."""
        from multihead.head_manager import _create_adapter

        ks = Mock()
        manifest = HeadManifest(
            head_id="bv-test", name="BV", adapter=AdapterKind.BOTVIBES,
            model="test", endpoint="http://x",
            extra={"api_key": "k", "project_id": "p", "target_capability": "c"},
        )
        adapter = _create_adapter(manifest, knowledge_store=ks)
        assert isinstance(adapter, BotVibesAdapter)
        assert adapter._knowledge_store is ks

    def test_botvibes_adapter_works_without_knowledge_store(self):
        """BotVibesAdapter works without knowledge_store."""
        from multihead.head_manager import _create_adapter

        manifest = HeadManifest(
            head_id="bv-test", name="BV", adapter=AdapterKind.BOTVIBES,
            model="test", endpoint="http://x",
            extra={"api_key": "k", "project_id": "p", "target_capability": "c"},
        )
        adapter = _create_adapter(manifest)
        assert isinstance(adapter, BotVibesAdapter)
        assert adapter._knowledge_store is None


# ── Integration: Cost Tracking Across Calls ──────────────────


class TestCostTracking:

    @pytest.mark.asyncio
    async def test_cost_accumulates(self, botvibes_manifest):
        adapter = BotVibesAdapter(botvibes_manifest)

        create_resp = _mock_response(200, {"task_id": "t1"})
        poll_resp = _mock_response(200, {"status": "complete", "output_ref": "ok"})

        with patch("httpx.AsyncClient") as mock_cls:
            client = AsyncMock()
            client.request = AsyncMock(side_effect=[
                create_resp, poll_resp,  # Call 1
                create_resp, poll_resp,  # Call 2
            ])
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = client

            await adapter.generate("prompt 1")
            await adapter.generate("prompt 2")

        assert adapter.call_count == 2
        assert adapter.total_cost == 1.00  # 0.50 * 2


# ── Healthcheck ──────────────────────────────────────────────


class TestHealthcheck:

    @pytest.mark.asyncio
    async def test_healthy(self, botvibes_manifest):
        adapter = BotVibesAdapter(botvibes_manifest)

        with patch("httpx.AsyncClient") as mock_cls:
            client = AsyncMock()
            client.get = AsyncMock(return_value=_mock_response(200))
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = client

            assert await adapter.healthcheck() is True

    @pytest.mark.asyncio
    async def test_unhealthy(self, botvibes_manifest):
        adapter = BotVibesAdapter(botvibes_manifest)

        with patch("httpx.AsyncClient") as mock_cls:
            client = AsyncMock()
            client.get = AsyncMock(side_effect=Exception("conn refused"))
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = client

            assert await adapter.healthcheck() is False
