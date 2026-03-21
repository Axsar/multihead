"""Tests for cloud marketplace bridge."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from multihead.cloud_marketplace import CloudMarketplaceBridge


@pytest.fixture
def head_manager():
    """Mock head manager with LLM and VLM heads."""
    hm = MagicMock()
    hm.get_states.return_value = {
        "core-llm": {"state": "OFF"},
        "vision-vlm": {"state": "OFF"},
    }

    llm_manifest = MagicMock()
    llm_manifest.kind = "llm"
    vlm_manifest = MagicMock()
    vlm_manifest.kind = "vlm"

    hm.manifests = {
        "core-llm": llm_manifest,
        "vision-vlm": vlm_manifest,
    }
    return hm


@pytest.fixture
def settings():
    return MagicMock()


@pytest.fixture
def runtime_config():
    svc = MagicMock()
    svc.cloud_marketplace = False
    svc.cloud_rfq_interval = 60
    svc.cloud_contract_interval = 30
    svc.cloud_auto_quote = True
    svc.cloud_max_contracts = 2
    rc = MagicMock()
    rc.services = svc
    return rc


@pytest.fixture
def bridge(head_manager, settings, runtime_config):
    return CloudMarketplaceBridge(
        head_manager=head_manager,
        settings=settings,
        cloud_url="https://cloud.example.com/api/v1",
        cloud_api_key="test-jwt-token",
        cloud_project_id="test-project-id",
        cloud_agent_id="test-agent",
        runtime_config=runtime_config,
    )


class TestInit:
    """Test initialization."""

    def test_basic_init(self, bridge):
        assert bridge._cloud_url == "https://cloud.example.com/api/v1"
        assert bridge._cloud_agent_id == "test-agent"
        assert bridge._cloud_api_key == "test-jwt-token"
        assert bridge._running is False

    def test_url_trailing_slash_stripped(self, head_manager, settings):
        b = CloudMarketplaceBridge(
            head_manager=head_manager,
            settings=settings,
            cloud_url="https://cloud.example.com/api/v1/",
            cloud_api_key="key",
            cloud_project_id="proj",
        )
        assert b._cloud_url == "https://cloud.example.com/api/v1"

    def test_default_agent_id(self, head_manager, settings):
        b = CloudMarketplaceBridge(
            head_manager=head_manager,
            settings=settings,
            cloud_url="https://cloud.example.com/api/v1",
            cloud_api_key="key",
            cloud_project_id="proj",
        )
        assert b._cloud_agent_id == "multihead-cloud-agent"

    def test_auth_headers(self, bridge):
        headers = bridge._auth_headers()
        assert headers["Authorization"] == "Bearer test-jwt-token"


class TestCapabilityMatching:
    """Test capability matching logic."""

    def test_exact_match(self, bridge):
        assert bridge._capability_matches("text_generation") is True

    def test_exact_match_vlm(self, bridge):
        assert bridge._capability_matches("visual_reasoning") is True

    def test_prefix_match(self, bridge):
        assert bridge._capability_matches("com.multihead.llm") is True

    def test_no_match(self, bridge):
        assert bridge._capability_matches("quantum_computing") is False

    def test_case_insensitive(self, bridge):
        assert bridge._capability_matches("TEXT_GENERATION") is True

    def test_code_generation(self, bridge):
        assert bridge._capability_matches("code_generation") is True

    def test_object_detection(self, bridge):
        assert bridge._capability_matches("object_detection") is True


class TestGetCapabilities:
    """Test capability enumeration."""

    def test_returns_capabilities(self, bridge):
        caps = bridge._get_capabilities()
        assert len(caps) > 0
        # Should have head-specific + generic caps
        assert any("com.multihead.llm" in c for c in caps)
        assert "text_generation" in caps
        assert "visual_reasoning" in caps

    def test_empty_on_error(self, bridge):
        bridge._heads.get_states.side_effect = RuntimeError("boom")
        caps = bridge._get_capabilities()
        assert caps == []


class TestComputeQuote:
    """Test quote computation."""

    @staticmethod
    def _seed_listings(bridge):
        """Pre-populate listings cache so _compute_quote finds a matching listing."""
        import time
        bridge._listings_cache = [
            {
                "listing_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                "capability_id": "text_generation",
                "unit_price": 0.50,
            },
        ]
        bridge._listings_cache_time = time.time()

    @pytest.mark.asyncio
    async def test_returns_quote_for_matching_capability(self, bridge):
        self._seed_listings(bridge)
        quote = await bridge._compute_quote({
            "rfq_id": "rfq-123",
            "capability_id": "text_generation",
        })
        assert quote is not None
        assert "unit_price" in quote
        assert "estimated_latency_ms" in quote
        assert "estimated_confidence" in quote
        assert quote["estimated_latency_ms"] == 5000

    @pytest.mark.asyncio
    async def test_returns_none_for_unmatched(self, bridge):
        self._seed_listings(bridge)
        quote = await bridge._compute_quote({
            "rfq_id": "rfq-456",
            "capability_id": "quantum_computing",
        })
        assert quote is None

    @pytest.mark.asyncio
    async def test_returns_none_without_listing(self, bridge):
        """No listing match means no quote (avoids empty listing_id 422)."""
        quote = await bridge._compute_quote({
            "rfq_id": "rfq-nolisting",
            "capability_id": "text_generation",
        })
        assert quote is None

    @pytest.mark.asyncio
    async def test_respects_max_price_constraint(self, bridge):
        self._seed_listings(bridge)
        quote = await bridge._compute_quote({
            "rfq_id": "rfq-789",
            "capability_id": "text_generation",
            "constraints": {"max_price": 0.20},
        })
        assert quote is not None
        assert quote["unit_price"] <= 0.20

    @pytest.mark.asyncio
    async def test_includes_agent_metadata(self, bridge):
        self._seed_listings(bridge)
        quote = await bridge._compute_quote({
            "rfq_id": "rfq-abc",
            "capability_id": "text_generation",
        })
        assert quote is not None
        assert quote["metadata"]["agent_id"] == "test-agent"


class TestRFQDedup:
    """Test that already-quoted RFQs are skipped."""

    def test_quoted_set_starts_empty(self, bridge):
        assert len(bridge._quoted_rfqs) == 0

    def test_add_quoted_rfq(self, bridge):
        bridge._quoted_rfqs.add("rfq-123")
        assert "rfq-123" in bridge._quoted_rfqs

    @pytest.mark.asyncio
    async def test_scan_skips_quoted(self, bridge):
        """Already-quoted RFQs should not be re-quoted."""
        bridge._quoted_rfqs.add("rfq-already-quoted")

        # Mock HTTP to return the already-quoted RFQ
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [{"rfq_id": "rfq-already-quoted", "capability_id": "text_generation"}],
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await bridge._scan_and_quote()

            # Should have called request for GET (scan) but not a second POST (quote)
            # Only the GET call for scanning, no POST for quoting
            assert all(
                call.args[0] == "GET"
                for call in mock_client.request.call_args_list
            )


class TestSelfBidPrevention:
    """Prevent bidding on our own RFQs."""

    @pytest.mark.asyncio
    async def test_scan_skips_own_agent_rfq(self, bridge):
        """RFQs with our agent_id should be skipped."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"rfq_id": "rfq-own", "capability_id": "text_generation",
                 "requester_id": "multihead-cloud-agent"},
                {"rfq_id": "rfq-other", "capability_id": "text_generation",
                 "requester_id": "some-other-agent"},
            ],
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            bridge._compute_quote = AsyncMock(
                return_value={"unit_price": 1.0, "estimated_latency_ms": 100},
            )
            bridge._submit_quote = AsyncMock()

            await bridge._scan_and_quote()

            # Own RFQ should be in quoted set (skipped), not quoted on
            assert "rfq-own" in bridge._quoted_rfqs

    @pytest.mark.asyncio
    async def test_scan_skips_own_tenant_rfq(self, bridge):
        """RFQs from our own tenant/project should be skipped."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"rfq_id": "rfq-tenant", "capability_id": "text_generation",
                 "tenant_id": "test-project-id"},
            ],
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            bridge._compute_quote = AsyncMock(
                return_value={"unit_price": 1.0, "estimated_latency_ms": 100},
            )
            bridge._submit_quote = AsyncMock()

            await bridge._scan_and_quote()

            # Own tenant RFQ skipped
            assert "rfq-tenant" in bridge._quoted_rfqs
            bridge._submit_quote.assert_not_called()


class TestContractConcurrency:
    """Test contract execution concurrency limit."""

    def test_active_contracts_starts_empty(self, bridge):
        assert len(bridge._active_contracts) == 0

    @pytest.mark.asyncio
    async def test_respects_max_contracts(self, bridge):
        """Should not start new contracts when at limit."""
        # Fill up active contracts
        for i in range(2):
            future: asyncio.Future[None] = asyncio.get_event_loop().create_future()
            bridge._active_contracts[f"contract-{i}"] = asyncio.ensure_future(future)

        # Mock HTTP to return a new contract
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "contracts": [{"contract_id": "new-contract", "task_id": "t1"}],
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await bridge._check_contracts()

        # Should NOT have spawned a new contract task (at limit)
        assert "new-contract" not in bridge._active_contracts

        # Cleanup futures
        for task in bridge._active_contracts.values():
            task.cancel()

    @pytest.mark.asyncio
    async def test_cleans_up_done_contracts(self, bridge):
        """Completed contract tasks should be removed."""
        done_future: asyncio.Future[None] = asyncio.get_event_loop().create_future()
        done_future.set_result(None)
        bridge._active_contracts["done-contract"] = asyncio.ensure_future(done_future)
        # Let event loop process
        await asyncio.sleep(0)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"contracts": []}

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await bridge._check_contracts()

        assert "done-contract" not in bridge._active_contracts


class TestServiceLifecycle:
    """Test service start/stop."""

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self, bridge):
        """Stop should be safe even if never started."""
        await bridge.stop()
        assert bridge._running is False

    @pytest.mark.asyncio
    async def test_stop_cancels_tasks(self, bridge):
        """Stop should cancel scanner and monitor tasks."""
        bridge._running = True
        bridge._scanner_task = asyncio.create_task(asyncio.sleep(100))
        bridge._monitor_task = asyncio.create_task(asyncio.sleep(100))

        await bridge.stop()

        assert bridge._running is False
        assert bridge._scanner_task.cancelled() or bridge._scanner_task.done()
        assert bridge._monitor_task.cancelled() or bridge._monitor_task.done()


class TestServiceWrapper:
    """Test the service_manager wrapper function."""

    @pytest.mark.asyncio
    async def test_exits_gracefully_without_env(self):
        """Service should exit if cloud env vars are not set."""
        from multihead.service_manager import cloud_marketplace_service

        with patch.dict("os.environ", {}, clear=True):
            # Should return without error (just logs warning)
            await cloud_marketplace_service(
                head_manager=MagicMock(),
                settings=MagicMock(),
                agentic_core=MagicMock(),
                knowledge_store=MagicMock(),
                runtime_config=MagicMock(),
            )

    @pytest.mark.asyncio
    async def test_creates_bridge_with_env(self):
        """Service should create bridge when env vars are set."""
        env = {
            "ACP_CLOUD_URL": "https://cloud.example.com/api/v1",
            "ACP_CLOUD_API_KEY": "test-key",
            "ACP_CLOUD_PROJECT_ID": "test-proj",
            "ACP_CLOUD_AGENT_ID": "test-agent",
        }

        with patch.dict("os.environ", env):
            with patch("multihead.cloud_marketplace.CloudMarketplaceBridge") as mock_cls:
                mock_bridge = AsyncMock()
                mock_bridge.run = AsyncMock(side_effect=asyncio.CancelledError)
                mock_bridge.stop = AsyncMock()
                mock_cls.return_value = mock_bridge

                from multihead.service_manager import cloud_marketplace_service

                with pytest.raises(asyncio.CancelledError):
                    await cloud_marketplace_service(
                        head_manager=MagicMock(),
                        settings=MagicMock(),
                        agentic_core=MagicMock(),
                        knowledge_store=MagicMock(),
                        runtime_config=MagicMock(),
                    )

                mock_cls.assert_called_once()
                mock_bridge.run.assert_called_once()


class TestConfigIntegration:
    """Test runtime config integration."""

    def test_cloud_marketplace_field_exists(self):
        from multihead.runtime_config import ServicesConfig
        svc = ServicesConfig()
        assert svc.cloud_marketplace is False
        assert svc.cloud_rfq_interval == 600
        assert svc.cloud_contract_interval == 600
        assert svc.cloud_auto_quote is True
        assert svc.cloud_max_contracts == 2

    def test_config_set_value(self):
        from multihead.runtime_config import RuntimeConfig
        rc = RuntimeConfig()
        result = rc.set_value("services.cloud_marketplace", "true")
        assert "cloud_marketplace" in result
        assert rc.services.cloud_marketplace is True


class TestKnowledgeClaims:
    """Test knowledge store integration."""

    def test_deposit_claim_without_store(self, bridge):
        """Should not crash when knowledge_store is None."""
        bridge._knowledge_store = None
        bridge._deposit_claim("test.key", "test statement")

    # test_deposit_claim_with_store — extracted to 10-failed/test_cloud_marketplace_bugs.py
    # Production bug: EntityRef(entity_label=...) should be EntityRef(label=...)


class TestJWTTokenRefresh:
    """Test JWT token refresh for cloud marketplace."""

    def test_jwt_exp_valid_token(self):
        """Parse expiry from a valid JWT."""
        import base64, json, time
        payload = {"exp": time.time() + 3600, "sub": "agent"}
        payload_b64 = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).decode().rstrip("=")
        token = f"header.{payload_b64}.signature"
        exp = CloudMarketplaceBridge._jwt_exp(token)
        assert exp is not None
        assert abs(exp - payload["exp"]) < 1

    def test_jwt_exp_no_exp_claim(self):
        """Token without exp claim returns None-ish (0.0)."""
        import base64, json
        payload = {"sub": "agent"}
        payload_b64 = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).decode().rstrip("=")
        token = f"header.{payload_b64}.signature"
        exp = CloudMarketplaceBridge._jwt_exp(token)
        # Returns 0.0 (falsy) when no exp claim
        assert not exp

    def test_jwt_exp_invalid_token(self):
        """Invalid token returns None."""
        assert CloudMarketplaceBridge._jwt_exp("not-a-jwt") is None
        assert CloudMarketplaceBridge._jwt_exp("") is None

    @pytest.mark.asyncio
    async def test_refresh_token_updates_key(self, bridge):
        """Successful refresh should update _cloud_api_key."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"token": "new-jwt-token"}

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            await bridge._refresh_token()

        assert bridge._cloud_api_key == "new-jwt-token"

    @pytest.mark.asyncio
    async def test_refresh_token_handles_failure(self, bridge):
        """Failed refresh should not crash."""
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = Exception("network error")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            await bridge._refresh_token()

        # Should keep old key
        assert bridge._cloud_api_key == "test-jwt-token"

    @pytest.mark.asyncio
    async def test_token_refresh_loop_sleeps_for_non_jwt(self, bridge):
        """Loop should sleep and retry (not exit) for opaque tokens."""
        bridge._running = True
        bridge._cloud_api_key = "not-a-jwt"
        # Loop sleeps for opaque tokens instead of exiting; mock sleep
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            # Stop loop after first iteration
            async def _stop_after_one(*_a, **_kw):
                bridge._running = False
            mock_sleep.side_effect = _stop_after_one
            await bridge._token_refresh_loop()
        # Verify it called sleep (not early return)
        mock_sleep.assert_called()


class TestListingRegistration:
    """Test marketplace listing auto-registration."""

    @pytest.mark.asyncio
    async def test_register_skips_existing(self, bridge):
        """Should not re-register listings that already exist."""
        bridge._listings_cache = [
            {"capability_id": "com.multihead.llm.core-llm", "listing_id": "l1"},
            {"capability_id": "com.multihead.vlm.vision-vlm", "listing_id": "l2"},
        ]
        bridge._listings_cache_time = 9999999999.0

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            await bridge._register_listings()

        # Should NOT have called POST (all listings exist)
        mock_client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_register_creates_missing(self, bridge):
        """Should register listings for heads not yet listed."""
        bridge._listings_cache = []
        bridge._listings_cache_time = 9999999999.0

        mock_response = MagicMock()
        mock_response.status_code = 201

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            await bridge._register_listings()

        # Should have registered 2 listings (LLM + VLM) via _cloud_request
        post_calls = [c for c in mock_client.request.call_args_list if c.args[0] == "POST"]
        assert len(post_calls) == 2

    @pytest.mark.asyncio
    async def test_register_handles_conflict(self, bridge):
        """409 Conflict should be treated as success (already exists)."""
        bridge._listings_cache = []
        bridge._listings_cache_time = 9999999999.0

        mock_response = MagicMock()
        mock_response.status_code = 409

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            # Should not raise
            await bridge._register_listings()


class TestTrustScore:
    """Test trust score tracking."""

    # test_fetch_trust_score, test_trust_score_emits_on_change
    # extracted to 10-failed/test_cloud_marketplace_bugs.py
    # Mock targets httpx.AsyncClient but code uses self._cloud_request

    @pytest.mark.asyncio
    async def test_fetch_trust_score_handles_error(self, bridge):
        """Should return None on network error."""
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.request.side_effect = Exception("timeout")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            score = await bridge._fetch_trust_score()

        assert score is None


class TestEmitActivity:
    """Test activity emission."""

    def test_emit_calls_callback(self, bridge):
        events = []
        bridge.on_activity = lambda t, m: events.append((t, m))
        bridge._emit("test", "hello")
        assert events == [("test", "hello")]

    def test_emit_no_callback(self, bridge):
        bridge.on_activity = None
        bridge._emit("test", "hello")  # Should not crash

    def test_emit_callback_error(self, bridge):
        bridge.on_activity = MagicMock(side_effect=RuntimeError("oops"))
        bridge._emit("test", "hello")  # Should not crash


class TestStatsInit:
    """Test stats tracking initialization."""

    def test_stats_init(self, bridge):
        assert bridge._stats["rfqs_seen"] == 0
        assert bridge._stats["quotes_sent"] == 0
        assert bridge._stats["contracts_won"] == 0
        assert bridge._stats["contracts_done"] == 0

    def test_token_refresh_task_init(self, bridge):
        assert bridge._token_refresh_task is None


class TestServiceWrapperSharedData:
    """Test that service wrapper exposes stats via shared_data."""

    @pytest.mark.asyncio
    async def test_shared_data_gets_marketplace_stats(self):
        """Stats dict should be placed in shared_data."""
        env = {
            "ACP_CLOUD_URL": "https://cloud.example.com/api/v1",
            "ACP_CLOUD_API_KEY": "test-key",
            "ACP_CLOUD_PROJECT_ID": "test-proj",
        }
        shared = {}

        with patch.dict("os.environ", env):
            with patch("multihead.cloud_marketplace.CloudMarketplaceBridge") as mock_cls:
                mock_bridge = AsyncMock()
                mock_bridge.run = AsyncMock(side_effect=asyncio.CancelledError)
                mock_bridge.stop = AsyncMock()
                mock_bridge._stats = {"quotes_sent": 0, "contracts_won": 0}
                mock_cls.return_value = mock_bridge

                from multihead.service_manager import cloud_marketplace_service

                with pytest.raises(asyncio.CancelledError):
                    await cloud_marketplace_service(
                        head_manager=MagicMock(),
                        settings=MagicMock(),
                        agentic_core=MagicMock(),
                        knowledge_store=MagicMock(),
                        runtime_config=MagicMock(),
                        shared_data=shared,
                    )

        assert "marketplace_stats" in shared


# ------------------------------------------------------------------
# Full Solve Pipeline routing
# ------------------------------------------------------------------


class TestFullPipelineRouting:
    """Test pipeline routing in cloud marketplace."""

    @pytest.fixture
    def bridge_with_infra(self, head_manager, settings, runtime_config):
        """Bridge with pipeline infrastructure available."""
        svc = runtime_config.services
        svc.cloud_full_pipeline = True
        svc.cloud_pipeline_max_steps = 10
        svc.cloud_pipeline_timeout = 60.0
        return CloudMarketplaceBridge(
            head_manager=head_manager,
            settings=settings,
            cloud_url="https://cloud.example.com/api/v1",
            cloud_api_key="test-key",
            cloud_project_id="test-proj",
            cloud_agent_id="test-agent",
            runtime_config=runtime_config,
            event_store=MagicMock(),
            artifact_store=MagicMock(),
            runs_dir="/tmp/test_runs",
        )

    def test_should_use_pipeline_no_infra(self, bridge):
        """Returns False when no event_store available."""
        assert bridge._should_use_full_pipeline("ai.task.v1", "hello") is False

    def test_should_use_pipeline_config_enabled(self, bridge_with_infra):
        """Returns True when cloud_full_pipeline is True."""
        assert bridge_with_infra._should_use_full_pipeline("ai.task.v1", "any") is True

    def test_should_use_pipeline_auto_complex(self, head_manager, settings, runtime_config):
        """Auto-detects complex tasks even without explicit flag."""
        svc = runtime_config.services
        svc.cloud_full_pipeline = False
        svc.cloud_pipeline_complexity_threshold = 0.5
        b = CloudMarketplaceBridge(
            head_manager=head_manager,
            settings=settings,
            cloud_url="https://cloud.example.com/api/v1",
            cloud_api_key="key",
            cloud_project_id="proj",
            runtime_config=runtime_config,
            event_store=MagicMock(),
            artifact_store=MagicMock(),
            runs_dir="/tmp/runs",
        )
        # Complex payload should trigger pipeline
        complex_payload = (
            "First analyze the architecture, then refactor the migration "
            "in multiple phases, implement the integration pipeline, "
            "and verify with tests after each step"
        )
        assert b._should_use_full_pipeline("ai.task.v1", complex_payload) is True

    def test_should_use_pipeline_simple_no_trigger(self, head_manager, settings, runtime_config):
        """Simple payloads don't trigger pipeline even with infra."""
        svc = runtime_config.services
        svc.cloud_full_pipeline = False
        svc.cloud_pipeline_complexity_threshold = 0.7
        b = CloudMarketplaceBridge(
            head_manager=head_manager,
            settings=settings,
            cloud_url="https://cloud.example.com/api/v1",
            cloud_api_key="key",
            cloud_project_id="proj",
            runtime_config=runtime_config,
            event_store=MagicMock(),
            artifact_store=MagicMock(),
            runs_dir="/tmp/runs",
        )
        assert b._should_use_full_pipeline("ai.task.v1", "hello world") is False


class TestComplexityEstimation:
    """Test task complexity heuristic."""

    def test_simple_task(self):
        score = CloudMarketplaceBridge._estimate_complexity("hello world")
        assert score < 0.3

    def test_medium_task(self):
        score = CloudMarketplaceBridge._estimate_complexity(
            "analyze the code and then implement the fix"
        )
        assert 0.1 < score < 0.7

    def test_complex_task(self):
        score = CloudMarketplaceBridge._estimate_complexity(
            "First investigate the architecture, then plan the migration "
            "in multiple phases. Implement the refactor, optimize performance, "
            "and verify with tests after each step. Research alternative approaches."
        )
        assert score >= 0.7

    def test_long_payload_adds_score(self):
        short = CloudMarketplaceBridge._estimate_complexity("do it")
        long = CloudMarketplaceBridge._estimate_complexity("a " * 300)
        assert long > short

    def test_capped_at_one(self):
        # All keywords + long payload
        payload = " ".join([
            "steps phases then first decompose plan analyze implement verify",
            "multiple several pipeline workflow",
            "refactor architecture migration integration optimize debug investigate research",
        ]) * 10
        score = CloudMarketplaceBridge._estimate_complexity(payload)
        assert score <= 1.0


class TestFulfillViaPipeline:
    """Test pipeline execution path."""

    @pytest.mark.asyncio
    async def test_fulfill_via_pipeline(self, head_manager, settings, runtime_config):
        """Should call SolvePipeline.solve and return result."""
        svc = runtime_config.services
        svc.cloud_full_pipeline = True
        svc.cloud_pipeline_max_steps = 10
        svc.cloud_pipeline_timeout = 60.0

        b = CloudMarketplaceBridge(
            head_manager=head_manager,
            settings=settings,
            cloud_url="https://cloud.example.com/api/v1",
            cloud_api_key="key",
            cloud_project_id="proj",
            cloud_agent_id="test-agent",
            runtime_config=runtime_config,
            event_store=MagicMock(),
            artifact_store=MagicMock(),
            runs_dir="/tmp/runs",
            knowledge_store=MagicMock(),
        )

        from multihead.solve_pipeline import SolveResult

        mock_result = SolveResult(
            run_id="run_test",
            status="done",
            output="pipeline output",
            confidence=0.92,
            steps_total=3,
            steps_succeeded=3,
            steps_failed=0,
            duration_seconds=5.0,
        )

        with patch("multihead.solve_pipeline.SolvePipeline") as MockPipeline:
            mock_instance = MockPipeline.return_value
            mock_instance.solve = AsyncMock(return_value=mock_result)

            output, confidence = await b._fulfill_via_pipeline(
                "ai.complex.v1", "do complex work", "contract-123"
            )

        assert output == "pipeline output"
        assert confidence == 0.92
        mock_instance.solve.assert_called_once()

    # test_pipeline_in_route_and_execute — extracted to 10-failed/test_cloud_marketplace_bugs.py
    # _route_and_execute returns None for unmatched capability
