"""Tests for the Router module."""

from __future__ import annotations

import pytest

from multihead.head_manager import HeadManager
from multihead.models import AdapterKind, HeadManifest
from multihead.observability import MetricsCollector
from multihead.router import Router


@pytest.fixture
def head_manager():
    """HeadManager with llm + vlm heads."""
    manifests = {
        "llm-a": HeadManifest(
            head_id="llm-a", name="LLM A", adapter=AdapterKind.MOCK,
            model="mock-v1", kind="llm", gpu_required=True, vram_hint_mb=6000,
        ),
        "llm-b": HeadManifest(
            head_id="llm-b", name="LLM B", adapter=AdapterKind.MOCK,
            model="mock-v1", kind="llm", gpu_required=False,
        ),
        "vlm-a": HeadManifest(
            head_id="vlm-a", name="VLM A", adapter=AdapterKind.MOCK,
            model="mock-v1", kind="vlm", gpu_required=True, vram_hint_mb=18000,
        ),
    }
    return HeadManager(manifests)


class TestRouterRoute:
    def test_routes_to_matching_kind(self, head_manager):
        """Router should return a head of the correct kind."""
        router = Router(head_manager)
        result = router.route("llm")
        assert result in ("llm-a", "llm-b")

    def test_routes_vlm(self, head_manager):
        """VLM kind should return vlm-a."""
        router = Router(head_manager)
        result = router.route("vlm")
        assert result == "vlm-a"

    def test_returns_none_for_unknown_kind(self, head_manager):
        """Unknown kind should return None."""
        router = Router(head_manager)
        result = router.route("embed")
        assert result is None

    def test_excludes_specified_heads(self, head_manager):
        """Exclude param should skip specified heads."""
        router = Router(head_manager)
        result = router.route("llm", exclude={"llm-a"})
        assert result == "llm-b"

    def test_exclude_all_returns_none(self, head_manager):
        """Excluding all candidates returns None."""
        router = Router(head_manager)
        result = router.route("vlm", exclude={"vlm-a"})
        assert result is None

    def test_prefers_active_head(self, head_manager):
        """Active head should score higher due to swap cost avoidance."""
        # Activate llm-b by setting its state
        head_manager._states["llm-b"] = head_manager._states["llm-b"]  # ensure exists
        # Manually set active flag
        original_get_states = head_manager.get_states

        def patched_get_states():
            states = original_get_states()
            states["llm-b"]["is_active"] = True
            return states

        head_manager.get_states = patched_get_states

        router = Router(head_manager)
        result = router.route("llm")
        assert result == "llm-b"

    def test_avoids_open_circuit_breaker(self, head_manager):
        """Head with open circuit breaker should be filtered out."""
        # Trip the breaker for llm-a
        breaker = head_manager.get_breaker("llm-a")
        if breaker:
            breaker._failure_count = breaker.failure_threshold
            breaker._state = "open"

        router = Router(head_manager)
        result = router.route("llm")
        assert result == "llm-b"

    def test_considers_error_rate(self, head_manager):
        """Head with lower error rate should score higher."""
        metrics = MetricsCollector()

        # llm-a: 10 calls, 5 errors (50% error rate)
        for _ in range(10):
            metrics.inc("head_generate_total", labels={"head_id": "llm-a"})
        for _ in range(5):
            metrics.inc("head_generate_errors_total", labels={"head_id": "llm-a"})

        # llm-b: 10 calls, 0 errors (0% error rate)
        for _ in range(10):
            metrics.inc("head_generate_total", labels={"head_id": "llm-b"})

        router = Router(head_manager, metrics=metrics)
        result = router.route("llm")
        # llm-b should win on error rate (other factors being equal)
        assert result == "llm-b"


class TestRouterRank:
    def test_rank_returns_sorted(self, head_manager):
        """rank() should return all candidates sorted by score descending."""
        router = Router(head_manager)
        ranked = router.rank("llm")
        assert len(ranked) == 2
        # Scores should be descending
        assert ranked[0][1] >= ranked[1][1]
        # All returned heads should be llm kind
        head_ids = {hid for hid, _ in ranked}
        assert head_ids == {"llm-a", "llm-b"}

    def test_rank_empty_for_unknown_kind(self, head_manager):
        """rank() for unknown kind should return empty list."""
        router = Router(head_manager)
        ranked = router.rank("embed")
        assert ranked == []


class TestRouterPrivacyFiltering:
    """Phase 2: Privacy-aware routing tests."""

    @pytest.fixture
    def privacy_head_manager(self):
        """HeadManager with local + external solvers."""
        from multihead.models import Capability

        manifests = {
            "local-llm": HeadManifest(
                head_id="local-llm",
                name="Local LLM",
                adapter=AdapterKind.TRANSFORMERS,
                model="local-model",
                kind="llm",
                is_local=True,
                capabilities=Capability(
                    solver_type="llm",
                    task_types=["reasoning", "text_generation"],
                    input_modalities=["text"],
                    output_modalities=["text"],
                ),
            ),
            "external-api": HeadManifest(
                head_id="external-api",
                name="External API",
                adapter=AdapterKind.OPENAI,
                model="gpt-4",
                kind="llm",
                is_local=False,
                privacy_level="external",
                capabilities=Capability(
                    solver_type="llm",
                    task_types=["reasoning", "text_generation"],
                    input_modalities=["text"],
                    output_modalities=["text"],
                ),
            ),
            "botvibes-encrypted": HeadManifest(
                head_id="botvibes-encrypted",
                name="BotVibes Encrypted",
                adapter=AdapterKind.BOTVIBES,
                model="expert-1",
                kind="llm",
                is_local=False,
                privacy_level="encrypted",
                endpoint="http://localhost:8000",  # Mock ACP URL
                extra={
                    "api_key": "test-token",
                    "project_id": "test-project",
                    "target_capability": "com.test.capability",
                },
                capabilities=Capability(
                    solver_type="llm",
                    task_types=["reasoning", "text_generation"],
                    input_modalities=["text"],
                    output_modalities=["text"],
                ),
            ),
        }
        return HeadManager(manifests)

    def test_confidential_data_blocks_external(self, privacy_head_manager):
        """CONFIDENTIAL data should only route to local solvers."""
        from multihead.models import DataSensitivity, PrivacyConstraint

        router = Router(privacy_head_manager)
        privacy = PrivacyConstraint(data_sensitivity=DataSensitivity.CONFIDENTIAL)

        result = router.route_by_task(
            task_types=["reasoning"],
            privacy=privacy
        )

        # Should select local-llm, not external solvers
        assert result == "local-llm"

    def test_internal_data_allows_local_and_encrypted(self, privacy_head_manager):
        """INTERNAL data should allow local + encrypted BotVibes."""
        from multihead.models import DataSensitivity, PrivacyConstraint

        router = Router(privacy_head_manager)
        privacy = PrivacyConstraint(data_sensitivity=DataSensitivity.INTERNAL)

        result = router.route_by_task(
            task_types=["reasoning"],
            privacy=privacy
        )

        # Should allow local-llm or botvibes-encrypted, but NOT external-api
        assert result in ("local-llm", "botvibes-encrypted")
        assert result != "external-api"

    def test_public_data_allows_all(self, privacy_head_manager):
        """PUBLIC data should allow all solvers."""
        from multihead.models import DataSensitivity, PrivacyConstraint

        router = Router(privacy_head_manager)
        privacy = PrivacyConstraint(data_sensitivity=DataSensitivity.PUBLIC)

        result = router.route_by_task(
            task_types=["reasoning"],
            privacy=privacy
        )

        # Should allow any solver
        assert result in ("local-llm", "external-api", "botvibes-encrypted")

    def test_allowed_providers_whitelist(self, privacy_head_manager):
        """allowed_providers should whitelist specific solvers."""
        from multihead.models import PrivacyConstraint

        router = Router(privacy_head_manager)
        privacy = PrivacyConstraint(
            allowed_providers=["local-llm"]
        )

        result = router.route_by_task(
            task_types=["reasoning"],
            privacy=privacy
        )

        # Should only route to whitelisted provider
        assert result == "local-llm"

    def test_blocked_providers_blacklist(self, privacy_head_manager):
        """blocked_providers should exclude specific solvers."""
        from multihead.models import PrivacyConstraint

        router = Router(privacy_head_manager)
        privacy = PrivacyConstraint(
            blocked_providers=["external-api"]
        )

        result = router.route_by_task(
            task_types=["reasoning"],
            privacy=privacy
        )

        # Should not route to blocked provider
        assert result != "external-api"
        assert result in ("local-llm", "botvibes-encrypted")

    def test_no_privacy_constraint_allows_all(self, privacy_head_manager):
        """No privacy constraint should allow all solvers."""
        router = Router(privacy_head_manager)

        result = router.route_by_task(
            task_types=["reasoning"],
            privacy=None
        )

        # Should allow any solver when no privacy constraint
        assert result in ("local-llm", "external-api", "botvibes-encrypted")
