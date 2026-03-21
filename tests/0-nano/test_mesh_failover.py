"""Tests for MeshFailoverPolicy."""

from __future__ import annotations

import pytest
from unittest.mock import Mock

from multihead.mesh.failover import MeshFailoverPolicy
from multihead.mesh.peer_registry import PeerHead
from multihead.models import AdapterKind, HeadManifest
from multihead.resilience import CircuitBreaker


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


@pytest.fixture
def head_manager():
    """Mock HeadManager with two local heads."""
    manager = Mock()
    manager._manifests = {
        "local-llm": HeadManifest(
            head_id="local-llm", name="Local LLM",
            adapter=AdapterKind.MOCK, model="test-llm", kind="llm",
        ),
        "local-vlm": HeadManifest(
            head_id="local-vlm", name="Local VLM",
            adapter=AdapterKind.MOCK, model="test-vlm", kind="vlm",
        ),
    }

    breakers = {
        "local-llm": CircuitBreaker(failure_threshold=5),
        "local-vlm": CircuitBreaker(failure_threshold=5),
    }
    manager.get_breaker = lambda hid: breakers.get(hid)
    return manager


@pytest.fixture
def peer_registry():
    """Mock PeerRegistry with two remote peers."""
    reg = Mock()
    reg.peer_heads = {
        "mesh-fast-llm": PeerHead(
            head_id="mesh-fast-llm", node_id="fast",
            peer_url="http://fast:7337", capability_kind="llm",
            status="available", latency_ms=20.0,
        ),
        "mesh-slow-llm": PeerHead(
            head_id="mesh-slow-llm", node_id="slow",
            peer_url="http://slow:7337", capability_kind="llm",
            status="available", latency_ms=500.0,
        ),
        "mesh-peer-vlm": PeerHead(
            head_id="mesh-peer-vlm", node_id="vlm-node",
            peer_url="http://vlm:7337", capability_kind="vlm",
            status="available", latency_ms=50.0,
        ),
    }
    return reg


@pytest.fixture
def policy(head_manager, peer_registry):
    return MeshFailoverPolicy(
        head_manager=head_manager,
        peer_registry=peer_registry,
    )


# -----------------------------------------------------------------------
# Basic fallback tests
# -----------------------------------------------------------------------


class TestGetFallbacks:
    """Test MeshFailoverPolicy.get_fallbacks()."""

    def test_excludes_failed_head(self, policy):
        """Should not include the failed head in fallbacks."""
        fallbacks = policy.get_fallbacks("local-llm", required_kind="llm")
        assert "local-llm" not in fallbacks

    def test_prefers_local_over_mesh(self, policy):
        """Should rank local heads before mesh peers."""
        # Fail local-llm, should get other local heads first (none for llm kind)
        # But local-vlm is vlm kind, so with required_kind="llm" it won't match
        # So we should get mesh peers
        fallbacks = policy.get_fallbacks("local-llm", required_kind="llm")
        # mesh-fast-llm and mesh-slow-llm are the only llm alternatives
        assert len(fallbacks) == 2
        assert all(fb.startswith("mesh-") for fb in fallbacks)

    def test_without_required_kind_returns_all(self, policy):
        """Should return all heads when no required_kind specified."""
        fallbacks = policy.get_fallbacks("local-llm")
        # local-vlm + 3 mesh peers = 4
        assert len(fallbacks) == 4
        # local-vlm should be first (local, score 100)
        assert fallbacks[0] == "local-vlm"

    def test_prefers_lower_latency_mesh_peer(self, policy):
        """Should rank lower-latency mesh peers higher."""
        fallbacks = policy.get_fallbacks("local-llm", required_kind="llm")
        fast_idx = fallbacks.index("mesh-fast-llm")
        slow_idx = fallbacks.index("mesh-slow-llm")
        assert fast_idx < slow_idx

    def test_excludes_additional_heads(self, policy):
        """Should respect explicit exclude set."""
        fallbacks = policy.get_fallbacks(
            "local-llm",
            required_kind="llm",
            exclude={"mesh-fast-llm"},
        )
        assert "mesh-fast-llm" not in fallbacks
        assert "mesh-slow-llm" in fallbacks

    def test_filters_by_required_kind(self, policy):
        """Should only return heads matching required_kind."""
        fallbacks = policy.get_fallbacks("local-vlm", required_kind="vlm")
        assert "local-llm" not in fallbacks
        assert "mesh-fast-llm" not in fallbacks
        assert "mesh-peer-vlm" in fallbacks

    def test_returns_empty_when_no_alternatives(self, policy):
        """Should return empty list when no alternatives available."""
        # Request embed kind which doesn't exist
        fallbacks = policy.get_fallbacks("local-llm", required_kind="embed")
        assert fallbacks == []


# -----------------------------------------------------------------------
# Circuit breaker integration
# -----------------------------------------------------------------------


class TestCircuitBreakerIntegration:
    """Test that circuit-broken heads are excluded."""

    def test_skips_circuit_broken_local_head(self, head_manager, peer_registry):
        """Should not include local heads with open circuit breakers."""
        # Trip the circuit breaker for local-vlm
        breaker = head_manager.get_breaker("local-vlm")
        for _ in range(5):
            breaker.record_failure()
        assert breaker.state == "open"

        policy = MeshFailoverPolicy(
            head_manager=head_manager,
            peer_registry=peer_registry,
        )
        fallbacks = policy.get_fallbacks("local-llm")
        assert "local-vlm" not in fallbacks


# -----------------------------------------------------------------------
# Offline peer handling
# -----------------------------------------------------------------------


class TestOfflinePeers:
    """Test that offline peers are excluded."""

    def test_skips_offline_mesh_peers(self, head_manager, peer_registry):
        """Should not include offline peers in fallbacks."""
        peer_registry.peer_heads["mesh-fast-llm"].status = "offline"

        policy = MeshFailoverPolicy(
            head_manager=head_manager,
            peer_registry=peer_registry,
        )
        fallbacks = policy.get_fallbacks("local-llm", required_kind="llm")
        assert "mesh-fast-llm" not in fallbacks
        assert "mesh-slow-llm" in fallbacks


# -----------------------------------------------------------------------
# Success/failure tracking
# -----------------------------------------------------------------------


class TestReportTracking:
    """Test success/failure tracking and ranking effects."""

    def test_failure_demotes_ranking(self, policy):
        """Heads with more failures should rank lower."""
        # Record failures for mesh-fast-llm
        for _ in range(10):
            policy.report_failure("mesh-fast-llm", "timeout")
        # Record successes for mesh-slow-llm
        for _ in range(10):
            policy.report_success("mesh-slow-llm")

        fallbacks = policy.get_fallbacks("local-llm", required_kind="llm")
        # mesh-slow-llm should now rank higher despite higher latency
        # because mesh-fast-llm has 100% error rate
        slow_idx = fallbacks.index("mesh-slow-llm")
        fast_idx = fallbacks.index("mesh-fast-llm")
        assert slow_idx < fast_idx

    def test_success_boosts_ranking(self, policy):
        """Heads with successes should maintain high ranking."""
        for _ in range(20):
            policy.report_success("mesh-fast-llm", latency_ms=15.0)

        fallbacks = policy.get_fallbacks("local-llm", required_kind="llm")
        assert fallbacks[0] == "mesh-fast-llm"

    def test_report_failure_increments_count(self, policy):
        """report_failure should increment failure count."""
        policy.report_failure("test-head", "error")
        assert policy._failure_counts["test-head"] == 1
        policy.report_failure("test-head", "error")
        assert policy._failure_counts["test-head"] == 2

    def test_report_success_increments_count(self, policy):
        """report_success should increment success count."""
        policy.report_success("test-head", latency_ms=50.0)
        assert policy._success_counts["test-head"] == 1
        assert policy._last_latency["test-head"] == 50.0


# -----------------------------------------------------------------------
# No peer registry
# -----------------------------------------------------------------------


class TestWithoutPeerRegistry:
    """Test behavior when no peer registry is configured."""

    def test_local_only_fallbacks(self, head_manager):
        """Should only return local heads when no peer registry."""
        policy = MeshFailoverPolicy(head_manager=head_manager)
        fallbacks = policy.get_fallbacks("local-llm")
        assert len(fallbacks) == 1
        assert fallbacks[0] == "local-vlm"
        assert not any(fb.startswith("mesh-") for fb in fallbacks)

    def test_empty_when_only_head_of_kind(self, head_manager):
        """Should return empty when failed head is the only one of its kind."""
        policy = MeshFailoverPolicy(head_manager=head_manager)
        fallbacks = policy.get_fallbacks("local-llm", required_kind="llm")
        assert fallbacks == []
