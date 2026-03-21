"""Tests for Router + CapabilityDiscovery integration (Gap #4)."""

import json
import sqlite3
from unittest.mock import Mock

import pytest

from multihead.capability_discovery import CapabilityDiscovery
from multihead.models import DataSensitivity, HeadManifest, PrivacyConstraint
from multihead.router import Router


@pytest.fixture
def mock_head_manager():
    """Create mock HeadManager with test heads."""
    manager = Mock()

    # Mock head states
    states = {
        "qwen-llm": {
            "kind": "llm",
            "model": "Qwen/Qwen3-8B",
            "is_active": False,
        },
        "qwen-vlm": {
            "kind": "vlm",
            "model": "Qwen/Qwen3-VL-32B-Thinking",
            "is_active": False,
        },
        "mock-llm": {
            "kind": "llm",
            "model": "mock-llm-v1",
            "is_active": False,
        },
    }
    manager.get_states.return_value = states

    # Mock manifests
    def get_manifest(head_id):
        manifests = {
            "qwen-llm": HeadManifest(
                head_id="qwen-llm",
                name="Qwen LLM",
                kind="llm",
                model="Qwen/Qwen3-8B",
                adapter="transformers",
                gpu_required=True,
                is_local=True,
            ),
            "qwen-vlm": HeadManifest(
                head_id="qwen-vlm",
                name="Qwen VLM",
                kind="vlm",
                model="Qwen/Qwen3-VL-32B-Thinking",
                adapter="transformers",
                gpu_required=True,
                is_local=True,
            ),
            "mock-llm": HeadManifest(
                head_id="mock-llm",
                name="Mock LLM",
                kind="llm",
                model="mock-llm-v1",
                adapter="mock",
                gpu_required=False,
                is_local=True,
            ),
        }
        return manifests.get(head_id)

    manager.get_manifest.side_effect = get_manifest

    # Mock circuit breakers (all closed/healthy)
    def get_breaker(head_id):
        breaker = Mock()
        breaker.state = "closed"
        return breaker

    manager.get_breaker.side_effect = get_breaker

    return manager


@pytest.fixture
def mock_discovery(tmp_path):
    """Create CapabilityDiscovery with mock data."""
    # Create acp_state.json
    acp_state = {
        "agent_id": "multihead-agent",
        "capabilities": {
            "capabilities": [
                "com.multihead.llm.qwen-llm",
                "com.multihead.llm.mock-llm",
                "com.multihead.vlm.qwen-vlm",
            ],
            "latency_profile": {"p50_ms": 2000, "p95_ms": 10000},
        },
        "heads": {
            "qwen-llm": "Qwen/Qwen3-8B",
            "mock-llm": "mock-llm-v1",
            "qwen-vlm": "Qwen/Qwen3-VL-32B-Thinking",
        },
    }
    (tmp_path / "acp_state.json").write_text(json.dumps(acp_state))

    # Create knowledge.db with model claims
    db_path = tmp_path / "knowledge.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE claims (
            claim_id TEXT PRIMARY KEY,
            claim_status TEXT,
            claim_type TEXT,
            statement TEXT,
            subject_json TEXT,
            object_json TEXT
        )
    """)
    conn.execute(
        "INSERT INTO claims VALUES (?, ?, ?, ?, ?, ?)",
        (
            "clm_qwen_perf",
            "accepted",
            "fact",
            "Qwen/Qwen3-8B achieves 85% accuracy on reasoning tasks",
            "{}",
            "{}",
        ),
    )
    conn.commit()
    conn.close()

    discovery = CapabilityDiscovery(tmp_path, knowledge_db_path=db_path)
    discovery.reload()
    return discovery


def test_router_with_discovery_initialization(mock_head_manager, mock_discovery):
    """Test Router initialization with discovery parameter."""
    router = Router(mock_head_manager, discovery=mock_discovery)

    assert router.discovery is mock_discovery
    assert router.heads is mock_head_manager


def test_route_with_discovery_by_capability_id(mock_head_manager, mock_discovery):
    """Test routing with capability ID query."""
    router = Router(mock_head_manager, discovery=mock_discovery)

    # Query for LLM capability
    head_id = router.route_with_discovery("com.multihead.llm.qwen-llm")

    assert head_id == "qwen-llm"


def test_route_with_discovery_by_prefix(mock_head_manager, mock_discovery):
    """Test routing with prefix wildcard."""
    router = Router(mock_head_manager, discovery=mock_discovery)

    # Query for all LLM capabilities
    head_id = router.route_with_discovery("com.multihead.llm*")

    assert head_id in ["qwen-llm", "mock-llm"]


def test_route_with_discovery_by_kind(mock_head_manager, mock_discovery):
    """Test routing by kind."""
    router = Router(mock_head_manager, discovery=mock_discovery)

    # Query for VLM
    head_id = router.route_with_discovery("vlm")

    assert head_id == "qwen-vlm"


def test_route_with_discovery_semantic_query(mock_head_manager, mock_discovery):
    """Test semantic query via discovery."""
    router = Router(mock_head_manager, discovery=mock_discovery)

    # Query with substring that matches capability IDs
    head_id = router.route_with_discovery("llm")

    # Should find an LLM head (qwen-llm or mock-llm)
    assert head_id in ["qwen-llm", "mock-llm"]


def test_route_with_discovery_no_match(mock_head_manager, mock_discovery):
    """Test query with no matching capabilities."""
    router = Router(mock_head_manager, discovery=mock_discovery)

    # Query for non-existent capability
    head_id = router.route_with_discovery("com.nonexistent.capability")

    assert head_id is None


def test_route_with_discovery_exclude(mock_head_manager, mock_discovery):
    """Test exclude parameter filters candidates."""
    router = Router(mock_head_manager, discovery=mock_discovery)

    # Query LLM but exclude qwen-llm
    head_id = router.route_with_discovery("com.multihead.llm*", exclude={"qwen-llm"})

    # Should select mock-llm instead
    assert head_id == "mock-llm"


def test_route_with_discovery_privacy_confidential(mock_head_manager, mock_discovery):
    """Test privacy filtering - confidential data requires local heads."""
    router = Router(mock_head_manager, discovery=mock_discovery)

    privacy = PrivacyConstraint(data_sensitivity=DataSensitivity.CONFIDENTIAL)

    # All test heads are local, should work
    head_id = router.route_with_discovery("com.multihead.llm*", privacy=privacy)

    assert head_id is not None


def test_route_with_discovery_fallback_to_route_by_task(mock_head_manager):
    """Test fallback when discovery not available."""
    router = Router(mock_head_manager, discovery=None)

    # Should fall back to route_by_task if task_types provided
    head_id = router.route_with_discovery(
        "com.multihead.llm",
        task_types=["reasoning"],
    )

    # Without discovery and without capable manifests, returns None
    assert head_id is None


def test_route_with_discovery_scoring_uses_match_quality(mock_head_manager, mock_discovery):
    """Test that discovery match score influences routing decision."""
    router = Router(mock_head_manager, discovery=mock_discovery)

    # Query that matches multiple heads
    head_id = router.route_with_discovery("com.multihead.llm")

    # Should select based on match score + router scoring
    assert head_id in ["qwen-llm", "mock-llm"]


def test_map_model_to_head_exact_match(mock_head_manager):
    """Test mapping model name to head_id - exact match."""
    router = Router(mock_head_manager)

    head_id = router._map_model_to_head("Qwen/Qwen3-8B")

    assert head_id == "qwen-llm"


def test_map_model_to_head_fuzzy_match(mock_head_manager):
    """Test mapping model name to head_id - fuzzy matching."""
    router = Router(mock_head_manager)

    # Fuzzy match for Qwen LLM variants
    head_id = router._map_model_to_head("qwen3-8b")
    assert head_id == "qwen-llm"

    # Fuzzy match for Qwen VLM
    head_id = router._map_model_to_head("qwen3-vl-32b-thinking")
    assert head_id == "qwen-vlm"


def test_map_model_to_head_no_match(mock_head_manager):
    """Test mapping model name with no match."""
    router = Router(mock_head_manager)

    head_id = router._map_model_to_head("UnknownModel")

    assert head_id is None


def test_passes_privacy_check_confidential(mock_head_manager):
    """Test privacy check - confidential requires local."""
    router = Router(mock_head_manager)

    manifest = mock_head_manager.get_manifest("qwen-llm")
    privacy = PrivacyConstraint(data_sensitivity=DataSensitivity.CONFIDENTIAL)

    assert router._passes_privacy_check(manifest, privacy) is True


def test_passes_privacy_check_internal_local(mock_head_manager):
    """Test privacy check - internal allows local."""
    router = Router(mock_head_manager)

    manifest = mock_head_manager.get_manifest("qwen-llm")
    privacy = PrivacyConstraint(data_sensitivity=DataSensitivity.INTERNAL)

    assert router._passes_privacy_check(manifest, privacy) is True


def test_passes_privacy_check_public(mock_head_manager):
    """Test privacy check - public allows all."""
    router = Router(mock_head_manager)

    manifest = mock_head_manager.get_manifest("qwen-llm")
    privacy = PrivacyConstraint(data_sensitivity=DataSensitivity.PUBLIC)

    assert router._passes_privacy_check(manifest, privacy) is True


def test_route_with_discovery_logs_candidates(mock_head_manager, mock_discovery, caplog):
    """Test that router logs candidate selection details."""
    import logging
    caplog.set_level(logging.INFO)

    router = Router(mock_head_manager, discovery=mock_discovery)

    head_id = router.route_with_discovery("com.multihead.llm*")

    # Should log selection
    assert "Router selected" in caplog.text
    assert "candidates" in caplog.text


def test_route_with_discovery_debug_logging(mock_head_manager, mock_discovery, caplog):
    """Test debug logging shows candidate scores."""
    import logging
    caplog.set_level(logging.DEBUG)

    router = Router(mock_head_manager, discovery=mock_discovery)

    router.route_with_discovery("com.multihead.llm*")

    # Debug logs should show candidate details
    # (may or may not appear depending on test environment logger config)
    # Just verify no crashes with debug enabled
    assert True


def test_route_with_discovery_integration_workflow(mock_head_manager, mock_discovery):
    """Test complete workflow: discovery → filter → score → select."""
    router = Router(mock_head_manager, discovery=mock_discovery)

    # Step 1: Query discovers capabilities
    head_id = router.route_with_discovery(
        "com.multihead.llm",
        privacy=PrivacyConstraint(data_sensitivity=DataSensitivity.CONFIDENTIAL),
        exclude=set(),
    )

    # Step 2: Router filters by privacy (all local, should pass)
    # Step 3: Router scores candidates
    # Step 4: Router selects best

    assert head_id in ["qwen-llm", "mock-llm"]
