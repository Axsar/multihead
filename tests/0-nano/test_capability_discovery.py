"""Tests for capability discovery (Gap #3 fix)."""

import json
import sqlite3

import pytest

from multihead.capability_discovery import (
    CapabilityDiscovery,
    CapabilityMatch,
    discover_capabilities,
)


@pytest.fixture
def mock_data_dir(tmp_path):
    """Create mock data directory with acp_state.json."""
    acp_state = {
        "agent_id": "multihead-agent",
        "claude_agent_id": "claude-session-agent",
        "status": "idle",
        "capabilities": {
            "capabilities": [
                "com.multihead.llm.mock-llm",
                "com.multihead.llm.qwen-llm",
                "com.multihead.vlm.qwen-vlm",
                "com.multihead.llm.claude-sonnet",
            ],
            "latency_profile": {"p50_ms": 2000, "p95_ms": 10000},
            "cost_model": {"unit": "task", "price": 0.0},
        },
        "claude_capabilities": {
            "capabilities": [
                "com.claude.code.edit",
                "com.claude.code.test",
                "com.claude.code.review",
            ],
            "latency_profile": {"p50_ms": 5000, "p95_ms": 30000},
            "cost_model": {"unit": "task", "price": 0.0},
        },
        "heads": {
            "mock-llm": "mock-llm-v1",
            "qwen-llm": "Qwen/Qwen3-8B",
            "qwen-vlm": "Qwen/Qwen3-VL-32B-Thinking",
            "claude-sonnet": "sonnet",
        },
    }

    state_file = tmp_path / "acp_state.json"
    state_file.write_text(json.dumps(acp_state, indent=2))

    return tmp_path


@pytest.fixture
def mock_knowledge_db(tmp_path):
    """Create mock knowledge database with model performance claims."""
    db_path = tmp_path / "knowledge.db"
    conn = sqlite3.connect(str(db_path))

    # Create claims table (simplified schema)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS claims (
            claim_id TEXT PRIMARY KEY,
            claim_status TEXT,
            claim_type TEXT,
            statement TEXT,
            subject_json TEXT,
            object_json TEXT
        )
    """)

    # Insert model performance claims
    claims = [
        {
            "claim_id": "clm_yolo_perf_001",
            "claim_status": "accepted",
            "claim_type": "fact",
            "statement": "YOLOv8m achieving 91.85% mAP50 for object detection on 8 classes",
            "subject_json": "{}",
            "object_json": "{}",
        },
        {
            "claim_id": "clm_sam2_perf_001",
            "claim_status": "accepted",
            "claim_type": "fact",
            "statement": "SAM2 segmentation with 10.5s/page processing speed",
            "subject_json": "{}",
            "object_json": "{}",
        },
        {
            "claim_id": "clm_unet_perf_001",
            "claim_status": "accepted",
            "claim_type": "fact",
            "statement": (
                "UNet achieving 99.58% IoU for image segmentation,"
                " 11x faster at 0.9s/page"
            ),
            "subject_json": "{}",
            "object_json": "{}",
        },
    ]

    for claim in claims:
        conn.execute(
            "INSERT INTO claims VALUES (?, ?, ?, ?, ?, ?)",
            (
                claim["claim_id"],
                claim["claim_status"],
                claim["claim_type"],
                claim["statement"],
                claim["subject_json"],
                claim["object_json"],
            ),
        )

    conn.commit()
    conn.close()

    return db_path


def test_discovery_initialization(mock_data_dir):
    """Test CapabilityDiscovery initialization."""
    discovery = CapabilityDiscovery(mock_data_dir)
    assert discovery.data_dir == mock_data_dir
    assert discovery.knowledge_db_path == mock_data_dir / "knowledge.db"


def test_load_acp_state(mock_data_dir):
    """Test loading acp_state.json."""
    discovery = CapabilityDiscovery(mock_data_dir)
    discovery.reload()

    assert discovery._acp_state is not None
    assert discovery._acp_state["agent_id"] == "multihead-agent"
    assert len(discovery._acp_state["capabilities"]["capabilities"]) == 4


def test_query_exact_match(mock_data_dir):
    """Test exact capability ID matching."""
    discovery = CapabilityDiscovery(mock_data_dir)
    discovery.reload()

    matches = discovery.query("com.multihead.llm.qwen-llm", exact_match=True)

    assert len(matches) == 1
    assert matches[0].capability_id == "com.multihead.llm.qwen-llm"
    assert matches[0].agent_id == "multihead-agent"
    assert matches[0].kind == "llm"
    assert matches[0].model == "Qwen/Qwen3-8B"
    assert matches[0].source == "local"


def test_query_prefix_match(mock_data_dir):
    """Test prefix matching with wildcard."""
    discovery = CapabilityDiscovery(mock_data_dir)
    discovery.reload()

    matches = discovery.query("com.multihead.llm*")

    assert len(matches) >= 3  # mock-llm, qwen-llm, claude-sonnet
    for match in matches:
        assert match.capability_id.startswith("com.multihead.llm")


def test_query_substring_match(mock_data_dir):
    """Test substring matching."""
    discovery = CapabilityDiscovery(mock_data_dir)
    discovery.reload()

    matches = discovery.query("qwen")

    assert len(matches) >= 2  # qwen-llm, qwen-vlm
    for match in matches:
        assert "qwen" in match.capability_id


def test_query_claude_capabilities(mock_data_dir):
    """Test querying Claude Code capabilities."""
    discovery = CapabilityDiscovery(mock_data_dir)
    discovery.reload()

    matches = discovery.query("com.claude.code")

    assert len(matches) >= 3  # edit, test, review
    for match in matches:
        assert match.agent_id == "claude-session-agent"
        assert match.capability_id.startswith("com.claude.code")


def test_query_by_kind_llm(mock_data_dir):
    """Test querying by kind: llm."""
    discovery = CapabilityDiscovery(mock_data_dir)
    discovery.reload()

    matches = discovery.query_by_kind("llm")

    assert len(matches) >= 3
    for match in matches:
        assert match.kind == "llm"


def test_query_by_kind_vlm(mock_data_dir):
    """Test querying by kind: vlm."""
    discovery = CapabilityDiscovery(mock_data_dir)
    discovery.reload()

    matches = discovery.query_by_kind("vlm")

    assert len(matches) >= 1
    for match in matches:
        assert match.kind == "vlm"


def test_query_knowledge_base_yolo(mock_data_dir, mock_knowledge_db):
    """Test querying knowledge base for YOLO model."""
    discovery = CapabilityDiscovery(mock_data_dir, knowledge_db_path=mock_knowledge_db)
    discovery.reload()

    matches = discovery.query("YOLO")

    # Should find YOLO from knowledge base
    kb_matches = [m for m in matches if m.source == "knowledge"]
    assert len(kb_matches) >= 1

    yolo_match = kb_matches[0]
    assert yolo_match.model == "YOLOv8m"
    assert yolo_match.kind == "model"
    assert yolo_match.performance is not None
    assert "mAP50" in yolo_match.performance
    assert yolo_match.performance["mAP50"] == pytest.approx(0.9185, rel=0.01)


def test_query_knowledge_base_sam2(mock_data_dir, mock_knowledge_db):
    """Test querying knowledge base for SAM2 model."""
    discovery = CapabilityDiscovery(mock_data_dir, knowledge_db_path=mock_knowledge_db)
    discovery.reload()

    matches = discovery.query("SAM2")

    kb_matches = [m for m in matches if m.source == "knowledge"]
    assert len(kb_matches) >= 1

    sam2_match = kb_matches[0]
    assert sam2_match.model == "SAM2"
    assert sam2_match.latency_p50_ms == 10500  # 10.5s/page


def test_query_knowledge_base_unet(mock_data_dir, mock_knowledge_db):
    """Test querying knowledge base for UNet model."""
    discovery = CapabilityDiscovery(mock_data_dir, knowledge_db_path=mock_knowledge_db)
    discovery.reload()

    matches = discovery.query("UNet")

    kb_matches = [m for m in matches if m.source == "knowledge"]
    assert len(kb_matches) >= 1

    unet_match = kb_matches[0]
    assert unet_match.model == "UNet"
    assert unet_match.performance is not None
    assert "iou" in unet_match.performance
    assert unet_match.performance["iou"] == pytest.approx(0.9958, rel=0.01)
    assert unet_match.latency_p50_ms == 900  # 0.9s/page


def test_query_semantic_detection(mock_data_dir, mock_knowledge_db):
    """Test semantic query for 'detection' finds YOLO."""
    discovery = CapabilityDiscovery(mock_data_dir, knowledge_db_path=mock_knowledge_db)
    discovery.reload()

    matches = discovery.query("detection")

    # Should find YOLO (detection model) from knowledge base
    kb_matches = [m for m in matches if m.source == "knowledge" and "YOLO" in m.model]
    assert len(kb_matches) >= 1


def test_query_semantic_segmentation(mock_data_dir, mock_knowledge_db):
    """Test semantic query for 'segmentation' finds SAM2 and UNet."""
    discovery = CapabilityDiscovery(mock_data_dir, knowledge_db_path=mock_knowledge_db)
    discovery.reload()

    matches = discovery.query("segmentation")

    # Should find SAM2 and UNet from knowledge base
    kb_matches = [m for m in matches if m.source == "knowledge"]
    models = {m.model for m in kb_matches}
    assert "SAM2" in models or "UNet" in models


def test_query_limit(mock_data_dir):
    """Test query result limit."""
    discovery = CapabilityDiscovery(mock_data_dir)
    discovery.reload()

    # Query with limit
    matches = discovery.query("com.multihead", limit=2)

    assert len(matches) <= 2


def test_match_scoring(mock_data_dir):
    """Test match score ordering."""
    discovery = CapabilityDiscovery(mock_data_dir)
    discovery.reload()

    matches = discovery.query("com.multihead.llm")

    # Results should be sorted by match_score descending
    for i in range(len(matches) - 1):
        assert matches[i].match_score >= matches[i + 1].match_score


def test_capability_match_to_dict():
    """Test CapabilityMatch.to_dict() serialization."""
    match = CapabilityMatch(
        agent_id="test-agent",
        capability_id="com.test.llm",
        source="local",
        kind="llm",
        model="test-model",
        performance={"accuracy": 0.95},
        latency_p50_ms=1000,
        match_score=0.9,
    )

    data = match.to_dict()

    assert data["agent_id"] == "test-agent"
    assert data["capability_id"] == "com.test.llm"
    assert data["performance"]["accuracy"] == 0.95
    assert data["match_score"] == 0.9


def test_convenience_function(mock_data_dir):
    """Test discover_capabilities convenience function."""
    matches = discover_capabilities("com.multihead.llm", data_dir=mock_data_dir)

    assert len(matches) >= 3
    for match in matches:
        assert "llm" in match.capability_id


def test_auto_reload(mock_data_dir):
    """Test automatic reload when stale."""
    discovery = CapabilityDiscovery(mock_data_dir)

    # First query triggers reload
    matches1 = discovery.query("com.multihead.llm")
    assert discovery._acp_state is not None

    # Immediate second query uses cache
    load_time_1 = discovery._last_reload
    matches2 = discovery.query("com.multihead.llm")
    load_time_2 = discovery._last_reload
    assert load_time_1 == load_time_2

    # Manual reload updates timestamp (sleep to advance Windows timer resolution)
    import time; time.sleep(0.01)
    discovery.reload()
    load_time_3 = discovery._last_reload
    assert load_time_3 > load_time_2


def test_no_acp_state_graceful(tmp_path):
    """Test graceful handling when acp_state.json doesn't exist."""
    discovery = CapabilityDiscovery(tmp_path)
    discovery.reload()

    # Should not crash, just return empty results
    matches = discovery.query("com.multihead.llm")
    assert matches == []


def test_no_knowledge_db_graceful(mock_data_dir):
    """Test graceful handling when knowledge.db doesn't exist."""
    discovery = CapabilityDiscovery(mock_data_dir)
    discovery.reload()

    # Should query local registry successfully, skip knowledge base
    matches = discovery.query("com.multihead.llm")
    assert len(matches) >= 3  # From acp_state.json only
    assert all(m.source == "local" for m in matches)


def test_extract_model_info():
    """Test model info extraction from claim statements."""
    discovery = CapabilityDiscovery("/tmp")

    # YOLO
    yolo_info = discovery._extract_model_info("YOLOv8m achieving 91.85% mAP50")
    assert yolo_info["name"] == "YOLOv8m"
    assert yolo_info["performance"]["mAP50"] == pytest.approx(0.9185)

    # SAM2
    sam2_info = discovery._extract_model_info("SAM2 segmentation with 10.5s/page processing")
    assert sam2_info["name"] == "SAM2"
    assert sam2_info["latency_ms"] == 10500

    # UNet
    unet_info = discovery._extract_model_info("UNet 99.58% IoU, 0.9s/page")
    assert unet_info["name"] == "UNet"
    assert unet_info["performance"]["iou"] == pytest.approx(0.9958)
    assert unet_info["latency_ms"] == 900

    # No match
    no_match = discovery._extract_model_info("Random text without model info")
    assert no_match is None


def test_semantic_score_calculation():
    """Test semantic match score calculation."""
    discovery = CapabilityDiscovery("/tmp")

    # Exact substring
    score1 = discovery._calculate_semantic_score("YOLO", "YOLOv8m is a detection model")
    assert score1 == 0.9

    # Keyword overlap
    score2 = discovery._calculate_semantic_score(
        "object detection", "YOLO performs object detection",
    )
    assert 0.0 < score2 < 1.0

    # No overlap
    score3 = discovery._calculate_semantic_score("banana", "YOLO detection model")
    assert score3 == 0.0
