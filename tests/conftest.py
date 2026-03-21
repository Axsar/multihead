"""Shared fixtures and factory functions for MultiHead test suite."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from multihead.adapters.mock import MockAdapter
from multihead.api.app import create_app
from multihead.artifact_store import ArtifactStore
from multihead.config import Settings
from multihead.context_packs import PackBuilder
from multihead.event_store import EventStore
from multihead.head_manager import HeadManager
from multihead.knowledge_models import (
    Claim,
    ClaimCanonical,
    ClaimScope,
    ClaimStatus,
    ClaimType,
    EntityRef,
    EvidencePointer,
    EventType,
    KnowledgeEvent,
    Provenance,
    Record,
    ScopeType,
    SpanRef,
    TimeBlock,
    ValueObject,
)
from multihead.knowledge_store import KnowledgeStore
from multihead.models import AdapterKind, HeadManifest
from multihead.record_store import RecordStore
from multihead.session import SessionManager
from multihead.tool_registry import ToolRegistry


# ---------------------------------------------------------------------------
# Factory functions (not fixtures — call these directly in tests)
# ---------------------------------------------------------------------------


def make_prov(**overrides) -> Provenance:
    """Create a test Provenance."""
    defaults = {"produced_by": {"kind": "test", "id": "unit"}}
    defaults.update(overrides)
    return Provenance(**defaults)


def make_record(uri: str = "file:///test.txt") -> Record:
    """Create a test Record."""
    return Record(uri=uri)


def make_evidence(record_id: str, uri: str = "file:///test.txt") -> EvidencePointer:
    """Create a test EvidencePointer."""
    return EvidencePointer(
        record_id=record_id,
        uri=uri,
        span=SpanRef(start=0, end=100),
    )


def make_event(title: str = "Test event", **overrides) -> KnowledgeEvent:
    """Create a test KnowledgeEvent."""
    defaults = dict(
        event_type=EventType.DECISION,
        title=title,
        time=TimeBlock(happened_at=datetime.now(timezone.utc)),
        provenance=make_prov(),
    )
    defaults.update(overrides)
    return KnowledgeEvent(**defaults)


def make_claim(
    claim_key: str = "test.key",
    statement: str = "Test claim.",
    **overrides,
) -> Claim:
    """Create a test Claim."""
    defaults = dict(
        claim_type=ClaimType.DECISION,
        scope=ClaimScope(scope_type=ScopeType.PROJECT, scope_id="multihead"),
        canonical=ClaimCanonical(
            claim_key=claim_key,
            subject=EntityRef(entity_type="component", entity_id="core"),
            predicate="is",
            object=ValueObject(value_type="string", value="true"),
        ),
        statement=statement,
        provenance=make_prov(),
    )
    defaults.update(overrides)
    return Claim(**defaults)


def seed_records(rs: RecordStore, count: int = 3) -> list[str]:
    """Insert test records and return their record_ids."""
    ids = []
    for i in range(count):
        rec = rs.ingest_text(
            f"Record {i}: MultiHead uses event sourcing. The core LLM runs on CPU. "
            f"VRAM is reserved for worker heads. This is test data entry number {i}.",
            uri=f"test://record_{i}",
        )
        ids.append(rec.record_id)
    return ids


# ---------------------------------------------------------------------------
# Manifest fixtures
# ---------------------------------------------------------------------------

MOCK_LLM_MANIFEST = HeadManifest(
    head_id="mock-llm",
    name="Mock LLM",
    adapter=AdapterKind.MOCK,
    model="mock-v1",
    kind="llm",
    gpu_required=False,
)

MOCK_VLM_MANIFEST = HeadManifest(
    head_id="mock-vlm",
    name="Mock VLM",
    adapter=AdapterKind.MOCK,
    model="mock-v1",
    kind="vlm",
    gpu_required=False,
)


@pytest.fixture
def mock_manifests() -> dict[str, HeadManifest]:
    """Two mock heads: mock-llm and mock-vlm."""
    return {
        "mock-llm": MOCK_LLM_MANIFEST,
        "mock-vlm": MOCK_VLM_MANIFEST,
    }


@pytest.fixture
def mock_head_manager(mock_manifests) -> HeadManager:
    """HeadManager with mock-llm and mock-vlm."""
    return HeadManager(mock_manifests)


@pytest.fixture
def mock_adapter() -> MockAdapter:
    """A single MockAdapter instance."""
    return MockAdapter(MOCK_LLM_MANIFEST)


# ---------------------------------------------------------------------------
# Store fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def knowledge_store(tmp_path) -> KnowledgeStore:
    return KnowledgeStore(tmp_path / "knowledge.db")


@pytest.fixture
def artifact_store(tmp_path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts", tmp_path / "artifacts.db")


@pytest.fixture
def event_store(tmp_path) -> EventStore:
    return EventStore(tmp_path / "runs", tmp_path / "state.db")


@pytest.fixture
def record_store(knowledge_store, artifact_store) -> RecordStore:
    return RecordStore(knowledge_store, artifact_store)


@pytest.fixture
def pack_builder(knowledge_store, tmp_path) -> PackBuilder:
    return PackBuilder(knowledge_store, tmp_path / "packs")


@pytest.fixture
def session_manager(tmp_path) -> SessionManager:
    return SessionManager(tmp_path / "sessions")


@pytest.fixture
def tool_registry() -> ToolRegistry:
    return ToolRegistry()


# ---------------------------------------------------------------------------
# API client fixture
# ---------------------------------------------------------------------------

HEADS_YAML = """\
heads:
  - head_id: mock-llm
    name: Mock LLM
    adapter: mock
    model: mock-v1
    kind: llm
    gpu_required: false
  - head_id: mock-vlm
    name: Mock VLM
    adapter: mock
    model: mock-v1
    kind: vlm
    gpu_required: false
"""


@pytest.fixture
def api_client(tmp_path) -> TestClient:
    """Full API TestClient with mock heads and a test recipe."""
    settings = Settings(
        data_dir=tmp_path / "data",
        config_dir=tmp_path / "config",
    )
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    recipes_dir = config_dir / "recipes"
    recipes_dir.mkdir()

    (config_dir / "heads.yaml").write_text(HEADS_YAML)
    (recipes_dir / "test-pipeline.yaml").write_text("""\
goal: "Test pipeline"
steps:
  - name: plan
    head_id: mock-llm
    prompt_template: "Create a plan for testing"
  - name: extract
    head_id: mock-vlm
    prompt_template: "Extract data"
""")

    app = create_app(settings)
    with TestClient(app) as c:
        yield c
