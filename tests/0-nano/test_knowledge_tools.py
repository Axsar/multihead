"""Tests for knowledge tools (claims, events, stats)."""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from multihead.knowledge_models import (
    ActorRef,
    Claim,
    ClaimCanonical,
    ClaimScope,
    ClaimStatus,
    ClaimType,
    EntityRef,
    EventStatus,
    EventType,
    KnowledgeEvent,
    Provenance,
    ScopeType,
    Stability,
    TimeBlock,
    TimePrecision,
    ValueObject,
)
from multihead.knowledge_tools import register_knowledge_tools
from multihead.tool_registry import ToolRegistry


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def make_test_claim(
    claim_id: str = "clm_test1",
    statement: str = "MultiHead uses event-sourced orchestration",
    claim_type: ClaimType = ClaimType.FACT,
    claim_status: ClaimStatus = ClaimStatus.ACCEPTED,
    scope_id: str = "multihead-project",
    claim_key: str = "project.multihead.architecture.event_sourced",
    subject_id: str = "multihead",
    confidence: float = 0.9,
) -> Claim:
    """Helper to create a test claim with required fields."""
    return Claim(
        claim_id=claim_id,
        claim_type=claim_type,
        claim_status=claim_status,
        scope=ClaimScope(
            scope_type=ScopeType.PROJECT,
            scope_id=scope_id,
        ),
        canonical=ClaimCanonical(
            claim_key=claim_key,
            subject=EntityRef(entity_type="project", entity_id=subject_id),
            predicate="uses",
            object=ValueObject(value_type="string", value="event-sourced orchestration"),
        ),
        statement=statement,
        confidence=confidence,
        stability=Stability.STABLE,
        provenance=Provenance(
            produced_by={"kind": "extractor", "id": "night_shift"},
        ),
    )


def make_test_event(
    event_id: str = "evt_test1",
    title: str = "Project created",
    summary: str = "MultiHead project was initialized",
    event_type: EventType = EventType.MILESTONE,
    event_status: EventStatus = EventStatus.CONFIRMED,
) -> KnowledgeEvent:
    """Helper to create a test event with required fields."""
    return KnowledgeEvent(
        event_id=event_id,
        event_type=event_type,
        event_status=event_status,
        title=title,
        summary=summary,
        time=TimeBlock(
            happened_at=datetime.now(timezone.utc),
            time_precision=TimePrecision.DAY,
        ),
        actors=[ActorRef(actor_type="user", actor_id="dev1", display="Developer")],
        provenance=Provenance(
            produced_by={"kind": "extractor", "id": "night_shift"},
        ),
    )


# ---------------------------------------------------------------------------
# Test registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_three_tools(self):
        """Registry should have knowledge.claims, knowledge.events, knowledge.stats."""
        registry = ToolRegistry()
        mock_store = MagicMock()

        initial_count = len(registry.list_tools())
        register_knowledge_tools(registry, mock_store)

        assert len(registry.list_tools()) == initial_count + 3
        assert registry.get_spec("knowledge.claims") is not None
        assert registry.get_spec("knowledge.events") is not None
        assert registry.get_spec("knowledge.stats") is not None

    def test_claims_spec(self):
        """Verify knowledge.claims spec has correct params."""
        registry = ToolRegistry()
        mock_store = MagicMock()
        register_knowledge_tools(registry, mock_store)

        spec = registry.get_spec("knowledge.claims")
        assert spec is not None
        assert "search" in spec.params_schema
        assert "status" in spec.params_schema
        assert "claim_type" in spec.params_schema
        assert "limit" in spec.params_schema

    def test_events_spec(self):
        """Verify knowledge.events spec has correct params."""
        registry = ToolRegistry()
        mock_store = MagicMock()
        register_knowledge_tools(registry, mock_store)

        spec = registry.get_spec("knowledge.events")
        assert spec is not None
        assert "search" in spec.params_schema
        assert "event_type" in spec.params_schema
        assert "limit" in spec.params_schema

    def test_stats_spec(self):
        """Verify knowledge.stats spec has empty params."""
        registry = ToolRegistry()
        mock_store = MagicMock()
        register_knowledge_tools(registry, mock_store)

        spec = registry.get_spec("knowledge.stats")
        assert spec is not None
        assert spec.params_schema == {}


# ---------------------------------------------------------------------------
# Test knowledge.claims
# ---------------------------------------------------------------------------


class TestQueryClaims:
    @pytest.mark.asyncio
    async def test_query_claims_basic(self):
        """Returns formatted JSON with claim fields."""
        registry = ToolRegistry()
        mock_store = MagicMock()

        claim1 = make_test_claim(
            claim_id="clm_1",
            statement="Test claim one",
            claim_key="test.claim.one",
        )
        claim2 = make_test_claim(
            claim_id="clm_2",
            statement="Test claim two",
            claim_key="test.claim.two",
        )
        mock_store.list_claims.return_value = [claim1, claim2]

        register_knowledge_tools(registry, mock_store)
        result = await registry.execute("knowledge.claims", {})

        assert result.success
        assert result.tool == "knowledge.claims"

        output = json.loads(result.output)
        assert len(output) == 2
        assert output[0]["claim_id"] == "clm_1"
        assert output[0]["statement"] == "Test claim one"
        assert output[0]["key"] == "test.claim.one"
        assert output[0]["type"] == "fact"
        assert output[0]["status"] == "accepted"
        assert output[0]["confidence"] == 0.9
        assert output[0]["subject"] == "multihead"

    @pytest.mark.asyncio
    async def test_query_claims_with_filters(self):
        """Passes status, claim_type to store."""
        registry = ToolRegistry()
        mock_store = MagicMock()
        mock_store.list_claims.return_value = []

        register_knowledge_tools(registry, mock_store)
        await registry.execute("knowledge.claims", {
            "status": "accepted",
            "claim_type": "decision",
            "limit": 50,
        })

        mock_store.list_claims.assert_called_once_with(
            status="accepted",
            claim_type="decision",
            scope_id=None,
            limit=50,
        )

    @pytest.mark.asyncio
    async def test_query_claims_limit_cap(self):
        """Limit 200 is capped to 100."""
        registry = ToolRegistry()
        mock_store = MagicMock()
        mock_store.list_claims.return_value = []

        register_knowledge_tools(registry, mock_store)
        await registry.execute("knowledge.claims", {"limit": 200})

        mock_store.list_claims.assert_called_once()
        call_kwargs = mock_store.list_claims.call_args[1]
        assert call_kwargs["limit"] == 100

    @pytest.mark.asyncio
    async def test_query_claims_default_limit(self):
        """Default limit is 20."""
        registry = ToolRegistry()
        mock_store = MagicMock()
        mock_store.list_claims.return_value = []

        register_knowledge_tools(registry, mock_store)
        await registry.execute("knowledge.claims", {})

        call_kwargs = mock_store.list_claims.call_args[1]
        assert call_kwargs["limit"] == 20

    @pytest.mark.asyncio
    async def test_query_claims_search_filter_statement(self):
        """Search term filters by statement match (lowercase)."""
        registry = ToolRegistry()
        mock_store = MagicMock()

        claim1 = make_test_claim(
            claim_id="clm_1",
            statement="MultiHead uses GPU mutex",
            subject_id="project-a",
            claim_key="project.a.gpu",
        )
        claim2 = make_test_claim(
            claim_id="clm_2",
            statement="BotVibes is a marketplace",
            subject_id="project-b",
            claim_key="project.b.marketplace",
        )
        claim3 = make_test_claim(
            claim_id="clm_3",
            statement="MultiHead has event sourcing",
            subject_id="project-c",
            claim_key="project.c.eventsourcing",
        )
        mock_store.list_claims.return_value = [claim1, claim2, claim3]

        register_knowledge_tools(registry, mock_store)
        result = await registry.execute("knowledge.claims", {"search": "multihead"})

        assert result.success
        output = json.loads(result.output)
        assert len(output) == 2
        assert output[0]["claim_id"] == "clm_1"
        assert output[1]["claim_id"] == "clm_3"

    @pytest.mark.asyncio
    async def test_query_claims_search_by_key(self):
        """Search matches claim_key."""
        registry = ToolRegistry()
        mock_store = MagicMock()

        claim1 = make_test_claim(
            claim_id="clm_1",
            statement="Something else",
            claim_key="project.multihead.core",
            subject_id="project-x",
        )
        claim2 = make_test_claim(
            claim_id="clm_2",
            statement="Another thing",
            claim_key="project.botvibes.api",
            subject_id="project-y",
        )
        mock_store.list_claims.return_value = [claim1, claim2]

        register_knowledge_tools(registry, mock_store)
        result = await registry.execute("knowledge.claims", {"search": "multihead"})

        assert result.success
        output = json.loads(result.output)
        assert len(output) == 1
        assert output[0]["claim_id"] == "clm_1"

    @pytest.mark.asyncio
    async def test_query_claims_search_by_subject(self):
        """Search matches subject.entity_id."""
        registry = ToolRegistry()
        mock_store = MagicMock()

        claim1 = make_test_claim(
            claim_id="clm_1",
            statement="Uses transformers",
            claim_key="model.qwen.adapter",
            subject_id="qwen-llm",
        )
        claim2 = make_test_claim(
            claim_id="clm_2",
            statement="Uses OpenAI API",
            claim_key="model.gpt.adapter",
            subject_id="openai-gpt4o",
        )
        mock_store.list_claims.return_value = [claim1, claim2]

        register_knowledge_tools(registry, mock_store)
        result = await registry.execute("knowledge.claims", {"search": "qwen"})

        assert result.success
        output = json.loads(result.output)
        assert len(output) == 1
        assert output[0]["claim_id"] == "clm_1"

    @pytest.mark.asyncio
    async def test_query_claims_search_case_insensitive(self):
        """Search is case-insensitive."""
        registry = ToolRegistry()
        mock_store = MagicMock()

        claim1 = make_test_claim(claim_id="clm_1", statement="GPU MUTEX")
        mock_store.list_claims.return_value = [claim1]

        register_knowledge_tools(registry, mock_store)
        result = await registry.execute("knowledge.claims", {"search": "gpu"})

        assert result.success
        output = json.loads(result.output)
        assert len(output) == 1

    @pytest.mark.asyncio
    async def test_query_claims_statement_truncation(self):
        """Statement is truncated to 200 chars in output."""
        registry = ToolRegistry()
        mock_store = MagicMock()

        long_statement = "x" * 300
        claim = make_test_claim(claim_id="clm_1", statement=long_statement)
        mock_store.list_claims.return_value = [claim]

        register_knowledge_tools(registry, mock_store)
        result = await registry.execute("knowledge.claims", {})

        assert result.success
        output = json.loads(result.output)
        assert len(output[0]["statement"]) == 200

    @pytest.mark.asyncio
    async def test_query_claims_error(self):
        """Store raises exception → ToolResult error."""
        registry = ToolRegistry()
        mock_store = MagicMock()
        mock_store.list_claims.side_effect = RuntimeError("Database connection failed")

        register_knowledge_tools(registry, mock_store)
        result = await registry.execute("knowledge.claims", {})

        assert not result.success
        assert result.tool == "knowledge.claims"
        assert "Database connection failed" in result.error


# ---------------------------------------------------------------------------
# Test knowledge.events
# ---------------------------------------------------------------------------


class TestQueryEvents:
    @pytest.mark.asyncio
    async def test_query_events_basic(self):
        """Returns event data in JSON format."""
        registry = ToolRegistry()
        mock_store = MagicMock()

        event1 = make_test_event(
            event_id="evt_1",
            title="Commit pushed",
            summary="Added new feature X",
            event_type=EventType.COMMIT,
        )
        event2 = make_test_event(
            event_id="evt_2",
            title="Task created",
            summary="Implement feature Y",
            event_type=EventType.TASK_CREATED,
        )
        mock_store.list_events.return_value = [event1, event2]

        register_knowledge_tools(registry, mock_store)
        result = await registry.execute("knowledge.events", {})

        assert result.success
        assert result.tool == "knowledge.events"

        output = json.loads(result.output)
        assert len(output) == 2
        assert output[0]["event_id"] == "evt_1"
        assert output[0]["type"] == "commit"
        assert output[0]["title"] == "Commit pushed"
        assert output[0]["summary"] == "Added new feature X"

    @pytest.mark.asyncio
    async def test_query_events_with_filters(self):
        """Passes event_type, limit to store."""
        registry = ToolRegistry()
        mock_store = MagicMock()
        mock_store.list_events.return_value = []

        register_knowledge_tools(registry, mock_store)
        await registry.execute("knowledge.events", {
            "event_type": "decision",
            "limit": 30,
        })

        mock_store.list_events.assert_called_once_with(
            event_type="decision",
            status=None,
            limit=30,
        )

    @pytest.mark.asyncio
    async def test_query_events_limit_cap(self):
        """Limit 150 is capped to 100."""
        registry = ToolRegistry()
        mock_store = MagicMock()
        mock_store.list_events.return_value = []

        register_knowledge_tools(registry, mock_store)
        await registry.execute("knowledge.events", {"limit": 150})

        call_kwargs = mock_store.list_events.call_args[1]
        assert call_kwargs["limit"] == 100

    @pytest.mark.asyncio
    async def test_query_events_with_search_title(self):
        """Search filters by title (lowercase)."""
        registry = ToolRegistry()
        mock_store = MagicMock()

        event1 = make_test_event(event_id="evt_1", title="Deploy to production")
        event2 = make_test_event(event_id="evt_2", title="Deploy to staging")
        event3 = make_test_event(event_id="evt_3", title="Code review completed")
        mock_store.list_events.return_value = [event1, event2, event3]

        register_knowledge_tools(registry, mock_store)
        result = await registry.execute("knowledge.events", {"search": "deploy"})

        assert result.success
        output = json.loads(result.output)
        assert len(output) == 2
        assert output[0]["event_id"] == "evt_1"
        assert output[1]["event_id"] == "evt_2"

    @pytest.mark.asyncio
    async def test_query_events_with_search_summary(self):
        """Search filters by summary."""
        registry = ToolRegistry()
        mock_store = MagicMock()

        event1 = make_test_event(
            event_id="evt_1",
            title="Event A",
            summary="Fixed critical bug in GPU mutex",
        )
        event2 = make_test_event(
            event_id="evt_2",
            title="Event B",
            summary="Updated documentation",
        )
        mock_store.list_events.return_value = [event1, event2]

        register_knowledge_tools(registry, mock_store)
        result = await registry.execute("knowledge.events", {"search": "gpu"})

        assert result.success
        output = json.loads(result.output)
        assert len(output) == 1
        assert output[0]["event_id"] == "evt_1"

    @pytest.mark.asyncio
    async def test_query_events_search_case_insensitive(self):
        """Search is case-insensitive."""
        registry = ToolRegistry()
        mock_store = MagicMock()

        event1 = make_test_event(event_id="evt_1", title="GPU Mutex Update")
        mock_store.list_events.return_value = [event1]

        register_knowledge_tools(registry, mock_store)
        result = await registry.execute("knowledge.events", {"search": "gpu"})

        assert result.success
        output = json.loads(result.output)
        assert len(output) == 1

    @pytest.mark.asyncio
    async def test_query_events_title_truncation(self):
        """Title is truncated to 160 chars in output."""
        registry = ToolRegistry()
        mock_store = MagicMock()

        long_title = "x" * 200
        event = make_test_event(event_id="evt_1", title=long_title)
        mock_store.list_events.return_value = [event]

        register_knowledge_tools(registry, mock_store)
        result = await registry.execute("knowledge.events", {})

        assert result.success
        output = json.loads(result.output)
        assert len(output[0]["title"]) == 160

    @pytest.mark.asyncio
    async def test_query_events_summary_truncation(self):
        """Summary is truncated to 200 chars in output."""
        registry = ToolRegistry()
        mock_store = MagicMock()

        long_summary = "y" * 250
        event = make_test_event(event_id="evt_1", summary=long_summary)
        mock_store.list_events.return_value = [event]

        register_knowledge_tools(registry, mock_store)
        result = await registry.execute("knowledge.events", {})

        assert result.success
        output = json.loads(result.output)
        assert len(output[0]["summary"]) == 200

    @pytest.mark.asyncio
    async def test_query_events_error(self):
        """Error handling when store raises exception."""
        registry = ToolRegistry()
        mock_store = MagicMock()
        mock_store.list_events.side_effect = Exception("Query timeout")

        register_knowledge_tools(registry, mock_store)
        result = await registry.execute("knowledge.events", {})

        assert not result.success
        assert result.tool == "knowledge.events"
        assert "Query timeout" in result.error


# ---------------------------------------------------------------------------
# Test knowledge.stats
# ---------------------------------------------------------------------------


class TestKnowledgeStats:
    @pytest.mark.asyncio
    async def test_knowledge_stats(self):
        """Returns counts breakdown."""
        registry = ToolRegistry()
        mock_store = MagicMock()

        claim1 = make_test_claim(
            claim_id="clm_1",
            claim_type=ClaimType.FACT,
            claim_status=ClaimStatus.ACCEPTED,
        )
        claim2 = make_test_claim(
            claim_id="clm_2",
            claim_type=ClaimType.FACT,
            claim_status=ClaimStatus.PROPOSED,
        )
        claim3 = make_test_claim(
            claim_id="clm_3",
            claim_type=ClaimType.DECISION,
            claim_status=ClaimStatus.ACCEPTED,
        )

        event1 = make_test_event(event_id="evt_1", event_type=EventType.COMMIT)
        event2 = make_test_event(event_id="evt_2", event_type=EventType.COMMIT)
        event3 = make_test_event(event_id="evt_3", event_type=EventType.MILESTONE)

        mock_store.list_claims.return_value = [claim1, claim2, claim3]
        mock_store.list_events.return_value = [event1, event2, event3]

        register_knowledge_tools(registry, mock_store)
        result = await registry.execute("knowledge.stats", {})

        assert result.success
        assert result.tool == "knowledge.stats"

        stats = json.loads(result.output)
        assert stats["total_claims"] == 3
        assert stats["total_events"] == 3
        assert stats["claim_types"]["fact"] == 2
        assert stats["claim_types"]["decision"] == 1
        assert stats["claim_statuses"]["accepted"] == 2
        assert stats["claim_statuses"]["proposed"] == 1
        assert stats["event_types"]["commit"] == 2
        assert stats["event_types"]["milestone"] == 1

    @pytest.mark.asyncio
    async def test_knowledge_stats_empty(self):
        """Empty store returns zero counts."""
        registry = ToolRegistry()
        mock_store = MagicMock()
        mock_store.list_claims.return_value = []
        mock_store.list_events.return_value = []

        register_knowledge_tools(registry, mock_store)
        result = await registry.execute("knowledge.stats", {})

        assert result.success
        stats = json.loads(result.output)
        assert stats["total_claims"] == 0
        assert stats["total_events"] == 0
        assert stats["claim_types"] == {}
        assert stats["claim_statuses"] == {}
        assert stats["event_types"] == {}

    @pytest.mark.asyncio
    async def test_knowledge_stats_fetches_large_limit(self):
        """Stats queries with limit=10000 to get all data."""
        registry = ToolRegistry()
        mock_store = MagicMock()
        mock_store.list_claims.return_value = []
        mock_store.list_events.return_value = []

        register_knowledge_tools(registry, mock_store)
        await registry.execute("knowledge.stats", {})

        # Should fetch with high limit to get all claims/events
        mock_store.list_claims.assert_called_once_with(limit=10000)
        mock_store.list_events.assert_called_once_with(limit=10000)

    @pytest.mark.asyncio
    async def test_knowledge_stats_error(self):
        """Error handling when stats query fails."""
        registry = ToolRegistry()
        mock_store = MagicMock()
        mock_store.list_claims.side_effect = Exception("Stats calculation failed")

        register_knowledge_tools(registry, mock_store)
        result = await registry.execute("knowledge.stats", {})

        assert not result.success
        assert result.tool == "knowledge.stats"
        assert "Stats calculation failed" in result.error
