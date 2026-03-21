"""Tests for MCP server tools (HTTP proxy + direct SQLite knowledge tools)."""

import json

import httpx
import pytest

import multihead.mcp_server as mcp_mod
from multihead.mcp_server import (
    _briefing,
    _chat,
    _config,
    _deposit_claim,
    _generate,
    _heads,
    _knowledge,
    _report_event,
    _run_recipe,
    _run_status,
    _swap_head,
)
from multihead.knowledge_store import KnowledgeStore


# ---------------------------------------------------------------------------
# Fixture: temp KnowledgeStore injected into the module-level _knowledge_store
# ---------------------------------------------------------------------------

@pytest.fixture()
def ks_tmp(tmp_path):
    """Create a temp KnowledgeStore and patch it into mcp_server._knowledge_store."""
    db_path = tmp_path / "knowledge.db"
    ks = KnowledgeStore(db_path)
    old = mcp_mod._knowledge_store
    mcp_mod._knowledge_store = ks
    yield ks
    mcp_mod._knowledge_store = old


# ---------------------------------------------------------------------------
# HTTP-proxied tools (unchanged — still need server)
# ---------------------------------------------------------------------------

class TestChat:
    @pytest.mark.asyncio
    async def test_chat_success(self, httpx_mock):
        httpx_mock.add_response(
            url="http://127.0.0.1:7337/chat",
            json={"session_id": "s1", "response": "Hello!"},
        )
        result = await _chat("Hi there")
        data = json.loads(result)
        assert data["response"] == "Hello!"

    @pytest.mark.asyncio
    async def test_chat_with_session(self, httpx_mock):
        httpx_mock.add_response(
            url="http://127.0.0.1:7337/chat",
            json={"session_id": "s1", "response": "Continued"},
        )
        result = await _chat("Continue", session_id="s1")
        data = json.loads(result)
        assert data["session_id"] == "s1"

    @pytest.mark.asyncio
    async def test_chat_server_down(self, httpx_mock):
        httpx_mock.add_exception(httpx.ConnectError("Connection refused"))
        result = await _chat("test")
        assert "not running" in result


class TestGenerate:
    @pytest.mark.asyncio
    async def test_generate_success(self, httpx_mock):
        httpx_mock.add_response(
            url="http://127.0.0.1:7337/heads/qwen-llm/generate",
            json={"text": "Generated text", "tokens_in": 10, "tokens_out": 20},
        )
        result = await _generate("qwen-llm", "Hello")
        data = json.loads(result)
        assert data["text"] == "Generated text"

    @pytest.mark.asyncio
    async def test_generate_not_found(self, httpx_mock):
        httpx_mock.add_response(
            url="http://127.0.0.1:7337/heads/nonexistent/generate",
            status_code=404,
        )
        result = await _generate("nonexistent", "Hello")
        assert "not found" in result


class TestHeads:
    @pytest.mark.asyncio
    async def test_heads_success(self, httpx_mock):
        httpx_mock.add_response(
            url="http://127.0.0.1:7337/heads",
            json={"mock-llm": {"state": "OFF"}, "qwen-llm": {"state": "ACTIVE"}},
        )
        result = await _heads()
        data = json.loads(result)
        assert "mock-llm" in data
        assert data["qwen-llm"]["state"] == "ACTIVE"


class TestSwapHead:
    @pytest.mark.asyncio
    async def test_wake(self, httpx_mock):
        httpx_mock.add_response(
            url="http://127.0.0.1:7337/heads/qwen-llm/wake",
            json={"head_id": "qwen-llm", "state": "ACTIVE"},
        )
        result = await _swap_head("qwen-llm", "wake")
        data = json.loads(result)
        assert data["state"] == "ACTIVE"

    @pytest.mark.asyncio
    async def test_invalid_action(self):
        result = await _swap_head("qwen-llm", "invalid")
        assert "Error" in result


class TestRunRecipe:
    @pytest.mark.asyncio
    async def test_run(self, httpx_mock):
        httpx_mock.add_response(
            url="http://127.0.0.1:7337/runs",
            json={"run_id": "r1", "status": "completed"},
        )
        result = await _run_recipe("test-pipeline")
        data = json.loads(result)
        assert data["run_id"] == "r1"


class TestRunStatus:
    @pytest.mark.asyncio
    async def test_status(self, httpx_mock):
        httpx_mock.add_response(
            url="http://127.0.0.1:7337/runs/r1",
            json={"run_id": "r1", "status": "completed"},
        )
        result = await _run_status("r1")
        data = json.loads(result)
        assert data["status"] == "completed"


class TestConfig:
    @pytest.mark.asyncio
    async def test_show(self, httpx_mock):
        httpx_mock.add_response(
            url="http://127.0.0.1:7337/config",
            json={"web_tools_enabled": True, "disabled_tools": []},
        )
        result = await _config("show")
        data = json.loads(result)
        assert data["web_tools_enabled"] is True

    @pytest.mark.asyncio
    async def test_set(self, httpx_mock):
        httpx_mock.add_response(
            url="http://127.0.0.1:7337/config/set",
            json={"key": "generation.temperature", "value": "0.5", "message": "ok"},
        )
        result = await _config("set", key="generation.temperature", value="0.5")
        data = json.loads(result)
        assert data["key"] == "generation.temperature"

    @pytest.mark.asyncio
    async def test_set_missing_params(self):
        result = await _config("set")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_invalid_action(self):
        result = await _config("invalid")
        assert "Error" in result


# ---------------------------------------------------------------------------
# Direct-SQLite knowledge tools (no server needed)
# ---------------------------------------------------------------------------

class TestKnowledge:
    @pytest.mark.asyncio
    async def test_claims_empty(self, ks_tmp):
        result = await _knowledge("claims")
        data = json.loads(result)
        assert data == []

    @pytest.mark.asyncio
    async def test_events_empty(self, ks_tmp):
        result = await _knowledge("events")
        data = json.loads(result)
        assert data == []

    @pytest.mark.asyncio
    async def test_invalid_type(self):
        result = await _knowledge("invalid")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_claims_with_data(self, ks_tmp):
        # Deposit a claim first, then query
        await _deposit_claim("test.key", "Test statement")
        result = await _knowledge("claims")
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["claim_key"] == "test.key"
        assert data[0]["statement"] == "Test statement"
        assert data[0]["claim_status"] == "accepted"

    @pytest.mark.asyncio
    async def test_claims_filter_status(self, ks_tmp):
        await _deposit_claim("test.key", "Test statement")
        # Query with non-matching status
        result = await _knowledge("claims", status="proposed")
        data = json.loads(result)
        assert data == []
        # Query with matching status
        result = await _knowledge("claims", status="accepted")
        data = json.loads(result)
        assert len(data) == 1

    @pytest.mark.asyncio
    async def test_events_with_data(self, ks_tmp):
        await _report_event("Test event", summary="Something happened")
        result = await _knowledge("events")
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["title"] == "Test event"
        assert data[0]["event_status"] == "confirmed"


class TestDepositClaim:
    @pytest.mark.asyncio
    async def test_basic(self, ks_tmp):
        result = await _deposit_claim("h2v.test.status", "Balloon layout works")
        data = json.loads(result)
        assert data["claim_key"] == "h2v.test.status"
        assert data["statement"] == "Balloon layout works"
        assert data["claim_status"] == "accepted"
        assert data["claim_id"].startswith("clm_")

    @pytest.mark.asyncio
    async def test_custom_type(self, ks_tmp):
        result = await _deposit_claim(
            "h2v.arch.decision", "Use SQLite for knowledge",
            claim_type="decision", confidence=0.95,
        )
        data = json.loads(result)
        assert data["claim_id"].startswith("clm_")
        # Verify it's actually in the DB
        claims = ks_tmp.list_claims(claim_type="decision")
        assert len(claims) == 1
        assert claims[0].confidence == 0.95

    @pytest.mark.asyncio
    async def test_persists_in_db(self, ks_tmp):
        await _deposit_claim("test.persist", "Persisted claim")
        claims = ks_tmp.list_claims()
        assert len(claims) == 1
        assert claims[0].statement == "Persisted claim"


class TestReportEvent:
    @pytest.mark.asyncio
    async def test_basic(self, ks_tmp):
        result = await _report_event("Deploy completed")
        data = json.loads(result)
        assert data["title"] == "Deploy completed"
        assert data["event_type"] == "note"
        assert data["event_status"] == "confirmed"
        assert data["event_id"].startswith("evt_")

    @pytest.mark.asyncio
    async def test_with_type_and_summary(self, ks_tmp):
        result = await _report_event(
            "Tests passing", summary="All 1434 tests pass",
            event_type="milestone",
        )
        data = json.loads(result)
        assert data["event_type"] == "milestone"
        # Verify in DB
        events = ks_tmp.list_events()
        assert len(events) == 1
        assert events[0].summary == "All 1434 tests pass"

    @pytest.mark.asyncio
    async def test_with_tags(self, ks_tmp):
        await _report_event("Tagged event", tags=["ci", "deploy"])
        events = ks_tmp.list_events()
        assert events[0].tags == ["ci", "deploy"]


class TestBriefing:
    @pytest.mark.asyncio
    async def test_empty_db(self, ks_tmp):
        result = await _briefing("balloon_layout")
        data = json.loads(result)
        assert data["component"] == "balloon_layout"
        assert data["claims"] == []
        assert data["related_claims"] == []
        assert data["recent_events"] == []

    @pytest.mark.asyncio
    async def test_direct_claim_match(self, ks_tmp):
        """Claims with component name in key are 'direct' claims."""
        await _deposit_claim("h2v.balloon_layout.status", "Layout is stable")
        result = await _briefing("balloon_layout")
        data = json.loads(result)
        assert len(data["claims"]) == 1
        assert data["claims"][0]["statement"] == "Layout is stable"

    @pytest.mark.asyncio
    async def test_related_claim_match(self, ks_tmp):
        """Claims mentioning component in statement are 'related' claims."""
        await _deposit_claim("h2v.rendering.pipeline", "Uses balloon_layout for positioning")
        result = await _briefing("balloon_layout")
        data = json.loads(result)
        assert len(data["related_claims"]) == 1
        assert "balloon_layout" in data["related_claims"][0]["statement"]

    @pytest.mark.asyncio
    async def test_event_match(self, ks_tmp):
        """Events mentioning component in title/summary are included."""
        await _report_event("Fixed balloon_layout overflow", summary="Resolved text clipping")
        result = await _briefing("balloon_layout")
        data = json.loads(result)
        assert len(data["recent_events"]) == 1
        assert "balloon_layout" in data["recent_events"][0]["title"]

    @pytest.mark.asyncio
    async def test_summary_counts(self, ks_tmp):
        await _deposit_claim("h2v.tails.length", "Tails are always 3 points")
        await _deposit_claim("h2v.rendering.info", "Rendering uses tails connector")
        await _report_event("tails updated")
        result = await _briefing("tails")
        data = json.loads(result)
        assert "1 direct" in data["summary"]
        assert "1 related" in data["summary"]
        assert "1 events" in data["summary"]
