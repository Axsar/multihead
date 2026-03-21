"""Tests for in-process MCP tools exposed to Claude SDK sessions.

Verifies that knowledge, head management, and process tools work
correctly when called through the SDK MCP tool interface.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from multihead.sdk_mcp_tools import (
    _build_head_tools,
    _build_knowledge_tools,
    _build_process_tools,
    build_sdk_mcp_server,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ks():
    """Mock KnowledgeStore."""
    ks = MagicMock()
    claim1 = MagicMock()
    claim1.statement = "The engine supports hot-swapping GPU models"
    claim1.confidence = 0.9
    claim1.canonical = MagicMock()
    claim1.canonical.claim_key = "engine.hotswap"

    claim2 = MagicMock()
    claim2.statement = "Router uses weighted scoring for head selection"
    claim2.confidence = 0.85
    claim2.canonical = MagicMock()
    claim2.canonical.claim_key = "router.scoring"

    ks.list_claims.return_value = [claim1, claim2]

    deposited = MagicMock()
    deposited.claim_id = "clm_test_123"
    ks.deposit_claim.return_value = deposited

    return ks


@pytest.fixture
def mock_hm():
    """Mock HeadManager."""
    hm = MagicMock()
    hm.get_states.return_value = {
        "core-llm": {"state": "active", "name": "Qwen3 8B", "adapter": "transformers"},
        "mock-llm": {"state": "off", "name": "Mock LLM", "adapter": "mock"},
    }
    hm.wake_head = AsyncMock()
    hm.sleep_head = AsyncMock()
    hm.ensure_active = AsyncMock()
    hm.generate = AsyncMock(return_value={"text": "Generated output from local model"})
    return hm


@pytest.fixture
def mock_pm():
    """Mock ProcessManager."""
    pm = MagicMock()
    proc = MagicMock()
    proc.pid = 12345
    proc.command = "echo hello"
    proc.status = "running"
    pm.spawn = AsyncMock(return_value=proc)
    pm.list_processes.return_value = [proc]
    pm.output.return_value = "hello\n"
    pm.kill = AsyncMock()
    return pm


# ---------------------------------------------------------------------------
# build_sdk_mcp_server
# ---------------------------------------------------------------------------


class TestBuildServer:
    """Test server creation."""

    def test_build_with_all_components(self, mock_ks, mock_hm, mock_pm):
        server = build_sdk_mcp_server(
            knowledge_store=mock_ks,
            head_manager=mock_hm,
            process_manager=mock_pm,
        )
        assert server is not None
        # Server config should have type "sdk"
        assert server.get("type") == "sdk" or hasattr(server, "type")

    def test_build_with_knowledge_only(self, mock_ks):
        server = build_sdk_mcp_server(knowledge_store=mock_ks)
        assert server is not None

    def test_build_with_heads_only(self, mock_hm):
        server = build_sdk_mcp_server(head_manager=mock_hm)
        assert server is not None

    def test_build_with_nothing(self):
        server = build_sdk_mcp_server()
        assert server is not None

    def test_build_without_sdk_raises(self):
        with patch("multihead.sdk_mcp_tools._SDK_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="claude-agent-sdk not installed"):
                build_sdk_mcp_server()


# ---------------------------------------------------------------------------
# Knowledge tools
# ---------------------------------------------------------------------------


class TestKnowledgeTools:
    """Knowledge store MCP tools."""

    def test_builds_three_tools(self, mock_ks):
        tools = _build_knowledge_tools(mock_ks, "multihead")
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert "knowledge_query" in names
        assert "knowledge_deposit" in names
        assert "knowledge_count" in names

    async def test_query_finds_matching_claims(self, mock_ks):
        tools = _build_knowledge_tools(mock_ks, "multihead")
        query_tool = next(t for t in tools if t.name == "knowledge_query")
        result = await query_tool.handler({"query": "engine hotswap"})
        text = result["content"][0]["text"]
        assert "engine.hotswap" in text
        assert "hot-swapping" in text

    async def test_query_no_results(self, mock_ks):
        mock_ks.list_claims.return_value = []
        tools = _build_knowledge_tools(mock_ks, "multihead")
        query_tool = next(t for t in tools if t.name == "knowledge_query")
        result = await query_tool.handler({"query": "nonexistent topic"})
        text = result["content"][0]["text"]
        assert "No claims found" in text

    async def test_deposit_creates_claim(self, mock_ks):
        tools = _build_knowledge_tools(mock_ks, "multihead")
        deposit_tool = next(t for t in tools if t.name == "knowledge_deposit")
        result = await deposit_tool.handler({
            "claim_key": "test.fact.something",
            "statement": "This is a test fact",
            "claim_type": "fact",
        })
        text = result["content"][0]["text"]
        assert "deposited" in text.lower()
        mock_ks.deposit_claim.assert_called_once()

    async def test_count_returns_total(self, mock_ks):
        # knowledge_count uses ks._connect() for SQL COUNT — mock the context manager
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = (2,)
        mock_ks._connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_ks._connect.return_value.__exit__ = MagicMock(return_value=False)

        tools = _build_knowledge_tools(mock_ks, "multihead")
        count_tool = next(t for t in tools if t.name == "knowledge_count")
        result = await count_tool.handler({})
        text = result["content"][0]["text"]
        assert "2 claims" in text

    async def test_count_with_scope(self, mock_ks):
        # knowledge_count uses ks._connect() for SQL COUNT with scope filter
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = (5,)
        mock_ks._connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_ks._connect.return_value.__exit__ = MagicMock(return_value=False)

        tools = _build_knowledge_tools(mock_ks, "multihead")
        count_tool = next(t for t in tools if t.name == "knowledge_count")
        result = await count_tool.handler({"scope_id": "h2v"})
        text = result["content"][0]["text"]
        assert "h2v" in text
        # Verify SQL was called with scope filter
        mock_conn.execute.assert_called_with(
            "SELECT COUNT(*) FROM claims WHERE scope_id = ?", ("h2v",)
        )

    async def test_query_respects_limit(self, mock_ks):
        tools = _build_knowledge_tools(mock_ks, "multihead")
        query_tool = next(t for t in tools if t.name == "knowledge_query")
        result = await query_tool.handler({"query": "engine", "limit": 1})
        text = result["content"][0]["text"]
        assert "1 claims" in text


# ---------------------------------------------------------------------------
# Head management tools
# ---------------------------------------------------------------------------


class TestHeadTools:
    """Head management MCP tools."""

    def test_builds_five_tools(self, mock_hm):
        tools = _build_head_tools(mock_hm)
        assert len(tools) == 5
        names = {t.name for t in tools}
        assert "heads_list" in names
        assert "heads_wake" in names
        assert "heads_sleep" in names
        assert "heads_swap" in names
        assert "heads_generate" in names

    async def test_list_shows_all_heads(self, mock_hm):
        tools = _build_head_tools(mock_hm)
        list_tool = next(t for t in tools if t.name == "heads_list")
        result = await list_tool.handler({})
        text = result["content"][0]["text"]
        assert "core-llm" in text
        assert "mock-llm" in text
        assert "active" in text

    async def test_wake_calls_head_manager(self, mock_hm):
        tools = _build_head_tools(mock_hm)
        wake_tool = next(t for t in tools if t.name == "heads_wake")
        result = await wake_tool.handler({"head_id": "core-llm"})
        mock_hm.wake_head.assert_called_once_with("core-llm")
        assert "waking" in result["content"][0]["text"].lower()

    async def test_sleep_calls_head_manager(self, mock_hm):
        tools = _build_head_tools(mock_hm)
        sleep_tool = next(t for t in tools if t.name == "heads_sleep")
        await sleep_tool.handler({"head_id": "core-llm"})
        mock_hm.sleep_head.assert_called_once_with("core-llm")

    async def test_swap_calls_ensure_active(self, mock_hm):
        tools = _build_head_tools(mock_hm)
        swap_tool = next(t for t in tools if t.name == "heads_swap")
        await swap_tool.handler({"head_id": "mock-llm"})
        mock_hm.ensure_active.assert_called_once_with("mock-llm")

    async def test_generate_calls_head(self, mock_hm):
        tools = _build_head_tools(mock_hm)
        gen_tool = next(t for t in tools if t.name == "heads_generate")
        result = await gen_tool.handler({"head_id": "core-llm", "prompt": "Hello"})
        mock_hm.generate.assert_called_once_with("core-llm", "Hello")
        assert "Generated output" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# Process management tools
# ---------------------------------------------------------------------------


class TestProcessTools:
    """Process management MCP tools."""

    def test_builds_four_tools(self, mock_pm):
        tools = _build_process_tools(mock_pm)
        assert len(tools) == 4
        names = {t.name for t in tools}
        assert "process_spawn" in names
        assert "process_list" in names
        assert "process_output" in names
        assert "process_kill" in names

    async def test_spawn_returns_pid(self, mock_pm):
        tools = _build_process_tools(mock_pm)
        spawn_tool = next(t for t in tools if t.name == "process_spawn")
        result = await spawn_tool.handler({"command": "echo hello"})
        text = result["content"][0]["text"]
        assert "12345" in text
        mock_pm.spawn.assert_called_once_with("echo hello")

    async def test_list_shows_processes(self, mock_pm):
        tools = _build_process_tools(mock_pm)
        list_tool = next(t for t in tools if t.name == "process_list")
        result = await list_tool.handler({})
        text = result["content"][0]["text"]
        assert "12345" in text
        assert "running" in text

    async def test_list_empty(self, mock_pm):
        mock_pm.list_processes.return_value = []
        tools = _build_process_tools(mock_pm)
        list_tool = next(t for t in tools if t.name == "process_list")
        result = await list_tool.handler({})
        assert "No running" in result["content"][0]["text"]

    async def test_output_reads_process(self, mock_pm):
        tools = _build_process_tools(mock_pm)
        output_tool = next(t for t in tools if t.name == "process_output")
        result = await output_tool.handler({"pid": 12345})
        assert "hello" in result["content"][0]["text"]

    async def test_kill_calls_manager(self, mock_pm):
        tools = _build_process_tools(mock_pm)
        kill_tool = next(t for t in tools if t.name == "process_kill")
        await kill_tool.handler({"pid": 12345})
        mock_pm.kill.assert_called_once_with(12345)


# ---------------------------------------------------------------------------
# Tool schema validation
# ---------------------------------------------------------------------------


class TestToolSchemas:
    """Verify tool schemas are well-formed."""

    def test_all_tools_have_descriptions(self, mock_ks, mock_hm, mock_pm):
        all_tools = (
            _build_knowledge_tools(mock_ks, "multihead")
            + _build_head_tools(mock_hm)
            + _build_process_tools(mock_pm)
        )
        for tool in all_tools:
            assert tool.description, f"Tool {tool.name} has no description"
            assert len(tool.description) > 10, f"Tool {tool.name} description too short"

    def test_all_tools_have_schemas(self, mock_ks, mock_hm, mock_pm):
        all_tools = (
            _build_knowledge_tools(mock_ks, "multihead")
            + _build_head_tools(mock_hm)
            + _build_process_tools(mock_pm)
        )
        for tool in all_tools:
            schema = tool.input_schema
            assert isinstance(schema, dict), f"Tool {tool.name} schema is not a dict"
            assert schema.get("type") == "object", f"Tool {tool.name} schema not an object"

    def test_total_tool_count(self, mock_ks, mock_hm, mock_pm):
        all_tools = (
            _build_knowledge_tools(mock_ks, "multihead")
            + _build_head_tools(mock_hm)
            + _build_process_tools(mock_pm)
        )
        assert len(all_tools) == 12  # 3 knowledge + 5 heads + 4 process
