"""Tests for ACP tools registration and execution."""

import json

import httpx
import pytest

from multihead.acp_tools import ACP_CREATE_TASK_SPEC, _tool_acp_create_task, register_acp_tools
from multihead.tool_registry import ToolRegistry


class TestACPCreateTaskSpec:
    """Tests for the ACP_CREATE_TASK_SPEC definition."""

    def test_spec_name(self):
        assert ACP_CREATE_TASK_SPEC.name == "acp.create_task"

    def test_spec_description(self):
        assert "BotVibes" in ACP_CREATE_TASK_SPEC.description
        assert "claude-session-agent" in ACP_CREATE_TASK_SPEC.description

    def test_spec_has_required_params(self):
        schema = ACP_CREATE_TASK_SPEC.params_schema
        assert "capability" in schema
        assert schema["capability"]["required"] is True
        assert "prompt" in schema
        assert schema["prompt"]["required"] is True

    def test_spec_has_optional_params(self):
        schema = ACP_CREATE_TASK_SPEC.params_schema
        assert "target_agent_id" in schema
        assert schema["target_agent_id"]["default"] == "claude-session-agent"
        assert "priority" in schema
        assert schema["priority"]["default"] == "normal"
        assert "conversation_id" in schema

    def test_spec_requires_no_approval(self):
        assert ACP_CREATE_TASK_SPEC.requires_approval is False


class TestACPCreateTaskTool:
    """Tests for the _tool_acp_create_task handler function."""

    @pytest.mark.asyncio
    async def test_missing_capability(self):
        result = await _tool_acp_create_task({"prompt": "Do something"})
        assert not result.success
        assert "capability" in result.error
        assert "required" in result.error.lower()

    @pytest.mark.asyncio
    async def test_missing_prompt(self):
        result = await _tool_acp_create_task({"capability": "code.edit"})
        assert not result.success
        assert "prompt" in result.error
        assert "required" in result.error.lower()

    @pytest.mark.asyncio
    async def test_missing_both_params(self):
        result = await _tool_acp_create_task({})
        assert not result.success
        assert "required" in result.error.lower()

    @pytest.mark.asyncio
    async def test_normalizes_claude_code_target(self, httpx_mock):
        httpx_mock.add_response(
            url="http://127.0.0.1:7337/acp/tasks",
            method="POST",
            json={"task_id": "task-123"},
        )
        result = await _tool_acp_create_task({
            "capability": "code.edit",
            "prompt": "Fix bug",
            "target_agent_id": "claude code",
        })
        assert result.success
        # Verify that the normalized target_agent_id was sent
        request = httpx_mock.get_request()
        payload = json.loads(request.content)
        assert payload["target_agent_id"] == "claude-session-agent"

    @pytest.mark.asyncio
    async def test_normalizes_claude_target(self, httpx_mock):
        httpx_mock.add_response(
            url="http://127.0.0.1:7337/acp/tasks",
            method="POST",
            json={"task_id": "task-456"},
        )
        result = await _tool_acp_create_task({
            "capability": "reasoning.complex",
            "prompt": "Analyze data",
            "target_agent_id": "claude",
        })
        assert result.success
        request = httpx_mock.get_request()
        payload = json.loads(request.content)
        assert payload["target_agent_id"] == "claude-session-agent"

    @pytest.mark.asyncio
    async def test_normalizes_claude_hyphen_code_target(self, httpx_mock):
        httpx_mock.add_response(
            url="http://127.0.0.1:7337/acp/tasks",
            method="POST",
            json={"task_id": "task-789"},
        )
        result = await _tool_acp_create_task({
            "capability": "code.refactor",
            "prompt": "Refactor module",
            "target_agent_id": "claude-code",
        })
        assert result.success
        request = httpx_mock.get_request()
        payload = json.loads(request.content)
        assert payload["target_agent_id"] == "claude-session-agent"

    @pytest.mark.asyncio
    async def test_default_target_agent(self, httpx_mock):
        httpx_mock.add_response(
            url="http://127.0.0.1:7337/acp/tasks",
            method="POST",
            json={"task_id": "task-default"},
        )
        result = await _tool_acp_create_task({
            "capability": "code.review",
            "prompt": "Review PR",
        })
        assert result.success
        request = httpx_mock.get_request()
        # Should default to claude-session-agent
        payload = json.loads(request.content)
        assert payload["target_agent_id"] == "claude-session-agent"

    @pytest.mark.asyncio
    async def test_uses_description_as_prompt_fallback(self, httpx_mock):
        httpx_mock.add_response(
            url="http://127.0.0.1:7337/acp/tasks",
            method="POST",
            json={"task_id": "task-desc"},
        )
        result = await _tool_acp_create_task({
            "capability": "code.test",
            "description": "Write unit tests",
        })
        assert result.success
        request = httpx_mock.get_request()
        payload = json.loads(request.content)
        assert payload["payload_ref"] == "Write unit tests"

    @pytest.mark.asyncio
    async def test_success_response(self, httpx_mock):
        httpx_mock.add_response(
            url="http://127.0.0.1:7337/acp/tasks",
            method="POST",
            json={"task_id": "task-success-123"},
        )
        result = await _tool_acp_create_task({
            "capability": "reasoning.complex",
            "prompt": "Solve problem",
            "priority": "high",
        })
        assert result.success
        assert result.tool == "acp.create_task"
        assert "task-success-123" in result.output
        assert "reasoning.complex" in result.output
        assert "high" in result.output

    @pytest.mark.asyncio
    async def test_includes_conversation_id(self, httpx_mock):
        httpx_mock.add_response(
            url="http://127.0.0.1:7337/acp/tasks",
            method="POST",
            json={"task_id": "task-conv"},
        )
        result = await _tool_acp_create_task({
            "capability": "code.debug",
            "prompt": "Find memory leak",
            "conversation_id": "conv-abc-123",
        })
        assert result.success
        request = httpx_mock.get_request()
        payload = json.loads(request.content)
        assert payload["conversation_id"] == "conv-abc-123"

    @pytest.mark.asyncio
    async def test_omits_conversation_id_when_missing(self, httpx_mock):
        httpx_mock.add_response(
            url="http://127.0.0.1:7337/acp/tasks",
            method="POST",
            json={"task_id": "task-no-conv"},
        )
        result = await _tool_acp_create_task({
            "capability": "code.analyze",
            "prompt": "Check performance",
        })
        assert result.success
        request = httpx_mock.get_request()
        payload = json.loads(request.content)
        assert "conversation_id" not in payload

    @pytest.mark.asyncio
    async def test_http_503_error(self, httpx_mock):
        httpx_mock.add_response(
            url="http://127.0.0.1:7337/acp/tasks",
            method="POST",
            status_code=503,
        )
        result = await _tool_acp_create_task({
            "capability": "code.test",
            "prompt": "Write tests",
        })
        assert not result.success
        assert "bridge not connected" in result.error.lower()

    @pytest.mark.asyncio
    async def test_http_404_error(self, httpx_mock):
        httpx_mock.add_response(
            url="http://127.0.0.1:7337/acp/tasks",
            method="POST",
            status_code=404,
        )
        result = await _tool_acp_create_task({
            "capability": "code.format",
            "prompt": "Format code",
        })
        assert not result.success
        assert result.error

    @pytest.mark.asyncio
    async def test_connection_error(self, httpx_mock):
        httpx_mock.add_exception(httpx.ConnectError("Connection refused"))
        result = await _tool_acp_create_task({
            "capability": "code.lint",
            "prompt": "Lint files",
        })
        assert not result.success
        assert result.error

    @pytest.mark.asyncio
    async def test_timeout_error(self, httpx_mock):
        httpx_mock.add_exception(httpx.TimeoutException("Request timeout"))
        result = await _tool_acp_create_task({
            "capability": "code.optimize",
            "prompt": "Optimize queries",
        })
        assert not result.success
        assert result.error


class TestACPToolsRegistration:
    """Tests for the register_acp_tools function."""

    def test_registers_create_task_spec(self):
        registry = ToolRegistry()
        register_acp_tools(registry)
        spec = registry.get_spec("acp.create_task")
        assert spec is not None
        assert spec.name == "acp.create_task"

    def test_registers_create_task_handler(self):
        registry = ToolRegistry()
        register_acp_tools(registry)
        # Verify handler is callable by checking it's in the list of tools
        tools = registry.list_tools()
        tool_names = {t.name for t in tools}
        assert "acp.create_task" in tool_names

    @pytest.mark.asyncio
    async def test_execute_via_registry(self, httpx_mock):
        """Test that the registered tool can be executed through the registry."""
        registry = ToolRegistry()
        register_acp_tools(registry)

        httpx_mock.add_response(
            url="http://127.0.0.1:7337/acp/tasks",
            method="POST",
            json={"task_id": "registry-test-123"},
        )

        result = await registry.execute("acp.create_task", {
            "capability": "code.review",
            "prompt": "Review changes",
            "priority": "normal",
        })

        assert result.success
        assert "registry-test-123" in result.output
