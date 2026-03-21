"""Tests for the tool registry."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from multihead.tool_registry import ToolRegistry, ToolResult, ToolSpec


@pytest.fixture
def registry():
    r = ToolRegistry()
    if sys.platform == "win32":
        # On Windows, bypass _normalize_path WSL conversion so tests
        # write/read native Windows paths used by pytest tmp_path.
        original = ToolRegistry._normalize_path

        @staticmethod
        def _identity_path(raw: str) -> Path:
            return Path(raw.strip().strip('"').strip("'"))

        r._identity = _identity_path  # keep a reference

        # Monkey-patch the class method for this test run
        _patchers = []
        _patchers.append(patch.object(ToolRegistry, "_normalize_path", _identity_path))
        for p in _patchers:
            p.start()

        yield r

        for p in _patchers:
            p.stop()
    else:
        yield r


class TestRegistration:
    def test_builtins_registered(self, registry):
        tools = registry.list_tools()
        names = {t.name for t in tools}
        assert "files.read" in names
        assert "files.write" in names
        assert "shell.run" in names
        assert "python.run" in names
        assert "llm.call" in names
        assert "verify.jsonschema" in names

    def test_custom_tool(self, registry):
        async def my_handler(params):
            return ToolResult(tool="custom", success=True, output="custom result")

        spec = ToolSpec(name="custom", description="Custom tool")
        registry.register(spec, my_handler)
        assert registry.get_spec("custom") is not None

    def test_get_spec_missing(self, registry):
        assert registry.get_spec("nonexistent") is None


class TestExecution:
    @pytest.mark.asyncio
    async def test_files_read(self, registry, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")
        result = await registry.execute("files.read", {"path": str(f)})
        assert result.success
        assert result.output == "hello world"

    @pytest.mark.asyncio
    async def test_files_read_missing(self, registry):
        result = await registry.execute("files.read", {"path": "/nonexistent/file.txt"})
        assert not result.success
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_files_write(self, registry, tmp_path):
        f = tmp_path / "output.txt"
        result = await registry.execute("files.write", {"path": str(f), "content": "hello"})
        assert result.success
        assert f.read_text() == "hello"

    @pytest.mark.asyncio
    async def test_shell_run(self, registry):
        result = await registry.execute("shell.run", {"command": "echo hello"})
        assert result.success
        assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_python_run(self, registry):
        result = await registry.execute("python.run", {"code": "result = 2 + 2"})
        assert result.success
        assert "4" in str(result.output)

    @pytest.mark.asyncio
    async def test_python_run_error(self, registry):
        result = await registry.execute("python.run", {"code": "raise ValueError('boom')"})
        assert not result.success
        assert "boom" in result.error

    @pytest.mark.asyncio
    async def test_verify_jsonschema_valid(self, registry):
        result = await registry.execute("verify.jsonschema", {
            "data": {"key": "value"},
            "schema": {"type": "object"},
        })
        assert result.success

    @pytest.mark.asyncio
    async def test_verify_jsonschema_invalid(self, registry):
        result = await registry.execute("verify.jsonschema", {
            "data": "not an array",
            "schema": {"type": "array"},
        })
        assert not result.success

    @pytest.mark.asyncio
    async def test_unknown_tool(self, registry):
        result = await registry.execute("nonexistent", {})
        assert not result.success
        assert "Unknown tool" in result.error

    @pytest.mark.asyncio
    async def test_custom_tool_execution(self, registry):
        async def echo(params):
            return ToolResult(tool="echo", success=True, output=params.get("text", ""))

        registry.register(ToolSpec(name="echo", description="Echo"), echo)
        result = await registry.execute("echo", {"text": "test"})
        assert result.success
        assert result.output == "test"
