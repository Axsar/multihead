"""Tests for web tools (fetch + search)."""

from unittest.mock import MagicMock, patch

import pytest

from multihead.tool_registry import ToolRegistry
from multihead.web_tools import (
    _extract_text,
    _tool_web_fetch,
    _tool_web_search,
    register_web_tools,
)


class TestHTMLExtraction:
    def test_plain_text(self):
        assert _extract_text("Hello world") == "Hello world"

    def test_strip_tags(self):
        assert _extract_text("<p>Hello</p>") == "Hello"

    def test_strip_script(self):
        result = _extract_text("<script>var x = 1;</script>Hello")
        assert "var x" not in result
        assert "Hello" in result

    def test_strip_style(self):
        result = _extract_text("<style>.foo{color:red}</style>Content")
        assert "color" not in result
        assert "Content" in result

    def test_newlines_on_block_elements(self):
        result = _extract_text("<h1>Title</h1><p>Para</p>")
        assert "Title" in result
        assert "Para" in result


class TestWebFetch:
    @pytest.mark.asyncio
    async def test_fetch_missing_url(self):
        result = await _tool_web_fetch({})
        assert not result.success
        assert "required" in result.error.lower()

    @pytest.mark.asyncio
    async def test_fetch_success(self, httpx_mock):
        httpx_mock.add_response(
            url="https://example.com",
            text="<html><body><p>Hello World</p></body></html>",
            headers={"content-type": "text/html"},
        )
        result = await _tool_web_fetch({"url": "https://example.com"})
        assert result.success
        assert "Hello World" in result.output

    @pytest.mark.asyncio
    async def test_fetch_plain_text(self, httpx_mock):
        httpx_mock.add_response(
            url="https://example.com/api",
            text='{"key": "value"}',
            headers={"content-type": "application/json"},
        )
        result = await _tool_web_fetch({"url": "https://example.com/api"})
        assert result.success
        assert '"key"' in result.output

    @pytest.mark.asyncio
    async def test_fetch_truncation(self, httpx_mock):
        httpx_mock.add_response(
            url="https://example.com",
            text="x" * 1000,
            headers={"content-type": "text/plain"},
        )
        result = await _tool_web_fetch({"url": "https://example.com", "max_length": 100})
        assert result.success
        assert len(result.output) < 200
        assert "Truncated" in result.output

    @pytest.mark.asyncio
    async def test_fetch_http_error(self, httpx_mock):
        httpx_mock.add_response(url="https://example.com", status_code=404)
        result = await _tool_web_fetch({"url": "https://example.com"})
        assert not result.success
        assert "404" in result.error


class TestWebSearch:
    @pytest.mark.asyncio
    async def test_search_missing_query(self):
        result = await _tool_web_search({})
        assert not result.success
        assert "required" in result.error.lower()

    @pytest.mark.asyncio
    async def test_search_success(self):
        mock_results = [
            {"title": "Weather Today", "href": "https://weather.com", "body": "Sunny, 72F"},
            {"title": "Forecast", "href": "https://forecast.com", "body": "Clear skies"},
        ]
        with patch("duckduckgo_search.DDGS") as MockDDGS:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.text.return_value = mock_results
            MockDDGS.return_value = mock_instance

            result = await _tool_web_search({"query": "weather today"})
            assert result.success
            assert len(result.output) == 2
            assert result.output[0]["title"] == "Weather Today"
            assert result.output[0]["url"] == "https://weather.com"


class TestRegistration:
    def test_register_web_tools(self):
        registry = ToolRegistry()
        initial_count = len(registry.list_tools())
        register_web_tools(registry)
        assert len(registry.list_tools()) == initial_count + 2
        assert registry.get_spec("web.fetch") is not None
        assert registry.get_spec("web.search") is not None
