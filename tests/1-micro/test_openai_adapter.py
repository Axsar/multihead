"""Tests for the OpenAI-compatible API adapter."""

from __future__ import annotations

import httpx
import pytest

from multihead.adapters.openai_adapter import OpenAIAdapter
from multihead.models import AdapterKind, HeadManifest


def _make_manifest(**overrides) -> HeadManifest:
    defaults = {
        "head_id": "test-openai",
        "name": "Test OpenAI",
        "adapter": AdapterKind.OPENAI,
        "model": "gpt-4o-mini",
        "kind": "llm",
        "endpoint": "https://api.openai.com/v1",
        "gpu_required": False,
        "extra": {"api_key": "sk-test-key-12345"},
    }
    defaults.update(overrides)
    return HeadManifest(**defaults)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestOpenAILifecycle:
    @pytest.mark.asyncio
    async def test_load_with_api_key(self):
        adapter = OpenAIAdapter(_make_manifest())
        await adapter.load()
        assert adapter._loaded is True

    @pytest.mark.asyncio
    async def test_load_without_api_key_raises(self):
        manifest = _make_manifest(extra={})
        adapter = OpenAIAdapter(manifest)
        adapter._api_key = None
        with pytest.raises(RuntimeError, match="API key not configured"):
            await adapter.load()

    @pytest.mark.asyncio
    async def test_load_resolves_env_var(self, monkeypatch):
        monkeypatch.setenv("MY_OPENAI_KEY", "sk-from-env")
        manifest = _make_manifest(extra={"api_key": "${MY_OPENAI_KEY}"})
        adapter = OpenAIAdapter(manifest)
        await adapter.load()
        assert adapter._api_key == "sk-from-env"

    @pytest.mark.asyncio
    async def test_load_env_var_missing_raises(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_KEY", raising=False)
        manifest = _make_manifest(extra={"api_key": "${NONEXISTENT_KEY}"})
        adapter = OpenAIAdapter(manifest)
        with pytest.raises(RuntimeError, match="NONEXISTENT_KEY not set"):
            await adapter.load()

    @pytest.mark.asyncio
    async def test_unload(self):
        adapter = OpenAIAdapter(_make_manifest())
        await adapter.load()
        assert adapter._loaded is True
        await adapter.unload()
        assert adapter._loaded is False

    @pytest.mark.asyncio
    async def test_generate_before_load_raises(self):
        adapter = OpenAIAdapter(_make_manifest())
        with pytest.raises(RuntimeError, match="not loaded"):
            await adapter.generate("hello")


# ---------------------------------------------------------------------------
# Generate / Chat
# ---------------------------------------------------------------------------


def _mock_completion_response(text="Hello!", prompt_tokens=10, completion_tokens=5):
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
            "model": "gpt-4o-mini",
        },
    )


class TestOpenAIGenerate:
    @pytest.mark.asyncio
    async def test_generate_returns_text(self, httpx_mock):
        httpx_mock.add_response(
            url="https://api.openai.com/v1/chat/completions",
            json={
                "choices": [{"message": {"role": "assistant",
                    "content": "Hello world"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
                "model": "gpt-4o-mini",
            },
        )
        adapter = OpenAIAdapter(_make_manifest())
        await adapter.load()
        result = await adapter.generate("hi")
        assert result["text"] == "Hello world"
        assert result["tokens_in"] == 5
        assert result["tokens_out"] == 3

    @pytest.mark.asyncio
    async def test_chat_with_messages(self, httpx_mock):
        httpx_mock.add_response(
            url="https://api.openai.com/v1/chat/completions",
            json={
                "choices": [{"message": {"role": "assistant",
                    "content": "I am helpful"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 4},
                "model": "gpt-4o-mini",
            },
        )
        adapter = OpenAIAdapter(_make_manifest())
        await adapter.load()
        result = await adapter.chat([
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "What are you?"},
        ])
        assert result["text"] == "I am helpful"

    @pytest.mark.asyncio
    async def test_custom_endpoint(self, httpx_mock):
        httpx_mock.add_response(
            url="http://localhost:1234/v1/chat/completions",
            json={
                "choices": [{"message": {"content": "local response"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
                "model": "local-model",
            },
        )
        manifest = _make_manifest(endpoint="http://localhost:1234/v1")
        adapter = OpenAIAdapter(manifest)
        await adapter.load()
        result = await adapter.generate("test")
        assert result["text"] == "local response"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestOpenAIErrors:
    @pytest.mark.asyncio
    async def test_401_unauthorized(self, httpx_mock):
        httpx_mock.add_response(
            url="https://api.openai.com/v1/chat/completions",
            status_code=401,
        )
        adapter = OpenAIAdapter(_make_manifest())
        await adapter.load()
        with pytest.raises(RuntimeError, match="invalid or expired"):
            await adapter.generate("test")

    @pytest.mark.asyncio
    async def test_429_rate_limit(self, httpx_mock):
        httpx_mock.add_response(
            url="https://api.openai.com/v1/chat/completions",
            status_code=429,
        )
        adapter = OpenAIAdapter(_make_manifest())
        await adapter.load()
        with pytest.raises(RuntimeError, match="rate limit"):
            await adapter.generate("test")


# ---------------------------------------------------------------------------
# Healthcheck
# ---------------------------------------------------------------------------


class TestOpenAIHealthcheck:
    @pytest.mark.asyncio
    async def test_healthcheck_success(self, httpx_mock):
        httpx_mock.add_response(
            url="https://api.openai.com/v1/models",
            json={"data": [{"id": "gpt-4o-mini"}]},
        )
        adapter = OpenAIAdapter(_make_manifest())
        assert await adapter.healthcheck() is True

    @pytest.mark.asyncio
    async def test_healthcheck_no_key(self):
        manifest = _make_manifest(extra={})
        adapter = OpenAIAdapter(manifest)
        adapter._api_key = None
        assert await adapter.healthcheck() is False


# ---------------------------------------------------------------------------
# Adapter wiring
# ---------------------------------------------------------------------------


class TestOpenAIWiring:
    def test_adapter_kind_enum_has_openai(self):
        assert AdapterKind.OPENAI.value == "openai"

    def test_head_manager_creates_openai_adapter(self):
        from multihead.head_manager import _create_adapter
        manifest = _make_manifest()
        adapter = _create_adapter(manifest)
        assert isinstance(adapter, OpenAIAdapter)
