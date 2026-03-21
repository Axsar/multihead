"""Tests for MeshClient and MeshHeadAdapter."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

import httpx

from multihead.mesh.client import MeshClient
from multihead.adapters.mesh_adapter import MeshHeadAdapter
from multihead.models import AdapterKind, HeadManifest


def _mock_response(status_code: int, json_data=None):
    """Create a mock httpx.Response with a proper request object."""
    resp = httpx.Response(status_code, json=json_data or {})
    resp._request = httpx.Request("GET", "http://mock/")
    return resp


# -----------------------------------------------------------------------
# MeshClient tests
# -----------------------------------------------------------------------


class TestMeshClient:
    """Test the HTTP client for peer communication."""

    @pytest.fixture
    def client(self):
        return MeshClient(
            base_url="http://peer1.local:7337",
            auth_token="test-token-abc",
            timeout=5.0,
            max_retries=1,
        )

    async def test_health(self, client):
        """health() should GET /v1/health."""
        resp = _mock_response(200, {"status": "ok", "mesh_version": "1.0.0"})
        with patch("httpx.AsyncClient.request", new_callable=AsyncMock, return_value=resp):
            result = await client.health()
        assert result["status"] == "ok"

    async def test_capabilities(self, client):
        """capabilities() should GET /v1/capabilities."""
        caps = [{"name": "llm-8b", "kind": "llm", "status": "available"}]
        resp = _mock_response(200, caps)
        with patch("httpx.AsyncClient.request", new_callable=AsyncMock, return_value=resp):
            result = await client.capabilities()
        assert len(result) == 1
        assert result[0]["kind"] == "llm"

    async def test_capabilities_with_kind_filter(self, client):
        """capabilities(kind='vlm') should pass kind param."""
        resp = _mock_response(200, [])
        with patch(
            "httpx.AsyncClient.request",
            new_callable=AsyncMock, return_value=resp,
        ) as mock_req:
            await client.capabilities(kind="vlm")
        call_kwargs = mock_req.call_args
        assert call_kwargs.kwargs["params"] == {"kind": "vlm"}

    async def test_submit_task(self, client):
        """submit_task() should POST /v1/tasks."""
        resp_data = {
            "task_id": "t-1", "status": "completed",
            "result": "Hello world", "head_id": "mock-llm",
        }
        resp = _mock_response(200, resp_data)
        with patch("httpx.AsyncClient.request", new_callable=AsyncMock, return_value=resp):
            result = await client.submit_task("llm", "Say hello")
        assert result["status"] == "completed"
        assert result["result"] == "Hello world"

    async def test_node_info(self, client):
        """node_info() should GET /v1/node."""
        resp_data = {"node_id": "desktop-01", "version": "1.0.0", "capabilities": 3, "available": 2}
        resp = _mock_response(200, resp_data)
        with patch("httpx.AsyncClient.request", new_callable=AsyncMock, return_value=resp):
            result = await client.node_info()
        assert result["node_id"] == "desktop-01"
        assert result["capabilities"] == 3

    async def test_auth_header_injected(self, client):
        """Authorization header should be set when token provided."""
        assert client._headers["Authorization"] == "Bearer test-token-abc"

    async def test_no_auth_header_without_token(self):
        """No auth header when no token provided."""
        client = MeshClient(base_url="http://peer:7337")
        assert "Authorization" not in client._headers

    async def test_circuit_breaker_trips_on_failures(self):
        """Circuit breaker should open after repeated failures."""
        client = MeshClient(base_url="http://peer1.local:7337", max_retries=0)
        resp = _mock_response(500, {"detail": "error"})
        with patch("httpx.AsyncClient.request", new_callable=AsyncMock, return_value=resp):
            for _ in range(5):
                try:
                    await client.health()
                except httpx.HTTPStatusError:
                    pass

        from multihead.resilience import CircuitBreakerOpen
        with pytest.raises(CircuitBreakerOpen):
            await client.health()

    async def test_retry_on_server_error(self, client):
        """Should retry on 500 errors up to max_retries."""
        resp = _mock_response(500, {"detail": "error"})
        with patch(
            "httpx.AsyncClient.request",
            new_callable=AsyncMock, return_value=resp,
        ) as mock_req:
            with pytest.raises(httpx.HTTPStatusError):
                await client.health()
        # max_retries=1 means 2 total attempts
        assert mock_req.call_count == 2

    async def test_client_error_not_retried(self, client):
        """4xx errors should not be retried."""
        resp = _mock_response(404, {"detail": "not found"})
        with patch(
            "httpx.AsyncClient.request",
            new_callable=AsyncMock, return_value=resp,
        ) as mock_req:
            with pytest.raises(httpx.HTTPStatusError):
                await client.node_info()
        assert mock_req.call_count == 1


# -----------------------------------------------------------------------
# MeshHeadAdapter tests
# -----------------------------------------------------------------------


class TestMeshHeadAdapter:
    """Test the adapter that routes generate() to remote peers."""

    @pytest.fixture
    def manifest(self):
        return HeadManifest(
            head_id="mesh-peer1-llm",
            name="Peer1 LLM",
            adapter=AdapterKind.MESH,
            model="qwen3:8b",
            kind="llm",
            gpu_required=False,
            is_local=False,
            extra={
                "peer_url": "http://192.168.1.10:7337",
                "peer_node_id": "desktop-01",
                "auth_token": "secret-123",
            },
        )

    @pytest.fixture
    def adapter(self, manifest):
        return MeshHeadAdapter(manifest)

    async def test_generate_returns_text(self, adapter):
        """generate() should submit task and return text result."""
        mock_result = {
            "task_id": "", "status": "completed",
            "result": "The answer is 42", "head_id": "core-llm",
        }
        with patch.object(
            adapter._client, "submit_task",
            new_callable=AsyncMock, return_value=mock_result,
        ):
            result = await adapter.generate("What is the meaning of life?")
        assert result["text"] == "The answer is 42"
        assert result["peer_node_id"] == "desktop-01"
        assert "latency_ms" in result

    async def test_generate_handles_dict_result(self, adapter):
        """generate() should handle nested dict result."""
        mock_result = {
            "status": "completed",
            "result": {"text": "nested answer", "tokens": 5},
            "head_id": "core-llm",
        }
        with patch.object(
            adapter._client, "submit_task",
            new_callable=AsyncMock, return_value=mock_result,
        ):
            result = await adapter.generate("test")
        assert result["text"] == "nested answer"

    async def test_load_is_noop(self, adapter):
        """load() should be a no-op for remote peers."""
        await adapter.load()  # Should not raise

    async def test_unload_is_noop(self, adapter):
        """unload() should be a no-op for remote peers."""
        await adapter.unload()  # Should not raise

    async def test_healthcheck_success(self, adapter):
        """healthcheck() should return True when peer is healthy."""
        with patch.object(
            adapter._client, "health",
            new_callable=AsyncMock, return_value={"status": "ok"},
        ):
            assert await adapter.healthcheck() is True

    async def test_healthcheck_failure(self, adapter):
        """healthcheck() should return False when peer is unreachable."""
        with patch.object(
            adapter._client, "health",
            new_callable=AsyncMock,
            side_effect=Exception("connection refused"),
        ):
            assert await adapter.healthcheck() is False

    def test_requires_peer_url(self):
        """Should raise if peer_url not in extra."""
        manifest = HeadManifest(
            head_id="bad", name="Bad", adapter=AdapterKind.MESH,
            model="test", extra={},
        )
        with pytest.raises(ValueError, match="peer_url"):
            MeshHeadAdapter(manifest)


# -----------------------------------------------------------------------
# Head Manager integration
# -----------------------------------------------------------------------


class TestHeadManagerMeshAdapter:
    """Test that HeadManager can create MeshHeadAdapter."""

    def test_create_mesh_adapter(self):
        """_create_adapter should handle AdapterKind.MESH."""
        from multihead.head_manager import _create_adapter

        manifest = HeadManifest(
            head_id="mesh-test",
            name="Mesh Test",
            adapter=AdapterKind.MESH,
            model="qwen3:8b",
            kind="llm",
            gpu_required=False,
            extra={"peer_url": "http://test:7337", "peer_node_id": "test-node"},
        )
        adapter = _create_adapter(manifest)
        assert isinstance(adapter, MeshHeadAdapter)
        assert adapter._peer_node_id == "test-node"
