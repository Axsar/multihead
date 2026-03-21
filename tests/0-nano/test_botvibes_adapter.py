"""Tests for BotVibesAdapter."""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from multihead.adapters.botvibes_adapter import BotVibesAdapter
from multihead.models import AdapterKind, Capability, HeadManifest


@pytest.fixture
def bot_vibes_manifest():
    """Create a BotVibes provider manifest for testing."""
    return HeadManifest(
        head_id="botvibes-test-provider",
        name="Test BotVibes Provider",
        adapter=AdapterKind.BOTVIBES,
        model="test-model-v1",
        kind="llm",
        endpoint="http://localhost:8000/api/v1",
        extra={
            "api_key": "test-token-12345",
            "project_id": "test-project-123",
            "target_capability": "test_capability",
            "target_agent_id": "test-agent-456",
        },
        capabilities=Capability(
            solver_type="llm",
            input_modalities=["text"],
            output_modalities=["text"],
            task_types=["test_task"],
            latency_p50_ms=1000,
            cost_per_call=0.01,
        ),
    )


class TestBotVibesAdapter:
    """Test BotVibesAdapter functionality."""

    def test_init_with_valid_manifest(self, bot_vibes_manifest):
        """Test adapter initialization with valid manifest."""
        adapter = BotVibesAdapter(bot_vibes_manifest)

        assert adapter.manifest == bot_vibes_manifest
        assert adapter.acp_url == "http://localhost:8000/api/v1"
        assert adapter.acp_token == "test-token-12345"
        assert adapter.project_id == "test-project-123"
        assert adapter.target_capability == "test_capability"
        assert adapter.target_agent_id == "test-agent-456"

    def test_init_without_acp_url(self):
        """Test adapter initialization fails without ACP URL."""
        manifest = HeadManifest(
            head_id="test",
            name="Test",
            adapter=AdapterKind.BOTVIBES,
            model="test",
            kind="llm",
            # Missing endpoint
            extra={"api_key": "token", "project_id": "proj", "target_capability": "cap"},
        )

        with pytest.raises(ValueError, match="missing acp_url"):
            BotVibesAdapter(manifest)

    def test_init_without_token(self):
        """Test adapter initialization fails without token."""
        manifest = HeadManifest(
            head_id="test",
            name="Test",
            adapter=AdapterKind.BOTVIBES,
            model="test",
            kind="llm",
            endpoint="http://localhost:8000",
            # Missing api_key in extra
            extra={"project_id": "proj", "target_capability": "cap"},
        )

        with pytest.raises(ValueError, match="missing acp_token"):
            BotVibesAdapter(manifest)

    def test_init_without_capability(self):
        """Test adapter initialization fails without target capability."""
        manifest = HeadManifest(
            head_id="test",
            name="Test",
            adapter=AdapterKind.BOTVIBES,
            model="test",
            kind="llm",
            endpoint="http://localhost:8000",
            extra={"api_key": "token", "project_id": "proj"},  # Missing target_capability
        )

        with pytest.raises(ValueError, match="missing target_capability"):
            BotVibesAdapter(manifest)

    async def test_load_unload_are_noops(self, bot_vibes_manifest):
        """Test that load/unload are no-ops for remote providers."""
        adapter = BotVibesAdapter(bot_vibes_manifest)

        # Should not raise
        await adapter.load()
        await adapter.unload()
        await adapter.sleep()
        await adapter.wake()

    @patch("httpx.AsyncClient")
    async def test_healthcheck_success(self, mock_client_class, bot_vibes_manifest):
        """Test healthcheck when ACP server is reachable."""
        # Mock the async context manager and response
        mock_response = Mock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_client_class.return_value = mock_client

        adapter = BotVibesAdapter(bot_vibes_manifest)
        result = await adapter.healthcheck()

        assert result is True
        mock_client.get.assert_called_once()

    @patch("httpx.AsyncClient")
    async def test_healthcheck_failure(self, mock_client_class, bot_vibes_manifest):
        """Test healthcheck when ACP server is unreachable."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("Connection failed"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_client_class.return_value = mock_client

        adapter = BotVibesAdapter(bot_vibes_manifest)
        result = await adapter.healthcheck()

        assert result is False

    @patch("httpx.AsyncClient")
    async def test_generate_creates_task_and_waits(self, mock_client_class, bot_vibes_manifest):
        """Test generate creates ACP task and waits for result."""
        create_response = Mock()
        create_response.status_code = 200
        create_response.json = Mock(return_value={"task_id": "task-123"})
        create_response.raise_for_status = Mock()

        poll_response = Mock()
        poll_response.status_code = 200
        poll_response.json = Mock(return_value={
            "task_id": "task-123",
            "status": "complete",
            "output_ref": "Test result from BotVibes",
            "confidence": 0.95,
            "latency_ms": 1200,
        })

        mock_client = AsyncMock()
        # New adapter uses client.request() via _request_with_retry
        mock_client.request = AsyncMock(side_effect=[create_response, poll_response])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_client_class.return_value = mock_client

        adapter = BotVibesAdapter(bot_vibes_manifest)
        result = await adapter.generate("Test prompt")

        # Verify task creation via client.request
        assert mock_client.request.call_count >= 1
        create_call = mock_client.request.call_args_list[0]
        assert create_call[0][0] == "POST"  # method
        assert "tasks" in create_call[0][1]  # url

        # Verify result
        assert result["text"] == "Test result from BotVibes"
        assert result["tokens_in"] > 0
        assert result["tokens_out"] > 0
        assert result["latency_ms"] >= 0
        assert result["provider_metadata"]["task_id"] == "task-123"
        assert result["provider_metadata"]["confidence"] == 0.95

    @patch("httpx.AsyncClient")
    async def test_generate_handles_task_failure(self, mock_client_class, bot_vibes_manifest):
        """Test generate handles task failures gracefully."""
        create_response = Mock()
        create_response.status_code = 200
        create_response.json = Mock(return_value={"task_id": "task-456"})
        create_response.raise_for_status = Mock()

        poll_response = Mock()
        poll_response.status_code = 200
        poll_response.json = Mock(return_value={
            "task_id": "task-456",
            "status": "failed",
            "error_code": "EXECUTION_ERROR",
            "message": "Provider error: out of memory",
        })

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=[create_response, poll_response])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_client_class.return_value = mock_client

        adapter = BotVibesAdapter(bot_vibes_manifest)

        with pytest.raises(RuntimeError, match="BotVibes task failed"):
            await adapter.generate("Test prompt")

    @patch("httpx.AsyncClient")
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_generate_polls_until_complete(
        self, mock_sleep, mock_client_class, bot_vibes_manifest
    ):
        """Test generate polls multiple times until task completes."""
        create_response = Mock()
        create_response.status_code = 200
        create_response.json = Mock(return_value={"task_id": "task-789"})
        create_response.raise_for_status = Mock()

        poll_responses = [
            Mock(status_code=200, json=Mock(return_value={
                "task_id": "task-789", "status": "created"
            })),
            Mock(status_code=200, json=Mock(return_value={
                "task_id": "task-789", "status": "dispatched"
            })),
            Mock(status_code=200, json=Mock(return_value={
                "task_id": "task-789",
                "status": "complete",
                "output_ref": "Final result",
                "confidence": 0.9,
                "latency_ms": 3000,
            })),
        ]

        mock_client = AsyncMock()
        # First call is POST (create task), then 3 GET calls (poll)
        mock_client.request = AsyncMock(
            side_effect=[create_response] + poll_responses
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_client_class.return_value = mock_client

        adapter = BotVibesAdapter(bot_vibes_manifest)
        result = await adapter.generate("Test prompt")

        # 1 create + 3 polls = 4 total
        assert mock_client.request.call_count == 4
        assert result["text"] == "Final result"

    def test_auth_headers(self, bot_vibes_manifest):
        """Test authentication headers are correct."""
        adapter = BotVibesAdapter(bot_vibes_manifest)
        headers = adapter._auth_headers()

        assert headers["Authorization"] == "Bearer test-token-12345"
        assert headers["Content-Type"] == "application/json"
