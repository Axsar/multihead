"""Tests for multihead.client.MultiHeadClient."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from multihead.client import MultiHeadClient


@pytest.fixture
def client():
    """Create a MultiHeadClient instance for testing."""
    return MultiHeadClient(base_url="http://localhost:7337", timeout=10.0)


@pytest.fixture
def mock_httpx_client():
    """Mock httpx.Client context manager."""
    with patch("multihead.client.httpx.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__.return_value = mock_client
        yield mock_client


class TestClientInit:
    """Test client initialization."""

    def test_init_default(self):
        """Test client initialization with defaults."""
        client = MultiHeadClient()
        assert client.base_url == "http://localhost:7337"
        assert client.timeout == 10.0

    def test_init_custom_base_url(self):
        """Test client initialization with custom base URL."""
        client = MultiHeadClient(base_url="http://example.com:8080")
        assert client.base_url == "http://example.com:8080"

    def test_init_strips_trailing_slash(self):
        """Test that trailing slash is stripped from base URL."""
        client = MultiHeadClient(base_url="http://localhost:7337/")
        assert client.base_url == "http://localhost:7337"

    def test_init_custom_timeout(self):
        """Test client initialization with custom timeout."""
        client = MultiHeadClient(timeout=30.0)
        assert client.timeout == 30.0


class TestUrlConstruction:
    """Test URL construction."""

    def test_url_construction(self, client):
        """Test _url method constructs correct URLs."""
        assert client._url("/knowledge/claims") == "http://localhost:7337/knowledge/claims"
        assert client._url("/heads") == "http://localhost:7337/heads"
        assert client._url("/chat") == "http://localhost:7337/chat"


class TestDepositClaim:
    """Test deposit_claim method."""

    def test_deposit_claim_minimal(self, client, mock_httpx_client):
        """Test deposit_claim with minimal arguments."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"claim_id": "abc123", "status": "accepted"}
        mock_httpx_client.post.return_value = mock_response

        result = client.deposit_claim(
            claim_key="test.key",
            statement="Test statement",
        )

        assert result == {"claim_id": "abc123", "status": "accepted"}
        mock_httpx_client.post.assert_called_once()
        call_args = mock_httpx_client.post.call_args
        assert call_args[0][0] == "http://localhost:7337/knowledge/claims"
        assert call_args[1]["json"]["claim_key"] == "test.key"
        assert call_args[1]["json"]["statement"] == "Test statement"
        assert call_args[1]["json"]["produced_by"] == "external"
        mock_response.raise_for_status.assert_called_once()

    def test_deposit_claim_full_args(self, client, mock_httpx_client):
        """Test deposit_claim with all arguments."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"claim_id": "xyz789"}
        mock_httpx_client.post.return_value = mock_response

        result = client.deposit_claim(
            claim_key="project.auth.status",
            statement="Service completed",
            subject_type="component",
            subject_id="auth",
            predicate="has_state",
            value="completed",
            value_type="string",
            claim_type="fact",
            claim_status="accepted",
            scope_type="project",
            scope_id="default",
            confidence=0.95,
            stability="high",
            importance=0.8,
            rationale="Unit test",
            produced_by="test_suite",
        )

        assert result == {"claim_id": "xyz789"}
        call_args = mock_httpx_client.post.call_args
        body = call_args[1]["json"]
        assert body["claim_key"] == "project.auth.status"
        assert body["statement"] == "Service completed"
        assert body["subject_id"] == "auth"
        assert body["confidence"] == 0.95
        assert body["stability"] == "high"
        assert body["produced_by"] == "test_suite"

    def test_deposit_claim_subject_id_fallback(self, client, mock_httpx_client):
        """Test that subject_id defaults to first part of claim_key when empty."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"claim_id": "test"}
        mock_httpx_client.post.return_value = mock_response

        client.deposit_claim(
            claim_key="project.auth.status",
            statement="Test",
            subject_id="",
        )

        call_args = mock_httpx_client.post.call_args
        assert call_args[1]["json"]["subject_id"] == "project"


class TestQueryClaims:
    """Test query_claims method."""

    def test_query_claims_no_filters(self, client, mock_httpx_client):
        """Test query_claims with no filters."""
        mock_response = MagicMock()
        mock_response.json.return_value = [{"claim_id": "1"}, {"claim_id": "2"}]
        mock_httpx_client.get.return_value = mock_response

        result = client.query_claims()

        assert len(result) == 2
        mock_httpx_client.get.assert_called_once()
        call_args = mock_httpx_client.get.call_args
        assert call_args[0][0] == "http://localhost:7337/knowledge/claims"
        assert call_args[1]["params"] == {"limit": 100}

    def test_query_claims_with_filters(self, client, mock_httpx_client):
        """Test query_claims with all filters."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_httpx_client.get.return_value = mock_response

        result = client.query_claims(
            status="accepted",
            claim_type="fact",
            scope_id="default",
            limit=50,
        )

        assert result == []
        call_args = mock_httpx_client.get.call_args
        params = call_args[1]["params"]
        assert params["status"] == "accepted"
        assert params["claim_type"] == "fact"
        assert params["scope_id"] == "default"
        assert params["limit"] == 50


class TestGetClaim:
    """Test get_claim method."""

    def test_get_claim(self, client, mock_httpx_client):
        """Test get_claim retrieves specific claim."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "claim_id": "abc123",
            "statement": "Test claim",
        }
        mock_httpx_client.get.return_value = mock_response

        result = client.get_claim("abc123")

        assert result["claim_id"] == "abc123"
        assert result["statement"] == "Test claim"
        mock_httpx_client.get.assert_called_once_with(
            "http://localhost:7337/knowledge/claims/abc123",
            params=None,
        )


class TestGetBriefing:
    """Test get_briefing method."""

    def test_get_briefing_minimal(self, client, mock_httpx_client):
        """Test get_briefing with minimal arguments."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "direct_claims": [],
            "related_claims": [],
            "events": [],
        }
        mock_httpx_client.get.return_value = mock_response

        result = client.get_briefing("auth")

        assert "direct_claims" in result
        call_args = mock_httpx_client.get.call_args
        assert call_args[0][0] == "http://localhost:7337/knowledge/briefing"
        params = call_args[1]["params"]
        assert params["component"] == "auth"
        assert params["scope_id"] == "default"
        assert params["include_events"] is True
        assert params["max_claims"] == 20
        assert params["max_events"] == 10

    def test_get_briefing_custom_params(self, client, mock_httpx_client):
        """Test get_briefing with custom parameters."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": "test"}
        mock_httpx_client.get.return_value = mock_response

        result = client.get_briefing(
            "test_component",
            scope_id="custom",
            include_events=False,
            max_claims=5,
            max_events=3,
        )

        call_args = mock_httpx_client.get.call_args
        params = call_args[1]["params"]
        assert params["component"] == "test_component"
        assert params["scope_id"] == "custom"
        assert params["include_events"] is False
        assert params["max_claims"] == 5
        assert params["max_events"] == 3


class TestReportEvent:
    """Test report_event method."""

    def test_report_event_minimal(self, client, mock_httpx_client):
        """Test report_event with minimal arguments."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"event_id": "evt123"}
        mock_httpx_client.post.return_value = mock_response

        result = client.report_event("Test event")

        assert result["event_id"] == "evt123"
        call_args = mock_httpx_client.post.call_args
        body = call_args[1]["json"]
        assert body["title"] == "Test event"
        assert body["summary"] == ""
        assert body["event_type"] == "note"
        assert body["produced_by"] == "external"
        assert body["tags"] == []
        assert body["metrics"] == {}
        assert body["entities"] == []

    def test_report_event_full_args(self, client, mock_httpx_client):
        """Test report_event with all arguments."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"event_id": "evt456"}
        mock_httpx_client.post.return_value = mock_response

        result = client.report_event(
            "Layout completed",
            summary="Processed 12 panels",
            event_type="task_completed",
            event_status="confirmed",
            tags=["layout", "success"],
            metrics={"panels": 12.0, "duration": 5.2},
            produced_by="layout_pipeline",
            entities=[{"type": "component", "id": "auth"}],
        )

        assert result["event_id"] == "evt456"
        call_args = mock_httpx_client.post.call_args
        body = call_args[1]["json"]
        assert body["title"] == "Layout completed"
        assert body["summary"] == "Processed 12 panels"
        assert body["event_type"] == "task_completed"
        assert body["tags"] == ["layout", "success"]
        assert body["metrics"]["panels"] == 12.0
        assert body["produced_by"] == "layout_pipeline"


class TestQueryEvents:
    """Test query_events method."""

    def test_query_events_no_filters(self, client, mock_httpx_client):
        """Test query_events with no filters."""
        mock_response = MagicMock()
        mock_response.json.return_value = [{"event_id": "1"}, {"event_id": "2"}]
        mock_httpx_client.get.return_value = mock_response

        result = client.query_events()

        assert len(result) == 2
        call_args = mock_httpx_client.get.call_args
        assert call_args[1]["params"] == {"limit": 100}

    def test_query_events_with_filters(self, client, mock_httpx_client):
        """Test query_events with filters."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_httpx_client.get.return_value = mock_response

        result = client.query_events(
            status="confirmed",
            event_type="task_completed",
            limit=25,
        )

        assert result == []
        call_args = mock_httpx_client.get.call_args
        params = call_args[1]["params"]
        assert params["status"] == "confirmed"
        assert params["event_type"] == "task_completed"
        assert params["limit"] == 25


class TestChat:
    """Test chat method."""

    def test_chat_minimal(self, client, mock_httpx_client):
        """Test chat with minimal arguments."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Hello"}
        mock_httpx_client.post.return_value = mock_response

        result = client.chat("Hello world")

        assert result["response"] == "Hello"
        call_args = mock_httpx_client.post.call_args
        assert call_args[0][0] == "http://localhost:7337/chat"
        assert call_args[1]["json"]["message"] == "Hello world"
        assert "head_id" not in call_args[1]["json"]

    def test_chat_with_head_id(self, client, mock_httpx_client):
        """Test chat with specific head_id."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Hi"}
        mock_httpx_client.post.return_value = mock_response

        result = client.chat("Test message", head_id="qwen-llm")

        assert result["response"] == "Hi"
        call_args = mock_httpx_client.post.call_args
        assert call_args[1]["json"]["message"] == "Test message"
        assert call_args[1]["json"]["head_id"] == "qwen-llm"


class TestGenerate:
    """Test generate method."""

    def test_generate_minimal(self, client, mock_httpx_client):
        """Test generate with minimal arguments."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"text": "Generated text"}
        mock_httpx_client.post.return_value = mock_response

        result = client.generate("Test prompt")

        assert result["text"] == "Generated text"
        call_args = mock_httpx_client.post.call_args
        assert call_args[0][0] == "http://localhost:7337/generate"
        assert call_args[1]["json"]["prompt"] == "Test prompt"
        assert "head_id" not in call_args[1]["json"]

    def test_generate_with_head_id(self, client, mock_httpx_client):
        """Test generate with specific head_id."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"text": "Output"}
        mock_httpx_client.post.return_value = mock_response

        result = client.generate("Prompt", head_id="mock-llm")

        assert result["text"] == "Output"
        call_args = mock_httpx_client.post.call_args
        assert call_args[1]["json"]["head_id"] == "mock-llm"


class TestDecompose:
    """Test decompose method."""

    def test_decompose_minimal(self, client, mock_httpx_client):
        """Test decompose with minimal arguments."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"steps": []}
        mock_httpx_client.post.return_value = mock_response

        result = client.decompose("Achieve goal")

        assert result["steps"] == []
        call_args = mock_httpx_client.post.call_args
        assert call_args[0][0] == "http://localhost:7337/decompose"
        body = call_args[1]["json"]
        assert body["goal"] == "Achieve goal"
        assert body["context"] == ""
        assert body["max_depth"] == 4
        assert "head_id" not in body

    def test_decompose_full_args(self, client, mock_httpx_client):
        """Test decompose with all arguments."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"steps": ["step1", "step2"]}
        mock_httpx_client.post.return_value = mock_response

        result = client.decompose(
            "Complex goal",
            context="Additional context",
            head_id="qwen-llm",
            max_depth=6,
        )

        assert result["steps"] == ["step1", "step2"]
        call_args = mock_httpx_client.post.call_args
        body = call_args[1]["json"]
        assert body["goal"] == "Complex goal"
        assert body["context"] == "Additional context"
        assert body["max_depth"] == 6
        assert body["head_id"] == "qwen-llm"


class TestPing:
    """Test ping method."""

    def test_ping_success(self, client, mock_httpx_client):
        """Test ping returns True when server is reachable."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"heads": []}
        mock_httpx_client.get.return_value = mock_response

        result = client.ping()

        assert result is True
        mock_httpx_client.get.assert_called_once_with(
            "http://localhost:7337/heads",
            params=None,
        )

    def test_ping_failure(self, client, mock_httpx_client):
        """Test ping returns False when server is unreachable."""
        mock_httpx_client.get.side_effect = Exception("Connection error")

        result = client.ping()

        assert result is False

    def test_ping_http_error(self, client, mock_httpx_client):
        """Test ping returns False on HTTP error."""
        import httpx

        mock_httpx_client.get.side_effect = httpx.HTTPError("Server error")

        result = client.ping()

        assert result is False
