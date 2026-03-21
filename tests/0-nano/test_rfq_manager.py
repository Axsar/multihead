"""Tests for RFQManager."""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from multihead.rfq_manager import RFQManager, Quote


@pytest.fixture
def rfq_manager():
    """Create RFQManager instance for testing."""
    return RFQManager(
        acp_url="http://localhost:8000/api/v1",
        acp_token="test-token-123",
        project_id="test-project",
    )


class TestQuote:
    """Test Quote scoring logic."""

    def test_quote_score_balances_factors(self):
        """Test quote scoring considers price, latency, and quality."""
        quote = Quote(
            quote_id="q1",
            provider_id="p1",
            price=0.10,  # Good price
            estimated_latency_ms=3000,  # Good latency
            quality_score=0.95,  # Excellent quality
        )

        score = quote.score()
        assert 0.8 < score <= 1.0  # Should be high score

    def test_quote_score_penalizes_high_price(self):
        """Test expensive quotes score lower."""
        expensive = Quote("q1", "p1", price=0.90, estimated_latency_ms=2000, quality_score=0.95)
        cheap = Quote("q2", "p2", price=0.10, estimated_latency_ms=2000, quality_score=0.95)

        assert cheap.score() > expensive.score()

    def test_quote_score_penalizes_high_latency(self):
        """Test slow quotes score lower."""
        slow = Quote("q1", "p1", price=0.10, estimated_latency_ms=10000, quality_score=0.95)
        fast = Quote("q2", "p2", price=0.10, estimated_latency_ms=2000, quality_score=0.95)

        assert fast.score() > slow.score()

    def test_quote_score_rewards_high_quality(self):
        """Test high quality quotes score higher."""
        low_quality = Quote("q1", "p1", price=0.10, estimated_latency_ms=2000, quality_score=0.70)
        high_quality = Quote("q2", "p2", price=0.10, estimated_latency_ms=2000, quality_score=0.95)

        assert high_quality.score() > low_quality.score()


class TestRFQManager:
    """Test RFQManager RFQ workflow."""

    @patch("httpx.AsyncClient")
    async def test_submit_rfq(self, mock_client_class, rfq_manager):
        """Test RFQ submission."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value={
            "rfq_id": "rfq-abc123",
            "status": "open",
            "quotes_expected": 3,
        })

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_client_class.return_value = mock_client

        rfq_id = await rfq_manager.submit_rfq(
            capability="visual_reasoning",
            payload_ref="Analyze comic panel for speech bubbles",
            max_price=0.50,
            max_latency_ms=5000,
            min_quality=0.85,
        )

        assert rfq_id == "rfq-abc123"
        mock_client.post.assert_called_once()

        # Verify request payload
        call_args = mock_client.post.call_args
        request_data = call_args.kwargs["json"]
        assert request_data["capability_id"] == "visual_reasoning"
        assert request_data["constraints"]["max_price"] == 0.50
        assert request_data["constraints"]["max_latency_ms"] == 5000
        assert request_data["constraints"]["min_quality"] == 0.85

    @patch("httpx.AsyncClient")
    async def test_get_quotes_returns_quotes(self, mock_client_class, rfq_manager):
        """Test getting quotes for an RFQ."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value={
            "rfq_id": "rfq-abc123",
            "status": "closed",
            "quotes": [
                {
                    "quote_id": "quote-1",
                    "provider_id": "provider-a",
                    "price": 0.08,
                    "estimated_latency_ms": 4000,
                    "quality_score": 0.94,
                },
                {
                    "quote_id": "quote-2",
                    "provider_id": "provider-b",
                    "price": 0.12,
                    "estimated_latency_ms": 3000,
                    "quality_score": 0.98,
                },
            ],
        })

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_client_class.return_value = mock_client

        quotes = await rfq_manager.get_quotes("rfq-abc123", timeout=10.0)

        assert len(quotes) == 2
        assert quotes[0].quote_id == "quote-1"
        assert quotes[0].provider_id == "provider-a"
        assert quotes[0].price == 0.08
        assert quotes[1].quote_id == "quote-2"

    def test_select_best_quote(self, rfq_manager):
        """Test selecting best quote from multiple options."""
        quotes = [
            Quote("q1", "p1", price=0.15, estimated_latency_ms=5000, quality_score=0.85),
            Quote("q2", "p2", price=0.08, estimated_latency_ms=4000, quality_score=0.94),  # Best
            Quote("q3", "p3", price=0.20, estimated_latency_ms=3000, quality_score=0.98),
        ]

        best = rfq_manager.select_best_quote(quotes)

        assert best is not None
        assert best.quote_id == "q2"  # Best balance of price/latency/quality

    def test_select_best_quote_empty_list(self, rfq_manager):
        """Test selecting from empty quote list returns None."""
        best = rfq_manager.select_best_quote([])
        assert best is None

    @patch("httpx.AsyncClient")
    async def test_accept_quote(self, mock_client_class, rfq_manager):
        """Test accepting a quote."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value={
            "contract_id": "contract-123",
            "task_id": "task-456",
            "provider_id": "provider-a",
        })

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_client_class.return_value = mock_client

        result = await rfq_manager.accept_quote("quote-xyz789")

        assert result["contract_id"] == "contract-123"
        assert result["task_id"] == "task-456"
        assert result["provider_id"] == "provider-a"

    @patch("httpx.AsyncClient")
    async def test_post_receipt(self, mock_client_class, rfq_manager):
        """Test posting receipt after task completion."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_client_class.return_value = mock_client

        await rfq_manager.post_receipt(
            contract_id="contract-123",
            task_id="task-456",
            outcome="success",
            quality_rating=0.95,
            latency_ms=4200,
            notes="Excellent results",
        )

        mock_client.post.assert_called_once()

        # Verify receipt data
        call_args = mock_client.post.call_args
        receipt_data = call_args.kwargs["json"]
        assert receipt_data["contract_id"] == "contract-123"
        assert receipt_data["task_id"] == "task-456"
        assert receipt_data["outcome"] == "success"
        assert receipt_data["quality_rating"] == 0.95
        assert receipt_data["latency_ms"] == 4200
        assert receipt_data["notes"] == "Excellent results"

    @patch("httpx.AsyncClient")
    async def test_rfq_workflow_complete(self, mock_client_class, rfq_manager):
        """Test complete RFQ workflow from submission to acceptance."""
        # Mock RFQ submission
        submit_response = Mock()
        submit_response.raise_for_status = Mock()
        submit_response.json = Mock(return_value={"rfq_id": "rfq-abc"})

        # Mock quote polling
        quotes_response = Mock()
        quotes_response.raise_for_status = Mock()
        quotes_response.json = Mock(return_value={
            "status": "closed",
            "quotes": [
                {
                    "quote_id": "quote-1",
                    "provider_id": "provider-a",
                    "price": 0.08,
                    "estimated_latency_ms": 4000,
                    "quality_score": 0.94,
                }
            ],
        })

        # Mock quote acceptance
        accept_response = Mock()
        accept_response.raise_for_status = Mock()
        accept_response.json = Mock(return_value={
            "contract_id": "contract-123",
            "task_id": "task-456",
            "provider_id": "provider-a",
        })

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[submit_response, accept_response])
        mock_client.get = AsyncMock(return_value=quotes_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_client_class.return_value = mock_client

        result = await rfq_manager.rfq_workflow(
            capability="visual_reasoning",
            payload_ref="Test task",
            max_price=0.50,
        )

        assert result["contract_id"] == "contract-123"
        assert result["task_id"] == "task-456"
        assert result["provider_id"] == "provider-a"
        assert result["selected_quote"].quote_id == "quote-1"

    @patch("httpx.AsyncClient")
    async def test_rfq_workflow_no_quotes_raises(self, mock_client_class, rfq_manager):
        """Test RFQ workflow raises if no quotes received."""
        # Mock RFQ submission
        submit_response = Mock()
        submit_response.raise_for_status = Mock()
        submit_response.json = Mock(return_value={"rfq_id": "rfq-abc"})

        # Mock quote polling - no quotes
        quotes_response = Mock()
        quotes_response.raise_for_status = Mock()
        quotes_response.json = Mock(return_value={"status": "closed", "quotes": []})

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=submit_response)
        mock_client.get = AsyncMock(return_value=quotes_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_client_class.return_value = mock_client

        with pytest.raises(RuntimeError, match="No quotes received"):
            await rfq_manager.rfq_workflow(
                capability="test",
                payload_ref="test",
            )
