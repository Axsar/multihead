"""RFQ (Request for Quote) Manager for BotVibes marketplace.

Handles the quote-based provider selection workflow:
1. Submit RFQ to marketplace
2. Receive quotes (via WebSocket or polling)
3. Select best quote
4. Accept quote and create contract
5. Post receipt after task completion
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class Quote:
    """Represents a quote from a provider."""

    def __init__(
        self,
        quote_id: str,
        provider_id: str,
        price: float,
        estimated_latency_ms: int,
        quality_score: float,
        expires_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.quote_id = quote_id
        self.provider_id = provider_id
        self.price = price
        self.estimated_latency_ms = estimated_latency_ms
        self.quality_score = quality_score
        self.expires_at = expires_at
        self.metadata = metadata or {}

    def score(
        self,
        *,
        price_weight: float = 0.4,
        latency_weight: float = 0.3,
        quality_weight: float = 0.3,
    ) -> float:
        """Calculate weighted score for quote comparison (higher = better).

        Normalizes metrics to 0-1 range:
        - Price: inverted (lower is better)
        - Latency: inverted (lower is better)
        - Quality: direct (higher is better)
        """
        # Normalize price (assume max acceptable is $1.00)
        price_score = max(0, 1.0 - (self.price / 1.0))

        # Normalize latency (assume max acceptable is 30000ms)
        latency_score = max(0, 1.0 - (self.estimated_latency_ms / 30000))

        # Quality is already 0-1
        quality_score = self.quality_score

        return (
            price_score * price_weight
            + latency_score * latency_weight
            + quality_score * quality_weight
        )


class RFQManager:
    """Manages RFQ (Request for Quote) workflow with BotVibes marketplace."""

    def __init__(
        self,
        acp_url: str,
        acp_token: str,
        project_id: str | None = None,
    ):
        """Initialize RFQ manager.

        Args:
            acp_url: BotVibes ACP server URL
            acp_token: Authentication token
            project_id: Optional project ID for scoping
        """
        self.acp_url = acp_url.rstrip("/")
        self.acp_token = acp_token
        self.project_id = project_id

    async def submit_rfq(
        self,
        capability: str,
        payload_ref: str,
        *,
        max_price: float | None = None,
        max_latency_ms: int | None = None,
        min_quality: float | None = None,
        deadline: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Submit RFQ to marketplace.

        Args:
            capability: Required capability
            payload_ref: Task description or requirements
            max_price: Maximum acceptable price per call
            max_latency_ms: Maximum acceptable latency
            min_quality: Minimum quality score (0-1)
            deadline: ISO8601 deadline for quotes
            metadata: Additional metadata

        Returns:
            RFQ ID for tracking quotes

        Raises:
            httpx.HTTPError: If API call fails
        """
        request_data: dict[str, Any] = {
            "capability_id": capability,
            "payload_ref": payload_ref,
        }

        if max_price or max_latency_ms or min_quality:
            request_data["constraints"] = {}
            if max_price is not None:
                request_data["constraints"]["max_price"] = max_price
            if max_latency_ms is not None:
                request_data["constraints"]["max_latency_ms"] = max_latency_ms
            if min_quality is not None:
                request_data["constraints"]["min_quality"] = min_quality

        if deadline:
            request_data["deadline"] = deadline
        if self.project_id:
            request_data["project_id"] = self.project_id
        if metadata:
            request_data["metadata"] = metadata

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.acp_url}/api/v1/marketplace/rfqs",
                headers={"Authorization": f"Bearer {self.acp_token}"},
                json=request_data,
            )
            resp.raise_for_status()
            data = resp.json()

            rfq_id = data.get("rfq_id", "")
            logger.info(
                "RFQ submitted: %s, capability=%s, expected_quotes=%s",
                rfq_id, capability, data.get("quotes_expected", "unknown")
            )
            return rfq_id

    async def get_quotes(
        self,
        rfq_id: str,
        *,
        timeout: float = 30.0,
        poll_interval: float = 2.0,
    ) -> list[Quote]:
        """Get quotes for an RFQ (polling-based).

        Args:
            rfq_id: RFQ ID from submit_rfq()
            timeout: Maximum time to wait for quotes (seconds)
            poll_interval: Polling interval (seconds)

        Returns:
            List of Quote objects

        Note:
            WebSocket-based quote reception is preferred but requires
            additional setup. This polling fallback works for now.
        """
        start_time = asyncio.get_event_loop().time()
        quotes: list[Quote] = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            while (asyncio.get_event_loop().time() - start_time) < timeout:
                resp = await client.get(
                    f"{self.acp_url}/api/v1/marketplace/rfqs/{rfq_id}",
                    headers={"Authorization": f"Bearer {self.acp_token}"},
                )
                resp.raise_for_status()
                data = resp.json()

                status = data.get("status", "")
                quote_list = data.get("quotes", [])

                # Parse quotes
                quotes = [
                    Quote(
                        quote_id=q.get("quote_id", ""),
                        provider_id=q.get("provider_id", ""),
                        price=q.get("price", 0.0),
                        estimated_latency_ms=q.get("estimated_latency_ms", 0),
                        quality_score=q.get("quality_score", 0.0),
                        expires_at=q.get("expires_at"),
                        metadata=q.get("metadata", {}),
                    )
                    for q in quote_list
                ]

                # If RFQ is closed or we have quotes, return
                if status == "closed" or quotes:
                    logger.info("Received %d quotes for RFQ %s", len(quotes), rfq_id)
                    return quotes

                await asyncio.sleep(poll_interval)

        logger.warning("RFQ %s timed out after %.1fs with %d quotes", rfq_id, timeout, len(quotes))
        return quotes

    def select_best_quote(
        self,
        quotes: list[Quote],
        *,
        price_weight: float = 0.4,
        latency_weight: float = 0.3,
        quality_weight: float = 0.3,
    ) -> Quote | None:
        """Select best quote based on weighted scoring.

        Args:
            quotes: List of quotes to evaluate
            price_weight: Weight for price (lower is better)
            latency_weight: Weight for latency (lower is better)
            quality_weight: Weight for quality (higher is better)

        Returns:
            Best quote or None if no quotes available
        """
        if not quotes:
            return None

        # Score all quotes
        scored = [
            (q, q.score(price_weight=price_weight, latency_weight=latency_weight, quality_weight=quality_weight))
            for q in quotes
        ]

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        best_quote, best_score = scored[0]
        logger.info(
            "Selected best quote: %s (provider=%s, price=%.2f, score=%.2f)",
            best_quote.quote_id, best_quote.provider_id, best_quote.price, best_score
        )

        return best_quote

    async def accept_quote(self, quote_id: str) -> dict[str, Any]:
        """Accept a quote and create contract.

        Args:
            quote_id: Quote ID to accept

        Returns:
            Dict with contract_id, task_id, provider_id

        Raises:
            httpx.HTTPError: If API call fails
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.acp_url}/api/v1/marketplace/quotes/{quote_id}/accept",
                headers={"Authorization": f"Bearer {self.acp_token}"},
                json={},  # Empty body, quote_id is in URL
            )
            resp.raise_for_status()
            data = resp.json()

            contract_id = data.get("contract_id", "")
            task_id = data.get("task_id", "")
            provider_id = data.get("provider_id", "")

            logger.info(
                "Quote accepted: quote=%s, contract=%s, task=%s, provider=%s",
                quote_id, contract_id, task_id, provider_id
            )

            return {
                "contract_id": contract_id,
                "task_id": task_id,
                "provider_id": provider_id,
            }

    async def post_receipt(
        self,
        contract_id: str,
        task_id: str,
        *,
        outcome: str = "success",
        quality_rating: float | None = None,
        latency_ms: int | None = None,
        notes: str | None = None,
    ) -> None:
        """Post receipt after task completion (feedback/payment proof).

        Args:
            contract_id: Contract ID from accept_quote()
            task_id: Task ID that was executed
            outcome: Outcome status (success, failure, partial)
            quality_rating: Quality rating 0-1
            latency_ms: Actual latency observed
            notes: Optional feedback notes

        Raises:
            httpx.HTTPError: If API call fails
        """
        receipt_data: dict[str, Any] = {
            "contract_id": contract_id,
            "task_id": task_id,
            "outcome": outcome,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if quality_rating is not None:
            receipt_data["quality_rating"] = quality_rating
        if latency_ms is not None:
            receipt_data["latency_ms"] = latency_ms
        if notes:
            receipt_data["notes"] = notes

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.acp_url}/api/v1/marketplace/contracts/{contract_id}/receipts",
                headers={"Authorization": f"Bearer {self.acp_token}"},
                json=receipt_data,
            )
            resp.raise_for_status()

            logger.info(
                "Receipt posted: contract=%s, task=%s, outcome=%s",
                contract_id, task_id, outcome
            )

    async def rfq_workflow(
        self,
        capability: str,
        payload_ref: str,
        *,
        max_price: float | None = None,
        max_latency_ms: int | None = None,
        min_quality: float | None = None,
        quote_timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Complete RFQ workflow: submit → wait for quotes → select best → accept.

        Args:
            capability: Required capability
            payload_ref: Task description
            max_price: Maximum price filter
            max_latency_ms: Maximum latency filter
            min_quality: Minimum quality filter
            quote_timeout: Time to wait for quotes

        Returns:
            Dict with contract_id, task_id, provider_id, selected_quote

        Raises:
            RuntimeError: If no quotes received or quote acceptance fails
        """
        # 1. Submit RFQ
        rfq_id = await self.submit_rfq(
            capability,
            payload_ref,
            max_price=max_price,
            max_latency_ms=max_latency_ms,
            min_quality=min_quality,
        )

        # 2. Wait for quotes
        quotes = await self.get_quotes(rfq_id, timeout=quote_timeout)
        if not quotes:
            raise RuntimeError(f"No quotes received for RFQ {rfq_id}")

        # 3. Select best quote
        best_quote = self.select_best_quote(quotes)
        if not best_quote:
            raise RuntimeError(f"Failed to select best quote for RFQ {rfq_id}")

        # 4. Accept quote
        result = await self.accept_quote(best_quote.quote_id)
        result["selected_quote"] = best_quote

        return result
