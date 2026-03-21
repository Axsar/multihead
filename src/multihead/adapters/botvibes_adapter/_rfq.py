"""RFQ (Request for Quote) procurement cycle for BotVibes adapter."""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from typing import Any

import httpx

from ._constants import (
    _MAX_WAIT_TIME_S,
    _RFQ_BID_POLL_S,
    _RFQ_BID_WAIT_S,
    _TASK_POLL_INTERVAL_S,
)

logger = logging.getLogger(__name__)


class RfqMixin:
    """Mixin providing RFQ procurement methods.

    Expects the consuming class to have:
    - ``acp_url``: str
    - ``project_id``: str
    - ``target_capability``: str
    - ``manifest``: HeadManifest
    - ``total_cost``: float
    - ``call_count``: int
    - ``_use_rfq``: bool
    - ``_request_with_retry()``: from HttpMixin
    - ``_auth_headers()``: from HttpMixin
    - ``_record_to_knowledge()``: method
    - ``_generate_simple()``: method
    """

    async def _generate_rfq(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Full RFQ procurement: post RFQ -> collect bids -> award -> wait -> verify.

        This is the marketplace-native path where we act as a buyer:
        1. Post RFQ describing what we need
        2. Wait for bids from marketplace providers
        3. Score bids and award to best provider
        4. Wait for delivery
        5. Accept and release escrow (or dispute)
        """
        start_time = time.time()
        timeout_s = kwargs.get("timeout_s", _MAX_WAIT_TIME_S)
        bid_wait_s = kwargs.get("bid_wait_s", _RFQ_BID_WAIT_S)
        budget = kwargs.get("budget")
        max_cost = budget.max_cost_per_step if budget and budget.max_cost_per_step else 10.0

        from ._constants import _ACP_TIMEOUT

        async with httpx.AsyncClient(timeout=_ACP_TIMEOUT) as client:
            # Step 1: Post RFQ
            rfq_id = await self._post_rfq(client, prompt, max_cost)
            logger.info(
                "BotVibes RFQ %s posted for %s (budget=%.2f)",
                rfq_id[:8], self.target_capability, max_cost,  # type: ignore[attr-defined]
            )

            # Step 2: Wait for bids
            bids = await self._collect_bids(client, rfq_id, bid_wait_s)
            if not bids:
                # No bids — cancel RFQ and fall back to simple task mode
                await self._cancel_rfq(client, rfq_id)
                logger.warning("No bids on RFQ %s, falling back to simple task", rfq_id[:8])
                return await self._generate_simple(prompt, **kwargs)  # type: ignore[attr-defined]

            # Step 3: Score and award best bid
            best_bid = self._score_bids(bids)
            contract_id = await self._award_bid(client, rfq_id, best_bid)
            logger.info(
                "Awarded contract %s to %s (%.2f cr)",
                contract_id[:8],
                best_bid.get("agent_id", "?"),
                best_bid.get("unit_price", 0),
            )

            # Step 4: Wait for delivery
            delivery = await self._wait_for_delivery(client, contract_id, timeout_s)
            output_text = delivery.get("output_text", "")

            # Step 5: Accept delivery (auto-accept for now)
            await self._accept_delivery(client, contract_id)

        latency_ms = int((time.time() - start_time) * 1000)
        cost = best_bid.get("unit_price", 0.0)
        self.total_cost += cost  # type: ignore[attr-defined]
        self.call_count += 1  # type: ignore[attr-defined]

        self._record_to_knowledge(True, latency_ms, cost)  # type: ignore[attr-defined]

        return {
            "text": output_text,
            "tokens_in": len(prompt) // 4,
            "tokens_out": len(output_text) // 4,
            "latency_ms": latency_ms,
            "provider_metadata": {
                "rfq_id": rfq_id,
                "contract_id": contract_id,
                "provider_id": best_bid.get("agent_id", ""),
                "bid_price": cost,
                "bids_received": len(bids),
                "mode": "rfq",
            },
        }

    async def _post_rfq(
        self,
        client: httpx.AsyncClient,
        description: str,
        max_cost: float,
    ) -> str:
        """Post an RFQ to the marketplace."""
        body: dict[str, Any] = {
            "capability_id": self.target_capability,  # type: ignore[attr-defined]
            "budget_max": max_cost,
            "description": description[:500],
            "payload_ref": description,
            "constraints": {
                "max_price": max_cost,
                "max_latency_ms": int(_MAX_WAIT_TIME_S * 1000),
                "min_quality": 0.7,
            },
        }
        if self.project_id:  # type: ignore[attr-defined]
            body["project_id"] = self.project_id  # type: ignore[attr-defined]

        resp = await self._request_with_retry(  # type: ignore[attr-defined]
            client, "POST", f"{self.acp_url}/marketplace/rfqs",  # type: ignore[attr-defined]
            headers=self._auth_headers(), json=body,  # type: ignore[attr-defined]
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("rfq_id", data.get("id", ""))

    async def _collect_bids(
        self,
        client: httpx.AsyncClient,
        rfq_id: str,
        wait_s: float,
    ) -> list[dict[str, Any]]:
        """Poll for bids on an RFQ until timeout or sufficient bids."""
        start = time.time()
        best_bids: list[dict[str, Any]] = []

        while time.time() - start < wait_s:
            resp = await self._request_with_retry(  # type: ignore[attr-defined]
                client, "GET",
                f"{self.acp_url}/marketplace/rfqs/{rfq_id}/quotes",  # type: ignore[attr-defined]
                headers=self._auth_headers(),  # type: ignore[attr-defined]
            )
            if resp.status_code == 200:
                data = resp.json()
                quotes = data if isinstance(data, list) else data.get("quotes", [])
                if quotes:
                    best_bids = quotes
                    # Got at least one bid — wait a bit more for competition
                    if time.time() - start > wait_s * 0.5:
                        break

            await asyncio.sleep(_RFQ_BID_POLL_S)

        return best_bids

    def _score_bids(self, bids: list[dict[str, Any]]) -> dict[str, Any]:
        """Score bids and return the best one.

        Scoring: 50% price (lower is better) + 30% confidence + 20% latency.
        """
        if len(bids) == 1:
            return bids[0]

        max_price = max(b.get("unit_price", 1.0) for b in bids) or 1.0

        scored = []
        for bid in bids:
            price = bid.get("unit_price", max_price)
            confidence = bid.get("estimated_confidence", 0.5)
            latency = bid.get("estimated_latency_ms", 30000)

            price_score = 1.0 - (price / max_price) if max_price > 0 else 0.5
            latency_score = max(0, 1.0 - latency / 60000)

            total = 0.50 * price_score + 0.30 * confidence + 0.20 * latency_score
            scored.append((total, bid))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    async def _award_bid(
        self,
        client: httpx.AsyncClient,
        rfq_id: str,
        bid: dict[str, Any],
    ) -> str:
        """Accept a quote/bid and create a contract."""
        quote_id = bid.get("quote_id", bid.get("id", ""))

        resp = await self._request_with_retry(  # type: ignore[attr-defined]
            client, "POST",
            f"{self.acp_url}/marketplace/rfqs/{rfq_id}/accept",  # type: ignore[attr-defined]
            headers=self._auth_headers(),  # type: ignore[attr-defined]
            json={"quote_id": quote_id},
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("contract_id", data.get("id", ""))

    async def _cancel_rfq(self, client: httpx.AsyncClient, rfq_id: str) -> None:
        """Cancel an RFQ (best-effort)."""
        try:
            await self._request_with_retry(  # type: ignore[attr-defined]
                client, "POST",
                f"{self.acp_url}/marketplace/rfqs/{rfq_id}/cancel",  # type: ignore[attr-defined]
                headers=self._auth_headers(),  # type: ignore[attr-defined]
            )
        except Exception as e:
            logger.debug("Failed to cancel RFQ %s: %s", rfq_id[:8], e)

    async def _wait_for_delivery(
        self,
        client: httpx.AsyncClient,
        contract_id: str,
        timeout_s: float,
    ) -> dict[str, Any]:
        """Poll contract status until delivered or timeout."""
        start = time.time()
        poll_interval = _TASK_POLL_INTERVAL_S

        while time.time() - start < timeout_s:
            resp = await self._request_with_retry(  # type: ignore[attr-defined]
                client, "GET",
                f"{self.acp_url}/marketplace/contracts/{contract_id}",  # type: ignore[attr-defined]
                headers=self._auth_headers(),  # type: ignore[attr-defined]
            )
            if resp.status_code == 200:
                contract = resp.json()
                status = contract.get("status", "")

                if status == "delivered":
                    # Try to get vault content
                    output_text = await self._fetch_vault_content(client, contract_id)
                    return {"output_text": output_text, "contract": contract}

                if status in ("completed", "settled"):
                    output_text = contract.get("delivery_notes", contract.get("output_ref", ""))
                    return {"output_text": output_text, "contract": contract}

                if status in ("cancelled", "expired", "disputed"):
                    raise RuntimeError(f"Contract {contract_id[:8]} ended with status: {status}")

            await asyncio.sleep(poll_interval)
            # Adaptive polling: slow down after 30s
            if time.time() - start > 30:
                poll_interval = min(poll_interval * 1.5, 10.0)

        raise TimeoutError(f"Contract {contract_id[:8]} not delivered within {timeout_s}s")

    async def _fetch_vault_content(
        self,
        client: httpx.AsyncClient,
        contract_id: str,
    ) -> str:
        """Try to download vault content for a delivered contract."""
        try:
            # List vault entries
            resp = await self._request_with_retry(  # type: ignore[attr-defined]
                client, "GET",
                f"{self.acp_url}/vault/contracts/{contract_id}/entries",  # type: ignore[attr-defined]
                headers=self._auth_headers(),  # type: ignore[attr-defined]
            )
            if resp.status_code != 200:
                return ""

            entries = resp.json()
            if isinstance(entries, dict):
                entries = entries.get("entries", [])

            for entry in entries:
                entry_id = entry.get("entry_id", entry.get("id", ""))
                if not entry_id:
                    continue

                # Request download URL
                dl_resp = await self._request_with_retry(  # type: ignore[attr-defined]
                    client, "POST",
                    f"{self.acp_url}/vault/entries/{entry_id}/download",  # type: ignore[attr-defined]
                    headers=self._auth_headers(),  # type: ignore[attr-defined]
                )
                if dl_resp.status_code != 200:
                    continue

                dl_data = dl_resp.json()
                download_url = dl_data.get("download_url", "")
                if not download_url:
                    continue

                # Download the file
                file_resp = await client.get(download_url)
                if file_resp.status_code == 200:
                    # Try to decode as text
                    try:
                        return file_resp.text
                    except Exception:
                        return base64.b64encode(file_resp.content).decode("ascii")

        except Exception as e:
            logger.debug("Vault fetch failed for contract %s: %s", contract_id[:8], e)

        return ""

    async def _accept_delivery(
        self,
        client: httpx.AsyncClient,
        contract_id: str,
    ) -> None:
        """Accept delivery and release escrow."""
        try:
            resp = await self._request_with_retry(  # type: ignore[attr-defined]
                client, "POST",
                f"{self.acp_url}/marketplace/contracts/{contract_id}/accept",  # type: ignore[attr-defined]
                headers=self._auth_headers(),  # type: ignore[attr-defined]
                json={"rating": 5, "feedback": "Automated acceptance via MultiHead"},
            )
            if resp.status_code in (200, 201):
                logger.info("Delivery accepted for contract %s", contract_id[:8])
            else:
                logger.warning("Accept delivery returned %d for %s", resp.status_code, contract_id[:8])
        except Exception as e:
            logger.warning("Failed to accept delivery for %s: %s", contract_id[:8], e)
