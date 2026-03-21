"""RFQ scanner — browse open RFQs and auto-quote."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from ._constants import CLOUD_TIMEOUT, LISTINGS_CACHE_TTL, MAX_SEEN_IDS, logger


class RFQMixin:
    """Mixin providing RFQ scanning and quoting logic."""

    # Attributes defined on the main class.
    _cloud_url: str
    _cloud_api_key: str
    _cloud_agent_id: str
    _cloud_project_id: str
    _heads: Any
    _quoted_rfqs: set[str]
    _listings_cache: list[dict[str, Any]]
    _listings_cache_time: float
    _stats: dict[str, Any]
    _running: bool
    on_activity: Any

    def _emit(self, event_type: str, message: str) -> None: ...
    def _svc_config(self) -> Any: ...
    def _deposit_claim(self, claim_key: str, statement: str) -> None: ...
    async def _cloud_request(self, client: Any, method: str, url: str, **kw: Any) -> Any: ...

    @staticmethod
    def _cap_set(s: set, max_size: int = MAX_SEEN_IDS) -> None:
        """Evict ~half the entries when a set exceeds max_size."""
        if len(s) > max_size:
            # Sets are unordered; remove arbitrary half
            to_remove = list(s)[:len(s) // 2]
            for item in to_remove:
                s.discard(item)

    @staticmethod
    def _detect_gpu() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
        except ImportError:
            pass
        return "CPU"

    # ------------------------------------------------------------------
    # RFQ Scanner — browse open RFQs and auto-quote
    # ------------------------------------------------------------------

    async def _rfq_scanner_loop(self) -> None:
        """Periodically scan for open RFQs and submit quotes."""
        svc = self._svc_config()
        interval = getattr(svc, "cloud_rfq_interval", 60) if svc else 60

        logger.info("RFQ scanner started (interval=%ds)", interval)

        while self._running:
            try:
                await self._scan_and_quote()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("RFQ scan failed: %s", e)

            await asyncio.sleep(interval)

    async def _scan_and_quote(self) -> None:
        """Single scan iteration: find open RFQs, submit quotes."""
        # Prime listings cache so _get_capabilities includes cloud listing caps
        if not self._listings_cache:
            await self._get_our_listings()

        # Get our capabilities from head manager + cloud listings + whitelist
        capabilities = self._get_capabilities()
        if not capabilities:
            return

        async with httpx.AsyncClient(timeout=CLOUD_TIMEOUT) as client:
            for cap in capabilities:
                try:
                    resp = await self._cloud_request(
                        client, "GET",
                        f"{self._cloud_url}/marketplace/rfqs/search",
                        params={
                            "capability_id": cap,
                            "status": "open",
                            "cross_tenant": "true",
                            "limit": 10,
                        },
                    )
                    if resp.status_code != 200:
                        continue

                    rfqs = resp.json()
                    if isinstance(rfqs, dict):
                        rfqs = rfqs.get("results", rfqs.get("rfqs", []))

                    for rfq in rfqs:
                        rfq_id = rfq.get("rfq_id", "")
                        if not rfq_id or rfq_id in self._quoted_rfqs:
                            continue

                        # Skip our own RFQs — don't bid on what we posted
                        rfq_requester = rfq.get("requester_id", "") or rfq.get("agent_id", "")
                        rfq_tenant = rfq.get("tenant_id", "") or rfq.get("project_id", "")
                        if rfq_requester == self._cloud_agent_id:
                            logger.debug("Skipping own RFQ %s (same agent_id)", rfq_id[:8])
                            self._quoted_rfqs.add(rfq_id)  # Don't re-check
                            continue
                        if rfq_tenant and rfq_tenant == self._cloud_project_id:
                            logger.debug("Skipping own-tenant RFQ %s", rfq_id[:8])
                            self._quoted_rfqs.add(rfq_id)
                            continue

                        quote = await self._compute_quote(rfq)
                        if quote:
                            await self._submit_quote(client, rfq_id, quote)
                            self._quoted_rfqs.add(rfq_id)
                            self._cap_set(self._quoted_rfqs)

                except Exception as e:
                    logger.debug("RFQ search for %s failed: %s", cap, e)

    def _get_capabilities(self) -> list[str]:
        """Get our marketplace capability IDs from head manager + cloud listings."""
        caps: list[str] = []
        try:
            states = self._heads.get_states()
            for head_id, state in states.items():
                manifest = self._heads.manifests.get(head_id)
                if manifest is None:
                    continue
                kind = getattr(manifest, "kind", "llm")
                caps.append(f"com.multihead.{kind}.{head_id}")
            # Also add generic capabilities
            if any(
                getattr(m, "kind", "") == "llm"
                for m in self._heads.manifests.values()
            ):
                caps.extend(["text_generation", "reasoning", "code_generation"])
            if any(
                getattr(m, "kind", "") == "vlm"
                for m in self._heads.manifests.values()
            ):
                caps.extend(["visual_reasoning", "object_detection"])
        except Exception as e:
            logger.debug("Failed to get capabilities: %s", e)

        # Include capabilities from cloud listings cache (covers registered
        # marketplace services like image.describe.v1, vault.*, etc.)
        for listing in self._listings_cache:
            cap = listing.get("capability_id", "")
            if cap and cap not in caps:
                caps.append(cap)

        # Include capabilities from the auto-deliver whitelist in config
        svc = self._svc_config()
        whitelist: list[str] = (
            getattr(svc, "cloud_auto_deliver_capabilities", []) if svc else []
        )
        for cap in whitelist:
            if cap not in caps:
                caps.append(cap)

        return caps

    def _capability_matches(self, rfq_capability: str) -> bool:
        """Check if an RFQ capability matches our heads.

        Uses dot-boundary prefix matching so ``text_generation`` doesn't
        accidentally match ``text_generation_unsafe``.
        """
        caps = self._get_capabilities()
        rfq_cap_lower = rfq_capability.lower()
        for cap in caps:
            cap_lower = cap.lower()
            if rfq_cap_lower == cap_lower:
                return True
            # Dot-boundary prefix matching only
            if rfq_cap_lower.startswith(cap_lower + "."):
                return True
            if cap_lower.startswith(rfq_cap_lower + "."):
                return True
        return False

    async def _compute_quote(self, rfq: dict[str, Any]) -> dict[str, Any] | None:
        """Compute a quote for an RFQ based on our capabilities and pricing."""
        cap_id = rfq.get("capability_id", "")
        if not self._capability_matches(cap_id):
            return None

        # Get our listings to find matching listing_id and pricing
        listings = await self._get_our_listings()
        matching_listing = None
        for listing in listings:
            listing_cap = listing.get("capability_id", "")
            if listing_cap.lower() == cap_id.lower() or cap_id.lower() in listing_cap.lower():
                matching_listing = listing
                break

        # Require a matching listing (empty listing_id causes 422)
        if not matching_listing:
            return None
        listing_id = matching_listing.get("listing_id", "")
        if not listing_id:
            return None
        unit_price = matching_listing.get("unit_price", 0.50)

        # Check budget constraints
        constraints = rfq.get("constraints", {})
        max_price = constraints.get("max_price")
        if max_price is not None and unit_price > max_price:
            # Price down to meet constraint if reasonable
            unit_price = max_price * 0.95  # 5% under budget

        # Floor: never go below $0.01 (avoids zero/negative quotes)
        unit_price = max(0.01, unit_price)

        return {
            "listing_id": listing_id,
            "unit_price": unit_price,
            "estimated_latency_ms": 5000,
            "estimated_confidence": 0.85,
            "metadata": {
                "agent_id": self._cloud_agent_id,
                "gpu": self._detect_gpu(),
            },
        }

    async def _submit_quote(
        self,
        client: httpx.AsyncClient,
        rfq_id: str,
        quote: dict[str, Any],
    ) -> None:
        """Submit a quote on an RFQ."""
        try:
            resp = await self._cloud_request(
                client, "POST",
                f"{self._cloud_url}/marketplace/rfqs/{rfq_id}/quotes",
                json=quote,
            )
            if resp.status_code in (200, 201):
                self._stats["quotes_sent"] += 1
                price = float(quote["unit_price"])
                latency = int(quote["estimated_latency_ms"])
                logger.info(
                    "Submitted quote on RFQ %s: $%.2f, %dms latency",
                    rfq_id, price, latency,
                )
                self._emit("quote", f"Bid ${price:.2f} on RFQ {rfq_id[:8]}")
                self._deposit_claim(
                    f"cloud.marketplace.quote.{rfq_id}",
                    f"Auto-quoted on RFQ {rfq_id}: ${price:.2f}",
                )
            else:
                # 400 = already quoted; add to set so we don't retry
                if resp.status_code == 400:
                    self._quoted_rfqs.add(rfq_id)
                    self._cap_set(self._quoted_rfqs)
                    logger.debug(
                        "Already quoted on RFQ %s, skipping", rfq_id,
                    )
                else:
                    logger.warning(
                        "Quote submission failed for RFQ %s: %d %s",
                        rfq_id, resp.status_code, resp.text[:200],
                    )
        except Exception as e:
            logger.warning("Failed to submit quote on RFQ %s: %s", rfq_id, e)

    async def _get_our_listings(self) -> list[dict[str, Any]]:
        """Get our registered marketplace listings (cached)."""
        now = time.time()
        if self._listings_cache and (now - self._listings_cache_time) < LISTINGS_CACHE_TTL:
            return self._listings_cache

        try:
            async with httpx.AsyncClient(timeout=CLOUD_TIMEOUT) as client:
                resp = await self._cloud_request(
                    client, "GET",
                    f"{self._cloud_url}/marketplace/agents/{self._cloud_agent_id}/listings",
                )
                if resp.status_code == 200:
                    data = resp.json()
                    self._listings_cache = data if isinstance(data, list) else data.get("listings", [])
                    self._listings_cache_time = now
                    logger.debug("Refreshed %d cloud listings", len(self._listings_cache))
        except Exception as e:
            logger.debug("Failed to fetch cloud listings: %s", e)

        return self._listings_cache
