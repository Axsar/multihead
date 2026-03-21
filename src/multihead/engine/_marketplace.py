"""Marketplace mixin for the MultiHead Engine."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..rfq_manager import RFQManager

if TYPE_CHECKING:
    from ..knowledge_store import KnowledgeStore

logger = logging.getLogger(__name__)


class _MarketplaceMixin:
    """Marketplace procurement and search capabilities."""

    # These attributes are provided by the Engine base class.
    _knowledge_store: KnowledgeStore | None
    _started: bool

    def _ensure_started(self) -> None: ...  # pragma: no cover

    async def marketplace_procure(
        self,
        capability: str,
        payload: str,
        *,
        max_price: float | None = None,
        max_latency_ms: int | None = None,
        min_quality: float | None = None,
        quote_timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Procure work via BotVibes cloud marketplace.

        Submits an RFQ, collects quotes, selects the best, and accepts it.

        Args:
            capability: Required capability (e.g. 'text_generation').
            payload: Task description / requirements.
            max_price: Maximum acceptable unit price.
            max_latency_ms: Maximum acceptable latency.
            min_quality: Minimum quality score (0-1).
            quote_timeout: Seconds to wait for quotes.

        Returns:
            Dict with contract_id, task_id, provider_id, quote_price,
            quote_latency_ms.

        Raises:
            RuntimeError: If cloud env vars missing, no quotes, or
                acceptance fails.
        """
        self._ensure_started()

        import os

        cloud_url = os.environ.get("ACP_CLOUD_URL", "")
        cloud_key = os.environ.get("ACP_CLOUD_API_KEY", "")
        cloud_project = os.environ.get("ACP_CLOUD_PROJECT_ID", "")

        if not cloud_url or not cloud_key:
            raise RuntimeError(
                "Cloud marketplace not configured. "
                "Set ACP_CLOUD_URL and ACP_CLOUD_API_KEY in .env"
            )

        mgr = RFQManager(
            acp_url=cloud_url,
            acp_token=cloud_key,
            project_id=cloud_project or None,
        )

        result = await mgr.rfq_workflow(
            capability,
            payload,
            max_price=max_price,
            max_latency_ms=max_latency_ms,
            min_quality=min_quality,
            quote_timeout=quote_timeout,
        )

        # Extract quote details
        selected = result.get("selected_quote")
        out: dict[str, Any] = {
            "contract_id": result.get("contract_id", ""),
            "task_id": result.get("task_id", ""),
            "provider_id": result.get("provider_id", ""),
            "quote_price": getattr(selected, "price", 0.0),
            "quote_latency_ms": getattr(selected, "estimated_latency_ms", 0),
        }

        # Deposit knowledge claim
        if self._knowledge_store:
            try:
                from datetime import datetime, timezone
                from ..knowledge_models import (
                    Claim, ClaimCanonical, ClaimScope, ClaimStatus,
                    ClaimType, EntityRef, Provenance, ScopeType,
                    Stability, ValueObject,
                )
                now = datetime.now(timezone.utc)
                self._knowledge_store.insert_claim(Claim(
                    claim_status=ClaimStatus.ACCEPTED,
                    claim_type=ClaimType.FACT,
                    scope=ClaimScope(
                        scope_type=ScopeType.PROJECT,
                        scope_id="multihead",
                        valid_from=now,
                    ),
                    canonical=ClaimCanonical(
                        claim_key=f"marketplace.procure.{out['contract_id'][:12]}",
                        subject=EntityRef(entity_type="contract", entity_id=out["contract_id"]),
                        predicate="procured",
                        object=ValueObject(value_type="string", value=capability),
                    ),
                    statement=(
                        f"Procured '{capability}' from provider {out['provider_id']} "
                        f"via marketplace — contract {out['contract_id']}, "
                        f"price ${out['quote_price']:.2f}"
                    ),
                    confidence=0.95,
                    stability=Stability.STABLE,
                    provenance=Provenance(
                        produced_by={"kind": "system", "id": "engine-sdk"},
                    ),
                ))
            except Exception as exc:
                logger.debug("Failed to deposit marketplace claim: %s", exc)

        logger.info(
            "Marketplace procure: capability=%s, contract=%s, provider=%s, price=%.2f",
            capability, out["contract_id"], out["provider_id"], out["quote_price"],
        )
        return out

    async def marketplace_search(
        self,
        capability: str,
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search marketplace for providers offering a capability.

        Args:
            capability: Capability to search for.
            limit: Max results.

        Returns:
            List of listing dicts with listing_id, agent_id, capability_id,
            name, unit_price, quality_score.
        """
        self._ensure_started()

        import os
        import httpx

        cloud_url = os.environ.get("ACP_CLOUD_URL", "")
        cloud_key = os.environ.get("ACP_CLOUD_API_KEY", "")

        if not cloud_url or not cloud_key:
            raise RuntimeError(
                "Cloud marketplace not configured. "
                "Set ACP_CLOUD_URL and ACP_CLOUD_API_KEY in .env"
            )

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{cloud_url.rstrip('/')}/marketplace/listings/search",
                headers={"Authorization": f"Bearer {cloud_key}"},
                params={
                    "capability_id": capability,
                    "cross_tenant": "true",
                    "limit": limit,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in data.get("results", []):
            listing = item.get("listing", {})
            stats = item.get("stats", {})
            results.append({
                "listing_id": listing.get("listing_id", ""),
                "agent_id": listing.get("agent_id", ""),
                "capability_id": listing.get("capability_id", ""),
                "name": listing.get("name", ""),
                "unit_price": listing.get("unit_price", 0.0),
                "quality_score": stats.get("quality_score", 0.0),
            })

        return results
