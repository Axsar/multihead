"""Listing registration, trust score tracking, and knowledge claims."""

from __future__ import annotations

import hashlib
from typing import Any

import httpx

from ._constants import CLOUD_TIMEOUT, logger


class ListingsMixin:
    """Mixin providing listing registration, trust, and receipt/claim logic."""

    # Attributes defined on the main class.
    _cloud_url: str
    _cloud_agent_id: str
    _heads: Any
    _knowledge_store: Any
    _listings_cache: list[dict[str, Any]]
    _listings_cache_time: float
    _stats: dict[str, Any]
    _participant_id: str
    on_activity: Any

    def _emit(self, event_type: str, message: str) -> None: ...
    @staticmethod
    def _detect_gpu() -> str: ...
    async def _cloud_request(self, client: Any, method: str, url: str, **kw: Any) -> Any: ...
    async def _get_our_listings(self) -> list[dict[str, Any]]: ...

    # ------------------------------------------------------------------
    # Listing auto-registration
    # ------------------------------------------------------------------

    async def _register_listings(self) -> None:
        """Register our head capabilities as marketplace listings.

        Compares existing listings with current heads and creates
        any missing ones. Idempotent — safe to call on every startup.
        """
        existing = await self._get_our_listings()
        existing_caps = {l.get("capability_id", "").lower() for l in existing}

        caps_to_register = []
        for head_id, manifest in self._heads.manifests.items():
            kind = getattr(manifest, "kind", "llm")
            cap_id = f"com.multihead.{kind}.{head_id}"
            if cap_id.lower() not in existing_caps:
                caps_to_register.append((head_id, manifest, cap_id, kind))

        if not caps_to_register:
            logger.debug("All %d listings already registered", len(existing))
            return

        gpu = self._detect_gpu()
        async with httpx.AsyncClient(timeout=CLOUD_TIMEOUT) as client:
            for head_id, manifest, cap_id, kind in caps_to_register:
                name = getattr(manifest, "name", head_id)
                desc = getattr(manifest, "description", f"{kind.upper()} head: {head_id}")
                try:
                    resp = await self._cloud_request(
                        client, "POST",
                        f"{self._cloud_url}/marketplace/listings",
                        json={
                            "agent_id": self._cloud_agent_id,
                            "capability_id": cap_id,
                            "name": name,
                            "description": f"{desc} ({gpu})",
                            "pricing_model": "per_call",
                            "unit_price": 0.50,
                            "quality_score": 0.85,
                        },
                    )
                    if resp.status_code in (200, 201):
                        logger.info("Registered listing: %s (%s)", cap_id, name)
                        self._emit("listing", f"Registered {cap_id}")
                    elif resp.status_code == 409:
                        logger.debug("Listing %s already exists (conflict)", cap_id)
                    else:
                        logger.warning(
                            "Failed to register listing %s: %d %s",
                            cap_id, resp.status_code, resp.text[:200],
                        )
                except Exception as e:
                    logger.warning("Listing registration failed for %s: %s", cap_id, e)

        # Refresh cache after registration
        self._listings_cache.clear()
        self._listings_cache_time = 0.0

    # ------------------------------------------------------------------
    # Trust score tracking
    # ------------------------------------------------------------------

    async def _fetch_trust_score(self) -> float | None:
        """Fetch our current trust score from the marketplace."""
        try:
            async with httpx.AsyncClient(timeout=CLOUD_TIMEOUT) as client:
                resp = await self._cloud_request(
                    client, "GET",
                    f"{self._cloud_url}/marketplace/agents/{self._cloud_agent_id}/trust",
                )
                if resp.status_code == 200:
                    data = resp.json()
                    score = data.get("trust_score") or data.get("score")
                    if score is not None:
                        old = self._stats.get("trust_score")
                        self._stats["trust_score"] = float(score)
                        if old is not None and old != float(score):
                            self._emit("trust", f"Trust score: {old:.2f} -> {float(score):.2f}")
                        return float(score)
        except Exception as e:
            logger.debug("Failed to fetch trust score: %s", e)
        return None

    # ------------------------------------------------------------------
    # Receipt posting
    # ------------------------------------------------------------------

    async def _post_receipt(
        self,
        contract_id: str,
        task_id: str,
        *,
        outcome: str = "success",
        latency_ms: int = 0,
        confidence: float = 0.85,
        output: str = "",
        vault_entry_ids: list[str] | None = None,
    ) -> None:
        """Post completion receipt for a contract.

        Uses the cloud marketplace ReceiptCreate schema:
        receipt_type (completion|milestone|partial), artifact_hashes, metrics.
        """
        output_bytes = output.encode("utf-8") if output else b""
        output_hash = hashlib.sha256(output_bytes).hexdigest()

        artifacts = [
            {
                "ref": f"contract-{contract_id[:8]}-output",
                "sha256": output_hash,
                "size_bytes": len(output_bytes),
            }
        ]
        # Include vault output entries in artifact list
        if vault_entry_ids:
            for eid in vault_entry_ids:
                artifacts.append({
                    "ref": f"vault-{eid}",
                    "sha256": "",  # Already verified during upload
                    "size_bytes": 0,
                })

        receipt: dict[str, Any] = {
            "receipt_type": "completion" if outcome == "success" else "partial",
            "artifact_hashes": artifacts,
            "metrics": {
                "outcome": outcome,
                "confidence": confidence,
                "latency_ms": latency_ms,
                "output_preview": output[:500] if output else "",
                **({"vault_outputs": len(vault_entry_ids)} if vault_entry_ids else {}),
            },
        }

        try:
            async with httpx.AsyncClient(timeout=CLOUD_TIMEOUT) as client:
                resp = await self._cloud_request(
                    client, "POST",
                    f"{self._cloud_url}/marketplace/contracts/{contract_id}/receipts",
                    json=receipt,
                )
                if resp.status_code in (200, 201):
                    logger.info(
                        "Receipt posted: contract=%s, outcome=%s, %dms",
                        contract_id, outcome, latency_ms,
                    )
                else:
                    logger.warning(
                        "Receipt failed for contract %s: %d — %s",
                        contract_id, resp.status_code, resp.text[:200],
                    )
        except Exception as e:
            logger.warning("Failed to post receipt for %s: %s", contract_id, e)

    async def _decline_contract(self, contract_id: str, reason: str = "") -> None:
        """Decline a contract we can't fulfill.

        Calls the proper BotVibes decline endpoint which cancels the contract
        and refunds escrow. Tracks locally to stop re-polling.
        """
        self._declined_contracts.add(contract_id)
        self._cap_set(self._declined_contracts)

        try:
            params: dict[str, str] = {}
            if reason:
                params["reason"] = reason
            async with httpx.AsyncClient(timeout=CLOUD_TIMEOUT) as client:
                resp = await self._cloud_request(
                    client, "POST",
                    f"{self._cloud_url}/marketplace/contracts/{contract_id}/decline",
                    params=params,
                )
                if resp.status_code in (200, 201):
                    logger.info("Declined contract %s: %s", contract_id, reason)
                else:
                    logger.debug(
                        "Decline for %s returned %d (tracked locally)",
                        contract_id, resp.status_code,
                    )
        except Exception as e:
            logger.debug("Failed to decline %s: %s (tracked locally)", contract_id, e)

    # ------------------------------------------------------------------
    # Knowledge store integration
    # ------------------------------------------------------------------

    def _deposit_claim(self, claim_key: str, statement: str) -> None:
        """Deposit a knowledge claim for traceability."""
        if not self._knowledge_store:
            return
        try:
            from ..knowledge_models import (
                Claim, ClaimCanonical, ClaimScope, ClaimStatus, ClaimType,
                EntityRef, Provenance, ScopeType, ValueObject,
            )

            claim = Claim(
                claim_type=ClaimType.FACT,
                claim_status=ClaimStatus.ACCEPTED,
                scope=ClaimScope(
                    scope_type=ScopeType.PROJECT,
                    scope_id="multihead",
                    visibility="private",
                ),
                canonical=ClaimCanonical(
                    claim_key=claim_key,
                    subject=EntityRef(
                        entity_type="cloud_marketplace",
                        entity_id=self._cloud_agent_id,
                        entity_label="Cloud Marketplace",
                    ),
                    predicate="reports",
                    object=ValueObject(value_type="string", value=statement[:200]),
                ),
                statement=statement,
                confidence=0.95,
                provenance=Provenance(
                    produced_by={
                        "kind": "service",
                        "id": "cloud-marketplace",
                        **({"participant_id": self._participant_id} if getattr(self, "_participant_id", "") else {}),
                    },
                ),
            )
            self._knowledge_store.insert_claim(claim)
        except Exception as e:
            logger.debug("Failed to deposit claim: %s", e)
