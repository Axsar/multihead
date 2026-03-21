"""Pull-based knowledge replication between mesh peers."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..knowledge_store import KnowledgeStore
    from .peer_registry import PeerRegistry

logger = logging.getLogger(__name__)


class KnowledgeSync:
    """Replicates shared claims between mesh peers using pull-based sync.

    Each sync cycle:
    1. For each peer, fetch claims newer than our last sync timestamp
    2. Verify each claim's signature
    3. Import valid claims (skip duplicates, handle conflicts)
    4. Update last-sync watermark per peer
    """

    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        peer_registry: PeerRegistry,
        sync_interval: float = 60.0,
    ) -> None:
        self._store = knowledge_store
        self._peer_registry = peer_registry
        self._sync_interval = sync_interval
        # Track last sync timestamp per peer (node_id -> ISO timestamp)
        self._sync_watermarks: dict[str, str] = {}
        self._sync_task: asyncio.Task[None] | None = None

    async def sync_from_peer(self, peer_url: str, node_id: str, auth_token: str = "") -> int:
        """Pull new shared claims from a single peer.

        Returns:
            Number of claims imported.
        """
        from .client import MeshClient

        client = MeshClient(
            base_url=peer_url,
            auth_token=auth_token,
            timeout=15.0,
            max_retries=1,
        )

        since = self._sync_watermarks.get(node_id, "")
        try:
            claims_data = await client.fetch_claims(since=since)
        except Exception as e:
            logger.warning("Failed to fetch claims from %s (%s): %s", node_id, peer_url, e)
            return 0

        imported = 0
        for claim_dict in claims_data:
            try:
                if self._import_claim(claim_dict, node_id):
                    imported += 1
            except Exception as e:
                logger.warning(
                    "Failed to import claim %s from %s: %s",
                    claim_dict.get("claim_id", "?"), node_id, e,
                )

        # Update watermark to now
        self._sync_watermarks[node_id] = datetime.now(timezone.utc).isoformat()

        if imported:
            logger.info("Imported %d claims from peer %s", imported, node_id)
        return imported

    def _import_claim(self, claim_dict: dict[str, Any], source_node_id: str) -> bool:
        """Import a single foreign claim into local store.

        Returns True if claim was imported, False if skipped (duplicate/invalid).
        """
        from ..knowledge_models import Claim

        claim_id = claim_dict.get("claim_id", "")
        if not claim_id:
            return False

        # Skip if already exists
        existing = self._store.get_claim(claim_id)
        if existing is not None:
            return False

        # Only import shared claims
        scope_data = claim_dict.get("scope", {})
        if scope_data.get("visibility", "private") != "shared":
            return False

        # Reconstruct claim and verify signature
        try:
            claim = Claim.model_validate(claim_dict)
        except Exception as e:
            logger.warning("Invalid claim structure from %s: %s", source_node_id, e)
            return False

        # Verify signature if we have security configured
        sig_result = self._store.verify_claim(claim)
        if sig_result is False:
            logger.warning(
                "Rejected tampered claim %s from peer %s", claim_id, source_node_id
            )
            return False

        # Import as proposed (don't auto-accept foreign claims)
        from ..knowledge_models import ClaimStatus
        claim.claim_status = ClaimStatus.PROPOSED

        # Check for conflicts (same claim_key + scope with different content)
        existing_accepted = self._store.get_accepted_claim(
            claim.scope.scope_type.value,
            claim.scope.scope_id,
            claim.canonical.claim_key,
        )
        if existing_accepted and existing_accepted.statement != claim.statement:
            # Conflict: store both and record the conflict
            self._store.insert_claim(claim)
            self._store.add_claim_conflict(
                existing_accepted.claim_id,
                claim.claim_id,
                reason=f"Foreign claim from peer {source_node_id}",
            )
            # If foreign claim has higher confidence, mark conflict for review
            if claim.confidence > existing_accepted.confidence:
                logger.info(
                    "Higher-confidence foreign claim %s conflicts with local %s",
                    claim_id, existing_accepted.claim_id,
                )
            return True

        # No conflict — insert
        self._store.insert_claim(claim)
        return True

    async def sync_all_peers(self) -> dict[str, int]:
        """Sync claims from all known peers.

        Returns:
            Dict mapping node_id -> number of claims imported.
        """
        results: dict[str, int] = {}
        peer_heads = self._peer_registry.peer_heads

        # Deduplicate by node_id (a peer may have multiple heads)
        seen_nodes: dict[str, dict[str, str]] = {}
        for ph in peer_heads.values():
            if ph.status == "offline":
                continue
            if ph.node_id not in seen_nodes:
                seen_nodes[ph.node_id] = {
                    "peer_url": ph.peer_url,
                    "node_id": ph.node_id,
                }

        for node_id, info in seen_nodes.items():
            count = await self.sync_from_peer(
                peer_url=info["peer_url"],
                node_id=node_id,
            )
            results[node_id] = count

        return results

    async def start_periodic_sync(self) -> None:
        """Start background sync loop."""
        if self._sync_task is not None:
            return
        self._sync_task = asyncio.create_task(self._sync_loop())
        logger.info(
            "KnowledgeSync periodic sync started (interval=%.0fs)",
            self._sync_interval,
        )

    async def stop(self) -> None:
        """Stop background sync loop."""
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None

    async def _sync_loop(self) -> None:
        """Periodically sync claims from all peers."""
        while True:
            try:
                await self.sync_all_peers()
            except Exception as e:
                logger.warning("KnowledgeSync cycle failed: %s", e)
            await asyncio.sleep(self._sync_interval)
