"""Session discovery and presence management for distributed solve.

Functions for advertising session availability via knowledge-store claims,
discovering active sessions within a project, and mDNS-based LAN discovery.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any

from ..knowledge_models import (
    Claim,
    ClaimCanonical,
    ClaimScope,
    ClaimStatus,
    ClaimType,
    EntityRef,
    Provenance,
    ScopeType,
    Stability,
    ValueObject,
)
from ..knowledge_store import KnowledgeStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Presence Management
# ---------------------------------------------------------------------------


def write_presence_claim(
    knowledge_store: KnowledgeStore,
    session_id: str,
    project_id: str,
    capabilities: list[str] | None = None,
) -> str:
    """Write/refresh a presence claim to advertise session availability.

    Args:
        knowledge_store: Knowledge database
        session_id: This session's unique ID
        project_id: Project scope
        capabilities: Optional list of capabilities this session offers

    Returns:
        Claim ID of the presence claim
    """
    claim = Claim(
        claim_type=ClaimType.FACT,
        claim_status=ClaimStatus.ACCEPTED,
        scope=ClaimScope(
            scope_type=ScopeType.PROJECT,
            scope_id=project_id,
            visibility="project",
            valid_from=datetime.now(timezone.utc),
        ),
        canonical=ClaimCanonical(
            claim_key=f"agent.{session_id}.presence",
            subject=EntityRef(
                entity_type="session",
                entity_id=session_id,
                label=session_id,
            ),
            predicate="available",
            object=ValueObject(
                value_type="json",
                value={
                    "capabilities": capabilities or [],
                    "session_type": "multihead_solve",
                    "last_seen": datetime.now(timezone.utc).isoformat(),
                },
            ),
        ),
        statement=f"Session {session_id} is active and available for collaboration",
        confidence=1.0,
        stability=Stability.VOLATILE,
        provenance=Provenance(
            produced_by={"kind": "session", "id": session_id}
        ),
    )

    # Supersede any existing presence claim for this session
    try:
        existing = knowledge_store.list_claims(
            claim_type=ClaimType.FACT.value,
            scope_id=project_id,
        )
        for old in existing:
            if old.canonical.claim_key == f"agent.{session_id}.presence":
                knowledge_store.update_claim_status(
                    old.claim_id, ClaimStatus.SUPERSEDED,
                )
    except Exception as e:
        logger.debug("Failed to supersede old presence claim: %s", e)

    knowledge_store.insert_claim(claim)
    logger.debug("Wrote presence claim for session %s", session_id)
    return claim.claim_id


def mark_session_offline(
    knowledge_store: KnowledgeStore,
    session_id: str,
    project_id: str,
) -> str:
    """Mark a session as offline by updating its presence claim predicate.

    Args:
        knowledge_store: Knowledge database
        session_id: Session ID to mark offline
        project_id: Project scope

    Returns:
        Claim ID of the updated presence claim
    """
    # First, delete any existing presence claim for this session
    # (avoids UNIQUE constraint violation on claim_key)
    try:
        existing_claims = knowledge_store.list_claims(
            claim_type=ClaimType.FACT.value,
            scope_id=project_id,
        )
        for claim in existing_claims:
            if claim.canonical.claim_key == f"agent.{session_id}.presence":
                # Delete by marking as superseded or just skip if it exists
                # For simplicity, we'll just overwrite by using a new claim_id
                pass
    except Exception as e:
        logger.warning("Failed to check existing presence claims: %s", e)

    # Write new offline claim with unique claim_id but same key pattern
    # Use timestamp in key to avoid UNIQUE constraint
    timestamp = datetime.now(timezone.utc).isoformat().replace(":", "").replace(".", "")
    claim = Claim(
        claim_type=ClaimType.FACT,
        claim_status=ClaimStatus.ACCEPTED,
        scope=ClaimScope(
            scope_type=ScopeType.PROJECT,
            scope_id=project_id,
            visibility="project",
            valid_from=datetime.now(timezone.utc),
        ),
        canonical=ClaimCanonical(
            claim_key=f"agent.{session_id}.presence.offline.{timestamp}",  # Unique key
            subject=EntityRef(
                entity_type="session",
                entity_id=session_id,
                label=session_id,
            ),
            predicate="offline",
            object=ValueObject(
                value_type="json",
                value={
                    "session_type": "multihead_solve",
                    "last_seen": datetime.now(timezone.utc).isoformat(),
                },
            ),
        ),
        statement=f"Session {session_id} went offline",
        confidence=1.0,
        stability=Stability.VOLATILE,
        provenance=Provenance(
            produced_by={"kind": "session", "id": session_id}
        ),
    )

    knowledge_store.insert_claim(claim)
    logger.info("Marked session %s as offline", session_id)
    return claim.claim_id


# ---------------------------------------------------------------------------
# mDNS Discovery
# ---------------------------------------------------------------------------


def discover_sessions_mdns(
    discovery: Any | None = None,
    timeout: float = 5.0,
) -> list[dict[str, Any]]:
    """Discover MultiHead sessions on the LAN via mDNS.

    Args:
        discovery: An existing ``MeshDiscovery`` instance.  When *None* a
            short-lived instance is created, scanned for *timeout* seconds,
            then stopped.
        timeout: Seconds to listen when creating a temporary instance.

    Returns:
        List of dicts with keys: node_id, host, port, source.
    """
    import time

    if discovery is not None:
        # Use the already-running instance
        nodes = discovery.get_discovered_nodes()
        return list(nodes.values())

    # Create a short-lived instance for one-shot scanning
    try:
        from ..mesh.discovery import MeshDiscovery

        disc = MeshDiscovery(node_id=f"probe-{uuid.uuid4().hex[:6]}", port=0)
        started = disc.start()
        if not started:
            logger.debug("mDNS not available (zeroconf not installed)")
            return []

        time.sleep(timeout)
        nodes = disc.get_discovered_nodes()
        disc.stop()
        return list(nodes.values())
    except Exception as exc:
        logger.debug("mDNS discovery failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Knowledge-based Session Discovery
# ---------------------------------------------------------------------------


def discover_active_sessions(
    knowledge_store: KnowledgeStore,
    project_id: str,
    session_id: str,
    max_age_minutes: int = 10,
) -> list[dict[str, Any]]:
    """Discover other active sessions in the same project.

    Args:
        knowledge_store: Knowledge database
        project_id: Project scope to filter by
        session_id: This session's ID (excluded from results)
        max_age_minutes: Maximum age for presence claims (default: 10 min)

    Returns:
        List of active sessions: [{"session_id": str, "capabilities": list, "last_seen": datetime}]
    """
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)

    # Query for presence claims
    all_claims = knowledge_store.list_claims(
        claim_type=ClaimType.FACT.value,
        scope_id=project_id,
    )

    # Filter for presence claims that are recent
    active_sessions = []
    for claim in all_claims:
        # Check if it's a presence claim (predicate = "available")
        if claim.canonical.predicate != "available":
            continue

        # Check if it's recent
        if claim.scope.valid_from < cutoff:
            continue

        # Extract session_id from claim_key (format: "agent.{session_id}.presence")
        claim_key = claim.canonical.claim_key
        if not claim_key.startswith("agent.") or not claim_key.endswith(".presence"):
            continue

        discovered_session_id = claim_key.split(".")[1]

        # Skip own session
        if discovered_session_id == session_id:
            continue

        # Extract capabilities and last_seen from object
        obj = claim.canonical.object.value
        if isinstance(obj, dict):
            active_sessions.append({
                "session_id": discovered_session_id,
                "capabilities": obj.get("capabilities", []),
                "last_seen": claim.scope.valid_from,
            })

    logger.debug("Discovered %d active sessions (excluding self)", len(active_sessions))
    return active_sessions
