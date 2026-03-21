#!/usr/bin/env python3
"""Marketplace Activity Loop — keeps the BotVibes marketplace alive.

Runs every 10 minutes, cycling through marketplace actions:
  Round 0: Browse RFQs → bid on one
  Round 1: Post a new RFQ
  Round 2: Browse listings → buy cheapest
  Round 3: Check our trust score + stats

Usage:
    python scripts/marketplace_activity.py
    # or run in background via MultiHead process manager
"""

import asyncio
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

# Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [marketplace] %(message)s")
log = logging.getLogger("marketplace_activity")

# Add src to path for knowledge store
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── Config ────────────────────────────────────────────────────

CLOUD_URL = os.environ.get(
    "ACP_CLOUD_URL",
    "https://botvibes.io/api/v1",
)
CLOUD_KEY = os.environ.get("ACP_CLOUD_API_KEY", "")
CLOUD_AGENT = os.environ.get("ACP_CLOUD_AGENT_ID", "multihead-brain-agent")
CLOUD_PROJECT = os.environ.get("ACP_CLOUD_PROJECT_ID", "")

INTERVAL_SECONDS = 600  # 10 minutes
TIMEOUT = 30.0

# Capabilities we can bid on (we have local GPU for these)
OUR_CAPABILITIES = {
    "comic.detect.objects.v1": 0.05,
    "comic.enrich.relationships.v1": 0.01,
    "comic.segment.masks.v1": 0.15,
    "comic.generate.tails.v1": 0.02,
    "comic.extract.text.v1": 0.05,
    "comic.generate.catalog.v1": 0.02,
    "comic.attribute.speakers.v1": 1.00,
    "comic.analyze.narrative.v1": 1.50,
    "comic.observe.visual.v1": 2.00,
    "comic.fuse.claims.v1": 1.00,
    "comic.validate.claims.v1": 0.50,
    "ai.text.generate.v1": 0.50,
    "ai.vision.analyze.v1": 1.50,
    "ai.task.decompose.v1": 2.00,
    "knowledge.rag.query.v1": 0.10,
    "image.describe.v1": 2.00,
    "code.review.v1": 3.00,
    "ai.tree_of_thoughts.v1": 5.00,
}

# Capabilities we might want to buy (test procurement)
BUY_INTERESTS = [
    {"capability": "code.generation.v1", "budget": 5.0, "desc": "Generate a Python helper function for JSON validation"},
    {"capability": "code.review.v1", "budget": 3.0, "desc": "Review cloud_marketplace.py for edge cases"},
    {"capability": "nlp.research.v1", "budget": 2.0, "desc": "Research best practices for agent trust scoring"},
    {"capability": "comic.detect.objects.v1", "budget": 1.0, "desc": "Detect objects in a sample comic page"},
    {"capability": "comic.segment.masks.v1", "budget": 1.0, "desc": "Generate segmentation masks for comic panel"},
    {"capability": "ai.text.generate.v1", "budget": 1.0, "desc": "Summarize the MultiHead architecture in 3 sentences"},
    {"capability": "knowledge.rag.query.v1", "budget": 0.5, "desc": "Query knowledge base for recent consensus decisions"},
]


def _jwt_exp(token: str) -> float | None:
    """Extract 'exp' claim from JWT without verification."""
    import base64
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)  # pad base64
        data = json.loads(base64.urlsafe_b64decode(payload))
        return float(data["exp"])
    except Exception:
        return None


async def _refresh_token() -> bool:
    """Refresh the JWT token. Returns True if successful."""
    global CLOUD_KEY
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(
                f"{CLOUD_URL}/auth/refresh",
                headers={"Authorization": f"Bearer {CLOUD_KEY}"},
            )
            if resp.status_code == 200:
                data = resp.json()
                new_token = data.get("access_token", data.get("token", ""))
                if new_token:
                    CLOUD_KEY = new_token
                    log.info("Token refreshed successfully")
                    return True
            # Try re-login as fallback
            email = os.environ.get("ACP_CLOUD_EMAIL", "")
            password = os.environ.get("ACP_CLOUD_PASSWORD", "")
            if email and password:
                resp = await client.post(
                    f"{CLOUD_URL}/auth/login",
                    json={"email": email, "password": password},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    new_token = data.get("access_token", data.get("token", ""))
                    if new_token:
                        CLOUD_KEY = new_token
                        log.info("Re-login successful, token updated")
                        return True
            log.warning("Token refresh failed (no credentials for re-login)")
            return False
    except Exception as e:
        log.warning("Token refresh error: %s", e)
        return False


async def _ensure_token():
    """Proactively refresh token if close to expiry."""
    exp = _jwt_exp(CLOUD_KEY)
    if exp is not None and exp - time.time() < 300:  # 5 min margin
        await _refresh_token()


def _headers():
    return {
        "Authorization": f"Bearer {CLOUD_KEY}",
        "Content-Type": "application/json",
    }


def _log_to_knowledge(action: str, details: str):
    """Best-effort log to knowledge.db."""
    try:
        from multihead.knowledge_store import KnowledgeStore
        _data_dir = Path(os.environ.get("MULTIHEAD_DATA_DIR", str(Path.home() / ".multihead")))
        ks = KnowledgeStore(_data_dir / "knowledge.db")
        from multihead.knowledge_store import (
            Claim, ClaimScope, ClaimCanonical, EntityRef, ValueObject,
        )
        from multihead.knowledge_models import ClaimType, ScopeType, Provenance

        ts = datetime.now(timezone.utc).strftime("%H:%M")
        claim = Claim(
            claim_type=ClaimType.FACT,
            scope=ClaimScope(scope_type=ScopeType.PROJECT, scope_id="multihead"),
            canonical=ClaimCanonical(
                claim_key=f"marketplace.activity.{int(time.time())}",
                subject=EntityRef(entity_type="agent", entity_id=CLOUD_AGENT),
                predicate=action,
                object=ValueObject(value_type="activity", value=details[:200]),
            ),
            statement=f"[{ts}] Marketplace activity — {action}: {details}",
            confidence=1.0,
            importance=0.3,
            provenance=Provenance(produced_by={"id": "marketplace-activity-loop"}),
        )
        ks.insert_claim(claim)
    except Exception as e:
        log.debug("Knowledge log failed (non-critical): %s", e)


# ── Actions ───────────────────────────────────────────────────

async def action_browse_and_bid(client: httpx.AsyncClient) -> str:
    """Browse open RFQs and bid on one we can fulfill."""
    # Search for open RFQs
    resp = await client.get(
        f"{CLOUD_URL}/marketplace/rfqs/search",
        headers=_headers(),
        params={"status": "open", "cross_tenant": "true", "limit": 10},
    )
    if resp.status_code != 200:
        return f"RFQ search failed: {resp.status_code} {resp.text[:200]}"

    data = resp.json()
    rfqs = data.get("results", data.get("rfqs", []))
    if not rfqs:
        return "No open RFQs found"

    # Find one we can bid on
    for rfq in rfqs:
        cap = rfq.get("capability_id", "")
        rfq_id = rfq.get("rfq_id", rfq.get("id", ""))
        if cap in OUR_CAPABILITIES and rfq_id:
            our_price = OUR_CAPABILITIES[cap]
            # Get our listing ID for this capability
            listings_resp = await client.get(
                f"{CLOUD_URL}/marketplace/listings/search",
                headers=_headers(),
                params={"capability_id": cap, "agent_id": CLOUD_AGENT},
            )
            listing_id = None
            if listings_resp.status_code == 200:
                listings_data = listings_resp.json()
                results = listings_data.get("results", listings_data.get("listings", []))
                for item in results:
                    # Handle nested format: {listing: {listing_id: ...}} or flat
                    inner = item.get("listing", item)
                    lid = inner.get("listing_id", inner.get("id"))
                    aid = inner.get("agent_id", "")
                    if aid == CLOUD_AGENT and lid:
                        listing_id = lid
                        break
                # fallback: take first result if none matched our agent
                if not listing_id and results:
                    inner = results[0].get("listing", results[0])
                    listing_id = inner.get("listing_id", inner.get("id"))

            if not listing_id:
                log.debug("No listing found for %s, skipping", cap)
                continue

            quote_body = {
                "listing_id": listing_id,
                "unit_price": our_price,
                "estimated_latency_ms": 30000,
                "estimated_confidence": 0.90,
                "metadata": {"agent_id": CLOUD_AGENT, "gpu": "RTX 4090"},
            }

            bid_resp = await client.post(
                f"{CLOUD_URL}/marketplace/rfqs/{rfq_id}/quotes",
                headers=_headers(),
                json=quote_body,
            )
            if bid_resp.status_code in (200, 201):
                msg = f"Bid {our_price}cr on RFQ {rfq_id[:8]} for {cap}"
                log.info(msg)
                return msg
            else:
                log.warning("Bid failed: %s %s", bid_resp.status_code, bid_resp.text[:200])

    return f"Browsed {len(rfqs)} RFQs, none matched our capabilities or already quoted"


async def action_post_rfq(client: httpx.AsyncClient) -> str:
    """Post a new RFQ for something we want to buy."""
    interest = random.choice(BUY_INTERESTS)
    cap = interest["capability"]
    budget = interest["budget"]
    desc = interest["desc"]

    body = {
        "capability_id": cap,
        "budget_max": budget,
        "description": desc,
        "payload_ref": desc,
        "constraints": {
            "max_price": budget,
            "max_latency_ms": 120000,
            "min_quality": 0.7,
        },
    }
    if CLOUD_PROJECT:
        body["project_id"] = CLOUD_PROJECT

    resp = await client.post(
        f"{CLOUD_URL}/marketplace/rfqs",
        headers=_headers(),
        json=body,
    )
    if resp.status_code in (200, 201):
        rfq_data = resp.json()
        rfq_id = rfq_data.get("rfq_id", rfq_data.get("id", "?"))
        msg = f"Posted RFQ {str(rfq_id)[:8]} for {cap} (budget {budget}cr): {desc[:60]}"
        log.info(msg)
        return msg
    else:
        return f"RFQ post failed: {resp.status_code} {resp.text[:200]}"


async def action_browse_and_buy(client: httpx.AsyncClient) -> str:
    """Browse listings and accept a cheap one."""
    # Pick a random capability to search for
    interest = random.choice(BUY_INTERESTS)
    cap = interest["capability"]

    resp = await client.get(
        f"{CLOUD_URL}/marketplace/listings/search",
        headers=_headers(),
        params={"capability_id": cap, "cross_tenant": "true", "limit": 5},
    )
    if resp.status_code != 200:
        return f"Listing search failed: {resp.status_code}"

    data = resp.json()
    raw_listings = data.get("results", data.get("listings", []))
    if not raw_listings:
        return f"No listings found for {cap}"

    # Normalize nested format
    listings = []
    for item in raw_listings:
        inner = item.get("listing", item)
        listings.append(inner)

    # Find cheapest that isn't us
    cheapest = None
    for l in listings:
        provider = l.get("agent_id", l.get("provider_id", ""))
        price_val = float(l.get("unit_price", 999))
        if provider != CLOUD_AGENT:
            if cheapest is None or price_val < float(cheapest.get("unit_price", 999)):
                cheapest = l

    if not cheapest:
        return f"Found {len(listings)} listings for {cap}, all ours"

    price = cheapest.get("unit_price", "?")
    provider = cheapest.get("agent_id", cheapest.get("provider_id", "?"))
    msg = f"Found listing: {cap} @ {price}cr from {provider}"
    log.info(msg)
    return msg


async def action_check_stats(client: httpx.AsyncClient) -> str:
    """Check our trust score and marketplace stats."""
    parts = []

    # Contracts (shows trust via completed count)
    resp = await client.get(
        f"{CLOUD_URL}/marketplace/contracts",
        headers=_headers(),
        params={"limit": 50},
    )
    if resp.status_code == 200:
        all_contracts = resp.json() if isinstance(resp.json(), list) else resp.json().get("contracts", [])
        completed = sum(1 for c in all_contracts if c.get("status") in ("completed", "settled"))
        active = sum(1 for c in all_contracts if c.get("status") == "active")
        parts.append(f"contracts_completed={completed}")
        parts.append(f"contracts_active={active}")

    # Our listings count
    resp2 = await client.get(
        f"{CLOUD_URL}/marketplace/listings/search",
        headers=_headers(),
        params={"agent_id": CLOUD_AGENT, "limit": 50},
    )
    if resp2.status_code == 200:
        data = resp2.json()
        count = len(data.get("results", data.get("listings", [])))
        parts.append(f"listings={count}")

    # Open RFQs
    resp3 = await client.get(
        f"{CLOUD_URL}/marketplace/rfqs/search",
        headers=_headers(),
        params={"status": "open", "cross_tenant": "true", "limit": 50},
    )
    if resp3.status_code == 200:
        data = resp3.json()
        rfqs = data.get("results", data.get("rfqs", []))
        parts.append(f"open_rfqs={len(rfqs)}")

    msg = f"Stats: {', '.join(parts)}" if parts else "Stats: couldn't fetch"
    log.info(msg)
    return msg


# ── Main Loop ─────────────────────────────────────────────────

ACTIONS = [
    ("bid_on_rfq", action_browse_and_bid),
    ("post_rfq", action_post_rfq),
    ("browse_listings", action_browse_and_buy),
    ("check_stats", action_check_stats),
]


async def main():
    if not CLOUD_KEY:
        log.error("ACP_CLOUD_API_KEY not set. Export it or add to .env")
        sys.exit(1)

    log.info("Marketplace activity loop starting (every %ds)", INTERVAL_SECONDS)
    log.info("  URL: %s", CLOUD_URL)
    log.info("  Agent: %s", CLOUD_AGENT)

    round_num = 0
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        while True:
            action_name, action_fn = ACTIONS[round_num % len(ACTIONS)]
            log.info("Round %d — action: %s", round_num, action_name)

            try:
                await _ensure_token()
                result = await action_fn(client)
                # Retry once on 401
                if "401" in str(result) and await _refresh_token():
                    result = await action_fn(client)
                log.info("Result: %s", result)
                _log_to_knowledge(action_name, result)
            except httpx.HTTPStatusError as e:
                log.warning("HTTP error: %s", e)
            except httpx.ConnectError as e:
                log.warning("Connection error: %s", e)
            except Exception as e:
                log.error("Unexpected error: %s", e, exc_info=True)

            round_num += 1
            log.info("Sleeping %ds until next round...", INTERVAL_SECONDS)
            await asyncio.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    # Load .env if dotenv available
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent / ".env", override=True)
        # Re-read after loading
        globals()["CLOUD_URL"] = os.environ.get("ACP_CLOUD_URL", CLOUD_URL)
        globals()["CLOUD_KEY"] = os.environ.get("ACP_CLOUD_API_KEY", CLOUD_KEY)
        globals()["CLOUD_AGENT"] = os.environ.get("ACP_CLOUD_AGENT_ID", CLOUD_AGENT)
        globals()["CLOUD_PROJECT"] = os.environ.get("ACP_CLOUD_PROJECT_ID", CLOUD_PROJECT)
    except ImportError:
        pass

    asyncio.run(main())
