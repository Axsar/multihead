"""ACP marketplace routes: procure, search, RFQ, listings, contracts, discover."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from ._common import (
    CreateListingRequest,
    PostRFQRequest,
    ProcureRequest,
    SubmitQuoteRequest,
    _get_acp,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Marketplace (cloud BotVibes)
# ---------------------------------------------------------------------------


@router.post("/marketplace/procure")
async def marketplace_procure(body: ProcureRequest, request: Request):
    """Procure a service via cloud marketplace RFQ workflow."""
    from multihead.rfq_manager import RFQManager

    bridge = _get_acp(request)
    mgr = RFQManager(acp_url=bridge.acp_url, acp_token=bridge._api_key, project_id=bridge.project_id or None)

    try:
        result = await mgr.rfq_workflow(
            body.capability,
            body.payload,
            max_price=body.max_price,
            max_latency_ms=body.max_latency_ms,
            min_quality=body.min_quality,
            quote_timeout=body.quote_timeout,
        )
        selected = result.get("selected_quote")
        return {
            "contract_id": result.get("contract_id", ""),
            "task_id": result.get("task_id", ""),
            "provider_id": result.get("provider_id", ""),
            "quote_price": getattr(selected, "price", 0.0),
            "quote_latency_ms": getattr(selected, "estimated_latency_ms", 0),
        }
    except RuntimeError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.error("Marketplace procure failed: %s", e)
        raise HTTPException(502, f"Marketplace error: {e}")


@router.get("/marketplace/search")
async def marketplace_search(capability: str, request: Request, limit: int = 10):
    """Search cloud marketplace for providers offering a capability."""
    bridge = _get_acp(request)

    try:
        data = await bridge.request(
            "GET",
            "/marketplace/listings/search",
            params={
                "capability_id": capability,
                "cross_tenant": "true",
                "limit": limit,
            },
        )
    except Exception as e:
        logger.error("Marketplace search failed: %s", e)
        raise HTTPException(502, f"Marketplace error: {e}")

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

    return {"results": results}


# ---------------------------------------------------------------------------
# Marketplace playground: RFQ lifecycle
# ---------------------------------------------------------------------------


@router.post("/marketplace/rfq")
async def post_rfq(body: PostRFQRequest, request: Request):
    """Post an RFQ to the BotVibes marketplace."""
    bridge = _get_acp(request)
    try:
        return await bridge.request(
            "POST",
            "/marketplace/rfqs",
            json={
                "capability_id": body.capability_id,
                "payload_ref": body.payload_ref,
                "budget_max": body.budget_max,
                "max_latency_ms": body.max_latency_ms,
                "project_id": bridge.project_id or None,
                **({"description": body.description} if body.description else {}),
            },
        )
    except Exception as e:
        logger.error("Post RFQ failed: %s", e)
        raise HTTPException(502, f"Marketplace error: {e}")


@router.get("/marketplace/rfq/{rfq_id}")
async def get_rfq(rfq_id: str, request: Request):
    """Get RFQ details including quotes."""
    bridge = _get_acp(request)
    try:
        return await bridge.request("GET", f"/marketplace/rfqs/{rfq_id}")
    except Exception as e:
        logger.error("Get RFQ failed: %s", e)
        raise HTTPException(502, f"Marketplace error: {e}")


@router.get("/marketplace/rfqs/search")
async def search_rfqs(
    request: Request,
    capability_id: str | None = None,
    status: str = "open",
    cross_tenant: bool = True,
    limit: int = 20,
):
    """Search open RFQs matching our capabilities."""
    bridge = _get_acp(request)
    agent_id = bridge._agent_id

    params: dict = {"status": status, "cross_tenant": str(cross_tenant).lower(), "limit": limit}
    if capability_id:
        params["capability_id"] = capability_id

    try:
        data = await bridge.request("GET", "/marketplace/rfqs/search", params=params)
        rfqs = data.get("results", data.get("rfqs", data if isinstance(data, list) else []))
        # Filter out our own RFQs
        filtered = [
            r for r in rfqs
            if r.get("requester_id", "") != agent_id
        ]
        return {"results": filtered, "total": len(filtered)}
    except Exception as e:
        logger.error("Search RFQs failed: %s", e)
        raise HTTPException(502, f"Marketplace error: {e}")


@router.post("/marketplace/rfqs/{rfq_id}/quotes")
async def submit_quote(rfq_id: str, body: SubmitQuoteRequest, request: Request):
    """Submit a quote on an open RFQ."""
    bridge = _get_acp(request)
    try:
        return await bridge.request(
            "POST",
            f"/marketplace/rfqs/{rfq_id}/quotes",
            json={
                "listing_id": body.listing_id,
                "unit_price": body.unit_price,
                "estimated_latency_ms": body.estimated_latency_ms,
                **({"message": body.message} if body.message else {}),
            },
        )
    except Exception as e:
        logger.error("Submit quote failed: %s", e)
        raise HTTPException(502, f"Marketplace error: {e}")


@router.post("/marketplace/listings")
async def create_listing(body: CreateListingRequest, request: Request):
    """Publish a new marketplace listing on BotVibes."""
    bridge = _get_acp(request)
    payload = {
        "agent_id": bridge._agent_id,
        "capability_id": body.capability_id,
        "pricing_model": body.pricing_model,
        "unit_price": body.unit_price,
        "quality_score": body.quality_score,
    }
    if body.name:
        payload["name"] = body.name
    if body.description:
        payload["description"] = body.description

    try:
        return await bridge.request("POST", "/marketplace/listings", json=payload)
    except Exception as e:
        logger.error("Create listing failed: %s", e)
        raise HTTPException(502, f"Marketplace error: {e}")


@router.get("/marketplace/listings/mine")
async def my_listings(request: Request):
    """Get our marketplace listings."""
    bridge = getattr(request.app.state, "acp_bridge", None)
    if not bridge or not bridge.connected:
        return {"listings": [], "total": 0}

    agent_id = bridge._agent_id

    try:
        data = await bridge.request("GET", f"/marketplace/agents/{agent_id}/listings")
        results = data if isinstance(data, list) else data.get("results", []) if isinstance(data, dict) else []
        # Filter by agent + trim fields
        mine = []
        for r in results:
            listing = r.get("listing", r) if isinstance(r, dict) else {}
            stats = r.get("stats", {}) if isinstance(r, dict) else {}
            if listing.get("agent_id") == agent_id:
                mine.append({
                    "listing": {
                        "listing_id": listing.get("listing_id", ""),
                        "agent_id": listing.get("agent_id", ""),
                        "capability_id": listing.get("capability_id", ""),
                        "unit_price": listing.get("unit_price", ""),
                        "is_active": listing.get("is_active", True),
                    },
                    "stats": {
                        "total_quotes": stats.get("total_quotes", 0),
                        "accepted_quotes": stats.get("accepted_quotes", 0),
                        "quality_score": stats.get("quality_score", 0),
                    },
                })
        return {"listings": mine, "total": len(mine)}
    except Exception as e:
        logger.error("Get listings failed: %s", e)
        raise HTTPException(502, f"Marketplace error: {e}")


@router.post("/marketplace/quotes/{quote_id}/accept")
async def accept_quote(quote_id: str, request: Request):
    """Accept a quote to create a contract."""
    bridge = _get_acp(request)
    try:
        return await bridge.request("POST", f"/marketplace/quotes/{quote_id}/accept", json={})
    except Exception as e:
        logger.error("Accept quote failed: %s", e)
        raise HTTPException(502, f"Marketplace error: {e}")


@router.get("/marketplace/contracts")
async def list_contracts(request: Request, role: str = "buyer"):
    """List my contracts (as buyer or provider)."""
    bridge = getattr(request.app.state, "acp_bridge", None)
    if not bridge or not bridge.connected:
        return []
    try:
        data = await bridge.request("GET", f"/marketplace/contracts?role={role}&agent_id={bridge._agent_id}&limit=20")
        return data
    except Exception as e:
        logger.error("List contracts failed: %s", e)
        raise HTTPException(502, f"Marketplace error: {e}")


@router.post("/marketplace/contracts/{contract_id}/deliver")
async def deliver_contract(contract_id: str, body: dict, request: Request):
    """Deliver work on a contract by posting a completion receipt."""
    import hashlib

    bridge = _get_acp(request)

    output = body.get("output", "")
    output_bytes = output.encode("utf-8") if output else b""
    output_hash = hashlib.sha256(output_bytes).hexdigest()

    receipt = {
        "receipt_type": "completion",
        "artifact_hashes": [
            {
                "ref": f"contract-{contract_id[:8]}-output",
                "sha256": output_hash,
                "size_bytes": len(output_bytes),
            }
        ],
        "metrics": {
            "outcome": body.get("outcome", "success"),
            "confidence": body.get("confidence", 0.95),
            "latency_ms": body.get("latency_ms", 0),
            "output_preview": output[:500] if output else "",
        },
    }

    try:
        return await bridge.request("POST", f"/marketplace/contracts/{contract_id}/receipts", json=receipt)
    except Exception as e:
        logger.error("Deliver contract failed: %s", e)
        raise HTTPException(502, f"Delivery error: {e}")


@router.post("/marketplace/contracts/{contract_id}/accept")
async def accept_delivery(contract_id: str, request: Request, body: dict | None = None):
    """Buyer accepts a delivery on a contract."""
    bridge = _get_acp(request)
    payload = {}
    if body and body.get("message"):
        payload["message"] = body["message"]

    try:
        return await bridge.request("POST", f"/marketplace/contracts/{contract_id}/accept", json=payload)
    except Exception as e:
        logger.error("Accept delivery failed: %s", e)
        raise HTTPException(502, f"Accept delivery error: {e}")


@router.post("/marketplace/contracts/{contract_id}/reject")
async def reject_delivery(contract_id: str, request: Request, body: dict | None = None):
    """Buyer rejects a delivery, sending it back for revision."""
    bridge = _get_acp(request)
    payload = {}
    if body and body.get("reason"):
        payload["reason"] = body["reason"]

    try:
        return await bridge.request("POST", f"/marketplace/contracts/{contract_id}/reject", json=payload)
    except Exception as e:
        logger.error("Reject delivery failed: %s", e)
        raise HTTPException(502, f"Reject delivery error: {e}")


@router.post("/marketplace/contracts/{contract_id}/ratings")
async def rate_provider(contract_id: str, body: dict, request: Request):
    """Rate a provider after contract completion."""
    bridge = _get_acp(request)
    payload = {
        "rating": body.get("rating", 5),
    }
    if body.get("comment"):
        payload["comment"] = body["comment"]

    try:
        return await bridge.request("POST", f"/marketplace/contracts/{contract_id}/ratings", json=payload)
    except Exception as e:
        logger.error("Rate provider failed: %s", e)
        raise HTTPException(502, f"Rate provider error: {e}")


@router.get("/marketplace/discover")
async def marketplace_discover(request: Request, capability: str | None = None, limit: int = 20):
    """Browse marketplace listings (discover providers)."""
    bridge = _get_acp(request)
    params: dict = {"limit": limit, "cross_tenant": "true"}
    if capability:
        params["capability_id"] = capability

    try:
        data = await bridge.request("GET", "/marketplace/listings/search", params=params)
        # Trim listings to essential fields (same as listings/mine)
        raw_results = data.get("results", data if isinstance(data, list) else [])
        trimmed = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            listing = item.get("listing", item)
            stats = item.get("stats", {})
            trimmed.append({
                "listing": {
                    "listing_id": listing.get("listing_id", ""),
                    "agent_id": listing.get("agent_id", ""),
                    "capability_id": listing.get("capability_id", ""),
                    "name": listing.get("name", ""),
                    "unit_price": listing.get("unit_price", 0.0),
                    "is_active": listing.get("is_active", True),
                },
                "stats": {
                    "total_quotes": stats.get("total_quotes", 0),
                    "accepted_quotes": stats.get("accepted_quotes", 0),
                    "quality_score": stats.get("quality_score", 0),
                },
            })
        return {"results": trimmed, "total": len(trimmed)}
    except Exception as e:
        logger.error("Marketplace discover failed: %s", e)
        raise HTTPException(502, f"Marketplace discover error: {e}")
