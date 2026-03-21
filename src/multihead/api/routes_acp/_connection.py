"""ACP connection management routes: status, reconnect, token, env, dashboard."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from ._common import TokenLoginRequest

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/status")
async def acp_status(request: Request):
    """Return ACP/BotVibes connection status."""
    bridge = getattr(request.app.state, "acp_bridge", None)
    catalog = getattr(request.app.state, "skill_catalog", None)
    if not bridge:
        return {"connected": False, "acp_url": None, "agent_id": None, "project_id": None, "skills_loaded": 0}
    # Use catalog (single source of truth) for skill count
    skills_count = 0
    if catalog:
        try:
            counts = await catalog.count()
            skills_count = counts.get("installed", 0)
        except Exception:
            pass
    return {
        "connected": bridge.connected,
        "acp_url": bridge.acp_url,
        "agent_id": bridge._agent_id,
        "project_id": bridge._project_id,
        "skills_loaded": skills_count,
    }


@router.post("/reconnect")
async def acp_reconnect(request: Request):
    """Stop and restart the ACP bridge using current in-memory credentials.

    Does NOT reload .env — use POST /acp/env to persist, or login to refresh.
    This avoids overwriting fresh login tokens with stale .env values.
    """
    bridge = getattr(request.app.state, "acp_bridge", None)
    if not bridge:
        raise HTTPException(503, "ACP bridge not initialized")

    import os
    try:
        await bridge.stop()
        # Use current in-memory env vars (may have been updated by /acp/token)
        bridge.acp_url = os.environ.get("ACP_URL") or bridge.acp_url
        bridge._api_key = os.environ.get("ACP_API_KEY") or bridge._api_key
        bridge._project_id = os.environ.get("ACP_PROJECT_ID") or bridge._project_id
        bridge._agent_id = os.environ.get("ACP_AGENT_ID") or bridge._agent_id
        bridge._skip_registration = True  # skip re-registration on reconnect
        await bridge.start()
        bridge._skip_registration = False
        return {
            "status": "reconnected" if bridge.connected else "failed",
            "connected": bridge.connected,
            "agent_id": bridge._agent_id,
        }
    except Exception as e:
        logger.error("ACP reconnect failed: %s", e)
        raise HTTPException(502, f"Reconnect failed: {e}")


@router.post("/token")
async def acp_token(body: TokenLoginRequest, request: Request):
    """Login to BotVibes, update the ACP token, and reconnect."""
    import httpx
    import os

    url = os.environ.get("ACP_URL", "") or os.environ.get("ACP_CLOUD_URL", "")
    if not url:
        raise HTTPException(503, "ACP_URL not configured")

    # Login to BotVibes
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{url.rstrip('/')}/auth/login",
                json={"email": body.email, "password": body.password},
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, f"BotVibes login failed: {e.response.text[:200]}")
    except Exception as e:
        raise HTTPException(502, f"BotVibes login failed: {e}")

    new_token = data.get("access_token", "")
    if not new_token:
        raise HTTPException(502, "No access_token in login response")

    # Update env var in memory for bridge reconnect
    os.environ["ACP_API_KEY"] = new_token
    os.environ["ACP_CLOUD_API_KEY"] = new_token

    # Also store credentials for future auto-refresh
    os.environ["ACP_CLOUD_EMAIL"] = body.email
    os.environ["ACP_CLOUD_PASSWORD"] = body.password

    # Update agent identity from login response
    new_agent_id = data.get("agent_id", "")
    new_tenant_id = data.get("tenant_id", "")
    if new_agent_id:
        os.environ["ACP_AGENT_ID"] = new_agent_id
    if new_tenant_id:
        os.environ["ACP_PROJECT_ID"] = new_tenant_id

    # Persist token (and related keys) to .env so they survive restarts
    from ...acp_bridge._constants import persist_env_key
    try:
        persist_env_key("ACP_API_KEY", new_token)
        persist_env_key("ACP_CLOUD_API_KEY", new_token)
        if new_agent_id:
            persist_env_key("ACP_AGENT_ID", new_agent_id)
        if new_tenant_id:
            persist_env_key("ACP_PROJECT_ID", new_tenant_id)
        persist_env_key("ACP_CLOUD_EMAIL", body.email)
        persist_env_key("ACP_CLOUD_PASSWORD", body.password)
        logger.info("ACP credentials persisted to .env")
    except Exception as persist_err:
        logger.warning("Failed to persist ACP credentials to .env: %s", persist_err)

    # Reconnect the bridge with all updated credentials
    bridge = getattr(request.app.state, "acp_bridge", None)
    if bridge:
        try:
            await bridge.stop()
            bridge._api_key = new_token
            bridge._skip_registration = True  # login tokens can't re-register
            if new_agent_id:
                bridge._agent_id = new_agent_id
            if new_tenant_id:
                bridge._project_id = new_tenant_id
            await bridge.start()
            bridge._skip_registration = False
        except Exception as e:
            logger.warning("Bridge reconnect after token update failed: %s", e)

    return {
        "status": "ok",
        "connected": bridge.connected if bridge else False,
        "agent_id": new_agent_id,
        "tenant_id": new_tenant_id,
        "expires_in": data.get("expires_in_seconds", 0),
    }


@router.get("/env")
async def acp_env_get():
    """Return current ACP environment state (passwords masked)."""
    import os
    from pathlib import Path

    env_path = Path(__file__).parents[4] / ".env"
    return {
        "acp_url": os.environ.get("ACP_URL", ""),
        "acp_cloud_url": os.environ.get("ACP_CLOUD_URL", ""),
        "acp_agent_id": os.environ.get("ACP_AGENT_ID", ""),
        "acp_project_id": os.environ.get("ACP_PROJECT_ID", ""),
        "acp_cloud_email": os.environ.get("ACP_CLOUD_EMAIL", ""),
        "has_api_key": bool(os.environ.get("ACP_API_KEY", "")),
        "has_password": bool(os.environ.get("ACP_CLOUD_PASSWORD", "")),
        "env_file": str(env_path),
        "env_file_exists": env_path.exists(),
    }


@router.post("/env")
async def acp_env_save():
    """Persist current ACP credentials from os.environ to the .env file on disk."""
    import os
    from pathlib import Path

    env_path = Path(__file__).parents[4] / ".env"

    # Keys we manage
    acp_keys = [
        "ACP_CLOUD_EMAIL",
        "ACP_CLOUD_PASSWORD",
        "ACP_API_KEY",
        "ACP_AGENT_ID",
        "ACP_PROJECT_ID",
    ]

    # Read existing .env content
    existing_lines: list[str] = []
    if env_path.exists():
        existing_lines = env_path.read_text(encoding="utf-8").splitlines()

    # Parse existing key=value pairs, preserving order and comments
    updated_keys: set[str] = set()
    new_lines: list[str] = []
    for line in existing_lines:
        stripped = line.strip()
        # Detect KEY=... lines (skip comments and blank lines)
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in acp_keys:
                # Replace with current env value
                value = os.environ.get(key, "")
                new_lines.append(f"{key}={value}")
                updated_keys.add(key)
                continue
        new_lines.append(line)

    # Append any keys not already in the file
    keys_added: list[str] = []
    for key in acp_keys:
        if key not in updated_keys:
            value = os.environ.get(key, "")
            if value:  # Only add if there's a value
                new_lines.append(f"{key}={value}")
                keys_added.append(key)

    # Write back
    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    all_updated = list(updated_keys) + keys_added
    return {
        "status": "saved",
        "file": str(env_path),
        "keys_updated": sorted(all_updated),
    }


@router.get("/dashboard")
async def acp_dashboard(request: Request):
    """Return enriched BotVibes dashboard data for Cortex."""
    bridge = getattr(request.app.state, "acp_bridge", None)
    if not bridge or not bridge.connected:
        return {"connected": False}

    agent_id = bridge._agent_id
    results: dict = {"connected": True, "acp_url": bridge.acp_url, "agent_id": agent_id, "project_id": bridge._project_id}

    # Fetch balance, listings, reputation in parallel-ish
    import asyncio
    async def _fetch(method: str, path: str):
        try:
            return await bridge.request(method, path)
        except Exception:
            return None

    balance, listings, reputation = await asyncio.gather(
        _fetch("GET", f"/ledger/{agent_id}/balance"),
        _fetch("GET", f"/marketplace/agents/{agent_id}/listings"),
        _fetch("GET", f"/enforcement/plur/reputation/{agent_id}"),
    )
    results["balance"] = balance
    # Trim listing fields and filter by our agent_id
    raw_listings = listings if isinstance(listings, list) else []
    trimmed = []
    for item in raw_listings:
        lst = item.get("listing", item) if isinstance(item, dict) else {}
        if lst.get("agent_id") == agent_id:
            trimmed.append({
                "listing_id": lst.get("listing_id", ""),
                "capability_id": lst.get("capability_id", ""),
                "pricing_model": lst.get("pricing_model", ""),
                "unit_price": lst.get("unit_price", ""),
                "is_active": lst.get("is_active", True),
            })
    results["listings"] = trimmed
    results["reputation"] = reputation
    return results
