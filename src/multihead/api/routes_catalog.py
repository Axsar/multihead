"""API routes for the skill catalog — browse, search, install, publish."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_catalog(request: Request):
    catalog = getattr(request.app.state, "skill_catalog", None)
    if not catalog:
        raise HTTPException(503, "Skill catalog not initialized")
    return catalog


@router.get("/catalog")
async def list_catalog(
    request: Request,
    category: str | None = None,
    installed: bool = False,
    search: str | None = None,
    limit: int = 100,
):
    """Browse the skill catalog with optional filters."""
    catalog = _get_catalog(request)
    entries = await catalog.list_all(
        category=category, installed_only=installed, search=search, limit=limit
    )
    return {
        "skills": [e.to_dict() for e in entries],
        "count": len(entries),
    }


@router.get("/catalog/stats")
async def catalog_stats(request: Request):
    """Get catalog statistics."""
    catalog = _get_catalog(request)
    counts = await catalog.count()
    categories = await catalog.categories()
    last_sync = await catalog.get_meta("last_github_sync")
    return {
        **counts,
        "categories": categories,
        "last_github_sync": last_sync,
    }


@router.post("/catalog/sync")
async def sync_catalog(request: Request):
    """Sync catalog with GitHub (fetch available skills)."""
    from ..skill_catalog import fetch_github_skills

    catalog = _get_catalog(request)
    entries = await fetch_github_skills()
    if not entries:
        raise HTTPException(502, "Failed to fetch skills from GitHub")

    # Mark already-installed skills
    skills_dir = request.app.state.settings.skills_dir
    for entry in entries:
        existing = await catalog.get(entry.name)
        if existing:
            entry.installed = existing.installed
            entry.installed_at = existing.installed_at
            entry.published = existing.published
            entry.published_at = existing.published_at
        elif (skills_dir / entry.name / "SKILL.md").is_file():
            entry.installed = True
        await catalog.upsert(entry)

    from datetime import datetime, timezone
    await catalog.set_meta("last_github_sync", datetime.now(timezone.utc).isoformat())

    return {"synced": len(entries), "source": "github.com/anthropics/skills"}


class InstallRequest(BaseModel):
    skill_name: str


@router.post("/catalog/install")
async def install_skill(body: InstallRequest, request: Request):
    """Download and install a skill from GitHub."""
    from ..skill_catalog import install_skill_from_github

    catalog = _get_catalog(request)
    entry = await catalog.get(body.skill_name)

    # License check: warn if skill requires a license
    if entry and entry.requires_license:
        logger.info(
            "Installing licensed skill '%s' (type=%s, price=%.2f)",
            body.skill_name, entry.license_type, entry.license_price,
        )

    skills_dir = request.app.state.settings.skills_dir
    result = await install_skill_from_github(body.skill_name, skills_dir)
    if not result:
        raise HTTPException(502, f"Failed to install skill '{body.skill_name}'")

    # Mark installed in catalog
    await catalog.mark_installed(body.skill_name)

    # Load into active skill registry
    skill_registry = getattr(request.app.state, "skill_registry", None)
    if skill_registry:
        from ..skill_loader import load_skill
        try:
            skill = load_skill(result)
            skill_registry._skills[skill.name] = skill
            logger.info("Skill '%s' loaded into active registry", skill.name)
        except Exception as e:
            logger.warning("Skill installed but failed to load: %s", e)

    return {
        "status": "installed",
        "skill_name": body.skill_name,
        "path": str(result),
    }


class PublishRequest(BaseModel):
    skill_name: str | None = None  # None = publish all installed


@router.post("/catalog/publish")
async def publish_skills(body: PublishRequest, request: Request):
    """Publish installed skills to BotVibes marketplace."""
    catalog = _get_catalog(request)
    bridge = getattr(request.app.state, "acp_bridge", None)
    if not bridge or not bridge.connected:
        raise HTTPException(503, "BotVibes not connected")

    skill_registry = getattr(request.app.state, "skill_registry", None)
    if not skill_registry:
        raise HTTPException(503, "Skill registry not available")

    published = []
    errors = []
    skills_to_publish = (
        [skill_registry.get(body.skill_name)] if body.skill_name
        else skill_registry.list_skills()
    )

    for skill in skills_to_publish:
        if not skill:
            continue
        listing_status = "created"
        try:
            # Create marketplace listing via BotVibes API
            listing_data = {
                "capability_id": skill.capability_id,
                "name": f"Skill: {skill.name}",
                "description": skill.description[:500],
                "pricing_model": "per_call",
                "unit_price": 0.5 if skill.name != "mcp-builder" else 1.0,
                "is_active": True,
            }
            await bridge.request("POST", "/marketplace/listings", json=listing_data)
        except Exception as e:
            err_str = str(e)
            if "409" in err_str or "already" in err_str.lower() or "duplicate" in err_str.lower():
                listing_status = "already_exists"
                logger.debug("Listing for '%s' already exists on BotVibes", skill.name)
            else:
                listing_status = "error"
                errors.append({"skill": skill.name, "error": err_str})
                logger.warning("Listing create failed for '%s': %s", skill.name, e)
        # Mark published if listing was created or already exists
        if listing_status != "error":
            await catalog.mark_published(skill.name)
            published.append(skill.name)
            logger.info("Published skill '%s' to BotVibes (%s)", skill.name, listing_status)

    result: dict = {"published": published, "count": len(published)}
    if errors:
        result["errors"] = errors
    return result


@router.get("/catalog/royalties")
async def royalty_summary(request: Request, skill_name: str | None = None):
    """Get royalty transaction summary."""
    catalog = _get_catalog(request)
    summary = await catalog.royalty_summary(skill_name)
    return summary


class RoyaltyRecord(BaseModel):
    skill_name: str
    amount: float
    provider_id: str = ""
    consumer_id: str = ""
    transaction_type: str = "invoke"


@router.post("/catalog/royalties")
async def record_royalty(body: RoyaltyRecord, request: Request):
    """Record a royalty transaction for a licensed skill."""
    catalog = _get_catalog(request)
    entry = await catalog.get(body.skill_name)
    if not entry:
        raise HTTPException(404, f"Skill '{body.skill_name}' not in catalog")
    txn_id = await catalog.record_royalty(
        skill_name=body.skill_name,
        amount=body.amount,
        provider_id=body.provider_id,
        consumer_id=body.consumer_id,
        transaction_type=body.transaction_type,
    )
    return {"transaction_id": txn_id, "skill_name": body.skill_name, "amount": body.amount}


@router.get("/catalog/provider-dashboard")
async def provider_dashboard(request: Request):
    """Provider dashboard: catalog stats, royalties, and network status."""
    catalog = _get_catalog(request)
    counts = await catalog.count()
    royalties = await catalog.royalty_summary()
    categories = await catalog.categories()
    last_sync = await catalog.get_meta("last_github_sync")

    bridge = getattr(request.app.state, "acp_bridge", None)
    return {
        "catalog": counts,
        "categories": categories,
        "royalties": royalties,
        "last_sync": last_sync,
        "marketplace_connected": bridge.connected if bridge else False,
    }


@router.delete("/catalog/{skill_name}")
async def delete_catalog_entry(skill_name: str, request: Request):
    """Remove a skill from the catalog and unpublish from BotVibes."""
    catalog = _get_catalog(request)
    entry = await catalog.get(skill_name)
    if not entry:
        raise HTTPException(404, f"Skill '{skill_name}' not in catalog")

    # Unpublish from BotVibes if published
    unpublished = False
    if entry.published:
        bridge = getattr(request.app.state, "acp_bridge", None)
        if bridge and bridge.connected:
            capability_id = f"com.multihead.skill.{skill_name}"
            try:
                # Find and deactivate the listing on BotVibes
                dashboard = await bridge.request("GET", "/marketplace/listings/mine")
                listings = dashboard if isinstance(dashboard, list) else dashboard.get("listings", [])
                for listing in listings:
                    if listing.get("capability_id") == capability_id:
                        listing_id = listing["listing_id"]
                        await bridge.request("DELETE", f"/marketplace/listings/{listing_id}")
                        unpublished = True
                        logger.info("Unpublished listing '%s' from BotVibes", listing_id)
                        break
            except Exception as e:
                logger.warning("Failed to unpublish '%s' from BotVibes: %s", skill_name, e)

    await catalog.delete(skill_name)
    return {"status": "deleted", "skill_name": skill_name, "unpublished": unpublished}


# NOTE: This wildcard route MUST be last — it catches any /catalog/<name>
@router.get("/catalog/{skill_name}")
async def get_catalog_entry(skill_name: str, request: Request):
    """Get a specific skill from the catalog."""
    catalog = _get_catalog(request)
    entry = await catalog.get(skill_name)
    if not entry:
        raise HTTPException(404, f"Skill '{skill_name}' not in catalog")
    return entry.to_dict()
