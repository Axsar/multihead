"""MCP tools for skills, skill catalog, and ACP/BotVibes marketplace."""

from __future__ import annotations

import json
import logging
from typing import Any

from ._core import mcp, _ctx

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Skills
# -------------------------------------------------------------------

@mcp.tool()
async def list_skills() -> str:
    """List all loaded skills and their capabilities."""
    registry = _ctx.get("skill_registry")
    if not registry:
        return json.dumps({"skills": [], "count": 0})
    skills = [
        {"name": s.name, "description": s.description, "capability_id": s.capability_id}
        for s in registry.list_skills()
    ]
    return json.dumps({"skills": skills, "count": len(skills)})


@mcp.tool()
async def invoke_skill(skill_name: str, message: str) -> str:
    """Invoke a specific skill with a message.

    Args:
        skill_name: Name of the skill (e.g. 'pdf', 'xlsx', 'mcp-builder').
        message: The message/task for the skill.
    """
    registry = _ctx.get("skill_registry")
    if not registry:
        return json.dumps({"error": "No skills loaded"})
    skill = registry.get(skill_name)
    if not skill:
        return json.dumps({"error": f"Skill '{skill_name}' not found"})
    core = _ctx["agentic_core"]
    sm = _ctx["session_manager"]
    session_id = sm.create_session()
    response = await core.chat_with_skill(session_id, message, skill.build_system_prompt())
    return json.dumps({"skill": skill_name, "response": response, "session_id": session_id})


# -------------------------------------------------------------------
# Skill Catalog
# -------------------------------------------------------------------

@mcp.tool()
async def catalog_browse(category: str | None = None, search: str | None = None, installed_only: bool = False, limit: int = 20) -> str:
    """Browse the skill catalog -- available, installed, and published skills.

    Args:
        category: Filter by category (e.g. 'development', 'document', 'data').
        search: Search skills by name, description, or tags.
        installed_only: Only show installed skills.
        limit: Maximum results (default 20).
    """
    catalog = _ctx.get("skill_catalog")
    if not catalog:
        return json.dumps({"error": "Skill catalog not initialized"})
    entries = await catalog.list_all(category=category, installed_only=installed_only, search=search, limit=limit)
    slim = [
        {
            "name": e.name,
            "capability_id": f"com.multihead.skill.{e.name}",
            "version": e.version,
            "license": e.license_type,
        }
        for e in entries
    ]
    return json.dumps({"skills": slim, "count": len(slim)})


@mcp.tool()
async def catalog_stats() -> str:
    """Get skill catalog statistics: total, installed, published, categories."""
    catalog = _ctx.get("skill_catalog")
    if not catalog:
        return json.dumps({"error": "Skill catalog not initialized"})
    counts = await catalog.count()
    categories = await catalog.categories()
    return json.dumps({"counts": counts, "categories": categories})


@mcp.tool()
async def catalog_sync() -> str:
    """Sync the skill catalog with GitHub (fetch available skills from anthropics/skills)."""
    catalog = _ctx.get("skill_catalog")
    if not catalog:
        return json.dumps({"error": "Skill catalog not initialized"})
    from ..skill_catalog import fetch_github_skills
    entries = await fetch_github_skills()
    if not entries:
        return json.dumps({"error": "Failed to fetch skills from GitHub"})
    for entry in entries:
        existing = await catalog.get(entry.name)
        if existing:
            entry.installed = existing.installed
            entry.installed_at = existing.installed_at
            entry.published = existing.published
            entry.published_at = existing.published_at
        await catalog.upsert(entry)
    return json.dumps({"synced": len(entries), "source": "github.com/anthropics/skills"})


@mcp.tool()
async def catalog_install(skill_name: str) -> str:
    """Install a skill from GitHub into the local skills directory.

    Args:
        skill_name: Name of the skill to install (e.g. 'pdf', 'xlsx').
    """
    catalog = _ctx.get("skill_catalog")
    if not catalog:
        return json.dumps({"error": "Skill catalog not initialized"})
    from ..skill_catalog import install_skill_from_github
    settings = _ctx["settings"]
    skills_dir = settings.data_dir / "skills"
    result = await install_skill_from_github(skill_name, skills_dir)
    if not result:
        return json.dumps({"error": f"Failed to install skill '{skill_name}'"})
    await catalog.mark_installed(skill_name)
    # Load into active registry
    registry = _ctx.get("skill_registry")
    if registry:
        from ..skill_loader import load_skill
        try:
            skill = load_skill(result)
            registry._skills[skill.name] = skill
        except Exception as e:
            logger.warning("Skill installed but failed to load: %s", e)
    return json.dumps({"status": "installed", "skill_name": skill_name, "path": str(result)})


# -------------------------------------------------------------------
# ACP / BotVibes
# -------------------------------------------------------------------

@mcp.tool()
async def acp_status() -> str:
    """Check BotVibes marketplace connection status."""
    bridge = _ctx.get("acp_bridge")
    registry = _ctx.get("skill_registry")
    if not bridge:
        return json.dumps({"connected": False})
    return json.dumps({
        "connected": bridge.connected,
        "acp_url": bridge.acp_url,
        "agent_id": bridge._agent_id,
        "skills_loaded": len(registry) if registry else 0,
    })


@mcp.tool()
async def acp_reconnect() -> str:
    """Reconnect to BotVibes (reloads .env credentials)."""
    import os
    from dotenv import load_dotenv
    load_dotenv(override=True)

    bridge = _ctx.get("acp_bridge")
    if not bridge:
        return json.dumps({"error": "ACP bridge not initialized"})

    await bridge.stop()
    bridge.acp_url = os.environ.get("ACP_URL")
    bridge._api_key = os.environ.get("ACP_API_KEY")
    bridge._project_id = os.environ.get("ACP_PROJECT_ID")
    bridge._agent_id = os.environ.get("ACP_AGENT_ID", "multihead-agent")
    await bridge.start()
    return json.dumps({
        "status": "reconnected" if bridge.connected else "failed",
        "connected": bridge.connected,
        "agent_id": bridge._agent_id,
    })
