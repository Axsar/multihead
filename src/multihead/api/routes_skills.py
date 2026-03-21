"""API routes for skill management."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter()


@router.get("")
async def list_skills(request: Request) -> list[dict[str, Any]]:
    """List all loaded skills with their metadata."""
    registry = request.app.state.skill_registry
    return [s.to_dict() for s in registry.list_skills()]


@router.get("/{skill_name}")
async def get_skill(
    skill_name: str,
    request: Request,
    full: bool = Query(default=False, description="Include full instructions field"),
) -> dict[str, Any]:
    """Get a specific skill's details, optionally including full instructions."""
    registry = request.app.state.skill_registry
    skill = registry.get(skill_name)
    if skill is None:
        raise HTTPException(404, f"Skill not found: {skill_name}")
    result = skill.to_dict()
    if full:
        result["instructions"] = skill.instructions
    return result


@router.post("/{skill_name}/invoke")
async def invoke_skill(skill_name: str, request: Request) -> dict[str, Any]:
    """Invoke a skill directly with a message (for testing).

    Body: {"message": "your task description"}
    """
    registry = request.app.state.skill_registry
    skill = registry.get(skill_name)
    if skill is None:
        raise HTTPException(404, f"Skill not found: {skill_name}")

    body = await request.json()
    message = body.get("message", "")
    if not message:
        raise HTTPException(400, "message is required")

    core = request.app.state.agentic_core
    sessions = request.app.state.session_manager

    session = sessions.create_session()
    skill_prompt = skill.build_system_prompt()
    response = await core.chat_with_skill(session.session_id, message, skill_prompt)

    return {
        "skill": skill_name,
        "capability_id": skill.capability_id,
        "session_id": session.session_id,
        "response": response,
    }
