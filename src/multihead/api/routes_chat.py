"""API routes for chat sessions (Agentic Core)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=50_000)
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    response: str


@router.post("", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request):
    """Send a message to the Agentic Core."""
    core = request.app.state.agentic_core
    sessions = request.app.state.session_manager

    # Create or reuse session
    if body.session_id:
        session = sessions.get_session(body.session_id)
        if session is None:
            raise HTTPException(404, f"Session not found: {body.session_id}")
        sid = body.session_id
    else:
        session = sessions.create_session()
        sid = session.session_id

    response = await core.chat(sid, body.message)
    return ChatResponse(session_id=sid, response=response)


@router.get("/sessions")
async def list_sessions(request: Request) -> list[dict[str, Any]]:
    """List all chat sessions."""
    sessions = request.app.state.session_manager
    return sessions.list_sessions()


@router.get("/sessions/{session_id}")
async def get_session(
    session_id: str,
    request: Request,
    last_n: int | None = Query(default=None, description="Return only the last N messages"),
) -> dict[str, Any]:
    """Get session details and messages."""
    sessions = request.app.state.session_manager
    session = sessions.get_session(session_id)
    if session is None:
        raise HTTPException(404, f"Session not found: {session_id}")
    data = session.model_dump(mode="json")
    if last_n is not None and "messages" in data:
        data["messages"] = data["messages"][-last_n:]
    return data


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, request: Request) -> dict[str, str]:
    """Delete a chat session."""
    sessions = request.app.state.session_manager
    deleted = sessions.delete_session(session_id)
    if not deleted:
        raise HTTPException(404, f"Session not found: {session_id}")
    return {"status": "deleted", "session_id": session_id}
