"""WebSocket routes for live streaming."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()


WS_MAX_IDLE_SECONDS = 300  # 5 min idle timeout
WS_MAX_DURATION_SECONDS = 1800  # 30 min absolute timeout


@router.websocket("/runs/{run_id}/events")
async def ws_run_events(websocket: WebSocket, run_id: str):
    """Stream run events in real-time."""
    await websocket.accept()

    event_store = websocket.app.state.event_store
    last_count = 0
    import time
    start_time = time.monotonic()
    last_activity = start_time

    try:
        while True:
            now = time.monotonic()
            # Absolute timeout
            if now - start_time > WS_MAX_DURATION_SECONDS:
                await websocket.send_json({
                    "type": "stream_end", "reason": "max_duration_exceeded",
                })
                logger.info("WebSocket for run %s closed: max duration exceeded", run_id)
                break
            # Idle timeout
            if now - last_activity > WS_MAX_IDLE_SECONDS:
                await websocket.send_json({
                    "type": "stream_end", "reason": "idle_timeout",
                })
                logger.info("WebSocket for run %s closed: idle timeout", run_id)
                break

            events = event_store.read_events(run_id)
            current_count = len(events)

            if current_count > last_count:
                new_events = events[last_count:]
                for event in new_events:
                    await websocket.send_json(event.model_dump(mode="json"))
                last_count = current_count
                last_activity = time.monotonic()

                # Check if run is finished
                if events and events[-1].kind.value in ("run_done", "run_failed", "run_cancelled"):
                    await websocket.send_json({"type": "stream_end", "reason": "run_complete"})
                    break

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        logger.debug("Client disconnected from run events: %s", run_id)
    except Exception as e:
        logger.exception("Error in run events WebSocket for %s", run_id)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


@router.websocket("/chat/{session_id}")
async def ws_chat(websocket: WebSocket, session_id: str):
    """Token-streaming chat via WebSocket."""
    await websocket.accept()

    agentic_core = websocket.app.state.agentic_core
    sessions = websocket.app.state.session_manager

    # Verify session exists or create one
    session = sessions.get_session(session_id)
    if session is None:
        session = sessions.create_session()
        session_id = session.session_id
        await websocket.send_json({"type": "session_created", "session_id": session_id})

    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data) if data.startswith("{") else {"message": data}
            except json.JSONDecodeError:
                message = {"message": data}
            user_text = message.get("message", "")
            stream = message.get("stream", False)

            if not user_text:
                continue

            if stream:
                # Token streaming mode
                await _stream_response(websocket, agentic_core, session_id, user_text)
            else:
                # Full response mode
                response = await agentic_core.chat(session_id, user_text)
                await websocket.send_json({
                    "type": "response",
                    "session_id": session_id,
                    "content": response,
                })

    except WebSocketDisconnect:
        logger.debug("Client disconnected from chat: %s", session_id)
    except Exception as e:
        logger.exception("Error in chat WebSocket for %s", session_id)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


async def _stream_response(
    websocket: WebSocket,
    agentic_core: Any,
    session_id: str,
    user_text: str,
) -> None:
    """Stream tokens from the core head to the WebSocket client."""
    # Add user message to session
    agentic_core.sessions.add_message(session_id, "user", user_text)

    # Ensure core is available and get the adapter
    await agentic_core.vram.ensure_core_available()

    # Assemble context
    tool_descriptions = ", ".join(t.name for t in agentic_core.tools.list_tools())
    from multihead.agentic_core import SYSTEM_PROMPT
    system = SYSTEM_PROMPT.format(tools=tool_descriptions)
    messages = agentic_core.sessions.assemble_context(session_id, system_prompt=system)
    prompt = agentic_core._format_messages(messages)

    # Stream tokens
    full_response = []
    await websocket.send_json({"type": "stream_start", "session_id": session_id})

    async for chunk in agentic_core.heads.generate_stream(agentic_core.core_head_id, prompt):
        full_response.append(chunk)
        await websocket.send_json({
            "type": "stream_token",
            "session_id": session_id,
            "token": chunk,
        })

    response_text = "".join(full_response)
    agentic_core.sessions.add_message(session_id, "assistant", response_text)

    await websocket.send_json({
        "type": "stream_end",
        "session_id": session_id,
        "content": response_text,
    })
