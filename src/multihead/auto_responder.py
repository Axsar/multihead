"""Auto-responder for cross-session collaboration.

Detects pending messages from other sessions and prompts the user to participate.
Implements peer review feedback from collaborating agents.
"""

import asyncio
import logging
from typing import Any

from .knowledge_store import KnowledgeStore

logger = logging.getLogger(__name__)

# In-memory dedup to prevent re-prompting for same request
_responded_request_ids: set[str] = set()


async def check_for_pending_messages(
    knowledge_store: KnowledgeStore,
    session_id: str,
    project_id: str,
    timeout_seconds: float = 2.0,
) -> int:
    """
    Check for pending messages (non-blocking, with hard timeout).

    Peer review feedback: Hard 2s timeout with exception swallow - never block chat.

    Args:
        knowledge_store: Knowledge base to query
        session_id: This session's ID
        project_id: Project scope for filtering
        timeout_seconds: Hard timeout (default 2s)

    Returns:
        Number of pending messages found (0 if error/timeout)
    """
    try:
        # Hard timeout - never block chat
        pending = await asyncio.wait_for(
            _async_get_pending_messages(knowledge_store, session_id, project_id),
            timeout=timeout_seconds,
        )
        return len(pending)
    except (asyncio.TimeoutError, Exception) as e:
        logger.warning("Auto-responder check failed (timeout or error): %s", e)
        return 0  # Fail silently - never block chat


async def _async_get_pending_messages(
    knowledge_store: KnowledgeStore,
    session_id: str,
    project_id: str,
) -> list:
    """Async wrapper for knowledge store query."""
    # Use new inbox method with scope filtering
    pending = await asyncio.to_thread(
        knowledge_store.get_pending_messages,
        session_id=session_id,
        scope_id=project_id,
        max_age_hours=24,  # Ignore requests older than 24h (TTL)
        limit=50,
    )

    # Filter out already-responded requests (dedup)
    pending = [p for p in pending if p.claim_id not in _responded_request_ids]

    return pending


def get_pending_count_notification(count: int) -> str:
    """
    Generate end-of-response notification for pending messages.

    Peer review feedback: Flip notification to END of response + /collab command.
    This prevents interrupting user's question before it's processed.

    Args:
        count: Number of pending messages

    Returns:
        Notification string to append to response, or empty if no messages
    """
    if count == 0:
        return ""

    if count == 1:
        return "\n\n[collab] 1 request pending — type /collab to review"
    else:
        return f"\n\n[collab] {count} requests pending — type /collab to review"


async def handle_collab_command(
    knowledge_store: KnowledgeStore,
    session_id: str,
    project_id: str,
    head_manager: Any,
) -> str:
    """
    Handle /collab command - review and respond to pending messages.

    Peer review feedback: Use explicit /collab command instead of inline prompting.
    This gives user agency over when to engage with collaboration requests.

    Args:
        knowledge_store: Knowledge base
        session_id: This session's ID
        project_id: Project scope
        head_manager: HeadManager for decomposition

    Returns:
        Response text with pending messages and prompt for action
    """
    pending = await _async_get_pending_messages(knowledge_store, session_id, project_id)

    if not pending:
        return "No pending collaboration requests."

    # Batch prompt - show all requests in one interaction
    output = f"[Cross-Session Collaboration] {len(pending)} pending request(s):\n\n"

    for i, request in enumerate(pending[:10], 1):  # Limit to 10
        requester = request.provenance.produced_by.get("id", "unknown")
        task = request.statement[:150]
        output += f"{i}. FROM: {requester}\n"
        output += f"   Task: {task}...\n"
        output += f"   ID: {request.claim_id[:16]}...\n\n"

    if len(pending) > 10:
        output += f"... and {len(pending) - 10} more\n\n"

    output += "To respond: /collab-respond <number>\n"
    output += "To ignore: /collab-ignore <number>\n"
    output += "To ignore all: /collab-ignore-all"

    return output


async def respond_to_request(
    request_id: str,
    knowledge_store: KnowledgeStore,
    head_manager: Any,
    session_id: str,
    project_id: str,
) -> str:
    """
    Decompose a task and submit response to knowledge base.

    Args:
        request_id: The claim_id of the decomposition request
        knowledge_store: Knowledge base to submit response
        head_manager: HeadManager for decomposition
        session_id: This session's ID
        project_id: Project scope

    Returns:
        Status message
    """
    from .decomposer import TaskDecomposer
    from . import session_poller

    # Get the request claim
    pending = knowledge_store.get_pending_messages(session_id, project_id)
    request = next((p for p in pending if p.claim_id.startswith(request_id)), None)

    if not request:
        return f"Request {request_id} not found or already responded to."

    task = session_poller.get_request_task(request)
    logger.info("Decomposing task for request %s: %s", request.claim_id[:8], task[:60])

    # Create decomposer with session context
    decomposer = TaskDecomposer(
        head_manager,
        knowledge_store=knowledge_store,
        session_id=session_id,
        project_id=project_id,
    )

    # Decompose the task
    plan = await decomposer.decompose(task)

    # Submit response to knowledge base
    response_id = session_poller.submit_decomposition_proposal(
        knowledge_store=knowledge_store,
        request_id=request.claim_id,
        plan=plan,
        session_id=session_id,
        project_id=project_id,
    )

    # Mark as responded (dedup)
    _responded_request_ids.add(request.claim_id)

    logger.info("Submitted response %s for request %s", response_id[:8], request.claim_id[:8])
    return f"✓ Response submitted for request {request_id[:16]}"


def ignore_request(request_id: str) -> str:
    """Mark a request as ignored (add to dedup set).

    Args:
        request_id: The claim_id prefix to ignore

    Returns:
        Status message
    """
    _responded_request_ids.add(request_id)
    return f"✓ Request {request_id[:16]} ignored"
