"""Session poller - Check knowledge base for decomposition requests.

This module provides functions for Claude sessions to:
1. Poll knowledge base for unanswered decomposition requests
2. Prompt user to participate in consensus voting
3. Submit decomposition proposals as responses

Called on every user message to enable real-time collaboration.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from .knowledge_models import (
    Claim,
    ClaimCanonical,
    ClaimScope,
    ClaimStatus,
    ClaimType,
    EntityRef,
    Provenance,
    ScopeType,
    ValueObject,
)
from .knowledge_store import KnowledgeStore

logger = logging.getLogger(__name__)


def check_for_decomposition_requests(
    knowledge_store: KnowledgeStore,
    project_id: str = "multihead",
    session_id: str | None = None,
) -> list[Claim]:
    """Check knowledge base for unanswered decomposition requests.

    Called on every user message to see if other sessions need help
    with consensus voting.

    Args:
        knowledge_store: Knowledge base to query
        project_id: Project scope to search
        session_id: This session's ID (to avoid responding to own requests)

    Returns:
        List of unanswered decomposition request claims
    """
    # Query for unanswered QUESTION claims
    # Note: Check both proposed and accepted status to support both
    # session_poller protocol ("DECOMP_REQUEST:", proposed) and
    # solve.py protocol ("TASK DECOMPOSITION REQUEST", accepted)
    all_questions = knowledge_store.list_claims(
        claim_type="question", scope_id=project_id, limit=50
    )

    # Filter for decomposition requests (support both protocols)
    decomp_requests = [
        claim
        for claim in all_questions
        if (claim.statement.startswith("DECOMP_REQUEST:") or
            claim.statement.startswith("TASK DECOMPOSITION REQUEST"))
        and claim.claim_status in (ClaimStatus.PROPOSED, ClaimStatus.ACCEPTED)
    ]

    # Filter out requests from this session (if session_id provided)
    if session_id:
        decomp_requests = [
            claim
            for claim in decomp_requests
            if claim.provenance.produced_by.get("id") != session_id
        ]

    # Filter out requests we've already responded to
    # (Check if we have a PLAN claim with this request in related_claim_ids)
    our_responses = knowledge_store.list_claims(
        claim_type="plan", status="proposed", scope_id=project_id, limit=100
    )

    if session_id:
        our_responses = [
            r for r in our_responses if r.provenance.produced_by.get("id") == session_id
        ]

    responded_request_ids = set()
    for response in our_responses:
        responded_request_ids.update(response.related_claim_ids)

    unanswered_requests = [
        claim
        for claim in decomp_requests
        if claim.claim_id not in responded_request_ids
    ]

    count = len(unanswered_requests)
    if count and count != getattr(check_for_decomposition_requests, "_last_count", 0):
        logger.info(
            "Found %d unanswered decomposition requests", count
        )
    check_for_decomposition_requests._last_count = count  # type: ignore[attr-defined]

    return unanswered_requests


def submit_decomposition_proposal(
    knowledge_store: KnowledgeStore,
    request_id: str,
    plan: dict[str, Any],
    session_id: str = "claude-responder",
    project_id: str = "multihead",
) -> str:
    """Submit a decomposition proposal in response to a request.

    Args:
        knowledge_store: Knowledge base to write to
        request_id: The request claim ID being responded to
        plan: The decomposition plan (dict)
        session_id: This session's ID (for provenance)
        project_id: Project scope

    Returns:
        Response claim ID
    """
    # Create response claim
    plan_json = json.dumps(plan, indent=2)

    response_claim = Claim(
        claim_type=ClaimType.PLAN,
        claim_status=ClaimStatus.PROPOSED,
        scope=ClaimScope(
            scope_type=ScopeType.PROJECT,
            scope_id=project_id,
            visibility="private",
        ),
        canonical=ClaimCanonical(
            claim_key=f"decomp.proposal.{session_id}.{request_id}",
            subject=EntityRef(
                entity_type="decomposition_proposal",
                entity_id=f"{session_id}_{request_id}",
                entity_label="Decomposition Proposal",
            ),
            predicate="proposes_plan",
            object=ValueObject(value_type="json", value=plan_json),
        ),
        statement=f"DECOMP_PROPOSAL: {plan_json}",
        rationale=f"Decomposition proposal from {session_id}",
        confidence=0.8,  # Proposals are inherently uncertain
        related_claim_ids=[request_id],  # Link to request
        provenance=Provenance(produced_by={"kind": "session", "id": session_id}),
    )

    inserted = knowledge_store.insert_claim(response_claim)
    logger.info(
        "Submitted decomposition proposal %s for request %s",
        inserted.claim_id,
        request_id,
    )

    return inserted.claim_id


def get_request_task(request_claim: Claim) -> str:
    """Extract task description from decomposition request claim.

    Args:
        request_claim: The request claim

    Returns:
        Task description string
    """
    # Remove "DECOMP_REQUEST: " prefix
    statement = request_claim.statement
    if statement.startswith("DECOMP_REQUEST:"):
        return statement.replace("DECOMP_REQUEST:", "").strip()
    return statement


def format_request_for_display(request_claim: Claim) -> str:
    """Format decomposition request for user prompt.

    Args:
        request_claim: The request claim

    Returns:
        Formatted string for display
    """
    task = get_request_task(request_claim)
    requester = request_claim.provenance.produced_by.get("id", "unknown")
    created_at = request_claim.provenance.created_at

    return f"""
Decomposition Request from {requester}:
  Task: {task}
  Created: {created_at}
  Request ID: {request_claim.claim_id}
"""


# ---------------------------------------------------------------------------
# Execution requests — any agent can ask the central executor to run a plan
# ---------------------------------------------------------------------------

EXEC_REQUEST_PREFIX = "EXECUTION REQUEST:"


def post_execution_request(
    knowledge_store: KnowledgeStore,
    goal: str,
    plan_claim_id: str = "",
    session_id: str = "unknown",
    project_id: str = "multihead",
    scope_id: str = "default",
) -> str:
    """Post an execution request for the central executor to pick up.

    Any agent can call this when they have a locked plan and want it executed
    via claude -p subprocesses.

    Args:
        knowledge_store: Knowledge base to write to
        goal: What to execute (task description)
        plan_claim_id: Optional claim ID of the accepted plan to execute
        session_id: Requesting agent's session ID
        project_id: Project scope
        scope_id: Scope ID for the claim

    Returns:
        Execution request claim ID
    """
    statement = (
        f"{EXEC_REQUEST_PREFIX}\n\n"
        f"FROM: {session_id}\n"
        f"GOAL: {goal}\n"
    )
    if plan_claim_id:
        statement += f"PLAN: {plan_claim_id}\n"
    statement += f"\nRequested: {datetime.now(timezone.utc).isoformat()}\n"

    related = [plan_claim_id] if plan_claim_id else []

    claim = Claim(
        claim_type=ClaimType.QUESTION,
        claim_status=ClaimStatus.ACCEPTED,
        scope=ClaimScope(
            scope_type=ScopeType.PROJECT,
            scope_id=scope_id,
            visibility="project",
        ),
        canonical=ClaimCanonical(
            claim_key=f"exec.request.{session_id}",
            subject=EntityRef(
                entity_type="execution_request",
                entity_id=f"exec_{session_id}",
                entity_label="Execution Request",
            ),
            predicate="requests_execution",
            object=ValueObject(value_type="string", value=goal[:200]),
        ),
        statement=statement,
        rationale=f"Execution request from {session_id}",
        confidence=0.9,
        related_claim_ids=related,
        provenance=Provenance(produced_by={"kind": "session", "id": session_id}),
    )

    inserted = knowledge_store.insert_claim(claim)
    logger.info("Posted execution request %s from %s", inserted.claim_id, session_id)
    return inserted.claim_id


def check_for_execution_requests(
    knowledge_store: KnowledgeStore,
    project_id: str = "multihead",
    executor_id: str | None = None,
) -> list[Claim]:
    """Check for pending execution requests.

    Args:
        knowledge_store: Knowledge base to query
        project_id: Project scope to search (checks all scopes)
        executor_id: This executor's ID (to filter out own requests)

    Returns:
        List of unhandled execution request claims
    """
    # Check across common scopes
    all_requests = []
    for scope in [project_id, "collaboration"]:
        claims = knowledge_store.list_claims(
            claim_type="question", scope_id=scope, limit=50,
        )
        all_requests.extend(claims)

    # Deduplicate by claim_id
    seen = set()
    unique = []
    for c in all_requests:
        if c.claim_id not in seen:
            seen.add(c.claim_id)
            unique.append(c)

    # Filter for execution requests
    exec_requests = [
        c for c in unique
        if c.statement.startswith(EXEC_REQUEST_PREFIX)
        and c.claim_status in (ClaimStatus.PROPOSED, ClaimStatus.ACCEPTED)
    ]

    # Filter out own requests
    if executor_id:
        exec_requests = [
            c for c in exec_requests
            if c.provenance.produced_by.get("id") != executor_id
        ]

    # Filter out already-handled requests (check for result claims referencing them)
    result_claims = knowledge_store.list_claims(
        claim_type="fact", scope_id=project_id, limit=100,
    )
    handled_ids = set()
    for r in result_claims:
        if "EXECUTION RESULT" in r.statement:
            handled_ids.update(r.related_claim_ids)

    unhandled = [c for c in exec_requests if c.claim_id not in handled_ids]

    ucount = len(unhandled)
    if ucount and ucount != getattr(check_for_execution_requests, "_last_count", 0):
        logger.info("Found %d unhandled execution requests", ucount)
    check_for_execution_requests._last_count = ucount  # type: ignore[attr-defined]

    return unhandled
