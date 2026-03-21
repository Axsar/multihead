"""Claude Session Adapter - Query other Claude Code sessions via knowledge base.

This adapter enables multi-session consensus by posting decomposition requests
to the knowledge base and collecting responses from other Claude sessions.

Protocol (Option A - Prefix-based):
- Request: ClaimType.QUESTION with statement="DECOMP_REQUEST: {task}"
- Response: ClaimType.PLAN with related_claim_ids=[request_id]
- Responders poll knowledge base on every user message

Future Improvement (Option B):
- Add 'category' field to claims for explicit categorization
- More robust filtering without prefix parsing
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from ..knowledge_models import (
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
from ..knowledge_store import KnowledgeStore
from ..models import new_id
from .base import HeadAdapter

logger = logging.getLogger(__name__)


class ClaudeSessionAdapter(HeadAdapter):
    """Adapter that queries other Claude Code sessions via knowledge base.

    This enables distributed multi-session consensus where other Claude instances
    can vote on decomposition plans by responding to knowledge base requests.

    Key features:
    - Posts decomposition requests as QUESTION claims
    - Polls for PLAN responses with configurable timeout
    - Works with 0, 1, or N responses (resilient)
    - Late responses are still captured for future use
    """

    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        session_id: str = "claude-main",
        project_id: str = "multihead",
        timeout_seconds: float = 300.0,  # 5 minutes default
        min_responses: int = 1,
        poll_interval: float = 2.0,  # Check every 2 seconds
    ):
        """Initialize Claude session adapter.

        Args:
            knowledge_store: Knowledge base for posting/reading claims
            session_id: ID of this session (for provenance)
            project_id: Project scope for claims
            timeout_seconds: How long to wait for responses (0 = no timeout)
            min_responses: Minimum responses to collect before stopping
            poll_interval: How often to check for new responses (seconds)
        """
        # Note: HeadAdapter expects manifest in __init__, but ClaudeSessionAdapter
        # doesn't need it for its operation. We skip calling super().__init__()
        # since we don't have a manifest.
        self.knowledge_store = knowledge_store
        self.session_id = session_id
        self.project_id = project_id
        self.timeout_seconds = timeout_seconds
        self.min_responses = min_responses
        self.poll_interval = poll_interval

    async def load(self) -> None:
        """Load adapter (no-op for Claude session - stateless)."""
        pass

    async def unload(self) -> None:
        """Unload adapter (no-op for Claude session - stateless)."""
        pass

    async def generate(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Generate response by querying other Claude sessions.

        Posts a decomposition request to the knowledge base and waits for
        other Claude sessions to respond with their proposals.

        Args:
            prompt: The task/prompt to decompose
            **kwargs: Additional parameters (ignored for now)

        Returns:
            Dict with "text" key containing the response

        Raises:
            TimeoutError: If timeout > 0 and no responses received
        """
        # 1. Post decomposition request
        request_id = self._post_decomposition_request(prompt)
        logger.info(
            "Posted decomposition request %s, waiting for responses (timeout: %ss)",
            request_id,
            self.timeout_seconds if self.timeout_seconds > 0 else "none",
        )

        # 2. Wait for responses
        responses = await self._collect_responses(request_id)

        # 3. Handle no responses
        if not responses:
            if self.timeout_seconds > 0:
                logger.warning(
                    "No responses received for request %s within timeout", request_id
                )
                # System must work without responses - return empty
                # Consensus engine will handle this gracefully
                return {"text": "", "responses": []}
            else:
                # No timeout - this is expected, just return empty
                return {"text": "", "responses": []}

        # 4. Return responses for consensus voting
        # The consensus engine will pick the best response
        logger.info("Collected %d responses for request %s", len(responses), request_id)

        # Return first response's plan as text, but include all for consensus
        first_response_plan = self._extract_plan_from_response(responses[0])

        return {
            "text": json.dumps(first_response_plan) if first_response_plan else "",
            "responses": [self._extract_plan_from_response(r) for r in responses],
            "request_id": request_id,
        }

    def _post_decomposition_request(self, task: str) -> str:
        """Post decomposition request to knowledge base.

        Args:
            task: Task description to decompose

        Returns:
            Request claim ID
        """
        request_id = new_id("req_")

        request_claim = Claim(
            claim_type=ClaimType.QUESTION,
            claim_status=ClaimStatus.PROPOSED,  # Unanswered
            scope=ClaimScope(
                scope_type=ScopeType.PROJECT,
                scope_id=self.project_id,
                visibility="private",
            ),
            canonical=ClaimCanonical(
                claim_key=f"decomp.request.{request_id}",
                subject=EntityRef(
                    entity_type="decomposition_request",
                    entity_id=request_id,
                    entity_label="Decomposition Request",
                ),
                predicate="needs_decomposition",
                object=ValueObject(value_type="text", value=task),
            ),
            statement=f"DECOMP_REQUEST: {task}",
            rationale="Request for multi-session consensus decomposition",
            confidence=1.0,
            provenance=Provenance(
                produced_by={"kind": "session", "id": self.session_id}
            ),
        )

        inserted = self.knowledge_store.insert_claim(request_claim)
        return inserted.claim_id

    async def _collect_responses(self, request_id: str) -> list[Claim]:
        """Collect response claims from other sessions.

        Args:
            request_id: The request claim ID to find responses for

        Returns:
            List of response claims (may be empty)
        """
        responses: list[Claim] = []
        deadline = None

        if self.timeout_seconds > 0:
            deadline = datetime.now(timezone.utc) + timedelta(
                seconds=self.timeout_seconds
            )

        while True:
            # Poll for responses
            all_proposals = self.knowledge_store.list_claims(
                claim_type="plan", status="proposed", scope_id=self.project_id
            )

            # Filter for responses to this request
            new_responses = [
                claim
                for claim in all_proposals
                if request_id in claim.related_claim_ids
                and claim.claim_id not in [r.claim_id for r in responses]
            ]

            if new_responses:
                responses.extend(new_responses)
                logger.info(
                    "Received %d new responses for %s (total: %d)",
                    len(new_responses),
                    request_id,
                    len(responses),
                )

            # Check stopping conditions
            if len(responses) >= self.min_responses:
                logger.info("Collected minimum %d responses, stopping", self.min_responses)
                break

            if deadline and datetime.now(timezone.utc) >= deadline:
                logger.info("Timeout reached, collected %d responses", len(responses))
                break

            if self.timeout_seconds == 0 and len(responses) == 0:
                # No timeout and no responses yet - wait one poll cycle then return
                await asyncio.sleep(self.poll_interval)
                # Check one more time
                all_proposals = self.knowledge_store.list_claims(
                    claim_type="plan", status="proposed", scope_id=self.project_id
                )
                new_responses = [
                    claim
                    for claim in all_proposals
                    if request_id in claim.related_claim_ids
                ]
                if new_responses:
                    responses.extend(new_responses)
                break

            # Wait before next poll
            await asyncio.sleep(self.poll_interval)

        return responses

    def _extract_plan_from_response(self, response: Claim) -> dict[str, Any] | None:
        """Extract decomposition plan from response claim.

        Args:
            response: Response claim containing plan

        Returns:
            Plan dict or None if parsing fails
        """
        try:
            # Response statement should contain JSON plan
            # Format: "DECOMP_PROPOSAL: {json...}"
            statement = response.statement
            if statement.startswith("DECOMP_PROPOSAL:"):
                json_str = statement.replace("DECOMP_PROPOSAL:", "").strip()
                return json.loads(json_str)
            else:
                # Fallback: try to parse entire statement as JSON
                return json.loads(statement)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                "Failed to parse plan from response %s: %s", response.claim_id, e
            )
            return None

    async def generate_stream(self, prompt: str, **kwargs: Any):
        """Streaming not supported for multi-session consensus.

        Raises:
            NotImplementedError: Always (streaming doesn't make sense for voting)
        """
        raise NotImplementedError(
            "Streaming not supported for Claude session adapter - "
            "use generate() to collect all responses and vote"
        )

    def supports_streaming(self) -> bool:
        """Claude session adapter doesn't support streaming."""
        return False
