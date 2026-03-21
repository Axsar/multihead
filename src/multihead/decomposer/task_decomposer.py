"""TaskDecomposer — LLM-driven hierarchical task decomposition."""

from __future__ import annotations

import logging
from typing import Any

from multihead import session_poller  # Gap #2: Enable cross-session collaboration

from .models import DecompositionPlan, TaskNode, _trim_node
from .parsing import _extract_keywords, parse_json_array, parse_node, parse_plan
from .prompts import DECOMPOSE_PROMPT, REFINE_PROMPT

logger = logging.getLogger(__name__)


class TaskDecomposer:
    """Decomposes goals into hierarchical execution plans.

    Uses an LLM (via HeadManager) to reason about task structure,
    grounded by claims from the knowledge store.
    """

    def __init__(
        self,
        head_manager: Any,
        knowledge_store: Any | None = None,
        metrics: Any | None = None,
        session_id: str | None = None,  # Gap #2: For session poller
        project_id: str = "multihead",   # Gap #2: For session poller
    ):
        self.heads = head_manager
        self.knowledge = knowledge_store
        self._metrics = metrics
        self.session_id = session_id
        self.project_id = project_id

    async def decompose(
        self,
        goal: str,
        context: str = "",
        head_id: str | None = None,
        max_depth: int = 4,
    ) -> DecompositionPlan:
        """Decompose a goal into a hierarchical plan.

        Args:
            goal: The high-level task to decompose.
            context: Optional additional context from the user.
            head_id: Specific head to use. If None, uses the active head.
            max_depth: Maximum tree depth (default 4).

        Returns:
            DecompositionPlan with phases, steps, and substeps.
        """
        # 0. Check for pending decomposition requests from other sessions (Gap #2)
        if self.knowledge and self.session_id:
            try:
                pending_requests = session_poller.check_for_decomposition_requests(
                    self.knowledge,
                    project_id=self.project_id,
                    session_id=self.session_id,
                )
                if pending_requests:
                    # Log pending requests for visibility
                    logger.info(
                        "Found %d pending decomposition requests from other sessions",
                        len(pending_requests),
                    )
                    for req in pending_requests[:3]:  # Log first 3
                        task = session_poller.get_request_task(req)
                        requester = req.provenance.produced_by.get("id", "unknown")
                        logger.info(
                            "  - Request %s from %s: %s",
                            req.claim_id[:8],
                            requester,
                            task[:60],
                        )
                    if len(pending_requests) > 3:
                        logger.info("  ... and %d more", len(pending_requests) - 3)
            except Exception as e:
                # Don't fail decomposition if session polling fails
                logger.warning("Session poller check failed: %s", e)

        # 1. Gather knowledge context
        knowledge_context, context_keys = self._gather_context(goal)

        # 2. Build prompt
        prompt = DECOMPOSE_PROMPT.format(
            goal=goal,
            user_context=context or "(none)",
            knowledge_context=knowledge_context or "(no relevant claims found)",
        )

        # 3. Call LLM
        head = head_id or self._resolve_head()
        result = await self.heads.generate(head, prompt=prompt)
        raw = result.get("text", "") if isinstance(result, dict) else str(result)

        # 4. Parse response (with retry on JSON failures)
        max_retries = 2
        plan = None
        for attempt in range(max_retries):
            try:
                plan = parse_plan(raw, goal, context_keys)
                break
            except ValueError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"JSON parsing failed (attempt {attempt + 1}/{max_retries}): {e}")
                    # Retry with more explicit JSON-only prompt
                    retry_prompt = (
                        prompt +
                        "\n\nIMPORTANT: Respond with ONLY valid JSON. "
                        "Do not include any explanatory text before or after the JSON object."
                    )
                    result = await self.heads.generate(head, prompt=retry_prompt)
                    raw = result.get("text", "") if isinstance(result, dict) else str(result)
                else:
                    raise

        if plan is None:
            raise ValueError("Failed to decompose task after retries")

        # 5. Enforce depth limit
        self._enforce_depth(plan, max_depth)

        logger.info(
            "Decomposed '%s' -> %s, %d phases, %d leaf steps, depth %d",
            goal[:60], plan.complexity, len(plan.phases),
            plan.total_steps, plan.max_depth,
        )
        return plan

    async def refine(
        self,
        node: TaskNode,
        exploration_result: str = "",
        head_id: str | None = None,
    ) -> list[TaskNode]:
        """Refine a single node into sub-nodes.

        Useful for iterative deepening — take a complex leaf and
        break it down further after gathering more info.
        """
        prompt = REFINE_PROMPT.format(
            node_id=node.id,
            goal=node.goal,
            action_type=node.action_type or "unknown",
            files=", ".join(node.target_files) or "unknown",
            exploration_result=exploration_result or "(none)",
        )

        head = head_id or self._resolve_head()
        result = await self.heads.generate(head, prompt=prompt)
        raw = result.get("text", "") if isinstance(result, dict) else str(result)

        items = parse_json_array(raw)
        return [parse_node(item) for item in items]

    def _resolve_head(self) -> str:
        """Find a usable head — prefer already-active, then any LLM."""
        try:
            states = self.heads.get_states()
            # Prefer active head
            for hid, state in states.items():
                if state.get("state") == "active" and state.get("kind") == "llm":
                    return hid
            # Fall back to any registered LLM
            for hid, state in states.items():
                if state.get("kind") == "llm":
                    return hid
        except Exception:
            pass
        raise RuntimeError("No LLM head available for decomposition")

    def _gather_context(self, goal: str) -> tuple[str, list[str]]:
        """Query knowledge store for claims relevant to the goal."""
        if not self.knowledge:
            return "", []

        keywords = _extract_keywords(goal)
        if not keywords:
            return "", []

        claims_text: list[str] = []
        context_keys: list[str] = []

        try:
            all_claims = self.knowledge.list_claims(
                status="accepted",
                limit=100,
            )
            for claim in all_claims:
                statement_lower = claim.statement.lower()
                if any(kw in statement_lower for kw in keywords):
                    key = claim.canonical.claim_key
                    claims_text.append(f"- [{key}] {claim.statement}")
                    context_keys.append(key)
                    if len(claims_text) >= 20:
                        break
        except Exception as e:
            logger.warning("Knowledge query failed: %s", e)

        return "\n".join(claims_text), context_keys

    def _enforce_depth(self, plan: DecompositionPlan, max_depth: int) -> None:
        """Flatten nodes deeper than max_depth into leaves."""
        for phase in plan.phases:
            _trim_node(phase, current_depth=1, max_depth=max_depth)
