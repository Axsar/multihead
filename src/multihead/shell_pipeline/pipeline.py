"""ShellPipeline — middleware between Shell REPL and brain execution.

Automatically invokes MultiHead infrastructure on every message:
- Stage 1: Knowledge RAG (SQL-based, scope-aware, top 10)
- Stage 1b: Inbox context (pending claims/requests from knowledge.db)
- Stage 2: Intent classification (heuristic — chat vs complex task)
- Stage 3: Execution (direct brain or decompose+orchestrate)
- Stage 4: Knowledge recording (deposit key facts as claims)

Each stage is independently testable and toggled via PipelineConfig
in RuntimeConfig.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .constants import BrainFn
from .execution import detect_image_input, execute_as_task, format_run_results, route_to_vlm
from .intent import (
    _ACTION_VERBS,
    _MULTI_STEP_PATTERNS,
    classify_intent,
    has_action_verbs,
    has_multi_step_language,
    mentions_files,
)
from .knowledge_rag import (
    build_inbox_context,
    build_knowledge_context,
    extract_keywords,
    query_claims_by_keywords,
    query_claims_fts,
)
from .recording import maybe_record_knowledge, summarize_exchange

logger = logging.getLogger(__name__)


class ShellPipeline:
    """Middleware between Shell REPL and brain execution.

    Wraps the brain (Claude SDK or local AgenticCore) with automatic
    MultiHead infrastructure usage: knowledge RAG, intent routing,
    auto-decomposition, and knowledge recording.
    """

    # Class-level references for backward compatibility with tests
    # that access ShellPipeline._ACTION_VERBS etc.
    _ACTION_VERBS = _ACTION_VERBS
    _MULTI_STEP_PATTERNS = _MULTI_STEP_PATTERNS

    def __init__(
        self,
        knowledge_store: Any | None = None,
        head_manager: Any | None = None,
        router: Any | None = None,
        runtime_config: Any | None = None,
        orchestrator: Any | None = None,
        auto_decomposer: Any | None = None,
    ) -> None:
        self._ks = knowledge_store
        self._hm = head_manager
        self._router = router
        self._config = runtime_config
        self._orchestrator = orchestrator
        self._decomposer = auto_decomposer
        self._participant_id: str = ""  # Set by Shell after registration

        # Pipeline stats
        self._stats: dict[str, int] = {
            "messages_processed": 0,
            "tasks_decomposed": 0,
            "claims_recorded": 0,
            "knowledge_hits": 0,
            "vlm_routes": 0,
        }

    # -------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------

    async def process(
        self,
        user_input: str,
        brain_fn: BrainFn,
        session_id: str,
        on_status: Callable[[str], None] | None = None,
    ) -> str:
        """Run the full pipeline: RAG -> classify -> execute -> record.

        Args:
            user_input: Raw user message
            brain_fn: Async callable (session_id, user_input, knowledge_ctx) -> response
            session_id: Current session ID
            on_status: Optional callback to update status line (e.g. Rich spinner)
        """
        self._stats["messages_processed"] += 1
        _status = on_status or (lambda _s: None)

        # Check if pipeline is enabled
        if not self._pipeline_enabled():
            _status("Thinking...")
            return await brain_fn(session_id, user_input, "")

        # Stage 1: Knowledge RAG
        _status("Searching knowledge base...")
        knowledge_ctx = ""
        if self._config_enabled("knowledge_rag"):
            knowledge_ctx = self._build_knowledge_context(user_input)
            if knowledge_ctx:
                self._stats["knowledge_hits"] += 1

        # Stage 1b: Inbox context (pending claims/requests)
        _status("Checking inbox...")
        inbox_ctx = self._build_inbox_context(session_id)
        if inbox_ctx:
            knowledge_ctx = f"{knowledge_ctx}\n\n{inbox_ctx}" if knowledge_ctx else inbox_ctx

        # Stage 2: Check for VLM auto-routing (image input)
        if self._config_enabled("vlm_auto_route"):
            _status("Detecting image input...")
            image_path = self._detect_image_input(user_input)
            if image_path:
                _status("Routing to vision model...")
                vlm_response = await self._route_to_vlm(
                    user_input, image_path, knowledge_ctx, session_id, brain_fn,
                )
                if vlm_response is not None:
                    self._stats["vlm_routes"] += 1
                    response = vlm_response
                    if self._config_enabled("auto_record"):
                        _status("Recording knowledge...")
                        self._maybe_record_knowledge(user_input, response)
                    return response

        # Stage 3: Classify intent
        _status("Classifying intent...")
        intent = self._classify_intent(user_input)

        # Stage 4: Execute
        if intent == "task" and self._config_enabled("auto_decompose"):
            _status("Decomposing task...")
            response = await self._execute_as_task(
                user_input, knowledge_ctx, session_id, brain_fn,
            )
        else:
            _status("Thinking...")
            response = await brain_fn(session_id, user_input, knowledge_ctx)

        # Stage 5: Record knowledge
        if self._config_enabled("auto_record"):
            _status("Recording knowledge...")
            self._maybe_record_knowledge(user_input, response)

        return response

    # -------------------------------------------------------------------
    # Delegate to extracted modules (preserve method signatures for compat)
    # -------------------------------------------------------------------

    def _build_knowledge_context(self, user_message: str) -> str:
        return build_knowledge_context(self._ks, user_message)

    def _build_inbox_context(self, session_id: str) -> str:
        return build_inbox_context(self._ks, session_id)

    def _extract_keywords(self, text: str) -> list[str]:
        return extract_keywords(text)

    def _query_claims_fts(
        self, keywords: list[str], limit: int = 10,
    ) -> list[tuple[str, str, float]]:
        return query_claims_fts(self._ks, keywords, limit)

    def _query_claims_by_keywords(
        self, keywords: list[str], limit: int = 10,
    ) -> list[tuple[str, str, float]]:
        return query_claims_by_keywords(self._ks, keywords, limit)

    def _classify_intent(self, user_input: str) -> str:
        return classify_intent(user_input)

    def _has_action_verbs(self, text: str) -> bool:
        return has_action_verbs(text)

    def _mentions_files(self, text: str) -> bool:
        return mentions_files(text)

    def _has_multi_step_language(self, text: str) -> bool:
        return has_multi_step_language(text)

    def _detect_image_input(self, user_input: str) -> str | None:
        return detect_image_input(user_input)

    async def _route_to_vlm(
        self,
        user_input: str,
        image_path: str,
        knowledge_ctx: str,
        session_id: str,
        brain_fn: BrainFn,
    ) -> str | None:
        return await route_to_vlm(
            user_input, image_path, knowledge_ctx, session_id, brain_fn,
            self._router, self._hm,
        )

    async def _execute_as_task(
        self,
        user_input: str,
        knowledge_ctx: str,
        session_id: str,
        brain_fn: BrainFn,
    ) -> str:
        return await execute_as_task(
            user_input, knowledge_ctx, session_id, brain_fn,
            self._config, self._decomposer, self._orchestrator, self._stats,
        )

    def _format_run_results(self, run_state: Any) -> str:
        return format_run_results(run_state)

    def _maybe_record_knowledge(self, user_input: str, response: str) -> None:
        maybe_record_knowledge(
            self._ks, user_input, response, self._participant_id, self._stats,
        )

    def _summarize_exchange(self, user_input: str, response: str) -> str:
        return summarize_exchange(user_input, response)

    # -------------------------------------------------------------------
    # Config helpers
    # -------------------------------------------------------------------

    def _pipeline_enabled(self) -> bool:
        """Check if the pipeline is enabled in config."""
        if not self._config:
            return True  # Default: enabled
        pipeline = getattr(self._config, "pipeline", None)
        if pipeline is None:
            return True
        return getattr(pipeline, "enabled", True)

    def _config_enabled(self, feature: str) -> bool:
        """Check if a specific pipeline feature is enabled."""
        if not self._config:
            return True
        pipeline = getattr(self._config, "pipeline", None)
        if pipeline is None:
            return True
        return getattr(pipeline, feature, True)

    # -------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, int]:
        """Pipeline usage statistics."""
        return dict(self._stats)

    def stats_summary(self) -> str:
        """One-line summary of pipeline stats."""
        s = self._stats
        enabled = "ON" if self._pipeline_enabled() else "OFF"
        return (
            f"Pipeline: {enabled} | "
            f"{s['messages_processed']} messages | "
            f"{s['tasks_decomposed']} tasks decomposed | "
            f"{s['claims_recorded']} claims recorded | "
            f"{s['knowledge_hits']} knowledge hits"
        )
