"""Capability routing and simple fulfillment handlers for contract execution."""

from __future__ import annotations

from typing import Any

from ._constants import logger
from ._fulfillment_vision import VisionFulfillmentMixin


class FulfillmentMixin(VisionFulfillmentMixin):
    """Mixin providing capability routing and fulfillment handlers.

    Inherits vision and pipeline methods from VisionFulfillmentMixin.
    """

    # Attributes defined on the main class.
    _heads: Any
    _agentic_core: Any
    _knowledge_store: Any
    _settings: Any

    def _emit(self, event_type: str, message: str) -> None: ...
    def _svc_config(self) -> Any: ...
    def _deposit_claim(self, claim_key: str, statement: str) -> None: ...

    async def _route_and_execute(
        self,
        capability_id: str,
        payload: str,
        contract_id: str,
        vault_inputs: list[dict[str, Any]] | None = None,
    ) -> tuple[str, float] | tuple[str, float, list[tuple[str, bytes, str]]]:
        """Route contract to the right executor based on capability.

        Returns (output_text, confidence) or
        (output_text, confidence, binary_outputs) when vault files are produced.
        binary_outputs is a list of (filename, data_bytes, content_type) tuples.
        """
        import hashlib

        # Enrich payload with vault input metadata when files are present
        if vault_inputs:
            file_descs = ", ".join(
                f"{vi['filename']} ({len(vi['data'])} bytes)"
                for vi in vault_inputs
            )
            payload = (
                f"{payload}\n\n[Vault inputs: {file_descs}. "
                f"{len(vault_inputs)} file(s) available for processing.]"
            )

        # --- knowledge.rag.query.v1: query knowledge.db directly ---
        if capability_id == "knowledge.rag.query.v1":
            return await self._fulfill_rag_query(payload)

        # --- ai.task.decompose.v1: use auto-decomposer ---
        if capability_id == "ai.task.decompose.v1":
            return await self._fulfill_decompose(payload)

        # --- image.describe.v1: VLM image captioning (Vault I/O) ---
        if capability_id == "image.describe.v1" and vault_inputs:
            return await self._fulfill_image_describe(
                payload, vault_inputs, contract_id,
            )

        # --- code.review.v1: AI code review ---
        if capability_id == "code.review.v1":
            return await self._fulfill_code_review(payload)

        # --- ai.tree_of_thoughts.v1: multi-path reasoning ---
        if capability_id == "ai.tree_of_thoughts.v1":
            return await self._fulfill_tree_of_thoughts(payload)

        # --- Full solve pipeline for complex/unknown tasks ---
        if self._should_use_full_pipeline(capability_id, payload):
            return await self._fulfill_via_pipeline(capability_id, payload, contract_id)

        # --- Fallback: use agentic core chat ---
        if self._agentic_core:
            from ..session import SessionManager

            sessions = getattr(self._agentic_core, "sessions", None)
            if sessions:
                session = sessions.create_session()
                output = await self._agentic_core.chat(session.session_id, payload)
            else:
                output = await self._agentic_core.chat("cloud-contract", payload)
            return output, 0.85

        return (
            f"Contract {contract_id} received but no executor for {capability_id}",
            0.3,
        )

    async def _fulfill_rag_query(self, query: str) -> tuple[str, float]:
        """Fulfill a knowledge.rag.query.v1 request from knowledge.db."""
        try:
            # Use the injected knowledge store (not a new instance)
            ks = self._knowledge_store
            if ks is None:
                from ..knowledge_store import KnowledgeStore
                from pathlib import Path
                data_dir = Path(self._settings.data_dir) if self._settings else Path.home() / ".multihead"
                ks = KnowledgeStore(data_dir / "knowledge.db")

            # Search for matching claims via FTS
            results = ks.search_claims_fts(query, limit=20)
            if not results:
                return f"No claims found matching: {query}", 0.3

            # Format response — results are (claim_key, statement, confidence) tuples
            lines = [f"# RAG Query Results ({len(results)} claims)\n"]
            for item in results:
                if isinstance(item, tuple) and len(item) >= 2:
                    stmt = str(item[1])[:500]
                else:
                    stmt = str(item)[:500]
                lines.append(f"- {stmt}\n")

            output = "\n".join(lines)
            return output, 0.90

        except Exception as e:
            logger.warning("RAG query fulfillment failed: %s", e)
            # Fall back to agentic core if available
            if self._agentic_core:
                try:
                    sessions = getattr(self._agentic_core, "sessions", None)
                    if sessions:
                        session = sessions.create_session()
                        output = await self._agentic_core.chat(session.session_id, query)
                        return output, 0.75
                except Exception as fallback_err:
                    logger.warning("RAG fallback to agentic core also failed: %s", fallback_err)
            return f"RAG query failed: {e}", 0.2

    async def _fulfill_decompose(self, task_desc: str) -> tuple[str, float]:
        """Fulfill an ai.task.decompose.v1 request."""
        if self._agentic_core:
            prompt = (
                f"Decompose this task into atomic sub-tasks with a DAG structure. "
                f"For each step provide: id, description, dependencies, "
                f"estimated_capability. Task:\n\n{task_desc}"
            )
            output = await self._agentic_core.chat("decompose", prompt)
            return output, 0.88
        return f"No executor available for decomposition: {task_desc[:200]}", 0.3

    # ------------------------------------------------------------------
    # code.review.v1 — AI code review
    # ------------------------------------------------------------------

    async def _fulfill_code_review(self, payload: str) -> tuple[str, float]:
        """Review source code for bugs, security issues, and best practices.

        Returns a structured review report.
        """
        review_prompt = (
            "You are an expert code reviewer. Analyze the following code and "
            "provide a structured review covering:\n"
            "1. **Bugs & Issues**: Logic errors, edge cases, potential crashes\n"
            "2. **Security**: Injection risks, auth issues, data exposure\n"
            "3. **Performance**: Inefficiencies, unnecessary allocations, N+1 queries\n"
            "4. **Readability**: Naming, structure, documentation gaps\n"
            "5. **Suggestions**: Concrete improvements with code examples\n\n"
            "Rate each issue as: 🔴 Critical | 🟡 Warning | 🔵 Info\n\n"
            f"Code to review:\n```\n{payload}\n```"
        )

        if self._agentic_core:
            output = await self._agentic_core.chat("code-review", review_prompt)
            return output, 0.85

        try:
            result = await self._heads.generate("llm", review_prompt)
            output = result if isinstance(result, str) else str(result)
            return output, 0.80
        except Exception as e:
            logger.warning("Code review failed: %s", e)
            return f"Code review failed: {e}", 0.2

    # ------------------------------------------------------------------
    # ai.tree_of_thoughts.v1 — multi-path reasoning
    # ------------------------------------------------------------------

    async def _fulfill_tree_of_thoughts(self, payload: str) -> tuple[str, float]:
        """Solve a problem using Tree-of-Thoughts multi-path exploration.

        Uses the local ToT implementation with BFS/DFS/Beam search strategies.
        """
        try:
            from ..tree_of_thoughts import (
                LLMThoughtGenerator,
                LLMStateEvaluator,
                SearchStrategy,
                ToTEngine,
            )

            # Parse optional strategy from payload
            strategy_enum = SearchStrategy.BEAM  # default
            for s_name, s_enum in [("bfs", SearchStrategy.BFS), ("dfs", SearchStrategy.DFS), ("beam", SearchStrategy.BEAM)]:
                if f"strategy:{s_name}" in payload.lower() or f"strategy: {s_name}" in payload.lower():
                    strategy_enum = s_enum
                    break

            # Create generate function using the head manager
            async def _generate(prompt: str, **kwargs) -> str:
                active_heads = [
                    hid for hid, st in self._heads.get_states().items()
                    if str(st) == "active"
                ]
                head_id = active_heads[0] if active_heads else list(self._heads.get_states())[0]
                resp = await self._heads.generate(head_id, prompt)
                return resp.get("text", "")

            thought_gen = LLMThoughtGenerator(generate_func=_generate)
            state_eval = LLMStateEvaluator(generate_func=_generate)

            engine = ToTEngine(
                thought_generator=thought_gen,
                state_evaluator=state_eval,
                strategy=strategy_enum,
                max_depth=4,
                max_thoughts_per_state=3,
            )

            result = await engine.solve(
                problem=payload,
                initial_state=None,
                is_goal_reached=lambda s: False,  # Explore full depth
            )

            best_score = result.get("best_score", 0.0)
            explored = result.get("explored_count", 0)
            best_state = result.get("best_state", "No solution found")

            output = (
                f"# Tree-of-Thoughts Solution\n\n"
                f"**Strategy**: {strategy_enum.value}\n"
                f"**Paths explored**: {explored}\n"
                f"**Best path score**: {best_score:.2f}\n\n"
                f"## Solution\n{best_state}\n"
            )
            return output, min(0.95, best_score)

        except Exception as e:
            logger.warning("ToT fulfillment failed, falling back to agentic core: %s", e)

            # Fallback: use agentic core with ToT-style prompt
            if self._agentic_core:
                tot_prompt = (
                    "Solve this problem using tree-of-thoughts reasoning. "
                    "Consider multiple solution paths, evaluate each, and "
                    "select the best one. Show your reasoning trace.\n\n"
                    f"Problem: {payload}"
                )
                output = await self._agentic_core.chat("tot", tot_prompt)
                return output, 0.75

            return f"Tree-of-thoughts reasoning failed: {e}", 0.3
