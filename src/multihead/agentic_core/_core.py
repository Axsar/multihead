"""Agentic Core: always-on LLM brain with structured action dispatch."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from multihead.action_types import (
    CallToolAction,
    CoreAction,
    CreateWorkOrderAction,
    MonitorWorkOrderAction,
    PauseAndAskAction,
    SayAction,
    parse_action,
)
from multihead.head_manager import HeadManager
from multihead.models import StepDef, WorkOrder
from multihead.orchestrator import Orchestrator
from multihead.session import SessionManager
from multihead.tool_registry import ToolRegistry
from multihead.vram_policy import VRAMManager

from ._constants import (
    _APPROVAL_WORDS,
    _MESH_TUTORIAL,
    _TOOL_CAPABLE_ADAPTERS,
    MAX_TOOL_LOOP_DEPTH,
    MAX_VALIDATION_RETRIES,
    SIMPLE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class AgenticCore:
    """The always-on core loop that interprets user input and dispatches actions."""

    def __init__(
        self,
        head_manager: HeadManager,
        orchestrator: Orchestrator,
        tool_registry: ToolRegistry,
        session_manager: SessionManager,
        vram_manager: VRAMManager,
        core_head_id: str = "core-llm",
        strip_thinking: bool = True,
        narrative_pipeline: Any | None = None,
        knowledge_store: Any | None = None,
        project_id: str = "multihead",
        mesh_discovery: Any | None = None,
    ) -> None:
        self.heads = head_manager
        self.orchestrator = orchestrator
        self.tools = tool_registry
        self.sessions = session_manager
        self.vram = vram_manager
        self.core_head_id = core_head_id
        self.strip_thinking = strip_thinking
        self.narrative_pipeline = narrative_pipeline
        self.knowledge_store = knowledge_store
        self.project_id = project_id
        self.mesh_discovery = mesh_discovery
        self._mesh_tutorial_shown: bool = False

    async def chat(self, session_id: str, user_message: str) -> str:
        """Process a user message and return a response."""
        # Check for pending approval before normal processing
        pending = self._get_pending_approval(session_id)
        if pending:
            return await self._handle_approval_response(session_id, user_message, pending)

        self.sessions.add_message(session_id, "user", user_message)
        response = await self._core_loop(session_id)
        self.sessions.add_message(session_id, "assistant", response)
        self._ingest_exchange(session_id, user_message, response)

        # Check for pending cross-session messages (end-of-response notification)
        if self.knowledge_store:
            from multihead import auto_responder
            pending_count = await auto_responder.check_for_pending_messages(
                self.knowledge_store,
                session_id,
                self.project_id,
            )
            notification = auto_responder.get_pending_count_notification(pending_count)
            response += notification

        # Print mesh tutorial hint on first multi-session detection
        if not self._mesh_tutorial_shown:
            self._maybe_show_mesh_tutorial()

        return response

    async def chat_with_skill(self, session_id: str, user_message: str, skill_prompt: str) -> str:
        """Process a message using a skill's instructions as the system prompt.

        This is the execution path for marketplace skill-tasks: the skill's
        SKILL.md body becomes the system prompt instead of the default one.
        """
        self.sessions.add_message(session_id, "user", user_message)

        await self.vram.ensure_core_available()
        adapter = self.heads.get_adapter(self.core_head_id)

        messages = self.sessions.assemble_context(session_id, system_prompt=skill_prompt)

        # Generate response (same path as _core_loop but with skill context)
        if hasattr(adapter, "chat"):
            response = await adapter.chat(messages)
        else:
            prompt = self._format_messages(messages)
            response = await adapter.generate(prompt)

        raw_text = response.get("text", "")
        if self.strip_thinking:
            raw_text = re.sub(r"<think>.*?</think>\s*", "", raw_text, flags=re.DOTALL)

        self.sessions.add_message(session_id, "assistant", raw_text)
        return raw_text

    def _detect_peers(self) -> list[str]:
        """Return node IDs of currently reachable mesh peers.

        Tries mDNS discovery first; falls back to presence claims in the
        knowledge store so shared-DB setups work even without zeroconf.
        """
        # mDNS path
        if self.mesh_discovery is not None:
            try:
                nodes = self.mesh_discovery.get_discovered_nodes()
                if nodes:
                    return list(nodes.keys())
            except Exception:
                pass

        # Shared-DB fallback: look for accepted presence claims from other nodes
        if self.knowledge_store is not None:
            try:
                claims = self.knowledge_store.list_claims(
                    scope_id="multihead",
                    status="accepted",
                    limit=100,
                )
                peer_ids: list[str] = []
                for c in claims:
                    key = c.canonical.claim_key if c.canonical else ""
                    if not key.startswith("mesh.presence."):
                        continue
                    node = key[len("mesh.presence."):]
                    # Skip ourselves and offline/absent nodes
                    value = (
                        c.canonical.object.value
                        if c.canonical and c.canonical.object
                        else {}
                    )
                    status = value.get("status", "") if isinstance(value, dict) else ""
                    if status in ("offline", "absent"):
                        continue
                    if node and node != getattr(self, "project_id", ""):
                        peer_ids.append(node)
                return peer_ids
            except Exception:
                pass

        return []

    def _maybe_show_mesh_tutorial(self) -> None:
        """Print a Rich tutorial panel on the first detection of mesh peers."""
        peers = self._detect_peers()
        if not peers:
            return

        self._mesh_tutorial_shown = True
        try:
            from rich.console import Console
            from rich.panel import Panel

            console = Console()
            peer_list = ", ".join(peers[:5])
            if len(peers) > 5:
                peer_list += f" (+{len(peers) - 5} more)"
            body = _MESH_TUTORIAL + f"\n[dim]Detected peers: {peer_list}[/dim]"
            console.print(Panel(body, title="[bold green]MultiHead Mesh[/bold green]",
                                border_style="green", expand=False))
        except Exception as e:
            logger.debug("Could not display mesh tutorial panel: %s", e)

    def _ingest_exchange(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        """Feed user↔assistant exchange to the narrative pipeline (if wired)."""
        if not self.narrative_pipeline:
            return
        try:
            messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
            self.narrative_pipeline.ingest_chat_messages(
                messages, session_id, source_uri=f"agentic_core://{session_id}",
            )
        except Exception as e:
            logger.debug("Narrative ingest error: %s", e)

    async def _core_loop(self, session_id: str, depth: int = 0) -> str:
        """Main core loop: assemble context, generate, parse, execute."""
        if depth >= MAX_TOOL_LOOP_DEPTH:
            return "Reached maximum tool loop depth. Please provide additional guidance."

        # Ensure core is available
        await self.vram.ensure_core_available()
        adapter = self.heads.get_adapter(self.core_head_id)

        # Choose system prompt based on adapter capability
        adapter_type = getattr(adapter, 'manifest', None)
        adapter_name = str(adapter_type.adapter) if adapter_type else ""
        logger.info("Core loop: adapter=%s, tool_capable=%s", adapter_name, adapter_name in _TOOL_CAPABLE_ADAPTERS)
        if adapter_name in _TOOL_CAPABLE_ADAPTERS:
            # Full prompt with tool descriptions
            tool_lines = []
            for t in self.tools.list_tools():
                params = ", ".join(
                    f"{k}" + (" (required)" if isinstance(v, dict) and v.get("required") else "")
                    for k, v in t.params_schema.items()
                )
                tool_lines.append(f"- {t.name}: {t.description} (params: {params})")
            tool_descriptions = "\n".join(tool_lines)
            system = SYSTEM_PROMPT.format(tools=tool_descriptions)
        else:
            # Simple prompt — inject live system state so Claude can answer directly
            heads_info = self.heads.get_states()
            heads_lines = []
            for hid, info in heads_info.items():
                heads_lines.append(
                    f"  - {hid}: {info['name']} (adapter={info['adapter']}, "
                    f"model={info['model']}, state={info['state']})"
                )
            context_parts = [
                f"## Registered Heads ({len(heads_info)} total)\n" + "\n".join(heads_lines),
                f"\n## Core Head\nYou are running as: {self.core_head_id}",
            ]
            system = SIMPLE_SYSTEM_PROMPT.format(context="\n".join(context_parts))

        messages = self.sessions.assemble_context(session_id, system_prompt=system)

        # Generate response
        action = await self._generate_with_retry(adapter, messages)

        # Execute action
        return await self._execute_action(session_id, action, depth)

    async def _generate_with_retry(self, adapter: Any, messages: list[dict[str, str]]) -> CoreAction:
        """Generate and parse action with retry on validation failure."""
        for attempt in range(MAX_VALIDATION_RETRIES):
            try:
                # Use chat() to apply proper chat template for the model
                if hasattr(adapter, "chat"):
                    response = await adapter.chat(messages)
                else:
                    prompt = self._format_messages(messages)
                    response = await adapter.generate(prompt)
                raw_text = response.get("text", "")

                # Strip <think>...</think> blocks if enabled
                if self.strip_thinking:
                    raw_text = re.sub(r"<think>.*?</think>\s*", "", raw_text, flags=re.DOTALL)

                action = parse_action(raw_text)
                return action

            except Exception as e:
                logger.warning("Generation attempt %d failed: %s", attempt + 1, e)
                if attempt == MAX_VALIDATION_RETRIES - 1:
                    return SayAction(content=f"I encountered an error: {e}")

        return SayAction(content="Failed to generate a valid response.")

    async def _execute_action(self, session_id: str, action: CoreAction, depth: int) -> str:
        """Dispatch action and return result."""
        if isinstance(action, SayAction):
            return action.content

        elif isinstance(action, CallToolAction):
            # Wire up llm.call to head_manager (special case, no approval needed)
            if action.tool == "llm.call":
                return await self._handle_llm_call(action, session_id)

            # Check approval requirement BEFORE executing
            spec = self.tools.get_spec(action.tool)
            if spec and spec.requires_approval:
                return self._request_approval(session_id, action)

            result = await self.tools.execute(action.tool, action.params)

            # Add tool result to session and loop back
            result_text = json.dumps({"tool": result.tool, "success": result.success,
                                       "output": str(result.output)[:2000], "error": result.error})
            self.sessions.add_message(session_id, "tool", result_text)
            return await self._core_loop(session_id, depth + 1)

        elif isinstance(action, CreateWorkOrderAction):
            return await self._handle_create_workorder(action)

        elif isinstance(action, MonitorWorkOrderAction):
            return await self._handle_monitor_workorder(action)

        elif isinstance(action, PauseAndAskAction):
            return self._format_question(action)

        return f"Unknown action type: {type(action).__name__}"

    # -------------------------------------------------------------------
    # Approval flow
    # -------------------------------------------------------------------

    def _request_approval(self, session_id: str, action: CallToolAction) -> str:
        """Store pending action and ask user for approval."""
        session = self.sessions.get_session(session_id)
        if session is None:
            return "Session not found."

        # Serialize the pending action into session metadata
        pending = {
            "tool": action.tool,
            "params": action.params,
        }
        # Store on the latest message's metadata (we'll add a system message)
        self.sessions.add_message(
            session_id, "assistant",
            self._format_approval_request(action),
            pending_approval=json.dumps(pending),
        )

        return self._format_approval_request(action)

    def _get_pending_approval(self, session_id: str) -> dict[str, Any] | None:
        """Check if the session has a pending approval request."""
        session = self.sessions.get_session(session_id)
        if session is None or not session.messages:
            return None

        last_msg = session.messages[-1]
        if last_msg.role != "assistant":
            return None

        pending_json = last_msg.metadata.get("pending_approval")
        if not pending_json:
            return None

        try:
            return json.loads(pending_json)
        except (json.JSONDecodeError, TypeError):
            return None

    async def _handle_approval_response(
        self, session_id: str, user_message: str, pending: dict[str, Any]
    ) -> str:
        """Handle user's response to an approval request."""
        # Clear the pending approval by adding user message
        self.sessions.add_message(session_id, "user", user_message)

        normalized = user_message.strip().lower().rstrip("!.")
        if normalized in _APPROVAL_WORDS:
            # Execute the approved tool
            tool_name = pending["tool"]
            params = pending["params"]
            result = await self.tools.execute(tool_name, params)

            result_text = json.dumps({
                "tool": result.tool, "success": result.success,
                "output": str(result.output)[:2000], "error": result.error,
            })
            self.sessions.add_message(session_id, "tool", result_text)

            # Continue the core loop to interpret the result
            response = await self._core_loop(session_id, depth=1)
            self.sessions.add_message(session_id, "assistant", response)
            return response
        else:
            response = f"Tool '{pending['tool']}' execution cancelled."
            self.sessions.add_message(session_id, "assistant", response)
            return response

    @staticmethod
    def _format_approval_request(action: CallToolAction) -> str:
        """Format an approval request message for the user."""
        params_str = json.dumps(action.params, indent=2) if action.params else "{}"
        return (
            f"Tool '{action.tool}' requires approval before execution.\n"
            f"Parameters:\n{params_str}\n\n"
            f"Reply 'yes' to approve or anything else to cancel."
        )

    # -------------------------------------------------------------------
    # Tool handlers
    # -------------------------------------------------------------------

    async def _handle_llm_call(self, action: CallToolAction, session_id: str) -> str:
        """Handle llm.call tool by routing to head_manager."""
        head_id = action.params.get("head_id", self.core_head_id)
        prompt = action.params.get("prompt", "")

        try:
            await self.vram.prepare_for_batch(head_id)
            response = await self.heads.generate(head_id, prompt)
            await self.vram.restore_after_batch()

            result_text = response.get("text", "")
            self.sessions.add_message(session_id, "tool", f"llm.call result: {result_text[:2000]}")
            return await self._core_loop(session_id, depth=1)
        except Exception as e:
            return f"LLM call failed: {e}"

    async def _handle_create_workorder(self, action: CreateWorkOrderAction) -> str:
        """Create and execute a work order."""
        try:
            steps = [StepDef(**s) for s in action.steps] if action.steps else []
        except Exception as e:
            return f"Could not create work order: invalid step format ({e}). Steps need 'name' and 'head_id' fields."
        work_order = WorkOrder(goal=action.goal, steps=steps, inputs=action.inputs)

        try:
            state = await self.orchestrator.create_run(work_order)

            await self.vram.prepare_for_batch(
                steps[0].head_id if steps else self.core_head_id
            )
            state = await self.orchestrator.execute_run(state.run_id)
            await self.vram.restore_after_batch()

            status = state.status.value
            return f"Work order completed: {state.run_id} (status: {status})"
        except Exception as e:
            return f"Work order failed: {e}"

    async def _handle_monitor_workorder(self, action: MonitorWorkOrderAction) -> str:
        """Monitor or intervene on a running work order."""
        state = self.orchestrator.get_run_state(action.run_id)
        if state is None:
            return f"Run not found: {action.run_id}"

        if action.decision == "cancel":
            state = await self.orchestrator.cancel_run(action.run_id)
            return f"Run {action.run_id} cancelled."
        else:
            return (
                f"Run {action.run_id}: status={state.status.value}, "
                f"step={state.current_step_index}"
            )

    @staticmethod
    def _format_question(action: PauseAndAskAction) -> str:
        """Format a question for the user."""
        lines = [action.question]
        if action.options:
            for i, opt in enumerate(action.options, 1):
                lines.append(f"  {i}. {opt}")
        return "\n".join(lines)

    @staticmethod
    def _format_messages(messages: list[dict[str, str]]) -> str:
        """Format chat messages into a prompt string."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"[{role}]\n{content}")
        return "\n\n".join(parts)
