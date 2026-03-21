"""Head and tool management command handlers.

/tools, /heads, /wake, /sleep, /swap, /model, /verbose, /brain, /responsive
"""

from __future__ import annotations

from typing import Any


class HeadsMixin:
    """Mixin providing head/tool/brain command handlers."""

    # -------------------------------------------------------------------
    # /tools
    # -------------------------------------------------------------------

    def _handle_tools(self, args: list[str]) -> str:
        if not args or args[0] == "list":
            return self._format_tools_list()
        if args[0] == "enable" and len(args) >= 2:
            return self._toggle_tool(args[1], enable=True)
        if args[0] == "disable" and len(args) >= 2:
            return self._toggle_tool(args[1], enable=False)
        return "Usage: /tools list | /tools enable <name> | /tools disable <name>"

    def _format_tools_list(self) -> str:
        all_tools = self.tools.list_all_tools()
        if not all_tools:
            return "No tools registered."
        lines = ["Tools:"]
        for spec in all_tools:
            status = "enabled" if spec.enabled else "DISABLED"
            approval = " [requires approval]" if spec.requires_approval else ""
            lines.append(f"  {spec.name}: {spec.description} ({status}{approval})")
        return "\n".join(lines)

    def _toggle_tool(self, tool_name: str, *, enable: bool) -> str:
        spec = self.tools.get_spec(tool_name)
        if spec is None:
            return f"Unknown tool: {tool_name}"
        spec.enabled = enable
        if enable:
            self.config.enable_tool(tool_name)
        else:
            self.config.disable_tool(tool_name)
        self.config.save(self.config_path)
        state = "enabled" if enable else "disabled"
        return f"Tool '{tool_name}' {state}."

    # -------------------------------------------------------------------
    # /heads
    # -------------------------------------------------------------------

    def _handle_heads(self) -> str:
        try:
            states = self._head_states_fn()
        except Exception:
            return "Could not retrieve head states."
        if not states:
            return "No heads registered."
        lines = ["Heads:"]
        for head_id, info in states.items():
            state = info.get("state", "unknown")
            name = info.get("name", head_id)
            adapter = info.get("adapter", "?")
            lines.append(f"  {head_id}: {name} ({adapter}) [{state}]")
        return "\n".join(lines)

    # -------------------------------------------------------------------
    # /wake, /sleep, /swap
    # -------------------------------------------------------------------

    async def _handle_wake(self, args: list[str]) -> str:
        if not args:
            return "Usage: /wake <head_id>"
        if not self.head_manager:
            return "Head manager not available."
        head_id = args[0]
        try:
            await self.head_manager.wake_head(head_id)
            return f"Head '{head_id}' woken up."
        except Exception as e:
            return f"Failed to wake '{head_id}': {e}"

    async def _handle_sleep(self, args: list[str]) -> str:
        if not args:
            return "Usage: /sleep <head_id>"
        if not self.head_manager:
            return "Head manager not available."
        head_id = args[0]
        try:
            await self.head_manager.sleep_head(head_id)
            return f"Head '{head_id}' put to sleep."
        except Exception as e:
            return f"Failed to sleep '{head_id}': {e}"

    async def _handle_swap(self, args: list[str]) -> str:
        if not args:
            return "Usage: /swap <head_id>"
        if not self.head_manager:
            return "Head manager not available."
        head_id = args[0]
        try:
            await self.head_manager.ensure_active(head_id)
            return f"Swapped to '{head_id}' (now active)."
        except Exception as e:
            return f"Failed to swap to '{head_id}': {e}"

    # -------------------------------------------------------------------
    # /model
    # -------------------------------------------------------------------

    # Common model aliases -> full model IDs
    _MODEL_ALIASES: dict[str, str] = {
        "opus": "claude-opus-4-6",
        "sonnet": "claude-sonnet-4-6",
        "haiku": "claude-haiku-4-5-20251001",
        "opus-4": "claude-opus-4-6",
        "sonnet-4": "claude-sonnet-4-6",
        "haiku-4": "claude-haiku-4-5-20251001",
    }

    def _handle_model(self, args: list[str]) -> str:
        """Show or switch the Claude brain model."""
        shell = self.shell
        if not shell:
            return "Shell reference not available."

        adapter = getattr(shell, "_claude_adapter", None)

        # No args: show current model
        if not args:
            if adapter:
                return f"Current model: {adapter._model}"
            return "No Claude adapter configured. Use /brain claude first."

        if not adapter:
            return "No Claude adapter configured. Use /brain claude first."

        requested = args[0].lower()
        model_id = self._MODEL_ALIASES.get(requested, requested)

        old_model = adapter._model
        adapter._model = model_id
        return f"Model switched: {old_model} -> {model_id}\nNext turn will use {model_id} (session continues)."

    # -------------------------------------------------------------------
    # /verbose
    # -------------------------------------------------------------------

    def _handle_verbose(self) -> str:
        """Toggle verbose output showing model, cost, turns after each response."""
        shell = self.shell
        if not shell:
            return "Shell reference not available."

        shell._verbose = not shell._verbose
        state = "ON" if shell._verbose else "OFF"
        return f"Verbose output: {state}"

    # -------------------------------------------------------------------
    # /brain
    # -------------------------------------------------------------------

    async def _handle_brain(self, args: list[str]) -> str:
        if not self.shell:
            return "Brain switching requires the shell. Use `multihead shell`."
        if not args:
            return f"Current brain: {self.shell.brain}\nUsage: /brain local | /brain claude"
        mode = args[0].lower()
        return await self.shell.switch_brain(mode)

    # -------------------------------------------------------------------
    # /responsive
    # -------------------------------------------------------------------

    def _handle_responsive(self) -> str:
        """Toggle responsive mode: auto-quote, auto-deliver, event auto-handle."""
        ew = self.config.pipeline.event_watcher
        svc = self.config.services
        currently_on = ew.auto_handle and svc.cloud_auto_deliver

        if currently_on:
            ew.auto_handle = False
            svc.auto_responder = False
            svc.cloud_auto_deliver = False
            self.config.save(self.config_path)
            return "Responsive mode: OFF (watching only, no auto-bid/deliver)"
        else:
            ew.auto_handle = True
            svc.auto_responder = True
            svc.cloud_marketplace = True
            svc.cloud_auto_deliver = True
            self.config.save(self.config_path)
            # Start services if not running
            svc_mgr = getattr(self, 'service_manager', None) or getattr(self.shell, 'service_manager', None)
            if svc_mgr:
                import asyncio
                for name in ("auto-responder", "cloud-marketplace"):
                    entry = svc_mgr._services.get(name)
                    if entry and entry.status != "running":
                        asyncio.create_task(svc_mgr.start(name))
            msg = "Responsive mode: ON (auto-quote, auto-deliver, event auto-handle)"
            # Show current marketplace stats if available
            if svc_mgr:
                mkt = svc_mgr.shared_data.get("marketplace_stats")
                if mkt and any(v for v in mkt.values() if v):
                    trust = mkt.get("trust_score")
                    trust_str = f", trust {trust:.2f}" if trust is not None else ""
                    msg += (
                        f"\n  Stats: {mkt.get('quotes_sent', 0)} quotes, "
                        f"{mkt.get('contracts_won', 0)} won, "
                        f"{mkt.get('contracts_done', 0)} delivered"
                        f"{trust_str}"
                    )
            return msg
