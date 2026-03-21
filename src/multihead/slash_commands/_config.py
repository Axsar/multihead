"""Config command handlers for SlashCommandHandler.

/config show | /config set <key> <value> | /config interactive
"""

from __future__ import annotations

import io
from typing import Any

from rich.console import Console as RichConsole
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.tree import Tree


class ConfigMixin:
    """Mixin providing /config command handlers."""

    # Section definitions for interactive config
    _CONFIG_SECTIONS: list[tuple[str, str, list[tuple[str, str]]]] = [
        ("General", "web tools, thinking, VRAM", [
            ("web_tools_enabled", "Web tools enabled"),
            ("strip_thinking", "Strip <think> tags"),
            ("vram_core_mode", "VRAM core mode"),
        ]),
        ("Generation", "temperature, tokens, top_p", [
            ("generation.temperature", "Temperature"),
            ("generation.max_tokens", "Max tokens"),
            ("generation.top_p", "Top P"),
        ]),
        ("Pipeline", "decompose, RAG, conversation, events", [
            ("pipeline.enabled", "Pipeline enabled"),
            ("pipeline.auto_decompose", "Auto decompose complex tasks"),
            ("pipeline.auto_record", "Auto record facts to knowledge.db"),
            ("pipeline.knowledge_rag", "Knowledge RAG injection"),
            ("pipeline.vlm_auto_route", "Auto route images to VLM"),
            ("pipeline.decompose_threshold", "Decompose confidence threshold"),
            ("pipeline.decompose_head", "Decompose head (empty=auto)"),
            ("pipeline.prompt_color", "Prompt color"),
            ("pipeline.conversation.enabled", "Conversation context enabled"),
            ("pipeline.conversation.recent_count", "Recent messages in context"),
            ("pipeline.conversation.summary_interval", "Summary rebuild interval"),
            ("pipeline.conversation.max_summary_chars", "Max summary chars"),
            ("pipeline.conversation.max_recent_chars", "Max recent chars"),
            ("pipeline.event_watcher.enabled", "Event watcher enabled"),
            ("pipeline.event_watcher.poll_interval", "Event poll interval (s)"),
            ("pipeline.event_watcher.auto_handle", "Auto handle events"),
            ("pipeline.event_watcher.watch_acp", "Watch BotVibes/ACP"),
            ("pipeline.event_watcher.watch_knowledge", "Watch knowledge.db"),
        ]),
        ("Services", "responder, worker, night shift, marketplace", [
            ("services.auto_responder", "Auto responder"),
            ("services.worker_daemon", "Worker daemon"),
            ("services.serve", "API server"),
            ("services.night_shift", "Night shift"),
            ("services.acp_auto_execute", "ACP auto execute"),
            ("services.responder_interval", "Responder interval (s)"),
            ("services.responder_strategy", "Responder strategy"),
            ("services.worker_mode", "Worker mode"),
            ("services.night_shift_interval", "Night shift interval (s)"),
            ("services.night_shift_head", "Night shift head (empty=default)"),
            ("services.night_shift_concurrency", "Night shift concurrency"),
            ("services.cloud_marketplace", "Cloud marketplace"),
            ("services.cloud_rfq_interval", "RFQ scan interval (s)"),
            ("services.cloud_contract_interval", "Contract check interval (s)"),
            ("services.cloud_auto_quote", "Auto quote RFQs"),
            ("services.cloud_auto_deliver", "Auto deliver contracts"),
            ("services.cloud_full_pipeline", "Full solve pipeline for contracts"),
            ("services.cloud_pipeline_complexity_threshold", "Pipeline complexity threshold"),
            ("services.cloud_pipeline_max_steps", "Pipeline max steps"),
            ("services.cloud_pipeline_timeout", "Pipeline timeout (s)"),
            ("services.cloud_max_contracts", "Max parallel contracts"),
        ]),
    ]

    # Fields with fixed choices
    _CONFIG_CHOICES: dict[str, list[str]] = {
        "vram_core_mode": ["keep_loaded", "cpu_fallback", "unload_during_batch"],
        "services.responder_strategy": ["plan-only", "execute"],
        "services.worker_mode": ["sdk", "headless", "interactive"],
    }

    async def _handle_config(self, args: list[str]) -> str:
        if not args or args[0] == "show":
            return self._format_config()
        if args[0] in ("interactive", "i"):
            import asyncio
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._config_interactive_sync)
        if args[0] == "set" and len(args) >= 3:
            key = args[1]
            value = " ".join(args[2:])
            try:
                result = self.config.set_value(key, value)
                self.config.save(self.config_path)
                return f"Set {result}"
            except ValueError as e:
                return f"Error: {e}"
        return "Usage: /config show | /config set <key> <value> | /config interactive"

    def _format_config(self) -> str:
        """Render config as a Rich Tree with color-coded values."""
        tree = Tree("[bold]Runtime Config[/bold]")

        def _fmt(val: Any) -> str:
            if isinstance(val, bool):
                return f"[green]True[/green]" if val else f"[red]False[/red]"
            if isinstance(val, list):
                if not val:
                    return "[dim]\\[][/dim]"
                return ", ".join(str(v) for v in val)
            return str(val)

        def _add_section(parent: Tree, title: str, data: dict[str, Any]) -> None:
            branch = parent.add(f"[bold cyan]{title}[/bold cyan]")
            for key, value in data.items():
                if isinstance(value, dict):
                    _add_section(branch, key, value)
                else:
                    branch.add(f"{key}: {_fmt(value)}")

        dump = self.config.model_dump()

        # Top-level scalars
        general = tree.add("[bold cyan]General[/bold cyan]")
        for key in ("web_tools_enabled", "strip_thinking", "vram_core_mode", "disabled_tools"):
            general.add(f"{key}: {_fmt(dump[key])}")

        # Generation
        _add_section(tree, "Generation", dump["generation"])

        # Pipeline (with nested conversation + event_watcher)
        _add_section(tree, "Pipeline", dump["pipeline"])

        # Services
        _add_section(tree, "Services", dump["services"])

        # Render tree to string
        buf = io.StringIO()
        console = RichConsole(file=buf, force_terminal=True, width=100)
        console.print(tree)
        return buf.getvalue().rstrip()

    def _get_config_value(self, key: str) -> Any:
        """Get current config value by dotted key."""
        parts = key.split(".")
        obj: Any = self.config
        for part in parts:
            obj = getattr(obj, part)
        return obj

    def _config_interactive_sync(self) -> str:
        """Interactive config editor — runs in executor (blocking Rich prompts)."""
        console = RichConsole()

        console.rule("[bold]Interactive Config Editor[/bold]")
        console.print()

        # Show section menu
        for i, (name, desc, _) in enumerate(self._CONFIG_SECTIONS, 1):
            console.print(f"  [bold cyan][{i}][/bold cyan] {name:12s} [dim]({desc})[/dim]")
        console.print(f"  [bold cyan][5][/bold cyan] {'All':12s} [dim](walk through everything)[/dim]")
        console.print(f"  [bold cyan][0][/bold cyan] {'Cancel':12s}")
        console.print()

        try:
            choice = Prompt.ask(
                "Section",
                choices=[str(i) for i in range(len(self._CONFIG_SECTIONS) + 2)],
                default="0",
                console=console,
            )
        except (EOFError, KeyboardInterrupt):
            return "Config editor cancelled."

        choice_int = int(choice)
        if choice_int == 0:
            return "Config editor cancelled."

        # Determine which sections to edit
        if choice_int == 5:
            sections = list(self._CONFIG_SECTIONS)
        elif 1 <= choice_int <= len(self._CONFIG_SECTIONS):
            sections = [self._CONFIG_SECTIONS[choice_int - 1]]
        else:
            return "Invalid selection."

        changes: list[str] = []

        for section_name, _, fields in sections:
            console.print()
            console.rule(f"[bold]{section_name}[/bold]")
            console.print()

            for key, label in fields:
                current = self._get_config_value(key)

                try:
                    if key in self._CONFIG_CHOICES:
                        new_val = Prompt.ask(
                            f"  {label}",
                            choices=self._CONFIG_CHOICES[key],
                            default=str(current),
                            console=console,
                        )
                    elif isinstance(current, bool):
                        new_val = Confirm.ask(
                            f"  {label}",
                            default=current,
                            console=console,
                        )
                    elif isinstance(current, int):
                        new_val = IntPrompt.ask(
                            f"  {label}",
                            default=current,
                            console=console,
                        )
                    elif isinstance(current, float):
                        new_val = FloatPrompt.ask(
                            f"  {label}",
                            default=current,
                            console=console,
                        )
                    else:
                        new_val = Prompt.ask(
                            f"  {label}",
                            default=str(current) if current else "",
                            console=console,
                        )
                except (EOFError, KeyboardInterrupt):
                    console.print("\n[dim]Cancelled — saving changes so far.[/dim]")
                    break

                if new_val != current:
                    try:
                        self.config.set_value(key, str(new_val))
                        changes.append(f"  {key}: {current} → {new_val}")
                    except ValueError as e:
                        console.print(f"  [red]Error: {e}[/red]")

        # Save
        if changes:
            self.config.save(self.config_path)
            console.print()
            console.rule("[bold green]Changes Saved[/bold green]")
            for change in changes:
                console.print(change)
            return f"\n{len(changes)} setting(s) updated."
        else:
            return "No changes made."
