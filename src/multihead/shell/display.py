"""Display and rendering mixin — banner, response output, Rich helpers."""

from __future__ import annotations

import shutil
from io import StringIO
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .prompts import BRAIN_CLAUDE, BRAIN_LOCAL


class DisplayMixin:
    """Mixin providing display/rendering methods for the Shell class.

    Expects the host class to provide:
    - self.console (Console)
    - self.ac (AgenticCore)
    - self.hm (HeadManager)
    - self.ks (KnowledgeStore)
    - self.service_manager (ServiceManager | None)
    - self._brain (str)
    - self._verbose (bool)
    - self._last_meta (dict)
    - self._app (Application | None)
    - self._output_pane (_OutputPane | None)
    - self._count_claims() -> int
    - self._get_peers() -> list[str]
    """

    # -------------------------------------------------------------------
    # Rich rendering helpers
    # -------------------------------------------------------------------

    def _get_width(self) -> int:
        """Get effective terminal width for Rich rendering."""
        if self._app and self._app.output:
            try:
                size = self._app.output.get_size()
                return max(size.columns - 2, 40)
            except Exception:
                pass
        return max(shutil.get_terminal_size().columns - 2, 40)

    def _render_rich(self, *renderables, width: int | None = None) -> str:
        """Render Rich objects (Panel, Markdown, Table, markup strings) to ANSI."""
        buf = StringIO()
        c = Console(file=buf, force_terminal=True, width=width or self._get_width())
        for r in renderables:
            if isinstance(r, str):
                c.print(r, highlight=False)
            else:
                c.print(r)
        return buf.getvalue()

    def _tui_print(self, *args: Any, **kwargs: Any) -> None:
        """Output to TUI pane when active, otherwise fall back to self.console.

        Accepts the same arguments as ``Console.print()`` (Rich renderables,
        markup strings, etc.).  When the TUI output pane exists the content
        is rendered to ANSI via a scratch Console and appended to the pane.
        When there is no pane (unit tests, non-TUI mode) the call is
        forwarded to ``self.console.print()``.
        """
        if self._output_pane is not None:
            if not args:
                self._output_pane.append_plain("", "")
            else:
                ansi = self._render_rich(*args)
                self._output_pane.append_ansi(ansi)
            # Auto-scroll: keep at bottom when following
            if self._output_pane.follow:
                self._output_pane.scroll_offset = 0
            if self._app:
                self._app.invalidate()
        else:
            self.console.print(*args, **kwargs)

    # -------------------------------------------------------------------
    # Response display
    # -------------------------------------------------------------------

    def _display_response(self, response: str) -> None:
        """Display agent response with Rich formatting."""
        if not response:
            return
        # If response looks like it has markdown, render it
        if any(marker in response for marker in ("```", "**", "##", "- ", "| ")):
            try:
                self._tui_print()
                self._tui_print(Markdown(response))
                self._tui_print()
            except Exception:
                self._tui_print(f"[green]assistant:[/green] {response}\n")
        else:
            # Plain text fallback
            self._tui_print(f"[green]assistant:[/green] {response}\n")

        # Verbose metadata footer
        if self._verbose and self._last_meta:
            m = self._last_meta
            parts = []
            if m.get("model"):
                parts.append(f"model={m['model']}")
            if m.get("cost_usd"):
                parts.append(f"cost=${m['cost_usd']:.4f}")
            if m.get("turns"):
                parts.append(f"sdk_turns={m['turns']}")
            if m.get("session_id"):
                parts.append(f"session={m['session_id'][:12]}")
            parts.append(f"conv_turn={m.get('conv_turns', '?')}")
            if m.get("has_summary"):
                parts.append("summary=yes")
            self._tui_print(f"[dim]  [{' | '.join(parts)}][/dim]")
            self._last_meta = {}

    # -------------------------------------------------------------------
    # Banner
    # -------------------------------------------------------------------

    def _print_banner(self) -> None:
        """Print Rich panel showing system status."""
        lines: list[str] = []

        # Version
        try:
            from multihead import __version__
            lines.append(f"[bold cyan]MultiHead Agent Terminal[/bold cyan] v{__version__}")
        except Exception:
            lines.append("[bold cyan]MultiHead Agent Terminal[/bold cyan]")

        lines.append("")

        # Heads
        try:
            states = self.hm.get_states()
            if states:
                head_parts = []
                for hid, info in states.items():
                    state = info.get("state", "off")
                    # Show claude-sdk as active when it's the brain
                    if hid == "claude-sdk" and self._brain == BRAIN_CLAUDE:
                        head_parts.append(f"[green]{hid}[/green] (brain)")
                    elif state == "active":
                        head_parts.append(f"[green]{hid}[/green] (active)")
                    elif state == "sleeping":
                        head_parts.append(f"[yellow]{hid}[/yellow] (sleeping)")
                    else:
                        head_parts.append(f"[dim]{hid}[/dim] (off)")
                lines.append(f"  Heads: {', '.join(head_parts)}")
            else:
                lines.append("  Heads: [dim]none loaded[/dim]")
        except Exception:
            lines.append("  Heads: [dim]unavailable[/dim]")

        # Knowledge
        claim_count = self._count_claims()
        lines.append(f"  Knowledge: {claim_count:,} claims")

        # Mesh peers
        peers = self._get_peers()
        if peers:
            lines.append(f"  Mesh: {len(peers)} peer(s) online")
        else:
            lines.append("  Mesh: [dim]no peers[/dim]")

        # Brain mode
        if self._brain == BRAIN_CLAUDE:
            lines.append("  Brain: [bold magenta]Claude SDK[/bold magenta]")
        else:
            lines.append("  Brain: [bold blue]Local GPU[/bold blue]")

        # Services
        if self.service_manager and self.service_manager.registered_names:
            lines.append(f"  {self.service_manager.status_line()}")

        # PLUR
        lines.append("")
        lines.append("[dim]PLUR: Peace, Love, Unity, Respect[/dim]")

        body = "\n".join(lines)
        self._tui_print(Panel(
            body,
            title="[bold green]multihead shell[/bold green]",
            border_style="green",
            expand=False,
            padding=(1, 2),
        ))

        # Starter prompts based on current state
        self._print_starter_prompts()

    def _print_starter_prompts(self) -> None:
        """Show contextual starter prompts based on current state."""
        is_mock = self.ac.core_head_id.startswith("mock")
        claim_count = self._count_claims()

        prompts: list[str] = []
        if is_mock and claim_count == 0:
            # Fresh install with mock brain
            prompts = [
                '[bold]Try one of these to get started:[/bold]',
                '  [cyan]1.[/cyan] "What can you do?" — see MultiHead capabilities',
                '  [cyan]2.[/cyan] /heads — list available model heads',
                '  [cyan]3.[/cyan] /help — see all slash commands',
                '',
                '[dim]Tip: run `multihead init --auto` to detect your hardware and enable real models.[/dim]',
            ]
        elif is_mock:
            # Returning user with mock brain but has knowledge
            prompts = [
                '[bold]Try one of these:[/bold]',
                '  [cyan]1.[/cyan] /knowledge — browse your knowledge base',
                '  [cyan]2.[/cyan] /heads — see available heads',
                '  [cyan]3.[/cyan] /status — system overview',
            ]
        elif claim_count == 0:
            # Real brain but empty knowledge
            prompts = [
                '[bold]Try one of these:[/bold]',
                '  [cyan]1.[/cyan] "Summarize this project" — let the brain explore',
                '  [cyan]2.[/cyan] /status — see system overview',
                '  [cyan]3.[/cyan] /help — see all commands',
            ]
        else:
            # Real brain with existing knowledge
            prompts = [
                '[bold]Try one of these:[/bold]',
                '  [cyan]1.[/cyan] "What happened since I was last here?" — catch up',
                '  [cyan]2.[/cyan] /knowledge — browse claims and events',
                '  [cyan]3.[/cyan] /status — system overview',
            ]

        if prompts:
            self._tui_print("\n".join(prompts))
            self._tui_print()
