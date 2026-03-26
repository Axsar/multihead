"""Display and rendering mixin — banner, response output, Rich helpers."""

from __future__ import annotations

import shutil
from io import StringIO
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .prompts import BRAIN_CLAUDE, BRAIN_DUAL, BRAIN_LOCAL


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
        # Dual brain: deep response gets a [deep] label instead of assistant:
        label = (
            "[bold magenta]\\[deep][/bold magenta]"
            if self._brain == BRAIN_DUAL
            else "[green]assistant:[/green]"
        )
        # If response looks like it has markdown, render it
        if any(marker in response for marker in ("```", "**", "##", "- ", "| ")):
            try:
                self._tui_print()
                self._tui_print(Markdown(response))
                self._tui_print()
            except Exception:
                self._tui_print(f"{label} {response}\n")
        else:
            # Plain text fallback
            self._tui_print(f"{label} {response}\n")

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

        # Knowledge — breakdown by status + domains
        claim_count = self._count_claims()
        lines.append(f"  Knowledge: {claim_count:,} claims")
        if claim_count > 0:
            try:
                import sqlite3
                conn = sqlite3.connect(str(self.ks.db_path), timeout=5.0)
                # Status breakdown
                rows = conn.execute(
                    "SELECT claim_status, COUNT(*) FROM claims GROUP BY claim_status"
                ).fetchall()
                status_map = dict(rows)
                parts = []
                corr = status_map.get("corroborated", 0)
                if corr:
                    parts.append(f"[green]{corr}[/green] corroborated")
                stale = status_map.get("stale", 0)
                if stale:
                    parts.append(f"[yellow]{stale}[/yellow] stale")
                contested = status_map.get("contested", 0)
                if contested:
                    parts.append(f"[red]{contested}[/red] contested")
                if parts:
                    lines.append(f"    {', '.join(parts)}")
                # Top domains from accepted/corroborated claim keys
                domain_rows = conn.execute(
                    "SELECT DISTINCT substr(claim_key, 1, instr(claim_key || '.', '.') - 1) "
                    "FROM claims WHERE claim_status IN ('accepted','corroborated') "
                    "AND claim_key LIKE '%.%' LIMIT 8"
                ).fetchall()
                domains = [r[0] for r in domain_rows if r[0] and len(r[0]) > 1]
                if domains:
                    lines.append(f"    Domains: {', '.join(domains[:6])}")
                conn.close()
            except Exception:
                pass  # Graceful fallback — just show total count

        # Mesh peers
        peers = self._get_peers()
        if peers:
            lines.append(f"  Mesh: {len(peers)} peer(s) online")
        else:
            lines.append("  Mesh: [dim]no peers[/dim]")

        # Brain mode
        if self._brain == BRAIN_DUAL:
            fast = getattr(self, "_fast_head_id", None) or "fast-llm"
            lines.append(
                f"  Brain: [bold yellow]Dual[/bold yellow] "
                f"([yellow]{fast}[/yellow] → [magenta]Claude SDK[/magenta])"
            )
        elif self._brain == BRAIN_CLAUDE:
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
            # Fresh install with mock brain — guide to setup
            prompts = [
                '[dim]No models configured yet. Run:[/dim]',
                '  [cyan]multihead init --auto[/cyan]  — detect hardware and enable real models',
                '',
                '[dim]Or explore with mock heads:[/dim]',
                '  /heads  ·  /help  ·  /status',
            ]
        elif claim_count < 10:
            # Empty or near-empty knowledge store
            prompts = [
                '[dim]Knowledge store is empty. Populate it:[/dim]',
                '  [cyan]multihead nightshift run --head <your-head> --batch[/cyan]',
                '',
                '[dim]Or deposit manually:[/dim]',
                '  [cyan]multihead deposit "JWT tokens expire in 24h" -k auth.jwt.expiry[/cyan]',
            ]
        else:
            # Real brain with existing knowledge — grouped prompts
            prompts = [
                '[bold]Try:[/bold]',
                '  [cyan]Explore:[/cyan]  "what does the auth system do?"  ·  "explain payments flow"',
                '  [cyan]Inspect:[/cyan]  "show known constraints"  ·  "what contradictions exist?"',
                '  [cyan]Verify:[/cyan]   "verify: tokens expire in 24h"  ·  "what might be stale?"',
            ]

        if prompts:
            self._tui_print("\n".join(prompts))
            self._tui_print()
