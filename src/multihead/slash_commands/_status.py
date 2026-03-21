"""Status and dashboard command handlers.

/status, /dashboard, /dashboard live
"""

from __future__ import annotations

from datetime import timedelta

from rich.columns import Columns
from rich.console import Console as RichConsole
from rich.panel import Panel
from rich.table import Table


class StatusMixin:
    """Mixin providing /status and /dashboard command handlers."""

    # -------------------------------------------------------------------
    # /status
    # -------------------------------------------------------------------

    def _handle_status(self) -> str:
        lines = ["System Status:"]

        # Heads
        try:
            states = self._head_states_fn()
            lines.append(f"  Heads ({len(states)}):")
            for hid, info in states.items():
                state = info.get("state", "unknown")
                name = info.get("name", hid)
                lines.append(f"    {hid}: {name} [{state}]")
        except Exception:
            lines.append("  Heads: unavailable")

        # Knowledge
        if self.knowledge_store:
            try:
                claims = self.knowledge_store.list_claims(
                    status="accepted", limit=100_000,
                )
                lines.append(f"  Knowledge: {len(claims):,} accepted claims")
            except Exception:
                lines.append("  Knowledge: unavailable")
        else:
            lines.append("  Knowledge: not connected")

        # Pipeline stats
        if self.pipeline and hasattr(self.pipeline, "_stats"):
            stats = self.pipeline._stats
            total = stats.get("messages_processed", 0)
            k_hits = stats.get("knowledge_hits", 0)
            complex_tasks = stats.get("complex_tasks", 0)
            lines.append(f"  Pipeline: {total} messages, {k_hits} knowledge hits, {complex_tasks} complex tasks")

        # Marketplace stats
        svc_mgr = getattr(self.shell, "service_manager", None) if self.shell else None
        if svc_mgr:
            mkt = svc_mgr.shared_data.get("marketplace_stats")
            if mkt:
                trust = mkt.get("trust_score")
                trust_str = f", trust {trust:.2f}" if trust is not None else ""
                lines.append(
                    f"  Marketplace: {mkt.get('quotes_sent', 0)} quotes, "
                    f"{mkt.get('contracts_won', 0)} won, "
                    f"{mkt.get('contracts_done', 0)} delivered"
                    f"{trust_str}"
                )

        # Mesh peers
        if self.knowledge_store:
            try:
                peers = self.knowledge_store.get_presence_peers()
                if peers:
                    lines.append(f"  Mesh: {len(peers)} peer(s)")
                    for p in peers[:5]:
                        node = getattr(p, "node_id", str(p))
                        lines.append(f"    {node}")
                else:
                    lines.append("  Mesh: no peers")
            except Exception:
                lines.append("  Mesh: unavailable")

        return "\n".join(lines)

    # -------------------------------------------------------------------
    # /dashboard
    # -------------------------------------------------------------------

    async def _handle_dashboard(self, args: list[str] | None = None) -> None:
        """Render a Rich dashboard showing full system state.

        Usage:
            /dashboard          — static snapshot
            /dashboard live     — auto-refresh every 5s (Ctrl+C to stop)
            /dashboard live 2   — auto-refresh every 2s
        """
        console = self.shell.console if self.shell else None
        if not console:
            return

        args = args or []
        if args and args[0] == "live":
            interval = float(args[1]) if len(args) > 1 else 5.0
            await self._dashboard_live(console, interval)
        else:
            self._dashboard_render(console)

    def _dashboard_render(self, console) -> None:
        """Render a static dashboard snapshot."""
        from ..resource_monitor import ResourceTracker, sparkline

        # --- Heads table ---
        heads_table = Table(title="Heads", expand=True, show_lines=False)
        heads_table.add_column("ID", style="cyan", no_wrap=True)
        heads_table.add_column("Name")
        heads_table.add_column("State", justify="center")
        try:
            states = self._head_states_fn()
            for hid, info in states.items():
                state = info.get("state", "unknown")
                name = info.get("name", hid)
                color = {"ACTIVE": "bold green", "SLEEPING": "yellow", "OFF": "dim"}.get(state, "white")
                heads_table.add_row(hid, name, f"[{color}]{state}[/{color}]")
        except Exception:
            heads_table.add_row("—", "unavailable", "[red]error[/red]")

        # --- Services table ---
        svc_table = Table(title="Services", expand=True, show_lines=False)
        svc_table.add_column("Name", style="cyan", no_wrap=True)
        svc_table.add_column("Status", justify="center")
        svc_table.add_column("Uptime")
        if self.service_manager:
            for svc in self.service_manager.status():
                status = svc["status"]
                color = {"running": "green", "failed": "red", "stopped": "dim", "stopping": "yellow"}.get(status, "white")
                uptime = ""
                if svc.get("uptime_seconds") is not None:
                    uptime = str(timedelta(seconds=int(svc["uptime_seconds"])))
                error = svc.get("error")
                status_text = f"[{color}]{status}[/{color}]"
                if error:
                    status_text += f" [red]({error[:30]})[/red]"
                svc_table.add_row(svc["name"], status_text, uptime)
        else:
            svc_table.add_row("—", "[dim]unavailable[/dim]", "")

        # --- Processes table ---
        proc_table = Table(title="Processes", expand=True, show_lines=False)
        proc_table.add_column("PID", style="cyan", justify="right")
        proc_table.add_column("Command")
        proc_table.add_column("Status", justify="center")
        if self.process_manager:
            procs = self.process_manager.list_processes()
            if procs:
                for p in procs:
                    info = p.to_dict()
                    status = info["status"]
                    color = {"running": "green", "exited": "dim"}.get(status, "white")
                    proc_table.add_row(
                        str(info["pid"]),
                        info["command"][:40],
                        f"[{color}]{status}[/{color}]",
                    )
            else:
                proc_table.add_row("—", "none", "")
        else:
            proc_table.add_row("—", "unavailable", "")

        # --- Resource sparklines ---
        tracker = getattr(self.shell, "_resource_tracker", None) if self.shell else None
        resource_lines: list[str] = []
        if tracker and isinstance(tracker, ResourceTracker) and tracker.latest:
            snap = tracker.latest
            # VRAM
            vram_spark = sparkline(tracker.series("gpu_vram_used_mb"))
            resource_lines.append(
                f"[cyan]VRAM:[/cyan]  {snap.gpu_vram_used_mb:,}/{snap.gpu_vram_total_mb:,} MB "
                f"({snap.gpu_vram_pct:.0f}%)  {vram_spark}"
            )
            # RAM
            ram_spark = sparkline(tracker.series("ram_used_mb"))
            resource_lines.append(
                f"[cyan]RAM:[/cyan]   {snap.ram_used_mb:,}/{snap.ram_total_mb:,} MB "
                f"({snap.ram_pct:.0f}%)  {ram_spark}"
            )
            # Disk
            disk_used = snap.disk_total_mb - snap.disk_free_mb
            resource_lines.append(
                f"[cyan]Disk:[/cyan]  {disk_used:,}/{snap.disk_total_mb:,} MB "
                f"({snap.disk_pct:.0f}%)"
            )
            # Process
            if snap.process_rss_mb:
                rss_spark = sparkline(tracker.series("process_rss_mb"))
                resource_lines.append(f"[cyan]RSS:[/cyan]   {snap.process_rss_mb:,} MB  {rss_spark}")
        else:
            # Take a quick sample if no tracker
            try:
                tmp_tracker = ResourceTracker()
                snap = tmp_tracker.sample()
                if snap.gpu_vram_total_mb:
                    resource_lines.append(
                        f"[cyan]VRAM:[/cyan]  {snap.gpu_vram_used_mb:,}/{snap.gpu_vram_total_mb:,} MB "
                        f"({snap.gpu_vram_pct:.0f}%)"
                    )
                if snap.ram_total_mb:
                    resource_lines.append(
                        f"[cyan]RAM:[/cyan]   {snap.ram_used_mb:,}/{snap.ram_total_mb:,} MB "
                        f"({snap.ram_pct:.0f}%)"
                    )
                disk_used = snap.disk_total_mb - snap.disk_free_mb
                if snap.disk_total_mb:
                    resource_lines.append(
                        f"[cyan]Disk:[/cyan]  {disk_used:,}/{snap.disk_total_mb:,} MB"
                    )
            except Exception:
                resource_lines.append("[dim]Resource data unavailable[/dim]")

        resource_panel = Panel("\n".join(resource_lines) if resource_lines else "[dim]No data[/dim]",
                               title="Resources", expand=True)

        # --- Marketplace stats ---
        mkt_lines: list[str] = []
        svc_mgr = getattr(self.shell, "service_manager", None) if self.shell else None
        if svc_mgr:
            mkt = svc_mgr.shared_data.get("marketplace_stats")
            if mkt:
                mkt_lines.append(f"[cyan]Quotes sent:[/cyan]     {mkt.get('quotes_sent', 0)}")
                mkt_lines.append(f"[cyan]Contracts won:[/cyan]   {mkt.get('contracts_won', 0)}")
                mkt_lines.append(f"[cyan]Delivered:[/cyan]       {mkt.get('contracts_done', 0)}")
                trust = mkt.get("trust_score")
                if trust is not None:
                    mkt_lines.append(f"[cyan]Trust score:[/cyan]    {trust:.2f}")
        if not mkt_lines:
            mkt_lines.append("[dim]No marketplace data[/dim]")
        mkt_panel = Panel("\n".join(mkt_lines), title="Marketplace", expand=True)

        # --- Knowledge + mesh summary ---
        info_lines: list[str] = []
        if self.knowledge_store:
            try:
                claims = self.knowledge_store.list_claims(status="accepted", limit=100_000)
                info_lines.append(f"[cyan]Claims:[/cyan]    {len(claims):,} accepted")
            except Exception:
                info_lines.append("[cyan]Claims:[/cyan]    [red]error[/red]")
        else:
            info_lines.append("[cyan]Claims:[/cyan]    not connected")

        # Pipeline stats
        if self.pipeline and hasattr(self.pipeline, "_stats"):
            stats = self.pipeline._stats
            info_lines.append(f"[cyan]Messages:[/cyan]  {stats.get('messages_processed', 0)}")
            info_lines.append(f"[cyan]K-hits:[/cyan]    {stats.get('knowledge_hits', 0)}")

        # Brain mode
        if self.shell:
            brain = getattr(self.shell, "_brain", "unknown")
            info_lines.append(f"[cyan]Brain:[/cyan]     {brain}")

        # Mesh peers
        if self.knowledge_store:
            try:
                peers = self.knowledge_store.get_presence_peers()
                info_lines.append(f"[cyan]Mesh:[/cyan]      {len(peers)} peer(s)")
            except Exception:
                pass

        # Responsive mode
        ew = self.config.pipeline.event_watcher
        svc_cfg = self.config.services
        responsive = ew.auto_handle and svc_cfg.cloud_auto_deliver
        info_lines.append(f"[cyan]Responsive:[/cyan] {'[green]ON[/green]' if responsive else '[dim]OFF[/dim]'}")

        info_panel = Panel("\n".join(info_lines), title="System", expand=True)

        # --- Render layout ---
        console.print()
        console.print(Panel(heads_table, border_style="blue"))
        console.print(Columns([svc_table, proc_table], equal=True, expand=True))
        console.print(Columns([resource_panel, info_panel], equal=True, expand=True))
        console.print(Columns([mkt_panel], equal=True, expand=True))
        console.print()

    async def _dashboard_live(self, console, interval: float = 5.0) -> None:
        """Auto-refreshing dashboard using Rich Live."""
        import asyncio

        # Ensure we have a resource tracker
        tracker = getattr(self.shell, "_resource_tracker", None) if self.shell else None
        if not tracker:
            from ..resource_monitor import ResourceTracker
            tracker = ResourceTracker()
            tracker.sample()  # initial sample

        console.print(f"[dim]Live dashboard (refresh every {interval}s, Ctrl+C to stop)[/dim]\n")

        try:
            while True:
                console.clear()
                # Sample resources
                if tracker:
                    tracker.sample()
                self._dashboard_render(console)
                console.print(f"[dim]Refreshing every {interval}s — Ctrl+C to stop[/dim]")
                await asyncio.sleep(interval)
        except (KeyboardInterrupt, asyncio.CancelledError):
            console.print("\n[dim]Live dashboard stopped.[/dim]")
