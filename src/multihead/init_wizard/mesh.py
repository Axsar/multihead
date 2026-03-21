"""MeshSetupWizard: interactive mesh setup for multi-session collaboration."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml


class MeshSetupWizard:
    """Interactive mesh setup: shared dir -> mDNS test -> ACP token -> confirm.

    Usage::

        wizard = MeshSetupWizard()
        result = wizard.run(config_dir=Path("config"), env_path=Path(".env"))
        # result keys: shared_dir, mdns_available, acp_configured,
        #              peers_found, config_written, env_written, aborted
    """

    STALE_CUTOFF_SECS = 90

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        config_dir: Path,
        env_path: Path | None = None,
    ) -> dict[str, Any]:
        """Run the full mesh-setup flow.

        Returns a result dict with keys:
            shared_dir, mdns_available, acp_configured, peers_found,
            config_written, env_written, aborted
        """
        from rich.console import Console

        self._console = Console()
        if env_path is None:
            env_path = Path(".env")

        c = self._console
        c.print("\n[bold cyan]MultiHead Mesh Setup Wizard[/bold cyan]")
        c.print("Configure collaboration between multiple MultiHead sessions.\n")

        # -- Step 1 / 4 : Shared Directory --
        c.rule("[dim]Step 1 / 4  \u00b7  Shared Directory[/dim]")
        current_dir = self._read_current_data_dir()
        initial_peers = self._probe_peers(current_dir) if current_dir else []
        shared_dir, peers = self._step_shared_dir(current_dir, initial_peers)

        # -- Step 2 / 4 : mDNS Discovery Test --
        c.rule("[dim]Step 2 / 4  \u00b7  mDNS Discovery Test[/dim]")
        mdns_ok = self._step_mdns_test()

        # -- Step 3 / 4 : ACP / BotVibes Connection --
        c.rule("[dim]Step 3 / 4  \u00b7  ACP / BotVibes Connection (optional)[/dim]")
        acp = self._step_acp_token(env_path)

        # -- Step 4 / 4 : Confirm --
        c.rule("[dim]Step 4 / 4  \u00b7  Confirm[/dim]")
        if not self._step_confirm(shared_dir, mdns_ok, peers, acp):
            c.print("[yellow]Aborted \u2014 no changes written.[/yellow]")
            return {"aborted": True}

        # -- Apply --
        config_written, env_written = self._apply_config(shared_dir, acp, env_path)

        c.print("\n[bold green]\u2713 Mesh configured![/bold green]")
        c.print(f"  Shared directory : {shared_dir}")
        c.print(f"  mDNS available   : {'yes' if mdns_ok else 'no (shared-DB fallback active)'}")
        if acp.get("acp_url"):
            c.print(f"  ACP URL          : {acp['acp_url']}")
        if peers:
            c.print(f"  [green]Connected to mesh with {len(peers)} active peer(s).[/green]")
        else:
            c.print("  [dim]No peers online yet \u2014 share the data dir path with collaborators.[/dim]")
            c.print("  [dim]They run: multihead init --mesh[/dim]")

        return {
            "shared_dir": shared_dir,
            "mdns_available": mdns_ok,
            "acp_configured": bool(acp.get("acp_url")),
            "peers_found": len(peers),
            "config_written": config_written,
            "env_written": env_written,
            "aborted": False,
        }

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _step_shared_dir(
        self,
        current_dir: str,
        initial_peers: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Prompt for shared data directory; re-probe the DB on change."""
        c = self._console
        peers = initial_peers

        if initial_peers:
            c.print(
                f"[green]Found {len(initial_peers)} active peer(s) in:[/green]  {current_dir}"
            )
            for p in initial_peers:
                c.print(f"  \u2022 {p['node_id']} @ {p['hostname']}:{p['port']}")
            c.print()
            keep = input(f"Keep this shared directory? [{current_dir}] (Y/n): ").strip().lower()
            if keep in ("n", "no"):
                shared_dir = self._prompt_dir(current_dir)
                peers = self._probe_peers(shared_dir)
            else:
                shared_dir = current_dir
        else:
            if current_dir:
                c.print(f"[dim]No active peers found at: {current_dir}[/dim]")
            default = current_dir or str(Path.home() / ".multihead")
            shared_dir = self._prompt_dir(default)
            if shared_dir != current_dir:
                peers = self._probe_peers(shared_dir)
            if peers:
                c.print(f"\n[green]Found {len(peers)} active peer(s) at that location![/green]")
                for p in peers:
                    c.print(f"  \u2022 {p['node_id']} @ {p['hostname']}:{p['port']}")

        return shared_dir, peers

    def _step_mdns_test(self) -> bool:
        """Try to import and instantiate zeroconf; report result with a spinner."""
        from rich.progress import Progress, SpinnerColumn, TextColumn

        c = self._console
        mdns_ok = False
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=c,
        ) as progress:
            progress.add_task("Testing mDNS (zeroconf) availability\u2026", total=None)
            try:
                from zeroconf import Zeroconf  # type: ignore[import]
                zc = Zeroconf()
                zc.close()
                mdns_ok = True
            except ImportError:
                mdns_ok = False
            except Exception:
                mdns_ok = False

        if mdns_ok:
            c.print("[green]\u2713 mDNS available[/green] \u2014 peers will be discovered automatically on LAN.")
        else:
            c.print(
                "[yellow]\u2717 mDNS unavailable[/yellow] \u2014 will use shared-DB fallback.\n"
                "  Install zeroconf for LAN auto-discovery: [dim]pip install zeroconf[/dim]"
            )
        return mdns_ok

    def _step_acp_token(self, env_path: Path) -> dict[str, str]:
        """Optionally prompt for ACP server URL and agent token, then ping."""
        c = self._console
        c.print("Connect to a BotVibes/ACP server to share tasks across sessions.")
        c.print("Press [bold]Enter[/bold] at each prompt to skip.\n")

        current_url = os.environ.get("ACP_URL", "")
        prompt_url = f"ACP server URL [{current_url or 'skip'}]: "
        raw_url = input(prompt_url).strip()
        if not raw_url:
            c.print("[dim]ACP setup skipped.[/dim]")
            return {}

        acp_url = raw_url

        current_key = (
            os.environ.get("ACP_CLAUDE_SESSION_KEY")
            or os.environ.get("ACP_SESSION_KEY", "")
        )
        masked = (
            "*" * max(0, len(current_key) - 6) + current_key[-6:]
            if len(current_key) > 6
            else "*" * len(current_key)
        )
        raw_key = input(f"ACP agent token [{masked or 'none'}]: ").strip()
        acp_key = raw_key if raw_key else current_key

        # Light connectivity test
        from rich.progress import Progress, SpinnerColumn, TextColumn

        reachable = False
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=c,
        ) as progress:
            progress.add_task(f"Pinging {acp_url}/api/v1/health \u2026", total=None)
            try:
                with urllib.request.urlopen(
                    f"{acp_url}/api/v1/health", timeout=3
                ) as resp:
                    reachable = resp.status < 500
            except Exception:
                reachable = False

        if reachable:
            c.print(f"[green]\u2713 ACP server reachable:[/green] {acp_url}")
        else:
            c.print(
                f"[yellow]\u26a0 ACP server not reachable[/yellow] at {acp_url}"
                " \u2014 credentials saved anyway."
            )

        return {"acp_url": acp_url, "acp_key": acp_key}

    def _step_confirm(
        self,
        shared_dir: str,
        mdns_ok: bool,
        peers: list[dict[str, Any]],
        acp: dict[str, str],
    ) -> bool:
        """Show configuration summary and ask for confirmation."""
        from rich.table import Table

        c = self._console
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_row("Shared data dir", shared_dir)
        table.add_row(
            "mDNS discovery",
            "enabled" if mdns_ok else "disabled (shared-DB fallback)",
        )
        table.add_row("Peers found", str(len(peers)) if peers else "none yet")
        table.add_row("ACP URL", acp.get("acp_url") or "(not configured)")
        c.print(table)
        c.print()
        answer = input("Apply this configuration? (Y/n): ").strip().lower()
        return answer not in ("n", "no")

    # ------------------------------------------------------------------
    # Apply & persist
    # ------------------------------------------------------------------

    def _apply_config(
        self,
        shared_dir: str,
        acp: dict[str, str],
        env_path: Path,
    ) -> tuple[bool, bool]:
        """Write ~/.multihead/config.yaml and mirror to .env if present.

        Returns (config_written, env_written).
        """
        home_cfg_dir = Path.home() / ".multihead"
        home_cfg_dir.mkdir(parents=True, exist_ok=True)
        config_file = home_cfg_dir / "config.yaml"

        existing: dict[str, Any] = {}
        if config_file.exists():
            try:
                existing = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
            except Exception:
                pass

        existing["MULTIHEAD_DATA_DIR"] = shared_dir
        if acp.get("acp_url"):
            existing["ACP_URL"] = acp["acp_url"]
        config_file.write_text(
            yaml.dump(existing, default_flow_style=False),
            encoding="utf-8",
        )
        config_written = True

        env_written = False
        if env_path.exists():
            self._update_env_key(env_path, "MULTIHEAD_DATA_DIR", shared_dir)
            if acp.get("acp_url"):
                self._update_env_key(env_path, "ACP_URL", acp["acp_url"])
            if acp.get("acp_key"):
                self._update_env_key(env_path, "ACP_CLAUDE_SESSION_KEY", acp["acp_key"])
            env_written = True

        return config_written, env_written

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_current_data_dir() -> str:
        """Read MULTIHEAD_DATA_DIR from ~/.multihead/config.yaml or environment."""
        config_file = Path.home() / ".multihead" / "config.yaml"
        if config_file.exists():
            try:
                data = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
                if val := data.get("MULTIHEAD_DATA_DIR"):
                    return str(val)
            except Exception:
                pass
        return os.environ.get("MULTIHEAD_DATA_DIR", "")

    def _probe_peers(self, data_dir: str) -> list[dict[str, Any]]:
        """Query knowledge.db for presence claims that are online and non-stale."""
        db_path = Path(data_dir) / "knowledge.db"
        if not db_path.exists():
            return []

        import sqlite3

        stale_cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.STALE_CUTOFF_SECS)
        try:
            conn = sqlite3.connect(str(db_path), timeout=5)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT claim_key, object_json FROM claims "
                "WHERE claim_status = 'accepted' AND claim_key LIKE 'mesh.presence.%'"
            ).fetchall()
            conn.close()
        except Exception:
            return []

        peers: list[dict[str, Any]] = []
        for row in rows:
            try:
                obj = json.loads(row["object_json"])
                value = obj.get("value", {})
                if value.get("status") != "online":
                    continue
                last_seen_str = value.get("last_seen", "")
                if last_seen_str:
                    last_seen = datetime.fromisoformat(last_seen_str)
                    if last_seen < stale_cutoff:
                        continue
                node_id = row["claim_key"].removeprefix("mesh.presence.")
                peers.append({
                    "node_id": node_id,
                    "hostname": value.get("hostname", "unknown"),
                    "port": value.get("port", 7337),
                    "last_seen": last_seen_str,
                })
            except Exception:
                continue
        return peers

    @staticmethod
    def _prompt_dir(default: str) -> str:
        """Prompt the user for a directory path, falling back to default."""
        raw = input(f"Shared data directory [{default}]: ").strip()
        return raw if raw else default

    @staticmethod
    def _update_env_key(env_path: Path, key: str, value: str) -> None:
        """Update or append a KEY=value line in an .env file."""
        lines = env_path.read_text(encoding="utf-8").splitlines()
        new_lines: list[str] = []
        updated = False
        for line in lines:
            if line.startswith(f"{key}="):
                new_lines.append(f"{key}={value}")
                updated = True
            else:
                new_lines.append(line)
        if not updated:
            new_lines.append(f"{key}={value}")
        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
