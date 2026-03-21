"""Init command and helpers."""

from __future__ import annotations

import json
import os

import click
from rich.table import Table

from ._helpers import (
    console,
    datetime,
    timezone,
    InitWizard,
    Path,
)
from ._core import main


def _detect_peers_from_db(data_dir: str) -> list[dict]:
    """Query knowledge.db for active peer presence claims (online within 90s)."""
    import sqlite3
    from datetime import timedelta

    db_path = Path(data_dir) / "knowledge.db"
    if not db_path.exists():
        return []

    stale_cutoff = datetime.now(timezone.utc) - timedelta(seconds=90)

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

    peers = []
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


def _prompt_for_dir(default: str) -> str:
    """Prompt for a shared data directory path, falling back to default."""
    raw = input(f"Shared data directory? [{default}]: ").strip()
    return raw if raw else default


def _update_env_data_dir(env_path: Path, data_dir: str) -> None:
    """Update or append MULTIHEAD_DATA_DIR in an existing .env file."""
    lines = env_path.read_text().splitlines()
    new_lines = []
    updated = False
    for line in lines:
        if line.startswith("MULTIHEAD_DATA_DIR="):
            new_lines.append(f"MULTIHEAD_DATA_DIR={data_dir}")
            updated = True
        else:
            new_lines.append(line)
    if not updated:
        new_lines.append(f"MULTIHEAD_DATA_DIR={data_dir}")
    env_path.write_text("\n".join(new_lines) + "\n")


def _show_init_status(config_dir: Path) -> None:
    """Print current init status for a returning user (skip re-running wizard)."""
    import yaml

    console.print("[bold]MultiHead — current configuration[/bold]\n")

    # Config files
    heads_yaml = config_dir / "heads.yaml"
    env_path = Path(".env")
    sentinel = config_dir / ".multihead_initialized"

    console.print("[bold]Config files:[/bold]")
    console.print(
        f"  {'[green]✓[/green]' if heads_yaml.exists() else '[yellow]✗[/yellow]'}  "
        f"heads.yaml  ({heads_yaml})"
    )
    console.print(
        f"  {'[green]✓[/green]' if env_path.exists() else '[yellow]✗[/yellow]'}  "
        f".env  ({env_path.resolve()})"
    )
    if sentinel.exists():
        console.print(f"  [dim]Initialized: {sentinel.read_text().strip()}[/dim]")

    # Mesh / shared data dir
    mesh_config_file = Path.home() / ".multihead" / "config.yaml"
    data_dir = os.environ.get("MULTIHEAD_DATA_DIR", "")
    if not data_dir and mesh_config_file.exists():
        with open(mesh_config_file) as f:
            mc = yaml.safe_load(f) or {}
        data_dir = mc.get("MULTIHEAD_DATA_DIR", "")

    console.print("\n[bold]Mesh status:[/bold]")
    if data_dir:
        peers = _detect_peers_from_db(data_dir)
        console.print(f"  Shared data dir: {data_dir}")
        if peers:
            console.print(f"  [green]Active peers ({len(peers)}):[/green]")
            for p in peers:
                console.print(f"    • {p['node_id']} @ {p['hostname']}:{p['port']}")
        else:
            console.print("  [dim]No active peers detected (run multihead serve to appear)[/dim]")
    else:
        console.print("  [dim]Not configured — run: multihead init --mesh[/dim]")

    console.print(
        "\n[dim]To regenerate config run: multihead init --auto[/dim]"
        "\n[dim]To update mesh settings run: multihead init --mesh[/dim]"
    )


def _init_mesh_config(config_dir: Path | None = None):
    """Configure shared directory for multi-session collaboration."""
    from multihead.init_wizard import MeshSetupWizard

    if config_dir is None:
        config_dir = Path.home() / ".multihead"
    MeshSetupWizard().run(config_dir=config_dir, env_path=Path(".env"))


@main.command("init")
@click.option("--auto", is_flag=True, help="Non-interactive: auto-detect everything, generate .env and heads.yaml.")
@click.option("--mesh", is_flag=True, help="Configure shared directory for multi-session collaboration.")
@click.pass_context
def init_cmd(ctx, auto, mesh):
    """Initialize MultiHead: detect hardware and generate config."""
    settings = ctx.obj["settings"]

    # Handle --mesh: configure shared data directory for collaboration
    if mesh:
        _init_mesh_config(config_dir=settings.config_dir)
        return

    wizard = InitWizard()

    # Returning-user shortcut: skip wizard when already configured and no flags given
    if not auto and not wizard.is_first_run(settings.config_dir):
        _show_init_status(settings.config_dir)
        return

    if auto:
        # Place .env in cwd if .env.example exists (dev clone), else beside config
        env_path = Path(".env")
        if not Path(".env.example").exists() and not env_path.exists():
            env_path = settings.config_dir / ".env"
        result = wizard.run_auto(
            config_dir=settings.config_dir,
            env_path=env_path,
            templates_dir=settings.config_dir / "templates",
        )
    else:
        result = wizard.run_interactive(settings.config_dir)

    profile = result["profile"]
    suggestion = result["suggestion"]

    console.print("[bold green]MultiHead initialized![/bold green]")
    console.print(f"  Platform: {profile['platform']}")
    if profile["gpu_name"]:
        console.print(f"  GPU: {profile['gpu_name']} ({profile['gpu_vram_mb']} MB VRAM)")
    else:
        console.print("  GPU: None detected")
    console.print(f"  RAM: {profile['cpu_ram_mb']} MB")
    console.print(f"  Template: {suggestion['template']}")
    console.print(f"  Config written to: {result['config_dir']}")

    # Show adapter status in --auto mode
    if auto and "adapters" in result:
        adapters = result["adapters"]
        console.print("\n[bold]Adapter readiness:[/bold]")
        _adapter_line = lambda name, ok: console.print(
            f"  {'[green]OK[/green]' if ok else '[yellow]--[/yellow]'}  {name}"
        )
        _adapter_line("PyTorch + CUDA", adapters["cuda"])
        _adapter_line("Transformers", adapters["transformers"])
        _adapter_line("Ollama", adapters["ollama"])
        _adapter_line("Claude CLI", adapters["claude_cli"])
        _adapter_line("OpenAI API key", adapters["openai"])
        if result.get("env_generated"):
            console.print("\n  [dim].env generated (edit to customize)[/dim]")
