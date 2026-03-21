"""Core CLI commands: main group, run, status, inspect, heads, info, doctor, export/import."""

from __future__ import annotations

import json
import sys

import click
from rich.table import Table

from ._helpers import (
    _build_orchestrator,
    _get_settings,
    _setup_logging,
    asyncio,
    console,
    logger,
    ArtifactStore,
    BundleExporter,
    BundleImporter,
    Diagnostics,
    EventStore,
    InitWizard,
    KnowledgeStore,
    Path,
    Settings,
    load_heads,
    load_recipe,
)


@click.group(invoke_without_command=True)
@click.version_option(package_name="multihead")
@click.option("--data-dir", envvar="MULTIHEAD_DATA_DIR", help="Data directory")
@click.option("--config-dir", envvar="MULTIHEAD_CONFIG_DIR", default=None, help="Config directory (auto-detected if not set)")
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.pass_context
def main(ctx, data_dir, config_dir, debug):
    """MultiHead: local multimodal task-runner."""
    ctx.ensure_object(dict)
    settings = _get_settings(data_dir, config_dir)
    ctx.obj["settings"] = settings
    _setup_logging(settings, debug)

    if ctx.invoked_subcommand is None:
        # First-run nudge
        wizard = InitWizard()
        if wizard.is_first_run(settings.config_dir):
            console.print(
                "[bold yellow]Welcome to MultiHead![/bold yellow]\n"
                "  Run [bold]multihead init --auto[/bold] to detect your hardware and generate config.\n"
                "  Run [bold]multihead init[/bold] for the interactive setup wizard.\n"
            )
            return

        # Show help plus live session status when invoked with no subcommand
        console.print(ctx.get_help())
        console.print()
        try:
            from ..solve import discover_active_sessions
            knowledge_store = KnowledgeStore(settings.knowledge_db_path)
            other_sessions = discover_active_sessions(
                knowledge_store,
                "multihead",
                "multihead-coordinator",
            )
            if len(other_sessions) == 0:
                console.print("[dim]Sessions: none detected[/dim]")
            else:
                names = [s["session_id"] for s in other_sessions[:3]]
                extra = f" (and {len(other_sessions) - 3} more)" if len(other_sessions) > 3 else ""
                console.print(
                    f"[green]Sessions:[/green] {len(other_sessions)} active — "
                    f"{', '.join(names)}{extra}"
                )
        except Exception:
            pass  # Session discovery is best-effort on startup


@main.command()
@click.argument("recipe")
@click.option("--input", "-i", "input_path", help="Input file or directory path")
@click.option("--input-json", help="JSON string of inputs")
@click.pass_context
def run(ctx, recipe, input_path, input_json):
    """Run a recipe pipeline."""
    settings = ctx.obj["settings"]

    try:
        work_order = load_recipe(settings.config_dir, recipe)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    if input_path:
        work_order.inputs["input_path"] = input_path
    if input_json:
        work_order.inputs.update(json.loads(input_json))

    orchestrator, head_manager = _build_orchestrator(settings)

    async def _run():
        state = await orchestrator.create_run(work_order)
        console.print(f"[green]Run created:[/green] {state.run_id}")
        console.print(f"  Goal: {work_order.goal}")
        console.print(f"  Steps: {len(work_order.steps)}")
        console.print()

        state = await orchestrator.execute_run(state.run_id)

        if state.status.value == "done":
            console.print(f"[green]Run completed successfully![/green]")
        else:
            console.print(f"[red]Run finished with status: {state.status.value}[/red]")

        # Show results
        console.print()
        table = Table(title="Step Results")
        table.add_column("Step", style="cyan")
        table.add_column("Head", style="magenta")
        table.add_column("Status", style="green")
        table.add_column("Tokens In")
        table.add_column("Tokens Out")

        for step_id, result in state.step_results.items():
            status_style = "green" if result.status.value == "committed" else "red"
            table.add_row(
                step_id,
                result.head_id,
                f"[{status_style}]{result.status.value}[/{status_style}]",
                str(result.metrics.get("tokens_in", "-")),
                str(result.metrics.get("tokens_out", "-")),
            )
        console.print(table)

        # Show artifact locations
        run_dir = settings.runs_dir / state.run_id
        console.print(f"\n[dim]Run directory: {run_dir}[/dim]")
        console.print(f"[dim]Events log: {run_dir / 'events.jsonl'}[/dim]")

        await head_manager.shutdown()

    asyncio.run(_run())


@main.command()
@click.argument("run_id", required=False)
@click.pass_context
def status(ctx, run_id):
    """Show run status (or list all runs)."""
    settings = ctx.obj["settings"]
    settings.ensure_dirs()
    event_store = EventStore(settings.runs_dir, settings.db_path)

    if run_id:
        state = event_store.replay(run_id)
        if state.work_order is None:
            console.print(f"[red]Run not found: {run_id}[/red]")
            sys.exit(1)

        console.print(f"[bold]Run:[/bold] {state.run_id}")
        console.print(f"  Status: {state.status.value}")
        console.print(f"  Goal: {state.work_order.goal}")
        console.print(f"  Step: {state.current_step_index}/{len(state.work_order.steps)}")
        console.print(f"  Created: {state.created_at}")
        if state.ended_at:
            console.print(f"  Ended: {state.ended_at}")
    else:
        runs = event_store.list_runs()
        if not runs:
            console.print("[dim]No runs found.[/dim]")
            return

        table = Table(title="Runs")
        table.add_column("Run ID", style="cyan")
        table.add_column("Status")
        table.add_column("Goal")
        table.add_column("Created")

        for r in runs:
            status_style = "green" if r["status"] == "done" else "yellow" if r["status"] == "running" else "dim"
            table.add_row(
                r["run_id"],
                f"[{status_style}]{r['status']}[/{status_style}]",
                r.get("goal", ""),
                r.get("created_at", ""),
            )
        console.print(table)


@main.command()
@click.argument("run_id")
@click.pass_context
def inspect(ctx, run_id):
    """Inspect a run's events and artifacts."""
    settings = ctx.obj["settings"]
    settings.ensure_dirs()
    event_store = EventStore(settings.runs_dir, settings.db_path)

    events = event_store.read_events(run_id)
    if not events:
        console.print(f"[red]No events for run: {run_id}[/red]")
        sys.exit(1)

    console.print(f"[bold]Events for run {run_id}:[/bold]\n")
    for e in events:
        kind_color = {
            "run_created": "blue",
            "step_started": "yellow",
            "step_committed": "green",
            "step_failed": "red",
            "run_done": "green",
            "run_failed": "red",
        }.get(e.kind.value, "dim")

        step_info = f" [{e.step_id}]" if e.step_id else ""
        console.print(f"  [{kind_color}]{e.kind.value}[/{kind_color}]{step_info}  {e.timestamp.isoformat()}")

        if e.data:
            for key, val in e.data.items():
                if key != "work_order":
                    val_str = str(val)[:100]
                    console.print(f"    {key}: {val_str}")


@main.command()
@click.pass_context
def heads(ctx):
    """List registered heads and their states."""
    settings = ctx.obj["settings"]
    head_manifests = load_heads(settings.config_dir)

    if not head_manifests:
        console.print("[dim]No heads registered. Create config/heads.yaml[/dim]")
        return

    table = Table(title="Registered Heads")
    table.add_column("Head ID", style="cyan")
    table.add_column("Name")
    table.add_column("Adapter", style="magenta")
    table.add_column("Model")
    table.add_column("Kind")
    table.add_column("GPU")
    table.add_column("VRAM (MB)")

    for hid, m in head_manifests.items():
        table.add_row(
            hid, m.name, m.adapter.value, m.model,
            m.kind, "yes" if m.gpu_required else "no",
            str(m.vram_hint_mb) if m.vram_hint_mb else "-",
        )
    console.print(table)


@main.command()
@click.pass_context
def info(ctx):
    """Show system overview: config, hardware, knowledge, and heads."""
    settings = ctx.obj["settings"]
    head_manifests = load_heads(settings.config_dir)

    console.print("[bold]MultiHead System Info[/bold]\n")

    # Paths
    console.print(f"  Data dir:   {settings.data_dir}")
    console.print(f"  Config dir: {settings.config_dir}")
    console.print(f"  Knowledge:  {settings.knowledge_db_path}")

    # Knowledge stats
    try:
        import sqlite3
        conn = sqlite3.connect(str(settings.knowledge_db_path))
        count = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
        conn.close()
        console.print(f"  Claims:     {count:,}")
    except Exception:
        console.print("  Claims:     [dim]unavailable[/dim]")

    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory // (1024**2)
            console.print(f"  GPU:        {name} ({vram:,} MB)")
        else:
            console.print("  GPU:        [dim]none (CPU only)[/dim]")
    except ImportError:
        console.print("  GPU:        [dim]torch not installed[/dim]")

    # Heads summary
    if head_manifests:
        gpu_heads = [h for h in head_manifests.values() if h.gpu_required]
        cpu_heads = [h for h in head_manifests.values() if not h.gpu_required]
        console.print(f"  Heads:      {len(head_manifests)} ({len(gpu_heads)} GPU, {len(cpu_heads)} CPU)")
    else:
        console.print("  Heads:      [dim]none configured[/dim]")

    # Runs
    try:
        settings.ensure_dirs()
        event_store = EventStore(settings.runs_dir, settings.db_path)
        runs = event_store.list_runs()
        console.print(f"  Runs:       {len(runs)}")
    except Exception:
        console.print("  Runs:       [dim]unavailable[/dim]")

    console.print()


@main.command("export")
@click.argument("run_id", required=False)
@click.option("--output", "-o", default=".", help="Output directory")
@click.option("--project", is_flag=True, help="Export entire project")
@click.pass_context
def export_cmd(ctx, run_id, output, project):
    """Export a run or project as a zip bundle."""
    settings = ctx.obj["settings"]
    settings.ensure_dirs()
    artifact_store = ArtifactStore(settings.artifacts_dir, settings.db_path)
    event_store = EventStore(settings.runs_dir, settings.db_path)
    exporter = BundleExporter(event_store, artifact_store, settings.runs_dir)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    if project:
        path = exporter.export_project(output_path)
        console.print(f"[green]Project exported to:[/green] {path}")
    elif run_id:
        path = exporter.export_run(run_id, output_path)
        console.print(f"[green]Run exported to:[/green] {path}")
    else:
        console.print("[red]Specify a run_id or use --project[/red]")


@main.command("import")
@click.argument("zip_path")
@click.pass_context
def import_cmd(ctx, zip_path):
    """Import a bundle from a zip file."""
    settings = ctx.obj["settings"]
    settings.ensure_dirs()
    artifact_store = ArtifactStore(settings.artifacts_dir, settings.db_path)
    importer = BundleImporter(artifact_store, settings.runs_dir)

    result = importer.import_bundle(Path(zip_path))
    console.print(f"[green]Imported {result['type']} bundle[/green]")
    for k, v in result.items():
        console.print(f"  {k}: {v}")


@main.command("doctor")
@click.pass_context
def doctor(ctx):
    """Run diagnostic checks on the installation."""
    settings = ctx.obj["settings"]
    diag = Diagnostics(settings.data_dir, settings.config_dir)
    report = diag.run_all()

    for check in report.checks:
        icon = "[green]OK[/green]" if check.passed else "[red]FAIL[/red]"
        console.print(f"  {icon} {check.name}: {check.message}")
        if not check.passed and check.suggestion:
            console.print(f"      [dim]{check.suggestion}[/dim]")

    console.print(f"\n{report.summary}")
    if not report.all_passed:
        console.print("[dim]Run 'multihead init --auto' to fix configuration issues.[/dim]")
