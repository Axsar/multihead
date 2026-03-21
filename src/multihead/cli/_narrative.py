"""Narrative pipeline commands."""

from __future__ import annotations

import click

from ._helpers import (
    _build_knowledge_deps,
    _parse_since,
    asyncio,
    console,
    Path,
)
from ._core import main


@main.group()
def narrative():
    """Narrative pipeline commands."""


@narrative.command("ingest")
@click.option("--source", type=click.Choice(["git", "chat", "agent", "markdown"]), required=True,
              help="Source type to ingest")
@click.option("--path", type=click.Path(exists=True),
              help="Repo path (git), transcript file (chat), or markdown file")
@click.option("--session-id", default=None, help="Session ID (for chat source)")
@click.option("--since", default=None, help="ISO datetime or relative (e.g. '24h', '7d')")
@click.option("--limit", default=100, type=int, help="Max items to ingest")
@click.option("--fuse/--no-fuse", default=True, help="Run fusion after ingest")
@click.option("--doc-type", default="plan",
              type=click.Choice(["plan", "status", "recovery", "fixes", "decision", "constraint"]),
              help="Document type (for markdown source)")
@click.option("--source-project", default=None,
              help="Source project scope ID (for markdown source, e.g. 'h2v')")
@click.option("--enhance/--no-enhance", default=False,
              help="Use Claude worker daemons for deep extraction (markdown only)")
@click.pass_context
def narrative_ingest(ctx, source, path, session_id, since, limit, fuse, doc_type, source_project, enhance):
    """Ingest data into the narrative pipeline."""
    from multihead.narrative.pipeline import NarrativePipeline
    from multihead.narrative.context_gen import generate_daemon_context

    settings = ctx.obj["settings"]
    ks, _, art, _ = _build_knowledge_deps(settings)
    pipeline = NarrativePipeline(ks, project_id="multihead", artifact_store=art)
    since_dt = _parse_since(since)

    if source == "git":
        repo_path = Path(path) if path else Path.cwd()
        count = pipeline.ingest_git(repo_path, since=since_dt, limit=limit)
        console.print(f"[green]Ingested {count} git commits[/green] from {repo_path.name}")
    elif source == "chat":
        if not path:
            console.print("[red]--path required for chat source (JSONL file)[/red]")
            return
        count = pipeline.ingest_chat(Path(path), session_id=session_id)
        console.print(f"[green]Ingested {count} artifacts[/green] from {Path(path).name}")
    elif source == "markdown":
        if not path:
            console.print("[red]--path required for markdown source[/red]")
            return
        if enhance:
            console.print(f"[cyan]Claude-enhanced extraction[/cyan] for {Path(path).name}...")
            count = asyncio.run(pipeline.ingest_markdown_enhanced(
                Path(path), doc_type=doc_type, source_project=source_project,
            ))
            console.print(f"[green]Enhanced: {count} claims[/green] from {Path(path).name} ({doc_type})")
        else:
            count = pipeline.ingest_markdown(
                Path(path), doc_type=doc_type, source_project=source_project,
            )
            console.print(f"[green]Ingested {count} claims[/green] from {Path(path).name} ({doc_type})")
    elif source == "agent":
        console.print("[yellow]Agent ingestion requires JSON on stdin or --path[/yellow]")
        return

    if fuse:
        fused = pipeline.run_full()
        stored = pipeline.store_fused_claims(fused)
        console.print(
            f"  Fusion: {len(fused.accepted_claims)} accepted, "
            f"{len(fused.contested_claims)} contested, "
            f"{len(fused.conflicts)} conflicts"
        )
        console.print(f"  Stored: {stored} claims")

        # Update daemon context
        ctx_path = settings.data_dir / "context" / "daemon_narrative.md"
        generate_daemon_context(ks, ctx_path)
        console.print(f"  Context updated: {ctx_path}")


@narrative.command("fuse")
@click.pass_context
def narrative_fuse(ctx):
    """Run fusion on all pending evidence in the pipeline."""
    from multihead.narrative.pipeline import NarrativePipeline
    from multihead.narrative.context_gen import generate_daemon_context

    settings = ctx.obj["settings"]
    ks, _, art, _ = _build_knowledge_deps(settings)
    pipeline = NarrativePipeline(ks, project_id="multihead", artifact_store=art)

    fused = pipeline.run_full()
    stored = pipeline.store_fused_claims(fused)

    console.print(f"[green]Fusion complete[/green]")
    console.print(f"  Accepted: {len(fused.accepted_claims)}")
    console.print(f"  Contested: {len(fused.contested_claims)}")
    console.print(f"  Conflicts: {len(fused.conflicts)}")
    console.print(f"  Stored: {stored} claims")

    ctx_path = settings.data_dir / "context" / "daemon_narrative.md"
    generate_daemon_context(ks, ctx_path)
    console.print(f"  Context updated: {ctx_path}")


@narrative.command("status")
@click.pass_context
def narrative_status(ctx):
    """Show narrative pipeline state."""
    settings = ctx.obj["settings"]
    ks, _, _, _ = _build_knowledge_deps(settings)

    accepted = ks.list_claims(status="accepted", limit=1000)
    contested = ks.list_claims(status="contested", limit=1000)
    events = ks.list_events(limit=1000)

    console.print("[bold]Narrative Pipeline Status[/bold]")
    console.print(f"  Accepted claims: {len(accepted)}")
    console.print(f"  Contested claims: {len(contested)}")
    console.print(f"  Knowledge events: {len(events)}")

    ctx_path = settings.data_dir / "context" / "daemon_narrative.md"
    if ctx_path.exists():
        console.print(f"  Daemon context: {ctx_path} ({ctx_path.stat().st_size} bytes)")
    else:
        console.print("  Daemon context: [dim]not generated yet[/dim]")
