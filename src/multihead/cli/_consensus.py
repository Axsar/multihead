"""Consensus commands."""

from __future__ import annotations

import click
from rich.table import Table

from ._helpers import (
    _build_orchestrator,
    asyncio,
    console,
    ConsensusConfig,
    ConsensusEngine,
    ConsensusStrategy,
    HeadTask,
)
from ._core import main


@main.group()
def consensus():
    """Multi-head consensus operations."""


@consensus.command("strategies")
def consensus_strategies():
    """List available consensus strategies."""
    table = Table(title="Consensus Strategies")
    table.add_column("Strategy", style="cyan")
    table.add_column("Description")

    strategies = [
        ("majority", "Simple majority vote: most common output wins."),
        ("weighted", "Weighted vote: heads with higher weight have more influence."),
        ("unanimous", "All heads must produce identical output."),
        ("threshold", "Majority vote with minimum agreement percentage required."),
    ]
    for name, desc in strategies:
        table.add_row(name, desc)
    console.print(table)


@consensus.command("test")
@click.option("--prompt", "-p", default="What is 2+2?", help="Prompt to send to all heads")
@click.option("--strategy", "-s", default="majority", help="Consensus strategy")
@click.option("--head", "-h", "head_ids", multiple=True, help="Head IDs to use (repeat for multiple)")
@click.pass_context
def consensus_test(ctx, prompt, strategy, head_ids):
    """Test consensus across heads with a prompt."""
    settings = ctx.obj["settings"]
    _, head_manager = _build_orchestrator(settings)

    available = list(head_manager.get_states().keys())
    if not available:
        console.print("[red]No heads registered.[/red]")
        return

    # Use specified heads or all available
    use_heads = list(head_ids) if head_ids else available

    # Validate heads exist
    for hid in use_heads:
        if hid not in available:
            console.print(f"[red]Head '{hid}' not found. Available: {available}[/red]")
            return

    try:
        strat = ConsensusStrategy(strategy)
    except ValueError:
        valid = [s.value for s in ConsensusStrategy]
        console.print(f"[red]Invalid strategy '{strategy}'. Valid: {valid}[/red]")
        return

    config = ConsensusConfig(
        heads=[HeadTask(head_id=hid) for hid in use_heads],
        strategy=strat,
    )

    console.print(f"[bold]Consensus Test[/bold]")
    console.print(f"  Prompt: {prompt}")
    console.print(f"  Strategy: {strategy}")
    console.print(f"  Heads: {use_heads}")
    console.print()

    async def _run():
        engine = ConsensusEngine(head_manager)
        result = await engine.execute(config, prompt)

        # Show individual votes
        table = Table(title="Individual Votes")
        table.add_column("Head", style="cyan")
        table.add_column("Success")
        table.add_column("Output")
        table.add_column("Latency (ms)")

        for vote in result.all_votes:
            success_str = "[green]yes[/green]" if vote.success else f"[red]no: {vote.error}[/red]"
            text = vote.outputs.get("text", "")[:80] if vote.success else "-"
            table.add_row(
                vote.head_id, success_str, text, f"{vote.latency_ms:.1f}",
            )
        console.print(table)

        # Show consensus result
        console.print()
        console.print(f"[bold]Consensus Output:[/bold] {result.consensus_outputs.get('text', '')[:200]}")
        console.print(f"  Agreement: {result.agreement_score:.0%}")
        console.print(f"  Strategy: {result.strategy_used.value}")

        if result.red_flags:
            console.print(f"\n[yellow]Red Flags ({len(result.red_flags)}):[/yellow]")
            for flag in result.red_flags:
                sev_color = {"critical": "red", "high": "yellow", "medium": "dim"}.get(
                    flag["severity"], "dim"
                )
                console.print(f"  [{sev_color}]{flag['severity'].upper()}[/{sev_color}] {flag['message']}")
        else:
            console.print(f"\n[green]No red flags.[/green]")

        # Show metrics
        console.print(f"\n[dim]Metrics: {result.metrics}[/dim]")

        await head_manager.shutdown()

    asyncio.run(_run())
