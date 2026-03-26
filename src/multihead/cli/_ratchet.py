"""CLI command for the experiment ratchet."""

from __future__ import annotations

import click

from ._core import main
from ._helpers import (
    Path,
    asyncio,
    console,
    logger,
    HeadManager,
    KnowledgeStore,
    load_heads,
    validate_heads,
    VRAMManager,
    VRAMPolicy,
)


@main.command()
@click.argument("experiment_id")
@click.option("--target", "-t", "target_files", multiple=True, help="Target file(s) the agent can edit")
@click.option("--test", "test_command", default="", help="Test command to run after each change")
@click.option("--metric", "metric_name", default="quality_score", help="Metric to optimize")
@click.option("--goal", "metric_goal", default="maximize", type=click.Choice(["maximize", "minimize"]))
@click.option("--max-iterations", "-n", default=20, help="Maximum iterations")
@click.option("--time-budget", default=300, help="Time budget per iteration (seconds)")
@click.option("--head", "head_id", default="core-llm", help="Head to use for proposals")
@click.option("--work-dir", default=".", help="Working directory (git repo)")
@click.option("--threshold", default=0.0, help="Stop when metric reaches this value")
@click.pass_context
def ratchet(
    ctx,
    experiment_id,
    target_files,
    test_command,
    metric_name,
    metric_goal,
    max_iterations,
    time_budget,
    head_id,
    work_dir,
    threshold,
):
    """Run an automated experiment ratchet (AutoResearch-style loop).

    Proposes changes → tests them → keeps improvements → reverts failures.
    All results logged to knowledge.db.

    Example:
        multihead ratchet balloon-optimization \\
            -t src/layout.py \\
            --test "python -m pytest tests/ -x" \\
            --metric overlap_count --goal minimize \\
            -n 30
    """
    from ..experiment_ratchet import ExperimentRatchet, RatchetConfig
    from ..settings import Settings

    settings = Settings.load(ctx.obj.get("config_dir"))
    data_dir = Path(settings.data_dir)
    ks = KnowledgeStore(data_dir / "knowledge.db")

    # Load heads for proposal generation
    heads = load_heads(settings.config_dir)
    validate_heads(heads)
    vram = VRAMManager(policy=VRAMPolicy.ON_DEMAND)
    hm = HeadManager(heads, vram_manager=vram)

    config = RatchetConfig(
        experiment_id=experiment_id,
        target_files=list(target_files),
        test_command=test_command,
        metric_name=metric_name,
        metric_goal=metric_goal,
        max_iterations=max_iterations,
        time_budget_secs=time_budget,
        work_dir=work_dir,
        head_id=head_id,
        quality_threshold=threshold,
    )

    ratchet_engine = ExperimentRatchet(
        knowledge_store=ks,
        head_manager=hm,
        config=config,
    )

    console.print(f"[bold]Experiment Ratchet: {experiment_id}[/bold]")
    console.print(f"  Target: {', '.join(target_files) or 'any'}")
    console.print(f"  Test: {test_command or 'none'}")
    console.print(f"  Metric: {metric_name} ({metric_goal})")
    console.print(f"  Iterations: {max_iterations}, Budget: {time_budget}s/iter")
    console.print(f"  Head: {head_id}")
    console.print()

    report = asyncio.run(ratchet_engine.run())

    console.print()
    console.print(f"[bold]Ratchet Complete[/bold]")
    console.print(f"  Iterations: {report.iterations_run}")
    console.print(f"  Kept: {report.iterations_kept}")
    console.print(f"  Reverted: {report.iterations_reverted}")
    console.print(f"  Errors: {report.iterations_errored}")
    console.print(f"  Best: {metric_name}={report.best_metrics.get(metric_name)} at iteration {report.best_iteration}")
    console.print(f"  Duration: {report.total_duration_secs:.1f}s")
    console.print(f"  Stopped: {report.stopped_reason}")
