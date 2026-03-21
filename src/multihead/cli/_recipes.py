"""Recipe learning commands."""

from __future__ import annotations

import os

import click

from ._helpers import (
    _build_orchestrator,
    asyncio,
    console,
    Path,
)
from ._core import main


@main.group()
def recipes():
    """Recipe learning operations."""


@recipes.command("learn")
@click.argument("goal")
@click.option("--requirement", "-r", "requirements", multiple=True,
              help="Requirements as key=value pairs (repeat for multiple)")
@click.option("--test-data", type=click.Path(exists=True),
              help="Path to test data directory or file")
@click.option("--save-name", help="Name for saved recipe (auto-generated if not provided)")
@click.pass_context
def recipes_learn(ctx, goal, requirements, test_data, save_name):
    """Learn a new recipe from BotVibes experts.

    Example:
        multihead recipes learn "Extract entities from documents" \
            -r input_format=markdown \
            -r output_format=json \
            --test-data test_cases.json
    """
    settings = ctx.obj["settings"]

    # Parse requirements
    req_dict = {}
    for req in requirements:
        if "=" in req:
            key, value = req.split("=", 1)
            req_dict[key.strip()] = value.strip()
        else:
            console.print(f"[yellow]Warning: Skipping invalid requirement '{req}' (expected key=value)[/yellow]")

    # Load test data
    test_cases = []
    if test_data:
        import json
        test_path = Path(test_data)
        if test_path.suffix == ".json":
            test_cases = json.loads(test_path.read_text())
        else:
            console.print(f"[yellow]Warning: Test data format not recognized, using empty test cases[/yellow]")

    async def _learn():
        from ..acp_bridge import ACPBridge
        from ..recipe_learning import RecipeLearner, learn_recipe_workflow

        # Initialize ACP bridge
        acp_url = os.environ.get("ACP_URL", "http://localhost:8000")
        acp_token = os.environ.get("ACP_SESSION_KEY") or os.environ.get("ACP_TOKEN")

        if not acp_token:
            console.print("[red]Error: ACP_SESSION_KEY or ACP_TOKEN environment variable required[/red]")
            console.print("Set one of these to authenticate with BotVibes:")
            console.print("  export ACP_SESSION_KEY=your-token-here")
            return

        # Initialize orchestrator for benchmarking
        orchestrator, head_manager = _build_orchestrator(settings)

        acp_bridge = ACPBridge(
            head_manager=head_manager,
            settings=settings,
            acp_url=acp_url,
            api_key=acp_token,
        )

        # Initialize learner
        learner = RecipeLearner(
            acp_bridge=acp_bridge,
            recipes_dir=settings.config_dir / "recipes",
            orchestrator=orchestrator,
        )

        console.print(f"\n[cyan]Learning recipe:[/cyan] {goal}")
        if req_dict:
            console.print(f"[cyan]Requirements:[/cyan]")
            for key, value in req_dict.items():
                console.print(f"  {key}: {value}")

        # Run learning workflow
        result = await learn_recipe_workflow(
            goal=goal,
            requirements=req_dict,
            test_cases=test_cases,
            learner=learner,
            save_name=save_name,
        )

        if result["success"]:
            console.print("\n[green]✓ Recipe learning completed![/green]")
            console.print(f"[cyan]Recipe:[/cyan] {result['proposed_recipe']['goal']}")
            console.print(f"[cyan]Decision:[/cyan] {result['evaluation']['action']}")
            console.print(f"[cyan]Rationale:[/cyan] {result['evaluation']['rationale']}")

            if result.get("saved_path"):
                console.print(f"[cyan]Saved to:[/cyan] {result['saved_path']}")

            # Show benchmark results
            bench = result["benchmark_results"]
            if bench["test_cases_count"] > 0:
                success_rate = 100.0 * bench["success_count"] / bench["test_cases_count"]
                console.print(f"\n[cyan]Benchmark Results:[/cyan]")
                console.print(f"  Success: {bench['success_count']}/{bench['test_cases_count']} ({success_rate:.1f}%)")
                console.print(f"  Avg Latency: {bench['avg_latency_ms']:.1f}ms")
        else:
            console.print(f"\n[red]✗ Recipe learning failed:[/red] {result.get('error', 'Unknown error')}")

    asyncio.run(_learn())
