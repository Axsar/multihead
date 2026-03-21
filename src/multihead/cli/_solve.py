"""Solve commands: autonomous solve and distributed solve.

NOTE: The original cli.py defines two functions both named 'solve' registered
via @main.command(). In Click, the second registration overwrites the first.
We preserve this exact behavior here: the first 'solve' is defined and registered,
then the second 'solve' overwrites it. The net effect is that only the distributed
solve command is active.
"""

from __future__ import annotations

import os
import sys

import click

from ._helpers import (
    _build_orchestrator,
    asyncio,
    console,
    KnowledgeStore,
    ConsensusStrategy,
)
from ._core import main


# --- First solve definition (overridden by the second one below) ---

@main.command()
@click.argument("task", required=False, default=None)
@click.option("--heads", multiple=True, help="Heads to use for consensus decomposition (default: all LLM heads)")
@click.option("--strategy", default="first_to_ahead",
              type=click.Choice(["majority", "weighted", "unanimous", "first_to_ahead"]),
              help="Consensus strategy for decomposition")
@click.option("--max-steps", default=50, type=int, help="Maximum decomposition steps")
@click.option("--enable-knowledge", is_flag=True, default=True, help="Enable real-time knowledge extraction")
@click.option("--enable-tests", is_flag=True, default=True, help="Enable automatic test generation for code steps")
@click.option("--min-confidence", default=0.7, type=float, help="Min confidence for auto-accepting claims (0-1)")
@click.option("--show-plan", is_flag=True, help="Show decomposition plan before execution")
@click.option("--dry-run", is_flag=True, help="Decompose and show plan without executing")
@click.option("--gh-issue", default=None, help="Solve a GitHub issue (42, owner/repo#42, or URL)")
@click.option("--gh-track", is_flag=True, help="Create tracking issue for this solve run")
@click.option("--gh-subtasks", is_flag=True, help="Create child issues for decomposed steps")
@click.option("--gh-comment", is_flag=True, help="Post solve results as comment on source issue")
@click.option("--marketplace", is_flag=True, help="Enable BotVibes marketplace discovery for unroutable steps")
@click.pass_context
def solve(ctx, task, heads, strategy, max_steps, enable_knowledge, enable_tests,
          min_confidence, show_plan, dry_run, gh_issue, gh_track, gh_subtasks, gh_comment, marketplace):
    """Autonomous task solver: decompose -> route -> execute -> learn.

    This command implements the full autonomous loop:
    1. Multi-head consensus decomposition (multiple models propose plans, vote on best)
    2. DAG inference for parallel execution
    3. Auto-routing (best head per step)
    4. Real-time knowledge extraction
    5. Automatic test generation (for code steps)
    6. Learning summary

    Example:
        multihead solve "Build a web scraper for product prices"

        multihead solve --gh-issue 42 --gh-comment --auto-approve

        multihead solve "Refactor auth module" --gh-track

        multihead solve --gh-issue Axsar/multihead#42 --gh-subtasks
    """
    from ..auto_decomposition import AutoDecomposer
    from ..consensus import ConsensusStrategy

    # --gh-subtasks implies --gh-track
    if gh_subtasks:
        gh_track = True

    # --- GitHub input flow: fetch issue as task ---
    gh_issue_data = None
    if gh_issue:
        from ..github_integration import fetch_issue
        try:
            gh_issue_data = fetch_issue(gh_issue)
            issue_title = gh_issue_data.get("title", "")
            issue_body = gh_issue_data.get("body", "") or ""
            issue_num = gh_issue_data.get("number", gh_issue)
            task = f"GitHub Issue #{issue_num}: {issue_title}\n\n{issue_body}"
            console.print(f"[green]Fetched GitHub issue #{issue_num}:[/green] {issue_title}")
        except RuntimeError as e:
            console.print(f"[red]Failed to fetch issue: {e}[/red]")
            sys.exit(1)
    elif not task:
        console.print("[red]Error: provide a TASK argument or --gh-issue[/red]")
        sys.exit(1)

    settings = ctx.obj["settings"]
    orchestrator, head_manager = _build_orchestrator(settings)

    console.print(f"[bold cyan]MultiHead Autonomous Solver[/bold cyan]")
    console.print(f"Task: {task}\n")

    async def _solve():
        from ..solve_pipeline import SolveConstraints, SolvePipeline

        knowledge_store = KnowledgeStore(settings.knowledge_db_path)

        # Setup ACP bridge for remote delegation (if configured)
        acp_bridge = None
        acp_url = os.environ.get("ACP_URL")
        acp_key = os.environ.get("ACP_API_KEY") or os.environ.get("ACP_SESSION_KEY")
        if acp_url and acp_key:
            from ..acp_bridge import ACPBridge
            acp_bridge = ACPBridge(
                head_manager=head_manager,
                settings=settings,
                acp_url=acp_url,
                api_key=acp_key,
                project_id=os.environ.get("ACP_PROJECT_ID"),
                agent_id=os.environ.get("ACP_AGENT_ID", "multihead-agent"),
            )
            await acp_bridge.start()
            if acp_bridge.connected:
                console.print("[green]ACP bridge connected — remote delegation enabled[/green]\n")
            else:
                console.print("[dim]ACP bridge offline — local execution only[/dim]\n")

        enable_delegation = bool(marketplace and acp_bridge and acp_bridge.connected)

        constraints = SolveConstraints(
            max_steps=max_steps,
            strategy=strategy,
            enable_knowledge_hook=enable_knowledge,
            enable_test_generation=enable_tests,
            enable_marketplace_delegation=enable_delegation,
        )

        pipeline = SolvePipeline(
            head_manager=head_manager,
            event_store=orchestrator.events,
            artifact_store=orchestrator.artifacts,
            knowledge_store=knowledge_store,
            runs_dir=settings.runs_dir,
            acp_bridge=acp_bridge,
        )

        # --show-plan: decompose separately to display before execution
        if show_plan:
            auto_decomposer = AutoDecomposer(
                head_manager=head_manager,
                knowledge_store=knowledge_store,
            )
            heads_list = list(heads) if heads else None
            strategy_enum = {
                "majority": ConsensusStrategy.MAJORITY,
                "weighted": ConsensusStrategy.WEIGHTED,
                "unanimous": ConsensusStrategy.UNANIMOUS,
                "first_to_ahead": ConsensusStrategy.FIRST_TO_AHEAD,
            }[strategy]

            with console.status("[bold green]Decomposing task..."):
                plan, _ = await auto_decomposer.decompose_with_consensus(
                    goal=task, heads=heads_list, strategy=strategy_enum,
                    auto_validate=True, enable_research_features=True,
                )

            console.print("[bold]Decomposition Plan:[/bold]")
            for i, phase in enumerate(plan.phases, 1):
                console.print(f"  Phase {i}: {phase.goal}")
                for leaf in phase.leaves():
                    console.print(f"    - {leaf.goal} [{leaf.action_type}]")
            console.print()

        # Run the pipeline
        status_msg = "[bold green]Decomposing..." if dry_run else "[bold green]Solving..."
        with console.status(status_msg):
            result = await pipeline.solve(task, constraints=constraints, dry_run=dry_run)

        # Dry-run: show plan and exit
        if result.dry_run:
            console.print(f"\n{result.output}")
            console.print(f"\n[dim]Duration: {result.duration_seconds:.1f}s[/dim]")
            console.print(f"[dim]Run without --dry-run to execute.[/dim]")
            if acp_bridge:
                await acp_bridge.stop()
            return

        state = result.state

        # --- GitHub tracking: create tracking issue ---
        tracking_issue = None
        if gh_track and state and state.work_order:
            from ..github_integration import (
                create_issue as gh_create_issue,
                create_subtask_issues as gh_create_subtasks,
            )
            try:
                checklist_lines = [f"Autonomous solve run for: **{task[:200]}**\n"]
                for i, step in enumerate(state.work_order.steps, 1):
                    checklist_lines.append(f"- [ ] Step {i}: {step.name}")
                checklist_lines.append(f"\nRun ID: `{result.run_id}`")
                tracking_body = "\n".join(checklist_lines)

                goal_short = task[:60].split("\n")[0]
                tracking_issue = gh_create_issue(
                    title=f"MultiHead Solve: {goal_short}",
                    body=tracking_body,
                    labels=["multihead-solve"],
                )
                console.print(
                    f"  [green]Tracking issue created:[/green] "
                    f"#{tracking_issue['number']} {tracking_issue['url']}"
                )

                if gh_subtasks:
                    step_dicts = [
                        {"name": s.name, "description": s.prompt or "", "action_type": s.action or ""}
                        for s in state.work_order.steps
                    ]
                    gh_create_subtasks(str(tracking_issue["number"]), step_dicts)
                    console.print(f"  [green]Created subtask issues[/green]\n")
            except RuntimeError as e:
                console.print(f"  [yellow]GitHub tracking failed: {e}[/yellow]\n")

        # Final summary
        status_color = "green" if result.status in ("done", "completed", "committed") else "red"
        console.print(f"[bold {status_color}]{'✓' if 'green' in status_color else '✗'} Task {'Complete' if 'green' in status_color else 'Failed'}![/bold {status_color}]")
        console.print(f"\nResults:")
        console.print(f"  Run ID: {result.run_id}")
        console.print(f"  Status: {result.status}")
        console.print(f"  Steps: {result.steps_succeeded}/{result.steps_total} succeeded ({result.steps_failed} failed)")
        console.print(f"  Confidence: {result.confidence:.2f}")
        console.print(f"  Duration: {result.duration_seconds:.1f}s")
        console.print(f"  Plan: {result.plan_steps} steps ({result.parallel_steps} parallel)")
        console.print(f"\nInspect: multihead inspect {result.run_id}")

        # --- GitHub result posting ---
        should_post = gh_comment or gh_track
        if should_post and state:
            from ..github_integration import (
                format_solve_results,
                comment_on_issue as gh_comment_issue,
                close_issue as gh_close_issue,
            )

            step_dicts = []
            for step_def in (state.work_order.steps if state.work_order else []):
                sr = state.step_results.get(step_def.name, None)
                step_dicts.append({
                    "name": step_def.name,
                    "status": sr.status.value if sr else "unknown",
                    "head_id": sr.head_id if sr else "",
                })

            result_md = format_solve_results(
                goal=task,
                status=result.status,
                duration_seconds=result.duration_seconds,
                steps=step_dicts,
                run_id=result.run_id,
            )

            try:
                if gh_comment and gh_issue:
                    gh_comment_issue(gh_issue, result_md)
                    console.print(f"\n[green]Results posted as comment on issue[/green]")

                if gh_track and tracking_issue:
                    gh_comment_issue(str(tracking_issue["number"]), result_md)
                    if result.status in ("completed", "committed"):
                        gh_close_issue(
                            str(tracking_issue["number"]),
                            comment="Solve completed successfully.",
                        )
                        console.print(f"[green]Tracking issue #{tracking_issue['number']} closed[/green]")
                    else:
                        console.print(
                            f"[yellow]Tracking issue #{tracking_issue['number']} "
                            f"left open (status: {result.status})[/yellow]"
                        )
            except RuntimeError as e:
                console.print(f"[yellow]GitHub posting failed: {e}[/yellow]")

        # Cleanup ACP bridge
        if acp_bridge:
            await acp_bridge.stop()

    asyncio.run(_solve())


# --- Second solve definition (this one wins, overriding the first) ---

@main.command()
@click.argument("task", required=False, default=None)
@click.option("--auto-approve", is_flag=True, default=False,
              help="Auto-execute winning proposal (default: manual approval required)")
@click.option("--timeout", default=300, type=int,
              help="Proposal timeout in seconds (default: 300)")
@click.option("--project-id", default="multihead",
              help="Project scope for collaboration (default: multihead)")
@click.option("--strategy", default="majority",
              type=click.Choice(["majority", "weighted", "unanimous", "threshold", "first_to_ahead"]),
              help="Consensus strategy for proposal voting (default: majority)")
@click.option("--gh-issue", default=None, help="Solve a GitHub issue (42, owner/repo#42, or URL)")
@click.option("--gh-track", is_flag=True, help="Create tracking issue for this solve run")
@click.option("--gh-subtasks", is_flag=True, help="Create child issues for decomposed steps")
@click.option("--gh-comment", is_flag=True, help="Post solve results as comment on source issue")
@click.pass_context
def solve(ctx, task, auto_approve, timeout, project_id, strategy,  # noqa: F811
          gh_issue, gh_track, gh_subtasks, gh_comment):
    """Distribute task to multiple agents via consensus voting.

    TASK: Natural language description of task to solve

    Flow:
      1. Post task decomposition request to knowledge.db
      2. Wait for agent proposals (timeout: --timeout seconds)
      3. Run consensus voting on proposals
      4. Assign winning proposal to agent for execution
      5. Monitor execution (if --auto-approve enabled)
      6. Collect and return results

    Example:
      multihead solve "Check if we're on a git branch, if not create one"
      multihead solve "Implement feature X" --auto-approve
      multihead solve --gh-issue 42 --gh-comment --auto-approve
      multihead solve "Refactor auth module" --gh-track
    """
    from ..solve import SolveConfig, SolveCoordinator, discover_active_sessions

    # --gh-subtasks implies --gh-track
    if gh_subtasks:
        gh_track = True

    # --- GitHub input flow: fetch issue as task ---
    gh_issue_data = None
    if gh_issue:
        from ..github_integration import fetch_issue
        try:
            gh_issue_data = fetch_issue(gh_issue)
            issue_title = gh_issue_data.get("title", "")
            issue_body = gh_issue_data.get("body", "") or ""
            issue_num = gh_issue_data.get("number", gh_issue)
            task = f"GitHub Issue #{issue_num}: {issue_title}\n\n{issue_body}"
            console.print(f"[green]Fetched GitHub issue #{issue_num}:[/green] {issue_title}")
        except RuntimeError as e:
            console.print(f"[red]Failed to fetch issue: {e}[/red]")
            sys.exit(1)
    elif not task:
        console.print("[red]Error: provide a TASK argument or --gh-issue[/red]")
        sys.exit(1)

    settings = ctx.obj["settings"]

    console.print(f"\n[bold]MultiHead Solve:[/bold] {task}\n")
    console.print(f"Project: {project_id}")
    console.print(f"Timeout: {timeout}s")

    # Build dependencies
    orchestrator, head_manager = _build_orchestrator(settings)
    knowledge_store = KnowledgeStore(settings.knowledge_db_path)

    # Discover active sessions for display
    other_sessions = discover_active_sessions(
        knowledge_store,
        project_id,
        "multihead-coordinator",
    )

    # Display session count
    if len(other_sessions) == 0:
        console.print("[dim]No other sessions detected - solo mode[/dim]")
    else:
        names = [s['session_id'] for s in other_sessions[:3]]
        if len(other_sessions) <= 3:
            console.print(f"[green]{len(other_sessions)} session(s) active:[/green] {', '.join(names)}")
        else:
            console.print(f"[green]{len(other_sessions)} session(s) active:[/green] {', '.join(names)} (and {len(other_sessions) - 3} more)")

    # Show explicit flag status
    if auto_approve:
        console.print("Auto-approve: [green]ON[/green] (explicit)")
    else:
        console.print("Auto-approve: [yellow]adaptive[/yellow] (will prompt if multi-session)")
    console.print()

    # Map strategy string to enum
    from ..consensus import ConsensusStrategy as CS
    strategy_enum = {
        "majority": CS.MAJORITY,
        "weighted": CS.WEIGHTED,
        "unanimous": CS.UNANIMOUS,
        "threshold": CS.THRESHOLD,
        "first_to_ahead": CS.FIRST_TO_AHEAD,
    }[strategy]

    # Configure solve session
    config = SolveConfig(
        project_id=project_id,
        session_id="multihead-coordinator",
        proposal_timeout_seconds=float(timeout),
        auto_approve=False,  # Will be adapted by SolveCoordinator
        consensus_strategy=strategy_enum,
    )

    # Run coordinator with explicit flag
    coordinator = SolveCoordinator(
        knowledge_store=knowledge_store,
        head_manager=head_manager,
        orchestrator=orchestrator,
        config=config,
        explicit_auto_approve=auto_approve if auto_approve else None,
    )

    result = asyncio.run(coordinator.solve(task))

    # Display results
    console.print("\n[bold]Results:[/bold]")
    console.print(f"  Request ID: {result.request_id}")
    console.print(f"  Proposals received: {result.proposals_received}")

    if result.success:
        console.print(f"  [green]✓[/green] Winning proposal: {result.winning_proposal_id}")
        console.print(f"  [green]✓[/green] Assigned to: {result.assigned_agent}")

        if result.decomposition:
            console.print(f"\n[bold]Decomposition (execute these steps):[/bold]\n")
            console.print(result.decomposition)
        elif result.execution_started:
            if result.result_claim_id:
                console.print(f"  [green]✓[/green] Execution complete: {result.result_claim_id}")
            else:
                console.print(f"  [yellow]⏳[/yellow] Execution timed out (check knowledge.db for updates)")
        else:
            console.print(f"  [yellow]⏳[/yellow] Awaiting manual approval by agent operator")
    else:
        console.print(f"  [red]✗[/red] Error: {result.error}")

    # --- GitHub result posting ---
    if gh_comment or gh_track:
        from ..github_integration import (
            format_solve_results,
            comment_on_issue as gh_comment_issue,
            close_issue as gh_close_issue,
            create_issue as gh_create_issue,
            create_subtask_issues as gh_create_subtasks,
        )

        status_str = "completed" if result.success else "failed"
        result_md = format_solve_results(
            goal=task,
            status=status_str,
            duration_seconds=None,
            steps=[],  # distributed solve doesn't expose step details here
            run_id=result.request_id,
        )

        try:
            # Create tracking issue if requested
            if gh_track:
                tracking = gh_create_issue(
                    title=f"MultiHead Solve: {task[:60].split(chr(10))[0]}",
                    body=result_md,
                    labels=["multihead-solve"],
                )
                console.print(
                    f"  [green]Tracking issue created:[/green] "
                    f"#{tracking['number']} {tracking['url']}"
                )
                if result.success:
                    gh_close_issue(str(tracking["number"]))

            # Post comment on source issue
            if gh_comment and gh_issue:
                gh_comment_issue(gh_issue, result_md)
                console.print(f"  [green]Results posted as comment on issue[/green]")
        except RuntimeError as e:
            console.print(f"  [yellow]GitHub posting failed: {e}[/yellow]")

    console.print()
