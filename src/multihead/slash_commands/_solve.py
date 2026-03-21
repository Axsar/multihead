"""Solve, routing, discovery, and learning command handlers.

/solve, /resolve, /harvest, /route, /discover, /select, /learn
"""

from __future__ import annotations

from pathlib import Path


class SolveMixin:
    """Mixin providing solve/route/discover/select/learn command handlers."""

    # -------------------------------------------------------------------
    # /solve
    # -------------------------------------------------------------------

    async def _handle_solve(self, args: list[str]) -> str:
        """Run the full solve pipeline: decompose -> route -> execute."""
        if not args:
            return (
                "Usage: /solve <task> [--strategy majority|weighted|first_to_ahead] "
                "[--max-steps N] [--marketplace] [--privacy public|internal|confidential] "
                "[--dry-run]"
            )

        if not self.head_manager:
            return "Head manager not available. Use `multihead shell` for solve."

        if not self.event_store or not self.artifact_store:
            return "Event/artifact store not available. Solve requires shell infrastructure."

        # Parse flags from args
        strategy = "first_to_ahead"
        max_steps = 20
        dry_run = False
        enable_marketplace = False
        task_parts: list[str] = []
        i = 0
        while i < len(args):
            if args[i] == "--strategy" and i + 1 < len(args):
                strategy = args[i + 1]
                i += 2
            elif args[i] == "--max-steps" and i + 1 < len(args):
                try:
                    max_steps = int(args[i + 1])
                except ValueError:
                    return f"Invalid --max-steps value: {args[i + 1]}"
                i += 2
            elif args[i] == "--dry-run":
                dry_run = True
                i += 1
            elif args[i] == "--marketplace":
                enable_marketplace = True
                i += 1
            else:
                task_parts.append(args[i])
                i += 1

        task = " ".join(task_parts)
        if not task:
            return "Usage: /solve <task>"

        from ..solve_pipeline import SolveConstraints, SolvePipeline

        constraints = SolveConstraints(
            strategy=strategy,
            max_steps=max_steps,
            enable_marketplace_delegation=enable_marketplace,
        )

        pipeline = SolvePipeline(
            head_manager=self.head_manager,
            event_store=self.event_store,
            artifact_store=self.artifact_store,
            knowledge_store=self.knowledge_store,
            runs_dir=self.runs_dir,
        )

        result = await pipeline.solve(task, constraints=constraints, dry_run=dry_run)

        # Format as Rich-friendly output
        status_color = "green" if result.status in ("done", "completed", "committed") else "red"
        lines = [
            f"[bold]Solve Complete[/bold]",
            f"  Status: [{status_color}]{result.status}[/{status_color}]",
            f"  Run ID: {result.run_id}",
            f"  Steps: {result.steps_succeeded}/{result.steps_total} succeeded"
            f" ({result.steps_failed} failed)",
            f"  Confidence: {result.confidence:.2f}",
            f"  Duration: {result.duration_seconds:.1f}s",
            f"  Plan steps: {result.plan_steps} ({result.parallel_steps} parallel)",
        ]
        if result.output and len(result.output) < 500:
            lines.append(f"\n  Output: {result.output[:500]}")
        elif result.output:
            lines.append(f"\n  Output (truncated): {result.output[:500]}...")

        return "\n".join(lines)

    # -------------------------------------------------------------------
    # /resolve
    # -------------------------------------------------------------------

    def _handle_resolve(self, args: list[str]) -> str:
        """Mark a claim as resolved so it stops resurfacing in RAG/inbox."""
        if not args:
            return "Usage: /resolve <claim_id>"
        claim_id = args[0]
        if not self.knowledge_store:
            return "Knowledge store not available."
        if not hasattr(self.knowledge_store, "resolve_claim"):
            return "Knowledge store does not support resolve_claim."
        found = self.knowledge_store.resolve_claim(claim_id)
        if found:
            return f"Claim {claim_id} marked as resolved. It will no longer appear in search/inbox."
        return f"Claim {claim_id} not found."

    # -------------------------------------------------------------------
    # /harvest
    # -------------------------------------------------------------------

    async def _handle_harvest(self, args: list[str]) -> str:
        """Handle /harvest command with subcommands."""
        from ..session_harvester import SessionHarvester

        if not self.knowledge_store:
            return "Knowledge store not available."

        harvester = SessionHarvester(
            knowledge_store=self.knowledge_store,
            max_claims_per_project=getattr(
                getattr(self.config, "services", None),
                "harvester_max_claims", 100,
            ),
        )

        if not args or args[0] == "status":
            status = harvester.status()
            lines = [
                f"Session Harvester Status",
                f"  Projects found:  {status['projects_found']}",
                f"  Total claims:    {status['total_claims']}",
                f"  Last full scan:  {status['last_full_scan'] or 'never'}",
            ]
            return "\n".join(lines)

        subcmd = args[0].lower()

        if subcmd == "run":
            result = harvester.harvest_all()
            lines = [
                f"Harvest Complete",
                f"  Scanned:    {result.projects_scanned} projects",
                f"  Harvested:  {result.projects_harvested} projects",
                f"  Skipped:    {result.projects_skipped} (unchanged)",
                f"  Claims:     {result.claims_deposited} deposited",
                f"  Duration:   {result.duration_seconds}s",
            ]
            if result.errors:
                lines.append(f"  Errors:     {len(result.errors)}")
                for err in result.errors[:5]:
                    lines.append(f"    - {err}")
            return "\n".join(lines)

        if subcmd == "list":
            status = harvester.status()
            if not status["projects"]:
                return "No projects found."
            lines = ["Projects:"]
            for p in status["projects"]:
                marker = "M" if p["has_memory"] else " "
                marker += "C" if p["has_claude_md"] else " "
                claims = p["claim_count"]
                lines.append(
                    f"  [{marker}] {p['scope_id']:15s} {p['file_count']} files  "
                    f"{claims:3d} claims  {p['decoded_path']}"
                )
            lines.append(f"\nLegend: M=MEMORY.md C=CLAUDE.md  Total: {len(status['projects'])} projects")
            return "\n".join(lines)

        return (
            "Usage: /harvest [status|run|list]\n"
            "  status  — show harvest status (default)\n"
            "  run     — trigger immediate harvest\n"
            "  list    — list discovered projects"
        )

    # -------------------------------------------------------------------
    # /route
    # -------------------------------------------------------------------

    async def _handle_route(self, args: list[str]) -> str:
        """Route to best head for task types."""
        if not args:
            return "Usage: /route <task_type> [task_type2 ...] [--privacy public|internal|confidential]"

        if not self.head_manager:
            return "Head manager not available."

        from ..router import Router

        privacy = None
        task_types: list[str] = []
        i = 0
        while i < len(args):
            if args[i] == "--privacy" and i + 1 < len(args):
                from ..models import DataSensitivity, PrivacyConstraint
                try:
                    sensitivity = DataSensitivity(args[i + 1])
                    privacy = PrivacyConstraint(data_sensitivity=sensitivity)
                except ValueError:
                    return f"Invalid privacy level: {args[i + 1]}"
                i += 2
            else:
                task_types.append(args[i])
                i += 1

        if not task_types:
            return "Specify at least one task type."

        r = Router(self.head_manager)
        ranked = r.rank_by_task(task_types, privacy=privacy)

        if not ranked:
            return f"No head can handle: {task_types}"

        lines = [f"[bold]Routing for {task_types}[/bold]\n"]
        for hid, score in ranked[:5]:
            marker = "→" if hid == ranked[0][0] else " "
            lines.append(f"  {marker} {hid}: {score:.1f}")
        return "\n".join(lines)

    # -------------------------------------------------------------------
    # /discover
    # -------------------------------------------------------------------

    async def _handle_discover(self, args: list[str]) -> str:
        """Query the solver registry for discovered models."""
        import os

        data_dir = os.environ.get("MULTIHEAD_DATA_DIR", str(Path.home() / ".multihead"))
        db_path = Path(data_dir) / "solver_registry.db"

        if not db_path.exists():
            return "No solver registry found. Run Night Shift or `multihead nightshift` first."

        from ..registry.solver_registry import SolverRegistry
        registry = SolverRegistry(db_path)

        solver_type = None
        source = None
        for i, a in enumerate(args):
            if a == "--type" and i + 1 < len(args):
                solver_type = args[i + 1]
            elif a == "--source" and i + 1 < len(args):
                source = args[i + 1]

        solvers = registry.list_solvers(solver_type=solver_type, source=source)[:20]

        if not solvers:
            return "No solvers in registry."

        lines = [f"[bold]{len(solvers)} solver(s) in registry[/bold]\n"]
        for s in solvers:
            status = s.get("adoption_status", "candidate")
            color = "green" if status == "adopted" else "yellow" if status == "candidate" else "red"
            lines.append(
                f"  [{color}]{status:10s}[/{color}] {s['name']} ({s['solver_id']})"
                f" — {s['solver_type']} via {s['source']}"
            )
        return "\n".join(lines)

    # -------------------------------------------------------------------
    # /select
    # -------------------------------------------------------------------

    async def _handle_select(self, args: list[str]) -> str:
        """Show or run solver selection for a task type."""
        if not args:
            return "Usage: /select <task_type>  (e.g. /select object_detection)"

        import os

        task_type = args[0]
        data_dir = os.environ.get("MULTIHEAD_DATA_DIR", str(Path.home() / ".multihead"))
        db_path = Path(data_dir) / "solver_registry.db"

        if not db_path.exists():
            return "No solver registry found."

        from ..registry.solver_registry import SolverRegistry
        registry = SolverRegistry(db_path)

        pref = registry.get_preference(task_type)
        lines = [f"[bold]Solver Selection: {task_type}[/bold]\n"]

        if pref:
            lines.append(f"  Preferred: [green]{pref['preferred_solver_id']}[/green]")
            lines.append(f"  Confidence: {pref['confidence_score']}")
            lines.append(f"  Reasoning: {pref['reasoning']}")
            lines.append(f"  Selected: {pref['selected_at']}")
        else:
            lines.append("  No preference recorded yet.")
            lines.append("  Run meta-reasoning selection via Python API or Night Shift.")

        all_prefs = registry.list_preferences()
        if all_prefs:
            lines.append(f"\n  All preferences ({len(all_prefs)}):")
            for p in all_prefs[:5]:
                lines.append(f"    {p['task_type']}: {p['preferred_solver_id']} ({p['confidence_score']:.2f})")

        return "\n".join(lines)

    # -------------------------------------------------------------------
    # /learn
    # -------------------------------------------------------------------

    async def _handle_learn(self, args: list[str]) -> str:
        """Show recipe learning status and versions."""
        import os

        data_dir = os.environ.get("MULTIHEAD_DATA_DIR", str(Path.home() / ".multihead"))
        db_path = Path(data_dir) / "solver_registry.db"

        if not db_path.exists():
            return "No solver registry found."

        from ..registry.solver_registry import SolverRegistry
        registry = SolverRegistry(db_path)

        task_type = args[0] if args else None
        versions = registry.list_recipe_versions(task_type=task_type)

        lines = ["[bold]Recipe Learning Status[/bold]\n"]

        if not versions:
            lines.append("  No recipes tracked yet.")
            lines.append("  Recipes are learned via Night Shift stage 18 or Python API.")
        else:
            lines.append(f"  {len(versions)} recipe version(s):\n")
            for v in versions[:10]:
                status = v.get("adoption_status", "candidate")
                color = "green" if status == "adopted" else "yellow"
                perf = v.get("performance_score")
                perf_str = f"{perf:.2f}" if perf is not None else "N/A"
                lines.append(
                    f"  [{color}]{status:10s}[/{color}] {v['recipe_id']} v{v['version']}"
                    f" — {v['task_type']} (perf: {perf_str})"
                )

                evals = registry.get_recipe_evaluations(v["recipe_id"], v["version"])
                if evals:
                    for e in evals:
                        lines.append(
                            f"      Vote: {e['head_id']} → {e['vote']} ({e['confidence']:.2f})"
                        )

        return "\n".join(lines)
