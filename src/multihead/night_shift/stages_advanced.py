"""Night Shift stages 16-19: narrative fusion, solver discovery, recipe learning, backlog sweep."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AdvancedStagesMixin:
    """Mixin providing Night Shift stages 16 through 23."""

    async def _stage_behavioral_code_analysis(self, context: dict) -> dict:
        """Run LLM-based behavioral code analysis on project source files.

        Independent observation channel: analyzes WHAT code does (behavior,
        side effects, error handling) vs AST which only captures structure.
        Claims use observation_method='code_behavior_llm' for fusion.
        """
        from multihead.extractors.code_reader import scan_project_behavioral

        # Scan all known project repos for behavioral analysis
        from multihead._paths import get_known_project_roots
        repo_paths = [Path(r.rstrip("/")) for r in get_known_project_roots()]
        # Always include the current repo
        current_repo = Path(os.environ.get("MULTIHEAD_REPO", str(Path.cwd())))
        if current_repo not in repo_paths:
            repo_paths.insert(0, current_repo)

        # Dedup: check which repos already have behavioral claims from this run
        manifest_path = self.output_dir / ".behavioral_manifest.json"
        manifest: dict = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
            except Exception:
                manifest = {}

        adapter_fn = await self._adapter_or_fn()
        all_claims: list = []
        for repo_path in repo_paths:
            if not repo_path.is_dir():
                continue
            repo_key = str(repo_path)
            # Skip if already scanned today (dedup across restarts)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if manifest.get(repo_key, {}).get("date") == today:
                logger.info("Behavioral scan %s: already scanned today, skipping", repo_path.name)
                continue
            logger.info("Behavioral scan: %s", repo_path)
            repo_claims = await scan_project_behavioral(
                str(repo_path),
                adapter=adapter_fn,
                max_files=getattr(self.config, "behavioral_max_files", 100000),
                concurrency=getattr(self.config, "concurrency", 3),
                batch_mode=self.config.batch_mode,
                no_wait=self.config.no_wait,
            )
            # Tag each claim with repo root for provenance
            for c in repo_claims:
                c["repo_root"] = str(repo_path)
            all_claims.extend(repo_claims)
            manifest[repo_key] = {"date": today, "claims": len(repo_claims)}
            logger.info("Behavioral scan %s: %d claims", repo_path.name, len(repo_claims))

        manifest_path.write_text(json.dumps(manifest, indent=2))

        claims = all_claims

        # Insert behavioral claims into knowledge store
        from multihead.knowledge_models import (
            Claim, ClaimCanonical, ClaimScope, ClaimStatus, ClaimType,
            EntityRef, ScopeType, ValueObject,
        )
        from multihead.scope_inference import infer_scope
        from .models import _prov

        inserted = 0
        for claim_data in claims:
            stmt = claim_data.get("statement", "")
            if len(stmt.strip()) < 50:
                continue
            try:
                raw_type = claim_data.get("claim_type", "fact")
                type_map = {"architecture": "fact", "outcome": "fact", "observation": "fact"}
                claim_type_str = type_map.get(raw_type, raw_type)

                claim_obj = Claim(
                    claim_type=ClaimType(claim_type_str),
                    scope=ClaimScope(
                        scope_type=ScopeType.PROJECT,
                        scope_id=infer_scope(
                            claim_data.get("claim_key", ""),
                            stmt,
                        ),
                    ),
                    canonical=ClaimCanonical(
                        claim_key=claim_data.get("claim_key", "unknown.behavior"),
                        subject=EntityRef(
                            entity_type=claim_data.get("entity_type", "function"),
                            entity_id=claim_data.get("symbol", "unknown"),
                        ),
                        predicate="has_behavior",
                        object=ValueObject(value_type="string", value=stmt[:200]),
                    ),
                    statement=stmt,
                    confidence=claim_data.get("confidence", 0.75),
                    provenance=_prov(
                        observation_method="code_behavior_llm",
                        speaker="tool",
                        source_anchor={
                            k: v for k, v in {
                                "file_path": claim_data.get("file_path", ""),
                                "symbol": claim_data.get("symbol", ""),
                                "line": claim_data.get("line", ""),
                                "repo_root": claim_data.get("repo_root", ""),
                            }.items() if v
                        },
                        evidence=[e for e in claim_data.get("evidence", []) if isinstance(e, dict)],
                    ),
                )
                self.knowledge.insert_claim(claim_obj, dedup=True)
                inserted += 1
            except Exception as e:
                logger.debug("Failed to insert behavioral claim: %s", e)

        logger.info(
            "Behavioral code analysis: %d claims extracted, %d inserted",
            len(claims), inserted,
        )
        return {
            "behavioral_claims": len(claims),
            "behavioral_inserted": inserted,
            "metrics": {"behavioral_claims": inserted},
        }

    async def _stage_ci_results(self, context: dict) -> dict:
        """Extract CI/GitHub Actions results as claims.

        Strongest independent channel — code was actually executed and verified.
        Pass/fail is binary ground truth, no LLM judgment needed.
        """
        from multihead.extractors.ci_extractor import scan_ci
        from multihead.knowledge_models import (
            Claim, ClaimCanonical, ClaimScope, ClaimStatus, ClaimType,
            EntityRef, Provenance, ScopeType, ValueObject,
        )
        from multihead.scope_inference import infer_scope

        since = context.get("since")
        ci_claims = scan_ci(since=since, limit=100000)

        inserted = 0
        for c in ci_claims:
            stmt = c.get("statement", "")
            if len(stmt.strip()) < 50:
                continue
            try:
                scope_id = infer_scope(c.get("claim_key", ""), stmt)
                prov = Provenance(
                    produced_by={"kind": "extractor", "id": "ci_extractor"},
                    observation_method="ci_test",
                    speaker="tool",
                    source_anchor=c.get("source_anchor", {}),
                    evidence=c.get("evidence", []),
                )
                claim = Claim(
                    claim_type=ClaimType.FACT,
                    scope=ClaimScope(scope_type=ScopeType.PROJECT, scope_id=scope_id),
                    canonical=ClaimCanonical(
                        claim_key=c.get("claim_key", ""),
                        subject=EntityRef(entity_type="ci_run", entity_id=c.get("claim_key", "")),
                        predicate="concluded",
                        object=ValueObject(value_type="string", value=stmt[:200]),
                    ),
                    statement=stmt,
                    confidence=c.get("confidence", 0.95),
                    provenance=prov,
                )
                self.knowledge.insert_claim(claim, dedup=True)
                inserted += 1
            except Exception as e:
                logger.debug("Failed to insert CI claim: %s", e)

        logger.info("CI results: %d claims extracted, %d inserted", len(ci_claims), inserted)
        return {
            "ci_claims": len(ci_claims),
            "ci_inserted": inserted,
            "metrics": {"ci_claims": inserted},
        }

    async def _stage_narrative_fusion(self, context: dict) -> dict:
        """Run narrative pipeline: ingest git -> fuse -> store -> generate context.

        Scans all known project repos (not just cwd). Uses time-based window
        from last nightshift run, not fixed commit limit.
        """
        from multihead.narrative.context_gen import generate_daemon_context

        # Time-based window: since last successful nightshift run
        since = context.get("since")
        if not since:
            # Fallback: look up last run timestamp from DB
            try:
                with self.knowledge._connect() as conn:
                    row = conn.execute(
                        "SELECT created_at FROM claims WHERE statement LIKE 'Night Shift completed%' "
                        "ORDER BY created_at DESC LIMIT 1 OFFSET 1"
                    ).fetchone()
                    if row:
                        since = datetime.fromisoformat(row["created_at"])
            except Exception:
                pass

        # Scan all known project repos
        from multihead._paths import get_known_project_roots
        repo_paths = [Path(r.rstrip("/")) for r in get_known_project_roots()]
        current_repo = Path(os.environ.get("MULTIHEAD_REPO", str(Path.cwd())))
        if current_repo not in repo_paths:
            repo_paths.insert(0, current_repo)

        git_count = 0
        for repo_path in repo_paths:
            if repo_path.is_dir() and (repo_path / ".git").is_dir():
                try:
                    count = self.narrative_pipeline.ingest_git(repo_path, since=since)
                    git_count += count
                except Exception as e:
                    logger.warning("Git ingest failed for %s: %s", repo_path, e)

        # 2. Fuse pending evidence + apply to state
        fused = self.narrative_pipeline.run_full(
            unit_id=f"nightshift_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        )

        # 3. Store accepted claims
        stored = self.narrative_pipeline.store_fused_claims(fused)

        # 4. Generate daemon context file
        ctx_path = self.output_dir.parent / "context" / "daemon_narrative.md"
        generate_daemon_context(self.knowledge, ctx_path)

        logger.info(
            "Narrative fusion: %d git commits, %d accepted, %d stored",
            git_count, len(fused.accepted_claims), stored,
        )

        return {
            "narrative_git_commits": git_count,
            "narrative_claims_accepted": len(fused.accepted_claims),
            "narrative_claims_stored": stored,
            "metrics": {"narrative_claims": len(fused.accepted_claims)},
        }

    async def _stage_solver_discovery(self, context: dict) -> dict | None:
        """Weekly capability discovery: scan codebases for unique capabilities.

        Only runs if it's been 7+ days since last discovery run.
        Scans project directories for trained models, production pipelines,
        and domain-specific tools. Deposits findings into knowledge.db.
        """
        import os

        discovery_marker = self.output_dir / ".last_discovery"

        if discovery_marker.exists():
            mtime = datetime.fromtimestamp(
                os.path.getmtime(discovery_marker), tz=timezone.utc
            )
            if (datetime.now(timezone.utc) - mtime).days < 7:
                logger.info(
                    "Skipping solver discovery (last run %d days ago)",
                    (datetime.now(timezone.utc) - mtime).days,
                )
                return None

        try:
            from multihead.codebase_scanner import CodebaseScanner

            scanner = CodebaseScanner(knowledge_store=self.knowledge)
            project_paths = scanner.auto_discover_projects()

            if not project_paths:
                logger.info("No project paths auto-discovered, skipping")
                return {"capabilities_discovered": 0, "metrics": {}}

            results = scanner.scan_all(project_paths)

            # Deposit high-confidence capabilities as claims
            deposited = 0
            for result in results:
                for cap in result.capabilities:
                    if cap.confidence < 0.5:
                        continue
                    try:
                        claim = cap.to_claim(scope_id="capabilities")
                        self.knowledge.upsert_claim(**claim)
                        deposited += 1
                    except Exception:
                        pass

            # Write summary to output dir
            summary_path = self.output_dir / "discovery_report.md"
            summary_path.write_text(
                scanner.summary(results), encoding="utf-8"
            )

            # Update marker
            discovery_marker.write_text(
                datetime.now(timezone.utc).isoformat(), encoding="utf-8"
            )

            total_caps = sum(len(r.capabilities) for r in results)
            total_models = sum(len(r.model_checkpoints) for r in results)

            logger.info(
                "Solver discovery: %d capabilities found, %d deposited, %d models",
                total_caps, deposited, total_models,
            )

            # Phase 4: Run DiscoveryCoordinator for external model discovery
            external_results = await self._run_discovery_coordinator()

            return {
                "capabilities_discovered": total_caps,
                "capabilities_deposited": deposited,
                "model_checkpoints": total_models,
                "projects_scanned": len(results),
                "external_discovery": external_results,
                "metrics": {
                    "discovery_count": total_caps + external_results.get("discovered_count", 0),
                    "deposited_count": deposited,
                    "adopted_count": external_results.get("adopted_count", 0),
                },
            }

        except Exception as e:
            logger.error("Solver discovery failed: %s", e)
            return {
                "capabilities_discovered": 0,
                "error": str(e),
                "metrics": {"discovery_count": 0},
            }

    async def _run_discovery_coordinator(self) -> dict:
        """Run Phase 4 DiscoveryCoordinator for external model discovery.

        Discovers new models from HuggingFace, Ollama, BotVibes,
        and Papers with Code. Benchmarks and auto-adopts qualifying models.

        Returns:
            Discovery results summary dict
        """
        try:
            from multihead.discovery.coordinator import (
                create_discovery_job,
                load_discovery_config,
            )

            # Load config
            config = load_discovery_config()

            # Create coordinator
            registry_path = self.output_dir / "solver_registry.db"
            coordinator = create_discovery_job(
                registry_path,
                auto_benchmark=False,  # Night Shift benchmarking is separate
                auto_adopt=True,
                head_manager=self.heads,
                config=config,
            )

            # Run weekly discovery
            results = await coordinator.run_weekly_discovery(
                limit_per_source=config.get("discovery", {}).get("limit", 10),
            )

            # Register adopted solvers into HeadManager
            if results.get("adopted_count", 0) > 0:
                from multihead.discovery.adoption import register_adopted_solvers
                registered = register_adopted_solvers(
                    coordinator.registry, self.heads,
                )
                results["registered_heads"] = registered

            logger.info(
                "External discovery: %d discovered, %d adopted",
                results.get("discovered_count", 0),
                results.get("adopted_count", 0),
            )
            return results

        except Exception as e:
            logger.error("External discovery coordinator failed: %s", e)
            return {"discovered_count": 0, "adopted_count": 0, "error": str(e)}

    async def _stage_recipe_learning(self, context: dict) -> dict | None:
        """Weekly recipe learning: query experts for improved recipes.

        Only runs if it's been 7+ days since last recipe learning run.
        Queries BotVibes experts for recipe improvements, benchmarks them,
        evaluates via consensus, and adopts if better than current.
        """
        import os

        learning_marker = self.output_dir / ".last_recipe_learning"

        if learning_marker.exists():
            mtime = datetime.fromtimestamp(
                os.path.getmtime(learning_marker), tz=timezone.utc
            )
            if (datetime.now(timezone.utc) - mtime).days < 7:
                logger.info(
                    "Skipping recipe learning (last run %d days ago)",
                    (datetime.now(timezone.utc) - mtime).days,
                )
                return None

        try:
            return await self._run_recipe_learning()
        except Exception as e:
            logger.error("Recipe learning failed: %s", e)
            return {
                "recipes_evaluated": 0,
                "error": str(e),
                "metrics": {"recipes_learned": 0},
            }

    async def _run_recipe_learning(self) -> dict:
        """Run Phase 6 recipe learning pipeline.

        Discovers recipe improvements from BotVibes experts and evaluates
        them using multi-head consensus.

        Returns:
            Learning results summary dict
        """
        from multihead.recipe_learning import RecipeLearner, learn_recipe_workflow

        try:
            from multihead.acp_bridge import ACPBridge
            from multihead.registry.solver_registry import SolverRegistry

            # Initialize components
            registry_path = self.output_dir / "solver_registry.db"
            registry = SolverRegistry(registry_path)
            recipes_dir = self.output_dir / "learned_recipes"

            if not self.settings:
                raise RuntimeError("Settings required for recipe learning (pass settings to NightShift)")
            acp = ACPBridge(head_manager=self.heads, settings=self.settings)
            learner = RecipeLearner(
                acp_bridge=acp,
                recipes_dir=recipes_dir,
                head_manager=self.heads,
                registry=registry,
            )

            # Identify task types that could benefit from recipe learning
            # Use solver preferences to find active task types
            prefs = registry.list_preferences()
            task_types = list({p["task_type"] for p in prefs})

            if not task_types:
                # Default task types to try
                task_types = ["object_detection", "text_generation", "reasoning"]

            recipes_evaluated = 0
            recipes_adopted = 0

            for task_type in task_types[:5]:  # Cap at 5 task types per run
                try:
                    result = await learn_recipe_workflow(
                        goal=f"Optimal recipe for {task_type} tasks",
                        requirements={
                            "task_type": task_type,
                            "optimization": "accuracy and latency",
                            "max_steps": 5,
                        },
                        test_cases=[
                            {"input": f"test_{task_type}_1"},
                            {"input": f"test_{task_type}_2"},
                        ],
                        learner=learner,
                        save_name=f"learned-{task_type}",
                        task_type=task_type,
                    )

                    recipes_evaluated += 1
                    if result.get("evaluation", {}).get("action") == "adopt":
                        recipes_adopted += 1

                except Exception as e:
                    logger.error("Recipe learning failed for %s: %s", task_type, e)

            # Update marker
            learning_marker = self.output_dir / ".last_recipe_learning"
            learning_marker.write_text(
                datetime.now(timezone.utc).isoformat(), encoding="utf-8",
            )

            logger.info(
                "Recipe learning: %d evaluated, %d adopted",
                recipes_evaluated, recipes_adopted,
            )
            return {
                "recipes_evaluated": recipes_evaluated,
                "recipes_adopted": recipes_adopted,
                "task_types": task_types[:5],
                "metrics": {
                    "recipes_learned": recipes_adopted,
                    "recipes_evaluated": recipes_evaluated,
                },
            }

        except Exception as e:
            logger.error("Recipe learning pipeline failed: %s", e)
            return {"recipes_evaluated": 0, "recipes_adopted": 0, "error": str(e)}

    async def _stage_backlog_sweep(self, context: dict) -> dict | None:
        """Weekly backlog sweep: process all existing claims through analysis stages."""
        marker = self.output_dir / ".last_backlog_sweep"

        if marker.exists():
            import os
            mtime = datetime.fromtimestamp(os.path.getmtime(marker), tz=timezone.utc)
            days_ago = (datetime.now(timezone.utc) - mtime).days
            if days_ago < 7:
                logger.info("Skipping backlog sweep (last run %d days ago)", days_ago)
                return None

        logger.info("Starting weekly backlog sweep of all claims")
        summary = await self.run_backlog(batch_size=500, reset=True)
        marker.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
        return {
            "contradictions": summary.get("contradictions_found", 0),
            "links": summary.get("links_found", 0),
            "open_loops": summary.get("open_loops_found", 0),
            "metrics": {
                "contradictions_found": summary.get("contradictions_found", 0),
                "links_found": summary.get("links_found", 0),
                "open_loops_found": summary.get("open_loops_found", 0),
            },
        }
