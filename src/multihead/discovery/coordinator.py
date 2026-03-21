"""Discovery coordinator for automated solver discovery and benchmarking."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from multihead.benchmarking.base import BenchmarkRunner
from multihead.discovery.base import DiscoveryAgent, SolverCandidate
from multihead.registry import AdoptionRule, SolverRegistry

logger = logging.getLogger(__name__)


class DiscoveryCoordinator:
    """Coordinates solver discovery, benchmarking, and adoption.

    This is the orchestrator for Phase 4 automation:
    1. Run discovery agents (HuggingFace, Ollama, BotVibes)
    2. Benchmark new solvers
    3. Check adoption rules
    4. Auto-register qualifying solvers
    """

    def __init__(
        self,
        registry: SolverRegistry,
        benchmark_runner: BenchmarkRunner,
        discovery_agents: dict[str, DiscoveryAgent],
        *,
        auto_benchmark: bool = True,
        auto_adopt: bool = True,
    ):
        """Initialize discovery coordinator.

        Args:
            registry: Solver registry for persistence
            benchmark_runner: Benchmark runner with registered benchmarks
            discovery_agents: Dict of agent name -> discovery agent
            auto_benchmark: Automatically benchmark new discoveries
            auto_adopt: Automatically adopt solvers meeting criteria
        """
        self.registry = registry
        self.benchmarks = benchmark_runner
        self.agents = discovery_agents
        self.auto_benchmark = auto_benchmark
        self.auto_adopt = auto_adopt

    async def run_weekly_discovery(
        self,
        *,
        solver_types: list[str] | None = None,
        min_downloads: int = 1000,
        limit_per_source: int = 10,
    ) -> dict[str, Any]:
        """Run weekly discovery across all sources.

        Args:
            solver_types: Filter by types (None = all types)
            min_downloads: Minimum downloads for HuggingFace
            limit_per_source: Max solvers per source

        Returns:
            Summary dict with counts and adopted solvers
        """
        logger.info("Starting weekly discovery job")
        started_at = datetime.now(timezone.utc)
        updated_since = started_at - timedelta(days=7)  # Last week's updates

        results = {
            "started_at": started_at.isoformat(),
            "discovered_count": 0,
            "new_solvers": [],
            "benchmarked_count": 0,
            "adopted_count": 0,
            "adopted_solvers": [],
            "errors": [],
        }

        # Run discovery for each agent
        for agent_name, agent in self.agents.items():
            logger.info("Running discovery: %s", agent_name)
            try:
                candidates = await agent.discover_new_solvers(
                    solver_types=solver_types,
                    min_downloads=min_downloads,
                    updated_since=updated_since,
                    limit=limit_per_source,
                )

                for candidate in candidates:
                    # Check if already in registry
                    existing = self.registry.get_solver(candidate.solver_id)
                    if existing and existing.get("version") == candidate.version:
                        logger.debug("Skipping %s (already registered)", candidate.solver_id)
                        continue

                    # Add to registry
                    self.registry.add_solver(candidate, adoption_status="candidate")
                    results["discovered_count"] += 1
                    results["new_solvers"].append({
                        "solver_id": candidate.solver_id,
                        "name": candidate.name,
                        "source": candidate.source,
                        "type": candidate.solver_type,
                    })

                    logger.info(
                        "Discovered: %s (%s) from %s",
                        candidate.name, candidate.solver_type, candidate.source
                    )

            except Exception as e:
                logger.error("Discovery failed for %s: %s", agent_name, e)
                results["errors"].append({
                    "source": agent_name,
                    "error": str(e),
                    "stage": "discovery",
                })

        # Benchmark new solvers
        if self.auto_benchmark and results["new_solvers"]:
            logger.info("Auto-benchmarking %d new solvers", len(results["new_solvers"]))
            benchmark_results = await self._benchmark_solvers(
                [s["solver_id"] for s in results["new_solvers"]]
            )
            results["benchmarked_count"] = benchmark_results["benchmarked_count"]
            results["errors"].extend(benchmark_results["errors"])

        # Check adoption rules
        if self.auto_adopt:
            logger.info("Checking adoption rules for %d candidates", results["discovered_count"])
            adoption_results = self._check_adoptions()
            results["adopted_count"] = adoption_results["adopted_count"]
            results["adopted_solvers"] = adoption_results["adopted_solvers"]

        results["completed_at"] = datetime.now(timezone.utc).isoformat()
        logger.info(
            "Weekly discovery complete: %d discovered, %d benchmarked, %d adopted",
            results["discovered_count"],
            results["benchmarked_count"],
            results["adopted_count"],
        )

        return results

    async def _benchmark_solvers(self, solver_ids: list[str]) -> dict[str, Any]:
        """Benchmark a list of solvers.

        Args:
            solver_ids: List of solver IDs to benchmark

        Returns:
            Summary with counts and errors
        """
        results = {"benchmarked_count": 0, "errors": []}

        # Check if we have benchmarking integration
        if not hasattr(self, 'benchmark_integration') or self.benchmark_integration is None:
            logger.warning("No benchmark integration available - skipping benchmarking")
            return results

        for solver_id in solver_ids:
            solver = self.registry.get_solver(solver_id)
            if not solver:
                continue

            try:
                logger.info("Benchmarking solver %s", solver_id)

                # Run benchmarks via integration
                benchmark_results = await self.benchmark_integration.benchmark_solver(
                    solver_id=solver_id,
                    sample_limit=10,  # Quick eval
                    timeout_seconds=300.0,  # 5 min per solver
                )

                if benchmark_results:
                    # Update solver capabilities with results
                    await self.benchmark_integration.update_solver_capabilities(solver_id)
                    results["benchmarked_count"] += 1
                else:
                    logger.warning("No benchmark results for %s", solver_id)

            except Exception as e:
                logger.error("Benchmark failed for %s: %s", solver_id, e)
                results["errors"].append({
                    "solver_id": solver_id,
                    "error": str(e),
                    "stage": "benchmark",
                })

        return results

    def _check_adoptions(self) -> dict[str, Any]:
        """Check adoption rules for candidate solvers.

        Returns:
            Summary with adopted solvers
        """
        results = {"adopted_count": 0, "adopted_solvers": []}

        # Get all candidate solvers
        candidates = self.registry.list_solvers(adoption_status="candidate")

        for solver in candidates:
            solver_id = solver["solver_id"]

            # Check if solver meets any adoption rules
            matching_rules = self.registry.check_adoption_rules(solver_id)

            if matching_rules:
                rule_id = matching_rules[0]  # Use first matching rule
                logger.info(
                    "Adopting %s via rule %s",
                    solver["name"], rule_id
                )

                self.registry.update_adoption_status(
                    solver_id,
                    "adopted",
                    rule_id=rule_id,
                    notes=f"Auto-adopted by rule {rule_id} on {datetime.now(timezone.utc).isoformat()}",
                )

                results["adopted_count"] += 1
                results["adopted_solvers"].append({
                    "solver_id": solver_id,
                    "name": solver["name"],
                    "source": solver["source"],
                    "type": solver["solver_type"],
                    "rule_id": rule_id,
                })

        return results

    def add_default_adoption_rules(self) -> None:
        """Add sensible default adoption rules."""
        # High-quality LLMs
        self.registry.add_adoption_rule(AdoptionRule(
            rule_id="high-quality-llm",
            name="High Quality LLM",
            solver_type="llm",
            min_aggregate_score=0.80,
            required_benchmarks=["simple_reasoning", "mmlu"],
            min_benchmark_scores={"mmlu": 0.75, "simple_reasoning": 0.8},
            max_cost_per_call=0.10,
            max_latency_ms=5000,
            required_license=["apache-2.0", "mit", "llama2", "llama3"],
            excluded_sources=[],  # Accept all sources
            auto_register=False,  # Manual registration for now
            notify_user=True,
        ))

        # Fast inference models
        self.registry.add_adoption_rule(AdoptionRule(
            rule_id="fast-inference",
            name="Fast Inference Model",
            solver_type="llm",
            min_aggregate_score=0.60,  # Lower quality OK for speed
            max_latency_ms=500,  # Very fast
            max_cost_per_call=0.01,
            auto_register=False,
            notify_user=True,
        ))

        # Vision models
        self.registry.add_adoption_rule(AdoptionRule(
            rule_id="capable-vlm",
            name="Capable Vision Model",
            solver_type="vlm",
            min_aggregate_score=0.75,
            required_benchmarks=["image_classification"],
            min_benchmark_scores={"image_classification": 0.7},
            max_latency_ms=10000,
            auto_register=False,
            notify_user=True,
        ))

        # Local-only models (no cost, exclude BotVibes)
        self.registry.add_adoption_rule(AdoptionRule(
            rule_id="local-models",
            name="Local Models Only",
            solver_type="llm",
            min_aggregate_score=0.50,  # Accept lower quality for local
            max_cost_per_call=0.0,  # Must be free (local)
            excluded_sources=["botvibes"],  # Only local sources
            auto_register=False,
            notify_user=True,
        ))

        logger.info("Added 4 default adoption rules")


def create_discovery_job(
    registry_path: Path,
    *,
    auto_benchmark: bool = False,
    auto_adopt: bool = True,
    head_manager: Any = None,  # Optional HeadManager for benchmarking
    config: dict[str, Any] | None = None,  # Optional discovery_sources.yaml config
) -> DiscoveryCoordinator:
    """Factory function to create a discovery coordinator.

    Args:
        registry_path: Path to solver registry database
        auto_benchmark: Enable automatic benchmarking
        auto_adopt: Enable automatic adoption
        head_manager: Optional HeadManager for loading and benchmarking models
        config: Optional discovery config dict (from discovery_sources.yaml)

    Returns:
        Configured DiscoveryCoordinator
    """
    from multihead.benchmarking import (
        BenchmarkRunner,
        COCOBenchmark,
        GSM8KBenchmark,
        HumanEvalBenchmark,
        ImageClassificationBenchmark,
        LatencyBenchmark,
        MMLUBenchmark,
        SimpleReasoningBenchmark,
    )
    from multihead.discovery.botvibes import BotVibesDiscovery
    from multihead.discovery.huggingface import HuggingFaceDiscovery
    from multihead.discovery.ollama import OllamaDiscovery
    from multihead.discovery.paperswithcode import PapersWithCodeDiscovery

    sources_config = (config or {}).get("sources", {})

    # Create registry
    registry = SolverRegistry(registry_path)

    # Create benchmark runner with all benchmarks
    benchmark_runner = BenchmarkRunner()
    benchmark_runner.register_benchmark(SimpleReasoningBenchmark())
    benchmark_runner.register_benchmark(MMLUBenchmark())
    benchmark_runner.register_benchmark(GSM8KBenchmark())
    benchmark_runner.register_benchmark(HumanEvalBenchmark())
    benchmark_runner.register_benchmark(LatencyBenchmark())
    benchmark_runner.register_benchmark(ImageClassificationBenchmark())
    benchmark_runner.register_benchmark(COCOBenchmark())

    # Create discovery agents based on config
    discovery_agents: dict[str, DiscoveryAgent] = {}

    import os

    # HuggingFace
    hf_config = sources_config.get("huggingface", {})
    if hf_config.get("enabled", True):
        api_token = hf_config.get("api_token") or os.environ.get("HF_TOKEN")
        # Don't use unresolved env var placeholders as tokens
        if api_token and api_token.startswith("${"):
            api_token = None
        discovery_agents["huggingface"] = HuggingFaceDiscovery(api_token=api_token)

    # Ollama
    ollama_config = sources_config.get("ollama", {})
    if ollama_config.get("enabled", True):
        ollama_url = ollama_config.get("server_url") or os.environ.get("OLLAMA_URL", "http://localhost:11434")
        if ollama_url.startswith("${"):
            ollama_url = "http://localhost:11434"
        discovery_agents["ollama"] = OllamaDiscovery(ollama_url=ollama_url)

    # BotVibes
    bv_config = sources_config.get("botvibes", {})
    if bv_config.get("enabled", True):
        acp_url = bv_config.get("acp_url") or os.environ.get("ACP_URL")
        acp_token = bv_config.get("acp_token") or os.environ.get("ACP_SESSION_KEY") or os.environ.get("ACP_TOKEN")
        # Don't use unresolved env var placeholders
        if acp_url and not acp_url.startswith("${") and acp_token and not acp_token.startswith("${"):
            discovery_agents["botvibes"] = BotVibesDiscovery(acp_url, acp_token)

    # Papers with Code
    pwc_config = sources_config.get("paperswithcode", {})
    if pwc_config.get("enabled", True):
        pwc_tasks = pwc_config.get("tasks")
        pwc_datasets = pwc_config.get("datasets")
        discovery_agents["paperswithcode"] = PapersWithCodeDiscovery(
            tasks=pwc_tasks,
            datasets=pwc_datasets,
        )

    # Create coordinator
    coordinator = DiscoveryCoordinator(
        registry=registry,
        benchmark_runner=benchmark_runner,
        discovery_agents=discovery_agents,
        auto_benchmark=auto_benchmark,
        auto_adopt=auto_adopt,
    )

    # Add benchmarking integration if HeadManager provided
    if head_manager and auto_benchmark:
        from multihead.benchmarking.integration import BenchmarkingIntegration

        coordinator.benchmark_integration = BenchmarkingIntegration(
            head_manager=head_manager,
            benchmark_runner=benchmark_runner,
            registry=registry,
        )
        logger.info("Benchmarking integration enabled with HeadManager")
    else:
        coordinator.benchmark_integration = None
        if auto_benchmark:
            logger.warning(
                "auto_benchmark=True but no HeadManager provided - "
                "benchmarking will be skipped"
            )

    # Add default adoption rules
    coordinator.add_default_adoption_rules()

    return coordinator


def load_discovery_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load discovery configuration from YAML file.

    Args:
        config_path: Path to discovery_sources.yaml. If None,
            searches config/ directory.

    Returns:
        Config dict (empty dict if file not found)
    """
    import yaml

    if config_path is None:
        # Search standard locations
        candidates = [
            Path("config/discovery_sources.yaml"),
            Path(__file__).parent.parent.parent / "config" / "discovery_sources.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = candidate
                break

    if config_path is None or not config_path.exists():
        logger.debug("No discovery config found, using defaults")
        return {}

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
        logger.info("Loaded discovery config from %s", config_path)
        return config
    except Exception as e:
        logger.error("Failed to load discovery config: %s", e)
        return {}
