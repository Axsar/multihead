"""BotVibes marketplace discovery agent."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import httpx

from multihead.discovery.base import DiscoveryAgent, SolverCandidate

logger = logging.getLogger(__name__)


class BotVibesDiscovery(DiscoveryAgent):
    """Discovers providers from BotVibes marketplace."""

    def __init__(self, acp_url: str, acp_token: str):
        """Initialize BotVibes discovery agent.

        Args:
            acp_url: BotVibes ACP server URL
            acp_token: Authentication token
        """
        super().__init__(source_name="botvibes")
        self.acp_url = acp_url.rstrip("/")
        self.acp_token = acp_token

    async def discover_new_solvers(
        self,
        *,
        solver_types: list[str] | None = None,
        min_downloads: int | None = None,
        updated_since: datetime | None = None,
        limit: int = 50,
    ) -> list[SolverCandidate]:
        """Discover providers from BotVibes marketplace.

        Args:
            solver_types: Filter by solver types
            min_downloads: Not applicable for BotVibes
            updated_since: Not applicable for BotVibes
            limit: Maximum results

        Returns:
            List of discovered solver candidates
        """
        candidates = []

        # Discover for each solver type
        capabilities_to_query = self._solver_types_to_capabilities(solver_types)

        for capability in capabilities_to_query:
            try:
                providers = await self._discover_by_capability(
                    capability=capability,
                    limit=limit
                )
                candidates.extend(providers)

            except Exception as e:
                logger.error("BotVibes discovery failed for %s: %s", capability, e)

        # Deduplicate by solver_id
        seen = set()
        unique_candidates = []
        for candidate in candidates:
            if candidate.solver_id not in seen:
                seen.add(candidate.solver_id)
                unique_candidates.append(candidate)

        logger.info("BotVibes discovery: found %d unique candidates", len(unique_candidates))
        return unique_candidates[:limit]

    async def _discover_by_capability(
        self,
        capability: str,
        limit: int
    ) -> list[SolverCandidate]:
        """Discover providers for a specific capability.

        Args:
            capability: Capability to query for
            limit: Maximum results

        Returns:
            List of solver candidates
        """
        candidates = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{self.acp_url}/api/v1/marketplace/listings/search",
                headers={"Authorization": f"Bearer {self.acp_token}"},
                params={
                    "capability_id": capability,
                    "cross_tenant": "true",
                    "limit": limit,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", [])
            for result in results:
                candidate = self._listing_to_candidate(result, capability)
                if candidate:
                    candidates.append(candidate)

        return candidates

    async def get_solver_details(self, solver_id: str) -> SolverCandidate | None:
        """Get detailed information about a BotVibes provider.

        Args:
            solver_id: BotVibes provider ID (format: "botvibes-{agent_id}")

        Returns:
            Detailed solver candidate or None if not found
        """
        # Extract agent_id from solver_id
        if solver_id.startswith("botvibes-"):
            agent_id = solver_id[9:]  # Remove "botvibes-" prefix
        else:
            agent_id = solver_id

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Query listings for this specific agent
                resp = await client.get(
                    f"{self.acp_url}/api/v1/marketplace/listings/search",
                    headers={"Authorization": f"Bearer {self.acp_token}"},
                    params={
                        "agent_id": agent_id,
                        "cross_tenant": "true",
                        "limit": 1,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

                results = data.get("results", [])
                if not results:
                    return None

                # Get primary capability from first listing
                listing = results[0].get("listing", {})
                capability = listing.get("capability_id", "unknown")

                return self._listing_to_candidate(results[0], capability)

        except httpx.HTTPError as e:
            logger.warning("Failed to get BotVibes provider %s: %s", solver_id, e)
            return None
        except Exception as e:
            logger.error("Error getting BotVibes provider %s: %s", solver_id, e)
            return None

    def _listing_to_candidate(
        self,
        result: dict[str, Any],
        capability: str
    ) -> SolverCandidate | None:
        """Convert BotVibes listing to SolverCandidate.

        Args:
            result: Result from marketplace listings/search
            capability: Capability being queried

        Returns:
            SolverCandidate or None if invalid
        """
        listing = result.get("listing", {})
        stats = result.get("stats", {})
        scoring = result.get("scoring", {})

        agent_id = listing.get("agent_id", "")
        if not agent_id:
            return None

        # Infer solver type from capability
        solver_type = self._capability_to_solver_type(capability)

        # Build task types from capability
        task_types = [capability]

        # Infer modalities
        modalities = ["text"]  # Default
        if solver_type in ("vlm", "object_detection", "segmentation"):
            modalities.append("image")

        # Extract performance metrics
        quality_score = stats.get("quality_score", 0.0)
        latency_ms = stats.get("ewma_latency_ms", listing.get("sla_p95_ms", 0))
        cost = listing.get("unit_price", 0.0)

        # Build solver_id
        solver_id = f"botvibes-{agent_id}"

        # Build tags
        tags = ["botvibes", "marketplace"]
        privacy_level = listing.get("privacy_level", "external")
        tags.append(privacy_level)

        return SolverCandidate(
            solver_id=solver_id,
            name=listing.get("name", agent_id),
            source=self.source_name,
            solver_type=solver_type,
            task_types=task_types,
            modalities=modalities,
            benchmark_scores={"quality_score": quality_score},
            estimated_latency_ms=int(latency_ms),
            estimated_cost=cost,
            model_id=agent_id,
            description=listing.get("description"),
            tags=tags,
            discovery_metadata={
                "listing_id": listing.get("listing_id"),
                "capability": capability,
                "dispute_rate": stats.get("dispute_rate", 0.0),
                "accept_rate": stats.get("accept_rate", 1.0),
                "total_score": scoring.get("total_score", 0.0),
                "sla_p95_ms": listing.get("sla_p95_ms"),
                "privacy_level": privacy_level,
            },
        )

    def _solver_types_to_capabilities(
        self,
        solver_types: list[str] | None
    ) -> list[str]:
        """Convert solver types to BotVibes capabilities.

        Args:
            solver_types: List of solver types or None for all

        Returns:
            List of capability IDs to query
        """
        # Map solver types to BotVibes capabilities
        mapping = {
            "llm": ["text_generation", "reasoning", "code_generation"],
            "vlm": ["visual_reasoning", "image_analysis", "image_captioning"],
            "object_detection": ["object_detection"],
            "segmentation": ["segmentation", "image_segmentation"],
            "embedding": ["embedding", "semantic_search"],
            "external_service": ["custom_service"],
        }

        if not solver_types:
            # Return common capabilities
            return [
                "visual_reasoning",
                "text_generation",
                "reasoning",
                "code_generation",
                "object_detection",
            ]

        capabilities = []
        for solver_type in solver_types:
            caps = mapping.get(solver_type, [solver_type])
            capabilities.extend(caps)

        return list(set(capabilities))  # Deduplicate

    def _capability_to_solver_type(self, capability: str) -> str:
        """Infer solver type from BotVibes capability.

        Args:
            capability: BotVibes capability ID

        Returns:
            MultiHead solver type
        """
        # Reverse mapping from capabilities to solver types
        if capability in ("visual_reasoning", "image_analysis", "image_captioning"):
            return "vlm"
        elif capability in ("object_detection",):
            return "object_detection"
        elif capability in ("segmentation", "image_segmentation"):
            return "segmentation"
        elif capability in ("embedding", "semantic_search"):
            return "embedding"
        elif capability in ("text_generation", "reasoning", "code_generation"):
            return "llm"
        else:
            return "external_service"  # Default
