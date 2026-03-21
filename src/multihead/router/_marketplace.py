"""BotVibes marketplace integration mixin for the Router (Phase 3)."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from ..models import HeadManifest

logger = logging.getLogger(__name__)


class MarketplaceMixin:
    """BotVibes marketplace methods for the Router.

    Requires: self.heads, self.route_with_discovery, self._provider_cache,
              self._provider_cache_ttl_s
    """

    def _get_cached_providers(self, capability: str) -> list[dict[str, Any]] | None:
        """Return cached providers if still valid, else None."""
        import time as _time
        entry = self._provider_cache.get(capability)
        if entry is None:
            return None
        providers, ts = entry
        if _time.time() - ts > self._provider_cache_ttl_s:
            del self._provider_cache[capability]
            return None
        return providers

    def _set_cached_providers(self, capability: str, providers: list[dict[str, Any]]) -> None:
        """Cache providers for a capability."""
        import time as _time
        self._provider_cache[capability] = (providers, _time.time())

    def invalidate_provider_cache(self, capability: str | None = None) -> None:
        """Invalidate cached providers. Pass None to clear all."""
        if capability is None:
            self._provider_cache.clear()
        else:
            self._provider_cache.pop(capability, None)

    @staticmethod
    def _infer_kind_from_capabilities(capabilities: list[str]) -> str:
        """Infer head kind from a provider's capability list."""
        visual = {"visual_reasoning", "image_analysis", "image_captioning",
                  "image_classification", "image_description",
                  "visual_question_answering", "complex_image_analysis"}
        detection = {"object_detection", "image.detect.objects.v1"}
        segmentation = {"segmentation", "image.segment.masks.v1"}
        code = {"code.review.v1", "code.generation.v1"}

        cap_set = set(capabilities)
        if cap_set & visual:
            return "vlm"
        if cap_set & detection or cap_set & segmentation:
            return "tool"
        if cap_set & code:
            return "llm"
        # Default: if any caps contain "image" or "vision", it's a VLM
        for c in capabilities:
            if "image" in c or "vision" in c or "visual" in c:
                return "vlm"
        return "llm"

    async def route_with_marketplace_fallback(
        self,
        capability: str,
        *,
        task_types: list[str] | None = None,
        privacy: Any = None,
        exclude: set[str] | None = None,
        acp_url: str | None = None,
        acp_token: str | None = None,
        max_cost: float = 0.50,
        max_latency_ms: int = 10_000,
        min_reputation: float = 0.80,
    ) -> str | None:
        """Route locally first; if no local head matches, discover marketplace providers.

        1. Try route_with_discovery() for local heads
        2. If no local match, call discover_botvibes_providers()
        3. Register best marketplace provider as a temporary head
        4. Return head_id (either local or botvibes-*)

        Args:
            capability: Capability query (e.g. "visual_reasoning", "text_generation")
            task_types: Task types for local routing fallback
            privacy: Privacy constraints (CONFIDENTIAL blocks marketplace)
            exclude: Head IDs to skip
            acp_url: BotVibes ACP URL (default: from env)
            acp_token: ACP token (default: from env)
            max_cost: Maximum cost per call for marketplace ($)
            max_latency_ms: Maximum latency for marketplace (ms)
            min_reputation: Minimum provider reputation (0-1)

        Returns:
            head_id (local or botvibes-*) or None
        """
        # 1) Try local routing
        local = self.route_with_discovery(
            capability, task_types=task_types, privacy=privacy, exclude=exclude,
        )
        if local:
            return local

        # 2) Privacy gate: CONFIDENTIAL cannot leave local
        if privacy:
            from ..models import DataSensitivity
            if hasattr(privacy, "data_sensitivity") and privacy.data_sensitivity == DataSensitivity.CONFIDENTIAL:
                logger.info("CONFIDENTIAL data — marketplace discovery skipped")
                return None

        # 3) Check provider cache before hitting API
        cached = self._get_cached_providers(capability)
        if cached is not None:
            providers = cached
            logger.info("Using cached providers for capability=%s (%d)", capability, len(providers))
        else:
            logger.info("No local head for capability=%s, querying BotVibes marketplace", capability)
            providers = await self.discover_botvibes_providers(
                capability,
                acp_url=acp_url,
                acp_token=acp_token,
                min_reputation=min_reputation,
                max_cost=max_cost,
                max_latency_ms=max_latency_ms,
                limit=5,
            )
            # Cache the result (even if empty, to avoid repeated API calls)
            self._set_cached_providers(capability, providers)

        if not providers:
            logger.info("No marketplace providers found for capability=%s", capability)
            return None

        # 4) Register best provider as temporary head
        best = providers[0]
        manifest = self._provider_to_manifest(
            best,
            acp_url=acp_url or os.environ.get("ACP_URL", ""),
            acp_token=acp_token or os.environ.get("ACP_SESSION_KEY", ""),
        )

        head_id = manifest.head_id
        if head_id not in self.heads.get_states():
            from ..adapters.botvibes_adapter import BotVibesAdapter
            from ..resilience import CircuitBreaker
            from ..head_manager import HeadState
            self.heads._manifests[head_id] = manifest
            self.heads._adapters[head_id] = BotVibesAdapter(manifest)
            self.heads._states[head_id] = HeadState.OFF
            self.heads._breakers[head_id] = CircuitBreaker(5, 60.0)
            logger.info(
                "Registered marketplace provider %s (rep=%.2f, cost=$%.2f, latency=%dms)",
                head_id, best.get("reputation", 0), best.get("cost_per_call", 0),
                best.get("latency_p50_ms", 0),
            )

        return head_id

    async def discover_botvibes_providers(
        self,
        capability: str,
        *,
        acp_url: str | None = None,
        acp_token: str | None = None,
        min_reputation: float = 0.0,
        max_cost: float | None = None,
        max_latency_ms: int | None = None,
        privacy_levels: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Query BotVibes marketplace for providers matching criteria.

        This enables dynamic provider discovery - MultiHead can query the
        marketplace at runtime to find external solvers for specific tasks.

        Args:
            capability: Required capability filter (e.g., "visual_reasoning")
            acp_url: BotVibes ACP server URL (default: from env)
            acp_token: Authentication token (default: from env)
            min_reputation: Minimum provider reputation (0.0-1.0)
            max_cost: Maximum cost per call ($)
            max_latency_ms: Maximum acceptable latency (ms)
            privacy_levels: Allowed privacy levels (e.g., ["encrypted"])
            limit: Maximum providers to return

        Returns:
            List of provider dicts that can be converted to HeadManifests.
            Each dict contains:
            - provider_id: Provider's agent ID
            - provider_name: Display name
            - capabilities: List of capabilities
            - reputation: Score 0.0-1.0
            - cost_per_call: Price in USD
            - latency_p50_ms: Median latency
            - privacy_level: "encrypted" or "external"
            - metadata: Additional provider info

        Example:
            >>> providers = await router.discover_botvibes_providers(
            ...     capability="visual_reasoning",
            ...     min_reputation=0.85,
            ...     max_cost=0.50,
            ... )
            >>> # Convert to HeadManifests and add to solver registry
            >>> for p in providers:
            ...     manifest = router._provider_to_manifest(p, acp_url, acp_token)
            ...     # Register dynamically in head_manager
        """
        import os

        # Get ACP config from env if not provided
        if not acp_url:
            acp_url = os.getenv("ACP_URL", "http://localhost:8000")
        if not acp_token:
            acp_token = os.getenv("ACP_SESSION_KEY", "")

        # Strip /api/v1 suffix and re-add
        acp_url = acp_url.rstrip("/")
        for suffix in ("/api/v1", "/api/v1/", "/api"):
            if acp_url.endswith(suffix):
                acp_url = acp_url[: -len(suffix)]
                break
        acp_url += "/api/v1"

        if not acp_token:
            logger.warning("No ACP token provided, marketplace discovery unavailable")
            return []

        logger.info(
            "BotVibes marketplace discovery: capability=%s, filters=(rep>=%.2f, cost<=%.2f, latency<=%s)",
            capability, min_reputation, max_cost or float("inf"), max_latency_ms or "∞"
        )

        # Query BotVibes marketplace API
        # Endpoint: GET /api/v1/marketplace/listings/search
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                params = {
                    "capability_id": capability,
                    "cross_tenant": "true",
                    "limit": limit,
                }
                if max_cost is not None:
                    params["max_price"] = max_cost

                resp = await client.get(
                    f"{acp_url}/marketplace/listings/search",
                    headers={"Authorization": f"Bearer {acp_token}"},
                    params=params,
                )
                resp.raise_for_status()
                data = resp.json()

                # Parse response: results[].listing + results[].stats
                results = data.get("results", [])
                providers = []

                for result in results:
                    listing = result.get("listing", {})
                    stats = result.get("stats", {})
                    scoring = result.get("scoring", {})

                    # Extract and map fields per BotVibes API spec
                    provider = {
                        "provider_id": listing.get("agent_id", ""),
                        "provider_name": listing.get("name", listing.get("agent_id", "")),
                        "capabilities": [capability],  # Matched capability
                        "reputation": stats.get("quality_score", 0.0),  # 0-1 score
                        "cost_per_call": listing.get("unit_price", 0.0),
                        "latency_p50_ms": int(stats.get("ewma_latency_ms", listing.get("sla_p95_ms", 0))),
                        "privacy_level": "encrypted",  # BotVibes uses HTTPS, assume encrypted
                        "metadata": {
                            "dispute_rate": stats.get("dispute_rate", 0.0),
                            "accept_rate": stats.get("accept_rate", 1.0),
                            "total_score": scoring.get("total_score", 0.0),
                            "listing_id": listing.get("listing_id", ""),
                            "sla_p95_ms": listing.get("sla_p95_ms"),
                        },
                    }

                    # Apply client-side filters
                    if provider["reputation"] < min_reputation:
                        continue
                    if max_cost is not None and provider["cost_per_call"] > max_cost:
                        continue
                    if max_latency_ms is not None and provider["latency_p50_ms"] > max_latency_ms:
                        continue
                    if privacy_levels and provider["privacy_level"] not in privacy_levels:
                        continue

                    providers.append(provider)

                logger.info(
                    "Marketplace discovery returned %d providers (filtered from %d results)",
                    len(providers), len(results)
                )
                return providers

        except httpx.HTTPError as e:
            logger.error("BotVibes marketplace query failed: %s", e)
            return []
        except Exception as e:
            logger.error("Marketplace discovery error: %s", e)
            return []

    def _provider_to_manifest(
        self,
        provider: dict[str, Any],
        acp_url: str,
        acp_token: str,
        project_id: str | None = None,
    ) -> "HeadManifest":
        """Convert a BotVibes provider dict to a HeadManifest.

        This allows discovered providers to be registered dynamically
        in the head manager and used like local solvers.

        Args:
            provider: Provider info from marketplace discovery
            acp_url: ACP server URL
            acp_token: Authentication token
            project_id: ACP project ID

        Returns:
            HeadManifest configured for the BotVibes provider

        Example provider dict:
            {
                "provider_id": "vision-expert-agent-123",
                "provider_name": "Vision Analysis Expert",
                "capabilities": ["visual_reasoning", "image_classification"],
                "reputation": 0.94,
                "cost_per_call": 0.08,
                "latency_p50_ms": 4500,
                "privacy_level": "encrypted",
                "metadata": {
                    "queue_depth_avg": 2,
                    "success_rate": 0.98,
                    "uptime_pct": 99.5
                }
            }
        """
        from ..models import AdapterKind, Capability, HeadManifest

        solver_id = f"botvibes-{provider['provider_id']}"

        # Convert provider capabilities to our Capability model
        capability = Capability(
            solver_type="external_service",
            input_modalities=["text", "image"],  # TODO: from provider metadata
            output_modalities=["text", "json"],
            task_types=provider.get("capabilities", []),
            latency_p50_ms=provider.get("latency_p50_ms", 5000),
            cost_per_call=provider.get("cost_per_call", 0.0),
            accuracy_score=provider.get("reputation", 0.0),
        )

        # Create manifest
        manifest = HeadManifest(
            head_id=solver_id,
            name=provider.get("provider_name", solver_id),
            adapter=AdapterKind.BOTVIBES,
            model=provider.get("provider_id", "unknown"),
            kind=self._infer_kind_from_capabilities(provider.get("capabilities", [])),
            endpoint=acp_url,
            gpu_required=False,
            is_local=False,
            privacy_level=provider.get("privacy_level", "encrypted"),
            extra={
                "api_key": acp_token,
                "project_id": project_id or "",
                "target_capability": provider.get("capabilities", [""])[0],
                "target_agent_id": provider.get("provider_id"),
                "provider_metadata": provider.get("metadata", {}),
            },
            capabilities=capability,
        )

        return manifest
