"""Ollama library discovery agent."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import httpx

from multihead.discovery.base import DiscoveryAgent, SolverCandidate

logger = logging.getLogger(__name__)


class OllamaDiscovery(DiscoveryAgent):
    """Discovers models from Ollama library."""

    # Model size to capability mapping (heuristic)
    SIZE_TO_CAPABILITY = {
        # Small models (< 10B params)
        "small": {
            "task_types": ["text_generation", "reasoning", "code_generation"],
            "estimated_latency_ms": 500,
        },
        # Medium models (10-30B params)
        "medium": {
            "task_types": ["text_generation", "reasoning", "code_generation", "complex_reasoning"],
            "estimated_latency_ms": 2000,
        },
        # Large models (> 30B params)
        "large": {
            "task_types": ["text_generation", "reasoning", "code_generation", "complex_reasoning", "research"],
            "estimated_latency_ms": 5000,
        },
    }

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        """Initialize Ollama discovery agent.

        Args:
            ollama_url: Ollama server URL
        """
        super().__init__(source_name="ollama")
        self.ollama_url = ollama_url.rstrip("/")

    async def discover_new_solvers(
        self,
        *,
        solver_types: list[str] | None = None,
        min_downloads: int | None = None,
        updated_since: datetime | None = None,
        limit: int = 50,
    ) -> list[SolverCandidate]:
        """Discover models from Ollama library.

        Queries the Ollama library for available models.

        Args:
            solver_types: Filter by solver types (only LLM/VLM supported)
            min_downloads: Not applicable for Ollama
            updated_since: Not applicable for Ollama
            limit: Maximum results

        Returns:
            List of discovered solver candidates
        """
        # Filter: Ollama only has LLM and VLM models
        if solver_types and not any(st in ("llm", "vlm") for st in solver_types):
            return []

        candidates = []

        try:
            # Try Ollama library API first (public registry)
            candidates = await self._discover_from_library(limit=limit)

            # Fallback: Query local Ollama server
            if not candidates:
                candidates = await self._discover_from_local()

        except Exception as e:
            logger.error("Ollama discovery error: %s", e)

        logger.info("Ollama discovery: found %d candidates", len(candidates))
        return candidates

    async def _discover_from_library(self, limit: int = 50) -> list[SolverCandidate]:
        """Discover models from Ollama public library.

        Queries the local Ollama server's model list first, then falls back
        to a curated list of popular models for discovery suggestions.

        Args:
            limit: Maximum results

        Returns:
            List of solver candidates
        """
        candidates = []

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                # Try the Ollama API to list available (pulled) models
                resp = await client.get(f"{self.ollama_url}/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    models = data.get("models", [])
                    for model in models[:limit]:
                        name = model.get("name", "")
                        if not name:
                            continue
                        base_name = name.split(":")[0]
                        tag = name.split(":")[1] if ":" in name else "latest"
                        model_info = {
                            "name": base_name,
                            "tag": tag,
                            "size": model.get("size", 0),
                            "modified_at": model.get("modified_at"),
                        }
                        candidate = self._model_to_candidate(model_info, from_local=True)
                        if candidate:
                            candidates.append(candidate)

                    if candidates:
                        return candidates[:limit]

        except (httpx.HTTPError, httpx.ConnectError):
            logger.debug("Ollama server not reachable, using curated list")
        except Exception as e:
            logger.warning("Ollama library discovery failed: %s", e)

        # Fallback: curated list of popular models for discovery suggestions
        popular_models = [
            {"name": "llama3.2", "size": "3b", "type": "llm"},
            {"name": "llama3.2", "size": "1b", "type": "llm"},
            {"name": "llama3.1", "size": "8b", "type": "llm"},
            {"name": "llama3.1", "size": "70b", "type": "llm"},
            {"name": "qwen2.5", "size": "7b", "type": "llm"},
            {"name": "qwen2.5", "size": "14b", "type": "llm"},
            {"name": "qwen3", "size": "8b", "type": "llm"},
            {"name": "phi4", "size": "14b", "type": "llm"},
            {"name": "mistral", "size": "7b", "type": "llm"},
            {"name": "gemma3", "size": "12b", "type": "llm"},
            {"name": "codestral", "size": "22b", "type": "llm"},
            {"name": "llava", "size": "7b", "type": "vlm"},
            {"name": "llava", "size": "13b", "type": "vlm"},
            {"name": "llava-llama3", "size": "8b", "type": "vlm"},
        ]

        for model_info in popular_models[:limit]:
            candidate = self._model_to_candidate(model_info)
            if candidate:
                candidates.append(candidate)

        return candidates

    async def _discover_from_local(self) -> list[SolverCandidate]:
        """Discover models from local Ollama server.

        Queries the local Ollama instance for installed models.

        Returns:
            List of solver candidates from local installation
        """
        candidates = []

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.ollama_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()

                models = data.get("models", [])
                for model in models:
                    # Extract model info
                    name = model.get("name", "")
                    if not name:
                        continue

                    # Parse name (format: "model_name:tag")
                    base_name = name.split(":")[0]
                    tag = name.split(":")[1] if ":" in name else "latest"

                    model_info = {
                        "name": base_name,
                        "tag": tag,
                        "size": model.get("size", 0),
                        "modified_at": model.get("modified_at"),
                    }

                    candidate = self._model_to_candidate(model_info, from_local=True)
                    if candidate:
                        candidates.append(candidate)

        except httpx.HTTPError as e:
            logger.warning("Local Ollama query failed: %s", e)
        except Exception as e:
            logger.error("Error querying local Ollama: %s", e)

        return candidates

    async def get_solver_details(self, solver_id: str) -> SolverCandidate | None:
        """Get detailed information about an Ollama model.

        Args:
            solver_id: Ollama model name (e.g., "ollama-llama3.1-8b")

        Returns:
            Detailed solver candidate or None if not found
        """
        # Extract model name from solver_id
        if solver_id.startswith("ollama-"):
            model_name = solver_id[7:]  # Remove "ollama-" prefix
        else:
            model_name = solver_id

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self.ollama_url}/api/show",
                    json={"name": model_name},
                )
                resp.raise_for_status()
                data = resp.json()

                # Extract model info from response
                model_info = {
                    "name": model_name.split(":")[0],
                    "tag": model_name.split(":")[1] if ":" in model_name else "latest",
                    "size": data.get("size", 0),
                    "template": data.get("template"),
                    "parameters": data.get("parameters"),
                }

                return self._model_to_candidate(model_info, from_local=True)

        except httpx.HTTPError as e:
            logger.warning("Failed to get Ollama model %s: %s", solver_id, e)
            return None
        except Exception as e:
            logger.error("Error getting Ollama model %s: %s", solver_id, e)
            return None

    def _model_to_candidate(
        self,
        model: dict[str, Any],
        from_local: bool = False
    ) -> SolverCandidate | None:
        """Convert Ollama model dict to SolverCandidate.

        Args:
            model: Model data from Ollama
            from_local: Whether model is from local Ollama server

        Returns:
            SolverCandidate or None if model should be filtered out
        """
        name = model.get("name", "")
        if not name:
            return None

        # Determine size category
        size_str = model.get("size", "")
        if isinstance(size_str, str):
            size_lower = size_str.lower()
            if any(s in size_lower for s in ("1b", "3b", "7b")):
                size_category = "small"
            elif any(s in size_lower for s in ("8b", "13b", "14b")):
                size_category = "medium"
            else:
                size_category = "large"
        else:
            # Infer from model name
            if any(s in name.lower() for s in ("1b", "3b", "7b")):
                size_category = "small"
            elif any(s in name.lower() for s in ("8b", "13b", "14b")):
                size_category = "medium"
            else:
                size_category = "medium"  # Default

        # Get capabilities for size
        capabilities = self.SIZE_TO_CAPABILITY.get(size_category, self.SIZE_TO_CAPABILITY["medium"])

        # Determine solver type
        solver_type = model.get("type", "llm")
        if "llava" in name.lower() or "vision" in name.lower():
            solver_type = "vlm"

        # Build modalities
        modalities = ["text"]
        if solver_type == "vlm":
            modalities.append("image")

        # Build solver_id
        tag = model.get("tag", "latest")
        solver_id = f"ollama-{name}-{tag}" if tag != "latest" else f"ollama-{name}"

        # Build tags
        tags = ["ollama"]
        if size_str:
            tags.append(size_str)
        if from_local:
            tags.append("local")

        return SolverCandidate(
            solver_id=solver_id,
            name=f"{name}:{tag}",
            source=self.source_name,
            solver_type=solver_type,
            task_types=capabilities["task_types"],
            modalities=modalities,
            estimated_latency_ms=capabilities["estimated_latency_ms"],
            estimated_cost=0.0,  # Local models are free
            model_id=f"{name}:{tag}",
            version=tag,
            url=f"https://ollama.ai/library/{name}",
            tags=tags,
            discovery_metadata={
                "size": size_str,
                "from_local": from_local,
                "size_category": size_category,
            },
        )
