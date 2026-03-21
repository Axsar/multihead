"""HuggingFace Model Hub discovery agent."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import httpx

from multihead.discovery.base import DiscoveryAgent, SolverCandidate

logger = logging.getLogger(__name__)


class HuggingFaceDiscovery(DiscoveryAgent):
    """Discovers models from HuggingFace Model Hub."""

    # Task type mapping: HF pipeline tags → MultiHead task types
    TASK_TYPE_MAP = {
        # LLM tasks
        "text-generation": ["text_generation", "reasoning", "code_generation"],
        "text2text-generation": ["text_generation", "translation"],
        "conversational": ["conversation", "reasoning"],
        "question-answering": ["question_answering", "reasoning"],

        # Vision tasks
        "image-classification": ["image_classification"],
        "object-detection": ["object_detection"],
        "image-segmentation": ["segmentation"],
        "image-to-text": ["image_captioning", "visual_reasoning"],
        "visual-question-answering": ["visual_reasoning"],

        # Multimodal
        "zero-shot-image-classification": ["image_classification", "visual_reasoning"],

        # Embedding
        "feature-extraction": ["embedding"],
        "sentence-similarity": ["embedding", "semantic_search"],
    }

    # Solver type inference from pipeline tag
    SOLVER_TYPE_MAP = {
        "text-generation": "llm",
        "conversational": "llm",
        "question-answering": "llm",
        "text2text-generation": "llm",
        "object-detection": "object_detection",
        "image-segmentation": "segmentation",
        "image-classification": "vlm",
        "image-to-text": "vlm",
        "visual-question-answering": "vlm",
        "feature-extraction": "embedding",
        "sentence-similarity": "embedding",
    }

    def __init__(self, api_token: str | None = None):
        """Initialize HuggingFace discovery agent.

        Args:
            api_token: Optional HuggingFace API token for rate limits
        """
        super().__init__(source_name="huggingface")
        self.api_token = api_token
        self.api_base = "https://huggingface.co/api"

    async def discover_new_solvers(
        self,
        *,
        solver_types: list[str] | None = None,
        min_downloads: int | None = None,
        updated_since: datetime | None = None,
        limit: int = 50,
    ) -> list[SolverCandidate]:
        """Discover new models from HuggingFace.

        Queries the HuggingFace API for popular models matching criteria.

        Args:
            solver_types: Filter by solver types
            min_downloads: Minimum download count
            updated_since: Only models updated after this date
            limit: Maximum results

        Returns:
            List of discovered solver candidates
        """
        # Map solver types to HF pipeline tags
        pipeline_tags = []
        if solver_types:
            for solver_type in solver_types:
                # Reverse lookup in SOLVER_TYPE_MAP
                tags = [
                    tag for tag, st in self.SOLVER_TYPE_MAP.items()
                    if st == solver_type
                ]
                pipeline_tags.extend(tags)

        candidates = []

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Query HuggingFace models API
                params: dict[str, Any] = {
                    "sort": "downloads",
                    "direction": -1,
                    "limit": limit,
                }

                if pipeline_tags:
                    params["filter"] = ",".join(pipeline_tags)

                headers = {}
                if self.api_token:
                    headers["Authorization"] = f"Bearer {self.api_token}"

                resp = await client.get(
                    f"{self.api_base}/models",
                    params=params,
                    headers=headers,
                )
                resp.raise_for_status()
                models = resp.json()

                for model in models:
                    # Apply filters
                    if min_downloads and model.get("downloads", 0) < min_downloads:
                        continue

                    if updated_since:
                        last_modified = model.get("lastModified")
                        if last_modified:
                            model_date = datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
                            if model_date < updated_since:
                                continue

                    # Convert to SolverCandidate
                    candidate = self._model_to_candidate(model)
                    if candidate:
                        candidates.append(candidate)

        except httpx.HTTPError as e:
            logger.error("HuggingFace discovery failed: %s", e)
        except Exception as e:
            logger.error("HuggingFace discovery error: %s", e)

        logger.info("HuggingFace discovery: found %d candidates", len(candidates))
        return candidates

    async def get_solver_details(self, solver_id: str) -> SolverCandidate | None:
        """Get detailed information about a HuggingFace model.

        Args:
            solver_id: HuggingFace model ID (e.g., "meta-llama/Llama-2-7b")

        Returns:
            Detailed solver candidate or None if not found
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                headers = {}
                if self.api_token:
                    headers["Authorization"] = f"Bearer {self.api_token}"

                resp = await client.get(
                    f"{self.api_base}/models/{solver_id}",
                    headers=headers,
                )
                resp.raise_for_status()
                model = resp.json()

                return self._model_to_candidate(model)

        except httpx.HTTPError as e:
            logger.warning("Failed to get HuggingFace model %s: %s", solver_id, e)
            return None
        except Exception as e:
            logger.error("Error getting HuggingFace model %s: %s", solver_id, e)
            return None

    def _model_to_candidate(self, model: dict[str, Any]) -> SolverCandidate | None:
        """Convert HuggingFace model dict to SolverCandidate.

        Args:
            model: Model data from HuggingFace API

        Returns:
            SolverCandidate or None if model should be filtered out
        """
        model_id = model.get("id", "")
        if not model_id:
            return None

        # Infer solver type from pipeline tag
        pipeline_tag = model.get("pipeline_tag")
        solver_type = self.SOLVER_TYPE_MAP.get(pipeline_tag, "llm")

        # Get task types
        task_types = []
        if pipeline_tag:
            task_types = self.TASK_TYPE_MAP.get(pipeline_tag, [])

        # Infer modalities
        modalities = ["text"]  # Default
        if solver_type in ("vlm", "object_detection", "segmentation"):
            modalities.append("image")

        # Extract tags
        tags = model.get("tags", [])

        # Get license
        license_info = None
        card_data = model.get("cardData", {})
        if card_data:
            license_info = card_data.get("license")

        # Generate solver_id
        solver_id = f"hf-{model_id.replace('/', '-')}"

        return SolverCandidate(
            solver_id=solver_id,
            name=model_id.split("/")[-1] if "/" in model_id else model_id,
            source=self.source_name,
            solver_type=solver_type,
            task_types=task_types,
            modalities=modalities,
            model_id=model_id,
            license=license_info,
            description=card_data.get("description"),
            url=f"https://huggingface.co/{model_id}",
            tags=tags,
            discovery_metadata={
                "downloads": model.get("downloads", 0),
                "likes": model.get("likes", 0),
                "last_modified": model.get("lastModified"),
                "pipeline_tag": pipeline_tag,
                "library_name": model.get("library_name"),
            },
        )
