"""Papers with Code SOTA tracking discovery agent."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import httpx

from multihead.discovery.base import DiscoveryAgent, SolverCandidate

logger = logging.getLogger(__name__)


class PapersWithCodeDiscovery(DiscoveryAgent):
    """Discovers state-of-the-art models from Papers with Code.

    Tracks SOTA results for tasks like object detection, segmentation,
    and text generation, and surfaces new models that beat existing
    benchmarks.
    """

    API_BASE = "https://paperswithcode.com/api/v1"

    # Map PwC task slugs to MultiHead solver types and task types
    TASK_MAP = {
        "object-detection": {
            "solver_type": "object_detection",
            "task_types": ["object_detection"],
            "modalities": ["text", "image"],
        },
        "image-segmentation": {
            "solver_type": "segmentation",
            "task_types": ["segmentation", "image_segmentation"],
            "modalities": ["text", "image"],
        },
        "semantic-segmentation": {
            "solver_type": "segmentation",
            "task_types": ["segmentation", "semantic_segmentation"],
            "modalities": ["text", "image"],
        },
        "instance-segmentation": {
            "solver_type": "segmentation",
            "task_types": ["segmentation", "instance_segmentation"],
            "modalities": ["text", "image"],
        },
        "visual-question-answering": {
            "solver_type": "vlm",
            "task_types": ["visual_reasoning", "question_answering"],
            "modalities": ["text", "image"],
        },
        "image-classification": {
            "solver_type": "vlm",
            "task_types": ["image_classification"],
            "modalities": ["text", "image"],
        },
        "text-generation": {
            "solver_type": "llm",
            "task_types": ["text_generation", "reasoning"],
            "modalities": ["text"],
        },
        "question-answering": {
            "solver_type": "llm",
            "task_types": ["question_answering", "reasoning"],
            "modalities": ["text"],
        },
        "code-generation": {
            "solver_type": "llm",
            "task_types": ["code_generation"],
            "modalities": ["text"],
        },
    }

    # Well-known datasets to track
    DATASET_MAP = {
        "coco": {"task": "object-detection", "metric": "box AP"},
        "imagenet": {"task": "image-classification", "metric": "Top 1 Accuracy"},
        "mmlu": {"task": "text-generation", "metric": "Accuracy"},
        "gsm8k": {"task": "text-generation", "metric": "Accuracy"},
    }

    def __init__(self, tasks: list[str] | None = None, datasets: list[str] | None = None):
        """Initialize Papers with Code discovery agent.

        Args:
            tasks: PwC task slugs to track (default: all mapped tasks)
            datasets: Dataset names to track SOTA for
        """
        super().__init__(source_name="paperswithcode")
        self.tasks = tasks or list(self.TASK_MAP.keys())
        self.datasets = datasets or list(self.DATASET_MAP.keys())

    async def discover_new_solvers(
        self,
        *,
        solver_types: list[str] | None = None,
        min_downloads: int | None = None,
        updated_since: datetime | None = None,
        limit: int = 50,
    ) -> list[SolverCandidate]:
        """Discover SOTA models from Papers with Code.

        Queries PwC API for top-performing models across tracked tasks
        and datasets.

        Args:
            solver_types: Filter by solver types
            min_downloads: Not applicable
            updated_since: Only return models from papers after this date
            limit: Maximum results per task

        Returns:
            List of discovered solver candidates
        """
        candidates = []

        # Filter tasks by solver types if specified
        tasks_to_query = self.tasks
        if solver_types:
            tasks_to_query = [
                task for task in self.tasks
                if self.TASK_MAP.get(task, {}).get("solver_type") in solver_types
            ]

        for task_slug in tasks_to_query:
            try:
                task_candidates = await self._discover_for_task(
                    task_slug, limit=limit, updated_since=updated_since,
                )
                candidates.extend(task_candidates)
            except Exception as e:
                logger.error("PwC discovery failed for task %s: %s", task_slug, e)

        # Also search by dataset for SOTA tracking
        for dataset_name in self.datasets:
            try:
                dataset_candidates = await self._discover_for_dataset(
                    dataset_name, limit=limit,
                )
                candidates.extend(dataset_candidates)
            except Exception as e:
                logger.error("PwC discovery failed for dataset %s: %s", dataset_name, e)

        # Deduplicate by solver_id
        seen = set()
        unique = []
        for c in candidates:
            if c.solver_id not in seen:
                seen.add(c.solver_id)
                unique.append(c)

        logger.info("PwC discovery: found %d unique candidates", len(unique))
        return unique[:limit]

    async def _discover_for_task(
        self,
        task_slug: str,
        limit: int = 10,
        updated_since: datetime | None = None,
    ) -> list[SolverCandidate]:
        """Discover SOTA models for a specific PwC task.

        Args:
            task_slug: PwC task slug
            limit: Max results
            updated_since: Date filter

        Returns:
            List of solver candidates
        """
        candidates = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Get SOTA results for this task
            resp = await client.get(
                f"{self.API_BASE}/tasks/{task_slug}/",
            )

            if resp.status_code == 404:
                logger.debug("PwC task not found: %s", task_slug)
                return []

            resp.raise_for_status()
            task_data = resp.json()

            # Get evaluations (SOTA results)
            eval_resp = await client.get(
                f"{self.API_BASE}/evaluations/",
                params={"task": task_slug, "ordering": "-score", "items_per_page": limit},
            )

            if eval_resp.status_code == 200:
                eval_data = eval_resp.json()
                results = eval_data.get("results", [])

                for result in results:
                    candidate = self._result_to_candidate(result, task_slug, task_data)
                    if candidate:
                        # Apply date filter
                        if updated_since and candidate.discovered_at < updated_since:
                            continue
                        candidates.append(candidate)

        return candidates

    async def _discover_for_dataset(
        self,
        dataset_name: str,
        limit: int = 10,
    ) -> list[SolverCandidate]:
        """Discover SOTA models for a specific dataset.

        Args:
            dataset_name: Dataset name
            limit: Max results

        Returns:
            List of solver candidates
        """
        candidates = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Search for dataset
            resp = await client.get(
                f"{self.API_BASE}/datasets/",
                params={"name": dataset_name, "items_per_page": 5},
            )

            if resp.status_code != 200:
                return []

            data = resp.json()
            results = data.get("results", [])

            for dataset in results:
                dataset_id = dataset.get("id")
                if not dataset_id:
                    continue

                # Get evaluations for this dataset
                eval_resp = await client.get(
                    f"{self.API_BASE}/evaluations/",
                    params={
                        "dataset": dataset_id,
                        "ordering": "-score",
                        "items_per_page": limit,
                    },
                )

                if eval_resp.status_code != 200:
                    continue

                eval_data = eval_resp.json()
                for result in eval_data.get("results", []):
                    # Infer task from dataset
                    ds_info = self.DATASET_MAP.get(dataset_name, {})
                    task_slug = ds_info.get("task", "")

                    candidate = self._result_to_candidate(result, task_slug, {})
                    if candidate:
                        candidates.append(candidate)

        return candidates

    async def get_solver_details(self, solver_id: str) -> SolverCandidate | None:
        """Get detailed information about a PwC model.

        Args:
            solver_id: Solver identifier (format: "pwc-{paper_id}-{method}")

        Returns:
            Detailed solver candidate or None
        """
        # Extract paper URL from solver_id
        if not solver_id.startswith("pwc-"):
            return None

        # Parse solver_id to find the paper
        parts = solver_id[4:]  # Remove "pwc-" prefix

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{self.API_BASE}/papers/",
                    params={"q": parts.replace("-", " "), "items_per_page": 1},
                )

                if resp.status_code != 200:
                    return None

                data = resp.json()
                results = data.get("results", [])
                if not results:
                    return None

                paper = results[0]
                return self._paper_to_candidate(paper)

        except Exception as e:
            logger.error("Error getting PwC details for %s: %s", solver_id, e)
            return None

    def _result_to_candidate(
        self,
        result: dict[str, Any],
        task_slug: str,
        task_data: dict[str, Any],
    ) -> SolverCandidate | None:
        """Convert a PwC evaluation result to SolverCandidate.

        Args:
            result: Evaluation result from PwC API
            task_slug: PwC task slug
            task_data: Task metadata

        Returns:
            SolverCandidate or None
        """
        method_name = result.get("method", "")
        paper = result.get("paper", {}) or {}
        paper_title = paper.get("title", method_name)

        if not method_name:
            return None

        # Get task mapping
        task_info = self.TASK_MAP.get(task_slug, {
            "solver_type": "llm",
            "task_types": [task_slug],
            "modalities": ["text"],
        })

        # Build solver_id from method name
        method_slug = method_name.lower().replace(" ", "-").replace("/", "-")[:50]
        solver_id = f"pwc-{method_slug}"

        # Extract score
        score = result.get("score", 0.0)
        metric_name = result.get("metric", "score")
        dataset_name = result.get("dataset", "")

        # Normalize score to 0-1 if needed
        normalized_score = score
        if score > 1.0:
            normalized_score = score / 100.0  # Assume percentage

        # Build benchmark scores
        benchmark_key = f"{dataset_name}_{metric_name}".lower().replace(" ", "_")
        benchmark_scores = {benchmark_key: normalized_score}

        # Extract paper date
        paper_date = paper.get("published")

        # Determine URL
        paper_url = paper.get("url_abs") or paper.get("url")
        if not paper_url and paper.get("id"):
            paper_url = f"https://paperswithcode.com/paper/{paper['id']}"

        return SolverCandidate(
            solver_id=solver_id,
            name=method_name,
            source=self.source_name,
            solver_type=task_info["solver_type"],
            task_types=task_info["task_types"],
            modalities=task_info["modalities"],
            benchmark_scores=benchmark_scores,
            description=paper_title,
            url=paper_url,
            tags=["paperswithcode", "sota", task_slug],
            discovery_metadata={
                "task_slug": task_slug,
                "metric": metric_name,
                "raw_score": score,
                "dataset": dataset_name,
                "paper_title": paper_title,
                "paper_date": paper_date,
                "paper_id": paper.get("id"),
            },
        )

    def _paper_to_candidate(self, paper: dict[str, Any]) -> SolverCandidate | None:
        """Convert a PwC paper to SolverCandidate.

        Args:
            paper: Paper data from PwC API

        Returns:
            SolverCandidate or None
        """
        title = paper.get("title", "")
        if not title:
            return None

        paper_id = paper.get("id", "")
        method_slug = title.lower().replace(" ", "-")[:50]
        solver_id = f"pwc-{method_slug}"

        # Infer type from tasks
        tasks = paper.get("tasks", [])
        solver_type = "llm"
        task_types = []
        modalities = ["text"]

        for task in tasks:
            task_slug = task.get("slug", "")
            if task_slug in self.TASK_MAP:
                info = self.TASK_MAP[task_slug]
                solver_type = info["solver_type"]
                task_types.extend(info["task_types"])
                modalities = info["modalities"]
                break

        return SolverCandidate(
            solver_id=solver_id,
            name=title,
            source=self.source_name,
            solver_type=solver_type,
            task_types=list(set(task_types)) or ["research"],
            modalities=modalities,
            url=f"https://paperswithcode.com/paper/{paper_id}" if paper_id else None,
            tags=["paperswithcode", "paper"],
            discovery_metadata={
                "paper_id": paper_id,
                "abstract": paper.get("abstract", "")[:500],
                "published": paper.get("published"),
                "arxiv_id": paper.get("arxiv_id"),
            },
        )
