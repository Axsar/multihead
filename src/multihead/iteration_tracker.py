"""Iteration Tracker — records experiment attempts in knowledge.db.

Phase 1: Captures human-agent iteration loops (trying approaches, reverting failures).
Phase 2: Powers the automated experiment ratchet (AutoResearch-style).

Every attempt — manual or automated — deposits a structured claim:
  - What was tried (params, approach, code change)
  - What happened (metrics, pass/fail, error)
  - Why it was kept or reverted
  - Git commit SHA (if committed) or "reverted"

This gives any future agent full visibility into what's been tried.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .knowledge_models import (
    Claim,
    ClaimCanonical,
    ClaimScope,
    ClaimStatus,
    ClaimType,
    EntityRef,
    Provenance,
    ScopeType,
    Stability,
    ValueObject,
)
from .subprocess_utils import no_window_flags

logger = logging.getLogger(__name__)


@dataclass
class IterationResult:
    """Result of a single experiment iteration."""

    iteration: int
    status: str  # "success", "improved", "no_improvement", "failed", "error"
    metrics: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    git_sha: str = ""  # commit SHA if kept, "" if reverted
    duration_secs: float = 0.0
    error: str = ""
    claim_id: str = ""


class IterationTracker:
    """Records experiment iterations in knowledge.db.

    Usage:
        tracker = IterationTracker(ks, "balloon-layout-optimization")
        tracker.record_attempt(
            status="improved",
            metrics={"overlap_count": 0, "space_saved": 9.3},
            params={"mode": "bbox", "collision_check": "equator"},
            description="Mode 1 bbox collision — zero overlaps",
            git_sha="c893a88",
        )
    """

    def __init__(
        self,
        knowledge_store: Any,
        experiment_id: str,
        agent_id: str = "iteration-tracker",
        scope_id: str = "experiments",
        work_dir: str | None = None,
    ) -> None:
        self.ks = knowledge_store
        self.experiment_id = experiment_id
        self.agent_id = agent_id
        self.scope_id = scope_id
        self.work_dir = work_dir or "."
        self._iteration = self._load_iteration_count()

    @property
    def iteration(self) -> int:
        return self._iteration

    def _load_iteration_count(self) -> int:
        """Resume iteration count from knowledge.db."""
        if not self.ks:
            return 0
        try:
            claims = self.ks.search_claims(
                f"iteration.{self.experiment_id}",
                limit=1,
            )
            # Find highest iteration number
            max_iter = 0
            for claim in claims:
                key = getattr(claim, "claim_key", "") or ""
                parts = key.rsplit(".", 1)
                if parts and parts[-1].isdigit():
                    max_iter = max(max_iter, int(parts[-1]))
            return max_iter
        except Exception:
            return 0

    def record_attempt(
        self,
        status: str,
        metrics: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        description: str = "",
        git_sha: str = "",
        duration_secs: float = 0.0,
        error: str = "",
        supersedes_claim_id: str | None = None,
    ) -> IterationResult:
        """Record a single iteration attempt."""
        self._iteration += 1
        metrics = metrics or {}
        params = params or {}

        result = IterationResult(
            iteration=self._iteration,
            status=status,
            metrics=metrics,
            params=params,
            description=description,
            git_sha=git_sha,
            duration_secs=duration_secs,
            error=error,
        )

        if not self.ks:
            return result

        try:
            # Determine claim status based on result
            if status in ("success", "improved"):
                claim_status = ClaimStatus.CORROBORATED
                stability = Stability.HIGH
            elif status == "no_improvement":
                claim_status = ClaimStatus.SUPERSEDED
                stability = Stability.TEMPORARY
            elif status in ("failed", "error"):
                claim_status = ClaimStatus.SUPERSEDED
                stability = Stability.VOLATILE
            else:
                claim_status = ClaimStatus.PROPOSED
                stability = Stability.MEDIUM

            # Build statement
            metric_str = ", ".join(f"{k}={v}" for k, v in metrics.items()) if metrics else "no metrics"
            statement = (
                f"Experiment '{self.experiment_id}' iteration {self._iteration}: "
                f"{status}. {description}. Metrics: {metric_str}"
            )
            if error:
                statement += f". Error: {error[:200]}"
            if git_sha:
                statement += f". Committed: {git_sha[:8]}"
            else:
                statement += ". Reverted."

            claim = Claim(
                claim_status=claim_status,
                claim_type=ClaimType.FACT,
                scope=ClaimScope(
                    scope_type=ScopeType.PROJECT,
                    scope_id=self.scope_id,
                ),
                canonical=ClaimCanonical(
                    claim_key=f"iteration.{self.experiment_id}.{self._iteration:04d}",
                    subject=EntityRef(
                        entity_type="experiment",
                        entity_id=self.experiment_id,
                        label=self.experiment_id,
                    ),
                    predicate=f"iteration_{status}",
                    object=ValueObject(
                        value_type="json",
                        value={
                            "iteration": self._iteration,
                            "status": status,
                            "metrics": metrics,
                            "params": params,
                            "description": description,
                            "git_sha": git_sha,
                            "duration_secs": duration_secs,
                            "error": error[:500] if error else "",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    ),
                ),
                statement=statement[:500],
                confidence=0.95 if status in ("success", "improved") else 0.7,
                stability=stability,
                superseded_by_claim_id=supersedes_claim_id,
                provenance=Provenance(
                    produced_by={"kind": "agent", "id": self.agent_id},
                    observation_method="experiment_iteration",
                    source_anchor={
                        "experiment_id": self.experiment_id,
                        "iteration": str(self._iteration),
                        **({"git_sha": git_sha} if git_sha else {}),
                    },
                ),
            )

            self.ks.insert_claim(claim)
            result.claim_id = claim.claim_id
            logger.info(
                "Recorded iteration %d for %s: %s (claim=%s)",
                self._iteration, self.experiment_id, status, claim.claim_id,
            )

        except Exception as e:
            logger.warning("Failed to record iteration: %s", e)

        return result

    def get_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Retrieve iteration history from knowledge.db."""
        if not self.ks:
            return []
        try:
            claims = self.ks.search_claims(
                f"iteration.{self.experiment_id}",
                limit=limit,
            )
            history = []
            for claim in claims:
                obj = getattr(claim, "object", None)
                if obj and hasattr(obj, "value"):
                    history.append(obj.value if isinstance(obj.value, dict) else {"statement": str(obj.value)})
                else:
                    history.append({"statement": getattr(claim, "statement", "")})
            return history
        except Exception:
            return []

    def get_failures(self, limit: int = 20) -> list[dict[str, Any]]:
        """Retrieve only failed iterations — the negative signal."""
        history = self.get_history(limit=limit * 2)
        return [h for h in history if h.get("status") in ("failed", "error", "no_improvement")]

    def get_best(self) -> dict[str, Any] | None:
        """Get the best iteration so far (highest quality metric)."""
        history = self.get_history(limit=100)
        best = None
        best_score = float("-inf")
        for h in history:
            metrics = h.get("metrics", {})
            # Try common metric names
            for key in ("quality_score", "accuracy", "score", "mAP", "IoU"):
                if key in metrics:
                    val = float(metrics[key])
                    if val > best_score:
                        best_score = val
                        best = h
                    break
        return best


def _git_head_sha(work_dir: str = ".") -> str:
    """Get current HEAD SHA."""
    try:
        result = subprocess.run(
            ["git", "-C", work_dir, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            creationflags=no_window_flags(),
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def _git_commit(work_dir: str, message: str, files: list[str] | None = None) -> str:
    """Stage and commit. Returns SHA or empty on failure."""
    try:
        if files:
            subprocess.run(
                ["git", "-C", work_dir, "add"] + files,
                capture_output=True, timeout=10,
                creationflags=no_window_flags(),
            )
        else:
            subprocess.run(
                ["git", "-C", work_dir, "add", "-A"],
                capture_output=True, timeout=10,
                creationflags=no_window_flags(),
            )
        result = subprocess.run(
            ["git", "-C", work_dir, "commit", "-m", message],
            capture_output=True, text=True, timeout=15,
            creationflags=no_window_flags(),
        )
        if result.returncode == 0:
            return _git_head_sha(work_dir)
        return ""
    except Exception:
        return ""


def _git_reset_hard(work_dir: str, to_sha: str) -> bool:
    """Revert to a specific commit."""
    try:
        result = subprocess.run(
            ["git", "-C", work_dir, "reset", "--hard", to_sha],
            capture_output=True, timeout=10,
            creationflags=no_window_flags(),
        )
        return result.returncode == 0
    except Exception:
        return False
