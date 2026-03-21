"""CI/GitHub Actions extractor — independent observation channel.

Extracts claims from CI test results: pass/fail per commit SHA.
CI results are high-confidence evidence — code was actually executed and verified.

Configure repos via MULTIHEAD_CI_REPOS env var:
  export MULTIHEAD_CI_REPOS="owner/repo,owner/repo2"
"""

from __future__ import annotations

import json
import os
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def extract_ci_runs(
    owner: str = "Axsar",
    repo: str = "multihead",
    limit: int = 100000,
    since: datetime | None = None,
) -> list[dict[str, Any]]:
    """Extract CI run results from GitHub Actions via gh CLI.

    Returns list of claim dicts with observation_method='ci_test'.
    """
    claims: list[dict[str, Any]] = []

    # Fetch recent workflow runs
    try:
        result = subprocess.run(
            ["gh", "api", f"repos/{owner}/{repo}/actions/runs",
             "--jq", f".workflow_runs[:{limit}]"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            logger.error("gh api failed: %s", result.stderr)
            return []
        runs = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
        logger.error("CI extractor error: %s", e)
        return []

    for run in runs:
        conclusion = run.get("conclusion", "")
        sha = run.get("head_sha", "")[:8]
        full_sha = run.get("head_sha", "")
        name = run.get("name", "")
        run_id = run.get("id", 0)
        created = run.get("created_at", "")

        if not conclusion or not sha:
            continue

        # Filter by since
        if since and created:
            try:
                run_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                if run_dt < since:
                    continue
            except ValueError:
                pass

        # Run-level claim: overall pass/fail
        if conclusion == "success":
            stmt = f"CI workflow '{name}' passed for commit {sha} — all tests and checks succeeded"
            confidence = 0.95
        elif conclusion == "failure":
            stmt = f"CI workflow '{name}' failed for commit {sha} — one or more tests or checks failed"
            confidence = 0.95
        elif conclusion == "cancelled":
            stmt = f"CI workflow '{name}' was cancelled for commit {sha}"
            confidence = 0.80
        else:
            stmt = f"CI workflow '{name}' concluded with '{conclusion}' for commit {sha}"
            confidence = 0.70

        if len(stmt) >= 50:
            claims.append({
                "claim_type": "fact",
                "claim_key": f"ci.{repo}.{sha}.{name.replace(' ', '_').lower()}",
                "statement": stmt,
                "confidence": confidence,
                "observation_method": "ci_test",
                "speaker": "tool",
                "evidence": [{
                    "type": "ci_run",
                    "run_id": str(run_id),
                    "sha": full_sha,
                    "conclusion": conclusion,
                    "url": run.get("html_url", ""),
                }],
                "source_anchor": {
                    "git_sha": full_sha,
                },
                "durability": "durable",
            })

        # Fetch job-level details for failures (more specific signal)
        if conclusion == "failure":
            job_claims = _extract_job_results(owner, repo, run_id, sha, full_sha)
            claims.extend(job_claims)

    if len(claims) > 10000:
        logger.warning("CI extractor produced large claim set: %d claims from %d runs", len(claims), len(runs))
    logger.info("CI extractor: %d claims from %d runs (%s/%s)", len(claims), len(runs), owner, repo)
    return claims


def _extract_job_results(
    owner: str, repo: str, run_id: int, sha: str, full_sha: str,
) -> list[dict[str, Any]]:
    """Extract per-job pass/fail for a specific run."""
    claims: list[dict[str, Any]] = []

    try:
        result = subprocess.run(
            ["gh", "api", f"repos/{owner}/{repo}/actions/runs/{run_id}/jobs",
             "--jq", ".jobs"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return []
        jobs = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return []

    for job in jobs:
        job_name = job.get("name", "")
        job_conclusion = job.get("conclusion", "")
        if not job_name or not job_conclusion:
            continue

        if job_conclusion == "failure":
            stmt = f"CI job '{job_name}' failed for commit {sha} — this specific test configuration broke"
            confidence = 0.95
        elif job_conclusion == "success":
            stmt = f"CI job '{job_name}' passed for commit {sha}"
            confidence = 0.95
        elif job_conclusion == "cancelled":
            continue  # Not interesting
        else:
            continue

        if len(stmt) >= 50:
            claims.append({
                "claim_type": "fact",
                "claim_key": f"ci.{job_name.replace(' ', '_').replace('(', '').replace(')', '')}.{sha}",
                "statement": stmt,
                "confidence": confidence,
                "observation_method": "ci_test",
                "speaker": "tool",
                "evidence": [{
                    "type": "ci_job",
                    "job_name": job_name,
                    "sha": full_sha,
                    "conclusion": job_conclusion,
                }],
                "source_anchor": {
                    "git_sha": full_sha,
                },
                "durability": "durable",
            })

    return claims


def scan_ci(
    repos: list[tuple[str, str]] | None = None,
    limit: int = 100000,
    since: datetime | None = None,
) -> list[dict[str, Any]]:
    """Scan CI results across multiple repos.

    Args:
        repos: List of (owner, repo) tuples. Defaults to known repos.
        since: Only include runs after this time.
        limit: Max runs per repo.
    """
    if repos is None:
        # Configure via MULTIHEAD_CI_REPOS env var: "owner/repo,owner/repo"
        env_repos = os.environ.get("MULTIHEAD_CI_REPOS", "")
        if env_repos:
            repos = [tuple(r.strip().split("/", 1)) for r in env_repos.split(",") if "/" in r]
        else:
            repos = []  # No default repos — user must configure

    all_claims: list[dict[str, Any]] = []
    for owner, repo in repos:
        claims = extract_ci_runs(owner, repo, limit=limit, since=since)
        all_claims.extend(claims)

    if len(all_claims) > 10000:
        logger.warning("CI scan produced large total claim set: %d claims from %d repos", len(all_claims), len(repos))
    logger.info("CI scan: %d total claims from %d repos", len(all_claims), len(repos))
    return all_claims
