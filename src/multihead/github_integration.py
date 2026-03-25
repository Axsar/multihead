"""GitHub Issues integration via `gh` CLI.

Thin wrapper around the GitHub CLI for bidirectional issue support
in the `multihead solve` command.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass

from multihead.subprocess_utils import no_window_flags

logger = logging.getLogger(__name__)

_GH_TIMEOUT = 30  # seconds


# ---------------------------------------------------------------------------
# Ref parsing
# ---------------------------------------------------------------------------

@dataclass
class IssueRef:
    """Parsed GitHub issue reference."""
    owner: str | None
    repo: str | None
    number: int

    @property
    def repo_flag(self) -> list[str]:
        """Return ['--repo', 'owner/repo'] if owner/repo known, else []."""
        if self.owner and self.repo:
            return ["--repo", f"{self.owner}/{self.repo}"]
        return []


def parse_issue_ref(ref: str) -> IssueRef:
    """Parse issue ref: 42, owner/repo#42, or full URL.

    Raises ValueError for unparseable refs.
    """
    ref = ref.strip()

    # Full URL: https://github.com/owner/repo/issues/42
    url_match = re.match(
        r"https?://github\.com/([^/]+)/([^/]+)/issues/(\d+)", ref
    )
    if url_match:
        return IssueRef(
            owner=url_match.group(1),
            repo=url_match.group(2),
            number=int(url_match.group(3)),
        )

    # owner/repo#N
    slug_match = re.match(r"([^/]+)/([^#]+)#(\d+)$", ref)
    if slug_match:
        return IssueRef(
            owner=slug_match.group(1),
            repo=slug_match.group(2),
            number=int(slug_match.group(3)),
        )

    # Plain number: #42 or 42
    num_match = re.match(r"#?(\d+)$", ref)
    if num_match:
        return IssueRef(owner=None, repo=None, number=int(num_match.group(1)))

    raise ValueError(f"Cannot parse issue ref: {ref!r}")


# ---------------------------------------------------------------------------
# Repo inference
# ---------------------------------------------------------------------------

def infer_repo() -> str | None:
    """Infer owner/repo from git remote origin.

    Returns 'owner/repo' string or None if not determinable.
    """
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=10,
            creationflags=no_window_flags(),
        )
        if result.returncode != 0:
            return None
        url = result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

    # SSH: git@github.com:owner/repo.git
    ssh_match = re.match(r"git@github\.com:([^/]+)/(.+?)(?:\.git)?$", url)
    if ssh_match:
        return f"{ssh_match.group(1)}/{ssh_match.group(2)}"

    # HTTPS: https://github.com/owner/repo.git
    https_match = re.match(
        r"https?://github\.com/([^/]+)/(.+?)(?:\.git)?$", url
    )
    if https_match:
        return f"{https_match.group(1)}/{https_match.group(2)}"

    return None


# ---------------------------------------------------------------------------
# gh CLI helpers
# ---------------------------------------------------------------------------

def _run_gh(args: list[str], timeout: int = _GH_TIMEOUT) -> subprocess.CompletedProcess:
    """Run a `gh` command and return the CompletedProcess.

    Raises RuntimeError on non-zero exit or timeout.
    """
    cmd = ["gh"] + args
    logger.debug("gh command: %s", cmd)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            creationflags=no_window_flags(),
        )
    except FileNotFoundError:
        raise RuntimeError(
            "GitHub CLI (gh) not found. Install: https://cli.github.com/"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"gh command timed out after {timeout}s: {' '.join(cmd)}")

    if result.returncode != 0:
        raise RuntimeError(
            f"gh command failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    return result


def _resolve_repo(issue_ref: IssueRef) -> list[str]:
    """Get --repo flag, inferring from git remote if needed."""
    if issue_ref.owner and issue_ref.repo:
        return ["--repo", f"{issue_ref.owner}/{issue_ref.repo}"]
    repo = infer_repo()
    if repo:
        return ["--repo", repo]
    return []  # let gh use the current repo context


# ---------------------------------------------------------------------------
# Issue operations
# ---------------------------------------------------------------------------

def fetch_issue(ref: str) -> dict:
    """Fetch a GitHub issue by ref (42, owner/repo#42, or URL).

    Returns dict with keys: number, title, body, labels, assignees, state, url.
    """
    issue_ref = parse_issue_ref(ref)
    repo_flag = _resolve_repo(issue_ref)

    result = _run_gh([
        "issue", "view", str(issue_ref.number),
        *repo_flag,
        "--json", "number,title,body,labels,assignees,state,url",
    ])
    return json.loads(result.stdout)


def create_issue(
    title: str,
    body: str,
    labels: list[str] | None = None,
    repo: str | None = None,
) -> dict:
    """Create a GitHub issue. Returns dict with number and url."""
    cmd = ["issue", "create", "--title", title, "--body", body]
    if labels:
        for label in labels:
            cmd.extend(["--label", label])
    if repo:
        cmd.extend(["--repo", repo])
    else:
        inferred = infer_repo()
        if inferred:
            cmd.extend(["--repo", inferred])

    result = _run_gh(cmd)

    # gh issue create outputs the URL on stdout
    url = result.stdout.strip()
    # Extract issue number from URL
    num_match = re.search(r"/issues/(\d+)", url)
    number = int(num_match.group(1)) if num_match else 0

    return {"number": number, "url": url}


def comment_on_issue(ref: str, body: str) -> None:
    """Post a comment on a GitHub issue."""
    issue_ref = parse_issue_ref(ref)
    repo_flag = _resolve_repo(issue_ref)

    _run_gh([
        "issue", "comment", str(issue_ref.number),
        *repo_flag,
        "--body", body,
    ])


def update_issue_labels(
    ref: str,
    add: list[str] | None = None,
    remove: list[str] | None = None,
) -> None:
    """Add or remove labels on a GitHub issue."""
    issue_ref = parse_issue_ref(ref)
    repo_flag = _resolve_repo(issue_ref)

    cmd = ["issue", "edit", str(issue_ref.number), *repo_flag]

    if add:
        for label in add:
            cmd.extend(["--add-label", label])
    if remove:
        for label in remove:
            cmd.extend(["--remove-label", label])

    if add or remove:
        _run_gh(cmd)


def close_issue(ref: str, comment: str | None = None) -> None:
    """Close a GitHub issue, optionally with a comment."""
    issue_ref = parse_issue_ref(ref)
    repo_flag = _resolve_repo(issue_ref)

    cmd = ["issue", "close", str(issue_ref.number), *repo_flag]
    if comment:
        cmd.extend(["--comment", comment])
    _run_gh(cmd)


def create_subtask_issues(
    parent_ref: str,
    steps: list[dict],
    repo: str | None = None,
) -> list[dict]:
    """Create one issue per decomposed step, linking back to parent.

    Args:
        parent_ref: Issue ref of the parent (e.g. "42")
        steps: List of dicts with 'name' and optionally 'description', 'action_type'
        repo: Optional owner/repo override

    Returns:
        List of {"number": N, "url": "..."} for each created issue.
    """
    parent = parse_issue_ref(parent_ref)
    parent_num = parent.number
    created = []

    for i, step in enumerate(steps, 1):
        name = step.get("name", step.get("goal", f"Step {i}"))
        desc = step.get("description", "")
        action_type = step.get("action_type", "")

        body_lines = [
            f"Parent issue: #{parent_num}",
            "",
            f"**Step {i}** of decomposed solve task.",
            "",
        ]
        if desc:
            body_lines.append(desc)
            body_lines.append("")
        if action_type:
            body_lines.append(f"Action type: `{action_type}`")

        body = "\n".join(body_lines)
        title = f"Step {i}: {name}"

        result = create_issue(
            title=title,
            body=body,
            labels=["multihead-solve", "subtask"],
            repo=repo,
        )
        created.append(result)

    return created


# ---------------------------------------------------------------------------
# Solve result formatting
# ---------------------------------------------------------------------------

def format_solve_results(
    goal: str,
    status: str,
    duration_seconds: float | None,
    steps: list[dict],
    claims: list[dict] | None = None,
    run_id: str | None = None,
) -> str:
    """Format solve results as a markdown summary for GitHub comments.

    Args:
        goal: The original task/goal
        status: Final status string (e.g. "completed", "failed")
        duration_seconds: Total wall-clock time
        steps: List of dicts with 'name', 'status', 'head_id'
        claims: Optional list of knowledge claims with 'claim_key', 'statement'
        run_id: Optional run ID for reference
    """
    lines = [
        "## MultiHead Solve Results",
        f"**Goal**: {goal}",
        f"**Status**: {status}",
    ]

    if duration_seconds is not None:
        mins, secs = divmod(duration_seconds, 60)
        if mins > 0:
            lines.append(f"**Duration**: {int(mins)}m {secs:.1f}s")
        else:
            lines.append(f"**Duration**: {secs:.1f}s")

    if run_id:
        lines.append(f"**Run ID**: `{run_id}`")

    lines.append("")
    lines.append("### Steps")

    for i, step in enumerate(steps, 1):
        name = step.get("name", f"Step {i}")
        step_status = step.get("status", "unknown")
        head_id = step.get("head_id", "")

        if step_status == "committed":
            check = "x"
        elif step_status == "skipped":
            check = "-"
        else:
            check = " "

        head_info = f" ({head_id})" if head_id else ""
        status_info = f" — {step_status}" if step_status not in ("committed",) else ""
        lines.append(f"- [{check}] Step {i}: {name}{head_info}{status_info}")

    if claims:
        lines.append("")
        lines.append("### Knowledge Claims")
        for claim in claims[:10]:  # cap at 10 to avoid huge comments
            key = claim.get("claim_key", "")
            stmt = claim.get("statement", "")
            lines.append(f"- `{key}`: {stmt}")
        if len(claims) > 10:
            lines.append(f"- ... and {len(claims) - 10} more")

    lines.append("")
    lines.append("---")
    lines.append("*Generated by [MultiHead](https://github.com/Axsar/multihead) autonomous solver*")

    return "\n".join(lines)
