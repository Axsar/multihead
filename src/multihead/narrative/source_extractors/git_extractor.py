"""Extract narrative claims from git history.

Parses git log into Records (immutable evidence) and produces
KnowledgeEvents (commits) and Claims (what changed, why, who).
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multihead.knowledge_models import (
    ActorRef,
    Claim,
    ClaimCanonical,
    ClaimScope,
    ClaimType,
    EntityRef,
    EvidencePointer,
    EventType,
    KnowledgeEvent,
    Provenance,
    Record,
    ScopeType,
    Stability,
    TimeBlock,
    TimePrecision,
    ValueObject,
)
from multihead.models import new_id
from multihead.narrative.confidence import ConfidenceCalibrator, SourcePriority

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from multihead.artifact_store import ArtifactStore

logger = logging.getLogger(__name__)

_PROVENANCE = Provenance(
    produced_by={"kind": "extractor", "id": "narrative.git_extractor"},
    toolchain=[{"name": "git", "version": "2.x"}],
    observation_method="git_history",
)


def _git_prov(commit_hash: str = "", file_paths: list[str] | None = None) -> Provenance:
    """Create provenance with git SHA anchor for staleness tracking."""
    anchor = {}
    if commit_hash:
        anchor["git_sha"] = commit_hash
    if file_paths:
        anchor["file_path"] = file_paths[0]  # Primary file
    return Provenance(
        produced_by={"kind": "extractor", "id": "narrative.git_extractor"},
        toolchain=[{"name": "git", "version": "2.x"}],
        observation_method="git_history",
        source_anchor=anchor,
    )


class GitExtractor:
    """Extract narrative evidence from git repositories."""

    def __init__(self, repo_path: Path, project_id: str = "default", artifact_store: ArtifactStore | None = None):
        self.repo_path = repo_path
        self.project_id = project_id
        self.artifact_store = artifact_store
        self.calibrator = ConfidenceCalibrator()

    def extract_commits(
        self,
        since: datetime | None = None,
        limit: int = 100000,
    ) -> list[dict[str, Any]]:
        """Extract commits as narrative artifacts.

        Returns list of dicts, each containing:
            - record: Record (immutable evidence source)
            - event: KnowledgeEvent (the commit as a semantic event)
            - claims: list[Claim] (what the commit asserts)
            - evidence: list[EvidencePointer] (citations back to record)

        Expert recommendation: use since= (time-based) not fixed limit.
        Limit is a safety cap, not the primary filter.
        """
        log_args = [
            "git", "-C", str(self.repo_path), "log",
            f"--max-count={limit}",
            "--format=%H%n%an%n%ae%n%aI%n%s%n%b%n---END---",
        ]
        if since:
            log_args.append(f"--since={since.isoformat()}")

        try:
            result = subprocess.run(
                log_args, capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                logger.error("git log failed: %s", result.stderr)
                return []
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error("git log error: %s", e)
            return []

        raw = result.stdout
        commits = self._parse_log(raw)
        if len(commits) > 10000:
            logger.warning("Git extractor processing large commit set: %d commits", len(commits))
        artifacts = []

        for commit in commits:
            artifact = self._commit_to_artifacts(commit)
            if artifact:
                artifacts.append(artifact)

        return artifacts

    def _parse_log(self, raw: str) -> list[dict[str, str]]:
        """Parse git log output into structured commit dicts."""
        commits = []
        entries = raw.split("---END---")
        for entry in entries:
            lines = entry.strip().split("\n")
            if len(lines) < 4:
                continue
            commit = {
                "hash": lines[0].strip(),
                "author_name": lines[1].strip(),
                "author_email": lines[2].strip(),
                "date": lines[3].strip(),
                "subject": lines[4].strip() if len(lines) > 4 else "",
                "body": "\n".join(lines[5:]).strip() if len(lines) > 5 else "",
            }
            if commit["hash"]:
                commits.append(commit)
        return commits

    def _commit_to_artifacts(self, commit: dict[str, str]) -> dict[str, Any] | None:
        """Convert a parsed commit into Record + Event + Claims."""
        commit_hash = commit["hash"]
        short_hash = commit_hash[:8]
        subject = commit["subject"]
        body = commit["body"]
        full_message = f"{subject}\n{body}".strip() if body else subject

        # 1. Record (immutable evidence source)
        content_bytes = full_message.encode("utf-8")
        if self.artifact_store:
            ref = self.artifact_store.store(content_bytes, name=f"commit_{short_hash}", media_type="text/x-git-commit")
            sha = ref.artifact_id.removeprefix("sha256:")
        else:
            sha = hashlib.sha256(content_bytes).hexdigest()

        record = Record(
            uri=f"git://{self.repo_path.name}/{commit_hash}",
            sha256=sha,
            mime="text/x-git-commit",
        )

        # 2. Evidence pointer (citation into the record)
        evidence = EvidencePointer(
            record_id=record.record_id,
            uri=record.uri,
            sha256=record.sha256,
            quote=subject[:200],
        )

        # 3. Event (the commit as a semantic event)
        try:
            happened_at = datetime.fromisoformat(commit["date"])
        except ValueError:
            happened_at = datetime.now(timezone.utc)

        event = KnowledgeEvent(
            event_type=EventType.COMMIT,
            title=f"[{short_hash}] {subject[:80]}",
            summary=full_message[:500],
            time=TimeBlock(
                happened_at=happened_at,
                time_precision=TimePrecision.SECOND,
            ),
            actors=[
                ActorRef(
                    actor_type="person",
                    actor_id=commit["author_email"],
                    display=commit["author_name"],
                )
            ],
            entities=[
                EntityRef(
                    entity_type="repo",
                    entity_id=self.repo_path.name,
                    label=self.repo_path.name,
                )
            ],
            tags=self._extract_tags(subject, body),
            evidence_supports=[evidence],
            provenance=_PROVENANCE,
        )

        # 4. Claims from commit message (git_message sub-channel, 0.7 confidence)
        claims = self._extract_claims_from_message(
            subject, body, record, evidence, happened_at, commit_hash,
        )

        # 5. Claims from diff (git_diff sub-channel, 0.9-0.95 confidence)
        diff_stats = self._get_diff_stats(commit_hash)
        if diff_stats:
            diff_claims = self._extract_claims_from_diff(
                commit_hash, diff_stats, record, evidence, happened_at,
            )
            claims.extend(diff_claims)

            # 6. Internal cross-check: message vs diff consistency
            crosscheck_claims = self._check_message_vs_diff(
                subject, diff_stats, commit_hash,
            )
            claims.extend(crosscheck_claims)

        return {
            "record": record,
            "event": event,
            "claims": claims,
            "evidence": [evidence],
        }

    def _extract_claims_from_message(
        self,
        subject: str,
        body: str,
        record: Record,
        evidence: EvidencePointer,
        happened_at: datetime,
        commit_hash: str = "",
    ) -> list[Claim]:
        """Extract claims from commit subject and body."""
        claims: list[Claim] = []
        scope = ClaimScope(
            scope_type=ScopeType.PROJECT,
            scope_id=self.project_id,
        )
        prov = _git_prov(commit_hash) if commit_hash else _PROVENANCE

        # Claim 1: What was done (from subject line — author intent, not ground truth)
        action = self._classify_action(subject)
        cal = self.calibrator.calibrate(0.70, SourcePriority.GIT_COMMIT_MESSAGE)

        claims.append(Claim(
            claim_type=ClaimType.FACT,
            scope=scope,
            canonical=ClaimCanonical(
                claim_key=f"commit.{record.record_id}.action",
                subject=EntityRef(
                    entity_type="repo",
                    entity_id=self.project_id,
                ),
                predicate=action,
                object=ValueObject(value_type="string", value=subject),
            ),
            statement=subject,
            confidence=cal.calibrated,
            stability=Stability.STABLE,
            evidence_supports=[evidence],
            provenance=prov,
        ))

        # Claim 2: Rationale (from body, if present)
        if body and len(body) > 10:
            cal_body = self.calibrator.calibrate(
                0.80, SourcePriority.GIT_COMMIT_MESSAGE,
            )
            claims.append(Claim(
                claim_type=ClaimType.DECISION,
                scope=scope,
                canonical=ClaimCanonical(
                    claim_key=f"commit.{record.record_id}.rationale",
                    subject=EntityRef(
                        entity_type="repo",
                        entity_id=self.project_id,
                    ),
                    predicate="rationale",
                    object=ValueObject(value_type="string", value=body[:500]),
                ),
                statement=f"Rationale: {body[:300]}",
                rationale=body[:500],
                confidence=cal_body.calibrated,
                stability=Stability.STABLE,
                evidence_supports=[evidence],
                provenance=prov,
            ))

        return claims

    def _classify_action(self, subject: str) -> str:
        """Classify commit action from subject line."""
        lower = subject.lower()
        if lower.startswith(("fix", "bugfix", "hotfix")):
            return "fixed"
        if lower.startswith(("add", "feat", "implement")):
            return "added"
        if lower.startswith(("refactor", "restructure", "reorganize")):
            return "refactored"
        if lower.startswith(("update", "enhance", "improve")):
            return "updated"
        if lower.startswith(("remove", "delete", "drop")):
            return "removed"
        if lower.startswith(("test",)):
            return "tested"
        if lower.startswith(("doc", "readme")):
            return "documented"
        return "changed"

    def _extract_tags(self, subject: str, body: str) -> list[str]:
        """Extract tags from commit message."""
        tags: list[str] = ["git", "commit"]
        lower = subject.lower()
        if "fix" in lower:
            tags.append("bugfix")
        if "refactor" in lower:
            tags.append("refactor")
        if "test" in lower:
            tags.append("testing")
        if "wip" in lower or "work in progress" in lower:
            tags.append("wip")
        if "breaking" in lower:
            tags.append("breaking-change")
        return tags

    # -------------------------------------------------------------------
    # Diff-based extraction (expert recommendations 2, 3, 4)
    # -------------------------------------------------------------------

    def _get_diff_stats(self, commit_hash: str) -> list[dict[str, Any]]:
        """Get per-file diff stats for a commit: files changed, insertions, deletions."""
        try:
            result = subprocess.run(
                ["git", "-C", str(self.repo_path), "diff", "--numstat", f"{commit_hash}~1", commit_hash],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0:
                return []
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []

        stats = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                try:
                    added = int(parts[0]) if parts[0] != "-" else 0
                    deleted = int(parts[1]) if parts[1] != "-" else 0
                except ValueError:
                    continue
                stats.append({
                    "file": parts[2],
                    "added": added,
                    "deleted": deleted,
                })
        return stats

    def _get_diff_content(self, commit_hash: str, file_path: str, max_lines: int = 100) -> str:
        """Get the actual diff content for a specific file in a commit."""
        try:
            result = subprocess.run(
                ["git", "-C", str(self.repo_path), "diff", f"{commit_hash}~1", commit_hash, "--", file_path],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0:
                return ""
            lines = result.stdout.split("\n")
            return "\n".join(lines[:max_lines])
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return ""

    def _extract_claims_from_diff(
        self,
        commit_hash: str,
        diff_stats: list[dict[str, Any]],
        record: Record,
        evidence: EvidencePointer,
        happened_at: datetime,
    ) -> list[Claim]:
        """Extract file-level and function-level claims from git diff.

        Produces git_diff claims at 0.9 confidence (ground truth — what actually changed).
        """
        import re
        claims: list[Claim] = []
        scope = ClaimScope(scope_type=ScopeType.PROJECT, scope_id=self.project_id)

        for stat in diff_stats:
            fpath = stat["file"]
            added = stat["added"]
            deleted = stat["deleted"]

            # Skip binary files and non-code files
            if not fpath.endswith((".py", ".js", ".ts", ".yaml", ".yml", ".json", ".md", ".toml")):
                continue

            prov = _git_prov(commit_hash, [fpath])
            prov.observation_method = "git_diff"

            # File-level claim: what happened to this file
            if added > 0 and deleted == 0:
                action_desc = f"added {added} lines"
            elif deleted > 0 and added == 0:
                action_desc = f"removed {deleted} lines"
            else:
                action_desc = f"modified ({added} added, {deleted} deleted)"

            stmt = f"File {fpath} was {action_desc} in commit {commit_hash[:8]}"
            if len(stmt) >= 50:
                claims.append(Claim(
                    claim_type=ClaimType.FACT,
                    scope=scope,
                    canonical=ClaimCanonical(
                        claim_key=f"git_diff.{commit_hash[:8]}.{fpath.replace('/', '.')}",
                        subject=EntityRef(entity_type="file", entity_id=fpath),
                        predicate="was_modified",
                        object=ValueObject(value_type="string", value=action_desc),
                    ),
                    statement=stmt,
                    confidence=0.95,  # Diff is ground truth
                    stability=Stability.STABLE,
                    evidence_supports=[evidence],
                    provenance=prov,
                ))

            # Function-level claims: detect added/removed/modified functions
            if fpath.endswith(".py") and (added + deleted) > 3:
                diff_text = self._get_diff_content(commit_hash, fpath, max_lines=200)
                if diff_text:
                    # Extract function-level changes from diff
                    func_claims = self._extract_function_changes(
                        diff_text, fpath, commit_hash, scope, evidence, prov,
                    )
                    claims.extend(func_claims)

        return claims

    def _extract_function_changes(
        self,
        diff_text: str,
        file_path: str,
        commit_hash: str,
        scope: ClaimScope,
        evidence: EvidencePointer,
        prov: Provenance,
    ) -> list[Claim]:
        """Extract function-level claims from a diff.

        Detects added, removed, and modified function definitions.
        """
        import re
        claims: list[Claim] = []

        # Find function definitions in added/removed lines
        added_funcs = set()
        removed_funcs = set()
        modified_context_funcs = set()

        for line in diff_text.split("\n"):
            # Function def in added line
            m = re.match(r'^\+\s*(async\s+)?def\s+(\w+)\s*\(', line)
            if m:
                added_funcs.add(m.group(2))
                continue
            # Function def in removed line
            m = re.match(r'^-\s*(async\s+)?def\s+(\w+)\s*\(', line)
            if m:
                removed_funcs.add(m.group(2))
                continue
            # Hunk header shows function context
            m = re.match(r'^@@.*@@\s*(async\s+)?def\s+(\w+)', line)
            if m:
                modified_context_funcs.add(m.group(2))

        module = Path(file_path).stem

        # New functions
        for func in added_funcs - removed_funcs:
            stmt = f"Function {func} was added to {file_path} in commit {commit_hash[:8]}"
            if len(stmt) >= 50:
                claims.append(Claim(
                    claim_type=ClaimType.FACT,
                    scope=scope,
                    canonical=ClaimCanonical(
                        claim_key=f"git_diff.{module}.{func}.added",
                        subject=EntityRef(entity_type="function", entity_id=func),
                        predicate="was_added",
                        object=ValueObject(value_type="string", value=file_path),
                    ),
                    statement=stmt,
                    confidence=0.95,
                    evidence_supports=[evidence],
                    provenance=prov,
                ))

        # Removed functions
        for func in removed_funcs - added_funcs:
            stmt = f"Function {func} was removed from {file_path} in commit {commit_hash[:8]}"
            if len(stmt) >= 50:
                claims.append(Claim(
                    claim_type=ClaimType.FACT,
                    scope=scope,
                    canonical=ClaimCanonical(
                        claim_key=f"git_diff.{module}.{func}.removed",
                        subject=EntityRef(entity_type="function", entity_id=func),
                        predicate="was_removed",
                        object=ValueObject(value_type="string", value=file_path),
                    ),
                    statement=stmt,
                    confidence=0.95,
                    evidence_supports=[evidence],
                    provenance=prov,
                ))

        # Renamed/rewritten functions (in both added and removed)
        for func in added_funcs & removed_funcs:
            stmt = f"Function {func} was rewritten in {file_path} in commit {commit_hash[:8]}"
            if len(stmt) >= 50:
                claims.append(Claim(
                    claim_type=ClaimType.FACT,
                    scope=scope,
                    canonical=ClaimCanonical(
                        claim_key=f"git_diff.{module}.{func}.rewritten",
                        subject=EntityRef(entity_type="function", entity_id=func),
                        predicate="was_rewritten",
                        object=ValueObject(value_type="string", value=file_path),
                    ),
                    statement=stmt,
                    confidence=0.95,
                    evidence_supports=[evidence],
                    provenance=prov,
                ))

        # Modified functions (changed within function body)
        for func in modified_context_funcs - added_funcs - removed_funcs:
            stmt = f"Function {func} was modified in {file_path} in commit {commit_hash[:8]}"
            if len(stmt) >= 50:
                claims.append(Claim(
                    claim_type=ClaimType.FACT,
                    scope=scope,
                    canonical=ClaimCanonical(
                        claim_key=f"git_diff.{module}.{func}.modified",
                        subject=EntityRef(entity_type="function", entity_id=func),
                        predicate="was_modified",
                        object=ValueObject(value_type="string", value=file_path),
                    ),
                    statement=stmt,
                    confidence=0.95,
                    evidence_supports=[evidence],
                    provenance=prov,
                ))

        return claims

    def _check_message_vs_diff(
        self,
        subject: str,
        diff_stats: list[dict[str, Any]],
        commit_hash: str,
    ) -> list[Claim]:
        """Internal cross-check: does the commit message match the diff?

        Expert recommendation #4: message says 'removed' but diff says 'modified'?
        Flag it within the git channel before fusion.
        """
        claims: list[Claim] = []
        lower = subject.lower()
        scope = ClaimScope(scope_type=ScopeType.PROJECT, scope_id=self.project_id)
        prov = _git_prov(commit_hash)

        files_changed = [s["file"] for s in diff_stats]
        total_added = sum(s["added"] for s in diff_stats)
        total_deleted = sum(s["deleted"] for s in diff_stats)

        discrepancies = []

        # "removed" in message but mostly additions in diff
        if any(w in lower for w in ("remove", "delete", "drop")) and total_added > total_deleted * 2:
            discrepancies.append(
                f"Message says removal but diff shows net additions ({total_added} added, {total_deleted} deleted)"
            )

        # "added" in message but mostly deletions
        if any(w in lower for w in ("add", "feat", "implement", "new")) and total_deleted > total_added * 2:
            discrepancies.append(
                f"Message says addition but diff shows net deletions ({total_added} added, {total_deleted} deleted)"
            )

        # Single file mentioned in message but many files changed
        if len(diff_stats) > 5 and not any(w in lower for w in ("refactor", "rename", "move", "chore")):
            discrepancies.append(
                f"Message does not indicate broad change but {len(diff_stats)} files were modified"
            )

        for disc in discrepancies:
            stmt = f"Git internal cross-check [{commit_hash[:8]}]: {disc}"
            if len(stmt) >= 50:
                claims.append(Claim(
                    claim_type=ClaimType.FACT,
                    scope=scope,
                    canonical=ClaimCanonical(
                        claim_key=f"git_crosscheck.{commit_hash[:8]}",
                        subject=EntityRef(entity_type="commit", entity_id=commit_hash[:8]),
                        predicate="message_diff_discrepancy",
                        object=ValueObject(value_type="string", value=disc),
                    ),
                    statement=stmt,
                    confidence=0.85,
                    provenance=prov,
                ))

        return claims
