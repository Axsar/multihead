"""Night Shift stages 8-15: consistency, briefs, links, indexing, reports."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multihead.knowledge_models import (
    ClaimStatus,
    EntityRef,
    Link,
)

from .models import _prov

logger = logging.getLogger(__name__)


class LateStagesMixin:
    """Mixin providing Night Shift stages 8 through 15."""

    def _resolve_claim_key_to_id(self, claim_key: str) -> str | None:
        """Resolve a claim_key to its claim_id via DB lookup."""
        with self.knowledge._connect() as conn:
            row = conn.execute(
                "SELECT claim_id FROM claims WHERE claim_key = ? "
                "AND claim_status NOT IN ('superseded', 'rejected') LIMIT 1",
                (claim_key,),
            ).fetchone()
        return row["claim_id"] if row else None

    async def _stage_consistency_check(self, context: dict) -> dict:
        # Try context first (same-run), fall back to live DB
        claims = context.get("extracted_claims", [])
        if len(claims) < 2:
            # Query live DB for recent claims to check consistency on
            with self.knowledge._connect() as conn:
                rows = conn.execute(
                    "SELECT claim_key, statement, observation_method, confidence "
                    "FROM claims WHERE claim_status NOT IN ('superseded', 'rejected') "
                    "ORDER BY created_at DESC LIMIT 100000"
                ).fetchall()
                claims = [dict(r) for r in rows]
                logger.info("Consistency check: loaded %d claims from live DB (no context)", len(claims))
        if len(claims) < 2:
            return {"contradictions": [], "metrics": {"contradiction_count": 0}}

        result = await self.consistency_checker.extract(
            [], await self._adapter_or_fn(), claims=claims, concurrency=self.config.concurrency,
            checkpoint_dir=self.output_dir, stage_name="consistency_check",
            batch_mode=self.config.batch_mode, no_wait=self.config.no_wait,
        )

        # Persist contradiction findings to DB
        persisted = 0
        for conflict in result.items:
            key_a = conflict.get("claim_key_a", "")
            key_b = conflict.get("claim_key_b", "")
            if not (key_a and key_b):
                continue

            id_a = self._resolve_claim_key_to_id(key_a)
            id_b = self._resolve_claim_key_to_id(key_b)
            if not (id_a and id_b):
                logger.debug("Could not resolve contradiction pair: %s vs %s", key_a, key_b)
                continue

            reason = conflict.get("reason", "")
            severity = conflict.get("severity", "low")

            # Skip LLM false positives — model returned a conflict but then said "no contradiction"
            if severity == "none" or reason.lower().startswith("no contradiction"):
                logger.debug("Skipping non-contradiction pair: %s vs %s (%s)", key_a, key_b, reason[:60])
                continue

            tagged_reason = f"[{severity}] {reason}"

            try:
                self.knowledge.add_claim_conflict(id_a, id_b, tagged_reason)
            except Exception:
                pass  # duplicate conflict pair, skip

            for cid in (id_a, id_b):
                try:
                    self.knowledge.update_claim_status(
                        cid, ClaimStatus.CONTESTED, reason=tagged_reason,
                    )
                except Exception:
                    pass  # already contested or other constraint

            persisted += 1

        logger.info("Persisted %d/%d contradictions to DB", persisted, len(result.items))

        metrics = result.metrics
        metrics["persisted_count"] = persisted
        return {"contradictions": result.items, "metrics": metrics}

    async def _stage_conflict_resolution(self, context: dict) -> dict:
        """Auto-resolve conflicts flagged by consistency_check.

        Sends unresolved conflict pairs to the LLM for judgment:
        KEEP_A, KEEP_B, BOTH_VALID, TEMPORAL, NEEDS_HUMAN.
        Applies resolutions (supersede losers, un-contest winners).
        """
        import sqlite3

        # Get unresolved conflicts (not already processed)
        with self.knowledge._connect() as conn:
            rows = conn.execute(
                "SELECT cf.claim_id_a, cf.claim_id_b, cf.reason, "
                "c1.statement AS stmt_a, c1.claim_key AS key_a, "
                "c1.observation_method AS obs_a, c1.confidence AS conf_a, "
                "c2.statement AS stmt_b, c2.claim_key AS key_b, "
                "c2.observation_method AS obs_b, c2.confidence AS conf_b "
                "FROM claim_conflicts cf "
                "JOIN claims c1 ON cf.claim_id_a = c1.claim_id "
                "JOIN claims c2 ON cf.claim_id_b = c2.claim_id "
                "WHERE c1.claim_status = 'contested' "
                "ORDER BY cf.rowid DESC LIMIT 100000"
            ).fetchall()

        conflicts = [dict(r) for r in rows]
        if not conflicts:
            return {"metrics": {"conflicts_processed": 0, "resolved": 0}}

        if len(conflicts) >= 100000:
            logger.warning("Conflict resolution hit limit (100000) — some conflicts excluded")
        if len(conflicts) > 10000:
            logger.warning("Conflict resolution processing large set: %d conflicts", len(conflicts))

        logger.info("Conflict resolution: %d unresolved conflicts to process", len(conflicts))

        # Build resolution prompts in batches of 20
        BATCH = 20
        resolved = 0
        kept_a = 0
        kept_b = 0
        both_valid = 0
        temporal = 0
        needs_human = 0
        now = datetime.now(timezone.utc).isoformat()

        for i in range(0, len(conflicts), BATCH):
            batch = conflicts[i:i + BATCH]
            pairs_text = "\n".join(
                f"--- Pair {j} ---\n"
                f"Reason: {c['reason']}\n"
                f"Claim A [{c['key_a']}] (channel: {c.get('obs_a','?')}, confidence: {c.get('conf_a','?')}):\n{c['stmt_a']}\n"
                f"Claim B [{c['key_b']}] (channel: {c.get('obs_b','?')}, confidence: {c.get('conf_b','?')}):\n{c['stmt_b']}\n"
                for j, c in enumerate(batch)
            )

            prompt = (
                "Resolve these knowledge base contradictions. For each pair, decide:\n"
                "- KEEP_A: Claim A is correct, supersede B\n"
                "- KEEP_B: Claim B is correct, supersede A\n"
                "- BOTH_VALID: Both true (different contexts/times)\n"
                "- TEMPORAL: Before/after states, keep newer\n"
                "- NEEDS_HUMAN: Genuinely ambiguous\n\n"
                "Channel reliability (general guideline): direct code/test evidence is more reliable "
                "than conversation-derived claims. Consider the observation method when resolving.\n\n"
                "Output ONLY JSON array: [{\"pair_idx\": 0, \"resolution\": \"KEEP_A\", \"reason\": \"...\"}]\n\n"
                + pairs_text
            )

            try:
                response = await self.heads.generate(self.config.head_id, prompt)
                text = response.get("text", "")
                from multihead.extractors.base import BaseExtractor
                resolutions = BaseExtractor.parse_json_response(text)
                if not isinstance(resolutions, list):
                    resolutions = []
            except Exception as e:
                logger.warning("Resolution batch %d failed: %s", i // BATCH, e)
                continue

            # Apply resolutions
            for res in resolutions:
                try:
                    idx = int(res.get("pair_idx", -1))
                except (ValueError, TypeError):
                    continue
                if idx < 0 or idx >= len(batch):
                    continue

                c = batch[idx]
                resolution = (res.get("resolution") or "").upper()
                id_a, id_b = c["claim_id_a"], c["claim_id_b"]

                try:
                    with self.knowledge._connect() as conn:
                        if resolution == "KEEP_A":
                            conn.execute(
                                "UPDATE claims SET claim_status='superseded', "
                                "superseded_by_claim_id=?, updated_at=? WHERE claim_id=?",
                                (id_a, now, id_b),
                            )
                            conn.execute(
                                "DELETE FROM claim_conflicts WHERE claim_id_a=? AND claim_id_b=?",
                                (id_a, id_b),
                            )
                            kept_a += 1
                        elif resolution == "KEEP_B":
                            conn.execute(
                                "UPDATE claims SET claim_status='superseded', "
                                "superseded_by_claim_id=?, updated_at=? WHERE claim_id=?",
                                (id_b, now, id_a),
                            )
                            conn.execute(
                                "DELETE FROM claim_conflicts WHERE claim_id_a=? AND claim_id_b=?",
                                (id_a, id_b),
                            )
                            kept_b += 1
                        elif resolution in ("BOTH_VALID", "TEMPORAL"):
                            conn.execute(
                                "DELETE FROM claim_conflicts WHERE claim_id_a=? AND claim_id_b=?",
                                (id_a, id_b),
                            )
                            # Un-contest if no other conflicts remain
                            for cid in (id_a, id_b):
                                remaining = conn.execute(
                                    "SELECT COUNT(*) FROM claim_conflicts "
                                    "WHERE claim_id_a=? OR claim_id_b=?",
                                    (cid, cid),
                                ).fetchone()[0]
                                if remaining == 0:
                                    conn.execute(
                                        "UPDATE claims SET claim_status='proposed', "
                                        "contested_reason=NULL, updated_at=? "
                                        "WHERE claim_id=? AND claim_status='contested'",
                                        (now, cid),
                                    )
                            if resolution == "TEMPORAL":
                                temporal += 1
                            else:
                                both_valid += 1
                        else:
                            needs_human += 1
                            # Tag in DB so publish_report can surface them
                            nh_reason = res.get("reason", "Ambiguous — requires human judgment")
                            with self.knowledge._connect() as conn2:
                                conn2.execute(
                                    "UPDATE claim_conflicts SET reason=? "
                                    "WHERE claim_id_a=? AND claim_id_b=?",
                                    (f"[NEEDS_HUMAN] {nh_reason}", id_a, id_b),
                                )
                            continue  # Don't count as resolved

                    resolved += 1
                except Exception as e:
                    logger.debug("Resolution apply error: %s", e)

        logger.info(
            "Conflict resolution: %d resolved (keep_a=%d, keep_b=%d, both_valid=%d, temporal=%d, needs_human=%d)",
            resolved, kept_a, kept_b, both_valid, temporal, needs_human,
        )

        return {
            "metrics": {
                "conflicts_processed": len(conflicts),
                "resolved": resolved,
                "keep_a": kept_a,
                "keep_b": kept_b,
                "both_valid": both_valid,
                "temporal": temporal,
                "needs_human": needs_human,
            },
        }

    async def _stage_claim_fusion(self, context: dict) -> dict:
        """Cross-reference claims from different observation channels.

        Groups claims by file_path anchor, compares across observation methods
        (conversation, code_read, code_behavior_llm, git_history, bash_output), scores convergence,
        and updates confidence/lifecycle.

        Triangulation pattern: independent channels, gradient scoring,
        silence as signal.
        """
        import asyncio
        from multihead.claim_fusion import ClaimFusion
        from multihead.extractors.base import BaseExtractor

        adapter_fn = await self._adapter_or_fn()
        loop = asyncio.get_event_loop()

        def _sync_generate(prompt: str) -> dict:
            """Sync bridge: run async adapter call in the current event loop."""
            coro = BaseExtractor.call_generate(adapter_fn, prompt)
            # We're called from sync code inside an async stage — use a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()

        fusion = ClaimFusion(self.knowledge, generate_fn=_sync_generate)
        scope_id = context.get("scope_id")
        report = fusion.run_fusion(scope_id=scope_id)

        return {
            "metrics": {
                "topics_analyzed": report.topics_analyzed,
                "claims_boosted": report.claims_boosted,
                "claims_penalized": report.claims_penalized,
                "contradictions_found": report.contradictions_found,
                "corroborated": report.corroborated,
                "unverified": report.unverified,
                "temporal_drift": report.temporal_drift,
                "config_drift": report.config_drift,
                "improvement_unverified": report.improvement_unverified,
            },
        }

    async def _stage_open_loops(self, context: dict) -> dict:
        """Extract open questions and TODOs from knowledge store."""
        # Query live DB — all scopes, no artificial limits
        events = self.knowledge.list_events(limit=100000)
        open_events = [e for e in events if e.event_type.value in ("question", "task_created")]

        # Also check for question-type claims across all scopes
        with self.knowledge._connect() as conn:
            question_claims = conn.execute(
                "SELECT claim_key, substr(statement, 1, 200), scope_id FROM claims "
                "WHERE claim_type = 'question' AND claim_status = 'proposed' "
                "ORDER BY created_at DESC LIMIT 100000"
            ).fetchall()

        items = []
        for e in open_events:
            items.append({"title": e.title, "type": e.event_type.value, "priority": "medium"})
        for c in question_claims:
            items.append({"title": c[0], "type": "question", "priority": "medium", "scope": c[2]})

        return {"open_loops": items, "metrics": {"open_loop_count": len(items)}}

    async def _stage_canon_pack_build(self, context: dict) -> dict:
        packs = self.packs.build_standard_packs()
        return {
            "packs_built": [p.pack_id for p in packs],
            "metrics": {"packs_count": len(packs)},
        }

    async def _stage_daily_brief(self, context: dict) -> dict:
        """Generate daily brief summarizing today's events and claims."""
        # Query live DB for today's stats
        with self.knowledge._connect() as conn:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            claims_today = conn.execute(
                "SELECT COUNT(*) FROM claims WHERE created_at >= ?", (today,)
            ).fetchone()[0]
            events_today = conn.execute(
                "SELECT COUNT(*) FROM events WHERE created_at >= ?", (today,)
            ).fetchone()[0] if conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='events'"
            ).fetchone() else 0
            status_counts = conn.execute(
                "SELECT claim_status, COUNT(*) FROM claims GROUP BY claim_status ORDER BY 2 DESC"
            ).fetchall()
            conflicts = conn.execute(
                "SELECT COUNT(*) FROM claim_conflicts WHERE reason LIKE '[NEEDS_HUMAN]%'"
            ).fetchone()[0]
            total = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]

        status_summary = ", ".join(f"{r[0]}={r[1]}" for r in status_counts)
        prompt = (
            f"Write a concise daily brief for a knowledge management system.\n"
            f"Today: {claims_today} new claims, {events_today} events.\n"
            f"Total DB: {total} claims ({status_summary}).\n"
            f"Unresolved conflicts needing human review: {conflicts}.\n"
            f"Summarize the state and any action items."
        )
        try:
            response = await self.heads.generate(self.config.head_id, prompt)
            brief_text = response.get("text", "No brief generated.")
        except Exception:
            brief_text = "Daily brief generation failed."

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        brief_path = self.output_dir / f"daily_{today}.md"
        brief_path.write_text(f"# Daily Brief - {today}\n\n{brief_text}\n", encoding="utf-8")

        return {"daily_brief_path": str(brief_path), "metrics": {
            "total_claims": total, "claims_today": claims_today,
            "needs_human": conflicts, "brief_path": str(brief_path),
        }}

    async def _stage_weekly_rollup(self, context: dict) -> dict | None:
        """Weekly rollup — only run if it's been 7+ days since last."""
        last_weekly = None
        for f in sorted(self.output_dir.glob("weekly_*.md"), reverse=True):
            last_weekly = f
            break

        if last_weekly:
            # Check if it's been at least 7 days
            import os
            mtime = datetime.fromtimestamp(os.path.getmtime(last_weekly), tz=timezone.utc)
            if (datetime.now(timezone.utc) - mtime).days < 7:
                return None  # Skip

        # Query live DB for weekly stats
        with self.knowledge._connect() as conn:
            week_ago = (datetime.now(timezone.utc) - __import__('datetime').timedelta(days=7)).isoformat()
            claims_week = conn.execute(
                "SELECT COUNT(*) FROM claims WHERE created_at >= ?", (week_ago,)
            ).fetchone()[0]
            corroborated = conn.execute(
                "SELECT COUNT(*) FROM claims WHERE claim_status='corroborated'"
            ).fetchone()[0]
            contested = conn.execute(
                "SELECT COUNT(*) FROM claims WHERE claim_status='contested'"
            ).fetchone()[0]
            superseded = conn.execute(
                "SELECT COUNT(*) FROM claims WHERE claim_status='superseded' AND updated_at >= ?", (week_ago,)
            ).fetchone()[0]
            total = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]

        prompt = (
            f"Write a weekly rollup for a knowledge management system.\n"
            f"This week: {claims_week} new claims, {superseded} superseded.\n"
            f"Total: {total} claims, {corroborated} corroborated, {contested} contested.\n"
            f"Summarize trends, progress, and blockers."
        )
        try:
            response = await self.heads.generate(self.config.head_id, prompt)
            rollup_text = response.get("text", "No rollup generated.")
        except Exception:
            rollup_text = "Weekly rollup generation failed."

        week = datetime.now(timezone.utc).strftime("%Y-W%W")
        rollup_path = self.output_dir / f"weekly_{week}.md"
        rollup_path.write_text(f"# Weekly Rollup - {week}\n\n{rollup_text}\n", encoding="utf-8")

        return {"weekly_rollup_path": str(rollup_path), "metrics": {
            "total_claims": total, "claims_this_week": claims_week,
            "corroborated": corroborated, "contested": contested,
            "superseded_this_week": superseded,
        }}

    async def _stage_cross_topic_links(self, context: dict) -> dict:
        """Discover cross-topic connections from live DB.

        Finds files referenced by claims from multiple scopes or channels,
        indicating shared concerns across projects/agents.
        """
        with self.knowledge._connect() as conn:
            # Find file_paths claimed by multiple scopes — cross-project connections
            cross_scope = conn.execute(
                "SELECT json_extract(provenance_json, '$.source_anchor.file_path') as fp, "
                "COUNT(DISTINCT scope_id) as scope_count, "
                "GROUP_CONCAT(DISTINCT scope_id) as scopes, "
                "COUNT(*) as claim_count "
                "FROM claims WHERE json_extract(provenance_json, '$.source_anchor.file_path') IS NOT NULL "
                "AND claim_status NOT IN ('superseded', 'rejected') "
                "GROUP BY fp HAVING scope_count > 1 "
                "ORDER BY claim_count DESC"
            ).fetchall()

            # Find claim_key prefixes shared across different channels
            cross_channel = conn.execute(
                "SELECT substr(claim_key, 1, instr(claim_key || '.', '.') - 1) as topic, "
                "COUNT(DISTINCT observation_method) as channel_count, "
                "GROUP_CONCAT(DISTINCT observation_method) as channels, "
                "COUNT(*) as claim_count "
                "FROM claims WHERE claim_status NOT IN ('superseded', 'rejected') "
                "AND claim_key <> '' "
                "GROUP BY topic HAVING channel_count >= 3 "
                "ORDER BY claim_count DESC LIMIT 50"
            ).fetchall()

        entities_parts = []
        for r in cross_scope[:20]:
            entities_parts.append(f"- FILE {r[0]}: shared by scopes [{r[2]}], {r[3]} claims")
        for r in cross_channel[:20]:
            entities_parts.append(f"- TOPIC {r[0]}: observed by [{r[2]}], {r[3]} claims")

        if not entities_parts:
            return {"links": [], "metrics": {"link_count": 0, "cross_scope_files": 0, "cross_channel_topics": 0}}

        entities_text = "\n".join(entities_parts)
        prompt = (
            "Find meaningful connections between these entities. "
            "Return a JSON array with: from_id, to_id, reason_type, reason, score (0-1).\n\n"
            + entities_text
        )
        try:
            response = await self.heads.generate(self.config.head_id, prompt)
            from multihead.extractors.base import BaseExtractor
            links_data = BaseExtractor.parse_json_response(response.get("text", ""))
        except Exception as e:
            logger.warning("Cross-topic link extraction failed: %s", e)
            links_data = []

        # Store links
        for ld in links_data:
            try:
                link = Link(
                    from_entity=EntityRef(entity_type="concept", entity_id=ld.get("from_id", "")),
                    to_entity=EntityRef(entity_type="concept", entity_id=ld.get("to_id", "")),
                    reason_type=ld.get("reason_type", "unknown"),
                    reason=ld.get("reason", ""),
                    score=float(ld.get("score", 0)),
                    provenance=_prov(),
                )
                self.knowledge.insert_link(link)
            except (ValueError, KeyError) as e:
                logger.debug("Skipping malformed link data: %s", e)

        return {"links": links_data, "metrics": {
            "link_count": len(links_data),
            "cross_scope_files": len(cross_scope),
            "cross_channel_topics": len(cross_channel),
            "entities_analyzed": len(entities_parts),
        }}

    async def _stage_index_rebuild(self, context: dict) -> dict:
        """Build keyword index from live DB claims."""
        # Query live DB — not stale context chunks
        with self.knowledge._connect() as conn:
            rows = conn.execute(
                "SELECT claim_id, statement FROM claims "
                "WHERE claim_status NOT IN ('superseded', 'rejected') "
                "AND length(statement) > 20"
            ).fetchall()

        keyword_index: dict[str, list[str]] = {}
        for row in rows:
            words = set(row["statement"].lower().split())
            for word in words:
                if len(word) > 2:
                    keyword_index.setdefault(word, []).append(row["claim_id"])

        index_path = self.output_dir / "keyword_index.json"
        index_path.write_text(
            json.dumps({"keyword_count": len(keyword_index), "mode": "keyword_only"}, indent=2),
            encoding="utf-8",
        )

        return {
            "index_mode": "keyword_only",
            "metrics": {"keyword_count": len(keyword_index), "claims_indexed": len(rows)},
        }

    async def _stage_file_path_anchoring(self, context: dict) -> dict:
        """Extract file paths from claim statements and add to provenance.

        Claims that mention specific files (e.g. 'router.py') get
        a file_path added to their source_anchor, making them eligible for
        fusion against code_read/git_diff claims about the same file.
        """
        import re

        FILE_RE = re.compile(
            r'(?:'
            r'(?:[a-zA-Z_/][a-zA-Z0-9_./-]*)?(?:src|config|tests|scripts|docs|app|stage|tools)[a-zA-Z0-9_./-]*\.[a-z]{1,5}'
            r'|/mnt/[a-zA-Z]/[^\s,;\"\)\]]+\.[a-z]{1,5}'
            r'|[a-zA-Z_][a-zA-Z0-9_]*\.(?:py|yaml|yml|json|ts|tsx|md|toml|cfg|svg|png|html|css|js|sh|sql)'
            r')'
        )
        from multihead._paths import get_known_project_roots
        PREFIXES = get_known_project_roots()

        with self.knowledge._connect() as conn:
            rows = conn.execute(
                "SELECT claim_id, statement, provenance_json FROM claims "
                "WHERE observation_method IN ('user_statement','assistant_statement') "
                "AND json_extract(provenance_json, '$.source_anchor.file_path') IS NULL"
            ).fetchall()

            anchored = 0
            for row in rows:
                matches = FILE_RE.findall(row['statement'])
                if not matches:
                    continue
                best = max(matches, key=len)
                for prefix in PREFIXES:
                    if best.startswith(prefix):
                        best = best[len(prefix):]
                        break
                if not best or len(best) <= 3:
                    continue

                try:
                    prov = json.loads(row['provenance_json']) if row['provenance_json'] else {}
                except (json.JSONDecodeError, TypeError):
                    prov = {}
                prov.setdefault('source_anchor', {})['file_path'] = best
                conn.execute(
                    'UPDATE claims SET provenance_json=? WHERE claim_id=?',
                    (json.dumps(prov), row['claim_id']),
                )
                anchored += 1

        logger.info("File path anchoring: %d claims anchored out of %d candidates", anchored, len(rows))
        return {"metrics": {"candidates": len(rows), "anchored": anchored}}

    async def _stage_staleness_sweep(self, context: dict) -> dict:
        """Sweep claims for staleness based on git SHA anchors.

        For claims with source_anchor.git_sha, check if the source file
        has changed since that SHA. Mark stale claims with STALE status
        and decay their confidence.
        """
        import subprocess

        # Find claims with git_sha in source_anchor
        marked_stale = 0
        checked = 0
        with self.knowledge._connect() as conn:
            rows = conn.execute(
                "SELECT claim_id, provenance_json FROM claims "
                "WHERE claim_status NOT IN ('superseded', 'rejected', 'stale') "
                "AND provenance_json LIKE '%git_sha%' "
                "LIMIT 100000"
            ).fetchall()

            if not rows:
                return {"metrics": {"checked": 0, "marked_stale": 0}}

            if len(rows) > 10000:
                logger.warning("Staleness sweep processing large set: %d claims with git_sha", len(rows))

            # Group by SHA to minimize git calls
            sha_claims: dict[str, list[tuple[str, str]]] = {}  # sha → [(claim_id, file_path)]
            for row in rows:
                try:
                    prov = json.loads(row["provenance_json"])
                    anchor = prov.get("source_anchor", {})
                    sha = anchor.get("git_sha", "")
                    fpath = anchor.get("file_path", "")
                    if sha:
                        sha_claims.setdefault(sha, []).append((row["claim_id"], fpath))
                except (json.JSONDecodeError, KeyError):
                    continue

            # Known repos — map file_path prefixes to repo roots
            from multihead._paths import get_known_project_roots
            REPO_ROOTS = [Path(r.rstrip("/")) for r in get_known_project_roots()]

            def _find_repo(file_path: str) -> str:
                """Find which repo a file belongs to."""
                for repo in REPO_ROOTS:
                    if (repo / ".git").is_dir():
                        # Check if file_path is relative to this repo
                        try:
                            full = repo / file_path
                            if full.exists():
                                return str(repo)
                        except Exception:
                            pass
                # Default to cwd if no known roots configured
                return str(REPO_ROOTS[0]) if REPO_ROOTS else str(Path.cwd())

            # Group by (sha, repo) to run git diff in correct repo
            now = datetime.now(timezone.utc).isoformat()
            sha_repo_cache: dict[tuple[str, str], set[str]] = {}  # (sha, repo) → changed_files

            for sha, claims_list in sha_claims.items():
                for claim_id, file_path in claims_list:
                    checked += 1
                    repo_dir = _find_repo(file_path)
                    cache_key = (sha, repo_dir)

                    if cache_key not in sha_repo_cache:
                        try:
                            result = subprocess.run(
                                ["git", "diff", "--name-only", sha, "HEAD"],
                                capture_output=True, text=True, timeout=10,
                                cwd=repo_dir,
                            )
                            if result.returncode != 0:
                                sha_repo_cache[cache_key] = set()
                            else:
                                sha_repo_cache[cache_key] = set(
                                    f.strip() for f in result.stdout.strip().split("\n") if f.strip()
                                )
                        except (subprocess.TimeoutExpired, FileNotFoundError):
                            sha_repo_cache[cache_key] = set()

                    changed_files = sha_repo_cache[cache_key]
                    if file_path and any(
                        file_path.endswith(cf) or cf.endswith(file_path) for cf in changed_files
                    ):
                        conn.execute(
                            "UPDATE claims SET claim_status = 'stale', updated_at = ? "
                            "WHERE claim_id = ?",
                            (now, claim_id),
                        )
                        marked_stale += 1

                # Old per-claim loop removed — handled in repo-aware loop above

        logger.info("Staleness sweep: checked %d claims, marked %d stale", checked, marked_stale)
        return {"metrics": {"checked": checked, "marked_stale": marked_stale}}

    async def _stage_publish_report(self, context: dict) -> dict:
        """Write the final Night Shift report."""
        # Query live DB for current state — not stale context
        with self.knowledge._connect() as conn:
            total_claims = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
            status_rows = conn.execute(
                "SELECT claim_status, COUNT(*) FROM claims GROUP BY claim_status ORDER BY 2 DESC"
            ).fetchall()
            channel_rows = conn.execute(
                "SELECT observation_method, COUNT(*) FROM claims GROUP BY observation_method ORDER BY 2 DESC"
            ).fetchall()
            conflicts = conn.execute("SELECT COUNT(*) FROM claim_conflicts").fetchone()[0]
            needs_human = conn.execute(
                "SELECT COUNT(*) FROM claim_conflicts WHERE reason LIKE '[NEEDS_HUMAN]%'"
            ).fetchone()[0]
            producer_rows = conn.execute(
                "SELECT producer, COUNT(*) FROM claims WHERE producer != '' GROUP BY producer ORDER BY 2 DESC"
            ).fetchall()

        report_data = {
            "total_claims": total_claims,
            "claim_status": {r[0]: r[1] for r in status_rows},
            "channels": {r[0]: r[1] for r in channel_rows},
            "producers": {r[0]: r[1] for r in producer_rows},
            "claim_conflicts": conflicts,
            "needs_human": needs_human,
            "stage_metrics": context.get("_all_metrics", {}),
        }

        report_path = self.output_dir / "nightshift_report.json"
        report_path.write_text(json.dumps(report_data, indent=2, default=str), encoding="utf-8")

        report_md = self.output_dir / "nightshift_report.md"
        lines = ["# Night Shift Report", ""]
        for k, v in report_data.items():
            lines.append(f"- **{k}**: {v}")

        # Surface NEEDS_HUMAN conflicts
        with self.knowledge._connect() as conn:
            nh_rows = conn.execute(
                "SELECT cc.reason, cc.claim_id_a, cc.claim_id_b, "
                "c1.claim_key, c1.statement, c1.observation_method, c1.confidence, "
                "c2.claim_key, c2.statement, c2.observation_method, c2.confidence, "
                "json_extract(c1.provenance_json, '$.file_path') AS file_path_a "
                "FROM claim_conflicts cc "
                "JOIN claims c1 ON cc.claim_id_a = c1.claim_id "
                "JOIN claims c2 ON cc.claim_id_b = c2.claim_id "
                "WHERE cc.reason LIKE '[NEEDS_HUMAN]%' "
                "ORDER BY file_path_a, c1.claim_key"
            ).fetchall()

        if nh_rows:
            lines += ["", f"## Conflicts Requiring Human Review ({len(nh_rows)})", ""]
            lines.append("These contradictions could not be auto-resolved. Review and update the knowledge base.\n")

            current_group = None
            for row in nh_rows:
                reason, id_a, id_b, key_a, stmt_a, obs_a, conf_a, key_b, stmt_b, obs_b, conf_b, fp = row
                group = fp or key_a.split(".")[0] if key_a else "misc"
                if group != current_group:
                    lines += ["", f"### {group}", ""]
                    current_group = group

                llm_reason = reason.replace("[NEEDS_HUMAN] ", "")
                lines += [
                    f"**CLAIM A** `{key_a}` ({obs_a}, conf={conf_a:.2f})",
                    f"> {str(stmt_a)[:300]}",
                    "",
                    f"**CLAIM B** `{key_b}` ({obs_b}, conf={conf_b:.2f})",
                    f"> {str(stmt_b)[:300]}",
                    "",
                    f"**Conflict**: {llm_reason}",
                    "",
                    "---",
                    "",
                ]

        report_md.write_text("\n".join(lines), encoding="utf-8")

        return {"metrics": {
            "total_claims": total_claims,
            "corroborated": {r[0]: r[1] for r in status_rows}.get("corroborated", 0),
            "contested": {r[0]: r[1] for r in status_rows}.get("contested", 0),
            "needs_human_conflicts": len(nh_rows) if nh_rows else 0,
            "total_conflicts": conflicts,
        }}
