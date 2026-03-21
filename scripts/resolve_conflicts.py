"""Phase 2: LLM-assisted conflict resolution via Anthropic Batch API.

Groups remaining conflicts, sends to Haiku for resolution judgment,
then applies the resolution (supersede, accept both, merge, flag for human).

Usage:
    python scripts/resolve_conflicts.py --dry-run        # Preview batches
    python scripts/resolve_conflicts.py                   # Submit batch
    python scripts/resolve_conflicts.py --apply <batch_id>  # Apply results
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path(os.environ.get("MULTIHEAD_DATA_DIR", str(Path.home() / ".multihead"))) / "knowledge.db"
OUTPUT_DIR = Path(os.environ.get("MULTIHEAD_DATA_DIR", str(Path.home() / ".multihead"))) / "conflict_resolution"

RESOLUTION_PROMPT = """You are a knowledge base curator resolving contradictions between claims.

For each pair of conflicting claims, decide the resolution:

- KEEP_A: Claim A is correct, supersede Claim B
- KEEP_B: Claim B is correct, supersede Claim A
- BOTH_VALID: Both are true (different contexts, times, or scopes) — accept both, remove conflict
- TEMPORAL: One is the "before" state and one is "after" a fix/change — keep the newer one, supersede the older
- NEEDS_HUMAN: Genuinely ambiguous, needs human decision

Output ONLY valid JSON array (no markdown fences):
[
  {{"pair_idx": 0, "resolution": "KEEP_A|KEEP_B|BOTH_VALID|TEMPORAL|NEEDS_HUMAN", "reason": "brief explanation"}},
  ...
]

Here are the conflict pairs:

{pairs_text}
"""

BATCH_SIZE = 20  # Pairs per prompt


def load_conflicts(limit: int = 0) -> list[dict]:
    """Load remaining conflicts with claim statements."""
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    query = """
        SELECT cf.claim_id_a, cf.claim_id_b, cf.reason,
               c1.statement as stmt_a, c1.claim_key as key_a, c1.created_at as created_a,
               c2.statement as stmt_b, c2.claim_key as key_b, c2.created_at as created_b
        FROM claim_conflicts cf
        JOIN claims c1 ON cf.claim_id_a = c1.claim_id
        JOIN claims c2 ON cf.claim_id_b = c2.claim_id
        ORDER BY cf.reason DESC
    """
    if limit:
        query += f" LIMIT {limit}"
    rows = conn.execute(query).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def build_batches(conflicts: list[dict]) -> list[list[dict]]:
    """Group conflicts into batches for LLM processing."""
    batches = []
    for i in range(0, len(conflicts), BATCH_SIZE):
        batches.append(conflicts[i:i + BATCH_SIZE])
    return batches


def format_pairs(batch: list[dict]) -> str:
    """Format a batch of conflicts for the resolution prompt."""
    lines = []
    for i, c in enumerate(batch):
        lines.append(f"--- Pair {i} ---")
        lines.append(f"Conflict reason: {c['reason']}")
        lines.append(f"Claim A [{c['key_a']}] (created {c['created_a']}):")
        lines.append(f"  {c['stmt_a'][:400]}")
        lines.append(f"Claim B [{c['key_b']}] (created {c['created_b']}):")
        lines.append(f"  {c['stmt_b'][:400]}")
        lines.append("")
    return "\n".join(lines)


def submit_batch(batches: list[list[dict]], dry_run: bool = False) -> str | None:
    """Submit resolution requests via Anthropic Batch API."""
    import anthropic

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    requests = []
    for batch_idx, batch in enumerate(batches):
        pairs_text = format_pairs(batch)
        prompt = RESOLUTION_PROMPT.format(pairs_text=pairs_text)
        requests.append({
            "custom_id": f"resolve_batch_{batch_idx}",
            "params": {
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 2048,
                "messages": [{"role": "user", "content": prompt}],
            },
        })

    logger.info("Built %d batch requests (%d conflict pairs)", len(requests), sum(len(b) for b in batches))

    if dry_run:
        # Save preview
        preview_path = OUTPUT_DIR / "resolution_preview.json"
        with open(preview_path, "w") as f:
            json.dump({"total_pairs": sum(len(b) for b in batches),
                       "total_batches": len(requests),
                       "sample_request": requests[0] if requests else None}, f, indent=2)
        logger.info("Dry run — preview saved to %s", preview_path)
        return None

    # Write JSONL
    jsonl_path = OUTPUT_DIR / "resolution_requests.jsonl"
    with open(jsonl_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    # Submit via Anthropic Batch API
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id

    logger.info("Batch submitted: %s (%d requests)", batch_id, len(requests))

    # Save batch metadata
    meta = {
        "batch_id": batch_id,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "total_requests": len(requests),
        "total_pairs": sum(len(b) for b in batches),
    }
    with open(OUTPUT_DIR / f"batch_{batch_id}.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Save the original conflicts for apply phase
    with open(OUTPUT_DIR / f"conflicts_{batch_id}.json", "w") as f:
        json.dump(batches, f)

    return batch_id


def poll_batch(batch_id: str) -> dict | None:
    """Poll for batch completion, return results."""
    import anthropic

    client = anthropic.Anthropic()
    batch = client.messages.batches.retrieve(batch_id)
    logger.info("Batch %s: %s (%d/%d)", batch_id, batch.processing_status,
                batch.request_counts.succeeded, batch.request_counts.processing)

    if batch.processing_status != "ended":
        return None

    # Download results
    results = {}
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        if result.result.type == "succeeded":
            text = result.result.message.content[0].text
            try:
                resolutions = json.loads(text)
                results[custom_id] = resolutions
            except json.JSONDecodeError:
                # Try to extract JSON from text
                import re
                match = re.search(r'\[.*\]', text, re.DOTALL)
                if match:
                    try:
                        resolutions = json.loads(match.group())
                        results[custom_id] = resolutions
                    except json.JSONDecodeError:
                        logger.warning("Could not parse response for %s", custom_id)
                        results[custom_id] = []
        else:
            logger.warning("Request %s failed: %s", custom_id, result.result.type)

    # Save results
    results_path = OUTPUT_DIR / f"results_{batch_id}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)

    return results


def apply_results(batch_id: str) -> dict:
    """Apply resolution results to the database."""
    # Load original conflicts and results
    conflicts_path = OUTPUT_DIR / f"conflicts_{batch_id}.json"
    results_path = OUTPUT_DIR / f"results_{batch_id}.json"

    if not conflicts_path.exists() or not results_path.exists():
        # Try polling first
        results = poll_batch(batch_id)
        if results is None:
            logger.info("Batch not complete yet. Try again later.")
            return {"status": "pending"}
    else:
        with open(results_path) as f:
            results = json.load(f)

    with open(conflicts_path) as f:
        batches = json.load(f)

    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    now = datetime.now(timezone.utc).isoformat()

    stats = {"keep_a": 0, "keep_b": 0, "both_valid": 0, "temporal": 0, "needs_human": 0, "errors": 0}

    for batch_idx, batch in enumerate(batches):
        key = f"resolve_batch_{batch_idx}"
        resolutions = results.get(key, [])

        for res in resolutions:
            idx = res.get("pair_idx", -1)
            if idx < 0 or idx >= len(batch):
                stats["errors"] += 1
                continue

            conflict = batch[idx]
            resolution = res.get("resolution", "NEEDS_HUMAN").upper()
            id_a = conflict["claim_id_a"]
            id_b = conflict["claim_id_b"]

            if resolution == "KEEP_A":
                # Supersede B
                conn.execute(
                    "UPDATE claims SET claim_status = 'superseded', superseded_by_claim_id = ?, "
                    "updated_at = ? WHERE claim_id = ?", (id_a, now, id_b)
                )
                conn.execute(
                    "DELETE FROM claim_conflicts WHERE claim_id_a = ? AND claim_id_b = ?",
                    (id_a, id_b)
                )
                # Un-contest A if no other conflicts
                remaining = conn.execute(
                    "SELECT COUNT(*) FROM claim_conflicts WHERE claim_id_a = ? OR claim_id_b = ?",
                    (id_a, id_a)
                ).fetchone()[0]
                if remaining == 0:
                    conn.execute(
                        "UPDATE claims SET claim_status = 'proposed', contested_reason = NULL, "
                        "updated_at = ? WHERE claim_id = ? AND claim_status = 'contested'",
                        (now, id_a)
                    )
                stats["keep_a"] += 1

            elif resolution == "KEEP_B":
                conn.execute(
                    "UPDATE claims SET claim_status = 'superseded', superseded_by_claim_id = ?, "
                    "updated_at = ? WHERE claim_id = ?", (id_b, now, id_a)
                )
                conn.execute(
                    "DELETE FROM claim_conflicts WHERE claim_id_a = ? AND claim_id_b = ?",
                    (id_a, id_b)
                )
                remaining = conn.execute(
                    "SELECT COUNT(*) FROM claim_conflicts WHERE claim_id_a = ? OR claim_id_b = ?",
                    (id_b, id_b)
                ).fetchone()[0]
                if remaining == 0:
                    conn.execute(
                        "UPDATE claims SET claim_status = 'proposed', contested_reason = NULL, "
                        "updated_at = ? WHERE claim_id = ? AND claim_status = 'contested'",
                        (now, id_b)
                    )
                stats["keep_b"] += 1

            elif resolution in ("BOTH_VALID", "TEMPORAL"):
                # Remove conflict, un-contest both if no other conflicts
                conn.execute(
                    "DELETE FROM claim_conflicts WHERE claim_id_a = ? AND claim_id_b = ?",
                    (id_a, id_b)
                )
                for cid in (id_a, id_b):
                    remaining = conn.execute(
                        "SELECT COUNT(*) FROM claim_conflicts WHERE claim_id_a = ? OR claim_id_b = ?",
                        (cid, cid)
                    ).fetchone()[0]
                    if remaining == 0:
                        conn.execute(
                            "UPDATE claims SET claim_status = 'proposed', contested_reason = NULL, "
                            "updated_at = ? WHERE claim_id = ? AND claim_status = 'contested'",
                            (now, cid)
                        )
                stats[resolution.lower()] += 1

            else:  # NEEDS_HUMAN
                stats["needs_human"] += 1

    conn.commit()

    # Final counts
    final_conflicts = conn.execute("SELECT COUNT(*) FROM claim_conflicts").fetchone()[0]
    final_contested = conn.execute("SELECT COUNT(*) FROM claims WHERE claim_status='contested'").fetchone()[0]
    conn.close()

    stats["final_conflicts"] = final_conflicts
    stats["final_contested"] = final_contested
    logger.info("Applied: %s", json.dumps(stats, indent=2))
    return stats


def main():
    parser = argparse.ArgumentParser(description="Resolve knowledge conflicts via LLM")
    parser.add_argument("--dry-run", action="store_true", help="Preview without submitting")
    parser.add_argument("--apply", metavar="BATCH_ID", help="Apply results from a completed batch")
    parser.add_argument("--poll", metavar="BATCH_ID", help="Poll batch status")
    parser.add_argument("--limit", type=int, default=0, help="Limit conflicts to process")
    args = parser.parse_args()

    if args.apply:
        stats = apply_results(args.apply)
        print(json.dumps(stats, indent=2))
        return

    if args.poll:
        results = poll_batch(args.poll)
        if results is None:
            print("Batch still processing...")
        else:
            print(f"Complete: {len(results)} batch responses")
        return

    # Load and batch
    conflicts = load_conflicts(limit=args.limit)
    logger.info("Loaded %d conflicts", len(conflicts))
    batches = build_batches(conflicts)
    logger.info("Grouped into %d batches of %d", len(batches), BATCH_SIZE)

    batch_id = submit_batch(batches, dry_run=args.dry_run)
    if batch_id:
        print(f"\nBatch submitted: {batch_id}")
        print(f"Poll:  python scripts/resolve_conflicts.py --poll {batch_id}")
        print(f"Apply: python scripts/resolve_conflicts.py --apply {batch_id}")


if __name__ == "__main__":
    main()
