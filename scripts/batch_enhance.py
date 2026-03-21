"""Batch Claude-enhanced extraction for all project planning docs.

Runs each doc through ClaudeEnhancer → NarrativePipeline → KnowledgeStore.
Designed to run in the background while the worker daemon processes tasks.

Usage:
    export ACP_URL="http://localhost:8000/api/v1"
    export ACP_CLAUDE_SESSION_KEY="<jwt>"
    python scripts/batch_enhance.py [--dry-run] [--max N]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import sys
import time
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from multihead.narrative.claude_enhancer import ClaudeEnhancer
from multihead.narrative.pipeline import NarrativePipeline
from multihead.knowledge_store import KnowledgeStore
from multihead.artifact_store import ArtifactStore
from multihead.narrative.context_gen import generate_daemon_context

logging.basicConfig(
    level=logging.INFO,
    format="[batch] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_enhance")

# Project docs root (configure via DOCS_ROOT env var)
DOCS_ROOT = Path(os.environ.get("DOCS_ROOT", str(Path.home() / "docs")))
DATA_DIR = Path(os.environ.get("MULTIHEAD_DATA_DIR", str(Path.home() / ".multihead")))


def classify_doc_type(filename: str) -> str:
    """Guess doc_type from filename."""
    lower = filename.lower()
    if any(w in lower for w in ("plan", "proposal", "strategy")):
        return "plan"
    if any(w in lower for w in ("status", "progress", "update", "summary", "report")):
        return "status"
    if any(w in lower for w in ("fix", "recovery", "troubleshoot", "error")):
        return "fixes"
    if any(w in lower for w in ("rule", "requirement", "constraint")):
        return "constraint"
    if any(w in lower for w in ("decision", "formula")):
        return "decision"
    if any(w in lower for w in ("pipeline", "implementation", "complete", "production")):
        return "plan"
    if any(w in lower for w in ("training",)):
        return "plan"
    return "plan"  # Default


def find_project_docs() -> list[Path]:
    """Find all valuable project planning docs."""
    patterns = [
        "PLAN", "STATUS", "PIPELINE", "ENHANCEMENT", "IMPLEMENTATION",
        "FIXES", "RECOVERY", "RULES", "REQUIREMENT", "DECISION",
        "TRAINING", "PRODUCTION", "DEPLOYMENT", "VIRTUAL",
        "COMPLETE", "PROPOSAL", "STRATEGY",
    ]
    skip_patterns = [
        "CHANGELOG", "BACKUP", "Test0", "SESSION_RESTORE",
        "QUICKSTART", "README_GROUND",
    ]

    docs: list[Path] = []
    for md in DOCS_ROOT.rglob("*.md"):
        if ".git" in md.parts:
            continue
        if md.stat().st_size < 2048:
            continue
        name = md.name.upper()
        if any(skip in name for skip in skip_patterns):
            continue
        if any(pat in name for pat in patterns):
            docs.append(md)

    return sorted(docs)


async def process_doc(
    doc: Path,
    enhancer: ClaudeEnhancer,
    pipeline: NarrativePipeline,
    dry_run: bool = False,
) -> int:
    """Process a single doc through enhanced extraction."""
    doc_type = classify_doc_type(doc.name)
    stage = "unknown"
    for part in doc.parts:
        if part.startswith("Stage"):
            stage = part.lower()
            break

    logger.info("--- Processing: %s (%s, %s) ---", doc.name, doc_type, stage)

    if dry_run:
        content = doc.read_text(encoding="utf-8", errors="replace")
        sections = enhancer._split_sections(content)
        logger.info("  [DRY RUN] %d sections, %d bytes", len(sections), doc.stat().st_size)
        return 0

    try:
        # Phase 1: Heuristic
        heuristic_arts = pipeline.markdown_extractor.extract_from_file(
            doc, doc_type=doc_type, source_project="h2v",
        )
        heuristic_claims = []
        for art in heuristic_arts:
            heuristic_claims.extend(art.get("claims", []))

        # Phase 2: Claude enhancement
        artifacts = await enhancer.enhance_document(
            doc,
            doc_type=doc_type,
            source_project="h2v",
            heuristic_claims=heuristic_claims,
            synthesize=True,
        )

        if not artifacts:
            logger.warning("  No artifacts from %s", doc.name)
            return 0

        # Store via pipeline
        from multihead.narrative.confidence import SourcePriority
        count = 0
        for artifact in artifacts:
            pipeline._store_and_buffer(artifact, SourcePriority.LLM_INFERENCE)
            count += len(artifact.get("claims", []))

        # Fuse and store
        fused = pipeline.run_full()
        stored = pipeline.store_fused_claims(fused)

        logger.info(
            "  %s: %d claims extracted, %d stored (heuristic=%d)",
            doc.name, count, stored, len(heuristic_claims),
        )
        return count

    except Exception as e:
        logger.error("  FAILED %s: %s", doc.name, e)
        return 0


async def main():
    parser = argparse.ArgumentParser(description="Batch Claude-enhanced document extraction")
    parser.add_argument("--dry-run", action="store_true", help="List docs without processing")
    parser.add_argument("--max", type=int, default=0, help="Max docs to process (0=all)")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N docs")
    parser.add_argument("--parallel", type=int, default=1, help="Process N docs concurrently")
    args = parser.parse_args()

    # Not using ACP anymore - calling MultiHead REST API directly
    acp_url = "http://localhost:7337"  # MultiHead serve URL
    api_key = "local"  # Dummy for compatibility

    # Find docs
    docs = find_project_docs()
    logger.info("Found %d project docs to process", len(docs))

    if args.skip:
        docs = docs[args.skip:]
        logger.info("Skipping first %d, processing from #%d", args.skip, args.skip + 1)

    if args.max:
        docs = docs[:args.max]
        logger.info("Limited to %d docs", args.max)

    if args.dry_run:
        enhancer = ClaudeEnhancer(
            acp_url=acp_url, api_key=api_key or "dry-run",
            project_id="h2v",
        )
        total_sections = 0
        for doc in docs:
            content = doc.read_text(encoding="utf-8", errors="replace")
            sections = enhancer._split_sections(content)
            doc_type = classify_doc_type(doc.name)
            logger.info(
                "  [%3d secs] %-50s (%s, %d bytes)",
                len(sections), doc.name, doc_type, doc.stat().st_size,
            )
            total_sections += len(sections)
        logger.info("Total: %d docs, %d sections, ~%d Claude tasks",
                     len(docs), total_sections, total_sections + len(docs))
        estimated_mins = (total_sections + len(docs)) * 0.5  # ~30s per task
        logger.info("Estimated time: %.0f minutes (%.1f hours)", estimated_mins, estimated_mins / 60)
        return

    # Init pipeline
    db_path = DATA_DIR / "knowledge.db"
    ks = KnowledgeStore(db_path)
    art_store = ArtifactStore(DATA_DIR / "artifacts", DATA_DIR / "multihead.db")
    pipeline = NarrativePipeline(ks, project_id="multihead", artifact_store=art_store)

    enhancer = ClaudeEnhancer(
        acp_url=acp_url,
        api_key=api_key,
        project_id="h2v",
        poll_interval=15.0,
        max_wait=900.0,
        max_concurrent=5,
    )

    # Process all docs
    start = time.monotonic()
    total_claims = 0
    processed = 0
    failed = 0
    completed_count = 0
    doc_sem = asyncio.Semaphore(args.parallel)

    async def _process_one(idx: int, doc: Path) -> None:
        nonlocal total_claims, processed, failed, completed_count
        async with doc_sem:
            logger.info("=== Doc %d/%d: %s ===", idx + 1, len(docs), doc.name)
            count = await process_doc(doc, enhancer, pipeline)
            if count > 0:
                total_claims += count
                processed += 1
            else:
                failed += 1
            completed_count += 1

            elapsed = time.monotonic() - start
            rate = elapsed / completed_count
            remaining = rate * (len(docs) - completed_count)
            logger.info(
                "Progress: %d/%d done, %d claims, %.0f min elapsed, ~%.0f min remaining",
                completed_count, len(docs), total_claims, elapsed / 60, remaining / 60,
            )

    if args.parallel > 1:
        logger.info("Processing %d docs with parallelism=%d", len(docs), args.parallel)
        tasks = [_process_one(i, doc) for i, doc in enumerate(docs)]
        await asyncio.gather(*tasks, return_exceptions=True)
    else:
        for i, doc in enumerate(docs):
            await _process_one(i, doc)

    # Update daemon context
    ctx_path = DATA_DIR / "context" / "daemon_narrative.md"
    generate_daemon_context(ks, ctx_path)

    elapsed = time.monotonic() - start
    logger.info("=" * 60)
    logger.info("BATCH COMPLETE")
    logger.info("  Docs: %d processed, %d failed, %d total", processed, failed, len(docs))
    logger.info("  Claims: %d total", total_claims)
    logger.info("  Time: %.1f minutes (%.1f hours)", elapsed / 60, elapsed / 3600)
    logger.info("  Context updated: %s", ctx_path)


if __name__ == "__main__":
    asyncio.run(main())
