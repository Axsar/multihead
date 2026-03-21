#!/usr/bin/env python3
"""Generate instruction/response pairs from knowledge.db claims for fine-tuning.

Reads claims from the FAISS export (claims.jsonl), groups by topic,
sends batches to Claude to generate natural Q&A pairs, outputs
training-ready JSONL.

Usage:
    python scripts/generate_qa_pairs.py
    python scripts/generate_qa_pairs.py --batch-size 25 --max-batches 10 --dry-run
    python scripts/generate_qa_pairs.py --resume  # continues from last checkpoint
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_data_dir = Path(os.environ.get("MULTIHEAD_DATA_DIR", str(Path.home() / ".multihead")))
DEFAULT_CLAIMS_PATH = _data_dir / "rag" / "claims.jsonl"
DEFAULT_OUTPUT_DIR = _data_dir / "training"
BATCH_SIZE = 20  # Claims per Claude call
MIN_CLAIMS_PER_GROUP = 3  # Skip groups with too few claims

SYSTEM_PROMPT = """You are a training data generator for a local LLM fine-tune.

Given a batch of factual claims about a software project, generate natural
question-answer pairs that a developer working on this project would ask.

RULES:
- Generate 3-5 Q&A pairs per batch
- Questions should be natural and varied (how, what, why, when, which)
- Answers must be grounded ONLY in the provided claims — no hallucination
- Answers should synthesize across multiple claims when relevant
- Include specific numbers, file paths, and technical details from the claims
- Skip trivial claims (status updates, timestamps) — focus on architectural
  decisions, technical facts, and domain knowledge
- If the claims are too vague or trivial to generate useful Q&A, return an empty array

OUTPUT FORMAT (JSON array, nothing else):
[
  {
    "instruction": "What resolution are the SAM 2 balloon masks?",
    "response": "The SAM 2 balloon masks for Young Romance 16 are 3975x6150 pixels.",
    "source_claims": ["claim_key_1", "claim_key_2"]
  }
]"""


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def load_claims(path: Path) -> list[dict]:
    """Load claims from JSONL."""
    claims = []
    with open(path) as f:
        for line in f:
            claims.append(json.loads(line))
    logger.info("Loaded %d claims from %s", len(claims), path)
    return claims


def group_claims(claims: list[dict], batch_size: int = BATCH_SIZE) -> list[list[dict]]:
    """Group claims by topic prefix for coherent batches.

    Groups by first two key segments (e.g. doc.balloonlayout, claude.session),
    then splits large groups into batch_size chunks.
    """
    by_topic: dict[str, list[dict]] = defaultdict(list)

    for claim in claims:
        key = claim.get("claim_key", "")
        parts = key.split(".")
        # Group by first 2 segments for topic coherence
        topic = ".".join(parts[:2]) if len(parts) >= 2 else parts[0] if parts else "unknown"
        by_topic[topic].append(claim)

    batches: list[list[dict]] = []
    for topic, group in sorted(by_topic.items(), key=lambda x: -len(x[1])):
        if len(group) < MIN_CLAIMS_PER_GROUP:
            continue

        # Sort by confidence descending — best claims first
        group.sort(key=lambda c: c.get("confidence", 0), reverse=True)

        for i in range(0, len(group), batch_size):
            batch = group[i:i + batch_size]
            if len(batch) >= MIN_CLAIMS_PER_GROUP:
                batches.append(batch)

    logger.info("Created %d batches from %d topics", len(batches), len(by_topic))
    return batches


# ---------------------------------------------------------------------------
# Claude generation
# ---------------------------------------------------------------------------

def format_batch_prompt(batch: list[dict]) -> str:
    """Format a batch of claims into a prompt for Claude."""
    scope = batch[0].get("scope_id", "unknown")
    lines = [f"PROJECT SCOPE: {scope}\n\nCLAIMS:"]
    for c in batch:
        conf = c.get("confidence", 0)
        lines.append(f"- [{c['claim_key']}] (confidence={conf}) {c['statement']}")
    lines.append("\nGenerate Q&A pairs from these claims. JSON array only.")
    return "\n".join(lines)


def call_multihead(prompt: str, head_id: str = "claude-sonnet", timeout: int = 120) -> str | None:
    """Call MultiHead's generate API with the given head.

    Uses the running MultiHead server at localhost:7337.
    Supports any head: claude-sonnet, claude-sdk, qwen-llm, etc.
    """
    import httpx

    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
    url = f"http://localhost:7337/heads/{head_id}/generate"

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json={"prompt": full_prompt})
            resp.raise_for_status()
            data = resp.json()
            return data.get("text", "")
    except httpx.ConnectError:
        logger.error("MultiHead server not running. Start with: multihead serve")
        return None
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            logger.error("Head '%s' not found. Check: multihead heads", head_id)
        else:
            logger.warning("MultiHead API error: %s", e)
        return None
    except httpx.TimeoutException:
        logger.warning("MultiHead generate timed out after %ds", timeout)
        return None


def parse_qa_response(text: str) -> list[dict]:
    """Extract Q&A pairs from Claude's response."""
    if not text:
        return []

    # Try direct JSON parse
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict) and "instruction" in d]
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code blocks
    import re
    blocks = re.findall(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    for block in blocks:
        try:
            data = json.loads(block.strip())
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict) and "instruction" in d]
        except json.JSONDecodeError:
            continue

    # Try finding JSON array in text
    matches = re.findall(r"\[[\s\S]*\]", text)
    for match in matches:
        try:
            data = json.loads(match)
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict) and "instruction" in d]
        except json.JSONDecodeError:
            continue

    return []


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint(output_dir: Path) -> dict:
    """Load generation checkpoint."""
    path = output_dir / "qa_checkpoint.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {"batches_completed": 0, "pairs_generated": 0}


def save_checkpoint(output_dir: Path, state: dict) -> None:
    """Save generation checkpoint."""
    path = output_dir / "qa_checkpoint.json"
    path.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(
    claims_path: Path,
    output_dir: Path,
    batch_size: int = BATCH_SIZE,
    max_batches: int | None = None,
    dry_run: bool = False,
    resume: bool = False,
    head_id: str = "claude-sonnet",
) -> dict:
    """Generate Q&A pairs from claims."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "qa_pairs.jsonl"

    claims = load_claims(claims_path)
    batches = group_claims(claims, batch_size)

    if max_batches:
        batches = batches[:max_batches]

    # Resume support
    start_batch = 0
    if resume:
        ckpt = load_checkpoint(output_dir)
        start_batch = ckpt.get("batches_completed", 0)
        if start_batch > 0:
            logger.info("Resuming from batch %d", start_batch)

    if dry_run:
        # Show what would be generated
        total_claims = sum(len(b) for b in batches)
        print(f"Would process {len(batches)} batches ({total_claims} claims)")
        print(f"Estimated Claude calls: {len(batches)}")
        print(f"Estimated Q&A pairs: {len(batches) * 4} (avg 4 per batch)")
        print(f"\nSample batch (#{0}):")
        sample = batches[0] if batches else []
        for c in sample[:5]:
            print(f"  [{c['claim_key']}] {c['statement'][:100]}")
        return {"batches": len(batches), "dry_run": True}

    # Generate
    total_pairs = 0
    errors = 0
    t0 = time.time()

    # Open in append mode for resume
    mode = "a" if resume and start_batch > 0 else "w"
    with open(output_path, mode) as out_f:
        for i, batch in enumerate(batches):
            if i < start_batch:
                continue

            prompt = format_batch_prompt(batch)
            logger.info("Batch %d/%d (%d claims, scope=%s)",
                        i + 1, len(batches), len(batch),
                        batch[0].get("scope_id", "?"))

            response = call_multihead(prompt, head_id=head_id)
            pairs = parse_qa_response(response)

            if not pairs:
                errors += 1
                logger.warning("Batch %d: no pairs extracted", i + 1)
            else:
                for pair in pairs:
                    pair["batch_index"] = i
                    pair["scope_id"] = batch[0].get("scope_id", "unknown")
                    out_f.write(json.dumps(pair) + "\n")
                total_pairs += len(pairs)
                logger.info("Batch %d: %d pairs generated (total: %d)",
                            i + 1, len(pairs), total_pairs)

            # Checkpoint after each batch
            save_checkpoint(output_dir, {
                "batches_completed": i + 1,
                "pairs_generated": total_pairs,
                "errors": errors,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            })

            out_f.flush()

    elapsed = time.time() - t0
    result = {
        "batches_processed": len(batches) - start_batch,
        "pairs_generated": total_pairs,
        "errors": errors,
        "output_path": str(output_path),
        "elapsed_seconds": round(elapsed, 1),
    }

    # Save metadata
    meta_path = output_dir / "qa_metadata.json"
    meta_path.write_text(json.dumps(result, indent=2))

    logger.info("Done: %d pairs from %d batches in %.1fs → %s",
                total_pairs, len(batches) - start_batch, elapsed, output_path)
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate Q&A pairs from claims")
    parser.add_argument("--claims", type=Path, default=DEFAULT_CLAIMS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Limit number of batches (for testing)")
    parser.add_argument("--head", type=str, default="claude-sonnet",
                        help="MultiHead head to use (default: claude-sonnet)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be processed without calling LLM")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    args = parser.parse_args()

    generate(
        claims_path=args.claims,
        output_dir=args.output,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        dry_run=args.dry_run,
        resume=args.resume,
        head_id=args.head,
    )


if __name__ == "__main__":
    main()
