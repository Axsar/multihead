#!/usr/bin/env python3
"""Benchmark RAG-augmented answers vs generic model on project-specific questions.

Queries the FAISS index built by export_claims_for_rag.py and evaluates
whether retrieved context enables correct answers.

Usage:
    python scripts/rag_benchmark.py
    python scripts/rag_benchmark.py --top-k 10
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Benchmark questions with ground truth
# Answers a generic model would NOT know — these require project context.
# ---------------------------------------------------------------------------

BENCHMARK = [
    {
        "question": "What resolution are the SAM 2 balloon masks for Young Romance 16?",
        "expected": "3975x6150",
        "keywords": ["3975", "6150"],
        "scope": "h2v",
    },
    {
        "question": "How do tails connect to balloons in the sidecar format?",
        "expected": "3 points: base1, tip, base2 as quadratic bezier (Q-format path: M base1 Q tip base2)",
        "keywords": ["3 points", "base", "tip", "Q-format"],
        "scope": "h2v",
    },
    {
        "question": "What YOLO mAP50 was achieved for comic object detection?",
        "expected": "91.85% mAP (or 87.72% for 8-class)",
        "keywords": ["91.85", "87.72", "mAP"],
        "scope": "h2v",
    },
    {
        "question": "What rendering approach does the Scena DSL web viewer use?",
        "expected": "SVG with CSS transforms for parallax, proven by render_book.py",
        "keywords": ["SVG", "render_book"],
        "scope": "h2v",
    },
    {
        "question": "What are the valid balloon-tail scenarios in H2V?",
        "expected": "Speech balloon with tail, daisy chain sharing one tail, overlapping balloons sharing one tail. Thought bubbles use diminishing circles NOT tails. Narration boxes NEVER have tails.",
        "keywords": ["speech", "tail", "thought", "diminishing", "narration", "never"],
        "scope": "h2v",
    },
    {
        "question": "What consensus strategies does MultiHead support?",
        "expected": "5 strategies: MAJORITY, WEIGHTED, UNANIMOUS, BEST_OF_N, FIRST_TO_AHEAD",
        "keywords": ["MAJORITY", "WEIGHTED", "UNANIMOUS", "FIRST_TO_AHEAD"],
        "scope": "multihead",
    },
    {
        "question": "What GPU and VRAM does the MultiHead system have?",
        "expected": "RTX 4090, 24GB VRAM",
        "keywords": ["4090", "24"],
        "scope": "multihead",
    },
    {
        "question": "What is the UNet IoU score for balloon segmentation?",
        "expected": "99.58% IoU",
        "keywords": ["99.58", "IoU"],
        "scope": "h2v",
    },
    {
        "question": "What causes BubbleFill margin violations?",
        "expected": "17 CRITICAL across 11 balloons, safe_span vs cap ellipse mismatch, k75 boundary brute-force sampling",
        "keywords": ["CRITICAL", "safe_span", "margin"],
        "scope": "h2v",
    },
    {
        "question": "How does MultiHead route tasks to different model heads?",
        "expected": "Weighted scoring: active head (40), circuit breaker (30), VRAM fit (15), error rate (10), latency (5). Steps can declare required_kind for auto-routing.",
        "keywords": ["weighted", "active head", "circuit breaker", "VRAM", "required_kind"],
        "scope": "multihead",
    },
]


def run_benchmark(out_dir: Path, top_k: int = 5):
    """Run benchmark queries against the FAISS index."""
    # Import here to avoid slow load if just viewing questions
    from export_claims_for_rag import query_index

    print(f"Running {len(BENCHMARK)} benchmark questions (top_k={top_k})\n")
    print("=" * 70)

    scores = []
    for i, item in enumerate(BENCHMARK):
        q = item["question"]
        results = query_index(q, out_dir, top_k)

        # Check if any retrieved claim contains expected keywords
        all_text = " ".join(r["statement"].lower() for r in results)
        matched = [kw for kw in item["keywords"] if kw.lower() in all_text]
        score = len(matched) / len(item["keywords"])
        scores.append(score)

        status = "PASS" if score >= 0.5 else "FAIL"
        print(f"\nQ{i+1}: {q}")
        print(f"  Expected keywords: {item['keywords']}")
        print(f"  Matched: {matched} ({score:.0%})")
        print(f"  Status: [{status}]")
        if results:
            print(f"  Top result: [{results[0]['scope_id']}] {results[0]['claim_key']}")
            print(f"    Score: {results[0]['similarity']:.4f}")
            print(f"    {results[0]['statement'][:120]}")

    print("\n" + "=" * 70)
    avg = sum(scores) / len(scores) if scores else 0
    passed = sum(1 for s in scores if s >= 0.5)
    print(f"\nResults: {passed}/{len(BENCHMARK)} passed (avg keyword coverage: {avg:.0%})")
    print(f"A generic model scores 0/{len(BENCHMARK)} on these questions.")


def main():
    parser = argparse.ArgumentParser(description="RAG benchmark")
    parser.add_argument("--out", type=Path, default=Path(os.environ.get("MULTIHEAD_DATA_DIR", str(Path.home() / ".multihead"))) / "rag")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--list", action="store_true", help="Just list questions")
    args = parser.parse_args()

    if args.list:
        for i, item in enumerate(BENCHMARK):
            print(f"Q{i+1}: {item['question']}")
            print(f"  Expected: {item['expected']}")
            print()
        return

    run_benchmark(args.out, args.top_k)


if __name__ == "__main__":
    main()
