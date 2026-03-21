"""RAG Eval: Score FTS retrieval against Q&A ground truth.

Runs each Q&A question through the knowledge store's FTS search,
checks if the expected source_claims are in the results, and
computes precision, recall, MRR, and hit rate.

Usage:
    python scripts/eval_rag.py                    # Full eval
    python scripts/eval_rag.py --limit 100        # Quick sample
    python scripts/eval_rag.py --scope h2v        # Filter by scope
    python scripts/eval_rag.py --top-k 15         # Change retrieval depth
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.WARNING, format="%(message)s")

QA_PATH = Path(os.environ.get("MULTIHEAD_DATA_DIR", str(Path.home() / ".multihead"))) / "training" / "qa_pairs.jsonl"


def load_qa_pairs(path: Path, limit: int = 0, scope: str | None = None) -> list[dict]:
    pairs = []
    with open(path) as f:
        for line in f:
            pair = json.loads(line)
            if scope and pair.get("scope_id") != scope:
                continue
            pairs.append(pair)
            if limit and len(pairs) >= limit:
                break
    return pairs


def run_eval(pairs: list[dict], top_k: int = 15, mode: str = "fts") -> dict:
    from multihead.knowledge_store import KnowledgeStore

    db_path = Path(os.environ.get("MULTIHEAD_DATA_DIR", str(Path.home() / ".multihead"))) / "knowledge.db"
    ks = KnowledgeStore(db_path)

    total = len(pairs)
    precisions = []
    recalls = []
    mrrs = []
    hits = 0
    no_results = 0

    t0 = time.time()
    for i, pair in enumerate(pairs):
        question = pair["instruction"]
        expected_keys = set(pair.get("source_claims", []))

        if not expected_keys:
            continue

        # Run search based on mode
        try:
            if mode == "hybrid":
                results = ks.search_claims_hybrid(question, limit=top_k, min_confidence=0.0)
            elif mode == "vec":
                vec_results = ks.search_claims_vec(question, limit=top_k, min_confidence=0.0)
                results = [(k, s, c) for k, s, c, _d in vec_results]
            else:  # fts
                results = ks.search_claims_fts(question, limit=top_k, min_confidence=0.0)
        except Exception:
            results = []

        if not results:
            no_results += 1
            precisions.append(0.0)
            recalls.append(0.0)
            mrrs.append(0.0)
            continue

        returned_keys = [key for key, _stmt, _conf in results]

        # Match: exact key match OR shared prefix (up to last hash segment)
        def keys_match(returned: str, expected_set: set) -> bool:
            if returned in expected_set:
                return True
            # Prefix match: strip trailing hash and compare
            for ek in expected_set:
                # doc.h2v.alignment_fixes_plan.success_criteria.5eb10747
                # → doc.h2v.alignment_fixes_plan.success_criteria
                r_prefix = returned.rsplit(".", 1)[0] if "." in returned else returned
                e_prefix = ek.rsplit(".", 1)[0] if "." in ek else ek
                if r_prefix == e_prefix and len(r_prefix) > 10:
                    return True
            return False

        true_pos = sum(1 for k in returned_keys if keys_match(k, expected_keys))
        precision = true_pos / len(returned_keys) if returned_keys else 0.0
        precisions.append(precision)

        # Recall: of what was expected, how many did we return?
        returned_set = set(returned_keys)
        recall_hits = sum(1 for ek in expected_keys if any(keys_match(rk, {ek}) for rk in returned_keys))
        recall = recall_hits / len(expected_keys) if expected_keys else 0.0
        recalls.append(recall)

        # Hit rate: did we find at least one expected claim?
        if true_pos > 0:
            hits += 1

        # MRR: reciprocal rank of first relevant result
        rr = 0.0
        for rank, key in enumerate(returned_keys, 1):
            if keys_match(key, expected_keys):
                rr = 1.0 / rank
                break
        mrrs.append(rr)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{total} ({elapsed:.1f}s)...", file=sys.stderr)

    elapsed = time.time() - t0
    evaluated = len(precisions)

    return {
        "total_pairs": total,
        "evaluated": evaluated,
        "top_k": top_k,
        "elapsed_seconds": round(elapsed, 1),
        "metrics": {
            "precision_avg": round(sum(precisions) / evaluated, 4) if evaluated else 0,
            "recall_avg": round(sum(recalls) / evaluated, 4) if evaluated else 0,
            "mrr": round(sum(mrrs) / evaluated, 4) if evaluated else 0,
            "hit_rate": round(hits / evaluated, 4) if evaluated else 0,
            "no_results_pct": round(no_results / evaluated, 4) if evaluated else 0,
        },
        "distribution": {
            "precision_0": sum(1 for p in precisions if p == 0),
            "precision_partial": sum(1 for p in precisions if 0 < p < 1),
            "precision_1": sum(1 for p in precisions if p == 1),
            "recall_0": sum(1 for r in recalls if r == 0),
            "recall_partial": sum(1 for r in recalls if 0 < r < 1),
            "recall_1": sum(1 for r in recalls if r == 1),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval against Q&A ground truth")
    parser.add_argument("--limit", type=int, default=0, help="Max pairs to evaluate (0=all)")
    parser.add_argument("--scope", type=str, default=None, help="Filter by scope_id")
    parser.add_argument("--top-k", type=int, default=15, help="Number of results to retrieve")
    parser.add_argument("--mode", choices=["fts", "vec", "hybrid"], default="fts", help="Search mode")
    args = parser.parse_args()

    if not QA_PATH.exists():
        print(f"Q&A file not found: {QA_PATH}")
        sys.exit(1)

    pairs = load_qa_pairs(QA_PATH, limit=args.limit, scope=args.scope)
    print(f"Loaded {len(pairs)} Q&A pairs")
    print(f"Running {args.mode} retrieval (top_k={args.top_k})...\n")

    results = run_eval(pairs, top_k=args.top_k, mode=args.mode)

    m = results["metrics"]
    d = results["distribution"]

    print(f"=== RAG Eval Results ===")
    print(f"Pairs evaluated: {results['evaluated']}")
    print(f"Time: {results['elapsed_seconds']}s")
    print(f"")
    print(f"  Precision (avg):  {m['precision_avg']:.1%}")
    print(f"  Recall (avg):     {m['recall_avg']:.1%}")
    print(f"  MRR:              {m['mrr']:.1%}")
    print(f"  Hit Rate:         {m['hit_rate']:.1%}")
    print(f"  No Results:       {m['no_results_pct']:.1%}")
    print(f"")
    print(f"  Precision dist:   0%={d['precision_0']}  partial={d['precision_partial']}  100%={d['precision_1']}")
    print(f"  Recall dist:      0%={d['recall_0']}  partial={d['recall_partial']}  100%={d['recall_1']}")

    # Save results
    out_path = Path(os.environ.get("MULTIHEAD_DATA_DIR", str(Path.home() / ".multihead"))) / "training" / "rag_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
