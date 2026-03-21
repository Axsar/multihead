#!/usr/bin/env python3
"""Export knowledge.db claims to JSONL + FAISS index for RAG.

Filters out noise (mesh.presence, head.* compatibility), embeds with
Sentence-BERT (all-MiniLM-L6-v2, 384-dim), stores in FAISS index.

Usage:
    python scripts/export_claims_for_rag.py
    python scripts/export_claims_for_rag.py --limit 1000 --dry-run
    python scripts/export_claims_for_rag.py --query "How do tails connect to balloons?"
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_data_dir = Path(os.environ.get("MULTIHEAD_DATA_DIR", str(Path.home() / ".multihead")))
DEFAULT_DB = _data_dir / "knowledge.db"
DEFAULT_OUT_DIR = _data_dir / "rag"
MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, fast, good quality
EMBEDDING_DIM = 384
BATCH_SIZE = 256

# Key prefixes to SKIP (noise, not useful for RAG)
SKIP_PREFIXES = [
    "mesh.presence.",       # heartbeat spam
    "mesh.latency.",        # latency pings
    "head.",                # head compatibility probes
    "poller.",              # poller status
    "nightshift.stage.",    # per-stage metrics (keep rollups)
]

# Claim statuses to include
INCLUDE_STATUSES = ("accepted", "proposed")


def export_claims(db_path: Path, limit: int | None = None) -> list[dict]:
    """Export filtered claims from knowledge.db."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    query = """
        SELECT claim_id, claim_key, claim_type, claim_status,
               scope_id, statement, rationale, confidence,
               provenance_json, created_at
        FROM claims
        WHERE claim_status IN ('accepted', 'proposed')
        ORDER BY created_at DESC
    """
    if limit:
        query += f" LIMIT {limit}"

    rows = conn.execute(query).fetchall()
    conn.close()

    claims = []
    skipped = 0
    for row in rows:
        key = row["claim_key"]
        # Skip noise
        if any(key.startswith(prefix) for prefix in SKIP_PREFIXES):
            skipped += 1
            continue

        # Skip very short statements (< 20 chars = not useful)
        stmt = row["statement"] or ""
        if len(stmt) < 20:
            skipped += 1
            continue

        # Build RAG document: claim_key context + statement + rationale
        doc_parts = [f"[{row['scope_id']}] {key}"]
        doc_parts.append(stmt)
        if row["rationale"]:
            doc_parts.append(f"Rationale: {row['rationale']}")

        provenance = {}
        try:
            prov_raw = json.loads(row["provenance_json"] or "{}")
            provenance = prov_raw if isinstance(prov_raw, dict) else {}
        except (json.JSONDecodeError, TypeError):
            pass

        claims.append({
            "claim_id": row["claim_id"],
            "claim_key": key,
            "claim_type": row["claim_type"],
            "scope_id": row["scope_id"],
            "statement": stmt,
            "rationale": row["rationale"] or "",
            "confidence": row["confidence"] or 0.0,
            "produced_by": (provenance.get("produced_by") or {}).get("id", "unknown") if isinstance(provenance.get("produced_by"), dict) else str(provenance.get("produced_by", "unknown")),
            "created_at": row["created_at"],
            "document": "\n".join(doc_parts),  # full text for embedding
        })

    print(f"Exported {len(claims)} claims, skipped {skipped} noise")
    return claims


def embed_claims(claims: list[dict]) -> np.ndarray:
    """Embed claim documents with Sentence-BERT."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    documents = [c["document"] for c in claims]
    print(f"Embedding {len(documents)} documents (batch_size={BATCH_SIZE})...")
    start = time.time()
    embeddings = model.encode(
        documents,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # for cosine similarity via dot product
    )
    elapsed = time.time() - start
    print(f"Embedded in {elapsed:.1f}s ({len(documents)/elapsed:.0f} docs/sec)")
    return np.array(embeddings, dtype=np.float32)


def build_faiss_index(embeddings: np.ndarray) -> "faiss.Index":
    """Build FAISS index (flat IP for normalized vectors = cosine sim)."""
    import faiss

    dim = embeddings.shape[1]
    # Flat inner product on normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


def save_outputs(
    claims: list[dict],
    embeddings: np.ndarray,
    index: "faiss.Index",
    out_dir: Path,
):
    """Save JSONL manifest + FAISS index + embeddings."""
    import faiss

    out_dir.mkdir(parents=True, exist_ok=True)

    # JSONL manifest (one claim per line, no embeddings)
    jsonl_path = out_dir / "claims.jsonl"
    with open(jsonl_path, "w") as f:
        for c in claims:
            # Don't include 'document' field in output (redundant with statement)
            out = {k: v for k, v in c.items() if k != "document"}
            f.write(json.dumps(out) + "\n")
    print(f"Saved {len(claims)} claims to {jsonl_path}")

    # FAISS index
    index_path = out_dir / "claims.faiss"
    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index to {index_path}")

    # Numpy embeddings (for potential re-indexing)
    emb_path = out_dir / "claims_embeddings.npy"
    np.save(str(emb_path), embeddings)
    print(f"Saved embeddings to {emb_path} ({embeddings.nbytes / 1024 / 1024:.1f} MB)")

    # Metadata
    meta = {
        "model": MODEL_NAME,
        "dim": EMBEDDING_DIM,
        "count": len(claims),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "db_path": str(DEFAULT_DB),
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_path}")


def query_index(
    query: str,
    out_dir: Path,
    top_k: int = 5,
) -> list[dict]:
    """Query the FAISS index with a natural language question."""
    import faiss
    from sentence_transformers import SentenceTransformer

    # Load
    index = faiss.read_index(str(out_dir / "claims.faiss"))
    with open(out_dir / "claims.jsonl") as f:
        claims = [json.loads(line) for line in f]

    model = SentenceTransformer(MODEL_NAME)
    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype=np.float32)

    scores, indices = index.search(q_emb, top_k)
    results = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < 0:
            continue
        claim = claims[idx]
        claim["similarity"] = float(score)
        results.append(claim)
    return results


def main():
    parser = argparse.ArgumentParser(description="Export claims to RAG index")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Export only, no embedding")
    parser.add_argument("--query", type=str, help="Query existing index")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.query:
        results = query_index(args.query, args.out, args.top_k)
        print(f"\nTop {len(results)} results for: {args.query}\n")
        for i, r in enumerate(results):
            print(f"  {i+1}. [{r['scope_id']}] {r['claim_key']}")
            print(f"     Score: {r['similarity']:.4f} | Confidence: {r['confidence']}")
            print(f"     {r['statement'][:150]}")
            print()
        return

    # Export
    claims = export_claims(args.db, args.limit)
    if not claims:
        print("No claims to export")
        return

    if args.dry_run:
        print(f"Dry run: would embed {len(claims)} claims")
        # Show type distribution
        from collections import Counter
        types = Counter(c["claim_type"] for c in claims)
        scopes = Counter(c["scope_id"] for c in claims)
        print(f"Types: {dict(types)}")
        print(f"Scopes (top 10): {dict(scopes.most_common(10))}")
        return

    # Embed + index
    embeddings = embed_claims(claims)
    index = build_faiss_index(embeddings)
    save_outputs(claims, embeddings, index, args.out)

    print(f"\nDone! Query with: python {__file__} --query 'your question here'")


if __name__ == "__main__":
    main()
