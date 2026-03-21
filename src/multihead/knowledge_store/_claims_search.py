"""Claim search operations: FTS5, vector (sqlite-vec), hybrid, and LIKE fallback."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from ._retry import _sqlite_retry

# Stop words to filter from FTS queries
_STOP_WORDS = frozenset({
    "what", "how", "are", "the", "is", "was", "were", "and", "for", "its",
    "does", "from", "that", "this", "with", "has", "have", "had", "not", "but",
    "can", "did", "will", "would", "should", "could", "who", "which", "where",
    "when", "why", "many", "much", "use", "also", "been", "being", "all",
    "into", "than", "then", "them", "they", "their", "there", "these", "those",
    "about", "each", "make", "like", "just", "over", "such", "take", "other",
    "some", "only", "very", "after", "before", "between", "our", "you", "your",
})


def _clean_keywords(query: str, min_len: int = 3) -> list[str]:
    """Extract clean keywords from a query string.

    Strips punctuation, removes stop words, deduplicates.
    """
    words = re.findall(r'[a-zA-Z0-9_]+', query.lower())
    seen: set[str] = set()
    keywords: list[str] = []
    for w in words:
        if len(w) >= min_len and w not in _STOP_WORDS and w not in seen:
            seen.add(w)
            keywords.append(w)
    return keywords


class ClaimsSearchMixin:
    """Mixin providing claim search operations.

    Supports FTS5, vector similarity (sqlite-vec), hybrid RRF,
    and LIKE fallback. Expects self._connect() from the main class.
    """

    def search_claims_fts(
        self, query: str, limit: int = 10,
        min_confidence: float = 0.0,
        max_age_days: int | None = None,
    ) -> list[tuple[str, str, float]]:
        """Full-text search for claims using FTS5 MATCH.

        Args:
            query: Natural language query
            limit: Maximum results
            min_confidence: Minimum confidence threshold (0.0 = no filter)
            max_age_days: Only return claims within this many days (None = no filter)

        Returns:
            List of (claim_key, statement, confidence) tuples, ranked by relevance.
            Falls back to LIKE-based search if FTS5 is unavailable.
        """
        if not query.strip():
            return []

        try:
            with self._connect() as conn:
                keywords = _clean_keywords(query)
                if not keywords:
                    return []

                # FTS5 query with OR-combined clean keywords
                fts_query = " OR ".join(keywords)

                extra_where = ""
                params: list[Any] = [fts_query]
                if min_confidence > 0:
                    extra_where += " AND c.confidence >= ?"
                    params.append(min_confidence)
                if max_age_days is not None:
                    cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
                    extra_where += " AND c.created_at > ?"
                    params.append(cutoff)
                params.append(limit)

                # Use bm25 with claim_key boosted 5x over statement
                rows = conn.execute(
                    f"""
                    SELECT c.claim_key, c.statement, c.confidence
                    FROM claims_fts f
                    JOIN claims c ON c.rowid = f.rowid
                    WHERE claims_fts MATCH ?
                      AND c.claim_status IN ('accepted', 'corroborated', 'proposed')
                      {extra_where}
                    ORDER BY bm25(claims_fts, 5.0, 1.0)
                    LIMIT ?
                    """,
                    params,
                ).fetchall()
                return [(r[0] or "", r[1] or "", r[2] or 0.0) for r in rows]

        except Exception:
            # Fallback: LIKE-based search
            return self._search_claims_like(query, limit, min_confidence, max_age_days)

    def search_claims_vec(
        self, query: str, limit: int = 10,
        min_confidence: float = 0.0,
    ) -> list[tuple[str, str, float, float]]:
        """Vector similarity search using sqlite-vec embeddings.

        Args:
            query: Natural language query
            limit: Maximum results
            min_confidence: Minimum confidence threshold

        Returns:
            List of (claim_key, statement, confidence, distance) tuples.
            Returns empty list if sqlite-vec or embeddings not available.
        """
        try:
            model = self._get_embedding_model()
            if model is None:
                return []

            import sqlite_vec
            query_emb = model.encode(query, normalize_embeddings=True)

            with self._connect() as conn:
                sqlite_vec.load(conn)
                rows = conn.execute(
                    """
                    SELECT c.claim_key, c.statement, c.confidence, v.distance
                    FROM claims_vec v
                    JOIN claims c ON c.rowid = v.rowid
                    WHERE v.embedding MATCH ?
                      AND c.claim_status IN ('accepted', 'corroborated', 'proposed')
                      AND k = ?
                    ORDER BY v.distance
                    """,
                    [query_emb.tobytes(), limit * 2],
                ).fetchall()

                results = []
                for r in rows:
                    if min_confidence > 0 and (r[2] or 0) < min_confidence:
                        continue
                    results.append((r[0] or "", r[1] or "", r[2] or 0.0, r[3]))
                    if len(results) >= limit:
                        break
                return results

        except Exception:
            return []

    def search_claims_hybrid(
        self, query: str, limit: int = 10,
        min_confidence: float = 0.0,
        rrf_k: int = 60,
    ) -> list[tuple[str, str, float]]:
        """Hybrid search: FTS5 + vector similarity with Reciprocal Rank Fusion.

        Combines keyword and semantic search for best retrieval quality.
        Falls back to FTS-only if vector search is unavailable.

        Args:
            query: Natural language query
            limit: Maximum results
            min_confidence: Minimum confidence threshold
            rrf_k: RRF constant (default 60, higher = more weight to lower ranks)

        Returns:
            List of (claim_key, statement, confidence) tuples, ranked by RRF score.
        """
        # Get FTS results
        fts_results = self.search_claims_fts(query, limit=limit * 3, min_confidence=min_confidence)

        # Get vector results
        vec_results = self.search_claims_vec(query, limit=limit * 3, min_confidence=min_confidence)

        if not vec_results:
            # No vector search available — return FTS only
            return fts_results[:limit]

        # RRF fusion
        scores: dict[str, float] = {}
        claim_data: dict[str, tuple[str, float]] = {}  # key → (statement, confidence)

        for rank, (key, stmt, conf) in enumerate(fts_results):
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            claim_data[key] = (stmt, conf)

        for rank, (key, stmt, conf, _dist) in enumerate(vec_results):
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            if key not in claim_data:
                claim_data[key] = (stmt, conf)

        # Sort by RRF score
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        results = []
        for key, _score in ranked[:limit]:
            stmt, conf = claim_data[key]
            results.append((key, stmt, conf))

        return results

    def _get_embedding_model(self):
        """Lazy-load the sentence-transformers embedding model."""
        if not hasattr(self, "_embedding_model"):
            self._embedding_model = None
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(
                    "all-MiniLM-L6-v2",
                )
            except ImportError:
                pass
        return self._embedding_model

    def _search_claims_like(
        self, query: str, limit: int = 10,
        min_confidence: float = 0.0,
        max_age_days: int | None = None,
    ) -> list[tuple[str, str, float]]:
        """Fallback keyword search using SQL LIKE."""
        keywords = _clean_keywords(query)
        if not keywords:
            return []
        try:
            with self._connect() as conn:
                where_parts = ["LOWER(statement) LIKE ?"] * len(keywords[:8])
                params: list[Any] = [f"%{kw}%" for kw in keywords[:8]]

                extra_where = ""
                if min_confidence > 0:
                    extra_where += " AND confidence >= ?"
                    params.append(min_confidence)
                if max_age_days is not None:
                    cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
                    extra_where += " AND created_at > ?"
                    params.append(cutoff)
                params.append(limit)

                rows = conn.execute(
                    f"""
                    SELECT claim_key, statement, confidence
                    FROM claims
                    WHERE claim_status IN ('accepted', 'corroborated', 'proposed')
                      AND ({" OR ".join(where_parts)})
                      {extra_where}
                    ORDER BY confidence DESC
                    LIMIT ?
                    """,
                    params,
                ).fetchall()
                return [(r[0] or "", r[1] or "", r[2] or 0.0) for r in rows]
        except Exception:
            return []
