"""Embedding-based semantic search for knowledge claims.

Uses sentence-transformers to build a vector index of claim statements,
enabling conceptual queries (e.g. "portability" finds "friends can install it").

The index is built lazily on first query and cached on disk as a numpy .npz file.
Incremental updates add new claims without full rebuild.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default model — small (80MB), fast, good quality for short text
DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Cache file names
_EMBEDDINGS_FILE = "claim_embeddings.npz"
_IDS_FILE = "claim_ids.json"
_IDS_FILE_LEGACY = "claim_ids.npy"  # Old pickle-based format


class EmbeddingIndex:
    """In-memory vector index for knowledge claim semantic search.

    Lazily loads sentence-transformers model and builds/loads cached embeddings.
    """

    def __init__(
        self,
        db_path: Path,
        cache_dir: Path | None = None,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        self.db_path = db_path
        self.cache_dir = cache_dir or db_path.parent
        self.model_name = model_name

        self._model: Any = None
        self._embeddings: np.ndarray | None = None
        self._claim_ids: list[str] = []
        self._statements: list[str] = []
        self._claim_keys: list[str] = []
        self._confidences: list[float] = []
        self._built = False

    @property
    def _embeddings_path(self) -> Path:
        return self.cache_dir / _EMBEDDINGS_FILE

    @property
    def _ids_path(self) -> Path:
        return self.cache_dir / _IDS_FILE

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", self.model_name)
            t0 = time.monotonic()
            self._model = SentenceTransformer(self.model_name)
            logger.info("Model loaded in %.1fs", time.monotonic() - t0)
        return self._model

    def _fetch_claims(self) -> list[tuple[str, str, str, float]]:
        """Fetch (claim_id, claim_key, statement, confidence) from DB."""
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.execute("PRAGMA busy_timeout = 5000")
        rows = conn.execute(
            """
            SELECT claim_id, claim_key, statement, confidence
            FROM claims
            WHERE claim_status IN ('accepted', 'proposed')
              AND LENGTH(statement) > 10
            ORDER BY claim_id
            """,
        ).fetchall()
        conn.close()
        return [(r[0], r[1] or "", r[2], r[3] or 0.0) for r in rows]

    def build(self, force: bool = False) -> int:
        """Build or refresh the embedding index.

        Returns number of claims indexed.
        """
        claims = self._fetch_claims()
        if not claims:
            logger.warning("No claims to index")
            return 0

        current_ids = {c[0] for c in claims}

        # Check cache validity
        if not force and self._embeddings_path.exists() and self._ids_path.exists():
            try:
                cached_data = np.load(str(self._embeddings_path))
                cached_ids = json.loads(self._ids_path.read_text())
                cached_set = set(cached_ids)

                new_claims = [c for c in claims if c[0] not in cached_set]
                if not new_claims:
                    # Cache is up to date
                    self._embeddings = cached_data["embeddings"]
                    self._claim_ids = cached_ids
                    self._statements = [c[2] for c in claims if c[0] in cached_set]
                    self._claim_keys = [c[1] for c in claims if c[0] in cached_set]
                    self._confidences = [c[3] for c in claims if c[0] in cached_set]
                    self._built = True
                    logger.info("Loaded cached index: %d claims", len(cached_ids))
                    return len(cached_ids)

                # Incremental: embed only new claims
                logger.info("Incremental update: %d new claims", len(new_claims))
                model = self._load_model()
                new_texts = [c[2] for c in new_claims]
                new_embeddings = model.encode(new_texts, show_progress_bar=False,
                                              normalize_embeddings=True)

                # Merge
                self._embeddings = np.vstack([cached_data["embeddings"], new_embeddings])
                self._claim_ids = cached_ids + [c[0] for c in new_claims]
                # Re-fetch all metadata in order
                id_to_claim = {c[0]: c for c in claims}
                self._statements = [id_to_claim[cid][2] for cid in self._claim_ids if cid in id_to_claim]
                self._claim_keys = [id_to_claim[cid][1] for cid in self._claim_ids if cid in id_to_claim]
                self._confidences = [id_to_claim[cid][3] for cid in self._claim_ids if cid in id_to_claim]

            except Exception as e:
                logger.warning("Cache load failed, full rebuild: %s", e)
                force = True

        if force or not self._built:
            # Full build
            model = self._load_model()
            texts = [c[2] for c in claims]
            logger.info("Encoding %d claims...", len(texts))
            t0 = time.monotonic()
            self._embeddings = model.encode(texts, show_progress_bar=False,
                                            normalize_embeddings=True,
                                            batch_size=256)
            logger.info("Encoded in %.1fs", time.monotonic() - t0)

            self._claim_ids = [c[0] for c in claims]
            self._statements = [c[2] for c in claims]
            self._claim_keys = [c[1] for c in claims]
            self._confidences = [c[3] for c in claims]

        # Save cache
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(self._embeddings_path), embeddings=self._embeddings)
            self._ids_path.write_text(json.dumps(self._claim_ids))
            # Remove legacy pickle file if it exists
            legacy = self.cache_dir / _IDS_FILE_LEGACY
            if legacy.exists():
                legacy.unlink()
            logger.info("Saved embedding cache: %d claims", len(self._claim_ids))
        except Exception as e:
            logger.warning("Failed to save cache: %s", e)

        self._built = True
        return len(self._claim_ids)

    def search(
        self,
        query: str,
        limit: int = 20,
        min_score: float = 0.15,
    ) -> list[tuple[str, str, float, float]]:
        """Semantic search for claims.

        Args:
            query: Natural language query
            limit: Max results
            min_score: Minimum cosine similarity threshold

        Returns:
            List of (claim_key, statement, confidence, similarity_score) tuples,
            sorted by similarity descending.
        """
        if not self._built:
            self.build()

        if self._embeddings is None or len(self._claim_ids) == 0:
            return []

        model = self._load_model()
        query_embedding = model.encode([query], normalize_embeddings=True)

        # Cosine similarity (embeddings are already normalized)
        scores = np.dot(self._embeddings, query_embedding.T).flatten()

        # Top-k above threshold
        mask = scores >= min_score
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            return []

        # Sort by score descending
        top_indices = valid_indices[np.argsort(scores[valid_indices])[::-1]][:limit]

        results = []
        for idx in top_indices:
            results.append((
                self._claim_keys[idx],
                self._statements[idx],
                self._confidences[idx],
                float(scores[idx]),
            ))
        return results

    @property
    def indexed_count(self) -> int:
        return len(self._claim_ids) if self._built else 0
