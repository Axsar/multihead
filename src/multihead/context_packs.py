"""Context Pack builder with ranking formula and token-budget trimming."""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from multihead.knowledge_models import (
    ClaimStatus,
    ContextPack,
    PackItem,
)
from multihead.knowledge_store import KnowledgeStore

logger = logging.getLogger(__name__)


# Default ranking weights
DEFAULT_WEIGHTS = {
    "relevance": 0.3,
    "trust": 0.3,
    "recency": 0.3,
    "frequency": 0.1,
}

# Trust scores by status
TRUST_SCORES = {
    "accepted": 1.0,
    "confirmed": 1.0,
    "corroborated": 0.85,  # Second independent channel agrees — high trust but not yet fully accepted
    "proposed": 0.5,
    "draft": 0.5,
    "stale": 0.25,  # Source file changed since claim was derived — needs re-verification
    "contested": 0.3,
    "deprecated": 0.1,
    "superseded": 0.0,
    "rejected": 0.0,
    "retracted": 0.0,
}

# Recency half-life in days
RECENCY_HALF_LIFE_DAYS = 7.0


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class PackBuilder:
    """Build context packs from knowledge store data."""

    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        packs_dir: Path,
        embedding_adapter: Any | None = None,
    ) -> None:
        self.knowledge = knowledge_store
        self.packs_dir = packs_dir
        self.packs_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_adapter = embedding_adapter

    async def _embed_query(self, query: str) -> list[float] | None:
        """Embed a query string. Returns None if no embedding adapter."""
        if self.embedding_adapter is None:
            return None
        try:
            result = await self.embedding_adapter.embed(query)
            return result.get("embedding")
        except Exception as e:
            logger.warning("Failed to embed query: %s", e)
            return None

    async def _embed_text(self, text: str) -> list[float] | None:
        """Embed an item text. Returns None if no embedding adapter."""
        if self.embedding_adapter is None:
            return None
        try:
            result = await self.embedding_adapter.embed(text)
            return result.get("embedding")
        except Exception as e:
            logger.warning("Failed to embed text: %s", e)
            return None

    def build_pack(
        self,
        purpose: str,
        filters: dict[str, Any] | None = None,
        budgets: dict[str, int] | None = None,
        weights: dict[str, float] | None = None,
        query: str | None = None,
        query_embedding: list[float] | None = None,
    ) -> ContextPack:
        """Build a pack: query claims/events, rank, trim to budget, write files.

        Args:
            query_embedding: Pre-computed query embedding. If provided along with
                an embedding_adapter, items will be scored by cosine similarity
                against this vector instead of using the default 0.5 relevance.
        """
        filters = filters or {}
        budgets = budgets or {"max_tokens": 100000, "max_items": 100000}
        weights = weights or DEFAULT_WEIGHTS

        # Gather candidates
        candidates = self._gather_candidates(filters)

        # Score and sort
        for item in candidates:
            item.priority = self._score_item(item, weights, query_embedding=query_embedding)
        candidates.sort(key=lambda x: x.priority, reverse=True)

        # Trim to budget
        max_tokens = budgets.get("max_tokens", 100000)
        max_items = budgets.get("max_items", 100000)
        kept, dropped = self._trim_to_budget(candidates, max_tokens, max_items)

        pack = ContextPack(
            purpose=purpose,
            budgets=budgets,
            items=kept,
            dropped=dropped,
            metrics={
                "token_total": sum(i.token_estimate for i in kept),
                "item_count": len(kept),
                "dropped_count": len(dropped),
            },
        )

        # Write files
        self._write_pack_files(pack)
        return pack

    async def build_pack_with_embeddings(
        self,
        purpose: str,
        query: str,
        filters: dict[str, Any] | None = None,
        budgets: dict[str, int] | None = None,
        weights: dict[str, float] | None = None,
    ) -> ContextPack:
        """Build a pack using semantic similarity for relevance scoring.

        Embeds the query and all candidate items, then delegates to build_pack()
        with the pre-computed query embedding. Falls back to default relevance
        (0.5) if embedding fails.
        """
        query_embedding = await self._embed_query(query)

        # Pre-embed candidates if we have a query embedding
        if query_embedding is not None:
            filters = filters or {}
            candidates = self._gather_candidates(filters)
            texts = [item.text for item in candidates]
            if texts:
                try:
                    batch_result = await self.embedding_adapter.embed_batch(texts)
                    embeddings = batch_result.get("embeddings", [])
                    for item, emb in zip(candidates, embeddings):
                        item.embedding = emb
                except Exception as e:
                    logger.warning("Batch embedding failed, falling back: %s", e)

            # Now score with embeddings
            weights = weights or DEFAULT_WEIGHTS
            budgets = budgets or {"max_tokens": 100000, "max_items": 100000}
            for item in candidates:
                item.priority = self._score_item(item, weights, query_embedding=query_embedding)
            candidates.sort(key=lambda x: x.priority, reverse=True)

            max_tokens = budgets.get("max_tokens", 100000)
            max_items = budgets.get("max_items", 100000)
            kept, dropped = self._trim_to_budget(candidates, max_tokens, max_items)

            pack = ContextPack(
                purpose=purpose,
                budgets=budgets,
                items=kept,
                dropped=dropped,
                metrics={
                    "token_total": sum(i.token_estimate for i in kept),
                    "item_count": len(kept),
                    "dropped_count": len(dropped),
                    "embedding_powered": True,
                },
            )
            self._write_pack_files(pack)
            return pack

        # No embeddings available — fall back to synchronous build
        return self.build_pack(
            purpose=purpose, filters=filters, budgets=budgets, weights=weights,
        )

    def _gather_candidates(self, filters: dict) -> list[PackItem]:
        """Query knowledge store for pack item candidates."""
        items: list[PackItem] = []

        scope_id = filters.get("scope_id")
        since_str = filters.get("time_range")
        since = None
        if since_str:
            days = int(since_str.rstrip("d"))
            since = datetime.now(timezone.utc) - timedelta(days=days)

        # Accepted claims
        claims = self.knowledge.get_accepted_claims_for_pack(scope_id=scope_id, since=since)
        for c in claims:
            items.append(PackItem(
                type="claim",
                text=c.statement,
                evidence_refs=[c.claim_id] + c.derived_from_event_ids,
                token_estimate=self._estimate_tokens(c.statement),
                why_included=f"accepted {c.claim_type.value}",
                source_id=c.claim_id,
                status=c.claim_status.value,
                updated_at=c.provenance.updated_at,
            ))

        # Confirmed events
        events = self.knowledge.get_confirmed_events_for_pack(since=since)
        for e in events:
            text = f"{e.title}: {e.summary}" if e.summary else e.title
            items.append(PackItem(
                type="event",
                text=text,
                evidence_refs=[e.event_id],
                token_estimate=self._estimate_tokens(text),
                why_included=f"confirmed {e.event_type.value}",
                source_id=e.event_id,
                status=e.event_status.value,
                updated_at=e.provenance.updated_at,
            ))

        # Open loops
        loops = self.knowledge.get_open_loops()
        for loop in loops:
            text = f"[OPEN] {loop.title}"
            items.append(PackItem(
                type="event",
                text=text,
                evidence_refs=[loop.event_id],
                token_estimate=self._estimate_tokens(text),
                why_included="open loop",
                source_id=loop.event_id,
                status=loop.event_status.value,
                updated_at=loop.provenance.updated_at,
                priority=999.0,  # Open loops always kept
            ))

        return items

    def _score_item(
        self,
        item: PackItem,
        weights: dict[str, float],
        query_embedding: list[float] | None = None,
    ) -> float:
        """Compute: w_r*relevance + w_t*trust + w_c*recency + w_f*frequency - penalties."""
        w_r = weights.get("relevance", 0.3)
        w_t = weights.get("trust", 0.3)
        w_c = weights.get("recency", 0.3)
        w_f = weights.get("frequency", 0.1)

        # If already scored high (e.g., open loops), keep that score
        if item.priority > 100:
            return item.priority

        relevance = self._compute_relevance(item, query_embedding)
        trust = self._compute_trust(item)
        recency = self._compute_recency(item.updated_at)
        frequency = 0.5  # Default (could track reference counts)

        score = w_r * relevance + w_t * trust + w_c * recency + w_f * frequency

        # Penalties
        if item.status in ("superseded", "rejected", "retracted"):
            score -= 1.0

        return score

    def _compute_relevance(
        self, item: PackItem, query_embedding: list[float] | None = None,
    ) -> float:
        """Compute relevance score using cosine similarity if embeddings available."""
        if query_embedding is None:
            return 0.5

        item_embedding = getattr(item, "embedding", None)
        if item_embedding is None:
            return 0.5

        sim = cosine_similarity(query_embedding, item_embedding)
        # Normalize: cosine similarity is [-1, 1], map to [0, 1]
        return max(0.0, min(1.0, (sim + 1.0) / 2.0))

    def _compute_trust(self, item: PackItem) -> float:
        return TRUST_SCORES.get(item.status, 0.5)

    def _compute_recency(self, updated_at: datetime) -> float:
        """Exponential decay from updated_at."""
        now = datetime.now(timezone.utc)
        age_days = (now - updated_at).total_seconds() / 86400
        return math.exp(-age_days * math.log(2) / RECENCY_HALF_LIFE_DAYS)

    def _trim_to_budget(
        self, items: list[PackItem], max_tokens: int, max_items: int,
    ) -> tuple[list[PackItem], list[dict[str, str]]]:
        """Trim by priority: keep highest-priority items within budget."""
        kept: list[PackItem] = []
        dropped: list[dict[str, str]] = []
        token_total = 0

        for item in items:
            if len(kept) >= max_items:
                dropped.append({"item": item.source_id, "reason": "over_max_items"})
                continue
            if token_total + item.token_estimate > max_tokens:
                dropped.append({"item": item.source_id, "reason": "over_budget"})
                continue
            kept.append(item)
            token_total += item.token_estimate

        return kept, dropped

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token."""
        return max(1, len(text) // 4)

    def _write_pack_files(self, pack: ContextPack) -> tuple[Path, Path]:
        """Write <pack_id>.md and <pack_id>.pack.json files."""
        md_path = self.packs_dir / f"{pack.pack_id}.md"
        json_path = self.packs_dir / f"{pack.pack_id}.pack.json"

        # Write markdown
        lines = [f"# {pack.purpose}", f"*Built: {pack.created_at.isoformat()}*", ""]
        for item in pack.items:
            prefix = f"[{item.type.upper()}]"
            refs = ", ".join(item.evidence_refs[:3])
            lines.append(f"- {prefix} {item.text} [{refs}]")
        md_path.write_text("\n".join(lines), encoding="utf-8")

        # Write JSON
        json_path.write_text(
            json.dumps(pack.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )

        return md_path, json_path

    def build_standard_packs(self) -> list[ContextPack]:
        """Build standard Night Shift packs — no artificial limits."""
        packs = []

        packs.append(self.build_pack(
            purpose="Active Projects",
            filters={"time_range": "30d"},
        ))

        packs.append(self.build_pack(
            purpose="Recent Decisions",
            filters={"time_range": "7d"},
        ))

        packs.append(self.build_pack(
            purpose="Constraints",
            filters={},
        ))

        packs.append(self.build_pack(
            purpose="Glossary",
            filters={},
        ))

        packs.append(self.build_pack(
            purpose="Open Loops",
            filters={},
        ))

        return packs

    def list_packs(self) -> list[dict[str, Any]]:
        """List all built packs."""
        packs = []
        for json_path in sorted(self.packs_dir.glob("*.pack.json")):
            data = json.loads(json_path.read_text(encoding="utf-8"))
            packs.append({
                "pack_id": data.get("pack_id", ""),
                "purpose": data.get("purpose", ""),
                "item_count": len(data.get("items", [])),
                "created_at": data.get("created_at", ""),
            })
        return packs

    def load_pack(self, pack_id: str) -> ContextPack | None:
        """Load a pack from its .pack.json file."""
        json_path = self.packs_dir / f"{pack_id}.pack.json"
        if not json_path.exists():
            return None
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return ContextPack.model_validate(data)
