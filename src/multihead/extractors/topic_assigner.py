"""Stage 4: Topic assignment (clustering chunks into topics)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from multihead.adapters.base import HeadAdapter
from multihead.chunker import Chunk
from multihead.extractors.base import BaseExtractor, ExtractorResult

PROMPT_TEMPLATE = """Analyze the following text chunks and assign each to a topic.
Return a JSON array where each object has:
- chunk_id: the chunk identifier
- topic_id: a short kebab-case topic name (e.g. "head-management", "event-sourcing")
- topic_label: a human-readable topic name
- confidence: a number 0-1 indicating assignment confidence

Chunks:
{chunks_text}

Return ONLY the JSON array, no other text."""


class TopicAssigner(BaseExtractor):
    """Assign chunks to topics via LLM clustering."""

    async def extract(
        self, chunks: list[Chunk], adapter: HeadAdapter, **kwargs: Any
    ) -> ExtractorResult:
        if not chunks:
            return ExtractorResult(metrics={"unassigned_chunk_rate": 0.0, "topic_coherence": 1.0})

        # Process in batches to fit context windows
        batch_size = kwargs.get("batch_size", 10)
        concurrency = kwargs.get("concurrency", 1)
        all_assignments: list[dict[str, Any]] = []
        warnings: list[str] = []

        prompts: list[str] = []
        batch_indices: list[int] = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            chunks_text = "\n---\n".join(
                f"[{c.chunk_id}]: {c.text}" for c in batch
            )
            prompts.append(PROMPT_TEMPLATE.format(chunks_text=chunks_text))
            batch_indices.append(i)

        responses = await self.map_generate(
            adapter, prompts, concurrency=concurrency,
            checkpoint_dir=kwargs.get("checkpoint_dir"),
            stage_name=kwargs.get("stage_name", ""),
            batch_mode=kwargs.get("batch_mode", False),
            no_wait=kwargs.get("no_wait", False),
            on_chunk_progress=kwargs.get("on_chunk_progress"),
        )

        for idx, resp in zip(batch_indices, responses):
            if isinstance(resp, Exception):
                warnings.append(f"Topic assignment failed for batch {idx}: {resp}")
                continue
            parsed = self.parse_json_response(resp.get("text", ""))
            all_assignments.extend(parsed)

        # Compute metrics
        assigned_ids = {a.get("chunk_id") for a in all_assignments if a.get("topic_id")}
        total_chunks = len(chunks)
        unassigned = total_chunks - len(assigned_ids)
        unassigned_rate = unassigned / max(total_chunks, 1)

        # Topic coherence: rough measure — how many topics vs chunks
        topics = {a.get("topic_id") for a in all_assignments if a.get("topic_id")}
        coherence = 1.0 - min(len(topics) / max(total_chunks, 1), 1.0) if topics else 0.0

        metrics = {
            "unassigned_chunk_rate": unassigned_rate,
            "topic_coherence": coherence,
            "topic_count": len(topics),
            "chunks_processed": total_chunks,
        }

        return ExtractorResult(items=all_assignments, metrics=metrics, warnings=warnings)
