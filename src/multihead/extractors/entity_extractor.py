"""Stage 3: Entity extraction and canonicalization."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from multihead.adapters.base import HeadAdapter
from multihead.chunker import Chunk
from multihead.extractors.base import BaseExtractor, ExtractorResult

PROMPT_TEMPLATE = """Extract named entities from this text. Output ONLY a JSON array, no explanation.

Each object: {{"entity_type":"<TYPE>","entity_id":"<ID>","label":"<NAME>","aliases":[]}}
Types: project, repo, file, model, tool, company, person, concept, component
entity_id: lowercase hyphen-separated (e.g. "multihead", "qwen3-8b")

Example output:
[{{"entity_type":"model","entity_id":"qwen3-8b","label":"Qwen3-8B","aliases":["qwen"]}}]

Text:
{text}

JSON:"""


class EntityExtractor(BaseExtractor):
    """Extract entities from text chunks and canonicalize names."""

    async def extract(
        self, chunks: list[Chunk], adapter: HeadAdapter, **kwargs: Any
    ) -> ExtractorResult:
        all_entities: list[dict[str, Any]] = []
        total_tokens_in = 0
        warnings: list[str] = []
        concurrency = kwargs.get("concurrency", 1)

        prompts = [PROMPT_TEMPLATE.format(text=chunk.text) for chunk in chunks]
        responses = await self.map_generate(
            adapter, prompts, concurrency=concurrency,
            checkpoint_dir=kwargs.get("checkpoint_dir"),
            stage_name=kwargs.get("stage_name", ""),
            batch_mode=kwargs.get("batch_mode", False),
            no_wait=kwargs.get("no_wait", False),
        )

        for chunk, resp in zip(chunks, responses):
            if isinstance(resp, Exception):
                warnings.append(f"Entity extraction failed for chunk {chunk.chunk_id}: {resp}")
                continue
            parsed = self.parse_json_response(resp.get("text", ""))
            for ent in parsed:
                ent["source_chunk_id"] = chunk.chunk_id
                ent["source_record_id"] = chunk.record_id
            all_entities.extend(parsed)
            total_tokens_in += resp.get("tokens_in", len(chunk.text) // 4)

        # Canonicalize: merge entities with same entity_id or overlapping aliases
        deduped = self._canonicalize(all_entities)

        # Compute metrics
        token_count_k = max(total_tokens_in / 1000, 0.001)
        metrics = {
            "entity_count": len(deduped),
            "entity_yield_per_1k_tokens": len(deduped) / token_count_k,
            "alias_conflict_rate": self._compute_alias_conflicts(deduped),
            "chunks_processed": len(chunks),
        }

        return ExtractorResult(items=deduped, metrics=metrics, warnings=warnings)

    def _canonicalize(self, entities: list[dict]) -> list[dict]:
        """Merge entities with the same entity_id."""
        by_id: dict[str, dict] = {}
        for ent in entities:
            eid = ent.get("entity_id", "").lower().strip()
            if not eid:
                continue
            if eid in by_id:
                # Merge aliases
                existing = by_id[eid]
                existing_aliases = set(existing.get("aliases", []))
                new_aliases = set(ent.get("aliases", []))
                existing_aliases.update(new_aliases)
                if ent.get("label"):
                    existing_aliases.add(ent["label"])
                existing["aliases"] = list(existing_aliases)
            else:
                by_id[eid] = {
                    "entity_type": ent.get("entity_type", "concept"),
                    "entity_id": eid,
                    "label": ent.get("label", eid),
                    "aliases": list(set(ent.get("aliases", []))),
                }
        return list(by_id.values())

    def _compute_alias_conflicts(self, entities: list[dict]) -> float:
        """Compute rate of alias conflicts (same alias → different entity_id)."""
        alias_to_ids: dict[str, set[str]] = defaultdict(set)
        for ent in entities:
            for alias in ent.get("aliases", []):
                alias_to_ids[alias.lower()].add(ent["entity_id"])
        if not alias_to_ids:
            return 0.0
        conflicts = sum(1 for ids in alias_to_ids.values() if len(ids) > 1)
        return conflicts / len(alias_to_ids)
