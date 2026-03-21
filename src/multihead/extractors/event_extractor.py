"""Stage 5: Grounded event extraction."""

from __future__ import annotations

from typing import Any

from multihead.adapters.base import HeadAdapter
from multihead.chunker import Chunk
from multihead.extractors.base import BaseExtractor, ExtractorResult

PROMPT_TEMPLATE = """Extract events (things that happened) from this text. Output ONLY a JSON array, no explanation.

Each object: {{"event_type":"<TYPE>","title":"<SHORT TITLE>","summary":"<1-2 SENTENCES>","confidence":0.8}}
Types: decision, task_completed, tool_run, commit, note, milestone, question, answer

Example:
[{{"event_type":"decision","title":"Switched to 4-bit quantization","summary":"Team decided to use 4-bit quant for Qwen3 to fit in VRAM.","confidence":0.9}}]

Text:
{text}

JSON:"""


class EventExtractor(BaseExtractor):
    """Extract semantic events from text chunks."""

    async def extract(
        self, chunks: list[Chunk], adapter: HeadAdapter, **kwargs: Any
    ) -> ExtractorResult:
        all_events: list[dict[str, Any]] = []
        warnings: list[str] = []
        chunks_with_events = 0
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
                warnings.append(f"Event extraction failed for chunk {chunk.chunk_id}: {resp}")
                continue
            parsed = self.parse_json_response(resp.get("text", ""))
            if parsed:
                chunks_with_events += 1
            for evt in parsed:
                evt["source_chunk_id"] = chunk.chunk_id
                evt["source_record_id"] = chunk.record_id
                evt["source_span_start"] = chunk.span_start
            all_events.extend(parsed)

        coverage = chunks_with_events / max(len(chunks), 1)
        metrics = {
            "event_count": len(all_events),
            "event_extract_coverage": coverage,
            "chunks_processed": len(chunks),
        }

        return ExtractorResult(items=all_events, metrics=metrics, warnings=warnings)
