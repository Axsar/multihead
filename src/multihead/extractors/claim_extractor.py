"""Stage 6: Claim extraction with evidence backing."""

from __future__ import annotations

from typing import Any

from multihead.adapters.base import HeadAdapter
from multihead.chunker import Chunk
from multihead.extractors.base import BaseExtractor, ExtractorResult

PROMPT_TEMPLATE = """Extract DURABLE KNOWLEDGE from this text — decisions, architecture, outcomes, and constraints that will still be true and useful weeks from now.

DO extract:
- Decisions: "We chose X because Y" — WHY something was decided
- Architecture: "Component A connects to B via C" — how things are structured
- Outcomes: "Approach X worked/failed because Y" — results with reasons
- Constraints: "X must always be true" — rules and invariants
- Definitions: "X is Y" — what something means in this project

DO NOT extract:
- Debug measurements ("width: 115px", "y=3452", "span=970")
- Parameter snapshots ("margin=5", "epsilon=2.0", "jb=1.0")
- Git stats ("1 file changed", "commit abc123")
- Temporary state ("still broken", "not working yet")
- One-word or fragment observations

Each claim MUST be a complete, self-contained sentence of at least 50 characters that would be useful to someone who hasn't read this conversation.

For each claim, also identify:
- speaker: who made this claim? "human" (user typed it), "assistant" (AI said it), "bash_output" (command output showed it)
- observation_method: how was this observed? "user_statement", "assistant_statement", "bash_output", "code_read", "document_read"
- evidence: what specific text supports this claim? Include a short quote.
- file_path: if a specific file is mentioned, include its path

Output ONLY a JSON array:
[{{"claim_type":"<TYPE>","claim_key":"<KEY>","statement":"<SENTENCE>","confidence":0.8,"entity_type":"<ETYPE>","entity_id":"<EID>","predicate":"<PRED>","object_value":"<VAL>","durability":"durable","speaker":"<human|assistant|bash_output>","observation_method":"<METHOD>","evidence":[{{"type":"<TYPE>","text":"<QUOTE>"}}],"file_path":"<PATH_IF_ANY>"}}]

Types: decision, definition, fact, constraint, preference, plan
claim_key: dot-separated hierarchy (e.g. "multihead.router.scoring_weights")

Confidence rules:
- User stated directly → 0.8-0.9
- Bash output confirmed → 0.85-0.95
- Assistant inferred → 0.5-0.7 (needs corroboration)
- No evidence cited → cap at 0.4

Example:
[{{"claim_type":"decision","claim_key":"multihead.adapter.selection","statement":"We chose the Transformers adapter over Ollama for Qwen because it gives direct control over quantization and avoids the overhead of an HTTP round-trip for local inference.","confidence":0.9,"entity_type":"component","entity_id":"adapter","predicate":"chosen_because","object_value":"direct quantization control + no HTTP overhead","durability":"durable","speaker":"human","observation_method":"user_statement","evidence":[{{"type":"user_statement","text":"lets use transformers directly, ollama adds unnecessary HTTP overhead"}}],"file_path":"src/multihead/adapters/transformers_adapter.py"}}]

Text:
{text}

JSON:"""


class ClaimExtractor(BaseExtractor):
    """Extract claims from text chunks with evidence backing."""

    def __init__(
        self,
        auto_accept_confidence: float = 0.85,
        auto_accept_min_supports: int = 2,
        auto_accept_types: tuple[str, ...] = ("definition", "decision", "constraint"),
    ):
        self.auto_accept_confidence = auto_accept_confidence
        self.auto_accept_min_supports = auto_accept_min_supports
        self.auto_accept_types = auto_accept_types

    async def extract(
        self, chunks: list[Chunk], adapter: HeadAdapter, **kwargs: Any
    ) -> ExtractorResult:
        all_claims: list[dict[str, Any]] = []
        warnings: list[str] = []
        concurrency = kwargs.get("concurrency", 1)

        prompts = [PROMPT_TEMPLATE.format(text=chunk.text) for chunk in chunks]
        responses = await self.map_generate(
            adapter, prompts, concurrency=concurrency,
            checkpoint_dir=kwargs.get("checkpoint_dir"),
            stage_name=kwargs.get("stage_name", ""),
            batch_mode=kwargs.get("batch_mode", False),
            no_wait=kwargs.get("no_wait", False),
            on_chunk_progress=kwargs.get("on_chunk_progress"),
        )

        import logging as _log
        _logger = _log.getLogger(__name__)

        skipped_short = 0
        skipped_ephemeral = 0
        total_chunks = len(chunks)
        for chunk_idx, (chunk, resp) in enumerate(zip(chunks, responses)):
            if (chunk_idx + 1) % 500 == 0:
                _logger.info("Processing chunk %d/%d (%d claims so far)", chunk_idx + 1, total_chunks, len(all_claims))
            if isinstance(resp, Exception):
                warnings.append(f"Claim extraction failed for chunk {chunk.chunk_id}: {resp}")
                continue
            parsed = self.parse_json_response(resp.get("text", ""))
            for claim in parsed:
                # Quality gate: minimum statement length
                stmt = claim.get("statement", "")
                if len(stmt.strip()) < 50:
                    skipped_short += 1
                    continue
                # Quality gate: skip ephemeral claims
                if claim.get("durability") == "ephemeral":
                    skipped_ephemeral += 1
                    continue
                claim["source_chunk_id"] = chunk.chunk_id
                claim["source_record_id"] = chunk.record_id
                claim["source_span_start"] = chunk.span_start
                all_claims.append(claim)

        # Deduplicate by claim_key
        deduped = self._deduplicate(all_claims)

        # Mark auto-accept candidates
        for claim in deduped:
            claim["auto_accept"] = self._should_auto_accept(claim, deduped)

        metrics = {
            "claim_count": len(deduped),
            "auto_accept_count": sum(1 for c in deduped if c.get("auto_accept")),
            "chunks_processed": len(chunks),
            "skipped_short": skipped_short,
            "skipped_ephemeral": skipped_ephemeral,
        }

        return ExtractorResult(items=deduped, metrics=metrics, warnings=warnings)

    def _deduplicate(self, claims: list[dict]) -> list[dict]:
        """Keep the highest-confidence claim per claim_key."""
        by_key: dict[str, dict] = {}
        for claim in claims:
            key = claim.get("claim_key", "")
            if not key:
                continue
            existing = by_key.get(key)
            if not existing or claim.get("confidence", 0) > existing.get("confidence", 0):
                by_key[key] = claim
        return list(by_key.values())

    def _should_auto_accept(self, claim: dict, all_claims: list[dict]) -> bool:
        """Check if a claim qualifies for auto-acceptance."""
        claim_type = claim.get("claim_type", "")
        confidence = claim.get("confidence", 0)
        claim_key = claim.get("claim_key", "")

        if claim_type not in self.auto_accept_types:
            return False
        if confidence < self.auto_accept_confidence:
            return False

        # Count distinct supporting sources (chunks from different records)
        support_records = set()
        for c in all_claims:
            if c.get("claim_key") == claim_key:
                support_records.add(c.get("source_record_id", ""))
        support_records.discard("")

        return len(support_records) >= self.auto_accept_min_supports
