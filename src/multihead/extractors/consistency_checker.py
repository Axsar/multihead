"""Consistency check — detect contradictions between related claims.

Groups claims by topic (claim_key prefix or file_path) so only
related claims get compared. Unrelated claims never waste LLM calls.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from multihead.adapters.base import HeadAdapter
from multihead.chunker import Chunk
from multihead.extractors.base import BaseExtractor, ExtractorResult

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Analyze the following claims for contradictions or inconsistencies.
Two claims contradict if they make conflicting assertions about the same subject.

Claims:
{claims_text}

Return a JSON array of contradiction objects, each with:
- claim_key_a: the claim_key of the first conflicting claim
- claim_key_b: the claim_key of the second conflicting claim
- reason: a short explanation of the contradiction
- severity: "high" (direct contradiction) or "low" (tension but not direct) or "none" (no contradiction)

If no contradictions are found, return an empty array [].

Return ONLY the JSON array, no other text."""


class ConsistencyChecker(BaseExtractor):
    """Detect contradictions between related claims.

    Groups claims by topic before checking, so only related claims
    get compared. Much more efficient and accurate than random batching.
    """

    async def extract(
        self, chunks: list[Chunk], adapter: HeadAdapter, **kwargs: Any
    ) -> ExtractorResult:
        """Check claims for consistency. Pass claims via kwargs['claims']."""
        claims = kwargs.get("claims", [])
        if len(claims) < 2:
            return ExtractorResult(
                metrics={"contradiction_count": 0, "claims_checked": len(claims)}
            )

        # Group claims by topic for targeted comparison
        groups = self._group_by_topic(claims)
        logger.info(
            "Consistency check: %d claims → %d topic groups (avg %.1f claims/group)",
            len(claims), len(groups),
            sum(len(g) for g in groups.values()) / max(len(groups), 1),
        )

        # Only check groups with 2+ claims from different channels
        prompts: list[str] = []
        group_keys: list[str] = []
        for topic, group_claims in groups.items():
            if len(group_claims) < 2:
                continue

            # Skip if all claims from same channel (unlikely to contradict)
            channels = set(c.get("observation_method", "") for c in group_claims)
            if len(channels) < 2 and len(group_claims) < 5:
                continue

            # Build prompt with full statements + channel info
            claims_text = "\n".join(
                f"- [{c.get('claim_key', '?')}] ({c.get('observation_method', '?')}, "
                f"conf={c.get('confidence', '?')}): {c.get('statement', '?')}"
                for c in group_claims
            )
            prompts.append(PROMPT_TEMPLATE.format(claims_text=claims_text))
            group_keys.append(topic)

        logger.info(
            "Consistency check: %d groups need checking (%d skipped — single channel or <2 claims)",
            len(prompts), len(groups) - len(prompts),
        )

        if not prompts:
            return ExtractorResult(
                metrics={"contradiction_count": 0, "claims_checked": len(claims),
                         "groups_total": len(groups), "groups_checked": 0}
            )

        concurrency = kwargs.get("concurrency", 1)
        responses = await self.map_generate(
            adapter, prompts, concurrency=concurrency,
            checkpoint_dir=kwargs.get("checkpoint_dir"),
            stage_name=kwargs.get("stage_name", ""),
            batch_mode=kwargs.get("batch_mode", False),
            no_wait=kwargs.get("no_wait", False),
        )

        warnings: list[str] = []
        contradictions: list[dict[str, Any]] = []

        for topic, resp in zip(group_keys, responses):
            if isinstance(resp, Exception):
                warnings.append(f"Consistency check failed for group {topic}: {resp}")
                continue
            parsed = self.parse_json_response(resp.get("text", ""))
            contradictions.extend(parsed)

        metrics = {
            "contradiction_count": len(contradictions),
            "high_severity_count": sum(1 for c in contradictions if c.get("severity") == "high"),
            "claims_checked": len(claims),
            "groups_total": len(groups),
            "groups_checked": len(prompts),
        }

        return ExtractorResult(items=contradictions, metrics=metrics, warnings=warnings)

    @staticmethod
    def _group_by_topic(claims: list[dict]) -> dict[str, list[dict]]:
        """Group claims by topic for targeted consistency checking.

        Groups by (in priority order):
        1. file_path from source_anchor (most specific)
        2. claim_key prefix (first 2 dot-separated parts)
        3. First significant word in claim_key
        """
        import json

        groups: dict[str, list[dict]] = defaultdict(list)

        for c in claims:
            # Try file_path first
            file_path = None
            prov_str = c.get("provenance_json", "")
            if prov_str and isinstance(prov_str, str):
                try:
                    prov = json.loads(prov_str)
                    file_path = prov.get("source_anchor", {}).get("file_path", "")
                except (json.JSONDecodeError, AttributeError):
                    pass

            if file_path:
                groups[f"file:{file_path}"].append(c)
                continue

            # Fall back to claim_key prefix
            key = c.get("claim_key", "")
            if key:
                parts = key.split(".")
                if len(parts) >= 2:
                    topic = f"{parts[0]}.{parts[1]}"
                else:
                    topic = parts[0]
                groups[f"key:{topic}"].append(c)
            else:
                groups["uncategorized"].append(c)

        return dict(groups)
