"""Stage 4: Knowledge Recording — extract and deposit facts from conversation.

Records substantial exchanges as knowledge claims.
Uses heuristic summarization (no LLM call).
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def summarize_exchange(user_input: str, response: str) -> str:
    """Create a concise one-line summary of the exchange.

    Takes the user's question and the first sentence of the response.
    Max 200 chars. No LLM call — just text extraction.
    """
    # Get first sentence of response
    first_sentence = ""
    for sep in (".", "!", "\n"):
        idx = response.find(sep)
        if idx > 0:
            candidate = response[:idx + 1].strip()
            if not first_sentence or len(candidate) < len(first_sentence):
                first_sentence = candidate
    if not first_sentence:
        first_sentence = response[:100]

    # Clean up
    first_sentence = first_sentence.strip("*# \n")
    user_short = user_input[:80].strip()

    summary = f"Q: {user_short} → A: {first_sentence}"
    return summary[:200]


def maybe_record_knowledge(
    ks: Any,
    user_input: str,
    response: str,
    participant_id: str,
    stats: dict[str, int],
) -> None:
    """Extract and deposit key facts from the conversation.

    Only records substantial exchanges. Uses heuristic summarization
    (no LLM call). Silently skips on any error.
    """
    if not ks:
        return

    # Only record substantial exchanges
    if len(response.split()) < 20:
        return

    summary = summarize_exchange(user_input, response)
    if not summary or len(summary) < 10:
        return

    try:
        from ..knowledge_models import (
            Claim,
            ClaimCanonical,
            ClaimScope,
            ClaimStatus,
            ClaimType,
            EntityRef,
            Provenance,
            ScopeType,
            ValueObject,
        )

        claim = Claim(
            claim_status=ClaimStatus.ACCEPTED,
            claim_type=ClaimType.FACT,
            scope=ClaimScope(
                scope_type=ScopeType.PROJECT,
                scope_id="multihead",
            ),
            canonical=ClaimCanonical(
                claim_key=f"shell.conversation.{int(time.time())}",
                subject=EntityRef(
                    entity_type="session",
                    entity_id="shell",
                    label="MultiHead Shell",
                ),
                predicate="discussed",
                object=ValueObject(
                    value_type="string",
                    value=summary[:200],
                ),
            ),
            statement=summary,
            confidence=0.6,
            provenance=Provenance(
                produced_by={
                    "kind": "agent",
                    "id": "multihead-shell-pipeline",
                    **({"participant_id": participant_id} if participant_id else {}),
                },
            ),
        )

        ks.insert_claim(claim)
        stats["claims_recorded"] += 1
        logger.debug("Recorded shell conversation claim: %s", summary[:80])

    except Exception as e:
        logger.warning("Failed to record knowledge: %s", e)
