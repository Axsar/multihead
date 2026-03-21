"""Constants, type maps, and prompt templates for claude_enhancer."""

from __future__ import annotations

import logging

from multihead.knowledge_models import (
    ClaimType,
    Provenance,
    Stability,
)

logger = logging.getLogger("multihead.narrative.claude_enhancer")

_ACP_TIMEOUT = 30.0

_PROVENANCE = Provenance(
    produced_by={"kind": "extractor", "id": "narrative.claude_enhancer"},
    toolchain=[{"name": "claude-code", "version": "daemon"}],
)

# Map claim_type strings from Claude output to ClaimType enum
_CLAIM_TYPE_MAP: dict[str, ClaimType] = {
    "fact": ClaimType.FACT,
    "plan": ClaimType.PLAN,
    "decision": ClaimType.DECISION,
    "constraint": ClaimType.CONSTRAINT,
    "risk": ClaimType.RISK,
    "assumption": ClaimType.ASSUMPTION,
    "definition": ClaimType.DEFINITION,
    "preference": ClaimType.PREFERENCE,
    "question": ClaimType.QUESTION,
}

# Map doc_type to stability (same as markdown extractor)
_DOC_STABILITY: dict[str, Stability] = {
    "plan": Stability.MEDIUM,
    "status": Stability.VOLATILE,
    "recovery": Stability.MEDIUM,
    "fixes": Stability.MEDIUM,
    "decision": Stability.STABLE,
    "constraint": Stability.STABLE,
}

_SECTION_PROMPT_TEMPLATE = """\
You are a knowledge extraction agent. Analyze the following section from a {doc_type} document \
called "{doc_name}" and extract ALL claims, decisions, plans, constraints, risks, and facts.

<section heading="{section_heading}">
{section_text}
</section>

Extract structured claims as JSON. For each claim identify:
- text: the claim statement (1-2 sentences, precise)
- claim_type: one of [fact, plan, decision, constraint, risk, assumption, definition, question]
- predicate: relationship verb (e.g. "requires", "depends_on", "completed", "planned", "has_issue", "decided", "blocks")
- confidence: 0.0-1.0 (how certain is this claim based on the text)
- entities: list of entity names referenced (projects, components, tools, people)
- reasoning: brief explanation of why this is a claim (1 sentence)

Look beyond bullet points — extract implicit claims from:
- Dependencies between items
- Risks implied by the text
- Assumptions that are taken for granted
- Architectural decisions embedded in descriptions
- Constraints mentioned in passing
- Relationships between entities

Return ONLY valid JSON in this format:
```json
{{
  "claims": [
    {{
      "text": "...",
      "claim_type": "plan",
      "predicate": "requires",
      "confidence": 0.85,
      "entities": ["component_a", "component_b"],
      "reasoning": "..."
    }}
  ]
}}
```

Do not include any text outside the JSON block.\
"""

_SYNTHESIS_PROMPT_TEMPLATE = """\
You are a knowledge synthesis agent. You have extracted claims from multiple sections \
of a {doc_type} document called "{doc_name}". Now synthesize cross-section insights.

Here are the claims extracted per section:
{section_summaries}

Identify any ADDITIONAL claims that emerge from considering the document as a whole:
- Cross-section dependencies (section A depends on section B)
- Overall project risks or assumptions
- Implicit decisions about architecture or approach
- Ordering constraints between phases/sections

Return ONLY claims that are NOT already captured in the per-section extraction.
Return valid JSON:
```json
{{
  "claims": [
    {{
      "text": "...",
      "claim_type": "...",
      "predicate": "...",
      "confidence": 0.80,
      "entities": ["..."],
      "reasoning": "Cross-section: ..."
    }}
  ]
}}
```

If no additional cross-section claims exist, return: {{"claims": []}}
Do not include any text outside the JSON block.\
"""
