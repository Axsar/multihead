"""Night Shift data models, stage definitions, and constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from multihead.knowledge_models import Provenance


@dataclass
class StageGate:
    accept_if: list[dict[str, Any]] = field(default_factory=list)
    retry: dict[str, Any] = field(default_factory=lambda: {"max_attempts": 1})
    fallback: dict[str, Any] | None = None
    on_fail: str = "continue"  # abort | continue | skip | queue_review


@dataclass
class StageDefinition:
    stage_id: int
    name: str
    gate: StageGate
    requires_llm: bool = False
    depends_on: list[int] = field(default_factory=list)  # stage IDs this depends on


# Stage definitions with gating rules
STAGES = [
    # === PRODUCE: harvest data, extract claims from all channels ===
    StageDefinition(0, "session_harvest", StageGate(on_fail="continue")),
    StageDefinition(1, "select_input_window", StageGate(on_fail="abort")),
    StageDefinition(2, "normalize_chunk", StageGate(
        accept_if=[{"metric": "parse_success_rate", "op": ">=", "value": 0.995}],
        retry={"max_attempts": 2}, on_fail="abort",
    )),
    StageDefinition(3, "hot_signals", StageGate()),
    StageDefinition(4, "entity_extraction", StageGate(
        accept_if=[{"metric": "entity_yield_per_1k_tokens", "op": ">=", "value": 0.1}],
        retry={"max_attempts": 2}, on_fail="continue",
    ), requires_llm=True, depends_on=[3]),  # needs chunks
    StageDefinition(5, "topic_assignment", StageGate(
        accept_if=[{"metric": "unassigned_chunk_rate", "op": "<=", "value": 0.03}],
        retry={"max_attempts": 2}, on_fail="continue",
    ), requires_llm=True, depends_on=[3]),  # needs chunks (NOT entities)
    StageDefinition(6, "event_extraction", StageGate(
        accept_if=[{"metric": "event_extract_coverage", "op": ">=", "value": 0.80}],
        retry={"max_attempts": 2}, on_fail="continue",
    ), requires_llm=True, depends_on=[3]),  # needs chunks (NOT topics)
    StageDefinition(7, "claim_extraction", StageGate(on_fail="continue"), requires_llm=True, depends_on=[3]),  # needs chunks
    StageDefinition(8, "behavioral_code_analysis", StageGate(on_fail="continue"), requires_llm=True, depends_on=[]),  # independent — scans repos
    StageDefinition(9, "narrative_fusion", StageGate(on_fail="continue"), depends_on=[]),  # independent — scans git
    StageDefinition(10, "ci_results", StageGate(on_fail="continue"), depends_on=[]),  # independent — scans GitHub API
    # === ANALYZE: cross-reference, fuse, detect staleness ===
    StageDefinition(11, "consistency_check", StageGate(on_fail="continue"), requires_llm=True, depends_on=[7]),  # needs claims in DB
    StageDefinition(12, "conflict_resolution", StageGate(on_fail="continue"), requires_llm=True, depends_on=[11]),
    StageDefinition(13, "file_path_anchoring", StageGate(on_fail="continue"), depends_on=[7]),  # anchor claims before fusion
    StageDefinition(14, "claim_fusion", StageGate(on_fail="continue"), requires_llm=True, depends_on=[7]),  # needs claims (independent of 11)
    StageDefinition(15, "staleness_sweep", StageGate(on_fail="continue"), depends_on=[7]),  # needs claims with git_sha
    StageDefinition(16, "open_loops", StageGate(on_fail="continue"), requires_llm=True),
    # === OUTPUT: packs, briefs, reports, index, discovery ===
    StageDefinition(17, "canon_pack_build", StageGate()),
    StageDefinition(18, "daily_brief", StageGate(on_fail="continue"), requires_llm=True),
    StageDefinition(19, "weekly_rollup", StageGate(on_fail="skip"), requires_llm=True),
    StageDefinition(20, "cross_topic_links", StageGate(on_fail="continue"), requires_llm=True),
    StageDefinition(21, "index_rebuild", StageGate(
        fallback={"mode": "keyword_only"}, on_fail="continue",
    )),
    StageDefinition(22, "publish_report", StageGate()),
    StageDefinition(23, "solver_discovery", StageGate(on_fail="continue")),
    StageDefinition(24, "recipe_learning", StageGate(on_fail="continue")),
    StageDefinition(25, "backlog_sweep", StageGate(on_fail="continue"), requires_llm=True),
]


def _prov(
    observation_method: str = "conversation_analysis",
    speaker: str = "",
    valence: str = "",
    source_anchor: dict[str, str] | None = None,
    evidence: list[dict[str, str]] | None = None,
) -> Provenance:
    return Provenance(
        produced_by={"kind": "extractor", "id": "nightshift_v1"},
        observation_method=observation_method,
        speaker=speaker,
        valence=valence,
        source_anchor=source_anchor or {},
        evidence=evidence or [],
    )
