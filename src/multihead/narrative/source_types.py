"""Source type definitions with priority ordering and confidence calibration.

Each code source type has an inherent reliability ranking:
  git commits > test results > code comments > agent logs > chat transcripts > LLM inference

Confidence from any source is clamped to that source type's ceiling — an LLM
inference can never produce a 0.95-confidence claim, no matter how emphatic.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class SourceType(str, enum.Enum):
    """Known source types, ordered by decreasing reliability."""

    GIT_COMMIT = "git_commit"
    TEST_RESULT = "test_result"
    CODE_COMMENT = "code_comment"
    AGENT_TASK_RESULT = "agent_task_result"
    CHAT_TRANSCRIPT = "chat_transcript"
    LLM_INFERENCE = "llm_inference"


@dataclass(frozen=True)
class SourceProfile:
    """Confidence calibration profile for a source type."""

    source_type: SourceType
    priority: int  # Lower = higher priority (1 is highest)
    confidence_ceiling: float  # Max confidence a claim from this source can have
    base_confidence: float  # Default confidence when source doesn't provide one
    stability_default: str  # Default stability: volatile | medium | stable
    description: str


# Priority-ordered profiles: commit messages > test results > code comments >
# agent task results > chat transcripts > LLM inference.
SOURCE_PROFILES: dict[SourceType, SourceProfile] = {
    SourceType.GIT_COMMIT: SourceProfile(
        source_type=SourceType.GIT_COMMIT,
        priority=1,
        confidence_ceiling=0.95,
        base_confidence=0.85,
        stability_default="stable",
        description="Git commits: verified code changes with author, timestamp, diff",
    ),
    SourceType.TEST_RESULT: SourceProfile(
        source_type=SourceType.TEST_RESULT,
        priority=2,
        confidence_ceiling=0.95,
        base_confidence=0.90,
        stability_default="medium",
        description="Test results: pass/fail with deterministic evidence",
    ),
    SourceType.CODE_COMMENT: SourceProfile(
        source_type=SourceType.CODE_COMMENT,
        priority=3,
        confidence_ceiling=0.85,
        base_confidence=0.70,
        stability_default="medium",
        description="Code comments: developer-authored inline documentation",
    ),
    SourceType.AGENT_TASK_RESULT: SourceProfile(
        source_type=SourceType.AGENT_TASK_RESULT,
        priority=4,
        confidence_ceiling=0.80,
        base_confidence=0.65,
        stability_default="medium",
        description="Agent task results: outcomes of automated agent work",
    ),
    SourceType.CHAT_TRANSCRIPT: SourceProfile(
        source_type=SourceType.CHAT_TRANSCRIPT,
        priority=5,
        confidence_ceiling=0.75,
        base_confidence=0.55,
        stability_default="volatile",
        description="Chat transcripts: human/agent conversation records",
    ),
    SourceType.LLM_INFERENCE: SourceProfile(
        source_type=SourceType.LLM_INFERENCE,
        priority=6,
        confidence_ceiling=0.65,
        base_confidence=0.40,
        stability_default="volatile",
        description="LLM inference: model-generated claims without ground truth",
    ),
}


def get_profile(source_type: SourceType) -> SourceProfile:
    """Get the confidence profile for a source type."""
    return SOURCE_PROFILES[source_type]


def clamp_confidence(raw_confidence: float, source_type: SourceType) -> float:
    """Clamp a confidence value to the source type's ceiling.

    Epistemic humility: no source can exceed its inherent reliability ceiling.
    """
    profile = SOURCE_PROFILES[source_type]
    return min(max(raw_confidence, 0.0), profile.confidence_ceiling)


def resolve_conflict(source_a: SourceType, source_b: SourceType) -> SourceType:
    """When two sources conflict, the higher-priority source wins.

    Returns the winning source type.
    """
    profile_a = SOURCE_PROFILES[source_a]
    profile_b = SOURCE_PROFILES[source_b]
    return source_a if profile_a.priority <= profile_b.priority else source_b


def corroboration_boost(
    base_confidence: float, source_types: list[SourceType]
) -> float:
    """Boost confidence when multiple independent sources agree.

    Each additional corroborating source adds a diminishing boost,
    but the result is still clamped to the highest source's ceiling.
    """
    if not source_types:
        return base_confidence

    # Sort by priority (best first)
    sorted_types = sorted(source_types, key=lambda s: SOURCE_PROFILES[s].priority)
    best_ceiling = SOURCE_PROFILES[sorted_types[0]].confidence_ceiling

    # Each additional source adds a diminishing boost
    boost = 0.0
    for i, _ in enumerate(sorted_types[1:], start=1):
        boost += 0.05 / i  # +0.05, +0.025, +0.0167, ...

    return min(base_confidence + boost, best_ceiling)
