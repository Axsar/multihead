"""Confidence calibration for code narrative sources.

Each source type has a confidence cap reflecting its reliability.
Priority ordering determines which source wins on conflict.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class SourcePriority(enum.IntEnum):
    """Priority ordering for evidence sources (lower = higher priority)."""

    # Priority 1: Direct explicit text (highest trust)
    GIT_COMMIT_MESSAGE = 10
    CODE_COMMENT = 15

    # Priority 1b: Planning documents (authored intent, not verified reality)
    PLANNING_DOCUMENT = 18

    # Priority 2: Author/speaker attribution
    GIT_BLAME = 20
    CHAT_MESSAGE_AUTHOR = 25

    # Priority 3: Structural facts (deterministic)
    AST_DIFF = 30
    TEST_RESULT = 35
    CI_SIGNAL = 37

    # Priority 4: Identity registry
    DEVELOPER_PROFILE = 40
    MODULE_REGISTRY = 42
    AGENT_REGISTRY = 45

    # Priority 5: Observations (high confidence but indirect)
    CODE_ANALYSIS = 50
    DEPENDENCY_GRAPH = 55

    # Priority 6: Automated detection signals
    LINTER_OUTPUT = 60
    COVERAGE_REPORT = 65

    # Priority 7: LLM/VLM inference (lowest trust)
    LLM_INFERENCE = 70
    VLM_INFERENCE = 75
    AGENT_INTERPRETATION = 80


# Confidence caps per source type — source cannot exceed this
_CONFIDENCE_CAPS: dict[SourcePriority, float] = {
    SourcePriority.GIT_COMMIT_MESSAGE: 0.95,
    SourcePriority.PLANNING_DOCUMENT: 0.90,
    SourcePriority.CODE_COMMENT: 0.90,
    SourcePriority.GIT_BLAME: 0.95,
    SourcePriority.CHAT_MESSAGE_AUTHOR: 0.90,
    SourcePriority.AST_DIFF: 0.95,
    SourcePriority.TEST_RESULT: 0.95,
    SourcePriority.CI_SIGNAL: 0.90,
    SourcePriority.DEVELOPER_PROFILE: 0.85,
    SourcePriority.MODULE_REGISTRY: 0.85,
    SourcePriority.AGENT_REGISTRY: 0.85,
    SourcePriority.CODE_ANALYSIS: 0.80,
    SourcePriority.DEPENDENCY_GRAPH: 0.85,
    SourcePriority.LINTER_OUTPUT: 0.85,
    SourcePriority.COVERAGE_REPORT: 0.85,
    SourcePriority.LLM_INFERENCE: 0.75,
    SourcePriority.VLM_INFERENCE: 0.70,
    SourcePriority.AGENT_INTERPRETATION: 0.75,
}

# Corroboration bonus when multiple sources agree
_CORROBORATION_BONUS = 0.05
_CORROBORATION_CAP = 0.90

# Epistemic humility: absolute maximum confidence
EPISTEMIC_CEILING = 0.95

# Below this threshold, claims are marked as unknown/uncertain
UNKNOWN_THRESHOLD = 0.60


@dataclass
class CalibratedConfidence:
    """Result of confidence calibration."""

    raw: float
    calibrated: float
    source_priority: SourcePriority
    cap_applied: float
    corroborated: bool
    clamped: bool  # True if raw was outside [0, 1]


class ConfidenceCalibrator:
    """Calibrates confidence scores based on source type and corroboration."""

    def calibrate(
        self,
        raw_confidence: float,
        source: SourcePriority,
        corroborating_sources: list[SourcePriority] | None = None,
    ) -> CalibratedConfidence:
        """Apply confidence calibration rules.

        Args:
            raw_confidence: Raw confidence from extractor [0, 1].
            source: The primary source type.
            corroborating_sources: Other sources that agree with this claim.

        Returns:
            CalibratedConfidence with calibrated value.
        """
        clamped = raw_confidence < 0.0 or raw_confidence > 1.0
        confidence = max(0.0, min(1.0, raw_confidence))

        # Apply source-specific cap
        cap = _CONFIDENCE_CAPS.get(source, 0.75)
        confidence = min(confidence, cap)

        # Apply corroboration bonus
        corroborated = False
        if corroborating_sources:
            bonus = min(
                len(corroborating_sources) * _CORROBORATION_BONUS,
                _CORROBORATION_CAP - confidence,
            )
            if bonus > 0:
                confidence += bonus
                corroborated = True

        # Epistemic ceiling
        confidence = min(confidence, EPISTEMIC_CEILING)

        return CalibratedConfidence(
            raw=raw_confidence,
            calibrated=round(confidence, 4),
            source_priority=source,
            cap_applied=cap,
            corroborated=corroborated,
            clamped=clamped,
        )

    def is_unknown(self, confidence: float) -> bool:
        """Returns True if confidence is below the unknown threshold."""
        return confidence < UNKNOWN_THRESHOLD

    def resolve_conflict(
        self,
        source_a: SourcePriority,
        confidence_a: float,
        source_b: SourcePriority,
        confidence_b: float,
    ) -> str:
        """Determine which source wins a conflict.

        Returns 'a', 'b', or 'fork' (if too close to call).
        """
        # Higher priority (lower int) wins if confidence difference < 0.15
        if abs(confidence_a - confidence_b) < 0.15:
            if source_a < source_b:
                return "a"
            elif source_b < source_a:
                return "b"
            else:
                return "fork"

        # Otherwise, higher confidence wins
        if confidence_a > confidence_b:
            return "a"
        elif confidence_b > confidence_a:
            return "b"
        return "fork"
