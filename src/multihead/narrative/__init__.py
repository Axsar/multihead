"""Narrative pipeline: evidence extraction → fusion → verification → state tracking.

Extracts knowledge from code-domain sources
(git commits, chat transcripts, agent logs, LLM/VLM session outputs).
"""

from multihead.narrative.confidence import ConfidenceCalibrator, SourcePriority
from multihead.narrative.verification import NarrativeVerifier, VerificationResult
from multihead.narrative.fusion import NarrativeFusion, FusionBundle, FusedNarrative
from multihead.narrative.state_engine import NarrativeStateEngine
from multihead.narrative.context_gen import generate_daemon_context

__all__ = [
    "ConfidenceCalibrator",
    "SourcePriority",
    "NarrativeVerifier",
    "VerificationResult",
    "NarrativeFusion",
    "FusionBundle",
    "FusedNarrative",
    "NarrativeStateEngine",
    "generate_daemon_context",
]
