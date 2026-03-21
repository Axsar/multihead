"""Process Reward Models (PRM) for step-level quality scoring.

Process Reward Models evaluate the quality of each intermediate reasoning step,
not just the final outcome. This enables:
- Precise credit assignment (identify *which* step failed)
- Earlier error detection (fail fast)
- Better path ranking (prefer high-quality reasoning)
- Improved training signals for reinforcement learning

Key distinction from Outcome Reward Models (ORMs):
- ORMs: Only evaluate the final answer (right/wrong)
- PRMs: Evaluate each step's contribution to correct reasoning
- PRMs provide richer feedback for learning and debugging

Reference: "Let's Verify Step by Step" (OpenAI, 2023)
https://cdn.openai.com/improving-mathematical-reasoning-with-process-supervision/Lets_Verify_Step_by_Step.pdf
"""

from multihead.process_reward_models.models import PathScore, PRMScore, StepQuality
from multihead.process_reward_models.scorers import (
    CompositePRMScorer,
    LLMPRMScorer,
    PRMScorer,
    RubricPRMScorer,
)

__all__ = [
    "CompositePRMScorer",
    "LLMPRMScorer",
    "PRMScore",
    "PRMScorer",
    "PathScore",
    "RubricPRMScorer",
    "StepQuality",
]
