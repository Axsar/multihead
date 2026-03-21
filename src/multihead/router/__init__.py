"""Head router package — selects the best head for a step based on requirements and system state.

Re-exports the ``Router`` class so that ``from multihead.router import Router``
continues to work unchanged after the split from a single module to a package.
"""

from ._core import Router
from ._scoring import (
    _W_ACCURACY,
    _W_ACTIVE,
    _W_BREAKER,
    _W_CAPABILITY,
    _W_COST,
    _W_ERROR_RATE,
    _W_LATENCY,
    _W_PREFERENCE,
    _W_VRAM_FIT,
)

__all__ = [
    "Router",
    # Scoring weights (re-exported for backward compatibility)
    "_W_ACCURACY",
    "_W_ACTIVE",
    "_W_BREAKER",
    "_W_CAPABILITY",
    "_W_COST",
    "_W_ERROR_RATE",
    "_W_LATENCY",
    "_W_PREFERENCE",
    "_W_VRAM_FIT",
]
