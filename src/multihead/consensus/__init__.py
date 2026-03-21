"""Multi-head consensus engine: K-voting and cross-modal verification."""

from .engine import ConsensusEngine
from .models import (
    ConsensusConfig,
    ConsensusResult,
    ConsensusStrategy,
    FirstToAheadConfig,
    HeadTask,
    VoteResult,
)

__all__ = [
    "ConsensusConfig",
    "ConsensusEngine",
    "ConsensusResult",
    "ConsensusStrategy",
    "FirstToAheadConfig",
    "HeadTask",
    "VoteResult",
]
