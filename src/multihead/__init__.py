"""MultiHead: Local multimodal task-runner with hot-swappable specialist models."""

__version__ = "1.3.52"

# Layer 1: Python SDK
from .engine import Engine

# Core components (for advanced usage)
from .config import Settings, load_heads
from .head_manager import HeadManager
from .knowledge_store import KnowledgeStore
from .models import AdapterKind, HeadManifest, StepDef, WorkOrder
from .orchestrator import Orchestrator
from .rfq_manager import RFQManager
from .router import Router

# Consensus
from .consensus import (
    ConsensusConfig,
    ConsensusEngine,
    ConsensusResult,
    ConsensusStrategy,
    FirstToAheadConfig,
    HeadTask,
    VoteResult,
)

__all__ = [
    # SDK
    "Engine",
    # Core
    "HeadManager",
    "KnowledgeStore",
    "Router",
    "Orchestrator",
    "RFQManager",
    "Settings",
    "load_heads",
    # Models
    "AdapterKind",
    "HeadManifest",
    "StepDef",
    "WorkOrder",
    # Consensus
    "ConsensusConfig",
    "ConsensusEngine",
    "ConsensusResult",
    "ConsensusStrategy",
    "FirstToAheadConfig",
    "HeadTask",
    "VoteResult",
]
