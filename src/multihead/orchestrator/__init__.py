"""Run orchestrator: execute WorkOrders step-by-step with durable state.

This package splits the orchestrator into focused modules:
- _core: Main Orchestrator class with run lifecycle and execution strategies
- _delegation: Marketplace/ACP delegation logic
- _execution: Step execution and reflection loop
- _prompts: Prompt building from templates
- _helpers: Shared utilities

All public names are re-exported here so that existing imports
(``from multihead.orchestrator import Orchestrator``) continue to work.
"""

from ._core import Orchestrator
from ._helpers import _SafeFormatDict

__all__ = ["Orchestrator", "_SafeFormatDict"]
