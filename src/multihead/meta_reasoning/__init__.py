"""Meta-Reasoning Solver Selection (Phase 5).

MultiHead uses itself to decide which solver is best for a given task type.
The process:
1. Gather candidates from SolverRegistry
2. Run multi-head consensus to rank them
3. Optionally benchmark top candidates empirically
4. Synthesize final selection
5. Record preference for future routing
"""

from .models import SelectionResult
from .parsing import format_candidates_prompt, parse_consensus_output, try_parse_json
from .selector import MetaReasoningSelector

__all__ = [
    "MetaReasoningSelector",
    "SelectionResult",
    "format_candidates_prompt",
    "parse_consensus_output",
    "try_parse_json",
]
