"""knowledge_store package — re-exports all public symbols for backward compatibility.

All existing imports like ``from multihead.knowledge_store import KnowledgeStore``
continue to work unchanged.
"""

from ._helpers import _coerce_float, _safe_json_loads
from ._store import KnowledgeStore

__all__ = [
    "KnowledgeStore",
    "_safe_json_loads",
    "_coerce_float",
]
