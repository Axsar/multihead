"""ShellPipeline package — re-exports all public names for backward compatibility.

Usage:
    from multihead.shell_pipeline import ShellPipeline, AGENT_ID, _STOPWORDS
    # ... works exactly as before.
"""

from .constants import AGENT_ID, SELF_IDENTITIES, BrainFn, _STOPWORDS
from .execution import detect_image_input, execute_as_task, format_run_results, route_to_vlm
from .intent import classify_intent, has_action_verbs, has_multi_step_language, mentions_files
from .knowledge_rag import (
    build_inbox_context,
    build_knowledge_context,
    extract_keywords,
    query_claims_by_keywords,
    query_claims_fts,
)
from .pipeline import ShellPipeline
from .recording import maybe_record_knowledge, summarize_exchange

__all__ = [
    # Constants & types
    "AGENT_ID",
    "SELF_IDENTITIES",
    "BrainFn",
    "_STOPWORDS",
    # Main class
    "ShellPipeline",
    # Knowledge RAG (Stage 1)
    "build_knowledge_context",
    "build_inbox_context",
    "extract_keywords",
    "query_claims_fts",
    "query_claims_by_keywords",
    # Intent classification (Stage 2)
    "classify_intent",
    "has_action_verbs",
    "mentions_files",
    "has_multi_step_language",
    # Execution (Stage 3)
    "detect_image_input",
    "route_to_vlm",
    "execute_as_task",
    "format_run_results",
    # Recording (Stage 4)
    "maybe_record_knowledge",
    "summarize_exchange",
]
