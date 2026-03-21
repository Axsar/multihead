"""Shared constants and type aliases for the shell pipeline."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

# Stable agent identity — shared across all shell sessions so interaction
# tracking persists across restarts.
AGENT_ID = "claude-multihead-main"
SELF_IDENTITIES = frozenset({"claude-multihead-main", "claude_code_main"})

# Type alias for brain functions: (session_id, user_input, knowledge_ctx) -> response
BrainFn = Callable[[str, str, str], Awaitable[str]]

# ---------------------------------------------------------------------------
# Stopwords for keyword extraction
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset({
    "what", "when", "where", "which", "that", "this", "with",
    "from", "have", "been", "were", "will", "would", "could",
    "should", "about", "their", "there", "they", "them",
    "your", "some", "more", "than", "then", "also", "just",
    "like", "into", "over", "such", "very", "does", "each",
    "only", "most", "both", "here", "much", "many", "well",
    "even", "back", "make", "made", "come", "came", "went",
    "know", "want", "need", "help", "tell", "show", "give",
    "take", "look", "find", "keep", "think", "good", "best",
    "long", "same", "work", "part", "last", "next", "used",
    "using", "doing", "being", "going",
})
