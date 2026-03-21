"""Stage 2: Intent Classification — heuristic-based, no LLM call.

Classifies user input as 'chat' or 'task' based on action verbs,
file references, multi-step language, and message length.
"""

from __future__ import annotations

import re

_ACTION_VERBS = frozenset({
    "build", "fix", "implement", "create", "refactor", "add", "remove",
    "delete", "update", "modify", "change", "write", "rewrite", "deploy",
    "migrate", "install", "configure", "setup", "debug", "optimize",
    "test", "benchmark", "profile", "analyze", "generate", "convert",
    "extract", "transform", "parse", "validate", "integrate", "connect",
})

_MULTI_STEP_PATTERNS = [
    r"\bfirst\b.*\bthen\b",
    r"\bstep\s+\d",
    r"\b\d+\.\s+\w",
    r"\band\s+then\b",
    r"\bafter\s+that\b",
    r"\bfinally\b",
]


def has_action_verbs(text: str) -> bool:
    """Check if text contains programming action verbs."""
    text_lower = text.lower()
    return any(verb in text_lower.split() for verb in _ACTION_VERBS)


def mentions_files(text: str) -> bool:
    """Check if text references file paths or extensions."""
    file_patterns = [
        r"[\w/\\.-]+\.\w{1,4}\b",  # file.ext
        r"/[\w/.-]+",               # /path/to/file
        r"src/",                    # common path prefix
        r"tests/",
        r"config/",
    ]
    for pattern in file_patterns:
        if re.search(pattern, text):
            return True
    return False


def has_multi_step_language(text: str) -> bool:
    """Check if text uses multi-step/sequential language."""
    text_lower = text.lower()
    for pattern in _MULTI_STEP_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def classify_intent(user_input: str) -> str:
    """Classify user input as 'chat' or 'task'.

    Heuristic-based — no LLM call. Returns 'task' when the message
    looks like a complex actionable request.
    """
    words = user_input.split()

    # Quick exits -> chat
    if len(words) < 5:
        return "chat"
    if user_input.rstrip().endswith("?") and not has_action_verbs(user_input):
        return "chat"

    # Count task signals
    task_signals = 0
    if has_action_verbs(user_input):
        task_signals += 1
    if mentions_files(user_input):
        task_signals += 1
    if has_multi_step_language(user_input):
        task_signals += 1
    if len(words) > 20:
        task_signals += 1

    return "task" if task_signals >= 2 else "chat"
