"""Infer scope_id from claim_key and statement content.

Used at extraction time to prevent claims from landing in 'default' scope.
Reusable by nightshift, session harvester, SolveDirect, and MCP tools.
"""

from __future__ import annotations

import re

# Ordered list: first match wins. Patterns are case-insensitive.
_SCOPE_RULES: list[tuple[str, list[str], list[str]]] = [
    # (scope_id, claim_key patterns, statement keyword patterns)
    # Add project-specific scope rules here. Example:
    # ("myproject", [r"myproject[._]", r"auth", r"payment"], [r"auth", r"deploy", r"service"]),
    ("multihead", [
        r"multihead\.", r"project\.multihead\.", r"feature\.",
        r"packbuilder\.", r"nightshift", r"night_shift",
        r"agentic_core\.", r"conversation_harvester\.",
    ], [
        r"headmanager", r"orchestrator", r"night.shift",
        r"knowledge.store", r"recipe", r"head_manager", r"router",
        r"vram", r"consensus", r"head swap", r"adapter",
        r"circuit breaker", r"solve pipeline", r"mcp server",
        r"multihead",
    ]),
    ("botvibes", [
        r"botvibes\.", r"marketplace", r"rfq", r"escrow", r"listing",
        r"contract\.", r"agent\.marketplace", r"cloud_marketplace\.",
        r"cleanup_timed_out",
    ], [
        r"botvibes", r"marketplace", r"rfq", r"listing", r"escrow",
        r"contract", r"provider", r"vault", r"tender", r"bid",
        r"capability.publish",
    ]),
    ("vibebots", [
        r"vibebots\.", r"acp", r"worker_daemon\.",
        r"claude.vibebots",
    ], [
        r"acp", r"agent.register", r"websocket", r"vibebots",
        r"worker daemon",
    ]),
]


def infer_scope(
    claim_key: str = "",
    statement: str = "",
    default: str = "default",
) -> str:
    """Infer the best scope_id from claim_key and statement.

    Returns the first matching scope, or `default` if no match.
    """
    key_lower = (claim_key or "").lower()
    stmt_lower = (statement or "").lower()

    for scope_id, key_patterns, stmt_patterns in _SCOPE_RULES:
        for pat in key_patterns:
            if re.search(pat, key_lower):
                return scope_id
        for pat in stmt_patterns:
            if re.search(pat, stmt_lower):
                return scope_id

    return default
