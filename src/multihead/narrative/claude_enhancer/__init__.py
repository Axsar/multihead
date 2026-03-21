"""Enhance markdown claim extraction using Claude Code worker daemons.

Phase 1 (heuristic) extracts bullet points from markdown structure.
Phase 2 (Claude) sends each H2 section to a Claude worker daemon for
deep semantic extraction -- identifying implicit claims, relationships,
dependencies, and risks that heuristic parsing misses.

Sub-modules:
    constants  - type maps, prompt templates, shared config
    parsing    - section splitting, output parsing, claim conversion/merge
    client     - ACP/HTTP client for Claude worker communication
    enhancer   - ClaudeEnhancer orchestrator class
"""

from .enhancer import ClaudeEnhancer

__all__ = ["ClaudeEnhancer"]
