"""Execution strategies — swappable backends for plan step execution."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from typing import Any

from multihead.subprocess_utils import no_window_flags
from .models import StepExecutionResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Role-based tool restrictions for Claude sessions
# ---------------------------------------------------------------------------

ROLE_TOOL_MAP: dict[str, str] = {
    "explore": "Read,Grep,Glob,Task",
    "read": "Read,Grep,Glob",
    "implement": 'Read,Grep,Glob,Edit,Write,Bash(git:*)',
    "create": 'Read,Grep,Glob,Edit,Write,Bash(git:*)',
    "edit": 'Read,Grep,Glob,Edit,Write,Bash(git:*)',
    "refactor": 'Read,Grep,Glob,Edit,Write,Bash(git:*)',
    "review": "Read,Grep,Glob",
    "test": "Read,Grep,Glob,Edit,Write,Bash(*)",
    "verify": 'Read,Grep,Glob,Bash(git:*)',
    "debug": "Read,Grep,Glob,Edit,Write,Bash(*)",
    "fix": "Read,Grep,Glob,Edit,Write,Bash(*)",
}

DEFAULT_TOOLS = "Read,Grep,Glob"

# ---------------------------------------------------------------------------
# Role-specific prompt templates
# ---------------------------------------------------------------------------

ROLE_PROMPTS: dict[str, str] = {
    "explore": (
        "You are an EXPLORER. Your job is to understand the codebase.\n"
        "- Read files, search for patterns, understand architecture\n"
        "- DO NOT modify any files\n"
        "- Output a clear summary of what you found\n"
    ),
    "read": (
        "You are a READER. Your job is to read and understand specific files.\n"
        "- Read the requested files and extract relevant information\n"
        "- DO NOT modify any files\n"
        "- Output the key information found\n"
    ),
    "implement": (
        "You are an IMPLEMENTER. Your job is to write or modify code.\n"
        "- Follow existing code patterns and conventions\n"
        "- Make minimal, focused changes\n"
        "- Do NOT add unnecessary abstractions or comments\n"
        "- Stage changed files with git add\n"
    ),
    "create": (
        "You are a CREATOR. Your job is to create new files.\n"
        "- Follow existing code patterns and conventions\n"
        "- Keep code minimal and focused on the requirement\n"
        "- Stage new files with git add\n"
    ),
    "edit": (
        "You are an EDITOR. Your job is to modify existing code.\n"
        "- Read the file first, then make targeted edits\n"
        "- Follow existing code patterns\n"
        "- Stage changed files with git add\n"
    ),
    "refactor": (
        "You are a REFACTORER. Your job is to restructure code.\n"
        "- Preserve behavior while improving structure\n"
        "- Run existing tests to verify no regressions\n"
        "- Stage changed files with git add\n"
    ),
    "review": (
        "You are a REVIEWER. Your job is to review code quality.\n"
        "- DO NOT modify any files\n"
        "- Check for bugs, security issues, style violations\n"
        "- Output a structured review with issues found\n"
    ),
    "test": (
        "You are a TESTER. Your job is to run tests and fix failures.\n"
        "- Run the relevant test suite\n"
        "- If tests fail, fix the code and re-run\n"
        "- Output test results (pass/fail counts)\n"
    ),
    "verify": (
        "You are a VERIFIER. Your job is to check acceptance criteria.\n"
        "- DO NOT modify any files\n"
        "- Verify the changes meet the stated requirements\n"
        "- Output PASS or FAIL with reasoning\n"
    ),
    "debug": (
        "You are a DEBUGGER. Your job is to diagnose and fix issues.\n"
        "- Investigate the problem, find root cause\n"
        "- Apply a targeted fix\n"
        "- Verify the fix works\n"
    ),
    "fix": (
        "You are a FIXER. Your job is to fix a specific issue.\n"
        "- Read the relevant code, understand the bug\n"
        "- Apply a minimal fix\n"
        "- Run tests to verify\n"
    ),
}

DEFAULT_ROLE_PROMPT = (
    "You are an autonomous agent executing a step in a larger plan.\n"
    "- Be precise and focused on the specific goal\n"
    "- Output a clear summary of what you did\n"
)


# ---------------------------------------------------------------------------
# Execution strategies (ABC + implementations)
# ---------------------------------------------------------------------------


class ExecutionStrategy(ABC):
    """Swappable execution backend for plan steps."""

    @abstractmethod
    async def execute_step(
        self, step_id: str, prompt: str, action_type: str,
        timeout: int = 300,
    ) -> StepExecutionResult:
        """Execute a single step and return the result."""

    @abstractmethod
    def check_quality(self, result: StepExecutionResult) -> tuple[float, str]:
        """Heuristic quality check. Returns (score 0-1, feedback)."""


class LocalLLMStrategy(ExecutionStrategy):
    """Plan-only strategy — returns the prompt as the 'output' (no execution)."""

    async def execute_step(
        self, step_id: str, prompt: str, action_type: str,
        timeout: int = 300,
    ) -> StepExecutionResult:
        return StepExecutionResult(
            step_id=step_id,
            step_goal=prompt[:200],
            action_type=action_type,
            success=True,
            output=f"[plan-only] Step planned: {prompt[:500]}",
            quality_score=1.0,
        )

    def check_quality(self, result: StepExecutionResult) -> tuple[float, str]:
        return (1.0, "plan-only mode — no quality check")


class ClaudeSessionStrategy(ExecutionStrategy):
    """Spawns `claude -p` per step with role-specific tool restrictions."""

    def __init__(
        self,
        model: str = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6"),
        max_budget_usd: float = 1.0,
        work_dir: str = "",
        subprocess_timeout: int = 600,
    ):
        self.model = model
        self.max_budget_usd = max_budget_usd
        self.work_dir = work_dir or os.environ.get("CLAUDE_WORK_DIR", os.getcwd())
        self.subprocess_timeout = subprocess_timeout

    async def execute_step(
        self, step_id: str, prompt: str, action_type: str,
        timeout: int = 300,
    ) -> StepExecutionResult:
        """Spawn a claude -p subprocess for this step."""
        claude_bin = shutil.which("claude")
        if not claude_bin:
            return StepExecutionResult(
                step_id=step_id, step_goal=prompt[:200],
                action_type=action_type, success=False,
                output="", error="claude CLI not found in PATH",
            )

        allowed_tools = ROLE_TOOL_MAP.get(action_type, DEFAULT_TOOLS)

        cmd = [
            claude_bin, "-p", prompt,
            "--output-format", "json",
            "--model", self.model,
            "--max-budget-usd", str(self.max_budget_usd),
            "--allowedTools", allowed_tools,
            "--permission-mode", "acceptEdits",
        ]

        # Clean env so subprocess can spawn
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

        logger.info(
            "Step %s (%s): spawning claude -p (tools=%s, budget=$%.2f)",
            step_id, action_type, allowed_tools, self.max_budget_usd,
        )
        start = time.monotonic()

        try:
            proc = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd, capture_output=True, text=True,
                    cwd=self.work_dir,
                    timeout=timeout or self.subprocess_timeout,
                    env=env,
                    creationflags=no_window_flags(),
                ),
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start
            return StepExecutionResult(
                step_id=step_id, step_goal=prompt[:200],
                action_type=action_type, success=False,
                output="", duration_secs=elapsed,
                error=f"Subprocess timed out after {elapsed:.0f}s",
            )

        elapsed = time.monotonic() - start
        logger.info("Step %s completed in %.1fs (exit=%d)", step_id, elapsed, proc.returncode)

        if proc.returncode != 0:
            return StepExecutionResult(
                step_id=step_id, step_goal=prompt[:200],
                action_type=action_type, success=False,
                output=proc.stdout[:2000] if proc.stdout else "",
                duration_secs=elapsed,
                error=f"Exit {proc.returncode}: {proc.stderr[:500] if proc.stderr else ''}",
            )

        # Parse output
        parsed = self._parse_output(proc.stdout)
        return StepExecutionResult(
            step_id=step_id,
            step_goal=prompt[:200],
            action_type=action_type,
            success=not parsed.get("is_error", False),
            output=parsed.get("result", proc.stdout[:2000]),
            cost_usd=parsed.get("cost_usd", 0.0),
            duration_secs=elapsed,
            session_id=parsed.get("session_id", ""),
            error=parsed.get("error", ""),
        )

    def check_quality(self, result: StepExecutionResult) -> tuple[float, str]:
        """Heuristic quality check (Tier 1 — zero cost)."""
        score = 1.0
        issues = []

        # Empty output
        if not result.output or len(result.output.strip()) < 20:
            score -= 0.5
            issues.append("Output too short or empty")

        # Error indicators in output
        error_terms = ["error", "failed", "traceback", "exception"]
        output_lower = result.output.lower()
        for term in error_terms:
            if term in output_lower:
                score -= 0.2
                issues.append(f"Contains '{term}'")
                break

        # Action-specific checks
        if result.action_type in ("test", "verify"):
            if "pass" not in output_lower and "success" not in output_lower:
                score -= 0.3
                issues.append("No pass/success indicator for test/verify step")

        if result.action_type in ("implement", "create", "edit"):
            if "edit" not in output_lower and "write" not in output_lower and "created" not in output_lower:
                score -= 0.1
                issues.append("No write/edit indicator for implementation step")

        score = max(0.0, min(1.0, score))
        feedback = "; ".join(issues) if issues else "Looks good"
        return (score, feedback)

    @staticmethod
    def _parse_output(stdout: str) -> dict:
        """Parse Claude's JSON output (same pattern as claude_worker.py)."""
        stdout = stdout.strip()
        if not stdout:
            return {"error": "Empty output"}

        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            # Try line-by-line
            for line in stdout.split("\n"):
                line = line.strip()
                if line.startswith(("{", "[")):
                    try:
                        parsed = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
            else:
                return {"result": stdout[:2000]}

        # Normalize message list to flat dict
        if isinstance(parsed, list):
            return _normalize_claude_output(parsed)
        return parsed


# ---------------------------------------------------------------------------
# Shared output normalizer (extracted from claude_worker.py pattern)
# ---------------------------------------------------------------------------


def _normalize_claude_output(messages: list[dict]) -> dict:
    """Extract session_id and result text from Claude's message list."""
    result: dict = {}

    for msg in messages:
        if msg.get("type") == "system" and msg.get("subtype") == "init":
            result["session_id"] = msg.get("session_id", "")
            break

    for msg in reversed(messages):
        if msg.get("type") == "result":
            result["cost_usd"] = msg.get("cost_usd", 0)
            result["result"] = msg.get("result", "")
            result["is_error"] = msg.get("is_error", False)
            return result

    # Fallback: concatenate assistant text blocks
    texts = []
    for msg in messages:
        if msg.get("type") == "assistant":
            inner = msg.get("message", {})
            for block in inner.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block["text"])
                elif isinstance(block, str):
                    texts.append(block)

    result["result"] = "\n".join(texts) if texts else "No output"
    return result
