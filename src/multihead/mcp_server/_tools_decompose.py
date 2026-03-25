"""Task decomposition: 3-tier strategy (self, claude-p, qwen)."""

from __future__ import annotations

import json

import httpx

from multihead.subprocess_utils import no_window_flags
from ._core import _get_ks, _request, logger


_DECOMPOSE_TEMPLATE = """You are a task decomposer for software engineering.
Given a goal and codebase context, break the goal into a hierarchical execution plan where each leaf step is a single concrete action.

## Goal
{goal}

## Additional Context
{user_context}

## Codebase Knowledge (from knowledge.db)
{knowledge_context}

## Rules
1. Group steps into phases (e.g. understand, diagnose, implement, verify)
2. Leaf steps must be SINGLE concrete actions: read a file, edit specific code, run a test
3. Scale depth to complexity:
   - Simple bug fix: 1-2 phases, 3-6 leaf steps
   - Moderate feature: 2-4 phases, 8-15 leaf steps
   - Complex refactor: 3-6 phases, 15-40 leaf steps
4. Action types: explore, read, edit, create, test, verify, refactor, delete
5. Include target file paths when known from context
6. Each step should describe what success looks like
7. Steps within a phase should be parallelizable when independent
8. Each leaf step should target at most ONE file (m=1 atomicity)

## Output
Return ONLY valid JSON (no markdown fences, no commentary):
{{
  "goal": "{goal_escaped}",
  "complexity": "simple|moderate|complex",
  "phases": [
    {{
      "id": "1",
      "goal": "Phase description",
      "rationale": "Why this phase is needed",
      "action_type": "explore",
      "children": [
        {{
          "id": "1.1",
          "goal": "Concrete step description",
          "action_type": "read|edit|test|...",
          "target_files": ["path/to/file.py"],
          "expected_output": "What you'll have after this step"
        }}
      ]
    }}
  ]
}}"""


def _gather_knowledge_context(goal: str) -> tuple[str, list[str]]:
    """Query knowledge.db via FTS for claims relevant to the goal."""
    try:
        ks = _get_ks()
        results = ks.search_claims_hybrid(goal, limit=20, min_confidence=0.5)
        if results:
            claims_text = [f"- [{key}] {stmt[:200]}" for key, stmt, _conf in results]
            context_keys = [key for key, _stmt, _conf in results]
            return "\n".join(claims_text), context_keys
    except Exception:
        pass

    # Fallback: keyword matching
    import re
    stop_words = {
        "the", "a", "an", "in", "on", "at", "to", "for", "of", "is", "and",
        "or", "with", "from", "by", "it", "this", "that", "be", "are", "was",
        "do", "does", "did", "not", "no", "we", "i", "my", "our", "us",
    }
    words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", goal.lower())
    keywords = [w for w in words if w not in stop_words and len(w) > 2]

    if not keywords:
        return "(no keywords extracted)", []

    try:
        ks = _get_ks()
        all_claims = ks.list_claims(status="accepted", limit=200)
        claims_text: list[str] = []
        context_keys: list[str] = []
        for c in all_claims:
            stmt_lower = (c.statement or "").lower()
            if any(kw in stmt_lower for kw in keywords):
                key = c.canonical.claim_key or ""
                claims_text.append(f"- [{key}] {c.statement[:200]}")
                context_keys.append(key)
                if len(claims_text) >= 20:
                    break
        if not claims_text:
            return "(no relevant claims found)", []
        return "\n".join(claims_text), context_keys
    except Exception as e:
        return f"(knowledge query error: {e})", []


async def _decompose_self(goal: str, context: str = "") -> str:
    """Return decomposition prompt + knowledge context for calling session.

    The calling CLI session has the best context (loaded codebase, conversation
    history). We give it the structured template and knowledge claims so it
    can decompose the task itself using its own tools (Read, Grep, etc.).
    """
    knowledge_context, context_keys = _gather_knowledge_context(goal)
    goal_escaped = goal.replace('"', '\\"')

    prompt = _DECOMPOSE_TEMPLATE.format(
        goal=goal,
        user_context=context or "(none)",
        knowledge_context=knowledge_context,
        goal_escaped=goal_escaped,
    )

    return (
        f"## Self-Decomposition Mode\n\n"
        f"You have the best context for this task — you can read the actual code.\n"
        f"Use the template below to structure your decomposition.\n\n"
        f"**Instructions:**\n"
        f"1. Read the relevant source files to ground your plan in real code\n"
        f"2. Follow the JSON format below\n"
        f"3. Post result to knowledge.db via multihead_deposit_claim\n\n"
        f"**Knowledge claims found:** {len(context_keys)}\n\n"
        f"---\n\n"
        f"{prompt}\n\n"
        f"---\n\n"
        f"After reading relevant code, produce the JSON plan above and post it "
        f"to knowledge.db using multihead_deposit_claim with:\n"
        f"- claim_key: 'decomp.proposal.<your_session_id>.<task_short_name>'\n"
        f"- claim_type: 'plan'\n"
        f"- statement: 'DECOMP_PROPOSAL: <your JSON plan>'\n"
    )


async def _decompose_claude_p(goal: str, context: str = "") -> str:
    """Decompose via claude -p subprocess."""
    import asyncio
    import os
    import shutil
    import subprocess

    claude_bin = shutil.which("claude")
    if not claude_bin:
        return "Error: claude CLI not found in PATH. Falling back to qwen."

    knowledge_context, _ = _gather_knowledge_context(goal)
    goal_escaped = goal.replace('"', '\\"')

    prompt = _DECOMPOSE_TEMPLATE.format(
        goal=goal,
        user_context=context or "(none)",
        knowledge_context=knowledge_context,
        goal_escaped=goal_escaped,
    )

    cmd = [
        claude_bin, "-p", prompt,
        "--output-format", "json",
        "--model", os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6"),
        "--max-budget-usd", "1.0",
        "--allowedTools", "Read,Grep,Glob",
        "--permission-mode", "acceptEdits",
    ]

    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    work_dir = os.environ.get("CLAUDE_WORK_DIR", os.getcwd())

    try:
        loop = asyncio.get_event_loop()
        proc = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=work_dir, timeout=300, env=env,
                creationflags=no_window_flags(),
            ),
        )
    except subprocess.TimeoutExpired:
        return "Error: claude -p decomposition timed out after 300s"

    if proc.returncode != 0:
        return f"Error: claude -p exit {proc.returncode}: {proc.stderr[:500]}"

    stdout = proc.stdout.strip()
    if not stdout:
        return "Error: empty output from claude -p"

    # Parse message list format
    try:
        parsed = json.loads(stdout)
        if isinstance(parsed, list):
            # Extract text from message list
            texts = []
            for msg in parsed:
                if isinstance(msg, dict):
                    for block in msg.get("content", []):
                        if isinstance(block, dict) and block.get("type") == "text":
                            texts.append(block["text"])
            result_text = "\n".join(texts)
        elif isinstance(parsed, dict) and "phases" in parsed:
            result_text = json.dumps(parsed, indent=2)
        else:
            result_text = stdout
    except json.JSONDecodeError:
        result_text = stdout

    return f"## Decomposition (via claude -p)\n\n{result_text}"


async def _decompose_qwen(goal: str, context: str = "", head_id: str | None = None, max_depth: int = 4) -> str:
    """Decompose via local LLM (Qwen) through API."""
    payload: dict = {"goal": goal, "context": context, "max_depth": max_depth}
    if head_id:
        payload["head_id"] = head_id
    try:
        result = await _request("POST", "/decompose", json=payload)
        tree = result.get("tree", "")
        meta = (
            f"Complexity: {result.get('complexity')}\n"
            f"Total steps: {result.get('total_steps')}\n"
            f"Max depth: {result.get('max_depth')}\n"
            f"Context claims used: {len(result.get('context_used', []))}\n"
        )
        return f"{meta}\n{tree}\n\n---\nFull JSON phases available via /decompose API."
    except httpx.ConnectError:
        return "Error: MultiHead server not running. Start it with: multihead serve"
    except httpx.HTTPStatusError as e:
        return f"Error ({e.response.status_code}): {e.response.text}"
    except Exception as e:
        return f"Error: {e}"


async def _decompose(goal: str, context: str = "", head_id: str | None = None, max_depth: int = 4, strategy: str = "self") -> str:
    """Route decomposition to the appropriate strategy.

    Strategies (in priority order):
      1. "self"     — returns prompt+context for calling session to decompose
                      (best quality: session has loaded codebase + conversation)
      2. "claude-p" — spawns claude -p subprocess
      3. "qwen"     — routes through local LLM via API
    """
    if strategy == "self":
        return await _decompose_self(goal, context)
    elif strategy == "claude-p":
        result = await _decompose_claude_p(goal, context)
        if result.startswith("Error:") and "Falling back" in result:
            logger.warning("claude -p failed, falling back to qwen: %s", result)
            return await _decompose_qwen(goal, context, head_id, max_depth)
        return result
    elif strategy == "qwen":
        return await _decompose_qwen(goal, context, head_id, max_depth)
    else:
        return f"Error: unknown strategy '{strategy}'. Use 'self', 'claude-p', or 'qwen'."


async def _refine_step(node_id: str, node_goal: str, action_type: str = "", target_files: list[str] | None = None, exploration_result: str = "", head_id: str | None = None) -> str:
    payload: dict = {
        "node_id": node_id,
        "node_goal": node_goal,
        "action_type": action_type,
        "target_files": target_files or [],
        "exploration_result": exploration_result,
    }
    if head_id:
        payload["head_id"] = head_id
    try:
        result = await _request("POST", "/decompose/refine", json=payload)
        children = result.get("children", [])
        lines = [f"Refined {node_id} into {len(children)} sub-steps:\n"]
        for c in children:
            action = f" [{c.get('action_type', '')}]" if c.get('action_type') else ""
            files = f" -> {', '.join(c.get('target_files', []))}" if c.get('target_files') else ""
            lines.append(f"  {c['id']}. {c['goal']}{action}{files}")
        return "\n".join(lines)
    except httpx.ConnectError:
        return "Error: MultiHead server not running."
    except httpx.HTTPStatusError as e:
        return f"Error ({e.response.status_code}): {e.response.text}"
    except Exception as e:
        return f"Error: {e}"
