"""Claude Code CLI adapter for using Claude as a head.

Shells out to `claude -p --model <model>` for each generate call.
No GPU required — uses Claude's API via the CLI.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import sys
from typing import Any

from ..models import HeadManifest
from .base import HeadAdapter

logger = logging.getLogger(__name__)

_GENERATE_TIMEOUT = 300.0  # Claude API can be slow on large prompts


class ClaudeAdapter(HeadAdapter):
    """Adapter that shells out to the Claude Code CLI.

    Runs `claude -p --model <model> "prompt"` as a subprocess.
    The model defaults to 'sonnet' but can be configured in heads.yaml.
    """

    def __init__(self, manifest: HeadManifest) -> None:
        super().__init__(manifest)
        self.model_name = manifest.model or "sonnet"
        self.max_tokens = manifest.extra.get("max_tokens", 16384)
        self._claude_path: str | None = None

    async def load(self) -> None:
        """Verify claude CLI is available."""
        path = shutil.which("claude")
        if path is None:
            raise RuntimeError("claude CLI not found on PATH")
        self._claude_path = path
        logger.info("Claude adapter ready (model=%s)", self.model_name)

    async def unload(self) -> None:
        """No-op — stateless subprocess adapter."""

    async def generate(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Run claude -p and return the response."""
        model = kwargs.pop("model", self.model_name)
        system_prompt = kwargs.pop("system_prompt", None) or (
            "You are a helpful assistant executing a step in an automated pipeline. "
            "Follow the instructions exactly. Return only what is asked for."
        )
        cmd = [
            self._claude_path or "claude",
            "-p",
            "--model", model,
            "--output-format", "json",
            "--no-session-persistence",
            "--allowedTools", "",
            "--mcp-config", '{"mcpServers":{}}',
            "--strict-mcp-config",
            "--system-prompt", system_prompt,
        ]

        # Build env: must unset CLAUDECODE to allow nested invocation
        env = {**os.environ}
        env.pop("CLAUDECODE", None)

        exec_kwargs: dict[str, Any] = dict(
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        if sys.platform == "win32":
            exec_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        proc = await asyncio.create_subprocess_exec(*cmd, **exec_kwargs)

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=prompt.encode("utf-8")),
                timeout=kwargs.get("timeout", _GENERATE_TIMEOUT),
            )
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError(f"Claude CLI timed out after {_GENERATE_TIMEOUT}s")

        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Claude CLI exited {proc.returncode}: {err}")

        raw = stdout.decode("utf-8", errors="replace")

        # --output-format json returns a JSON array: [init, assistant, result]
        # The last element (type=result) has the text, cost, and usage info
        import json
        text = raw.strip()
        cost = 0.0
        tokens_in = 0
        tokens_out = 0
        try:
            items = json.loads(raw)
            if isinstance(items, list):
                # Find the result object
                for item in items:
                    if item.get("type") == "result":
                        text = item.get("result", text)
                        cost = item.get("total_cost_usd", 0)
                        usage = item.get("usage", {})
                        tokens_in = usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
                        tokens_out = usage.get("output_tokens", 0)
                        break
            elif isinstance(items, dict):
                text = items.get("result", text)
                cost = items.get("total_cost_usd", 0)
                usage = items.get("usage", {})
                tokens_in = usage.get("input_tokens", 0)
                tokens_out = usage.get("output_tokens", 0)
        except (json.JSONDecodeError, KeyError):
            pass

        # Estimate tokens if not provided
        if tokens_out == 0:
            tokens_out = len(text) // 4  # rough estimate

        return {
            "text": text,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "model": model,
            "cost_usd": cost,
            "finish_reason": "stop",
        }

    async def healthcheck(self) -> bool:
        """Check that claude CLI is available."""
        return shutil.which("claude") is not None

    async def sleep(self, level: int = 1) -> None:
        """No-op."""

    async def wake(self) -> None:
        """Re-verify CLI availability."""
        await self.load()
