"""Claude Code ACP Worker Daemon.

Listens for ACP tasks targeting claude-session-agent via WebSocket + HTTP
polling, then processes them by invoking Claude Code.

Three execution modes per task:
- sdk (default): use Claude Agent SDK in-process — typed messages, streaming,
  session resume, in-process MCP tools for MultiHead integration
- headless: spawn `claude -p` subprocess (legacy fallback)
- interactive: send keystroke to a running Claude Code tmux session

Multi-turn support: conversation_id maps to Claude session_id via --resume.

Usage:
    export ACP_URL="http://localhost:8000/api/v1"
    export ACP_SESSION_KEY="<jwt>"
    python scripts/claude_worker.py [--mode sdk|headless|interactive]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="[worker] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("claude_worker")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ACP_URL = os.environ.get("ACP_URL", "http://localhost:8000/api/v1").strip()
ACP_API_KEY = (os.environ.get("ACP_CLAUDE_SESSION_KEY") or os.environ.get("ACP_SESSION_KEY") or "").strip()
AGENT_ID = "claude-session-agent"
POLL_CAPABILITIES = ["com.multihead", "com.claude.code"]  # Our namespace + shared fallback
# Capabilities reserved for the interactive session — daemon skips these
INTERACTIVE_CAPABILITIES = {"com.multihead.interactive"}
POLL_INTERVAL = 30
HEARTBEAT_INTERVAL = 120  # seconds between heartbeats
PROJECT_ID = os.environ.get("ACP_PROJECT_ID", "")
CLAUDE_WORK_DIR = os.environ.get("CLAUDE_WORK_DIR", os.getcwd())
_DATA_DIR_DEFAULT = os.environ.get("MULTIHEAD_DATA_DIR", os.path.join(os.path.expanduser("~"), ".multihead"))
CONTEXT_PACK = os.environ.get(
    "CLAUDE_CONTEXT_PACK",
    os.path.join(_DATA_DIR_DEFAULT, "context", "daemon_briefing.md"),
)
MAX_BUDGET_USD = float(os.environ.get("CLAUDE_MAX_BUDGET", "2.0"))
SUBPROCESS_TIMEOUT = int(os.environ.get("CLAUDE_TIMEOUT", "300"))
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
ALLOWED_TOOLS = os.environ.get(
    "CLAUDE_ALLOWED_TOOLS",
    "Read,Grep,Glob,Edit,Write,WebSearch,WebFetch,Bash(git:*)",
)


class ClaudeWorker:
    """ACP worker that processes tasks via Claude Agent SDK or subprocess."""

    def __init__(
        self,
        mode: str = "sdk",
        tmux_target: str = "claude",
    ) -> None:
        self.acp_url = ACP_URL
        self.api_key = ACP_API_KEY
        self.agent_id = AGENT_ID
        self.default_mode = mode  # "sdk", "headless", or "interactive"
        self.tmux_target = tmux_target
        self.active_sessions: dict[str, str] = {}  # conversation_id → claude session_id
        self._seen: set[str] = set()
        self._processing: set[str] = set()  # tasks currently being processed
        self._sdk_adapter = None  # Lazy-init ClaudeAgentSDKAdapter
        self.narrative_pipeline = self._init_narrative()

    def _init_narrative(self):
        """Initialize narrative pipeline if KnowledgeStore is available."""
        try:
            from pathlib import Path as P
            from multihead.knowledge_store import KnowledgeStore
            from multihead.artifact_store import ArtifactStore
            from multihead.narrative.pipeline import NarrativePipeline
            data_dir = P(os.environ.get("MULTIHEAD_DATA_DIR", P.home() / ".multihead"))
            db_path = data_dir / "knowledge.db"
            if db_path.parent.exists():
                ks = KnowledgeStore(db_path)
                art = ArtifactStore(data_dir / "artifacts", data_dir / "multihead.db")
                logger.info("Narrative pipeline initialized (%s)", db_path)
                return NarrativePipeline(ks, project_id="multihead", artifact_store=art)
        except Exception as e:
            logger.warning("Narrative pipeline unavailable: %s", e)
        return None

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Agent registration & heartbeat
    # ------------------------------------------------------------------

    def _register_and_heartbeat_sync(self) -> None:
        """Register agent with ACP and send initial heartbeat."""
        try:
            with httpx.Client(timeout=10.0) as client:
                # Register (idempotent — returns 201 on first, 200/409 if exists)
                reg = {
                    "agent_id": self.agent_id,
                    "project_id": PROJECT_ID,
                    "visibility": "private",
                    "descriptor": {
                        "capabilities": POLL_CAPABILITIES,
                        "input_schema": ["application/json", "text/plain"],
                        "output_schema": ["application/json", "text/plain"],
                        "latency_profile": {"p50_ms": 30000, "p95_ms": 120000},
                        "cost_model": {"unit": "task", "price": 0.05, "currency": "USD"},
                        "max_concurrency": 1,
                        "description": "Claude Code worker daemon — headless task execution",
                    },
                }
                resp = client.post(
                    f"{self.acp_url}/agents/register",
                    headers=self._headers(),
                    json=reg,
                )
                logger.info("Agent register: %d", resp.status_code)

                # Heartbeat
                resp = client.post(
                    f"{self.acp_url}/agents/{self.agent_id}/heartbeat",
                    headers=self._headers(),
                    json={"status": "idle"},
                )
                if resp.status_code == 200:
                    logger.info("Heartbeat: idle")
                else:
                    logger.warning("Heartbeat failed: %d %s", resp.status_code, resp.text[:200])
        except Exception as e:
            logger.warning("Registration/heartbeat error: %s", e)

    async def _heartbeat_loop(self) -> None:
        """Periodically send heartbeat to keep agent active."""
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            await asyncio.to_thread(self._register_and_heartbeat_sync)

    # ------------------------------------------------------------------
    # WebSocket URL
    # ------------------------------------------------------------------

    def _ws_url(self) -> str:
        """Build WebSocket URL (WS router is at root, not under /api/v1)."""
        url = self.acp_url.replace("https://", "wss://").replace("http://", "ws://")
        for suffix in ("/api/v1", "/api/v1/", "/api"):
            if url.endswith(suffix):
                url = url[: -len(suffix)]
                break
        return f"{url}/ws/agents/{self.agent_id}?token={self.api_key}"

    # ------------------------------------------------------------------
    # Task claiming
    # ------------------------------------------------------------------

    async def _claim_task(self, client: httpx.AsyncClient, task_id: str) -> bool:
        """Reserve + dispatch a task. Returns True on success."""
        try:
            resp = await client.post(
                f"{self.acp_url}/tasks/{task_id}/reserve",
                headers=self._headers(),
                json={"agent_id": self.agent_id, "reservation_ttl_ms": 300_000},
            )
            if resp.status_code != 200 or not resp.json().get("accepted"):
                logger.warning("Reserve failed for %s: %s", task_id[:8], resp.text[:200])
                return False

            resp = await client.post(
                f"{self.acp_url}/tasks/{task_id}/dispatch",
                headers=self._headers(),
                json={"agent_id": self.agent_id},
            )
            if resp.status_code != 200:
                logger.warning("Dispatch failed for %s: %s", task_id[:8], resp.text[:200])
                return False

            return True
        except Exception as e:
            logger.error("Claim error for %s: %s", task_id[:8], e)
            return False

    async def _submit_result(
        self,
        client: httpx.AsyncClient,
        task_id: str,
        output: str,
        status: str = "complete",
        confidence: float = 0.9,
    ) -> None:
        """Submit task result back to ACP."""
        try:
            payload: dict = {
                "status": status,
                "output_ref": output[:5000],  # Cap output size
                "confidence": confidence,
            }
            if status == "failed":
                payload["error_code"] = "EXECUTION_ERROR"
                payload["message"] = output[:500]

            await client.post(
                f"{self.acp_url}/tasks/{task_id}/result",
                headers=self._headers(),
                json=payload,
            )
            logger.info("Result submitted for %s (status=%s)", task_id[:8], status)
        except Exception as e:
            logger.error("Result submission error for %s: %s", task_id[:8], e)

    # ------------------------------------------------------------------
    # Headless Claude Code invocation
    # ------------------------------------------------------------------

    def _load_context_pack(self) -> str:
        """Load context pack for enriching prompts.

        Combines the static daemon briefing with auto-generated narrative
        context (accepted claims + confirmed events from the knowledge store).
        """
        parts: list[str] = []
        for filename in (CONTEXT_PACK, os.path.join(os.path.dirname(CONTEXT_PACK), "daemon_narrative.md")):
            try:
                if os.path.isfile(filename):
                    with open(filename) as f:
                        content = f.read().strip()
                        if content:
                            parts.append(content)
            except Exception as e:
                logger.warning("Context pack load error (%s): %s", filename, e)
        return "\n\n".join(parts)

    async def _ensure_sdk_adapter(self) -> None:
        """Lazy-init the Claude Agent SDK adapter with MCP tools."""
        if self._sdk_adapter is not None:
            return

        try:
            from multihead.adapters.claude_agent_sdk import ClaudeAgentSDKAdapter
            from multihead.models import AdapterKind, HeadManifest

            manifest = HeadManifest(
                head_id="worker-claude-sdk",
                name="Claude Worker SDK",
                adapter=AdapterKind.CLAUDE_AGENT_SDK,
                model=CLAUDE_MODEL,
                kind="llm",
                gpu_required=False,
                extra={
                    "max_turns": 25,
                    "max_budget_usd": MAX_BUDGET_USD,
                    "permission_mode": "acceptEdits",
                    "effort": "high",
                    "cwd": CLAUDE_WORK_DIR,
                    "allowed_tools": [t.strip() for t in ALLOWED_TOOLS.split(",") if t.strip()],
                },
            )
            self._sdk_adapter = ClaudeAgentSDKAdapter(manifest)
            await self._sdk_adapter.load()

            # Inject in-process MCP tools if knowledge store is available
            if self.narrative_pipeline:
                try:
                    from multihead.sdk_mcp_tools import build_sdk_mcp_server
                    ks = self.narrative_pipeline._knowledge_store
                    server = build_sdk_mcp_server(knowledge_store=ks)
                    self._sdk_adapter.set_mcp_servers({"multihead": server})
                    logger.info("SDK adapter ready with MCP tools")
                except Exception as e:
                    logger.warning("SDK MCP tools unavailable: %s", e)
            else:
                logger.info("SDK adapter ready (no MCP tools)")

        except ImportError as e:
            logger.error("Claude Agent SDK not installed: %s", e)
            raise
        except Exception as e:
            logger.error("SDK adapter init failed: %s", e)
            self._sdk_adapter = None
            raise

    async def _invoke_claude_sdk(self, prompt: str, session_id: str | None = None,
                                  conversation_id: str | None = None) -> dict:
        """Invoke Claude via Agent SDK (in-process, typed messages)."""
        await self._ensure_sdk_adapter()

        # Enrich prompt with context pack (skip for resumed sessions)
        if not session_id:
            context = self._load_context_pack()
            if context:
                prompt = f"{context}\n\n---\n\nTask:\n{prompt}"

        start = time.monotonic()
        kwargs: dict = {"conversation_id": conversation_id or "worker-default"}
        if session_id:
            kwargs["resume"] = session_id

        try:
            result = await self._sdk_adapter.generate(prompt, **kwargs)
            elapsed = time.monotonic() - start

            logger.info("SDK completed in %.1fs (cost=$%.4f, turns=%d)",
                       elapsed, result.get("cost_usd", 0), result.get("turns", 0))

            return {
                "result": result.get("text", ""),
                "session_id": result.get("session_id", ""),
                "cost_usd": result.get("cost_usd", 0),
                "turns": result.get("turns", 0),
            }
        except Exception as e:
            elapsed = time.monotonic() - start
            logger.error("SDK error after %.1fs: %s", elapsed, e)
            return {"error": str(e)}

    def _invoke_claude(self, prompt: str, session_id: str | None = None) -> dict:
        """Spawn headless Claude Code and return parsed JSON output."""
        claude_bin = shutil.which("claude")
        if not claude_bin:
            return {"error": "claude CLI not found in PATH"}

        # Enrich prompt with context pack (skip for resumed sessions — they have context)
        if not session_id:
            context = self._load_context_pack()
            if context:
                prompt = f"{context}\n\n---\n\nTask:\n{prompt}"

        cmd = [claude_bin, "-p", prompt, "--output-format", "json"]
        if CLAUDE_MODEL:
            cmd.extend(["--model", CLAUDE_MODEL])
        if session_id:
            cmd.extend(["--resume", session_id])
        cmd.extend([
            "--allowedTools", ALLOWED_TOOLS,
            "--permission-mode", "acceptEdits",
            "--max-budget-usd", str(MAX_BUDGET_USD),
        ])

        # Build clean env: unset CLAUDECODE to allow subprocess spawning
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

        logger.info("Spawning: claude -p '%s...' (session=%s)", prompt[:60], session_id or "new")
        start = time.monotonic()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=CLAUDE_WORK_DIR,
                timeout=SUBPROCESS_TIMEOUT,
                env=env,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start
            return {"error": f"Claude subprocess timed out after {elapsed:.0f}s"}

        elapsed = time.monotonic() - start
        logger.info("Claude completed in %.1fs (exit=%d)", elapsed, result.returncode)

        if result.returncode != 0:
            stderr = result.stderr.strip()
            return {"error": f"Exit {result.returncode}: {stderr[:500]}"}

        # Parse JSON output
        stdout = result.stdout.strip()
        if not stdout:
            return {"error": "Empty output from Claude"}

        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            # Claude might output text before JSON in some modes
            for line in stdout.split("\n"):
                line = line.strip()
                if line.startswith(("{", "[")):
                    try:
                        parsed = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
            else:
                return {"result": stdout[:2000], "error": "Non-JSON output"}

        # claude -p --output-format json returns a list of message objects.
        # Normalize to a flat dict with session_id, result, cost_usd.
        if isinstance(parsed, list):
            return self._normalize_claude_output(parsed)
        return parsed

    @staticmethod
    def _normalize_claude_output(messages: list[dict]) -> dict:
        """Extract session_id and result text from Claude's message list."""
        result: dict = {}

        # session_id is in the init message
        for msg in messages:
            if msg.get("type") == "system" and msg.get("subtype") == "init":
                result["session_id"] = msg.get("session_id", "")
                break

        # Cost info is in the last system/result message
        for msg in reversed(messages):
            if msg.get("type") == "result":
                result["cost_usd"] = msg.get("cost_usd", 0)
                result["result"] = msg.get("result", "")
                result["is_error"] = msg.get("is_error", False)
                return result

        # Fallback: concatenate assistant message text
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

    # ------------------------------------------------------------------
    # Interactive mode (tmux send-keys)
    # ------------------------------------------------------------------

    def _wake_interactive(self, message: str) -> dict:
        """Send keystroke to interactive Claude Code tmux session."""
        try:
            # Check if tmux session exists
            check = subprocess.run(
                ["tmux", "has-session", "-t", self.tmux_target],
                capture_output=True, timeout=5,
            )
            if check.returncode != 0:
                return {"error": f"tmux session '{self.tmux_target}' not found. "
                        f"Start Claude Code with: tmux new-session -d -s {self.tmux_target} claude"}

            # Send the message as keystrokes
            subprocess.run(
                ["tmux", "send-keys", "-t", self.tmux_target, message, "Enter"],
                check=True, timeout=5,
            )
            return {"result": f"Sent to tmux session '{self.tmux_target}'", "mode": "interactive"}

        except FileNotFoundError:
            return {"error": "tmux not installed"}
        except subprocess.TimeoutExpired:
            return {"error": "tmux command timed out"}
        except subprocess.CalledProcessError as e:
            return {"error": f"tmux error: {e}"}

    # ------------------------------------------------------------------
    # Task processing
    # ------------------------------------------------------------------

    async def _process_task(self, task: dict) -> None:
        """Claim task, invoke Claude (headless or interactive), submit result."""
        tid = task.get("task_id", "?")
        if tid in self._seen or tid in self._processing:
            return

        self._processing.add(tid)
        payload = task.get("payload_ref", "")
        conv_id = task.get("conversation_id")
        # Allow task metadata to override mode
        mode = task.get("mode", self.default_mode)

        logger.info("=== NEW TASK %s ===", tid[:8])
        logger.info("Payload: %s", payload[:200])
        logger.info("Mode: %s | Conversation: %s", mode, conv_id or "none")

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Claim
            if not await self._claim_task(client, tid):
                self._seen.add(tid)
                self._processing.discard(tid)
                return

            # Execute based on mode
            if mode == "interactive":
                output = await asyncio.to_thread(self._wake_interactive, payload)
                result_text = output.get("result", output.get("error", "No output"))
                await self._submit_result(client, tid, result_text)
            elif mode == "sdk":
                # SDK: in-process Claude Agent SDK (preferred)
                session_id = self.active_sessions.get(conv_id) if conv_id else None
                try:
                    output = await self._invoke_claude_sdk(payload, session_id, conv_id)
                except Exception as e:
                    logger.warning("SDK failed, falling back to headless: %s", e)
                    output = await asyncio.to_thread(self._invoke_claude, payload, session_id)
            else:
                # Headless: subprocess fallback
                session_id = self.active_sessions.get(conv_id) if conv_id else None
                output = await asyncio.to_thread(self._invoke_claude, payload, session_id)

                # Track session for multi-turn continuity
                if conv_id and output.get("session_id"):
                    self.active_sessions[conv_id] = output["session_id"]
                    logger.info("Session tracked: %s → %s", conv_id[:8], output["session_id"][:8])

                # Extract result
                if "error" in output and "result" not in output:
                    result_text = output["error"]
                    status = "failed"
                    confidence = 0.0
                else:
                    result_text = output.get("result", json.dumps(output, indent=2))
                    status = "complete"
                    confidence = 0.9

                # Log cost if available
                cost = output.get("cost_usd") or output.get("total_cost_usd")
                if cost:
                    logger.info("Task cost: $%.4f", cost)

                # Record in narrative pipeline
                if self.narrative_pipeline and status == "complete":
                    try:
                        self.narrative_pipeline.ingest_agent_result(
                            result={"task": payload, "output": result_text,
                                    "status": status, "model": "claude-code",
                                    "cost_usd": cost},
                            agent_id=self.agent_id,
                            agent_type="claude",
                            parent_session_id=conv_id,
                        )
                    except Exception as e:
                        logger.debug("Narrative ingest error: %s", e)

                await self._submit_result(client, tid, result_text, status, confidence)

            self._seen.add(tid)
            self._processing.discard(tid)
            logger.info("=== TASK %s DONE ===\n", tid[:8])

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def _poll_once_sync(self) -> list[dict]:
        """HTTP poll for available tasks across all our capabilities.

        Polls with both schema formats to catch tasks created with either
        MIME-type ("application/json") or JSON Schema ('{"type":"object"}').
        """
        all_tasks: list[dict] = []
        seen_ids: set[str] = set()
        # Some agents create tasks with JSON Schema strings instead of MIME types.
        # BotVibes /tasks/available defaults to application/json but we need both.
        schema_variants = [
            ("application/json", "application/json"),
            ('{"type":"object"}', '{"type":"object"}'),
            ("text/plain", "text/plain"),
        ]
        try:
            with httpx.Client(timeout=10.0) as client:
                for cap in POLL_CAPABILITIES:
                    for in_schema, out_schema in schema_variants:
                        resp = client.get(
                            f"{self.acp_url}/tasks/available",
                            headers=self._headers(),
                            params={
                                "capability": cap,
                                "limit": 10,
                                "input_schema": in_schema,
                                "output_schema": out_schema,
                            },
                        )
                        if resp.status_code == 401:
                            logger.warning("JWT expired or invalid (401)")
                            return []
                        if resp.status_code != 200:
                            logger.warning("Poll failed (%s/%s): %d %s", cap, in_schema[:15], resp.status_code, resp.text[:200])
                            continue

                        for task in resp.json():
                            tid = task.get("task_id")
                            task_cap = task.get("required_capability", "")
                            if tid and tid not in seen_ids and task_cap not in INTERACTIVE_CAPABILITIES:
                                seen_ids.add(tid)
                                all_tasks.append(task)
                            elif task_cap in INTERACTIVE_CAPABILITIES:
                                logger.info("Skipping interactive task %s (%s)", tid[:8] if tid else "?", task_cap)

                if all_tasks:
                    logger.info("Found %d available task(s)", len(all_tasks))
                return all_tasks

        except httpx.ConnectError:
            logger.debug("ACP not reachable")
            return []
        except Exception as e:
            logger.error("Poll error: %s", e)
            return []

    async def poll_and_process(self) -> None:
        """Poll for tasks and process them."""
        tasks = await asyncio.to_thread(self._poll_once_sync)
        for task in tasks:
            tid = task.get("task_id")
            if tid and tid not in self._seen and tid not in self._processing:
                await self._process_task(task)

    # ------------------------------------------------------------------
    # WebSocket loop
    # ------------------------------------------------------------------

    async def ws_loop(self) -> None:
        """Primary: WebSocket connection for instant task notifications."""
        try:
            import websockets
        except ImportError:
            logger.warning("websockets not installed — WebSocket disabled, HTTP-only mode")
            return

        backoff = 1.0
        max_backoff = 60.0
        ws_url = self._ws_url()

        while True:
            try:
                logger.info("WebSocket connecting: %s", ws_url.split("?")[0])
                async with websockets.connect(ws_url) as ws:
                    backoff = 1.0
                    logger.info("WebSocket connected — waiting for task notifications")

                    async for message in ws:
                        try:
                            data = json.loads(message)
                            event_type = data.get("type", "unknown")

                            if event_type == "connected":
                                trust = data.get("trust_level", "?")
                                logger.info("WebSocket hello (trust=%s)", trust)
                                continue

                            if event_type in ("task.created", "task.available", "notify"):
                                logger.info("WebSocket event: %s — polling now", event_type)
                                await self.poll_and_process()

                        except json.JSONDecodeError:
                            logger.debug("WebSocket non-JSON: %s", str(message)[:100])
                            await self.poll_and_process()

            except Exception as e:
                logger.warning("WebSocket error (retry in %.0fs): %s", backoff, e)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main loop: WebSocket + HTTP fallback in parallel."""
        async def http_fallback():
            while True:
                await self.poll_and_process()
                await asyncio.sleep(POLL_INTERVAL)

        logger.info("Starting Claude Worker Daemon")
        logger.info("Agent: %s", self.agent_id)
        logger.info("ACP: %s", self.acp_url)
        logger.info("Work dir: %s", CLAUDE_WORK_DIR)
        logger.info("Mode: %s", self.default_mode)
        logger.info("Max budget: $%.2f per task", MAX_BUDGET_USD)
        logger.info("Timeout: %ds per task", SUBPROCESS_TIMEOUT)
        logger.info("WebSocket + HTTP fallback (%ds)", POLL_INTERVAL)

        # Register agent and send initial heartbeat
        await asyncio.to_thread(self._register_and_heartbeat_sync)

        await asyncio.gather(self.ws_loop(), http_fallback(), self._heartbeat_loop())


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Claude Code ACP Worker Daemon")
    parser.add_argument(
        "--mode",
        choices=["sdk", "headless", "interactive"],
        default=os.environ.get("CLAUDE_WORKER_MODE", "sdk"),
        help="Execution mode: sdk (Agent SDK, default), headless (subprocess), interactive (tmux)",
    )
    parser.add_argument(
        "--tmux-target",
        default=os.environ.get("CLAUDE_TMUX_SESSION", "claude"),
        help="tmux session name for interactive mode (default: 'claude')",
    )
    # Legacy flag — maps to --mode interactive
    parser.add_argument(
        "--interactive-only",
        action="store_true",
        help="Shortcut for --mode interactive",
    )
    args = parser.parse_args()

    if not ACP_API_KEY:
        print("Set ACP_SESSION_KEY env var to the claude-session-agent JWT")
        sys.exit(1)

    mode = "interactive" if args.interactive_only else args.mode

    worker = ClaudeWorker(
        mode=mode,
        tmux_target=args.tmux_target,
    )
    asyncio.run(worker.run())


if __name__ == "__main__":
    main()
