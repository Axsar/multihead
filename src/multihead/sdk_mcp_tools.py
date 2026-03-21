"""In-process MCP tools for Claude Agent SDK sessions.

Exposes MultiHead capabilities (knowledge store, head management, process
management) as SDK MCP tools. These run in-process with zero IPC overhead
when injected into a Claude SDK session via set_mcp_servers().

Usage:
    from multihead.sdk_mcp_tools import build_sdk_mcp_server
    server = build_sdk_mcp_server(knowledge_store=ks, head_manager=hm)
    adapter.set_mcp_servers({"multihead": server})
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from claude_agent_sdk import SdkMcpTool, create_sdk_mcp_server

    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False


def build_sdk_mcp_server(
    knowledge_store: Any = None,
    head_manager: Any = None,
    process_manager: Any = None,
    project_id: str = "multihead",
) -> Any:
    """Build an in-process MCP server with MultiHead tools.

    Args:
        knowledge_store: KnowledgeStore instance for knowledge tools
        head_manager: HeadManager instance for head management tools
        process_manager: ProcessManager instance for process tools
        project_id: Default project scope for knowledge queries

    Returns:
        McpSdkServerConfig that can be passed to ClaudeAgentOptions.mcp_servers
    """
    if not _SDK_AVAILABLE:
        raise RuntimeError("claude-agent-sdk not installed")

    tools: list[SdkMcpTool] = []

    # --- Knowledge tools ---
    if knowledge_store is not None:
        tools.extend(_build_knowledge_tools(knowledge_store, project_id))

    # --- Head management tools ---
    if head_manager is not None:
        tools.extend(_build_head_tools(head_manager))

    # --- Process management tools ---
    if process_manager is not None:
        tools.extend(_build_process_tools(process_manager))

    return create_sdk_mcp_server(
        name="multihead",
        version="1.2.0",
        tools=tools,
    )


# ---------------------------------------------------------------------------
# Knowledge tools
# ---------------------------------------------------------------------------


def _build_knowledge_tools(ks: Any, project_id: str) -> list[SdkMcpTool]:
    """Build knowledge store tools."""

    @_sdk_tool(
        "knowledge_query",
        "Search the knowledge store for claims matching a query. "
        "Returns relevant facts, decisions, and constraints from past sessions.",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query — keywords to match against claim statements",
                },
                "scope_id": {
                    "type": "string",
                    "description": "Project scope to search within (default: multihead)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default: 10)",
                },
            },
            "required": ["query"],
        },
    )
    async def knowledge_query(args: dict) -> dict[str, Any]:
        query = args["query"]
        scope = args.get("scope_id", project_id)
        limit = args.get("limit", 10)

        keywords = [w.lower() for w in query.split() if len(w) > 2]
        claims = ks.list_claims(scope_id=scope, status="accepted", limit=200)

        matching = []
        for c in claims:
            stmt_lower = c.statement.lower()
            if any(k in stmt_lower for k in keywords):
                matching.append({
                    "claim_id": c.canonical.claim_key if hasattr(c, "canonical") and c.canonical else "",
                    "statement": c.statement[:500],
                    "confidence": getattr(c, "confidence", 0),
                })
                if len(matching) >= limit:
                    break

        return {
            "content": [{
                "type": "text",
                "text": f"Found {len(matching)} claims matching '{query}':\n"
                + "\n".join(
                    f"- [{m['claim_id']}] {m['statement']}" for m in matching
                ) if matching else f"No claims found matching '{query}'",
            }],
        }

    @_sdk_tool(
        "knowledge_deposit",
        "Deposit a new claim into the knowledge store. "
        "Use this to record facts, decisions, or observations.",
        {
            "type": "object",
            "properties": {
                "claim_key": {
                    "type": "string",
                    "description": "Dot-separated key (e.g. 'project.component.fact')",
                },
                "statement": {
                    "type": "string",
                    "description": "Human-readable statement of the claim",
                },
                "claim_type": {
                    "type": "string",
                    "description": "Type: fact, decision, constraint, preference, plan",
                    "enum": ["fact", "decision", "constraint", "preference", "plan"],
                },
                "scope_id": {
                    "type": "string",
                    "description": "Project scope (default: multihead)",
                },
            },
            "required": ["claim_key", "statement"],
        },
    )
    async def knowledge_deposit(args: dict) -> dict[str, Any]:
        claim_key = args["claim_key"]
        statement = args["statement"]
        claim_type = args.get("claim_type", "fact")
        scope = args.get("scope_id", project_id)

        claim = ks.deposit_claim(
            claim_key=claim_key,
            statement=statement,
            claim_type=claim_type,
            scope_id=scope,
            produced_by="claude-sdk-session",
            confidence=0.9,
        )

        claim_id = claim.claim_id if hasattr(claim, "claim_id") else str(claim)
        return {
            "content": [{
                "type": "text",
                "text": f"Claim deposited: {claim_id} ({claim_key})",
            }],
        }

    @_sdk_tool(
        "knowledge_count",
        "Get the number of claims in the knowledge store.",
        {
            "type": "object",
            "properties": {
                "scope_id": {
                    "type": "string",
                    "description": "Project scope to count (default: all scopes)",
                },
            },
        },
    )
    async def knowledge_count(args: dict) -> dict[str, Any]:
        scope = args.get("scope_id")
        # Use direct SQL COUNT instead of loading all records into memory
        try:
            conn_ctx = ks._connect()
            with conn_ctx as conn:
                if scope:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM claims WHERE scope_id = ?", (scope,)
                    ).fetchone()
                else:
                    row = conn.execute("SELECT COUNT(*) FROM claims").fetchone()
                count = row[0]
        except Exception:
            # Fallback: limited list_claims if _connect not available
            if scope:
                claims = ks.list_claims(scope_id=scope, limit=100)
            else:
                claims = ks.list_claims(limit=100)
            count = len(claims)
            if count >= 100:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Knowledge store contains 100+ claims"
                        + (f" in scope '{scope}'" if scope else "")
                        + " (approximate — showing capped count)",
                    }],
                }
        return {
            "content": [{
                "type": "text",
                "text": f"Knowledge store contains {count} claims"
                + (f" in scope '{scope}'" if scope else ""),
            }],
        }

    return [knowledge_query, knowledge_deposit, knowledge_count]


# ---------------------------------------------------------------------------
# Head management tools
# ---------------------------------------------------------------------------


def _build_head_tools(hm: Any) -> list[SdkMcpTool]:
    """Build head management tools."""

    @_sdk_tool(
        "heads_list",
        "List all registered model heads and their states (active, sleeping, off).",
        {"type": "object", "properties": {}},
    )
    async def heads_list(args: dict) -> dict[str, Any]:
        states = hm.get_states()
        lines = []
        for hid, info in states.items():
            state = info.get("state", "unknown")
            name = info.get("name", hid)
            adapter = info.get("adapter", "?")
            lines.append(f"- {hid}: {state} ({name}, adapter={adapter})")

        return {
            "content": [{
                "type": "text",
                "text": f"Registered heads ({len(states)}):\n" + "\n".join(lines),
            }],
        }

    @_sdk_tool(
        "heads_wake",
        "Wake (load) a model head to GPU. Required before the head can generate.",
        {
            "type": "object",
            "properties": {
                "head_id": {
                    "type": "string",
                    "description": "ID of the head to wake (e.g. 'core-llm', 'vision-vlm')",
                },
            },
            "required": ["head_id"],
        },
    )
    async def heads_wake(args: dict) -> dict[str, Any]:
        head_id = args["head_id"]
        await hm.wake_head(head_id)
        return {
            "content": [{
                "type": "text",
                "text": f"Head '{head_id}' is now waking/active",
            }],
        }

    @_sdk_tool(
        "heads_sleep",
        "Put a model head to sleep (offload from GPU to save VRAM).",
        {
            "type": "object",
            "properties": {
                "head_id": {
                    "type": "string",
                    "description": "ID of the head to sleep",
                },
            },
            "required": ["head_id"],
        },
    )
    async def heads_sleep(args: dict) -> dict[str, Any]:
        head_id = args["head_id"]
        await hm.sleep_head(head_id)
        return {
            "content": [{
                "type": "text",
                "text": f"Head '{head_id}' is now sleeping",
            }],
        }

    @_sdk_tool(
        "heads_swap",
        "Switch the active head (GPU mutex — unloads current, loads new).",
        {
            "type": "object",
            "properties": {
                "head_id": {
                    "type": "string",
                    "description": "ID of the head to make active",
                },
            },
            "required": ["head_id"],
        },
    )
    async def heads_swap(args: dict) -> dict[str, Any]:
        head_id = args["head_id"]
        await hm.ensure_active(head_id)
        return {
            "content": [{
                "type": "text",
                "text": f"Active head is now '{head_id}'",
            }],
        }

    @_sdk_tool(
        "heads_generate",
        "Generate text using a specific local model head. "
        "Use this when you need a local GPU model to process something.",
        {
            "type": "object",
            "properties": {
                "head_id": {
                    "type": "string",
                    "description": "Head to use (e.g. 'core-llm' for Qwen3-8B)",
                },
                "prompt": {
                    "type": "string",
                    "description": "Text prompt to generate from",
                },
            },
            "required": ["head_id", "prompt"],
        },
    )
    async def heads_generate(args: dict) -> dict[str, Any]:
        head_id = args["head_id"]
        prompt = args["prompt"]
        result = await hm.generate(head_id, prompt)
        text = result.get("text", "")
        return {
            "content": [{
                "type": "text",
                "text": text,
            }],
        }

    return [heads_list, heads_wake, heads_sleep, heads_swap, heads_generate]


# ---------------------------------------------------------------------------
# Process management tools
# ---------------------------------------------------------------------------


def _build_process_tools(pm: Any) -> list[SdkMcpTool]:
    """Build process management tools."""

    @_sdk_tool(
        "process_spawn",
        "Spawn a background subprocess. Returns the PID for monitoring.",
        {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to run",
                },
            },
            "required": ["command"],
        },
    )
    async def process_spawn(args: dict) -> dict[str, Any]:
        command = args["command"]
        proc = await pm.spawn(command)
        return {
            "content": [{
                "type": "text",
                "text": f"Spawned process PID={proc.pid}: {command}",
            }],
        }

    @_sdk_tool(
        "process_list",
        "List all running background processes.",
        {"type": "object", "properties": {}},
    )
    async def process_list(args: dict) -> dict[str, Any]:
        procs = pm.list_processes()
        if not procs:
            return {"content": [{"type": "text", "text": "No running processes"}]}
        lines = [
            f"- PID {p.pid}: {p.command} ({p.status})" for p in procs
        ]
        return {
            "content": [{
                "type": "text",
                "text": f"Processes ({len(procs)}):\n" + "\n".join(lines),
            }],
        }

    @_sdk_tool(
        "process_output",
        "Read recent output from a background process.",
        {
            "type": "object",
            "properties": {
                "pid": {
                    "type": "integer",
                    "description": "Process ID",
                },
                "lines": {
                    "type": "integer",
                    "description": "Number of recent lines to return (default: 20)",
                },
            },
            "required": ["pid"],
        },
    )
    async def process_output(args: dict) -> dict[str, Any]:
        pid = args["pid"]
        lines = args.get("lines", 20)
        text = pm.output(pid, lines)
        return {
            "content": [{
                "type": "text",
                "text": text if text else f"No output from PID {pid}",
            }],
        }

    @_sdk_tool(
        "process_kill",
        "Kill a running background process.",
        {
            "type": "object",
            "properties": {
                "pid": {
                    "type": "integer",
                    "description": "Process ID to kill",
                },
            },
            "required": ["pid"],
        },
    )
    async def process_kill(args: dict) -> dict[str, Any]:
        pid = args["pid"]
        await pm.kill(pid)
        return {
            "content": [{
                "type": "text",
                "text": f"Process PID={pid} killed",
            }],
        }

    return [process_spawn, process_list, process_output, process_kill]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _sdk_tool(name: str, description: str, input_schema: dict) -> Any:
    """Decorator that creates an SdkMcpTool from a function.

    This is a wrapper around claude_agent_sdk.tool() that works with
    dict-based input schemas (JSON Schema) instead of Pydantic models.
    """
    if not _SDK_AVAILABLE:
        raise RuntimeError("claude-agent-sdk not installed")

    from claude_agent_sdk import SdkMcpTool

    def decorator(fn):
        return SdkMcpTool(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=fn,
        )
    return decorator
