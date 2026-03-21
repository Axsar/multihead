"""MCP server: thin bridge from Claude Code to MultiHead HTTP API.

Runs as a stdio MCP server that proxies all requests to the running
MultiHead FastAPI server at localhost:7337. Zero GPU state — just HTTP calls.

Usage:
    multihead mcp                      # default localhost:7337
    multihead mcp --port 8080          # custom port
    python -m multihead.mcp_server     # direct invocation
"""

from __future__ import annotations

# --- Core shared state ---
from ._core import (
    _BASE_URL as _BASE_URL,
    _TIMEOUT as _TIMEOUT,
    _get_ks as _get_ks,
    _knowledge_store as _knowledge_store,
    _request as _request,
    logger as logger,
    mcp as mcp,
)

# --- Core tool logic ---
from ._tools_core import (
    _briefing as _briefing,
    _chat as _chat,
    _config as _config,
    _deposit_action_claim as _deposit_action_claim,
    _deposit_claim as _deposit_claim,
    _generate as _generate,
    _heads as _heads,
    _knowledge as _knowledge,
    _report_event as _report_event,
    _run_recipe as _run_recipe,
    _run_status as _run_status,
    _swap_head as _swap_head,
)

# --- ACP / marketplace tool logic ---
from ._tools_acp import (
    _check_tasks as _check_tasks,
    _claim_task as _claim_task,
    _complete_task as _complete_task,
    _create_task as _create_task,
    _delegate_claude as _delegate_claude,
)

# --- Solve pipeline ---
from ._tools_solve import _solve as _solve

# --- Decomposition ---
from ._tools_decompose import (
    _DECOMPOSE_TEMPLATE as _DECOMPOSE_TEMPLATE,
    _decompose as _decompose,
    _decompose_claude_p as _decompose_claude_p,
    _decompose_qwen as _decompose_qwen,
    _decompose_self as _decompose_self,
    _gather_knowledge_context as _gather_knowledge_context,
    _refine_step as _refine_step,
)

# --- Harvest ---
from ._tools_harvest import _harvest as _harvest

# --- MCP tool registrations (importing triggers @mcp.tool() decoration) ---
from ._registrations import (  # noqa: F401
    multihead_briefing,
    multihead_chat,
    multihead_check_inbox,
    multihead_check_tasks,
    multihead_claim_task,
    multihead_collect_votes,
    multihead_complete_task,
    multihead_config,
    multihead_create_task,
    multihead_decompose,
    multihead_delegate_claude,
    multihead_deposit_action,
    multihead_deposit_claim,
    multihead_generate,
    multihead_harvest,
    multihead_heads,
    multihead_knowledge,
    multihead_marketplace_procure,
    multihead_refine_step,
    multihead_report_event,
    multihead_run_recipe,
    multihead_run_status,
    multihead_complete_step,
    multihead_finalize_solve,
    multihead_solve,
    multihead_swap_head,
)


# --- Entry point ---

def run_mcp_server(
    host: str = "127.0.0.1",
    port: int = 7337,
    transport: str = "stdio",
    mcp_port: int = 8338,
) -> None:
    """Start the MCP server.

    Args:
        host: MultiHead API host to proxy to.
        port: MultiHead API port to proxy to.
        transport: 'stdio' for Claude Code CLI, 'sse' or 'streamable-http' for Cowork/Desktop.
        mcp_port: Port to listen on when using HTTP transports (default 8338).
    """
    from . import _core
    _core._BASE_URL = f"http://{host}:{port}"
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport=transport, host="0.0.0.0", port=mcp_port)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_mcp_server()
