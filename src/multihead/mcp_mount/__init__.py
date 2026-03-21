"""MCP-over-HTTP endpoint mounted directly on the MultiHead FastAPI app.

Unlike mcp_server.py (stdio proxy), this module creates MCP tools that call
core logic directly -- no HTTP round-trip. Mounted at /mcp on the same port.

Usage:
    # In app.py lifespan, after all components are initialized:
    from ..mcp_mount import mcp, wire_mcp
    wire_mcp(app.state)
    app.mount("/mcp", mcp.http_app(transport="streamable-http", stateless_http=True))
"""

# Core: FastMCP instance, _ctx, wire_mcp, health, heads, chat, config, nightshift
from ._core import mcp, _ctx, wire_mcp  # noqa: F401
from ._core import health, list_heads, wake_head, sleep_head, generate  # noqa: F401
from ._core import chat, list_sessions  # noqa: F401
from ._core import nightshift  # noqa: F401
from ._core import config  # noqa: F401

# Knowledge, runs, solve, packs, consensus
from ._knowledge import query_knowledge, create_claim, create_event, briefing  # noqa: F401
from ._knowledge import list_runs, run_recipe, run_status  # noqa: F401
from ._knowledge import solve, decompose  # noqa: F401
from ._knowledge import list_packs, build_pack  # noqa: F401
from ._knowledge import consensus  # noqa: F401

# Skills, catalog, ACP
from ._skills import list_skills, invoke_skill  # noqa: F401
from ._skills import catalog_browse, catalog_stats, catalog_sync, catalog_install  # noqa: F401
from ._skills import acp_status, acp_reconnect  # noqa: F401
