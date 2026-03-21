"""Allow running as: python -m multihead.mcp_server"""

from dotenv import load_dotenv
load_dotenv()

from . import run_mcp_server
run_mcp_server()
