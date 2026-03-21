"""ACP Bridge package — connects MultiHead to BotVibes/ACP.

Re-exports everything that was previously available from the
``multihead.acp_bridge`` module so that all existing imports
continue to work unchanged.
"""

from ._bridge import ACPBridge

__all__ = ["ACPBridge"]
