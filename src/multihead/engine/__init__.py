"""MultiHead Engine — embeddable Python SDK.

Layer 1: Use MultiHead as a library in your own applications.
You manage context and conversation; MultiHead provides heads,
routing, knowledge, tools, and mesh infrastructure.

Usage:
    from multihead import Engine

    engine = Engine(config_dir="./config")
    await engine.start()

    # Generate text (auto-routes to best available head)
    result = await engine.generate("Explain this code", kind="llm")
    print(result["text"])

    # Route to specific head
    head_id = engine.route("vlm")
    result = await engine.generate("Describe image", head_id=head_id)

    # Access knowledge store
    claims = engine.knowledge.list_claims(status="accepted", limit=10)

    # Clean shutdown
    await engine.stop()
"""

from ._core import Engine
from ._marketplace import _MarketplaceMixin
from ._solve import _SolveMixin

# Re-export for backward compatibility (tests patch multihead.engine.RFQManager)
from ..rfq_manager import RFQManager

__all__ = ["Engine", "_MarketplaceMixin", "_SolveMixin", "RFQManager"]
