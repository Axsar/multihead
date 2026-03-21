"""Cloud marketplace bridge — seller-side integration with BotVibes cloud.

Connects to the BotVibes cloud marketplace to:
1. Accept direct tasks via a cloud ACPBridge instance
2. Auto-quote on open RFQs matching our capabilities
3. Execute awarded contracts locally and post receipts
4. Build trust score through successful contract completion

Runs as a background service managed by ServiceManager.

This package was refactored from a single module into sub-modules for
maintainability.  All public names are re-exported here so that existing
``from multihead.cloud_marketplace import CloudMarketplaceBridge``
(and wildcard imports) continue to work unchanged.
"""

from ._bridge import CloudMarketplaceBridge

__all__ = ["CloudMarketplaceBridge"]
