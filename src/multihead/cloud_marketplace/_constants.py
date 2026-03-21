"""Shared constants for cloud marketplace modules."""

from __future__ import annotations

import logging

logger = logging.getLogger("multihead.cloud_marketplace")

# API timeout for cloud requests
CLOUD_TIMEOUT = 30.0
# Cache TTL for our listings (seconds)
LISTINGS_CACHE_TTL = 300  # 5 minutes
# Refresh JWT this many seconds before expiry
TOKEN_REFRESH_MARGIN_S = 300  # 5 minutes
# Max remembered RFQ/contract IDs (prevents unbounded memory growth)
MAX_SEEN_IDS = 10_000
# Max vault download size (100 MB)
MAX_VAULT_DOWNLOAD_BYTES = 100 * 1024 * 1024
