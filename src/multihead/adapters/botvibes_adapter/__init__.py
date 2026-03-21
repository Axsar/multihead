"""BotVibes Adapter: Execute tasks on BotVibes marketplace providers.

This package was refactored from a single module into a sub-package.
All public names are re-exported here so existing imports continue to work:

    from multihead.adapters.botvibes_adapter import BotVibesAdapter
    from multihead.adapters.botvibes_adapter import PrivacyViolation
"""

from ._adapter import BotVibesAdapter
from ._constants import (
    PrivacyViolation,
    _ACP_TIMEOUT,
    _MAX_RETRIES,
    _MAX_WAIT_TIME_S,
    _RETRYABLE_STATUS_CODES,
    _RETRY_BACKOFF_BASE_S,
    _RFQ_BID_POLL_S,
    _RFQ_BID_WAIT_S,
    _TASK_POLL_INTERVAL_S,
)

__all__ = [
    "BotVibesAdapter",
    "PrivacyViolation",
    "_ACP_TIMEOUT",
    "_MAX_RETRIES",
    "_MAX_WAIT_TIME_S",
    "_RETRYABLE_STATUS_CODES",
    "_RETRY_BACKOFF_BASE_S",
    "_RFQ_BID_POLL_S",
    "_RFQ_BID_WAIT_S",
    "_TASK_POLL_INTERVAL_S",
]
