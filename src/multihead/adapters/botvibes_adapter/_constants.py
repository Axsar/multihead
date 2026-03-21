"""BotVibes adapter constants and exceptions."""

_ACP_TIMEOUT = 30.0
_TASK_POLL_INTERVAL_S = 2.0
_MAX_WAIT_TIME_S = 300.0  # 5 minutes max wait for task completion
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE_S = 1.0
_RFQ_BID_WAIT_S = 60.0  # How long to wait for bids on an RFQ
_RFQ_BID_POLL_S = 5.0  # Poll interval for bids
_RETRYABLE_STATUS_CODES = {429, 502, 503, 504}


class PrivacyViolation(Exception):
    """Raised when data sensitivity blocks marketplace delegation."""
