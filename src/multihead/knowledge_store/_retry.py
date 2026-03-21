"""SQLite retry decorator with exponential backoff."""

from __future__ import annotations

import functools
import random
import sqlite3
import time

_RETRY_MAX = 5
_RETRY_BASE_MS = 200
_RETRY_MAX_MS = 2000
_BUSY_TIMEOUT_MS = 10000
_WAL_CHECKPOINT_INTERVAL = 1000


def _sqlite_retry(fn):
    """Retry on SQLite BUSY/LOCKED with exponential backoff (max 5 retries, 50-500ms).

    After a successful write, increments the store's write counter and triggers
    a passive WAL checkpoint every _WAL_CHECKPOINT_INTERVAL writes to prevent
    WAL file bloat in shared-DB multi-session scenarios.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # Import here to avoid circular dependency at module level
        from ._store import KnowledgeStore

        for attempt in range(_RETRY_MAX + 1):
            try:
                result = fn(*args, **kwargs)
                # Trigger periodic WAL checkpoint when called on a KnowledgeStore instance
                if args and isinstance(args[0], KnowledgeStore):
                    args[0]._maybe_checkpoint()
                return result
            except sqlite3.OperationalError as exc:
                msg = str(exc).lower()
                if attempt < _RETRY_MAX and ("locked" in msg or "busy" in msg):
                    delay_ms = min(_RETRY_BASE_MS * (2 ** attempt), _RETRY_MAX_MS)
                    jitter_ms = random.randint(0, delay_ms // 2)
                    time.sleep((delay_ms + jitter_ms) / 1000.0)
                    continue
                raise
    return wrapper
