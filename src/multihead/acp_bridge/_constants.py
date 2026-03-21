"""ACP Bridge constants and shared configuration."""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger("multihead.acp_bridge")

# Heartbeat interval (ACP recommends 10-15s, TTL is 30s)
HEARTBEAT_INTERVAL_S = 15
# Task polling interval (fallback when WebSocket is unavailable)
TASK_POLL_INTERVAL_S = 5
ACP_TIMEOUT = 30.0
# Refresh JWT this many seconds before expiry
TOKEN_REFRESH_MARGIN_S = 300  # 5 minutes


def _find_env_path() -> Path:
    """Locate the project .env file (three levels up from this module).

    _constants.py is at src/multihead/acp_bridge/_constants.py,
    so parents[3] == project root (D:/Dev/Axar/multihead).
    """
    return Path(__file__).resolve().parents[3] / ".env"


def persist_env_key(key: str, value: str, env_path: Path | None = None) -> None:
    """Safely update a single key in the .env file without clobbering other values.

    If the key already exists, its line is replaced in-place.
    If it does not exist, the key=value pair is appended.
    If the file does not exist, it is created.

    Args:
        key: Environment variable name (e.g. ``ACP_API_KEY``).
        value: New value to write.
        env_path: Path to .env file; defaults to project root .env.
    """
    if env_path is None:
        env_path = _find_env_path()

    key_line = f"{key}={value}"
    pattern = re.compile(rf"^{re.escape(key)}\s*=.*$", re.MULTILINE)

    if env_path.exists():
        original = env_path.read_text(encoding="utf-8")
        if pattern.search(original):
            updated = pattern.sub(key_line, original)
        else:
            sep = "" if original.endswith("\n") else "\n"
            updated = original + sep + key_line + "\n"
        env_path.write_text(updated, encoding="utf-8")
    else:
        env_path.write_text(f"{key_line}\n", encoding="utf-8")

    logger.info("Persisted %s to %s", key, env_path)
