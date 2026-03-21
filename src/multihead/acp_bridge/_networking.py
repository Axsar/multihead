"""ACP networking: heartbeat, WebSocket doorbell, and token refresh.

Mixin class providing background networking loops for the ACPBridge.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any

import httpx

from ._constants import (
    ACP_TIMEOUT,
    HEARTBEAT_INTERVAL_S,
    TOKEN_REFRESH_MARGIN_S,
    logger,
    persist_env_key,
)


class NetworkingMixin:
    """Heartbeat, WebSocket doorbell, and token refresh methods."""

    # These attributes are provided by ACPBridge.__init__
    heads: Any
    settings: Any
    acp_url: str | None
    _api_key: str | None
    _agent_id: str
    _connected: bool
    _poll_now: asyncio.Event
    _ws_task: asyncio.Task | None
    _auth_headers: Any  # method

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """Send heartbeats for both agents at regular intervals."""
        while self._connected:
            try:
                await self._send_heartbeats()
            except Exception as e:
                logger.warning("ACP heartbeat failed: %s", e)
            await asyncio.sleep(HEARTBEAT_INTERVAL_S)

    async def _send_heartbeats(self) -> None:
        """Send heartbeat for MultiHead agent."""
        active = self.heads.active_head
        async with httpx.AsyncClient(timeout=ACP_TIMEOUT) as client:
            await client.post(
                f"{self.acp_url}/agents/{self._agent_id}/heartbeat",
                headers=self._auth_headers(),
                json={
                    "status": "busy" if active else "idle",
                    "queue_depth": 0,
                    "active_tasks": 1 if active else 0,
                    "metadata": {
                        "active_head": active,
                        "endpoint": f"http://{self.settings.api_host}:{self.settings.api_port}",
                    },
                },
            )

    # ------------------------------------------------------------------
    # WebSocket doorbell (instant task notifications)
    # ------------------------------------------------------------------

    def _ws_url(self) -> str:
        """Build WebSocket URL from ACP HTTP URL."""
        # Convert http(s)://host:port/api/v1 -> ws(s)://host:port/ws/agents/{id}
        # WebSocket router is mounted at root, not under /api/v1
        url = self.acp_url or ""
        url = url.replace("https://", "wss://").replace("http://", "ws://")
        # Strip /api/v1 suffix — WS endpoint is at ws://host:port/ws/agents/{id}
        for suffix in ("/api/v1", "/api/v1/", "/api"):
            if url.endswith(suffix):
                url = url[: -len(suffix)]
                break
        return f"{url}/ws/agents/{self._agent_id}?token={self._api_key}"

    async def _ws_doorbell_loop(self) -> None:
        """Connect to BotVibes WebSocket for instant task notifications.

        On receiving a message, sets _poll_now to trigger an immediate poll.
        Reconnects with exponential backoff on failure. Falls back to
        polling-only if WebSocket is unavailable.
        """
        backoff = 1.0
        max_backoff = 60.0

        while self._connected:
            try:
                import websockets
                ws_url = self._ws_url()
                logger.info("WebSocket doorbell connecting: %s", ws_url.split("?")[0])

                async with websockets.connect(
                    ws_url,
                    ping_interval=30,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    backoff = 1.0  # Reset on successful connection
                    logger.info("WebSocket doorbell connected")

                    async for message in ws:
                        try:
                            data = json.loads(message)
                            event_type = data.get("type", "unknown")
                            logger.info("WebSocket event: %s", event_type)

                            if event_type in ("task.created", "task.available", "notify"):
                                self._poll_now.set()
                        except json.JSONDecodeError:
                            logger.debug("WebSocket non-JSON message: %s", message[:100])
                            self._poll_now.set()  # Trigger poll on any message

            except ImportError:
                logger.warning("websockets package not installed — doorbell disabled")
                return
            except Exception as e:
                if not self._connected:
                    return
                err_msg = str(e)
                if "no close frame" in err_msg or "keepalive" in err_msg:
                    logger.debug("WebSocket reconnecting: %s", err_msg)
                else:
                    logger.warning("WebSocket doorbell error (retry in %.0fs): %s", backoff, e)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    # ------------------------------------------------------------------
    # Token refresh (auto-renew JWT before expiry)
    # ------------------------------------------------------------------

    @staticmethod
    def _jwt_exp(token: str) -> float | None:
        """Extract expiry timestamp from a JWT without verification."""
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None
            # Decode payload (base64url -> JSON)
            payload_b64 = parts[1]
            # Add padding
            padding = 4 - len(payload_b64) % 4
            if padding != 4:
                payload_b64 += "=" * padding
            payload = json.loads(base64.urlsafe_b64decode(payload_b64))
            return float(payload.get("exp", 0))
        except Exception:
            return None

    async def _token_refresh_loop(self) -> None:
        """Monitor JWT expiry and refresh before it expires."""
        while self._connected:
            try:
                exp = self._jwt_exp(self._api_key or "")
                if not exp:
                    logger.debug("Token has no exp claim — refresh disabled")
                    return

                remaining = exp - time.time()
                if remaining <= 0:
                    logger.warning("JWT already expired — attempting refresh now")
                    await self._refresh_token()
                elif remaining <= TOKEN_REFRESH_MARGIN_S:
                    logger.info("JWT expires in %.0fs — refreshing", remaining)
                    await self._refresh_token()
                else:
                    # Sleep until margin window
                    sleep_for = remaining - TOKEN_REFRESH_MARGIN_S
                    logger.debug("JWT expires in %.0fs, refresh in %.0fs", remaining, sleep_for)
                    await asyncio.sleep(min(sleep_for, 300))  # Check at least every 5 min
                    continue

            except Exception as e:
                logger.warning("Token refresh check failed: %s", e)
            await asyncio.sleep(60)

    async def _refresh_token(self) -> None:
        """Call BotVibes POST /auth/refresh to get a new JWT."""
        if not self.acp_url or not self._api_key:
            return

        try:
            async with httpx.AsyncClient(timeout=ACP_TIMEOUT) as client:
                resp = await client.post(
                    f"{self.acp_url}/auth/refresh",
                    headers=self._auth_headers(),
                )
                resp.raise_for_status()
                data = resp.json()
                new_token = data.get("token") or data.get("access_token")
                if new_token:
                    self._api_key = new_token
                    # Persist to .env so the token survives restarts
                    import os
                    os.environ["ACP_API_KEY"] = new_token
                    try:
                        persist_env_key("ACP_API_KEY", new_token)
                    except Exception as persist_err:
                        logger.warning("Failed to persist refreshed token to .env: %s", persist_err)
                    logger.info("JWT refreshed successfully (new exp: %s)",
                                self._jwt_exp(new_token))
                    # Cancel WebSocket so it reconnects with the new token
                    if self._ws_task and not self._ws_task.done():
                        self._ws_task.cancel()
                        self._ws_task = asyncio.create_task(self._ws_doorbell_loop())
                else:
                    logger.warning("Token refresh response missing token field: %s",
                                   list(data.keys()))
        except Exception as e:
            logger.error("Token refresh failed: %s", e)
