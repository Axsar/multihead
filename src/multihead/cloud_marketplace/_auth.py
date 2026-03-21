"""Authentication, JWT handling, and token refresh for cloud marketplace."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import Any

import httpx

from ._constants import CLOUD_TIMEOUT, TOKEN_REFRESH_MARGIN_S, logger


class AuthMixin:
    """Mixin providing authentication and token refresh logic."""

    # These attributes are defined on the main class; declared here for typing.
    _cloud_url: str
    _cloud_api_key: str
    _cloud_agent_id: str
    _cloud_email: str
    _cloud_password: str
    _cloud_bridge: Any
    _running: bool
    on_activity: Any

    def _emit(self, event_type: str, message: str) -> None: ...

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._cloud_api_key}"}

    async def _cloud_request(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        *,
        retried: bool = False,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make an authenticated cloud API request with proactive token refresh.

        Before each request, checks if the JWT is expired or about to expire
        (within 30s) and refreshes proactively. Falls back to reactive 401
        retry if the proactive check misses.
        """
        # Proactive refresh: avoid the 401 round-trip entirely
        if not retried:
            exp = self._jwt_exp(self._cloud_api_key)
            if exp and exp - time.time() < 30:
                logger.info("Cloud JWT near-expiry — proactive refresh before %s %s", method, url)
                await self._refresh_token()

        if "headers" not in kwargs:
            kwargs["headers"] = self._auth_headers()
        resp = await client.request(method, url, **kwargs)
        if resp.status_code == 401 and not retried:
            logger.warning("Cloud API 401 on %s %s — refreshing token", method, url)
            await self._refresh_token()
            kwargs["headers"] = self._auth_headers()
            return await self._cloud_request(
                client, method, url, retried=True, **kwargs
            )
        return resp

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
            payload_b64 = parts[1]
            padding = 4 - len(payload_b64) % 4
            if padding != 4:
                payload_b64 += "=" * padding
            payload = json.loads(base64.urlsafe_b64decode(payload_b64))
            return float(payload.get("exp", 0))
        except Exception:
            return None

    async def _token_refresh_loop(self) -> None:
        """Monitor JWT expiry and refresh before it expires."""
        while self._running:
            try:
                exp = self._jwt_exp(self._cloud_api_key)
                if not exp:
                    # Opaque token — re-check periodically in case a refresh
                    # produces a proper JWT later
                    logger.debug("Cloud token has no exp claim — sleeping 5m before retry")
                    await asyncio.sleep(300)
                    continue

                remaining = exp - time.time()
                if remaining <= 0:
                    logger.warning("Cloud JWT expired — refreshing now")
                    await self._refresh_token()
                elif remaining <= TOKEN_REFRESH_MARGIN_S:
                    logger.info("Cloud JWT expires in %.0fs — refreshing", remaining)
                    await self._refresh_token()
                else:
                    sleep_for = remaining - TOKEN_REFRESH_MARGIN_S
                    logger.debug("Cloud JWT expires in %.0fs, refresh in %.0fs", remaining, sleep_for)
                    await asyncio.sleep(min(sleep_for, 300))
                    continue

            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning("Cloud token refresh check failed: %s", e)
            await asyncio.sleep(60)

    async def _refresh_token(self) -> None:
        """Call BotVibes POST /auth/refresh to get a new cloud JWT.

        Falls back to full re-login if refresh returns 401 (expired token).
        """
        if not self._cloud_url or not self._cloud_api_key:
            return

        try:
            async with httpx.AsyncClient(timeout=CLOUD_TIMEOUT) as client:
                resp = await client.post(
                    f"{self._cloud_url}/auth/refresh",
                    headers=self._auth_headers(),
                )
                if resp.status_code == 401:
                    logger.warning("Cloud token refresh got 401 — attempting re-login")
                    await self._re_login()
                    return
                resp.raise_for_status()
                data = resp.json()
                new_token = data.get("token") or data.get("access_token")
                if new_token:
                    self._cloud_api_key = new_token
                    self._propagate_token(new_token)
                    logger.info("Cloud JWT refreshed (new exp: %s)", self._jwt_exp(new_token))
                    self._emit("token", "JWT refreshed")
                else:
                    logger.warning("Cloud token refresh missing token field: %s", list(data.keys()))
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logger.warning("Cloud token refresh got 401 — attempting re-login")
                await self._re_login()
            else:
                logger.error("Cloud token refresh failed: %s", e)
        except Exception as e:
            logger.error("Cloud token refresh failed: %s", e)

    async def _re_login(self) -> None:
        """Re-authenticate with BotVibes using stored credentials.

        Called when JWT refresh fails (token already expired).
        Reads credentials from ACP_CLOUD_EMAIL / ACP_CLOUD_PASSWORD env vars
        or instance fields set by the service wrapper.
        """
        import os

        email = self._cloud_email or os.environ.get("ACP_CLOUD_EMAIL", "")
        password = self._cloud_password or os.environ.get("ACP_CLOUD_PASSWORD", "")

        if not email or not password:
            logger.error(
                "Cloud re-login failed: ACP_CLOUD_EMAIL / ACP_CLOUD_PASSWORD not set. "
                "Cannot re-authenticate — set these in .env to enable auto-login."
            )
            return

        try:
            async with httpx.AsyncClient(timeout=CLOUD_TIMEOUT) as client:
                resp = await client.post(
                    f"{self._cloud_url}/auth/login",
                    json={"email": email, "password": password},
                )
                resp.raise_for_status()
                data = resp.json()
                new_token = data.get("access_token", "")
                if new_token:
                    self._cloud_api_key = new_token
                    self._propagate_token(new_token)
                    logger.info(
                        "Cloud re-login successful (agent=%s, exp=%s)",
                        data.get("agent_id", "?"),
                        self._jwt_exp(new_token),
                    )
                    self._emit("token", "Re-authenticated with BotVibes")
                else:
                    logger.error("Cloud re-login response missing access_token")
        except Exception as e:
            logger.error("Cloud re-login failed: %s", e)

    def _propagate_token(self, token: str) -> None:
        """Propagate a refreshed JWT to internal subsystems."""
        if self._cloud_bridge and hasattr(self._cloud_bridge, "_api_key"):
            self._cloud_bridge._api_key = token
