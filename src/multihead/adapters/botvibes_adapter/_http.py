"""HTTP helpers with retry and token refresh for BotVibes adapter."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

from ._constants import _MAX_RETRIES, _RETRYABLE_STATUS_CODES, _RETRY_BACKOFF_BASE_S

logger = logging.getLogger(__name__)


class HttpMixin:
    """Mixin providing HTTP request helpers with retry and token refresh.

    Expects the consuming class to have:
    - ``acp_url``: str
    - ``acp_token``: str
    - ``manifest``: HeadManifest
    - ``_login_email``: str
    - ``_login_password``: str
    """

    acp_url: str
    acp_token: str
    _login_email: str
    _login_password: str

    def _auth_headers(self) -> dict[str, str]:
        """Build authentication headers for ACP requests."""
        return {
            "Authorization": f"Bearer {self.acp_token}",
            "Content-Type": "application/json",
        }

    async def _request_with_retry(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        *,
        max_retries: int = _MAX_RETRIES,
        **kwargs: Any,
    ) -> httpx.Response:
        """HTTP request with exponential backoff retry on transient errors."""
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                resp = await client.request(method, url, **kwargs)

                # Token expired — try refresh once
                if resp.status_code == 401 and attempt == 0:
                    if await self._refresh_token(client):
                        kwargs.setdefault("headers", {})
                        kwargs["headers"] = self._auth_headers()
                        continue

                if resp.status_code in _RETRYABLE_STATUS_CODES and attempt < max_retries:
                    wait = _RETRY_BACKOFF_BASE_S * (2 ** attempt)
                    logger.warning(
                        "BotVibes %s %s returned %d, retry %d/%d in %.1fs",
                        method, url, resp.status_code, attempt + 1, max_retries, wait,
                    )
                    await asyncio.sleep(wait)
                    continue

                return resp
            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                last_exc = e
                if attempt < max_retries:
                    wait = _RETRY_BACKOFF_BASE_S * (2 ** attempt)
                    logger.warning(
                        "BotVibes %s %s failed (%s), retry %d/%d in %.1fs",
                        method, url, e, attempt + 1, max_retries, wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise

        # Should not reach here, but just in case
        if last_exc:
            raise last_exc
        raise RuntimeError("Exhausted retries")  # pragma: no cover

    async def _refresh_token(self, client: httpx.AsyncClient) -> bool:
        """Try to refresh the JWT token via /auth/refresh or re-login."""
        # Try refresh endpoint first
        try:
            resp = await client.post(
                f"{self.acp_url}/auth/refresh",
                headers=self._auth_headers(),
            )
            if resp.status_code == 200:
                data = resp.json()
                new_token = data.get("access_token", data.get("token", ""))
                if new_token:
                    self.acp_token = new_token
                    logger.info("BotVibes adapter %s: token refreshed", self.manifest.head_id)  # type: ignore[attr-defined]
                    return True
        except Exception:
            pass

        # Fallback: re-login
        email = self._login_email or os.environ.get("ACP_CLOUD_EMAIL", "")
        password = self._login_password or os.environ.get("ACP_CLOUD_PASSWORD", "")
        if email and password:
            try:
                resp = await client.post(
                    f"{self.acp_url}/auth/login",
                    json={"email": email, "password": password},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    new_token = data.get("access_token", data.get("token", ""))
                    if new_token:
                        self.acp_token = new_token
                        logger.info("BotVibes adapter %s: re-login successful", self.manifest.head_id)  # type: ignore[attr-defined]
                        return True
            except Exception:
                pass

        logger.warning("BotVibes adapter %s: token refresh failed", self.manifest.head_id)  # type: ignore[attr-defined]
        return False
