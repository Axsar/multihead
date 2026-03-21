"""HTTP client for communicating with peer MultiHead nodes."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from ..resilience import CircuitBreaker, CircuitBreakerOpen

logger = logging.getLogger(__name__)


class MeshClient:
    """Async HTTP client for peer mesh node REST API.

    Wraps the endpoints defined in mesh_routes.py:
      GET  /v1/health
      GET  /v1/capabilities
      POST /v1/tasks
      GET  /v1/node
    """

    def __init__(
        self,
        base_url: str,
        auth_token: str = "",
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.timeout = timeout
        self.max_retries = max_retries
        self.breaker = CircuitBreaker(failure_threshold=5, reset_timeout_s=60.0)
        self._headers: dict[str, str] = {}
        if auth_token:
            self._headers["Authorization"] = f"Bearer {auth_token}"

    async def health(self) -> dict[str, Any]:
        """GET /v1/health — public endpoint, no auth required."""
        return await self._get("/v1/health")

    async def capabilities(self, kind: str | None = None) -> list[dict[str, Any]]:
        """GET /v1/capabilities — list peer's capabilities."""
        params: dict[str, str] = {}
        if kind:
            params["kind"] = kind
        result = await self._get("/v1/capabilities", params=params)
        return result if isinstance(result, list) else []

    async def submit_task(
        self,
        capability_kind: str,
        prompt: str,
        *,
        model: str = "",
        task_id: str = "",
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """POST /v1/tasks — submit a task to the peer."""
        body = {
            "capability_kind": capability_kind,
            "prompt": prompt,
            "model": model,
            "task_id": task_id,
            "params": params or {},
        }
        return await self._post("/v1/tasks", json_body=body)

    async def node_info(self) -> dict[str, Any]:
        """GET /v1/node — get peer node information."""
        return await self._get("/v1/node")

    async def fetch_claims(
        self,
        since: str = "",
        scope_id: str = "",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """GET /v1/claims — fetch shared claims from peer."""
        params: dict[str, str] = {}
        if since:
            params["since"] = since
        if scope_id:
            params["scope_id"] = scope_id
        params["limit"] = str(limit)
        result = await self._get("/v1/claims", params=params)
        return result if isinstance(result, list) else []

    async def push_claims(self, claims: list[dict[str, Any]]) -> dict[str, Any]:
        """POST /v1/claims/import — push claims to peer."""
        return await self._post("/v1/claims/import", json_body={"claims": claims})

    async def _get(
        self, path: str, params: dict[str, str] | None = None
    ) -> Any:
        """Execute a GET request with retry and circuit breaker."""
        return await self._request("GET", path, params=params)

    async def _post(self, path: str, json_body: dict[str, Any]) -> Any:
        """Execute a POST request with retry and circuit breaker."""
        return await self._request("POST", path, json_body=json_body)

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        """Execute HTTP request with circuit breaker and retry."""
        if self.breaker.state == "open":
            raise CircuitBreakerOpen(
                f"Circuit open for peer {self.base_url}"
            )
        url = f"{self.base_url}{path}"

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    t0 = time.perf_counter()
                    response = await client.request(
                        method,
                        url,
                        headers=self._headers,
                        params=params,
                        json=json_body,
                    )
                    latency_ms = (time.perf_counter() - t0) * 1000

                    if response.status_code >= 500:
                        self.breaker.record_failure()
                        raise httpx.HTTPStatusError(
                            f"Server error {response.status_code}",
                            request=response.request,
                            response=response,
                        )

                    if response.status_code >= 400:
                        raise httpx.HTTPStatusError(
                            f"Client error {response.status_code}",
                            request=response.request,
                            response=response,
                        )

                    self.breaker.record_success()
                    logger.debug(
                        "Mesh %s %s → %d (%.0fms)",
                        method, path, response.status_code, latency_ms,
                    )
                    return response.json()

            except httpx.HTTPStatusError as e:
                last_error = e
                # Don't retry client errors (4xx)
                if e.response.status_code < 500:
                    raise
                if attempt < self.max_retries:
                    logger.warning(
                        "Mesh %s %s attempt %d failed: %s",
                        method, path, attempt + 1, e,
                    )
                    continue
                raise
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = e
                self.breaker.record_failure()
                if attempt < self.max_retries:
                    logger.warning(
                        "Mesh %s %s attempt %d failed: %s",
                        method, path, attempt + 1, e,
                    )
                    continue
                raise

        raise last_error  # type: ignore[misc]
