"""Web access tools for the Agentic Core's tool registry.

Gives the local LLM the ability to fetch URLs and search the web.
Both tools are registered in ToolRegistry and toggleable via RuntimeConfig.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from html.parser import HTMLParser
from io import StringIO
from typing import Any
from urllib.parse import urlparse

import httpx

from .tool_registry import ToolRegistry, ToolResult, ToolSpec

logger = logging.getLogger(__name__)

_FETCH_TIMEOUT = 30.0
_SEARCH_TIMEOUT = 15.0
_MAX_RESPONSE_BYTES = 5 * 1024 * 1024  # 5 MB
_ALLOWED_SCHEMES = {"http", "https"}

# Private/internal IP ranges that must be blocked
_PRIVATE_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def _is_safe_url(url: str) -> tuple[bool, str]:
    """Validate URL for SSRF protection.

    Returns (is_safe, error_message).
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "Invalid URL"

    # Scheme check
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        return False, f"Blocked scheme: {parsed.scheme}:// (only http/https allowed)"

    hostname = parsed.hostname
    if not hostname:
        return False, "No hostname in URL"

    # Resolve hostname to IP and check against private ranges
    try:
        infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for family, _, _, _, sockaddr in infos:
            ip = ipaddress.ip_address(sockaddr[0])
            for network in _PRIVATE_NETWORKS:
                if ip in network:
                    return False, f"Blocked: {hostname} resolves to private IP {ip}"
    except socket.gaierror:
        return False, f"Cannot resolve hostname: {hostname}"

    return True, ""


# ---------------------------------------------------------------------------
# HTML text extraction (stdlib only, no external deps)
# ---------------------------------------------------------------------------

class _HTMLTextExtractor(HTMLParser):
    """Strip HTML tags and return plain text."""

    def __init__(self) -> None:
        super().__init__()
        self._result = StringIO()
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip = False
        if tag in ("p", "br", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"):
            self._result.write("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._result.write(data)

    def get_text(self) -> str:
        return self._result.getvalue().strip()


def _extract_text(html: str) -> str:
    """Extract readable text from HTML."""
    extractor = _HTMLTextExtractor()
    extractor.feed(html)
    return extractor.get_text()


# ---------------------------------------------------------------------------
# Tool specs
# ---------------------------------------------------------------------------

WEB_FETCH_SPEC = ToolSpec(
    name="web.fetch",
    description="Fetch a URL and return its text content (HTML tags stripped)",
    params_schema={
        "url": {"type": "string", "required": True},
        "max_length": {"type": "integer", "default": 20000},
    },
    requires_approval=False,
)

WEB_SEARCH_SPEC = ToolSpec(
    name="web.search",
    description="Search the web and return results with titles, URLs, and snippets",
    params_schema={
        "query": {"type": "string", "required": True},
        "max_results": {"type": "integer", "default": 5},
    },
    requires_approval=False,
)


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

async def _tool_web_fetch(params: dict[str, Any]) -> ToolResult:
    """Fetch a URL and return text content."""
    url = params.get("url", "")
    max_length = params.get("max_length", 20000)

    if not url:
        return ToolResult(tool="web.fetch", success=False, error="URL is required")

    # SSRF protection
    safe, reason = _is_safe_url(url)
    if not safe:
        return ToolResult(tool="web.fetch", success=False, error=reason)

    try:
        async with httpx.AsyncClient(
            timeout=_FETCH_TIMEOUT,
            follow_redirects=True,
            max_redirects=5,
            headers={"User-Agent": "MultiHead/1.0"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()

            # Check response size
            content_length = resp.headers.get("content-length")
            if content_length and int(content_length) > _MAX_RESPONSE_BYTES:
                return ToolResult(
                    tool="web.fetch", success=False,
                    error=f"Response too large: {int(content_length)} bytes (max {_MAX_RESPONSE_BYTES})",
                )

        content_type = resp.headers.get("content-type", "")
        body = resp.text

        # Strip HTML if it looks like HTML
        if "html" in content_type or body.lstrip().startswith("<!") or body.lstrip().startswith("<html"):
            text = _extract_text(body)
        else:
            text = body

        # Truncate
        if len(text) > max_length:
            text = text[:max_length] + f"\n\n[Truncated at {max_length} characters]"

        return ToolResult(tool="web.fetch", success=True, output=text)

    except httpx.HTTPStatusError as e:
        return ToolResult(tool="web.fetch", success=False, error=f"HTTP {e.response.status_code}: {url}")
    except Exception as e:
        return ToolResult(tool="web.fetch", success=False, error=f"Fetch failed: {e}")


async def _tool_web_search(params: dict[str, Any]) -> ToolResult:
    """Search the web using DuckDuckGo."""
    query = params.get("query", "")
    max_results = params.get("max_results", 5)

    if not query:
        return ToolResult(tool="web.search", success=False, error="Query is required")

    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        formatted = []
        for r in results:
            formatted.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            })

        return ToolResult(
            tool="web.search",
            success=True,
            output=formatted,
        )

    except ImportError:
        return ToolResult(
            tool="web.search",
            success=False,
            error="duckduckgo-search package not installed. Run: pip install duckduckgo-search",
        )
    except Exception as e:
        return ToolResult(tool="web.search", success=False, error=f"Search failed: {e}")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_web_tools(registry: ToolRegistry) -> None:
    """Register web tools into an existing ToolRegistry."""
    registry.register(WEB_FETCH_SPEC, _tool_web_fetch)
    registry.register(WEB_SEARCH_SPEC, _tool_web_search)
