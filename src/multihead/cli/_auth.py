"""Auth commands."""

from __future__ import annotations

import json
import os

import click

from ._helpers import (
    console,
    datetime,
    timezone,
)
from ._core import main


@main.group()
def auth():
    """ACP/BotVibes authentication status."""


@auth.command("status")
@click.pass_context
def auth_status(ctx):
    """Check JWT validity, ACP connection, and agent identity.

    Example:

        multihead auth status
    """
    settings = ctx.obj["settings"]

    console.print("[bold]Auth Status[/bold]\n")

    # ACP environment
    acp_url = os.environ.get("ACP_URL", "")
    agent_id = os.environ.get("ACP_AGENT_ID", "multihead-agent")
    project_id = os.environ.get("ACP_PROJECT_ID", "")

    console.print(f"  Agent ID:     {agent_id}")
    console.print(f"  ACP URL:      {acp_url or '[dim]not set[/dim]'}")
    console.print(f"  Project ID:   {project_id or '[dim]not set[/dim]'}")

    # Check tokens
    session_key = os.environ.get("ACP_SESSION_KEY") or os.environ.get("ACP_API_KEY", "")
    claude_key = os.environ.get("ACP_CLAUDE_SESSION_KEY", "")

    def _check_jwt(label: str, token: str) -> None:
        if not token:
            console.print(f"  {label}: [dim]not set[/dim]")
            return
        try:
            import base64
            parts = token.split(".")
            if len(parts) != 3:
                console.print(f"  {label}: [yellow]invalid format (not a JWT)[/yellow]")
                return
            padded = parts[1] + "=" * (4 - len(parts[1]) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded))
            exp = payload.get("exp")
            sub = payload.get("sub", payload.get("agent_id", "?"))
            if exp:
                exp_dt = datetime.fromtimestamp(exp, tz=timezone.utc)
                now = datetime.now(timezone.utc)
                remaining = exp_dt - now
                if remaining.total_seconds() > 0:
                    hours = remaining.total_seconds() / 3600
                    console.print(
                        f"  {label}: [green]valid[/green] "
                        f"(sub={sub}, expires in {hours:.1f}h)"
                    )
                else:
                    console.print(
                        f"  {label}: [red]EXPIRED[/red] "
                        f"(sub={sub}, expired {abs(remaining.total_seconds()) / 3600:.1f}h ago)"
                    )
            else:
                console.print(f"  {label}: [green]valid (no expiry)[/green] (sub={sub})")
        except Exception as e:
            console.print(f"  {label}: [yellow]error decoding: {e}[/yellow]")

    _check_jwt("Session JWT", session_key)
    _check_jwt("Claude JWT ", claude_key)

    # Try ACP connectivity
    if acp_url:
        console.print()
        try:
            import httpx
            resp = httpx.get(f"{acp_url}/api/v1/health", timeout=5.0,
                             headers={"Authorization": f"Bearer {session_key}"} if session_key else {})
            if resp.status_code == 200:
                console.print(f"  ACP server:   [green]reachable[/green] ({resp.status_code})")
            else:
                console.print(f"  ACP server:   [yellow]{resp.status_code}[/yellow]")
        except Exception as e:
            console.print(f"  ACP server:   [red]unreachable[/red] ({type(e).__name__})")
