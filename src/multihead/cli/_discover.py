"""Capability discovery commands."""

from __future__ import annotations

import json
import os

import click

from ._helpers import (
    console,
    KnowledgeStore,
)
from ._core import main


@main.command("discover")
@click.option("--paths", "-p", multiple=True, help="Project paths to scan (repeatable).")
@click.option("--register", is_flag=True, help="Register discovered capabilities on marketplace.")
@click.option("--json-output", "json_out", is_flag=True, help="Output as JSON.")
@click.option("--min-confidence", default=0.5, help="Minimum confidence for marketplace-ready.")
@click.pass_context
def discover(ctx, paths, register, json_out, min_confidence):
    """Discover unique capabilities from local codebases.

    Scans project directories for trained models, production pipelines,
    and domain-specific tools. Identifies what's unique to THIS machine
    vs commodity capabilities that ship with every MultiHead install.

    \b
    Examples:
      multihead discover                     # Auto-discover from Claude sessions
      multihead discover -p /path/to/project1 -p /path/to/project2
      multihead discover --register          # Also register on marketplace
      multihead discover --json-output       # Machine-readable output
    """
    from ..codebase_scanner import CodebaseScanner

    settings = ctx.obj["settings"]
    ks = None
    try:
        ks = KnowledgeStore(settings.data_dir / "knowledge.db")
    except Exception:
        pass

    scanner = CodebaseScanner(knowledge_store=ks)

    # Determine project paths
    scan_paths = list(paths) if paths else []
    if not scan_paths:
        auto = scanner.auto_discover_projects()
        if auto:
            scan_paths = [str(p) for p in auto]
            if not json_out:
                console.print(
                    f"[dim]Auto-discovered {len(scan_paths)} project(s) "
                    f"from Claude sessions[/dim]"
                )
        else:
            console.print(
                "[yellow]No project paths provided and none auto-discovered.[/yellow]\n"
                "Use: multihead discover -p /path/to/project"
            )
            return

    if not json_out:
        console.print(f"Scanning {len(scan_paths)} project(s)...\n")

    results = scanner.scan_all(scan_paths)

    # Curate: auto-classify unique vs commodity
    from ..capability_curator import CapabilityCurator

    curator = CapabilityCurator(min_unique_score=min_confidence)
    curated = curator.curate_scan_results(results)

    if json_out:
        out = {
            "summary": curated.summary(),
            "projects": [
                {
                    "name": r.project_name,
                    "path": r.project_path,
                    "capabilities": len(r.capabilities),
                    "models": len(r.model_checkpoints),
                    "scan_seconds": r.scan_duration_s,
                }
                for r in results
            ],
            "unique": [
                {**cc.capability.to_listing(), "uniqueness_score": cc.uniqueness_score,
                 "tier": cc.tier.value, "reasons": cc.reasons}
                for cc in curated.unique
            ],
            "specialized": [
                {**cc.capability.to_listing(), "uniqueness_score": cc.uniqueness_score,
                 "tier": cc.tier.value, "reasons": cc.reasons}
                for cc in curated.specialized
            ],
            "marketplace_ready": [
                {**cc.capability.to_listing(), "suggested_price": cc.suggested_price}
                for cc in curated.listable
            ],
        }
        click.echo(json.dumps(out, indent=2))
        return

    # Rich output — curation report
    console.print(curator.summary_report(curated))

    listable = curated.listable
    if listable:
        console.print(
            f"\n[green]{len(listable)} capabilities ready for marketplace[/green]"
        )

    if register and listable:
        _register_discovered(ctx, [cc.capability for cc in listable])
    elif register and not listable:
        console.print("[yellow]No capabilities met the curation threshold.[/yellow]")


def _register_discovered(ctx, capabilities):
    """Register discovered capabilities on BotVibes marketplace."""
    cloud_url = os.environ.get("ACP_CLOUD_URL", "")
    email = os.environ.get("ACP_CLOUD_EMAIL", "")
    password = os.environ.get("ACP_CLOUD_PASSWORD", "")
    agent_id = os.environ.get("ACP_CLOUD_AGENT_ID", "multihead-cloud-agent")

    if not cloud_url or not email or not password:
        console.print(
            "[yellow]Marketplace registration requires ACP_CLOUD_URL, "
            "ACP_CLOUD_EMAIL, ACP_CLOUD_PASSWORD in .env[/yellow]"
        )
        return

    try:
        import httpx

        # Login
        resp = httpx.post(
            f"{cloud_url}/auth/login",
            json={"email": email, "password": password},
            timeout=15,
        )
        if resp.status_code != 200:
            console.print(f"[red]Login failed: {resp.status_code}[/red]")
            return

        token = resp.json().get("access_token") or resp.json().get("token")
        headers = {"Authorization": f"Bearer {token}"}

        # Get existing listings
        r = httpx.get(
            f"{cloud_url}/marketplace/agents/{agent_id}/listings",
            headers=headers,
            timeout=15,
        )
        existing_caps = set()
        if r.status_code == 200:
            existing_caps = {
                l.get("capability_id", "").lower() for l in r.json()
            }

        registered = 0
        skipped = 0
        for cap in capabilities:
            listing = cap.to_listing()
            cap_id = listing["capability_id"]
            if cap_id.lower() in existing_caps:
                skipped += 1
                continue

            r = httpx.post(
                f"{cloud_url}/marketplace/listings",
                headers=headers,
                json={
                    "agent_id": agent_id,
                    "capability_id": cap_id,
                    "name": listing["name"],
                    "description": listing["description"],
                    "pricing_model": listing["pricing_model"],
                    "unit_price": listing["unit_price"],
                    "quality_score": listing["quality_score"],
                },
                timeout=15,
            )
            if r.status_code in (200, 201):
                console.print(f"  [green]Registered[/green] {cap_id} (${listing['unit_price']:.2f})")
                registered += 1
            else:
                console.print(f"  [red]Failed[/red] {cap_id}: {r.status_code}")

        console.print(
            f"\n[green]{registered} new listings registered[/green], "
            f"{skipped} already existed"
        )

    except ImportError:
        console.print("[red]httpx not installed — cannot register listings[/red]")
    except Exception as e:
        console.print(f"[red]Registration failed: {e}[/red]")
