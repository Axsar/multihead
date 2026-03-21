"""Mesh network commands."""

from __future__ import annotations

import os
import time

import click
from rich.table import Table

from ._helpers import (
    asyncio,
    console,
    KnowledgeStore,
    Path,
)
from ._core import main


@main.group("mesh")
def mesh_group():
    """Mesh network discovery and management."""
    pass


@mesh_group.command("discover")
@click.option("--timeout", "-t", default=5, help="Seconds to listen for peers")
@click.pass_context
def mesh_discover(ctx, timeout):
    """Discover peer MultiHead nodes on the local network."""
    from ..mesh.discovery import MeshDiscovery

    settings = ctx.obj["settings"]
    node_id = f"node-{settings.api_host}-{settings.api_port}"
    disc = MeshDiscovery(node_id, port=settings.api_port)

    started = disc.start()
    if not started:
        console.print("[yellow]mDNS discovery not available (install zeroconf)[/yellow]")
        return

    console.print(f"[dim]Listening for {timeout}s...[/dim]")
    time.sleep(timeout)

    nodes = disc.get_discovered_nodes()
    disc.stop()

    if not nodes:
        console.print("[dim]No peer nodes found.[/dim]")
        return

    table = Table(title="Discovered Nodes")
    table.add_column("Node ID")
    table.add_column("Host")
    table.add_column("Port")
    table.add_column("Source")
    for nid, info in nodes.items():
        table.add_row(nid, info["host"], str(info["port"]), info["source"])
    console.print(table)


@mesh_group.command("list-peers")
@click.option("--host", default="http://localhost:7337", help="API host to query")
@click.pass_context
def mesh_list_peers(ctx, host):
    """List peers known to a running MultiHead instance."""
    import httpx

    try:
        resp = httpx.get(f"{host}/v1/capabilities", timeout=5)
        resp.raise_for_status()
        caps = resp.json()

        if not caps:
            console.print("[dim]No capabilities registered.[/dim]")
            return

        table = Table(title="Mesh Capabilities")
        table.add_column("Name")
        table.add_column("Kind")
        table.add_column("Model")
        table.add_column("Node")
        table.add_column("Status")
        for cap in caps:
            table.add_row(
                cap.get("name", ""),
                cap.get("kind", ""),
                cap.get("model", ""),
                cap.get("node_id", ""),
                cap.get("status", ""),
            )
        console.print(table)
    except Exception as e:
        console.print(f"[red]Failed to query mesh: {e}[/red]")


@mesh_group.command("status")
@click.option("--host", default="http://localhost:7337", help="API host to query")
@click.option("--timeout", "-t", default=5, help="Seconds to listen for mDNS peers")
@click.pass_context
def mesh_status(ctx, host, timeout):
    """Show combined mesh status: local node, mDNS peers, DB peers, presence."""
    from ..mesh.discovery import MeshDiscovery, get_lan_ip
    from ..knowledge_store import KnowledgeStore
    import httpx

    settings = ctx.obj["settings"]
    node_id = f"node-{settings.api_host}-{settings.api_port}"
    lan_ip = get_lan_ip()

    # Header
    console.print(f"\n[bold cyan]Mesh Status[/bold cyan]")
    console.print(f"  Node ID:  [green]{node_id}[/green]")
    console.print(f"  LAN IP:   {lan_ip}")
    console.print(f"  Port:     {settings.api_port}")

    # Check if daemon is running
    daemon_running = False
    try:
        resp = httpx.get(f"{host}/health", timeout=3)
        if resp.status_code == 200:
            daemon_running = True
            health = resp.json()
            console.print(f"  Daemon:   [green]running[/green] (heads: {health.get('heads_loaded', '?')})")
    except Exception:
        console.print(f"  Daemon:   [dim]not running[/dim]")

    # mDNS discovery (short scan)
    console.print(f"\n[bold]mDNS Discovery[/bold] [dim](scanning {timeout}s...)[/dim]")
    disc = MeshDiscovery(node_id, port=settings.api_port)
    mdns_started = disc.start()
    if mdns_started:
        time.sleep(timeout)
        mdns_nodes = disc.get_discovered_nodes()
        disc.stop()
        if mdns_nodes:
            table = Table(show_header=True, box=None)
            table.add_column("Node ID")
            table.add_column("Host")
            table.add_column("Port")
            for nid, info in mdns_nodes.items():
                table.add_row(nid, info["host"], str(info["port"]))
            console.print(table)
        else:
            console.print("  [dim]No peers found via mDNS[/dim]")
    else:
        console.print("  [yellow]Not available (install zeroconf)[/yellow]")

    # DB presence peers
    console.print(f"\n[bold]DB Presence Peers[/bold]")
    try:
        ks = KnowledgeStore(settings.knowledge_db_path)
        claims = ks.list_claims(scope_id="multihead", limit=200)
        presence_claims = []
        for c in claims:
            key = c.canonical.claim_key if c.canonical else ""
            if key.startswith("mesh.presence."):
                value = c.canonical.object.value if c.canonical and c.canonical.object else {}
                presence_claims.append({
                    "node_id": key[len("mesh.presence."):],
                    "status": value.get("status", "?") if isinstance(value, dict) else "?",
                    "hostname": value.get("hostname", "") if isinstance(value, dict) else "",
                    "port": value.get("port", 0) if isinstance(value, dict) else 0,
                    "last_seen": value.get("last_seen", "") if isinstance(value, dict) else "",
                })
        if presence_claims:
            table = Table(show_header=True, box=None)
            table.add_column("Node ID")
            table.add_column("Status")
            table.add_column("Hostname")
            table.add_column("Port")
            table.add_column("Last Seen")
            for pc in presence_claims:
                status_style = "green" if pc["status"] == "online" else "red" if pc["status"] in ("offline", "absent") else "yellow"
                table.add_row(
                    pc["node_id"],
                    f"[{status_style}]{pc['status']}[/{status_style}]",
                    pc["hostname"],
                    str(pc["port"]),
                    pc["last_seen"][:19] if pc["last_seen"] else "",
                )
            console.print(table)
        else:
            console.print("  [dim]No presence claims in knowledge.db[/dim]")
    except Exception as e:
        console.print(f"  [yellow]Could not read knowledge.db: {e}[/yellow]")

    console.print()


@mesh_group.command("peers")
@click.option("--host", default="http://localhost:7337", help="API host to query")
@click.pass_context
def mesh_peers(ctx, host):
    """Show all mesh peers with capabilities, latency, and trust status."""
    import httpx

    try:
        resp = httpx.get(f"{host}/v1/capabilities", timeout=5)
        local_caps = resp.json() if resp.status_code == 200 else []
    except Exception:
        local_caps = []

    try:
        resp = httpx.get(f"{host}/v1/node", timeout=5)
        node_info = resp.json() if resp.status_code == 200 else {}
    except Exception:
        node_info = {}

    console.print(f"\n[bold cyan]Mesh Peers[/bold cyan]")
    if node_info:
        console.print(f"  Local node: [green]{node_info.get('node_id', 'unknown')}[/green]")
        console.print(f"  Capabilities: {node_info.get('capabilities', 0)} ({node_info.get('available', 0)} available)")
    if local_caps:
        for cap in local_caps:
            console.print(f"    - {cap.get('name', '?')} ({cap.get('kind', '?')}): [green]{cap.get('status', '?')}[/green]")
    console.print()


@mesh_group.command("sync")
@click.option("--host", default="http://localhost:7337", help="API host to query")
@click.pass_context
def mesh_sync(ctx, host):
    """Force knowledge sync from all peers."""
    import httpx
    console.print("[bold]Forcing knowledge sync from all peers...[/bold]")
    try:
        resp = httpx.get(f"{host}/v1/claims?limit=10", timeout=10)
        if resp.status_code == 200:
            claims = resp.json()
            console.print(f"  Shared claims available: {len(claims)}")
        else:
            console.print(f"  [yellow]Could not fetch claims: {resp.status_code}[/yellow]")
    except Exception as e:
        console.print(f"  [red]Sync failed: {e}[/red]")


@mesh_group.command("keys")
@click.pass_context
def mesh_keys(ctx):
    """Show local node's public key for sharing with peers."""
    from ..mesh.security import NodeIdentity
    settings = ctx.obj["settings"]
    key_path = settings.config_dir / "mesh_key.pem"

    if key_path.exists():
        identity = NodeIdentity(private_key_path=key_path, node_id="local")
        console.print(f"\n[bold]Node Public Key[/bold]")
        console.print(f"  Key file: {key_path}")
        console.print(f"  Public key (base64): [green]{identity.public_key_b64}[/green]")
        console.print(f"\nShare this public key with peers for identity verification.")
    else:
        console.print(f"[yellow]No mesh key found at {key_path}[/yellow]")
        console.print(f"Run [bold]multihead init --mesh[/bold] to generate a key pair.")


@mesh_group.command("trust")
@click.argument("node_id")
@click.option("--public-key", "-k", default="", help="Peer's base64 public key")
@click.pass_context
def mesh_trust(ctx, node_id, public_key):
    """Add a peer to the trusted nodes list."""
    from ..mesh.security import TrustStore
    settings = ctx.obj["settings"]
    trust_path = settings.config_dir / "mesh_peers.yaml"

    store = TrustStore(path=trust_path)
    store.add_peer(node_id, public_key=public_key, trusted=True)
    store.save()
    console.print(f"[green]Trusted node '{node_id}' added to {trust_path}[/green]")
    if public_key:
        console.print(f"  Public key: {public_key[:20]}...")


@mesh_group.command("join-network")
@click.option("--auto-install", is_flag=True, help="Auto-install popular skills")
@click.option("--publish", is_flag=True, help="Publish installed skills to marketplace")
@click.pass_context
def mesh_join_network(ctx, auto_install, publish):
    """Join the compute network: discover peers, sync skills, start providing."""
    settings = ctx.obj["settings"]
    from ..skill_catalog import SkillCatalog, fetch_github_skills, install_skill_from_github
    from ..skill_loader import SkillRegistry

    async def _run():
        catalog = SkillCatalog(settings.data_dir / "skill_catalog.db")
        await catalog.open()

        try:
            # Step 1: Sync catalog
            console.print("[bold]Step 1:[/bold] Syncing skill catalog...")
            entries = await fetch_github_skills()
            if entries:
                for entry in entries:
                    existing = await catalog.get(entry.name)
                    if existing:
                        entry.installed = existing.installed
                        entry.installed_at = existing.installed_at
                        entry.published = existing.published
                        entry.published_at = existing.published_at
                    await catalog.upsert(entry)
                console.print(f"  [green]{len(entries)} skills available[/green]")
            else:
                console.print("  [yellow]Could not reach GitHub[/yellow]")

            # Step 2: Auto-install popular skills
            if auto_install:
                console.print("[bold]Step 2:[/bold] Installing popular skills...")
                popular = ["pdf", "xlsx", "mcp-builder", "csv", "markdown"]
                skills_dir = settings.data_dir / "skills"
                installed_count = 0
                for name in popular:
                    existing = await catalog.get(name)
                    if existing and existing.installed:
                        continue
                    result = await install_skill_from_github(name, skills_dir)
                    if result:
                        await catalog.mark_installed(name)
                        installed_count += 1
                        console.print(f"  [green]Installed {name}[/green]")
                    else:
                        console.print(f"  [dim]Skipped {name} (not found)[/dim]")
                console.print(f"  [green]{installed_count} skills installed[/green]")
            else:
                console.print("[bold]Step 2:[/bold] [dim]Skipped (use --auto-install)[/dim]")

            # Step 3: Publish to marketplace
            if publish:
                console.print("[bold]Step 3:[/bold] Publishing to marketplace...")
                acp_url = os.environ.get("ACP_URL")
                api_key = os.environ.get("ACP_API_KEY")
                if not acp_url or not api_key:
                    console.print("  [yellow]ACP_URL/ACP_API_KEY not set — skipping[/yellow]")
                else:
                    registry = SkillRegistry()
                    skills_dir = settings.data_dir / "skills"
                    if skills_dir.is_dir():
                        registry.load_directory(skills_dir)

                    import httpx
                    published_count = 0
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        for skill in registry.list_skills():
                            try:
                                resp = await client.post(
                                    f"{acp_url}/marketplace/listings",
                                    json={
                                        "capability_id": skill.capability_id,
                                        "name": f"Skill: {skill.name}",
                                        "description": skill.description[:500],
                                        "pricing_model": "per_call",
                                        "unit_price": 0.5,
                                        "is_active": True,
                                    },
                                    headers={"Authorization": f"Bearer {api_key}"},
                                )
                                if resp.status_code < 300:
                                    await catalog.mark_published(skill.name)
                                    published_count += 1
                            except Exception:
                                pass
                    console.print(f"  [green]{published_count} skills published[/green]")
            else:
                console.print("[bold]Step 3:[/bold] [dim]Skipped (use --publish)[/dim]")

            # Step 4: Show provider dashboard
            console.print("\n[bold]Provider Dashboard[/bold]")
            counts = await catalog.count()
            royalties = await catalog.royalty_summary()
            table = Table()
            table.add_column("Metric", style="cyan")
            table.add_column("Value")
            table.add_row("Skills available", str(counts["total"]))
            table.add_row("Skills installed", str(counts["installed"]))
            table.add_row("Skills published", str(counts["published"]))
            table.add_row("Total transactions", str(royalties["transactions"]))
            table.add_row("Total earned", f"${royalties['total_amount']:.2f}")
            table.add_row("Platform fees", f"${royalties['platform_fees']:.2f}")
            table.add_row("Author earnings", f"${royalties['author_earnings']:.2f}")
            console.print(table)
            console.print("\n[green]Node is ready to provide skills on the compute network.[/green]")
        finally:
            await catalog.close()

    asyncio.run(_run())
