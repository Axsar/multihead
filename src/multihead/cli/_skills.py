"""Skills catalog commands."""

from __future__ import annotations

import os

import click

from ._helpers import (
    asyncio,
    console,
)
from ._core import main


@main.group()
def skills():
    """Skill catalog management: browse, install, publish."""


@skills.command("list")
@click.option("--installed", is_flag=True, help="Only show installed skills")
@click.option("--category", default=None, help="Filter by category")
@click.pass_context
def skills_list(ctx, installed, category):
    """List skills in the catalog."""
    settings = ctx.obj["settings"]
    from ..skill_catalog import SkillCatalog

    async def _run():
        from rich.table import Table

        catalog = SkillCatalog(settings.data_dir / "skill_catalog.db")
        await catalog.open()
        try:
            entries = await catalog.list_all(
                installed_only=installed, category=category, limit=200,
            )
            if not entries:
                console.print("[dim]No skills in catalog. Run 'multihead skills sync' first.[/dim]")
                return

            table = Table(title="Skill Catalog")
            table.add_column("Name", style="cyan")
            table.add_column("Category")
            table.add_column("Status")
            table.add_column("Author")
            table.add_column("License")

            for e in entries:
                status_parts = []
                if e.installed:
                    status_parts.append("[green]installed[/green]")
                if e.published:
                    status_parts.append("[blue]published[/blue]")
                if not status_parts:
                    status_parts.append("[dim]available[/dim]")
                table.add_row(
                    e.name, e.category, " ".join(status_parts),
                    e.author or "—", e.license_type,
                )
            console.print(table)
        finally:
            await catalog.close()

    asyncio.run(_run())


@skills.command("sync")
@click.pass_context
def skills_sync(ctx):
    """Sync catalog with GitHub (fetch available skills)."""
    settings = ctx.obj["settings"]
    from ..skill_catalog import SkillCatalog, fetch_github_skills

    async def _run():
        catalog = SkillCatalog(settings.data_dir / "skill_catalog.db")
        await catalog.open()
        try:
            console.print("[dim]Fetching skills from GitHub...[/dim]")
            entries = await fetch_github_skills()
            if not entries:
                console.print("[red]Failed to fetch skills from GitHub[/red]")
                return
            for entry in entries:
                existing = await catalog.get(entry.name)
                if existing:
                    entry.installed = existing.installed
                    entry.installed_at = existing.installed_at
                    entry.published = existing.published
                    entry.published_at = existing.published_at
                await catalog.upsert(entry)
            console.print(f"[green]Synced {len(entries)} skills from GitHub[/green]")
        finally:
            await catalog.close()

    asyncio.run(_run())


@skills.command("install")
@click.argument("skill_name")
@click.pass_context
def skills_install(ctx, skill_name):
    """Install a skill from GitHub."""
    settings = ctx.obj["settings"]
    from ..skill_catalog import SkillCatalog, install_skill_from_github

    async def _run():
        catalog = SkillCatalog(settings.data_dir / "skill_catalog.db")
        await catalog.open()
        try:
            skills_dir = settings.data_dir / "skills"
            console.print(f"[dim]Installing {skill_name}...[/dim]")
            result = await install_skill_from_github(skill_name, skills_dir)
            if result:
                await catalog.mark_installed(skill_name)
                console.print(f"[green]Installed {skill_name} to {result}[/green]")
            else:
                console.print(f"[red]Failed to install {skill_name}[/red]")
        finally:
            await catalog.close()

    asyncio.run(_run())


@skills.command("publish")
@click.option("--all", "publish_all", is_flag=True, help="Publish all installed skills")
@click.option("--name", default=None, help="Publish a specific skill")
@click.pass_context
def skills_publish(ctx, publish_all, name):
    """Publish installed skills to BotVibes marketplace."""
    if not publish_all and not name:
        console.print("[red]Specify --all or --name <skill>[/red]")
        return

    import httpx

    settings = ctx.obj["settings"]
    from ..skill_catalog import SkillCatalog
    from ..skill_loader import SkillRegistry

    async def _run():
        catalog = SkillCatalog(settings.data_dir / "skill_catalog.db")
        await catalog.open()
        try:
            # Load skill registry
            registry = SkillRegistry()
            skills_dir = settings.data_dir / "skills"
            project_skills = settings.config_dir.parent / "skills"
            for sd in [skills_dir, project_skills]:
                if sd.is_dir():
                    registry.load_directory(sd)

            to_publish = registry.list_skills() if publish_all else (
                [registry.get(name)] if name else []
            )
            to_publish = [s for s in to_publish if s is not None]

            if not to_publish:
                console.print("[dim]No skills to publish[/dim]")
                return

            acp_url = os.environ.get("ACP_URL")
            api_key = os.environ.get("ACP_API_KEY")
            if not acp_url or not api_key:
                console.print("[red]ACP_URL and ACP_API_KEY must be set in .env[/red]")
                return

            published = []
            async with httpx.AsyncClient(timeout=30.0) as client:
                for skill in to_publish:
                    try:
                        listing = {
                            "capability_id": skill.capability_id,
                            "name": f"Skill: {skill.name}",
                            "description": skill.description[:500],
                            "pricing_model": "per_call",
                            "unit_price": 0.5,
                            "is_active": True,
                        }
                        resp = await client.post(
                            f"{acp_url}/marketplace/listings",
                            json=listing,
                            headers={"Authorization": f"Bearer {api_key}"},
                        )
                        if resp.status_code < 300:
                            await catalog.mark_published(skill.name)
                            published.append(skill.name)
                            console.print(f"  [green]Published {skill.name}[/green]")
                        else:
                            console.print(f"  [yellow]{skill.name}: {resp.status_code}[/yellow]")
                    except Exception as e:
                        console.print(f"  [red]{skill.name}: {e}[/red]")

            console.print(f"\n[green]Published {len(published)} skill(s)[/green]")
        finally:
            await catalog.close()

    asyncio.run(_run())
