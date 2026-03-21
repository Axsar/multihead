"""Skill Catalog: persistent registry of available skills from GitHub and local sources.

Stores skill metadata in SQLite so skills can be browsed, searched, and installed
without having them all loaded in memory. Supports the full lifecycle:
  discover → install → load → publish to marketplace
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

CATALOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    name            TEXT PRIMARY KEY,
    description     TEXT NOT NULL DEFAULT '',
    source_url      TEXT NOT NULL DEFAULT '',
    category        TEXT NOT NULL DEFAULT 'general',
    license_type    TEXT NOT NULL DEFAULT 'free',
    license_price   REAL NOT NULL DEFAULT 0.0,
    author          TEXT NOT NULL DEFAULT '',
    version         TEXT NOT NULL DEFAULT '1.0.0',
    tags            TEXT NOT NULL DEFAULT '[]',
    compatibility   TEXT NOT NULL DEFAULT '',
    instruction_lines INTEGER NOT NULL DEFAULT 0,
    installed       INTEGER NOT NULL DEFAULT 0,
    installed_at    TEXT,
    published       INTEGER NOT NULL DEFAULT 0,
    published_at    TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS catalog_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS royalty_ledger (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_name      TEXT NOT NULL,
    provider_id     TEXT NOT NULL DEFAULT '',
    consumer_id     TEXT NOT NULL DEFAULT '',
    amount          REAL NOT NULL DEFAULT 0.0,
    platform_fee    REAL NOT NULL DEFAULT 0.0,
    author_share    REAL NOT NULL DEFAULT 0.0,
    transaction_type TEXT NOT NULL DEFAULT 'invoke',
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


@dataclass
class CatalogEntry:
    """A skill in the catalog (may or may not be installed locally)."""

    name: str
    description: str
    source_url: str = ""
    category: str = "general"
    license_type: str = "free"  # free, licensed, private
    license_price: float = 0.0
    author: str = ""
    version: str = "1.0.0"
    tags: list[str] = field(default_factory=list)
    compatibility: str = ""
    instruction_lines: int = 0
    installed: bool = False
    installed_at: str | None = None
    published: bool = False
    published_at: str | None = None

    @property
    def is_free(self) -> bool:
        return self.license_type == "free" or self.license_price <= 0

    @property
    def requires_license(self) -> bool:
        return self.license_type == "licensed" and self.license_price > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "source_url": self.source_url,
            "category": self.category,
            "license_type": self.license_type,
            "license_price": self.license_price,
            "author": self.author,
            "version": self.version,
            "tags": self.tags,
            "compatibility": self.compatibility,
            "instruction_lines": self.instruction_lines,
            "installed": self.installed,
            "published": self.published,
            "capability_id": f"com.multihead.skill.{self.name}",
            "is_free": self.is_free,
            "requires_license": self.requires_license,
        }


class SkillCatalog:
    """SQLite-backed skill catalog."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def open(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(CATALOG_SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    async def upsert(self, entry: CatalogEntry) -> None:
        """Insert or update a catalog entry."""
        assert self._db
        await self._db.execute(
            """INSERT INTO skills (name, description, source_url, category, license_type,
               license_price, author, version, tags, compatibility, instruction_lines,
               installed, installed_at, published, published_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
               ON CONFLICT(name) DO UPDATE SET
                 description=excluded.description, source_url=excluded.source_url,
                 category=excluded.category, license_type=excluded.license_type,
                 license_price=excluded.license_price, author=excluded.author,
                 version=excluded.version, tags=excluded.tags,
                 compatibility=excluded.compatibility,
                 instruction_lines=excluded.instruction_lines,
                 installed=excluded.installed, installed_at=excluded.installed_at,
                 published=excluded.published, published_at=excluded.published_at,
                 updated_at=datetime('now')""",
            (
                entry.name, entry.description, entry.source_url, entry.category,
                entry.license_type, entry.license_price, entry.author, entry.version,
                json.dumps(entry.tags), entry.compatibility, entry.instruction_lines,
                int(entry.installed), entry.installed_at,
                int(entry.published), entry.published_at,
            ),
        )
        await self._db.commit()

    async def delete(self, name: str) -> bool:
        """Delete a catalog entry by name and remember the deletion."""
        assert self._db
        cursor = await self._db.execute("DELETE FROM skills WHERE name = ?", (name,))
        # Remember deletion so startup sync doesn't re-add it
        await self._db.execute(
            "INSERT OR IGNORE INTO catalog_meta (key, value) VALUES (?, ?)",
            (f"deleted:{name}", "1"),
        )
        await self._db.commit()
        return cursor.rowcount > 0

    async def is_deleted(self, name: str) -> bool:
        """Check if a skill was explicitly deleted by the user."""
        assert self._db
        async with self._db.execute(
            "SELECT value FROM catalog_meta WHERE key = ?", (f"deleted:{name}",)
        ) as cur:
            return await cur.fetchone() is not None

    async def get(self, name: str) -> CatalogEntry | None:
        assert self._db
        async with self._db.execute("SELECT * FROM skills WHERE name = ?", (name,)) as cur:
            row = await cur.fetchone()
            return self._row_to_entry(row) if row else None

    async def list_all(self, category: str | None = None, installed_only: bool = False,
                       search: str | None = None, limit: int = 100) -> list[CatalogEntry]:
        assert self._db
        conditions = []
        params: list[Any] = []
        if category:
            conditions.append("category = ?")
            params.append(category)
        if installed_only:
            conditions.append("installed = 1")
        if search:
            conditions.append("(name LIKE ? OR description LIKE ? OR tags LIKE ?)")
            pat = f"%{search}%"
            params.extend([pat, pat, pat])
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM skills {where} ORDER BY name LIMIT ?"
        params.append(limit)
        async with self._db.execute(query, params) as cur:
            return [self._row_to_entry(row) async for row in cur]

    async def count(self) -> dict[str, int]:
        assert self._db
        async with self._db.execute("SELECT COUNT(*) as total, SUM(installed) as installed, SUM(published) as published FROM skills") as cur:
            row = await cur.fetchone()
            return {"total": row["total"], "installed": row["installed"] or 0, "published": row["published"] or 0}

    async def categories(self) -> list[dict[str, Any]]:
        assert self._db
        async with self._db.execute("SELECT category, COUNT(*) as count FROM skills GROUP BY category ORDER BY count DESC") as cur:
            return [{"category": row["category"], "count": row["count"]} async for row in cur]

    async def mark_installed(self, name: str) -> None:
        assert self._db
        await self._db.execute(
            "UPDATE skills SET installed = 1, installed_at = datetime('now'), updated_at = datetime('now') WHERE name = ?",
            (name,),
        )
        await self._db.commit()

    async def mark_published(self, name: str) -> None:
        assert self._db
        await self._db.execute(
            "UPDATE skills SET published = 1, published_at = datetime('now'), updated_at = datetime('now') WHERE name = ?",
            (name,),
        )
        await self._db.commit()

    async def set_meta(self, key: str, value: str) -> None:
        assert self._db
        await self._db.execute(
            "INSERT OR REPLACE INTO catalog_meta (key, value) VALUES (?, ?)", (key, value)
        )
        await self._db.commit()

    async def get_meta(self, key: str) -> str | None:
        assert self._db
        async with self._db.execute("SELECT value FROM catalog_meta WHERE key = ?", (key,)) as cur:
            row = await cur.fetchone()
            return row["value"] if row else None

    async def record_royalty(
        self, skill_name: str, amount: float,
        provider_id: str = "", consumer_id: str = "",
        transaction_type: str = "invoke",
        platform_fee_pct: float = 0.15, author_share_pct: float = 0.70,
    ) -> int:
        """Record a royalty transaction for a licensed skill invocation.

        Default split: 15% platform, 70% author, 15% provider.
        """
        assert self._db
        platform_fee = round(amount * platform_fee_pct, 4)
        author_share = round(amount * author_share_pct, 4)
        cursor = await self._db.execute(
            """INSERT INTO royalty_ledger
               (skill_name, provider_id, consumer_id, amount, platform_fee, author_share, transaction_type)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (skill_name, provider_id, consumer_id, amount, platform_fee, author_share, transaction_type),
        )
        await self._db.commit()
        return cursor.lastrowid or 0

    async def royalty_summary(self, skill_name: str | None = None) -> dict[str, Any]:
        """Get royalty summary, optionally filtered by skill."""
        assert self._db
        if skill_name:
            query = "SELECT COUNT(*) as txns, COALESCE(SUM(amount),0) as total, COALESCE(SUM(platform_fee),0) as fees, COALESCE(SUM(author_share),0) as author FROM royalty_ledger WHERE skill_name = ?"
            params: list[Any] = [skill_name]
        else:
            query = "SELECT COUNT(*) as txns, COALESCE(SUM(amount),0) as total, COALESCE(SUM(platform_fee),0) as fees, COALESCE(SUM(author_share),0) as author FROM royalty_ledger"
            params = []
        async with self._db.execute(query, params) as cur:
            row = await cur.fetchone()
            return {
                "transactions": row["txns"],
                "total_amount": row["total"],
                "platform_fees": row["fees"],
                "author_earnings": row["author"],
            }

    def _row_to_entry(self, row: aiosqlite.Row) -> CatalogEntry:
        tags_raw = row["tags"]
        try:
            tags = json.loads(tags_raw) if tags_raw else []
        except (json.JSONDecodeError, TypeError):
            tags = []
        return CatalogEntry(
            name=row["name"],
            description=row["description"],
            source_url=row["source_url"],
            category=row["category"],
            license_type=row["license_type"],
            license_price=row["license_price"],
            author=row["author"],
            version=row["version"],
            tags=tags,
            compatibility=row["compatibility"],
            instruction_lines=row["instruction_lines"],
            installed=bool(row["installed"]),
            installed_at=row["installed_at"],
            published=bool(row["published"]),
            published_at=row["published_at"],
        )


async def fetch_github_skills(repo_url: str = "https://api.github.com/repos/anthropics/skills/contents") -> list[CatalogEntry]:
    """Fetch skill list from GitHub API (just names + descriptions from directory listing).

    This doesn't download the full SKILL.md — just discovers what's available.
    """
    import httpx

    entries: list[CatalogEntry] = []
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(repo_url, headers={"Accept": "application/vnd.github.v3+json"})
            resp.raise_for_status()
            items = resp.json()

        for item in items:
            if item.get("type") != "dir":
                continue
            name = item["name"]
            if name.startswith(".") or name in ("docs", "tests", "scripts"):
                continue
            entries.append(CatalogEntry(
                name=name,
                description=f"Anthropic official skill: {name}",
                source_url=f"https://github.com/anthropics/skills/tree/main/{name}",
                category=_guess_category(name),
                author="anthropic",
                tags=["official", "anthropic"],
            ))

        logger.info("Fetched %d skills from GitHub", len(entries))
    except Exception as e:
        logger.error("Failed to fetch GitHub skills: %s", e)

    return entries


async def install_skill_from_github(
    skill_name: str,
    skills_dir: Path,
    repo_base: str = "https://raw.githubusercontent.com/anthropics/skills/main",
) -> Path | None:
    """Download a skill's SKILL.md and supporting files from GitHub."""
    import httpx

    skill_dir = skills_dir / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Download SKILL.md
            resp = await client.get(f"{repo_base}/{skill_name}/SKILL.md")
            resp.raise_for_status()
            (skill_dir / "SKILL.md").write_text(resp.text, encoding="utf-8")
            logger.info("Downloaded %s/SKILL.md (%d bytes)", skill_name, len(resp.text))

            # Try to download common supporting files
            for subpath in ["reference", "references", "scripts"]:
                try:
                    # Check if directory exists via GitHub API
                    api_resp = await client.get(
                        f"https://api.github.com/repos/anthropics/skills/contents/{skill_name}/{subpath}",
                        headers={"Accept": "application/vnd.github.v3+json"},
                    )
                    if api_resp.status_code == 200:
                        sub_dir = skill_dir / subpath
                        sub_dir.mkdir(exist_ok=True)
                        for file_info in api_resp.json():
                            if file_info.get("type") == "file" and file_info.get("download_url"):
                                file_resp = await client.get(file_info["download_url"])
                                if file_resp.status_code == 200:
                                    (sub_dir / file_info["name"]).write_text(
                                        file_resp.text, encoding="utf-8"
                                    )
                except Exception:
                    pass  # Supporting files are optional

        return skill_dir
    except Exception as e:
        logger.error("Failed to install skill %s: %s", skill_name, e)
        return None


def _guess_category(name: str) -> str:
    """Guess a category from the skill name."""
    categories = {
        "pdf": "document", "xlsx": "document", "csv": "document", "markdown": "document",
        "sql": "data", "database": "data", "postgres": "data",
        "mcp": "development", "api": "development", "docker": "development",
        "git": "development", "code": "development", "debug": "development",
        "web": "web", "html": "web", "css": "web", "scrape": "web",
        "image": "media", "video": "media", "audio": "media",
        "test": "testing", "pytest": "testing", "jest": "testing",
    }
    name_lower = name.lower()
    for keyword, cat in categories.items():
        if keyword in name_lower:
            return cat
    return "general"
