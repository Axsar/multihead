"""Knowledge, packs, and nightshift commands."""

from __future__ import annotations

import click
from rich.table import Table

from ._helpers import (
    _build_knowledge_deps,
    asyncio,
    console,
    HeadManager,
    KnowledgeStore,
    NightShiftConfig,
    NightShift,
    load_heads,
    validate_heads,
    Path,
    Settings,
)
from ._core import main


# -------------------------------------------------------------------
# Knowledge commands
# -------------------------------------------------------------------

def _search_claims_filtered(
    ks: KnowledgeStore, query: str, status: str | None = None,
    claim_type: str | None = None, scope: str | None = None, limit: int = 20,
) -> list[tuple[str, str, float]]:
    """FTS search with optional SQL post-filtering."""
    import sqlite3

    keywords = [w.strip() for w in query.split() if len(w.strip()) > 2]
    if not keywords:
        return []
    fts_query = " OR ".join(keywords)

    clauses = ["c.claim_status IN ('accepted', 'proposed')"]
    params: list = [fts_query]
    if status:
        clauses.append("c.claim_status = ?")
        params.append(status)
    if claim_type:
        clauses.append("c.claim_type = ?")
        params.append(claim_type)
    if scope:
        clauses.append("c.scope_id = ?")
        params.append(scope)
    params.append(limit)
    where = " AND ".join(clauses)

    try:
        conn = sqlite3.connect(str(ks.db_path), timeout=10.0)
        conn.execute("PRAGMA busy_timeout = 5000")
        rows = conn.execute(
            f"""
            SELECT c.claim_key, c.statement, c.confidence
            FROM claims_fts f
            JOIN claims c ON c.rowid = f.rowid
            WHERE claims_fts MATCH ? AND {where}
            ORDER BY rank
            LIMIT ?
            """,
            params,
        ).fetchall()
        conn.close()
        return [(r[0] or "", r[1] or "", r[2] or 0.0) for r in rows]
    except Exception:
        # Fallback to basic FTS
        return ks.search_claims_fts(query, limit=limit)


@main.group()
def knowledge():
    """Knowledge store inspection."""


@knowledge.command("claims")
@click.option("--status", default=None, help="Filter by status")
@click.option("--limit", default=20, type=int, help="Max results")
@click.pass_context
def knowledge_claims(ctx, status, limit):
    """List claims in the knowledge store."""
    settings = ctx.obj["settings"]
    ks, _, _, _ = _build_knowledge_deps(settings)
    claims = ks.list_claims(status=status, limit=limit)
    if not claims:
        console.print("[dim]No claims found.[/dim]")
        return

    table = Table(title="Claims")
    table.add_column("ID", style="cyan")
    table.add_column("Type")
    table.add_column("Status")
    table.add_column("Statement")
    table.add_column("Confidence")

    for c in claims:
        table.add_row(
            c.claim_id[:20], c.claim_type.value, c.claim_status.value,
            c.statement[:60], f"{c.confidence:.2f}",
        )
    console.print(table)


@knowledge.command("events")
@click.option("--status", default=None, help="Filter by status")
@click.option("--limit", default=20, type=int, help="Max results")
@click.pass_context
def knowledge_events(ctx, status, limit):
    """List knowledge events."""
    settings = ctx.obj["settings"]
    ks, _, _, _ = _build_knowledge_deps(settings)
    events = ks.list_events(status=status, limit=limit)
    if not events:
        console.print("[dim]No events found.[/dim]")
        return

    table = Table(title="Knowledge Events")
    table.add_column("ID", style="cyan")
    table.add_column("Type")
    table.add_column("Status")
    table.add_column("Title")

    for e in events:
        table.add_row(
            e.event_id[:20], e.event_type.value, e.event_status.value, e.title[:60],
        )
    console.print(table)


@knowledge.command("refresh-refs")
@click.option(
    "--memory-dir",
    type=click.Path(),
    default=None,
    help="Override Claude Code memory directory path.",
)
@click.pass_context
def knowledge_refresh_refs(ctx, memory_dir):
    """Regenerate reference bank files from knowledge.db.

    Writes topic-specific .md files (decisions, constraints, architecture,
    open loops, recent activity, capabilities) to Claude Code's project
    memory directory. These files survive SDK context compaction.
    """
    from ..ref_bank import RefBankBuilder

    settings = ctx.obj["settings"]
    ks, _, _, _ = _build_knowledge_deps(settings)

    if memory_dir:
        mem_path = Path(memory_dir)
    else:
        # Default: Claude Code project memory for current working directory
        cwd_folder = str(Path.cwd()).replace("/", "-").replace("\\", "-")
        if not cwd_folder.startswith("-"):
            cwd_folder = "-" + cwd_folder
        mem_path = Path.home() / ".claude" / "projects" / cwd_folder / "memory"

    builder = RefBankBuilder(ks, mem_path)
    results = builder.refresh_all()

    table = Table(title="Reference Bank Updated")
    table.add_column("File", style="cyan")
    table.add_column("Items", justify="right")
    table.add_column("Lines", justify="right")

    for r in results:
        table.add_row(r["filename"], str(r["items_count"]), str(r["lines"]))

    console.print(table)
    console.print(f"\n[green]Written to: {mem_path}[/green]")


@knowledge.command("search")
@click.argument("query")
@click.option("--status", default=None, help="Filter by claim status (accepted, proposed, etc.)")
@click.option("--type", "claim_type", default=None, help="Filter by claim type (fact, decision, etc.)")
@click.option("--limit", "-n", default=20, type=int, help="Max results")
@click.option("--scope", default=None, help="Filter by scope ID")
@click.option("--full", is_flag=True, help="Show full statements (no truncation)")
@click.option("--semantic", "-s", is_flag=True, help="Use embedding-based semantic search (conceptual matching)")
@click.pass_context
def knowledge_search(ctx, query, status, claim_type, limit, scope, full, semantic):
    """Search claims by keyword or semantically.

    Default uses FTS5 full-text search. Add --semantic for conceptual matching
    (e.g. "portability" finds "friends can install it").

        multihead knowledge search "portability"

        multihead knowledge search "how to install" --semantic

        multihead kb "authentication" --scope myproject -n 10 -s
    """
    settings = ctx.obj["settings"]
    ks, _, _, _ = _build_knowledge_deps(settings)

    if semantic:
        _run_semantic_search(settings, query, limit, full)
    else:
        results = _search_claims_filtered(ks, query, status, claim_type, scope, limit)
        _print_search_results(query, results, full)


def _run_semantic_search(settings: "Settings", query: str, limit: int, full: bool) -> None:
    """Run embedding-based semantic search."""
    try:
        from ..embedding_search import EmbeddingIndex
    except ImportError:
        console.print("[red]sentence-transformers not installed. Run: pip install sentence-transformers[/red]")
        return

    cache_dir = settings.data_dir / "embeddings"
    index = EmbeddingIndex(settings.knowledge_db_path, cache_dir=cache_dir)

    with console.status("[bold green]Building embedding index..."):
        count = index.build()

    if count == 0:
        console.print("[dim]No claims to search.[/dim]")
        return

    results_semantic = index.search(query, limit=limit)

    if not results_semantic:
        console.print(f"[dim]No semantic matches for '{query}'[/dim]")
        return

    table = Table(title=f"Semantic: '{query}' ({len(results_semantic)} results, {count} indexed)")
    table.add_column("Key", style="cyan", max_width=30)
    table.add_column("Statement", ratio=3)
    table.add_column("Conf", justify="right", style="green")
    table.add_column("Sim", justify="right", style="magenta")

    for claim_key, statement, confidence, similarity in results_semantic:
        stmt = statement if full else (statement[:120] + "..." if len(statement) > 120 else statement)
        table.add_row(
            claim_key[:30] if not full else claim_key,
            stmt,
            f"{confidence:.2f}",
            f"{similarity:.3f}",
        )
    console.print(table)


def _print_search_results(
    query: str, results: list[tuple[str, str, float]], full: bool,
) -> None:
    """Print FTS search results table."""
    if not results:
        console.print(f"[dim]No claims matching '{query}'[/dim]")
        return

    table = Table(title=f"Knowledge: '{query}' ({len(results)} results)")
    table.add_column("Key", style="cyan", max_width=30)
    table.add_column("Statement", ratio=3)
    table.add_column("Conf", justify="right", style="green")

    for claim_key, statement, confidence in results:
        stmt = statement if full else (statement[:120] + "..." if len(statement) > 120 else statement)
        table.add_row(
            claim_key[:30] if not full else claim_key,
            stmt,
            f"{confidence:.2f}",
        )
    console.print(table)


# Top-level shortcut: multihead kb "query"
@main.command("kb")
@click.argument("query")
@click.option("--status", default=None, help="Filter by claim status")
@click.option("--type", "claim_type", default=None, help="Filter by claim type")
@click.option("--limit", "-n", default=20, type=int, help="Max results")
@click.option("--scope", default=None, help="Filter by scope ID")
@click.option("--full", is_flag=True, help="Show full statements")
@click.option("--semantic", "-s", is_flag=True, help="Embedding-based semantic search")
@click.pass_context
def kb(ctx, query, status, claim_type, limit, scope, full, semantic):
    """Quick knowledge search (shortcut for 'knowledge search').

    Examples:

        multihead kb "authentication"

        multihead kb "consensus" --type decision --scope myproject

        multihead kb "how do agents communicate" -s
    """
    ctx.invoke(knowledge_search, query=query, status=status,
               claim_type=claim_type, limit=limit, scope=scope, full=full,
               semantic=semantic)


@main.command("briefing")
@click.argument("file_path")
@click.option("--compact", is_flag=True, help="Short output for hooks (max 3 per category, 1 line each).")
@click.pass_context
def file_briefing_cmd(ctx, file_path, compact):
    """Get knowledge briefing for a file before editing.

    Shows corroborated constraints, stale warnings, contested signals,
    superseded history, and unverified fixes for the given file.

    Use --compact for hook-friendly output (10-20 lines max).

    Examples:

        multihead briefing src/multihead/claim_fusion.py
        multihead briefing --compact src/multihead/claim_fusion.py
    """
    import sqlite3
    import os

    settings = ctx.obj["settings"]
    db_path = settings.knowledge_db_path

    from multihead._paths import normalize_file_path
    basename = os.path.basename(file_path)
    rel_path = normalize_file_path(file_path)

    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT claim_key, statement, claim_status, confidence, observation_method, producer, contested_reason "
        "FROM claims WHERE provenance_json LIKE ? OR provenance_json LIKE ? "
        "ORDER BY claim_status, confidence DESC LIMIT 100",
        (f"%{rel_path}%", f"%{basename}%"),
    ).fetchall()

    buckets = {"corroborated": [], "stale": [], "contested": [], "superseded": [], "other": []}
    for r in rows:
        producer = r['producer'] or '?'
        if compact:
            entry = f"  • {r['statement'][:120]}"
        else:
            entry = f"  [{r['observation_method']}|{producer}] {r['statement'][:200]}"
        status = r["claim_status"]
        if status in buckets:
            buckets[status].append(entry)
        else:
            buckets["other"].append(entry)

    # Check unverified
    unverified = conn.execute(
        "SELECT statement FROM claims WHERE contested_reason LIKE '%IMPROVEMENT_UNVERIFIED%' "
        "AND (provenance_json LIKE ? OR provenance_json LIKE ?) LIMIT 10",
        (f"%{rel_path}%", f"%{basename}%"),
    ).fetchall()
    conn.close()

    max_per_cat = 3 if compact else 10

    if compact:
        # Plain text output for hooks (stdout → injected into agent context)
        print(f"=== Knowledge: {rel_path} ===")
        if buckets["corroborated"]:
            print(f"CONSTRAINTS ({len(buckets['corroborated'])} corroborated — don't violate):")
            for e in buckets["corroborated"][:max_per_cat]:
                print(e)
        if buckets["stale"]:
            print(f"WARNINGS ({len(buckets['stale'])} stale — verify before assuming):")
            for e in buckets["stale"][:max_per_cat]:
                print(e)
        if buckets["superseded"]:
            print(f"HISTORY ({len(buckets['superseded'])} superseded — don't repeat):")
            for e in buckets["superseded"][:max_per_cat]:
                print(e)
        if buckets["contested"]:
            print(f"SIGNALS ({len(buckets['contested'])} contested — channels disagree)")
        if unverified:
            print(f"UNVERIFIED ({len(unverified)} claimed fixes not confirmed)")
        if not any(buckets.values()) and not unverified:
            print("No knowledge found for this file.")
        return

    console.print(f"[bold]Knowledge Briefing: {rel_path}[/bold]\n")
    if buckets["corroborated"]:
        console.print(f"[green]## CONSTRAINTS ({len(buckets['corroborated'])} corroborated — treat as invariants)[/green]")
        for e in buckets["corroborated"][:10]:
            console.print(e)
    if buckets["stale"]:
        console.print(f"\n[yellow]## WARNINGS ({len(buckets['stale'])} stale — verify before assuming)[/yellow]")
        for e in buckets["stale"][:10]:
            console.print(e)
    if buckets["contested"]:
        console.print(f"\n[red]## SIGNALS ({len(buckets['contested'])} contested — channels disagree)[/red]")
        for e in buckets["contested"][:10]:
            console.print(e)
    if buckets["superseded"]:
        console.print(f"\n[dim]## HISTORY ({len(buckets['superseded'])} superseded — don't repeat)[/dim]")
        for e in buckets["superseded"][:10]:
            console.print(e)
    if unverified:
        console.print(f"\n[magenta]## UNVERIFIED FIXES ({len(unverified)} — claimed fixed, not confirmed)[/magenta]")
        for u in unverified:
            console.print(f"  {u['statement'][:200]}")
    if not any(buckets.values()) and not unverified:
        console.print("[dim]No knowledge found for this file.[/dim]")
    else:
        total = sum(len(v) for v in buckets.values())
        console.print(f"\n[dim]{total} claims total[/dim]")


@knowledge.command("deposit")
@click.argument("statement")
@click.option("--key", "-k", required=True, help="Claim key (e.g. 'discussion.my_topic')")
@click.option("--producer", "-p", default="cli_user", help="Producer ID (your agent name)")
@click.option("--type", "claim_type", default="fact", help="Claim type: fact, decision, plan, question, definition, constraint, preference")
@click.option("--scope", default="multihead", help="Scope ID (e.g. multihead, default, botvibes)")
@click.option("--confidence", "-c", default=0.8, type=float, help="Confidence 0.0-1.0")
@click.option("--reply-to", default=None, help="Claim ID this responds to (sets related_claim_ids)")
@click.option("--observation-method", default="agent_deposit", help="How this was observed (agent_deposit, user_statement, etc.)")
@click.pass_context
def knowledge_deposit(ctx, statement, key, producer, claim_type, scope, confidence, reply_to, observation_method):
    """Deposit a claim to knowledge.db with proper formatting.

    Use this instead of raw SQL inserts. Handles timestamps, provenance,
    FTS indexing, and vector embeddings automatically.

    Examples:

        multihead knowledge deposit "Router uses weighted scoring" \\
            -k router.scoring.method -p claude_code_main

        multihead knowledge deposit "RESPONSE: I agree with the approach" \\
            -k discussion.grounding_response -p claude_code_main \\
            --reply-to clm_3C79D4CC

    Shortcut:

        multihead deposit "claim text" -k key -p producer
    """
    import uuid
    from datetime import datetime, timezone

    from multihead.knowledge_models import (
        Claim, ClaimCanonical, ClaimScope, ClaimStatus, ClaimType as CT,
        EntityRef, Provenance, ScopeType, ValueObject,
    )
    from multihead.scope_inference import infer_scope

    settings = ctx.obj["settings"]
    ks, _, _, _ = _build_knowledge_deps(settings)

    ct_map = {
        "fact": CT.FACT, "plan": CT.PLAN, "question": CT.QUESTION,
        "decision": CT.DECISION, "definition": CT.DEFINITION,
        "constraint": CT.CONSTRAINT, "preference": CT.PREFERENCE,
    }

    claim_id = f"clm_{uuid.uuid4().hex[:24].upper()}"
    resolved_scope = scope or infer_scope(key, statement)

    claim = Claim(
        claim_id=claim_id,
        claim_type=ct_map.get(claim_type, CT.FACT),
        claim_status=ClaimStatus.PROPOSED,
        scope=ClaimScope(
            scope_type=ScopeType.PROJECT,
            scope_id=resolved_scope,
        ),
        canonical=ClaimCanonical(
            claim_key=key,
            subject=EntityRef(entity_type="claim", entity_id=claim_id),
            predicate="states",
            object=ValueObject(value_type="string", value=statement[:200]),
        ),
        statement=statement,
        confidence=confidence,
        provenance=Provenance(
            produced_by={
                "kind": "agent",
                "id": producer,
                "folder": str(Path.cwd()),
            },
            observation_method=observation_method,
        ),
        related_claim_ids=[reply_to] if reply_to else [],
    )

    ks.insert_claim(claim, dedup=False)
    console.print(f"[green]Deposited:[/green] {claim_id}")
    console.print(f"  key: {key}")
    console.print(f"  producer: {producer}")
    console.print(f"  scope: {resolved_scope}")
    if reply_to:
        console.print(f"  reply-to: {reply_to}")


# Top-level shortcut: multihead deposit "text" -k key -p producer
@main.command("deposit")
@click.argument("statement")
@click.option("--key", "-k", required=True, help="Claim key")
@click.option("--producer", "-p", default="cli_user", help="Producer ID")
@click.option("--type", "claim_type", default="fact", help="Claim type")
@click.option("--scope", default="multihead", help="Scope ID")
@click.option("--confidence", "-c", default=0.8, type=float, help="Confidence")
@click.option("--reply-to", default=None, help="Claim ID this responds to")
@click.option("--observation-method", default="agent_deposit", help="Observation method")
@click.pass_context
def deposit_shortcut(ctx, statement, key, producer, claim_type, scope, confidence, reply_to, observation_method):
    """Quick claim deposit (shortcut for 'knowledge deposit')."""
    ctx.invoke(knowledge_deposit, statement=statement, key=key, producer=producer,
               claim_type=claim_type, scope=scope, confidence=confidence,
               reply_to=reply_to, observation_method=observation_method)


# -------------------------------------------------------------------
# Packs commands
# -------------------------------------------------------------------

@main.group()
def packs():
    """Context pack management."""


@packs.command("list")
@click.pass_context
def packs_list(ctx):
    """List all built context packs."""
    settings = ctx.obj["settings"]
    ks, _, _, pb = _build_knowledge_deps(settings)
    listing = pb.list_packs()
    if not listing:
        console.print("[dim]No packs built yet.[/dim]")
        return

    table = Table(title="Context Packs")
    table.add_column("Pack ID", style="cyan")
    table.add_column("Purpose")
    table.add_column("Items")
    table.add_column("Created")

    for p in listing:
        table.add_row(p["pack_id"], p["purpose"], str(p["item_count"]), p["created_at"])
    console.print(table)


@packs.command("build")
@click.option("--purpose", default="Custom Pack", help="Pack purpose")
@click.pass_context
def packs_build(ctx, purpose):
    """Build a context pack."""
    settings = ctx.obj["settings"]
    ks, _, _, pb = _build_knowledge_deps(settings)
    pack = pb.build_pack(purpose=purpose)
    console.print(f"[green]Pack built:[/green] {pack.pack_id}")
    console.print(f"  Purpose: {pack.purpose}")
    console.print(f"  Items: {len(pack.items)}")


@packs.command("build-standard")
@click.pass_context
def packs_build_standard(ctx):
    """Build all 5 standard Night Shift packs."""
    settings = ctx.obj["settings"]
    ks, _, _, pb = _build_knowledge_deps(settings)
    result = pb.build_standard_packs()
    console.print(f"[green]Built {len(result)} standard packs.[/green]")
    for p in result:
        console.print(f"  {p.purpose}: {len(p.items)} items")


@packs.command("show")
@click.argument("pack_id")
@click.pass_context
def packs_show(ctx, pack_id):
    """Show a pack's contents."""
    settings = ctx.obj["settings"]
    ks, _, _, pb = _build_knowledge_deps(settings)
    pack = pb.load_pack(pack_id)
    if pack is None:
        console.print(f"[red]Pack not found: {pack_id}[/red]")
        return
    console.print(f"[bold]{pack.purpose}[/bold] ({pack.pack_id})")
    console.print(f"  Items: {len(pack.items)}, Created: {pack.created_at.isoformat()}")
    for item in pack.items:
        console.print(f"  [{item.type.upper()}] {item.text[:80]}")


# -------------------------------------------------------------------
# Harvest commands
# -------------------------------------------------------------------

@main.command("harvest-conversations")
@click.option("--max-files", default=50, type=int, help="Max files to process per run.")
@click.option("--status-only", is_flag=True, help="Show status without harvesting.")
@click.pass_context
def harvest_conversations(ctx, max_files, status_only):
    """Harvest Claude Code session transcripts into RecordStore."""
    from multihead.conversation_harvester import ConversationHarvester

    settings = ctx.obj["settings"]
    ks, rs, art, pb = _build_knowledge_deps(settings)

    harvester = ConversationHarvester(
        record_store=rs,
        knowledge_store=ks,
        data_dir=settings.data_dir,
        max_files_per_run=max_files,
    )

    if status_only:
        status = harvester.status()
        console.print(f"[bold]Conversation Harvester Status[/bold]")
        console.print(f"  Total files: {status['total_files']}")
        console.print(f"  Total size: {status['total_size_mb']} MB")
        console.print(f"  Processed: {status['files_processed']}")
        console.print(f"  Remaining: {status['files_remaining']}")
        console.print(f"  Exchanges harvested: {status['total_exchanges']}")
        console.print(f"  Last harvest: {status['last_harvest'] or 'never'}")
        console.print(f"  Projects: {status['projects']}")
        for proj, count in sorted(status.get("project_breakdown", {}).items()):
            console.print(f"    {proj}: {count} files")
        return

    result = harvester.harvest_all()
    console.print(f"[green]Conversation harvest complete![/green]")
    console.print(f"  Files scanned: {result.files_scanned}")
    console.print(f"  Files processed: {result.files_processed}")
    console.print(f"  Files skipped (unchanged): {result.files_skipped}")
    console.print(f"  Exchanges ingested: {result.exchanges_ingested}")
    console.print(f"  Records created: {result.records_created}")
    console.print(f"  Duration: {result.duration_seconds}s")
    if result.errors:
        console.print(f"  [yellow]Errors: {len(result.errors)}[/yellow]")
        for err in result.errors[:5]:
            console.print(f"    {err[:120]}")


# -------------------------------------------------------------------
# Inbox command
# -------------------------------------------------------------------

@main.command()
@click.option("--scope", default=None, help="Filter by scope (default: all scopes).")
@click.option("--age", default=72, type=int, help="Max age in hours (default: 72).")
@click.option("--agent", default="claude-multihead-main", help="Agent identity.")
@click.pass_context
def inbox(ctx, scope, age, agent):
    """Show pending action items across all scopes."""
    from datetime import datetime, timezone

    from ._helpers import _get_settings
    settings = _get_settings()
    ks = KnowledgeStore(settings.data_dir / "knowledge.db")

    claims = ks.get_unhandled_claims(
        agent_id=agent,
        scope_id=scope,
        max_age_hours=age,
        limit=50,
        key_prefixes=[
            "action.", "solve.consensus.", "solve.request.",
            "solve.vote.", "solve.proposal.", "vote.", "consensus.",
        ],
    )

    if not claims:
        console.print("[dim]Inbox empty — no pending action items.[/dim]")
        return

    # Group by scope
    by_scope: dict[str, list] = {}
    for c in claims:
        s = c.scope.scope_id if hasattr(c, "scope") and hasattr(c.scope, "scope_id") else "?"
        by_scope.setdefault(s, []).append(c)

    now = datetime.now(timezone.utc)
    for scope_name, scope_claims in sorted(by_scope.items()):
        table = Table(title=f"[bold]{scope_name}[/bold]", show_lines=False)
        table.add_column("Key", style="cyan", max_width=50)
        table.add_column("Type", style="green", width=10)
        table.add_column("From", width=20)
        table.add_column("Deadline", width=12)
        table.add_column("Statement", max_width=60)

        for c in scope_claims:
            ctype = c.claim_type.value if hasattr(c.claim_type, "value") else str(c.claim_type)
            sender = c.provenance.produced_by.get("id", "?") if c.provenance.produced_by else "?"
            # Check expiry
            deadline_str = ""
            vt_raw = c.scope.valid_to if hasattr(c, "scope") and hasattr(c.scope, "valid_to") else None
            if vt_raw:
                try:
                    vt = vt_raw if hasattr(vt_raw, "tzinfo") else datetime.fromisoformat(str(vt_raw))
                    if vt.tzinfo and vt < now:
                        deadline_str = "[red]EXPIRED[/red]"
                    else:
                        deadline_str = str(vt_raw)[:10]
                except Exception:
                    deadline_str = str(vt_raw)[:10]
            stmt = c.statement[:80].replace("\n", " ")
            table.add_row(c.canonical.claim_key, ctype, sender, deadline_str, stmt)
        console.print(table)
        console.print()

    console.print(f"[dim]{len(claims)} items total[/dim]")


# -------------------------------------------------------------------
# Night Shift commands
# -------------------------------------------------------------------

@main.group()
def nightshift():
    """Night Shift memory refinery commands."""


@nightshift.command("run")
@click.option("--debug", is_flag=True, help="Show stage-by-stage progress output.")
@click.option("--head", default=None, help="Head for extraction stages (e.g. qwen-llm).")
@click.option("--reasoning-head", default=None, help="Head for reasoning stages (e.g. claude-sdk). Auto-switches for consistency_check, open_loops, daily_brief, weekly_rollup, cross_topic_links, backlog_sweep.")
@click.option("--model", default=None, help="Override model for claude-sdk head (e.g. claude-sonnet-4-6, claude-opus-4-6).")
@click.option("--concurrency", default=4, type=int, help="Parallel LLM calls per stage (default: 4).")
@click.option("--max-chunks", default=None, type=int, help="Cap chunks for faster end-to-end runs (e.g. 100).")
@click.option("--from-stage", default=0, type=int, help="Resume from stage N (0=full run). Loads saved state from previous run.")
@click.option("--to-stage", default=None, type=int, help="Stop after stage N (inclusive). Useful for running a single stage with --from-stage.")
@click.option("--batch", is_flag=True, help="Use Anthropic Batch API for extraction stages (50% cheaper, async).")
@click.option("--no-wait", is_flag=True, help="Submit batches and exit — don't poll. Resume next run to pick up results.")
@click.pass_context
def nightshift_run(ctx, debug, head, reasoning_head, model, concurrency, max_chunks, from_stage, to_stage, batch, no_wait):
    """Run the Night Shift pipeline."""
    settings = ctx.obj["settings"]
    ks, rs, art, pb = _build_knowledge_deps(settings)
    all_heads = load_heads(settings.config_dir)

    # Validate adapter availability
    heads, adapter_warnings = validate_heads(all_heads)
    for w in adapter_warnings:
        console.print(f"[yellow]{w} — disabled[/yellow]")

    if not heads:
        console.print("[red]No working heads found. Run 'multihead init --auto' to configure.[/red]")
        return

    # Warn if multihead serve is running (wastes GPU for API-based heads)
    if head and any(h in head for h in ("openai", "anthropic", "claude")):
        import subprocess as _sp
        try:
            serve_check = _sp.run(["pgrep", "-f", "multihead serve"], capture_output=True, timeout=2)
            if serve_check.returncode == 0:
                console.print("[yellow]⚠ multihead serve is running. Using API head '{head}' — serve wastes GPU. Consider killing serve.[/yellow]".format(head=head))
        except Exception:
            pass

    # Override model on claude_agent_sdk heads by patching manifest.extra before HeadManager loads them
    if model:
        for h_id, manifest in heads.items():
            adapter_str = str(manifest.adapter.value) if hasattr(manifest.adapter, "value") else str(manifest.adapter)
            if adapter_str == "claude_agent_sdk":
                if manifest.extra is None:
                    manifest.extra = {}
                manifest.extra["model"] = model
        console.print(f"[dim]Model override: {model}[/dim]")

    hm = HeadManager(heads, knowledge_store=ks)
    if head and head in heads:
        core_head = head
    else:
        core_head = settings.resolve_core_head_id(heads)

    reasoning_head_id = reasoning_head if reasoning_head and reasoning_head in heads else None
    if reasoning_head and not reasoning_head_id:
        console.print(f"[yellow]Warning: reasoning-head '{reasoning_head}' not found — falling back to extraction head[/yellow]")
    if reasoning_head_id:
        console.print(f"[dim]Reasoning stages → {reasoning_head_id}, extraction stages → {core_head}[/dim]")

    config = NightShiftConfig(head_id=core_head, concurrency=concurrency, reasoning_head_id=reasoning_head_id, batch_mode=batch, no_wait=no_wait)
    if batch:
        msg = "Batch mode: extraction stages will use Anthropic Batch API (50% cheaper)"
        if no_wait:
            msg += " — no-wait: submit and exit, resume next run"
        console.print(f"[dim]{msg}[/dim]")
    ns = NightShift(ks, rs, art, pb, hm, config, settings.nightshift_output_dir, settings=settings)

    if from_stage > 0:
        stage_names = [s.name for s in __import__("multihead.night_shift.models", fromlist=["STAGES"]).STAGES]
        resume_name = stage_names[from_stage] if from_stage < len(stage_names) else f"stage {from_stage}"
        console.print(f"[cyan]Resuming from stage {from_stage}: {resume_name}[/cyan]")

    if debug:
        def _debug_printer(evt: dict) -> None:
            e = evt.get("event")
            if e == "stage_start":
                console.print(f"[dim]>> Stage {evt['index'] + 1}/{evt['total']}: {evt['stage']}[/dim]")
            elif e == "stage_done":
                console.print(f"[green]   OK[/green] {evt['stage']} ({evt['elapsed_s']}s) -- {evt.get('summary', '')}")
            elif e == "stage_skip":
                console.print(f"[yellow]   SKIP[/yellow] {evt['stage']} ({evt['elapsed_s']}s)")
            elif e == "stage_fail":
                console.print(f"[red]   FAIL[/red] {evt['stage']} ({evt['elapsed_s']}s) -- {evt.get('error', '')[:120]}")
            elif e == "gate":
                console.print(f"[yellow]   GATE[/yellow] {evt['stage']}: {evt['decision']} (attempt {evt.get('attempt', '?')})")
            elif e == "llm_call":
                console.print(f"[blue]   LLM[/blue] {evt.get('prompt_chars', 0)} chars -> {evt.get('tokens', 0)} tokens ({evt['elapsed_s']}s)")
            elif e == "stage_pending":
                console.print(f"[cyan]   PENDING[/cyan] {evt['stage']} — batch {evt.get('batch_id', '?')} submitted, resume with --from-stage")
            elif e == "complete":
                console.print(f"[bold green]Pipeline done:[/bold green] {evt.get('records', 0)} records, {evt.get('events', 0)} events, {evt.get('claims', 0)} claims, {evt.get('packs', 0)} packs")
        ns.on_progress = _debug_printer

    async def _run():
        report = await ns.run(max_chunks=max_chunks, from_stage=from_stage, to_stage=to_stage)
        console.print(f"[green]Night Shift complete![/green]")
        console.print(f"  Records processed: {report.records_processed}")
        console.print(f"  Events created: {report.events_created}")
        console.print(f"  Claims created: {report.claims_created}")
        console.print(f"  Stages completed: {len(report.stages_completed)}")
        console.print(f"  Stages failed: {len(report.stages_failed)}")
        console.print(f"  Stages skipped: {len(report.stages_skipped)}")
        await hm.shutdown()

    asyncio.run(_run())


@nightshift.command("backlog")
@click.option("--batch-size", default=500, type=int, help="Claims per batch.")
@click.option("--reset", is_flag=True, help="Reset cursor and reprocess all claims.")
@click.option("--head", default=None, help="Head ID to use for LLM stages.")
@click.pass_context
def nightshift_backlog(ctx, batch_size, reset, head):
    """Run backlog sweep: process all existing claims through analysis stages."""
    settings = ctx.obj["settings"]
    ks, rs, art, pb = _build_knowledge_deps(settings)
    all_heads = load_heads(settings.config_dir)
    heads, adapter_warnings = validate_heads(all_heads)
    for w in adapter_warnings:
        console.print(f"[yellow]{w} — disabled[/yellow]")

    if not heads:
        console.print("[red]No working heads found.[/red]")
        return

    hm = HeadManager(heads, knowledge_store=ks)
    if head and head in heads:
        core_head = head
    else:
        core_head = settings.resolve_core_head_id(heads)
    config = NightShiftConfig(head_id=core_head)
    ns = NightShift(ks, rs, art, pb, hm, config, settings.nightshift_output_dir, settings=settings)

    def _progress(evt: dict) -> None:
        e = evt.get("event")
        if e == "backlog_batch":
            console.print(
                f"[dim]  Batch: offset={evt['offset']}, "
                f"size={evt['batch_size']}, total={evt['total']}[/dim]"
            )
        elif e == "backlog_complete":
            console.print(
                f"[bold green]Backlog complete:[/bold green] "
                f"{evt.get('claims_processed', 0)}/{evt.get('total_claims', 0)} claims, "
                f"{evt.get('contradictions_found', 0)} contradictions, "
                f"{evt.get('links_found', 0)} links, "
                f"{evt.get('open_loops_found', 0)} open loops"
            )
    ns.on_progress = _progress

    async def _run():
        summary = await ns.run_backlog(batch_size=batch_size, reset=reset)
        console.print(f"[green]Backlog sweep done![/green]")
        console.print(f"  Claims processed: {summary.get('claims_processed', 0)}")
        console.print(f"  Batches: {summary.get('batches', 0)}")
        console.print(f"  Contradictions: {summary.get('contradictions_found', 0)}")
        console.print(f"  Links: {summary.get('links_found', 0)}")
        console.print(f"  Open loops: {summary.get('open_loops_found', 0)}")
        await hm.shutdown()

    asyncio.run(_run())
