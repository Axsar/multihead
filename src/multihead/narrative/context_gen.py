"""Generate daemon context markdown from the KnowledgeStore.

Produces a `daemon_narrative.md` file containing accepted claims and
confirmed events, which the Claude worker daemon prepends to headless
session prompts alongside the static `daemon_briefing.md`.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from multihead.knowledge_store import KnowledgeStore

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_DIR: Path | None = None  # resolved from Settings.data_dir at call time


def generate_daemon_context(
    knowledge_store: KnowledgeStore,
    output_path: Path | None = None,
    max_claims: int = 50,
    max_events: int = 30,
    data_dir: Path | None = None,
) -> Path:
    """Generate narrative context markdown from accepted claims + confirmed events.

    Args:
        knowledge_store: KnowledgeStore to query.
        output_path: Where to write the markdown. Defaults to
            ``<data_dir>/context/daemon_narrative.md``.
        max_claims: Maximum accepted claims to include.
        max_events: Maximum confirmed events to include.
        data_dir: Data directory root. Falls back to $MULTIHEAD_DATA_DIR
            or ``~/.multihead``.

    Returns:
        Path to the written file.
    """
    if output_path is None:
        if data_dir is None:
            data_dir = Path(
                os.environ.get("MULTIHEAD_DATA_DIR", "")
            ) if os.environ.get("MULTIHEAD_DATA_DIR") else Path.home() / ".multihead"
        output_path = data_dir / "context" / "daemon_narrative.md"

    claims = knowledge_store.get_accepted_claims_for_pack(limit=max_claims)
    events = knowledge_store.get_confirmed_events_for_pack(limit=max_events)

    sections: list[str] = [
        "# Narrative Context (auto-generated)",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]

    if claims:
        sections.append("## Key Knowledge Claims")
        sections.append("")
        for claim in claims:
            conf = f"{claim.confidence:.2f}"
            sections.append(f"- [{conf}] {claim.statement}")
        sections.append("")

    if events:
        sections.append("## Recent Events")
        sections.append("")
        for event in events:
            ts = event.time.happened_at.strftime("%Y-%m-%d") if event.time else "?"
            sections.append(f"- [{ts}] {event.title}")
        sections.append("")

    if not claims and not events:
        sections.append("_No narrative data yet. Run `multihead narrative ingest` to populate._")
        sections.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections), encoding="utf-8")

    logger.info(
        "Daemon context written: %d claims, %d events → %s",
        len(claims), len(events), output_path,
    )
    return output_path
