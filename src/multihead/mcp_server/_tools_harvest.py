"""Session harvester tool logic."""

from __future__ import annotations

import json

from ._core import _get_ks


async def _harvest(action: str = "status") -> str:
    """Run session harvester actions directly (no API proxy needed)."""
    try:
        from multihead.session_harvester import SessionHarvester

        ks = _get_ks()
        harvester = SessionHarvester(knowledge_store=ks)

        if action == "status":
            status = harvester.status()
            return json.dumps(status, indent=2)
        elif action == "run":
            result = harvester.harvest_all()
            return json.dumps({
                "projects_scanned": result.projects_scanned,
                "projects_harvested": result.projects_harvested,
                "projects_skipped": result.projects_skipped,
                "claims_deposited": result.claims_deposited,
                "duration_seconds": result.duration_seconds,
                "errors": result.errors,
            }, indent=2)
        elif action == "list":
            status = harvester.status()
            return json.dumps(status.get("projects", []), indent=2)
        else:
            return f"Unknown action: {action}. Use 'status', 'run', or 'list'."
    except Exception as e:
        return f"Error: {e}"
