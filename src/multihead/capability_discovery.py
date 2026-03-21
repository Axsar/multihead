"""Capability discovery: query local registry + knowledge base for "who can handle X?"

Provides unified interface to discover capabilities from:
1. Local registry (acp_state.json) - agent capabilities
2. Knowledge base - model-specific performance data (YOLO, SAM2, UNet, etc.)
3. Optional: BotVibes marketplace (future)

Gap #3 fix: Enable dynamic capability lookup instead of static config.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CapabilityMatch:
    """A discovered capability that can handle a request."""

    # Core identity
    agent_id: str
    capability_id: str

    # Source of this capability
    source: str  # "local" | "knowledge" | "botvibes"

    # Metadata for scoring/routing
    kind: str = "llm"  # llm | vlm | tool | model
    model: str = ""  # e.g. "Qwen/Qwen3-8B" or "YOLOv8m"

    # Performance metrics (from knowledge base or ACP state)
    performance: dict[str, Any] | None = None  # e.g. {"mAP50": 0.9185, "iou": 0.9958}
    latency_p50_ms: int = 0
    latency_p95_ms: int = 0
    cost_per_call: float = 0.0

    # Availability
    available: bool = True

    # Confidence in this match
    match_score: float = 1.0  # 0.0-1.0, how well does this match the query?

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "capability_id": self.capability_id,
            "source": self.source,
            "kind": self.kind,
            "model": self.model,
            "performance": self.performance,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "cost_per_call": self.cost_per_call,
            "available": self.available,
            "match_score": self.match_score,
        }


class CapabilityDiscovery:
    """Discover capabilities from local registry and knowledge base.

    Usage:
        discovery = CapabilityDiscovery(data_dir="~/.multihead")

        # Query by capability ID (prefix or exact)
        matches = discovery.query("com.multihead.llm")

        # Query by task/model name (semantic search in knowledge base)
        matches = discovery.query("YOLO detection")

        # Query by kind
        matches = discovery.query_by_kind("vlm")
    """

    def __init__(
        self,
        data_dir: Path | str,
        knowledge_db_path: Path | str | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.knowledge_db_path = Path(knowledge_db_path) if knowledge_db_path else self.data_dir / "knowledge.db"

        # Cache loaded state
        self._acp_state: dict[str, Any] | None = None
        self._last_reload: float = 0.0

    def reload(self) -> None:
        """Reload capability data from files."""
        import time
        self._acp_state = self._load_acp_state()
        self._last_reload = time.time()
        logger.info(
            "Capability discovery reloaded: %d local capabilities",
            len(self._acp_state.get("capabilities", {}).get("capabilities", [])) if self._acp_state else 0,
        )

    def _load_acp_state(self) -> dict[str, Any] | None:
        """Load local capability registry from acp_state.json."""
        state_file = self.data_dir / "acp_state.json"
        if not state_file.exists():
            logger.warning("acp_state.json not found at %s", state_file)
            return None

        try:
            return json.loads(state_file.read_text())
        except Exception as e:
            logger.error("Failed to load acp_state.json: %s", e)
            return None

    def query(
        self,
        capability_query: str,
        *,
        exact_match: bool = False,
        limit: int = 20,
    ) -> list[CapabilityMatch]:
        """Query for capabilities matching the query string.

        Args:
            capability_query: Capability ID (e.g. "com.multihead.llm"),
                             prefix (e.g. "com.multihead.*"),
                             or semantic query (e.g. "YOLO detection")
            exact_match: Require exact capability ID match (no prefix matching)
            limit: Maximum results to return

        Returns:
            List of CapabilityMatch objects, sorted by match_score descending
        """
        # Auto-reload if stale (older than 60s)
        import time
        if not self._acp_state or (time.time() - self._last_reload > 60):
            self.reload()

        matches: list[CapabilityMatch] = []

        # 1. Query local registry (acp_state.json)
        matches.extend(self._query_local_registry(capability_query, exact_match))

        # 2. Query knowledge base for model-specific capabilities
        matches.extend(self._query_knowledge_base(capability_query))

        # Sort by match score (descending) and limit
        matches.sort(key=lambda m: m.match_score, reverse=True)
        return matches[:limit]

    def query_by_kind(self, kind: str, limit: int = 20) -> list[CapabilityMatch]:
        """Query capabilities by kind (llm, vlm, tool, model).

        Args:
            kind: Kind to filter by
            limit: Maximum results

        Returns:
            List of CapabilityMatch objects
        """
        import time
        if not self._acp_state or (time.time() - self._last_reload > 60):
            self.reload()

        matches: list[CapabilityMatch] = []

        # Query local registry filtered by kind
        if self._acp_state:
            agent_id = self._acp_state.get("agent_id", "multihead-agent")
            caps = self._acp_state.get("capabilities", {})
            latency = caps.get("latency_profile", {})
            cost = caps.get("cost_model", {})

            for cap_id in caps.get("capabilities", []):
                # Parse kind from capability ID: com.multihead.{kind}.{head_id}
                parts = cap_id.split(".")
                cap_kind = parts[2] if len(parts) >= 3 else "llm"

                if cap_kind == kind:
                    matches.append(CapabilityMatch(
                        agent_id=agent_id,
                        capability_id=cap_id,
                        source="local",
                        kind=cap_kind,
                        model=self._acp_state.get("heads", {}).get(parts[3] if len(parts) >= 4 else "", ""),
                        latency_p50_ms=latency.get("p50_ms", 0),
                        latency_p95_ms=latency.get("p95_ms", 0),
                        cost_per_call=cost.get("price", 0.0),
                        match_score=1.0,
                    ))

        # Also query knowledge base for model-specific matches
        kb_matches = self._query_knowledge_base(kind)
        matches.extend([m for m in kb_matches if m.kind == kind])

        matches.sort(key=lambda m: m.match_score, reverse=True)
        return matches[:limit]

    def _query_local_registry(
        self,
        query: str,
        exact_match: bool = False,
    ) -> list[CapabilityMatch]:
        """Query local acp_state.json for matching capabilities."""
        matches: list[CapabilityMatch] = []

        if not self._acp_state:
            return matches

        agent_id = self._acp_state.get("agent_id", "multihead-agent")
        caps = self._acp_state.get("capabilities", {})
        latency = caps.get("latency_profile", {})
        cost = caps.get("cost_model", {})

        for cap_id in caps.get("capabilities", []):
            # Match logic
            if exact_match:
                if cap_id != query:
                    continue
                score = 1.0
            else:
                # Prefix match (com.multihead.llm matches com.multihead.llm.qwen-llm)
                if query.endswith("*"):
                    prefix = query[:-1]
                    if not cap_id.startswith(prefix):
                        continue
                    score = 0.9
                elif cap_id.startswith(query):
                    score = 0.8
                elif query in cap_id:
                    score = 0.5
                else:
                    continue

            # Parse kind from capability ID: com.multihead.{kind}.{head_id}
            parts = cap_id.split(".")
            kind = parts[2] if len(parts) >= 3 else "llm"
            head_id = parts[3] if len(parts) >= 4 else ""

            matches.append(CapabilityMatch(
                agent_id=agent_id,
                capability_id=cap_id,
                source="local",
                kind=kind,
                model=self._acp_state.get("heads", {}).get(head_id, ""),
                latency_p50_ms=latency.get("p50_ms", 0),
                latency_p95_ms=latency.get("p95_ms", 0),
                cost_per_call=cost.get("price", 0.0),
                match_score=score,
            ))

        # Also check Claude capabilities if present
        claude_caps = self._acp_state.get("claude_capabilities", {})
        claude_agent_id = self._acp_state.get("claude_agent_id", "claude-session-agent")
        claude_latency = claude_caps.get("latency_profile", {})
        claude_cost = claude_caps.get("cost_model", {})

        for cap_id in claude_caps.get("capabilities", []):
            if exact_match:
                if cap_id != query:
                    continue
                score = 1.0
            else:
                if query.endswith("*"):
                    prefix = query[:-1]
                    if not cap_id.startswith(prefix):
                        continue
                    score = 0.9
                elif cap_id.startswith(query):
                    score = 0.8
                elif query in cap_id:
                    score = 0.5
                else:
                    continue

            matches.append(CapabilityMatch(
                agent_id=claude_agent_id,
                capability_id=cap_id,
                source="local",
                kind="tool",
                model="Claude Code",
                latency_p50_ms=claude_latency.get("p50_ms", 0),
                latency_p95_ms=claude_latency.get("p95_ms", 0),
                cost_per_call=claude_cost.get("price", 0.0),
                match_score=score,
            ))

        return matches

    def _query_knowledge_base(self, query: str) -> list[CapabilityMatch]:
        """Query knowledge base for model-specific capabilities.

        Searches for claims about model performance (YOLO, SAM2, UNet, etc.)
        """
        matches: list[CapabilityMatch] = []

        if not self.knowledge_db_path.exists():
            return matches

        try:
            import sqlite3
            conn = sqlite3.connect(str(self.knowledge_db_path), timeout=10.0)
            conn.row_factory = sqlite3.Row

            # Search for model-specific claims (YOLO, SAM2, UNet, etc.)
            # Look in statement field for model names and performance data
            cursor = conn.execute("""
                SELECT claim_id, statement, claim_type, subject_json, object_json
                FROM claims
                WHERE claim_status = 'accepted'
                  AND (
                    statement LIKE ? OR
                    statement LIKE '%YOLO%' OR
                    statement LIKE '%SAM2%' OR
                    statement LIKE '%UNet%' OR
                    statement LIKE '%model%' OR
                    statement LIKE '%capability%'
                  )
                LIMIT 50
            """, (f"%{query}%",))

            for row in cursor.fetchall():
                statement = row["statement"]

                # Extract model info from statement
                model_info = self._extract_model_info(statement)
                if not model_info:
                    continue

                # Create capability match from model claim
                matches.append(CapabilityMatch(
                    agent_id="multihead-agent",  # Assume local for now
                    capability_id=f"com.multihead.model.{model_info['name'].lower().replace(' ', '-')}",
                    source="knowledge",
                    kind="model",
                    model=model_info["name"],
                    performance=model_info.get("performance"),
                    latency_p50_ms=model_info.get("latency_ms", 0),
                    match_score=self._calculate_semantic_score(query, statement),
                ))

            conn.close()

        except Exception as e:
            logger.warning("Failed to query knowledge base: %s", e)

        return matches

    def _extract_model_info(self, statement: str) -> dict[str, Any] | None:
        """Extract model name and performance metrics from claim statement."""
        models = {
            "YOLO": {"name": "YOLOv8m", "task": "detection"},
            "SAM2": {"name": "SAM2", "task": "segmentation"},
            "UNet": {"name": "UNet", "task": "segmentation"},
        }

        for model_key, model_data in models.items():
            if model_key in statement:
                info = {"name": model_data["name"], "task": model_data["task"]}

                # Extract performance metrics using regex
                # mAP: 91.85% mAP50
                mAP_match = re.search(r"(\d+\.?\d*)%?\s*mAP", statement)
                if mAP_match:
                    info.setdefault("performance", {})["mAP50"] = float(mAP_match.group(1)) / 100.0

                # IoU: 99.58% IoU
                iou_match = re.search(r"(\d+\.?\d*)%?\s*IoU", statement)
                if iou_match:
                    info.setdefault("performance", {})["iou"] = float(iou_match.group(1)) / 100.0

                # Latency: 10.5s/page, 0.9s/page
                latency_match = re.search(r"(\d+\.?\d*)s/page", statement)
                if latency_match:
                    info["latency_ms"] = int(float(latency_match.group(1)) * 1000)

                return info

        return None

    def _calculate_semantic_score(self, query: str, statement: str) -> float:
        """Calculate semantic match score between query and statement.

        Simple keyword overlap for now. Could use embeddings later.
        """
        query_lower = query.lower()
        statement_lower = statement.lower()

        # Exact substring match
        if query_lower in statement_lower:
            return 0.9

        # Keyword overlap
        query_words = set(re.findall(r'\w+', query_lower))
        statement_words = set(re.findall(r'\w+', statement_lower))
        overlap = query_words & statement_words

        if not query_words:
            return 0.0

        overlap_ratio = len(overlap) / len(query_words)
        return overlap_ratio * 0.7  # Max 0.7 for keyword match


def discover_capabilities(
    query: str,
    data_dir: Path | str | None = None,
    **kwargs,
) -> list[CapabilityMatch]:
    """Convenience function to discover capabilities.

    Args:
        query: Capability query (ID, prefix, or semantic)
        data_dir: Path to MultiHead data directory (defaults to MULTIHEAD_DATA_DIR or ~/.multihead)
        **kwargs: Additional args for CapabilityDiscovery.query()

    Returns:
        List of CapabilityMatch objects
    """
    if data_dir is None:
        data_dir = os.environ.get("MULTIHEAD_DATA_DIR", str(Path.home() / ".multihead"))
    discovery = CapabilityDiscovery(data_dir)
    return discovery.query(query, **kwargs)
