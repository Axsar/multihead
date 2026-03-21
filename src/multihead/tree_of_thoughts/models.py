"""Core data models for Tree-of-Thoughts exploration.

Contains the SearchStrategy enum and ThoughtNode dataclass that form
the foundation of the ToT reasoning tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class SearchStrategy(str, Enum):
    """Search strategy for tree exploration."""

    BFS = "bfs"  # Breadth-first: explore all alternatives at each level
    DFS = "dfs"  # Depth-first: follow one path to completion before backtracking
    BEAM = "beam"  # Beam search: keep top-k most promising paths


@dataclass
class ThoughtNode:
    """Represents a state in the reasoning tree.

    Each node contains:
    - The current state/output
    - Link to parent node
    - List of child alternatives
    - Evaluation score for this state
    - Metadata about the thought
    """

    node_id: str
    state: Any  # The current state/output at this node
    step_description: str  # What this thought represents
    parent: ThoughtNode | None = None
    children: list[ThoughtNode] = field(default_factory=list)
    evaluation_score: float = 0.0  # How promising this state is (0-1)
    depth: int = 0
    is_terminal: bool = False  # Whether this is a final answer
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add_child(self, child: ThoughtNode) -> None:
        """Add a child node to this thought."""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

    def get_path(self) -> list[ThoughtNode]:
        """Get the path from root to this node."""
        path = []
        current = self
        while current is not None:
            path.insert(0, current)
            current = current.parent
        return path

    def get_path_description(self) -> str:
        """Get a text description of the path to this node."""
        path = self.get_path()
        return " → ".join(node.step_description for node in path)
