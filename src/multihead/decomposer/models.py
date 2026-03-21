"""Decomposition data models — TaskNode, DecompositionPlan, and tree helpers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from multihead.models import StepDef, WorkOrder


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TaskNode(BaseModel):
    """A node in the decomposition tree."""

    id: str                                     # "1", "1.1", "1.1.1"
    goal: str                                   # What this achieves
    rationale: str = ""                         # Why needed
    action_type: str = ""                       # explore, read, edit, create, test, verify
    target_files: list[str] = Field(default_factory=list)
    expected_output: str = ""
    children: list[TaskNode] = Field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.depth for c in self.children)

    def leaf_count(self) -> int:
        if self.is_leaf:
            return 1
        return sum(c.leaf_count() for c in self.children)

    def leaves(self) -> list[TaskNode]:
        if self.is_leaf:
            return [self]
        result: list[TaskNode] = []
        for c in self.children:
            result.extend(c.leaves())
        return result

    def find(self, node_id: str) -> TaskNode | None:
        """Find a node by id anywhere in the subtree."""
        if self.id == node_id:
            return self
        for c in self.children:
            found = c.find(node_id)
            if found:
                return found
        return None


class DecompositionPlan(BaseModel):
    """A complete decomposition of a goal into phases and steps."""

    goal: str
    complexity: str = "moderate"                # simple, moderate, complex
    phases: list[TaskNode] = Field(default_factory=list)
    context_used: list[str] = Field(default_factory=list)

    @property
    def total_steps(self) -> int:
        return sum(p.leaf_count() for p in self.phases)

    @property
    def max_depth(self) -> int:
        if not self.phases:
            return 0
        return 1 + max(p.depth for p in self.phases)

    def all_leaves(self) -> list[TaskNode]:
        result: list[TaskNode] = []
        for p in self.phases:
            result.extend(p.leaves())
        return result

    def find(self, node_id: str) -> TaskNode | None:
        for p in self.phases:
            found = p.find(node_id)
            if found:
                return found
        return None

    def to_work_order(self) -> WorkOrder:
        """Flatten tree into an executable WorkOrder with dependency chains."""
        steps: list[StepDef] = []
        prev_id = ""
        for phase in self.phases:
            for leaf in phase.leaves():
                deps = [prev_id] if prev_id else []
                step = StepDef(
                    step_id=leaf.id,
                    name=leaf.goal,
                    head_id="",
                    required_kind="llm",
                    prompt_template=_leaf_to_prompt(leaf),
                    depends_on=deps,
                    extra={
                        "action_type": leaf.action_type,
                        "target_files": leaf.target_files,
                        "expected_output": leaf.expected_output,
                    },
                )
                steps.append(step)
                prev_id = leaf.id
        return WorkOrder(goal=self.goal, steps=steps)

    def render_tree(self) -> str:
        """Render as readable text tree."""
        lines = [f"# {self.goal}", f"Complexity: {self.complexity}", ""]
        for phase in self.phases:
            _render_node(phase, lines, indent=0)
        lines.append(f"\nTotal leaf steps: {self.total_steps}")
        lines.append(f"Max depth: {self.max_depth}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tree helper functions
# ---------------------------------------------------------------------------

def _leaf_to_prompt(leaf: TaskNode) -> str:
    parts = [leaf.goal]
    if leaf.target_files:
        parts.append(f"Files: {', '.join(leaf.target_files)}")
    if leaf.expected_output:
        parts.append(f"Expected: {leaf.expected_output}")
    return "\n".join(parts)


def _render_node(node: TaskNode, lines: list[str], indent: int) -> None:
    prefix = "  " * indent
    icon = _action_icon(node.action_type)
    files = f" -> {', '.join(node.target_files)}" if node.target_files else ""
    if node.is_leaf:
        lines.append(f"{prefix}{node.id}. {icon}{node.goal}{files}")
    else:
        lines.append(f"{prefix}{node.id}. {node.goal}")
        for child in node.children:
            _render_node(child, lines, indent + 1)


def _action_icon(action_type: str) -> str:
    icons = {
        "explore": "[explore] ",
        "read": "[read] ",
        "edit": "[edit] ",
        "create": "[create] ",
        "test": "[test] ",
        "verify": "[verify] ",
        "refactor": "[refactor] ",
        "delete": "[delete] ",
    }
    return icons.get(action_type, "")


def _trim_node(node: TaskNode, current_depth: int, max_depth: int) -> None:
    """Recursively trim tree to max_depth."""
    if current_depth >= max_depth:
        node.children = []
        return
    for child in node.children:
        _trim_node(child, current_depth + 1, max_depth)
