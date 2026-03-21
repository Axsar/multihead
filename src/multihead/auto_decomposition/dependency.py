"""Dependency inference for decomposed task steps.

Analyzes step descriptions and metadata to infer DAG dependencies,
enabling parallel execution of independent steps.
"""

from __future__ import annotations

import re

from multihead.decomposer import TaskNode


class StepDependencyAnalyzer:
    """Infers DAG dependencies from step descriptions and metadata.

    Analyzes:
    - File dependencies (step A reads -> step B writes same file)
    - Output/input chains (step A produces artifact -> step B consumes)
    - Action ordering (test must follow edit, verify must follow create)
    """

    # Action types that must follow others
    ACTION_ORDER = {
        "test": ["edit", "create", "refactor"],  # Test after code changes
        "verify": ["edit", "create", "test"],  # Verify after changes/tests
        "refactor": ["test"],  # Refactor after tests confirm behavior
    }

    def __init__(self):
        self._file_reads: dict[str, list[str]] = {}  # file -> [step_ids]
        self._file_writes: dict[str, list[str]] = {}  # file -> [step_ids]
        self._outputs: dict[str, str] = {}  # artifact_name -> step_id

    def infer_dependencies(self, leaves: list[TaskNode]) -> dict[str, list[str]]:
        """Infer dependencies for all leaf nodes.

        Args:
            leaves: List of leaf TaskNodes in execution order

        Returns:
            Dict mapping step_id -> list[step_id] dependencies
        """
        self._file_reads.clear()
        self._file_writes.clear()
        self._outputs.clear()

        # First pass: gather all operations in order
        file_operations: list[tuple[str, str, str]] = []  # (step_id, action, file)
        for leaf in leaves:
            for file_path in leaf.target_files:
                if leaf.action_type in ("edit", "create", "delete", "refactor"):
                    file_operations.append((leaf.id, "write", file_path))
                elif leaf.action_type in ("read", "explore"):
                    file_operations.append((leaf.id, "read", file_path))

        # Second pass: build dependency graph
        deps: dict[str, list[str]] = {leaf.id: [] for leaf in leaves}

        for i, leaf in enumerate(leaves):
            # Rule 1: File write-after-read dependencies
            for file_path in leaf.target_files:
                if leaf.action_type in ("edit", "create", "delete", "refactor"):
                    # This step writes the file - depends on all prior reads
                    for j in range(i):
                        prior_leaf = leaves[j]
                        if (
                            file_path in prior_leaf.target_files
                            and prior_leaf.action_type in ("read", "explore")
                            and prior_leaf.id not in deps[leaf.id]
                        ):
                            deps[leaf.id].append(prior_leaf.id)

            # Rule 2: Action type ordering
            if leaf.action_type in self.ACTION_ORDER:
                required_actions = self.ACTION_ORDER[leaf.action_type]
                # Depends on most recent step of required type
                for j in range(i - 1, -1, -1):
                    prior = leaves[j]
                    if prior.action_type in required_actions:
                        if prior.id not in deps[leaf.id]:
                            deps[leaf.id].append(prior.id)
                        break  # Only depend on most recent

            # Rule 3: Artifact/output dependencies (inferred from descriptions)
            outputs = self._extract_artifact_names(leaf.expected_output)
            for out in outputs:
                self._outputs[out] = leaf.id

            inputs = self._extract_artifact_names(leaf.goal)
            for inp in inputs:
                if inp in self._outputs:
                    producer_id = self._outputs[inp]
                    if producer_id != leaf.id and producer_id not in deps[leaf.id]:
                        deps[leaf.id].append(producer_id)

        return deps


    def _extract_artifact_names(self, text: str) -> list[str]:
        """Extract likely artifact/output names from text.

        Looks for patterns like:
        - "produces X"
        - "outputs Y"
        - "generates Z"
        - "uses X from step Y"
        """
        if not text:
            return []

        patterns = [
            r"produces?\s+([a-z_][a-z0-9_]*)",
            r"outputs?\s+([a-z_][a-z0-9_]*)",
            r"generates?\s+([a-z_][a-z0-9_]*)",
            r"creates?\s+([a-z_][a-z0-9_]*)",
            r"uses?\s+([a-z_][a-z0-9_]*)",
        ]

        artifacts = []
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            artifacts.extend(matches)

        return list(set(artifacts))
