"""Result models and step context for the autonomous executor."""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


@dataclass
class StepExecutionResult:
    """Result from executing a single step."""

    step_id: str
    step_goal: str
    action_type: str
    success: bool
    output: str
    cost_usd: float = 0.0
    duration_secs: float = 0.0
    quality_score: float = 0.0
    quality_feedback: str = ""
    attempt_number: int = 1
    session_id: str = ""
    error: str = ""


@dataclass
class ExecutionReport:
    """Full execution report across all steps."""

    goal: str
    strategy: str
    total_steps: int
    completed_steps: int
    failed_steps: int
    skipped_steps: int
    total_cost_usd: float
    total_duration_secs: float
    step_results: list[StepExecutionResult] = field(default_factory=list)
    layers: list[list[str]] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.failed_steps == 0 and self.completed_steps == self.total_steps

    def summary(self) -> str:
        status = "SUCCESS" if self.success else "PARTIAL"
        lines = [
            f"Execution {status}: {self.completed_steps}/{self.total_steps} steps",
            f"Cost: ${self.total_cost_usd:.2f} | Duration: {self.total_duration_secs:.0f}s",
        ]
        if self.failed_steps:
            lines.append(f"Failed: {self.failed_steps} steps")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step context
# ---------------------------------------------------------------------------


@dataclass
class StepContext:
    """Carries contextual information for step execution."""

    goal: str
    plan_summary: str
    step_outputs: dict[str, str] = field(default_factory=dict)  # step_id -> output
    knowledge_claims: list[str] = field(default_factory=list)
    work_dir: str = ""
    max_context_chars: int = 2000

    def build_prompt(self, step_id: str, step_goal: str, action_type: str,
                     target_files: list[str], dependencies: list[str]) -> str:
        """Build the full prompt for a step, including dependency context."""
        from .strategies import ROLE_PROMPTS, DEFAULT_ROLE_PROMPT

        role_prompt = ROLE_PROMPTS.get(action_type, DEFAULT_ROLE_PROMPT)

        parts = [role_prompt]

        # Overall goal
        parts.append(f"\n## Overall Goal\n{self.goal}\n")

        # Plan summary (truncated)
        if self.plan_summary:
            summary = self.plan_summary[:1000]
            parts.append(f"## Plan Summary\n{summary}\n")

        # Dependency outputs (only direct dependencies, truncated)
        dep_context = self._dependency_context(dependencies)
        if dep_context:
            parts.append(f"## Context from Previous Steps\n{dep_context}\n")

        # Knowledge claims
        if self.knowledge_claims:
            claims_text = "\n".join(f"- {c}" for c in self.knowledge_claims[:10])
            parts.append(f"## Relevant Knowledge\n{claims_text}\n")

        # Target files
        if target_files:
            files_text = "\n".join(f"- {f}" for f in target_files)
            parts.append(f"## Target Files\n{files_text}\n")

        # The actual task
        parts.append(f"## Your Task\n{step_goal}\n")

        return "\n".join(parts)

    def _dependency_context(self, dependencies: list[str]) -> str:
        """Get truncated outputs from dependency steps."""
        if not dependencies:
            return ""
        parts = []
        budget = self.max_context_chars
        for dep_id in dependencies:
            output = self.step_outputs.get(dep_id, "")
            if not output:
                continue
            truncated = output[:budget]
            if len(output) > budget:
                truncated += "\n... (truncated)"
            parts.append(f"### Step {dep_id} output:\n{truncated}")
            budget -= len(truncated)
            if budget <= 0:
                break
        return "\n\n".join(parts)
