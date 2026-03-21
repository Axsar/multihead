"""Thought generators for Tree-of-Thoughts exploration.

Generators produce alternative reasoning paths from a given state,
enabling the search to explore multiple approaches.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ThoughtGenerator(ABC):
    """Base class for generating alternative thoughts from a state."""

    @abstractmethod
    async def generate_thoughts(
        self,
        current_state: Any,
        context: dict[str, Any],
        num_thoughts: int = 3,
    ) -> list[tuple[str, Any]]:
        """Generate alternative next thoughts from current state.

        Args:
            current_state: The current state to generate thoughts from
            context: Additional context (goal, constraints, etc.)
            num_thoughts: Number of alternative thoughts to generate

        Returns:
            List of (description, new_state) tuples
        """
        pass


class LLMThoughtGenerator(ThoughtGenerator):
    """Generates alternative thoughts using an LLM."""

    def __init__(self, generate_func: Callable[[str], Any]):
        """Initialize with an LLM generate function.

        Args:
            generate_func: Async function that takes a prompt and returns text
        """
        self.generate = generate_func

    async def generate_thoughts(
        self,
        current_state: Any,
        context: dict[str, Any],
        num_thoughts: int = 3,
    ) -> list[tuple[str, Any]]:
        """Generate alternative thoughts using LLM prompting.

        Args:
            current_state: Current state/output
            context: Context with goal, constraints
            num_thoughts: Number of alternatives

        Returns:
            List of (description, new_state) tuples
        """
        goal = context.get("goal", "solve the problem")
        current_desc = str(current_state) if current_state else "start"

        prompt = f"""Given the goal: {goal}

Current state: {current_desc}

Generate {num_thoughts} different alternative next steps or approaches.
Each alternative should be a distinct way to make progress.

Format your response as:
1. [Brief description of approach 1]
2. [Brief description of approach 2]
3. [Brief description of approach 3]

Focus on diverse alternatives, not variations of the same idea."""

        response = await self.generate(prompt)

        # Parse numbered alternatives from response
        thoughts = []
        lines = str(response).split("\n")
        for line in lines:
            line = line.strip()
            # Match numbered items like "1. " or "1) " or "- "
            if line and (line[0].isdigit() or line.startswith("-")):
                # Extract the description after the number/bullet
                if line.startswith("-"):
                    # Bullet point: remove the "-" and strip
                    description = line[1:].strip()
                    if description:
                        thoughts.append((description, description))
                else:
                    # Numbered item: split on "." or ")"
                    parts = line.split(".", 1) if "." in line else line.split(")", 1)
                    if len(parts) > 1:
                        description = parts[1].strip()
                        if description:
                            thoughts.append((description, description))

        # If parsing failed, return the full response as a single thought
        if not thoughts:
            thoughts = [(str(response), response)]

        return thoughts[:num_thoughts]
