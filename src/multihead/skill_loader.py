"""Skill loader: reads agentskills.io SKILL.md files and makes them available.

Scans a directory for skill folders (each containing a SKILL.md), parses
YAML frontmatter + markdown body, and provides them for ACP capability
registration and task execution.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ACP capability prefix for skill-based capabilities
SKILL_CAPABILITY_PREFIX = "com.multihead.skill"


@dataclass
class Skill:
    """A parsed agentskills.io skill."""

    name: str
    description: str
    instructions: str  # markdown body — the actual expertise
    license: str = ""
    compatibility: str = ""
    metadata: dict[str, str] = field(default_factory=dict)
    allowed_tools: list[str] = field(default_factory=list)
    # Path to skill directory (for loading references/scripts)
    skill_dir: Path | None = None

    @property
    def capability_id(self) -> str:
        """ACP capability ID for this skill."""
        return f"{SKILL_CAPABILITY_PREFIX}.{self.name}"

    def load_reference(self, filename: str) -> str | None:
        """Load a reference file from the skill directory."""
        if not self.skill_dir:
            return None
        # Check references/ subdir first, then root
        for path in [
            self.skill_dir / "references" / filename,
            self.skill_dir / "reference" / filename,
            self.skill_dir / filename,
        ]:
            if path.is_file():
                return path.read_text(encoding="utf-8")
        return None

    def build_system_prompt(self) -> str:
        """Build a system prompt from the skill instructions.

        Returns the skill body as-is — it's already written as instructions
        for an AI agent. We wrap it with minimal context about the execution
        environment.
        """
        parts = [
            f"# Skill: {self.name}",
            f"_{self.description}_",
            "",
            self.instructions,
        ]
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API responses."""
        return {
            "name": self.name,
            "description": self.description,
            "capability_id": self.capability_id,
            "license": self.license,
            "compatibility": self.compatibility,
            "metadata": self.metadata,
            "allowed_tools": self.allowed_tools,
            "instruction_lines": self.instructions.count("\n") + 1,
            "has_references": bool(
                self.skill_dir
                and any(
                    (self.skill_dir / d).is_dir()
                    for d in ("references", "reference", "scripts")
                )
            ),
        }


def parse_skill_md(text: str) -> tuple[dict[str, Any], str]:
    """Parse a SKILL.md file into frontmatter dict and body string."""
    # Match YAML frontmatter between --- delimiters
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
    if not match:
        raise ValueError("SKILL.md must start with YAML frontmatter (--- delimiters)")
    frontmatter_str, body = match.groups()
    frontmatter = yaml.safe_load(frontmatter_str) or {}
    return frontmatter, body.strip()


def load_skill(skill_dir: Path) -> Skill:
    """Load a single skill from a directory containing SKILL.md."""
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        raise FileNotFoundError(f"No SKILL.md in {skill_dir}")

    text = skill_md.read_text(encoding="utf-8")
    frontmatter, body = parse_skill_md(text)

    name = frontmatter.get("name", "")
    if not name:
        raise ValueError(f"Skill in {skill_dir} has no 'name' in frontmatter")

    description = frontmatter.get("description", "")
    if not description:
        raise ValueError(f"Skill '{name}' has no 'description' in frontmatter")

    # Parse allowed-tools (space-delimited string)
    allowed_tools_raw = frontmatter.get("allowed-tools", "")
    allowed_tools = allowed_tools_raw.split() if allowed_tools_raw else []

    return Skill(
        name=name,
        description=description,
        instructions=body,
        license=frontmatter.get("license", ""),
        compatibility=frontmatter.get("compatibility", ""),
        metadata=frontmatter.get("metadata", {}),
        allowed_tools=allowed_tools,
        skill_dir=skill_dir,
    )


class SkillRegistry:
    """Registry of loaded skills, indexed by name and capability ID."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def load_directory(self, skills_dir: Path) -> int:
        """Scan a directory for skill folders and load them all.

        Returns the number of skills loaded.
        """
        if not skills_dir.is_dir():
            logger.warning("Skills directory does not exist: %s", skills_dir)
            return 0

        loaded = 0
        for entry in sorted(skills_dir.iterdir()):
            if not entry.is_dir():
                continue
            skill_md = entry / "SKILL.md"
            if not skill_md.is_file():
                continue
            try:
                skill = load_skill(entry)
                self._skills[skill.name] = skill
                loaded += 1
                logger.info(
                    "Loaded skill: %s (%s) — %d lines",
                    skill.name,
                    skill.capability_id,
                    skill.instructions.count("\n") + 1,
                )
            except Exception as e:
                logger.warning("Failed to load skill from %s: %s", entry, e)

        logger.info("Skill registry: %d skills loaded from %s", loaded, skills_dir)
        return loaded

    def get(self, name: str) -> Skill | None:
        """Get a skill by name."""
        return self._skills.get(name)

    def get_by_capability(self, capability_id: str) -> Skill | None:
        """Get a skill by its ACP capability ID."""
        for skill in self._skills.values():
            if skill.capability_id == capability_id:
                return skill
        return None

    def find_by_capability_prefix(self, capability: str) -> Skill | None:
        """Find a skill whose capability ID matches (prefix or exact)."""
        # Exact match first
        skill = self.get_by_capability(capability)
        if skill:
            return skill
        # Check if capability starts with our prefix
        if capability.startswith(SKILL_CAPABILITY_PREFIX + "."):
            skill_name = capability[len(SKILL_CAPABILITY_PREFIX) + 1 :]
            return self._skills.get(skill_name)
        return None

    def list_skills(self) -> list[Skill]:
        """List all loaded skills."""
        return list(self._skills.values())

    def capability_ids(self) -> list[str]:
        """List all ACP capability IDs."""
        return [s.capability_id for s in self._skills.values()]

    def build_descriptor_capabilities(self) -> list[str]:
        """Build capability list for ACP agent descriptor."""
        return self.capability_ids()

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills
