"""ACP agent registration and descriptor building.

Mixin class providing agent registration and capability descriptor
construction for the ACPBridge.
"""

from __future__ import annotations

from typing import Any

import httpx

from ._constants import ACP_TIMEOUT, logger


class RegistrationMixin:
    """Agent registration and capability descriptor methods."""

    # These attributes are provided by ACPBridge.__init__
    heads: Any
    skills: Any
    acp_url: str | None
    _agent_id: str
    _project_id: str
    _auth_headers: Any  # method

    def _build_multihead_descriptor(self) -> dict[str, Any]:
        """Build ACP agent descriptor from registered heads + skills."""
        capabilities = []
        input_schemas: set[str] = {"application/json", "text/plain"}
        output_schemas: set[str] = {"application/json", "text/plain"}

        # Head-based capabilities
        for head_id, state_info in self.heads.get_states().items():
            kind = state_info.get("kind", "llm")
            capabilities.append(f"com.multihead.{kind}.{head_id}")
            if kind == "vlm":
                input_schemas.update(["image/jpeg", "image/png"])

        # Skill-based capabilities (from agentskills.io SKILL.md files)
        capabilities.extend(self.skills.build_descriptor_capabilities())
        if self.skills:
            logger.info(
                "ACP descriptor includes %d skill capabilities: %s",
                len(self.skills),
                ", ".join(s.name for s in self.skills.list_skills()),
            )

        return {
            "capabilities": capabilities,
            "input_schema": sorted(input_schemas),
            "output_schema": sorted(output_schemas),
            "latency_profile": {"p50_ms": 2000, "p95_ms": 10000},
            "cost_model": {"unit": "task", "price": 0.0},
            "max_concurrency": 1,  # GPU mutex
        }

    @staticmethod
    def _build_claude_descriptor() -> dict[str, Any]:
        """Build ACP descriptor for Claude Code proxy agent."""
        return {
            "capabilities": [
                "com.claude.code.edit",
                "com.claude.code.test",
                "com.claude.code.refactor",
                "com.claude.code.review",
            ],
            "input_schema": ["application/json", "text/plain"],
            "output_schema": ["application/json", "text/plain"],
            "latency_profile": {"p50_ms": 5000, "p95_ms": 30000},
            "cost_model": {"unit": "task", "price": 0.0},
            "max_concurrency": 1,
        }

    async def _register_agents(self) -> None:
        """Register MultiHead agent identity with ACP.

        Note: Claude Code agent (claude-session-agent) is registered separately
        by the BotVibes bootstrap process. We only register MultiHead here.
        If the agent is already registered (e.g. via BotVibes dashboard login),
        a 403/409 is treated as success.
        """
        async with httpx.AsyncClient(timeout=ACP_TIMEOUT) as client:
            resp = await client.post(
                f"{self.acp_url}/agents/register",
                headers=self._auth_headers(),
                json={
                    "agent_id": self._agent_id,
                    "project_id": self._project_id,
                    "visibility": "org",
                    "descriptor": self._build_multihead_descriptor(),
                },
            )
            if resp.status_code == 409:
                logger.info(
                    "ACP agent already registered (HTTP 409), continuing: %s",
                    self._agent_id,
                )
                return
            if resp.status_code == 403:
                body = resp.text.lower()
                if "already" in body or "registered" in body:
                    logger.info(
                        "ACP agent already registered (HTTP 403), continuing: %s",
                        self._agent_id,
                    )
                    return
                # 403 for other reasons (e.g. invalid/expired API key) is a real error
                resp.raise_for_status()
            resp.raise_for_status()
            logger.info("Registered ACP agent: %s", self._agent_id)
