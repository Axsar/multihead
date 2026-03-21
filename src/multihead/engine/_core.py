"""Core Engine class — lifecycle, routing, heads, generation, properties."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..config import Settings, load_heads
from ..event_store import EventStore
from ..head_manager import HeadManager
from ..knowledge_store import KnowledgeStore
from ..models import HeadManifest, WorkOrder
from ..orchestrator import Orchestrator
from ..router import Router

from ._marketplace import _MarketplaceMixin
from ._solve import _SolveMixin

logger = logging.getLogger(__name__)


class Engine(_SolveMixin, _MarketplaceMixin):
    """MultiHead as an embeddable engine.

    Provides direct Python access to MultiHead's core capabilities:
    - GPU model management (wake, sleep, swap, generate)
    - Intelligent routing (by kind, capability, or mesh)
    - Knowledge store (claims, events, institutional memory)
    - Orchestrator (durable step-by-step execution)

    All components are lazily initialized on ``start()``.
    """

    def __init__(
        self,
        config_dir: str | Path = "config",
        data_dir: str | Path | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if data_dir:
            kwargs["data_dir"] = Path(data_dir)

        self.settings = Settings(config_dir=Path(config_dir), **kwargs)
        self._head_manager: HeadManager | None = None
        self._router: Router | None = None
        self._knowledge_store: KnowledgeStore | None = None
        self._orchestrator: Orchestrator | None = None
        self._heads_config: dict[str, HeadManifest] = {}
        self._started = False

    async def start(self) -> None:
        """Initialize all components."""
        if self._started:
            return

        self.settings.ensure_dirs()

        # Load head manifests
        self._heads_config = load_heads(self.settings.config_dir)

        # Knowledge store
        self._knowledge_store = KnowledgeStore(self.settings.knowledge_db_path)

        # Head manager
        self._head_manager = HeadManager(
            self._heads_config,
            knowledge_store=self._knowledge_store,
        )

        # Router
        self._router = Router(head_manager=self._head_manager)

        # Event store + Orchestrator
        from ..artifact_store import ArtifactStore

        artifact_store = ArtifactStore(
            self.settings.artifacts_dir, self.settings.db_path,
        )
        event_store = EventStore(self.settings.runs_dir, self.settings.db_path)
        self._orchestrator = Orchestrator(
            event_store, artifact_store, self._head_manager, self.settings.runs_dir,
        )

        self._started = True
        logger.info(
            "Engine started: %d heads, knowledge at %s",
            len(self._heads_config),
            self.settings.knowledge_db_path,
        )

    async def stop(self) -> None:
        """Graceful shutdown — unload models, close connections."""
        if not self._started:
            return
        if self._head_manager:
            await self._head_manager.shutdown()
        self._started = False
        logger.info("Engine stopped.")

    # -------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        kind: str = "llm",
        head_id: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate text through a model head.

        Args:
            prompt: The input text.
            kind: Head kind to route to (llm, vlm, embed).
                  Ignored if head_id is provided.
            head_id: Specific head to use. If empty, routes by kind.
            **kwargs: Passed to adapter.generate() (temperature, max_tokens, etc.)

        Returns:
            dict with at least "text" key.
        """
        self._ensure_started()
        hid = head_id or self.route(kind)
        if not hid:
            raise RuntimeError(f"No head available for kind='{kind}'")
        return await self._head_manager.generate(hid, prompt, **kwargs)

    async def chat(
        self,
        messages: list[dict[str, str]],
        kind: str = "llm",
        head_id: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Multi-turn chat through a model head.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": "..."}.
            kind: Head kind to route to.
            head_id: Specific head to use.
            **kwargs: Passed to adapter.chat().

        Returns:
            dict with at least "text" key.
        """
        self._ensure_started()
        hid = head_id or self.route(kind)
        if not hid:
            raise RuntimeError(f"No head available for kind='{kind}'")
        await self._head_manager.ensure_active(hid)
        adapter = self._head_manager.get_adapter(hid)
        if hasattr(adapter, "chat"):
            return await adapter.chat(messages, **kwargs)
        # Fallback: flatten messages to prompt
        prompt = "\n".join(f"[{m['role']}] {m['content']}" for m in messages)
        return await self._head_manager.generate(hid, prompt, **kwargs)

    # -------------------------------------------------------------------
    # Routing
    # -------------------------------------------------------------------

    def route(self, kind: str = "llm") -> str | None:
        """Route to the best head for a given kind.

        Returns head_id or None if no suitable head found.
        """
        self._ensure_started()
        return self._router.route(kind)

    def route_mesh(self, kind: str = "llm") -> str | None:
        """Route across local heads and mesh peers."""
        self._ensure_started()
        return self._router.route_mesh(kind)

    # -------------------------------------------------------------------
    # Head Management
    # -------------------------------------------------------------------

    async def wake(self, head_id: str) -> None:
        """Wake a head (load to GPU)."""
        self._ensure_started()
        await self._head_manager.wake_head(head_id)

    async def sleep(self, head_id: str) -> None:
        """Put a head to sleep (offload from GPU)."""
        self._ensure_started()
        await self._head_manager.sleep_head(head_id)

    async def swap(self, head_id: str) -> None:
        """Switch the active head (GPU mutex — unloads current, loads new)."""
        self._ensure_started()
        await self._head_manager.ensure_active(head_id)

    def get_states(self) -> dict[str, dict[str, Any]]:
        """Get all head states."""
        self._ensure_started()
        return self._head_manager.get_states()

    # -------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------

    @property
    def head_manager(self) -> HeadManager:
        """Direct access to HeadManager for advanced usage."""
        self._ensure_started()
        return self._head_manager

    @property
    def knowledge(self) -> KnowledgeStore:
        """Direct access to KnowledgeStore."""
        self._ensure_started()
        return self._knowledge_store

    @property
    def router(self) -> Router:
        """Direct access to Router."""
        self._ensure_started()
        return self._router

    @property
    def orchestrator(self) -> Orchestrator:
        """Direct access to Orchestrator for running work orders."""
        self._ensure_started()
        return self._orchestrator

    @property
    def heads(self) -> dict[str, HeadManifest]:
        """Registered head manifests."""
        return self._heads_config

    # -------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------

    def _ensure_started(self) -> None:
        if not self._started:
            raise RuntimeError("Engine not started. Call await engine.start() first.")
