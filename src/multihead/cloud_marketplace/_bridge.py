"""CloudMarketplaceBridge — main class composing all mixin functionality.

Seller-side cloud marketplace bridge connecting to BotVibes.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from ._auth import AuthMixin
from ._constants import logger
from ._contracts import ContractMixin
from ._fulfillment import FulfillmentMixin
from ._listings import ListingsMixin
from ._rfq import RFQMixin
from ._vault import VaultMixin

from ..config import Settings
from ..head_manager import HeadManager


class CloudMarketplaceBridge(
    AuthMixin,
    RFQMixin,
    ContractMixin,
    FulfillmentMixin,
    VaultMixin,
    ListingsMixin,
):
    """Seller-side cloud marketplace bridge.

    Three background loops:
    1. Cloud ACPBridge — heartbeat + task polling for direct tasks
    2. RFQ Scanner — browse open RFQs, auto-quote on matching ones
    3. Contract Monitor — track active contracts, execute, post receipts
    """

    def __init__(
        self,
        head_manager: HeadManager,
        settings: Settings,
        cloud_url: str,
        cloud_api_key: str,
        cloud_project_id: str,
        cloud_agent_id: str = "multihead-cloud-agent",
        agentic_core: Any = None,
        knowledge_store: Any = None,
        runtime_config: Any = None,
        # Pipeline infrastructure (enables full solve for complex contracts)
        event_store: Any = None,
        artifact_store: Any = None,
        runs_dir: Any = None,
        acp_bridge: Any = None,
    ) -> None:
        self._heads = head_manager
        self._settings = settings
        self._cloud_url = cloud_url.rstrip("/")
        self._cloud_api_key = cloud_api_key
        self._cloud_project_id = cloud_project_id
        self._cloud_agent_id = cloud_agent_id
        self._agentic_core = agentic_core
        self._knowledge_store = knowledge_store
        self._config = runtime_config

        # State tracking
        self._quoted_rfqs: set[str] = set()
        self._declined_contracts: set[str] = set()  # Track declined to avoid re-polling
        self._failed_contracts: dict[str, int] = {}  # Track failure counts to prevent retry loops
        self._active_contracts: dict[str, asyncio.Task[None]] = {}
        self._listings_cache: list[dict[str, Any]] = []
        self._listings_cache_time: float = 0.0
        self._stats = {"rfqs_seen": 0, "quotes_sent": 0, "contracts_won": 0, "contracts_done": 0}

        # Pipeline infrastructure
        self._event_store = event_store
        self._artifact_store = artifact_store
        self._runs_dir = runs_dir or (getattr(settings, "runs_dir", None) if settings else None)
        self._acp_bridge = acp_bridge

        # Login credentials for re-authentication when JWT expires
        self._cloud_email: str = ""
        self._cloud_password: str = ""

        # Participant identity (set by service wrapper if available)
        self._participant_id: str = ""

        # Optional callback for shell visibility: fn(event_type, message)
        self.on_activity: Any = None

        # Background tasks
        self._cloud_bridge: Any = None  # ACPBridge instance
        self._scanner_task: asyncio.Task[None] | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self._token_refresh_task: asyncio.Task[None] | None = None
        self._running = False

    def _emit(self, event_type: str, message: str) -> None:
        """Emit activity event for shell visibility."""
        if self.on_activity:
            try:
                self.on_activity(event_type, message)
            except Exception:
                pass

    def _svc_config(self) -> Any:
        """Get services config, or None."""
        return getattr(self._config, "services", None)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start all background loops. Blocks until cancelled."""
        self._running = True
        logger.info(
            "Cloud marketplace bridge starting: %s (agent=%s)",
            self._cloud_url, self._cloud_agent_id,
        )

        # Early token check — re-login if JWT is already expired
        exp = self._jwt_exp(self._cloud_api_key)
        if exp and exp <= time.time():
            logger.warning("Cloud JWT already expired at startup — re-authenticating")
            await self._re_login()

        # Start cloud ACPBridge for direct task polling + heartbeat
        try:
            from ..acp_bridge import ACPBridge

            self._cloud_bridge = ACPBridge(
                head_manager=self._heads,
                settings=self._settings,
                acp_url=self._cloud_url,
                api_key=self._cloud_api_key,
                project_id=self._cloud_project_id,
                agent_id=self._cloud_agent_id,
                auto_execute=False,  # Route to brain, don't auto-execute
                skip_registration=True,  # Cloud has its own auth (login+JWT)
            )
            if self._agentic_core:
                self._cloud_bridge.set_agentic_core(self._agentic_core)
            await self._cloud_bridge.start()
        except Exception as e:
            logger.warning("Cloud ACPBridge failed to start: %s", e)
            self._cloud_bridge = None

        # Launch scanner and monitor loops
        svc = self._svc_config()
        auto_quote = getattr(svc, "cloud_auto_quote", True) if svc else True

        tasks: list[asyncio.Task[None]] = []

        if auto_quote:
            self._scanner_task = asyncio.create_task(
                self._rfq_scanner_loop(), name="cloud-rfq-scanner",
            )
            tasks.append(self._scanner_task)

        self._monitor_task = asyncio.create_task(
            self._contract_monitor_loop(), name="cloud-contract-monitor",
        )
        tasks.append(self._monitor_task)

        # Token refresh loop (keeps cloud API key valid)
        self._token_refresh_task = asyncio.create_task(
            self._token_refresh_loop(), name="cloud-token-refresh",
        )
        tasks.append(self._token_refresh_task)

        # Register our capabilities as marketplace listings
        await self._register_listings()

        logger.info(
            "Cloud marketplace bridge running (%d loops, auto_quote=%s)",
            len(tasks), auto_quote,
        )

        # Block until cancelled
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False

        for task in [self._scanner_task, self._monitor_task, self._token_refresh_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=3.0)
                except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                    pass

        # Cancel active contract executions and wait for them
        pending: list[asyncio.Task[None]] = []
        for cid, task in self._active_contracts.items():
            if not task.done():
                task.cancel()
                pending.append(task)
        if pending:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                pass
        self._active_contracts.clear()

        if self._cloud_bridge:
            try:
                await self._cloud_bridge.stop()
            except Exception:
                pass

        logger.info("Cloud marketplace bridge stopped")
