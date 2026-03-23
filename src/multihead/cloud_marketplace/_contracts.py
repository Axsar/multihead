"""Contract monitor — execute awarded contracts."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from ._constants import CLOUD_TIMEOUT, logger


_MAX_CONTRACT_RETRIES = 3


class ContractMixin:
    """Mixin providing contract monitoring and execution logic."""

    # Attributes defined on the main class.
    _cloud_url: str
    _cloud_agent_id: str
    _active_contracts: dict[str, asyncio.Task[None]]
    _declined_contracts: set[str]
    _failed_contracts: dict[str, int]  # contract_id → failure count
    _stats: dict[str, Any]
    _running: bool
    _agentic_core: Any
    _knowledge_store: Any
    _acp_bridge: Any
    on_activity: Any

    def _emit(self, event_type: str, message: str) -> None: ...
    def _svc_config(self) -> Any: ...
    def _deposit_claim(self, claim_key: str, statement: str) -> None: ...
    @staticmethod
    def _cap_set(s: set, max_size: int = 10_000) -> None: ...
    async def _cloud_request(self, client: Any, method: str, url: str, **kw: Any) -> Any: ...
    async def _download_vault_inputs(self, contract_id: str) -> list[dict[str, Any]]: ...
    async def _upload_vault_outputs(self, contract_id: str, outputs: list) -> list[str]: ...
    async def _route_and_execute(self, capability_id: str, payload: str, contract_id: str, vault_inputs: Any = None) -> Any: ...
    async def _post_receipt(self, contract_id: str, task_id: str, **kw: Any) -> None: ...
    async def _fetch_trust_score(self) -> Any: ...
    async def _decline_contract(self, contract_id: str, reason: str = "") -> None: ...

    # ------------------------------------------------------------------
    # Contract Monitor — execute awarded contracts
    # ------------------------------------------------------------------

    def _load_failed_contracts_from_kb(self) -> None:
        """Load previously failed contract IDs from knowledge store on startup."""
        if not self._knowledge_store:
            return
        try:
            import sqlite3
            conn = sqlite3.connect(str(self._knowledge_store.db_path), timeout=5.0)
            rows = conn.execute(
                "SELECT claim_key FROM claims "
                "WHERE claim_key LIKE 'cloud.marketplace.contract.failed.%' "
                "AND claim_status IN ('accepted', 'proposed')"
            ).fetchall()
            conn.close()
            for row in rows:
                # claim_key = cloud.marketplace.contract.failed.<contract_id>
                contract_id = row[0].rsplit(".", 1)[-1]
                if contract_id:
                    self._failed_contracts[contract_id] = _MAX_CONTRACT_RETRIES
            if self._failed_contracts:
                logger.info(
                    "Loaded %d previously failed contracts from knowledge store",
                    len(self._failed_contracts),
                )
        except Exception as e:
            logger.debug("Failed to load failed contracts from KB: %s", e)

    async def _contract_monitor_loop(self) -> None:
        """Periodically check for active contracts and execute them."""
        svc = self._svc_config()
        interval = getattr(svc, "cloud_contract_interval", 30) if svc else 30

        # Load failed contracts from knowledge store (persisted across restarts)
        self._load_failed_contracts_from_kb()

        logger.info("Contract monitor started (interval=%ds)", interval)

        while self._running:
            try:
                await self._check_contracts()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Contract check failed: %s", e)

            await asyncio.sleep(interval)

    async def _check_contracts(self) -> None:
        """Single check iteration: find active contracts, execute new ones."""
        svc = self._svc_config()
        max_contracts = getattr(svc, "cloud_max_contracts", 2) if svc else 2

        # Clean up completed contract tasks
        done = [cid for cid, t in self._active_contracts.items() if t.done()]
        for cid in done:
            del self._active_contracts[cid]

        # Check concurrency limit
        if len(self._active_contracts) >= max_contracts:
            return

        async with httpx.AsyncClient(timeout=CLOUD_TIMEOUT) as client:
            try:
                resp = await self._cloud_request(
                    client, "GET",
                    f"{self._cloud_url}/marketplace/contracts",
                    params={
                        "role": "provider",
                        "status": "active",
                        "agent_id": self._cloud_agent_id,
                    },
                )
                if resp.status_code != 200:
                    return

                data = resp.json()
                contracts = data if isinstance(data, list) else data.get("contracts", [])

            except Exception as e:
                logger.debug("Contract poll failed: %s", e)
                return

        for contract in contracts:
            contract_id = contract.get("contract_id", "")
            if not contract_id or contract_id in self._active_contracts:
                continue
            if contract_id in self._declined_contracts:
                continue
            if self._failed_contracts.get(contract_id, 0) >= _MAX_CONTRACT_RETRIES:
                continue
            # Also check partial matches (claim keys may store short or full IDs)
            if any(contract_id.startswith(fid) or fid.startswith(contract_id)
                   for fid in self._failed_contracts
                   if self._failed_contracts[fid] >= _MAX_CONTRACT_RETRIES):
                continue

            if len(self._active_contracts) >= max_contracts:
                break

            # Spawn execution task
            task = asyncio.create_task(
                self._execute_contract(contract),
                name=f"contract-{contract_id[:8]}",
            )
            self._active_contracts[contract_id] = task
            self._stats["contracts_won"] += 1
            logger.info("Executing contract %s", contract_id)
            self._emit("contract", f"Won contract {contract_id[:8]} — executing")

    async def _execute_contract(self, contract: dict[str, Any]) -> None:
        """Execute a single contract: run task locally, post receipt.

        Checks cloud_auto_deliver config and capability whitelist before
        executing. Non-whitelisted capabilities are logged but skipped
        (manual fulfillment).
        """
        contract_id = contract.get("contract_id", "")
        # Contract may nest under 'contract' key in search results
        inner = contract.get("contract", contract)
        task_id = inner.get("task_id", "")
        capability_id = inner.get("capability_id", "") or inner.get(
            "required_capability", ""
        )
        rfq_id = inner.get("rfq_id", "")
        payload = (
            inner.get("payload_ref", "")
            or inner.get("description", "")
            or inner.get("notes", "")
        )

        if not payload and not capability_id:
            logger.warning("Contract %s has no payload, declining", contract_id)
            await self._decline_contract(contract_id, "No payload or capability specified")
            return

        # --- Download vault inputs (buyer-uploaded files) ---
        vault_inputs: list[dict[str, Any]] = []
        try:
            vault_inputs = await self._download_vault_inputs(contract_id)
            if vault_inputs:
                logger.info(
                    "Contract %s: downloaded %d vault input(s)",
                    contract_id[:8], len(vault_inputs),
                )
        except Exception as e:
            logger.warning("Vault input download for %s failed: %s", contract_id[:8], e)

        # --- Auto-deliver gate ---
        svc = self._svc_config()
        auto_deliver = getattr(svc, "cloud_auto_deliver", False) if svc else False
        whitelist: list[str] = (
            getattr(svc, "cloud_auto_deliver_capabilities", []) if svc else []
        )

        if not auto_deliver:
            logger.info(
                "Contract %s received but auto-deliver disabled, skipping",
                contract_id,
            )
            # Track so we don't re-poll this contract every cycle
            self._declined_contracts.add(contract_id)
            self._cap_set(self._declined_contracts)
            self._deposit_claim(
                f"cloud.marketplace.contract.pending.{contract_id}",
                f"Contract {contract_id} received (auto-deliver OFF). "
                f"Capability: {capability_id}. Manual fulfillment required.",
            )
            return

        if capability_id and whitelist and capability_id not in whitelist:
            logger.info(
                "Contract %s capability %s not in auto-deliver whitelist, declining",
                contract_id,
                capability_id,
            )
            await self._decline_contract(
                contract_id,
                f"Capability {capability_id} not in auto-deliver whitelist",
            )
            return

        start_time = time.time()
        outcome = "success"
        output = ""
        confidence = 0.85
        vault_output_ids: list[str] = []

        try:
            # Capability-aware routing (with vault inputs if any)
            result = await self._route_and_execute(
                capability_id, payload, contract_id,
                vault_inputs=vault_inputs or None,
            )
            output, confidence = result[0], result[1]
            # Check for binary outputs to upload to vault
            binary_outputs: list[tuple[str, bytes, str]] = (
                result[2] if len(result) > 2 else []
            )
            if binary_outputs:
                vault_output_ids = await self._upload_vault_outputs(
                    contract_id, binary_outputs,
                )
                logger.info(
                    "Contract %s: uploaded %d vault output(s)",
                    contract_id[:8], len(vault_output_ids),
                )
        except Exception as e:
            logger.error("Contract %s execution failed: %s", contract_id, e)
            outcome = "failure"
            output = str(e)
            confidence = 0.0
            # Track failure count to prevent infinite retry loops
            self._failed_contracts[contract_id] = self._failed_contracts.get(contract_id, 0) + 1
            if self._failed_contracts[contract_id] >= _MAX_CONTRACT_RETRIES:
                logger.warning(
                    "Contract %s failed %d times, giving up",
                    contract_id[:8], self._failed_contracts[contract_id],
                )
                self._emit("failed", f"Contract {contract_id[:8]} — max retries reached, skipping")
                # Persist to knowledge store so we remember across restarts
                self._deposit_claim(
                    f"cloud.marketplace.contract.failed.{contract_id}",
                    f"Contract {contract_id} failed {self._failed_contracts[contract_id]} times. "
                    f"Last error: {output[:200]}. Giving up — will not retry.",
                )

        latency_ms = int((time.time() - start_time) * 1000)

        # Post receipt
        await self._post_receipt(
            contract_id, task_id,
            outcome=outcome,
            latency_ms=latency_ms,
            confidence=confidence,
            output=output,
            vault_entry_ids=vault_output_ids or None,
        )

        # Also submit task result if we have a task_id
        if task_id and outcome == "success":
            await self._submit_task_result(task_id, output, latency_ms, confidence)

        if outcome == "success":
            self._stats["contracts_done"] += 1
            self._emit("delivered", f"Delivered contract {contract_id[:8]} ({latency_ms}ms)")
            # Update trust score after successful delivery
            await self._fetch_trust_score()
        else:
            self._emit("failed", f"Contract {contract_id[:8]} failed: {output[:80]}")

        self._deposit_claim(
            f"cloud.marketplace.contract.{contract_id}",
            f"Contract {contract_id} auto-delivered: {outcome} "
            f"({capability_id}, {latency_ms}ms, conf={confidence:.2f})",
        )

    async def _submit_task_result(
        self,
        task_id: str,
        output: str,
        latency_ms: int,
        confidence: float,
    ) -> None:
        """Submit task result back to cloud ACP."""
        try:
            async with httpx.AsyncClient(timeout=CLOUD_TIMEOUT) as client:
                await self._cloud_request(
                    client, "POST",
                    f"{self._cloud_url}/tasks/{task_id}/result",
                    json={
                        "status": "complete",
                        "output_ref": output[:2000],
                        "confidence": confidence,
                        "latency_ms": latency_ms,
                        "subtasks": [],
                    },
                )
        except Exception as e:
            logger.debug("Failed to submit cloud task result for %s: %s", task_id, e)
