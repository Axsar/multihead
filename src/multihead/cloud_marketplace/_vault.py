"""Vault integration — BotVibes pre-signed URL file escrow."""

from __future__ import annotations

import hashlib
from typing import Any

import httpx

from ._constants import CLOUD_TIMEOUT, MAX_VAULT_DOWNLOAD_BYTES, logger


class VaultMixin:
    """Mixin providing vault upload/download operations."""

    # Attributes defined on the main class.
    _cloud_url: str

    async def _cloud_request(self, client: Any, method: str, url: str, **kw: Any) -> Any: ...

    async def _vault_list_entries(self, contract_id: str) -> list[dict[str, Any]]:
        """List vault entries for a contract."""
        try:
            async with httpx.AsyncClient(timeout=CLOUD_TIMEOUT) as client:
                resp = await self._cloud_request(
                    client, "GET",
                    f"{self._cloud_url}/api/v1/vault/contracts/{contract_id}/entries",
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("entries", data) if isinstance(data, dict) else data
                logger.debug(
                    "Vault list for %s returned %d", contract_id, resp.status_code,
                )
        except Exception as e:
            logger.debug("Vault list failed for %s: %s", contract_id, e)
        return []

    async def _vault_download(self, entry_id: str) -> tuple[str, bytes, str]:
        """Download a vault entry via pre-signed URL.

        Returns (filename, content_bytes, sha256_hex).
        """
        async with httpx.AsyncClient(timeout=CLOUD_TIMEOUT) as client:
            # Step 1: get pre-signed download URL
            resp = await self._cloud_request(
                client, "POST",
                f"{self._cloud_url}/api/v1/vault/entries/{entry_id}/download",
            )
            resp.raise_for_status()
            info = resp.json()
            download_url = info["download_url"]
            filename = info.get("filename", f"vault_{entry_id}")
            expected_sha = info.get("sha256", "")

            # Step 2: GET the actual bytes (with size limit)
            dl_resp = await client.get(download_url)
            dl_resp.raise_for_status()
            content = dl_resp.content
            if len(content) > MAX_VAULT_DOWNLOAD_BYTES:
                raise ValueError(
                    f"Vault entry {entry_id} exceeds size limit: "
                    f"{len(content)} > {MAX_VAULT_DOWNLOAD_BYTES} bytes"
                )
            actual_sha = hashlib.sha256(content).hexdigest()

            if expected_sha and actual_sha != expected_sha:
                raise ValueError(
                    f"Vault integrity check failed for {entry_id}: "
                    f"expected {expected_sha[:16]}…, got {actual_sha[:16]}…"
                )

            return filename, content, actual_sha

    async def _vault_upload(
        self,
        contract_id: str,
        filename: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload a file to the vault as provider_output.

        Returns the vault entry_id.
        """
        sha256_hex = hashlib.sha256(data).hexdigest()

        async with httpx.AsyncClient(timeout=CLOUD_TIMEOUT) as client:
            # Step 1: request upload URL
            resp = await self._cloud_request(
                client, "POST",
                f"{self._cloud_url}/api/v1/vault/contracts/{contract_id}/upload",
                json={
                    "filename": filename,
                    "role": "provider_output",
                    "content_type": content_type,
                    "sha256_declared": sha256_hex,
                },
            )
            resp.raise_for_status()
            info = resp.json()
            entry_id = info["entry_id"]
            upload_url = info["upload_url"]
            upload_headers = info.get("upload_headers", {})

            # Step 2: PUT bytes to pre-signed URL
            put_headers = {**upload_headers, "Content-Type": content_type}
            put_resp = await client.put(upload_url, content=data, headers=put_headers)
            put_resp.raise_for_status()

            # Step 3: confirm upload
            confirm_resp = await self._cloud_request(
                client, "POST",
                f"{self._cloud_url}/api/v1/vault/entries/{entry_id}/confirm",
                json={"sha256": sha256_hex, "size_bytes": len(data)},
            )
            confirm_resp.raise_for_status()

        logger.info(
            "Vault upload: %s → entry %s (%d bytes)", filename, entry_id, len(data),
        )
        return entry_id

    async def _download_vault_inputs(
        self, contract_id: str,
    ) -> list[dict[str, Any]]:
        """Download all buyer_input vault entries for a contract.

        Returns list of {filename, data, sha256, entry_id} dicts.
        Empty list for text-only contracts (no vault entries).
        """
        entries = await self._vault_list_entries(contract_id)
        inputs: list[dict[str, Any]] = []

        for entry in entries:
            role = entry.get("role", "")
            state = entry.get("state", "")
            entry_id = entry.get("entry_id", "") or entry.get("id", "")

            if role != "buyer_input" or state not in ("active", "confirmed"):
                continue
            if not entry_id:
                continue

            try:
                filename, data, sha256 = await self._vault_download(entry_id)
                inputs.append({
                    "filename": filename,
                    "data": data,
                    "sha256": sha256,
                    "entry_id": entry_id,
                })
                logger.info(
                    "Downloaded vault input: %s (%d bytes) for contract %s",
                    filename, len(data), contract_id[:8],
                )
            except Exception as e:
                logger.warning(
                    "Failed to download vault entry %s: %s", entry_id, e,
                )

        return inputs

    async def _upload_vault_outputs(
        self,
        contract_id: str,
        outputs: list[tuple[str, bytes, str]],
    ) -> list[str]:
        """Upload output files to the vault as provider_output.

        Args:
            contract_id: The contract to attach outputs to.
            outputs: List of (filename, data_bytes, content_type) tuples.

        Returns list of vault entry IDs.
        """
        entry_ids: list[str] = []
        for filename, data, content_type in outputs:
            try:
                eid = await self._vault_upload(
                    contract_id, filename, data, content_type,
                )
                entry_ids.append(eid)
            except Exception as e:
                logger.warning(
                    "Failed to upload vault output %s: %s", filename, e,
                )
        return entry_ids
