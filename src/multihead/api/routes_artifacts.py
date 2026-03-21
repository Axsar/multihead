"""API routes for artifact management."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_CHUNK_SIZE = 64 * 1024  # 64KB read chunks


@router.get("")
async def list_artifacts(request: Request, limit: int = Query(default=20, le=100)):
    """List artifacts in the store (most recent first)."""
    artifact_store = request.app.state.artifact_store
    all_artifacts = artifact_store.list_all()
    return all_artifacts[-limit:] if len(all_artifacts) > limit else all_artifacts


@router.post("")
async def upload_artifact(
    request: Request,
    file: UploadFile,
    name: str | None = None,
    media_type: str | None = None,
):
    """Upload an artifact to the content-addressed store."""
    artifact_store = request.app.state.artifact_store
    max_size = artifact_store.max_size_bytes

    # Stream-read with early abort if file exceeds max size
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(UPLOAD_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_size:
            raise HTTPException(
                413, f"Upload too large: exceeds {max_size} byte limit"
            )
        chunks.append(chunk)
    data = b"".join(chunks)

    ref = artifact_store.store(
        data=data,
        name=name or file.filename or "unnamed",
        media_type=media_type or file.content_type or "application/octet-stream",
    )

    return {
        "artifact_id": ref.artifact_id,
        "uri": ref.uri,
        "name": ref.name,
        "media_type": ref.media_type,
        "size_bytes": ref.size_bytes,
    }


@router.get("/{artifact_id:path}")
async def download_artifact(artifact_id: str, request: Request):
    """Download artifact bytes."""
    artifact_store = request.app.state.artifact_store
    data = artifact_store.fetch(artifact_id)
    if data is None:
        raise HTTPException(404, f"Artifact not found: {artifact_id}")

    meta = artifact_store.get_meta(artifact_id)
    media_type = meta.get("media_type", "application/octet-stream") if meta else "application/octet-stream"

    return Response(content=data, media_type=media_type)


@router.get("/{artifact_id:path}/meta")
async def get_artifact_meta(artifact_id: str, request: Request):
    """Get artifact metadata."""
    artifact_store = request.app.state.artifact_store
    meta = artifact_store.get_meta(artifact_id)
    if meta is None:
        raise HTTPException(404, f"Artifact not found: {artifact_id}")
    return meta
