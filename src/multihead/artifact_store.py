"""Content-addressed artifact store with SHA-256 sharding."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import ArtifactRef


class ArtifactStore:
    """SHA-256 content-addressed local artifact store with 2-level sharding."""

    DEFAULT_MAX_SIZE_BYTES: int = 100 * 1024 * 1024  # 100 MB

    def __init__(self, root: Path, db_path: Path, max_size_bytes: int = DEFAULT_MAX_SIZE_BYTES) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.max_size_bytes = max_size_bytes
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    name TEXT,
                    media_type TEXT,
                    size_bytes INTEGER,
                    created_at TEXT,
                    annotations TEXT
                )
            """)
            conn.commit()

    def _shard_path(self, sha: str) -> Path:
        return self.root / sha[:2] / sha[2:4] / sha

    def store(
        self,
        data: bytes,
        name: str = "",
        media_type: str = "",
        annotations: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        """Store bytes and return an ArtifactRef."""
        if len(data) > self.max_size_bytes:
            raise ValueError(
                f"Artifact too large: {len(data)} bytes exceeds limit of {self.max_size_bytes} bytes"
            )

        sha = hashlib.sha256(data).hexdigest()
        dest = self._shard_path(sha)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if not dest.exists():
            # Atomic write: write to temp file then rename
            fd, tmp_path = tempfile.mkstemp(dir=dest.parent)
            try:
                with open(fd, "wb") as f:
                    f.write(data)
                Path(tmp_path).replace(dest)
            except BaseException:
                Path(tmp_path).unlink(missing_ok=True)
                raise

        artifact_id = f"sha256:{sha}"
        now = datetime.now(timezone.utc).isoformat()

        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            conn.execute(
                """INSERT OR IGNORE INTO artifacts
                   (artifact_id, name, media_type, size_bytes, created_at, annotations)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (artifact_id, name, media_type, len(data), now,
                 json.dumps(annotations or {})),
            )
            conn.commit()

        return ArtifactRef(
            artifact_id=artifact_id,
            uri=f"artifact://{artifact_id}",
            name=name,
            media_type=media_type,
            size_bytes=len(data),
        )

    def store_file(
        self,
        path: Path,
        name: str = "",
        media_type: str = "",
    ) -> ArtifactRef:
        """Store a file by path."""
        data = path.read_bytes()
        return self.store(data, name=name or path.name, media_type=media_type)

    def fetch(self, artifact_id: str) -> bytes | None:
        """Fetch artifact bytes by ID."""
        sha = artifact_id.removeprefix("sha256:")
        path = self._shard_path(sha)
        if path.exists():
            return path.read_bytes()
        return None

    def fetch_path(self, artifact_id: str) -> Path | None:
        """Get the filesystem path for an artifact."""
        sha = artifact_id.removeprefix("sha256:")
        path = self._shard_path(sha)
        return path if path.exists() else None

    def exists(self, artifact_id: str) -> bool:
        sha = artifact_id.removeprefix("sha256:")
        return self._shard_path(sha).exists()

    def get_meta(self, artifact_id: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            row = conn.execute(
                "SELECT artifact_id, name, media_type, size_bytes, created_at, annotations FROM artifacts WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "artifact_id": row[0],
            "name": row[1],
            "media_type": row[2],
            "size_bytes": row[3],
            "created_at": row[4],
            "annotations": json.loads(row[5]) if row[5] else {},
        }

    def list_all(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            rows = conn.execute(
                "SELECT artifact_id, name, media_type, size_bytes, created_at FROM artifacts ORDER BY created_at DESC"
            ).fetchall()
        return [
            {"artifact_id": r[0], "name": r[1], "media_type": r[2], "size_bytes": r[3], "created_at": r[4]}
            for r in rows
        ]

    def delete(self, artifact_id: str) -> bool:
        sha = artifact_id.removeprefix("sha256:")
        path = self._shard_path(sha)
        if path.exists():
            path.unlink()
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            conn.execute("DELETE FROM artifacts WHERE artifact_id = ?", (artifact_id,))
            conn.commit()
        return True
