"""Append-only event store for durable execution."""

from __future__ import annotations

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore[assignment]  # Windows — file locking unavailable
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

from .models import (
    EventKind,
    RunEvent,
    RunState,
    RunStatus,
    StageResult,
    StepStatus,
    WorkOrder,
)


class EventStore:
    """Event-sourced state store: JSONL per run + SQLite index."""

    def __init__(self, runs_dir: Path, db_path: Path) -> None:
        self.runs_dir = runs_dir
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'queued',
                    goal TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            conn.commit()

    def _run_dir(self, run_id: str) -> Path:
        d = self.runs_dir / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _events_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "events.jsonl"

    def append(self, event: RunEvent) -> None:
        """Append an event to the run's JSONL log and update index.

        Order: SQLite index first (can be rebuilt from JSONL), then JSONL (append-only).
        Both are independently recoverable — index from replay, JSONL from file lock.
        """
        # 1. Update SQLite index (recoverable via replay if inconsistent)
        status = self._status_from_event(event.kind)
        if status:
            now = datetime.now(timezone.utc).isoformat()
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                conn.execute(
                    """INSERT INTO runs (run_id, status, created_at, updated_at)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(run_id) DO UPDATE SET status=?, updated_at=?""",
                    (event.run_id, status, now, now, status, now),
                )
                conn.commit()

        # 2. Append to JSONL (source of truth, file-locked)
        path = self._events_path(event.run_id)
        line = event.model_dump_json() + "\n"
        with open(path, "a") as f:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()
            finally:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _status_from_event(self, kind: EventKind) -> str | None:
        mapping = {
            EventKind.RUN_CREATED: "queued",
            EventKind.STEP_STARTED: "running",
            EventKind.RUN_DONE: "done",
            EventKind.RUN_FAILED: "failed",
            EventKind.RUN_CANCELLED: "cancelled",
        }
        return mapping.get(kind)

    def read_events(self, run_id: str) -> list[RunEvent]:
        """Read all events for a run. Skips corrupted/truncated lines."""
        path = self._events_path(run_id)
        if not path.exists():
            return []
        events = []
        with open(path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(RunEvent.model_validate_json(line))
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(
                        "Skipping corrupted event line %d in %s: %s",
                        line_num, run_id, e,
                    )
        return events

    def replay(self, run_id: str, work_order: WorkOrder | None = None) -> RunState:
        """Replay events to reconstruct RunState. Returns state with correct step index."""
        events = self.read_events(run_id)
        state = RunState(run_id=run_id)

        for event in events:
            match event.kind:
                case EventKind.RUN_CREATED:
                    state.status = RunStatus.QUEUED
                    state.created_at = event.timestamp
                    if "work_order" in event.data:
                        try:
                            state.work_order = WorkOrder.model_validate(event.data["work_order"])
                        except Exception as e:
                            logger.warning(
                                "Corrupted work_order in run %s event replay: %s", run_id, e
                            )

                case EventKind.STEP_STARTED:
                    state.status = RunStatus.RUNNING
                    if state.started_at is None:
                        state.started_at = event.timestamp

                case EventKind.STEP_COMMITTED:
                    if event.step_id:
                        state.step_results[event.step_id] = StageResult(
                            step_id=event.step_id,
                            head_id=event.data.get("head_id", ""),
                            status=StepStatus.COMMITTED,
                            outputs=event.data.get("outputs", {}),
                            metrics=event.data.get("metrics", {}),
                        )
                    # Advance to next step
                    state.current_step_index = len(state.step_results)

                case EventKind.STEP_FAILED:
                    if event.step_id:
                        state.step_results[event.step_id] = StageResult(
                            step_id=event.step_id,
                            head_id=event.data.get("head_id", ""),
                            status=StepStatus.FAILED,
                            error=event.data.get("error", "unknown"),
                        )

                case EventKind.RUN_DONE:
                    state.status = RunStatus.DONE
                    state.ended_at = event.timestamp

                case EventKind.RUN_FAILED:
                    state.status = RunStatus.FAILED
                    state.ended_at = event.timestamp

                case EventKind.RUN_CANCELLED:
                    state.status = RunStatus.CANCELLED
                    state.ended_at = event.timestamp

        # Attach work_order from arg if not found in events
        if state.work_order is None and work_order is not None:
            state.work_order = work_order

        return state

    def list_runs(self) -> list[dict]:
        """List all runs from the index."""
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            rows = conn.execute(
                "SELECT run_id, status, goal, created_at, updated_at FROM runs ORDER BY created_at DESC"
            ).fetchall()
        return [
            {"run_id": r[0], "status": r[1], "goal": r[2], "created_at": r[3], "updated_at": r[4]}
            for r in rows
        ]

    def get_run_summary(self, run_id: str) -> dict | None:
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            row = conn.execute(
                "SELECT run_id, status, goal, created_at, updated_at FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if not row:
            return None
        return {"run_id": row[0], "status": row[1], "goal": row[2], "created_at": row[3], "updated_at": row[4]}
