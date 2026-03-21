# Event-Sourced Durable Execution

## Overview

MultiHead uses an event-sourced architecture for crash-proof execution. The system can resume after failure without losing where it was. This is the lightweight equivalent of Temporal's "durable execution" pattern, without adopting a full platform.

## The Thesis

MultiHead v0.1 = a durable step-runner that can swap model "heads" without losing state. If you can kill the process mid-run and it resumes cleanly from the last committed step, you've crossed the line into "real infrastructure."

## Append-Only Event Log

Every run produces an append-only event log stored as JSONL:

```
runs/<run_id>/events.jsonl
```

The orchestrator can reconstruct state from the log and continue.

## Required Events

| Event | Description |
|-------|-------------|
| `RUN_CREATED` | WorkOrder accepted, run initialized |
| `STEP_PLANNED` | Step queued for execution |
| `STEP_STARTED` | Step execution began, head loaded |
| `STEP_OUTPUT_WRITTEN` | Artifacts written (paths + hashes recorded) |
| `STEP_COMMITTED` | Checkpoint boundary -- step is done |
| `STEP_FAILED` | Step failed (error details included) |
| `RUN_DONE` | All steps completed successfully |
| `RUN_CANCELLED` | Run cancelled by user or policy |

## Replay Rules

### On Startup

1. Scan `events.jsonl`
2. Find last `STEP_COMMITTED` event
3. Resume at next step

### Idempotency Rule

If a step has `STEP_COMMITTED`, do **not** re-run it unless user forces replay. This prevents wasted work and ensures determinism.

### Replay Command

```
POST /runs/{run_id}/replay?step_id=...
```

Force rerun from a specific step (useful for debugging or when a step produced bad output).

## Checkpoint Modes

### Sync Checkpointing (Default in v0.1)

Every step ends with this sequence, synchronously:

1. Validate output schema
2. Write artifacts to disk
3. Append `STEP_OUTPUT_WRITTEN` event
4. Append `STEP_COMMITTED` event

Only after all four succeed does the orchestrator advance to the next step.

**Trade-off:** Slower, but guarantees no step is lost even if the process dies mid-execution.

### Async Checkpointing (Optional, Later)

Write artifacts and events asynchronously. Faster, but can lose the last step if the process dies mid-write.

**v0.1 policy:** Default to sync for correctness. Allow per-step `checkpoint_mode: async` override for steps where speed matters more than safety.

## Kill-9 Resilience

The "Definition of Done" for v0.1:

> You can `kill -9` the orchestrator mid-run and it resumes from the last committed step. No redoing completed steps.

### How It Works

1. Process is killed mid-step (e.g., during `STEP_STARTED` but before `STEP_COMMITTED`)
2. On restart, orchestrator reads `events.jsonl`
3. Finds last `STEP_COMMITTED` -- say it was step 2
4. Step 3 had `STEP_STARTED` but no `STEP_COMMITTED` -- it's incomplete
5. Orchestrator re-runs step 3 from scratch
6. Steps 1 and 2 are not re-executed (their artifacts already exist)

### What Gets Preserved

- All committed step artifacts (on disk, content-addressed)
- All committed events (append-only log)
- The WorkOrder definition itself

### What May Be Lost

- In-progress work from the uncommitted step (this is by design -- it gets re-run)
- Any state held only in memory (this is why events are the source of truth)

## Event Log Format

Each event is a single JSON line in `events.jsonl`:

```json
{"event": "RUN_CREATED", "run_id": "run_abc123", "ts": "2026-02-11T20:12:33Z", "data": {"goal": "Image folder to structured report", "step_count": 3}}
{"event": "STEP_PLANNED", "run_id": "run_abc123", "step_id": "step_01", "ts": "2026-02-11T20:12:33Z", "data": {"tool": "llm.plan", "head_id": "qwen3-8b"}}
{"event": "STEP_STARTED", "run_id": "run_abc123", "step_id": "step_01", "ts": "2026-02-11T20:12:34Z", "data": {"head_id": "qwen3-8b", "head_state": "ACTIVE"}}
{"event": "STEP_OUTPUT_WRITTEN", "run_id": "run_abc123", "step_id": "step_01", "ts": "2026-02-11T20:13:01Z", "data": {"artifacts": [{"path": "artifacts/step_01_plan.json", "sha256": "abcd..."}]}}
{"event": "STEP_COMMITTED", "run_id": "run_abc123", "step_id": "step_01", "ts": "2026-02-11T20:13:01Z"}
```

## Storage Layout

```
runs/
  <run_id>/
    workorder.json      # The original WorkOrder
    events.jsonl        # Append-only event log
    artifacts/          # Step outputs, content-addressed
      step_01_plan.json
      step_02_vlm_raw.jsonl
      step_03_items.json
      report.md
```

Plus a SQLite index for fast listing/status queries:

```
runs.db   # SQLite: run_id, status, created_at, step_count, etc.
```

## API Surface for Runs

```
POST /runs                            -> create run from recipe + inputs
GET  /runs/{run_id}                   -> status, current step, timestamps
GET  /runs/{run_id}/events?tail=200   -> stream/tail JSONL
POST /runs/{run_id}/cancel
POST /runs/{run_id}/replay?step_id=.. -> force rerun from step
```

