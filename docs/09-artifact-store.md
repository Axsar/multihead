# Artifact Store

## Overview

The Artifact Store is the content-addressed storage layer for all intermediate and final outputs produced by MultiHead runs. Every artifact is immutable, hashable, and trackable.

## Design Principles

- **Content-addressed**: Artifact IDs are `sha256:<hex>` digests of the content
- **Immutable**: Once written, an artifact is never modified (new versions get new hashes)
- **Cacheable**: Same input + same tool + same params = same hash = skip recomputation
- **Local-first**: Files on disk, metadata in SQLite
- **Deletable**: Cleanup truly deletes files + rows + rebuilds indexes

## SHA-256 Sharding

Artifacts are stored in a content-addressed directory structure:

```
~/.acp/artifacts/
  <sha256_prefix>/
    <full_sha256>/
      data            # The actual artifact bytes
      meta.json       # Metadata sidecar
```

Alternative flat layout for simplicity in v0.1:

```
runs/<run_id>/artifacts/
  step_01_plan.json
  step_02_vlm_raw.jsonl
  step_03_items.json
  report.md
```

The global content-addressed store can coexist with per-run artifact directories. The per-run directory provides human-readable access; the global store provides deduplication.

## Artifact Metadata

Every artifact has associated metadata:

```json
{
  "artifact_id": "sha256:abcd1234...",
  "uri": "artifact://sha256:abcd1234...",
  "media_type": "application/json",
  "name": "step_01_plan.json",
  "size_bytes": 4096,
  "sha256": "abcd1234...",
  "created_at": "2026-02-11T20:13:01Z",
  "annotations": {
    "run_id": "run_abc123",
    "step_id": "step_01",
    "tool": "llm.plan",
    "model": "qwen3:8b"
  }
}
```

### Key Fields

| Field | Description |
|-------|-------------|
| `artifact_id` | Content hash, format `sha256:<hex>` |
| `uri` | Stable pointer: `artifact://sha256:<hex>` |
| `media_type` | MIME type (`application/json`, `image/png`, `text/plain`, etc.) |
| `name` | Human-readable filename |
| `size_bytes` | File size |
| `sha256` | Hex digest (without prefix) |
| `created_at` | When the artifact was first stored |
| `annotations` | Arbitrary metadata (run context, tool info, etc.) |

## Artifact References

Steps reference artifacts via `ArtifactRef` objects:

```json
{
  "uri": "artifact://sha256:abcd1234...",
  "media_type": "image/png",
  "name": "panel_001.png",
  "role": "input_image"
}
```

The `role` field provides semantic context: `input_image`, `mask`, `patch`, `report`, `raw_output`, etc.

## Content-Addressed Storage (CAS) Patterns

### Why CAS

1. **Deduplication**: Identical content is stored once regardless of how many runs produce it
2. **Cache validation**: Hash comparison instantly confirms whether a cached result is still valid
3. **Integrity**: Any corruption is detectable via hash mismatch
4. **Provenance**: Every artifact's lineage is traceable through run events

### Stage Caching

The caching formula:

```
cache_key = hash(inputs_hash + tool_version + params_hash)
```

If the cache key matches a prior run, skip the stage and reuse artifacts. This is what makes repeated runs and incremental pipelines efficient.

### Cache Invalidation

Caches are invalidated when:

- Input artifacts change (different hash)
- Tool version changes
- Parameters change
- User explicitly forces replay via `POST /runs/{run_id}/replay?step_id=...`

## Storage Hierarchy

```
~/.acp/
  artifacts/                    # Global content-addressed store
    <sha256>/
      data
      meta.json
  state.db                      # SQLite metadata index
  vault/                        # User-controlled memory (markdown/JSON)
  runs/
    <run_id>/
      workorder.json
      events.jsonl
      artifacts/                # Symlinks or copies for human access
        step_01_plan.json
        step_02_vlm_raw.jsonl
  memory/
    chunks.jsonl
    index.sqlite
```

## API Endpoints

### Upload Artifact

```
POST /v1/artifacts
Content-Type: application/octet-stream (or multipart/form-data)
```

Server computes SHA-256, stores content, returns `ArtifactMeta` with stable URI.

### Download Artifact

```
GET /v1/artifacts/{artifact_id}
```

Returns raw bytes with appropriate Content-Type header.

### Get Artifact Metadata

```
GET /v1/artifacts/{artifact_id}/meta
```

Returns `ArtifactMeta` JSON.

## Deletion

Real deletion means:

1. Delete the artifact file from disk
2. Delete the metadata row from SQLite
3. Remove from any index (embeddings, keyword)
4. Rebuild affected indexes

Artifact cleanup performs complete deletion with no ghost data.

## Relationship to Runs

Each run produces artifacts that are:

1. Written to disk (content-addressed)
2. Referenced in `STEP_OUTPUT_WRITTEN` events (paths + hashes)
3. Committed via `STEP_COMMITTED` events
4. Available for subsequent steps as input references

Artifacts survive run deletion if they're referenced by other runs (deduplication). Orphan cleanup is a maintenance task.

