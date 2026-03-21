# Head Manager: Model Swapping

## Overview

The Head Manager is responsible for enforcing "only one GPU-heavy head active at a time" while providing fast switching between models. It supports multiple backends with a unified interface.

## Head Manifest (Concept)

Each head declares:

| Field | Description |
|-------|-------------|
| `head_id` | Unique identifier |
| `kind` | `llm` / `vlm` / `embed` / `tool` |
| `backend` | `ollama` / `vllm_sleep` / `hf_transformers` / `remote` |
| `endpoints` | generate/chat, plus sleep/wake/unload if supported |
| `resources` | `gpu_required`, `vram_hint`, `cpu_ram_hint` |

## Head States

A head can be in one of these states:

| State | Description |
|-------|-------------|
| `OFF` | Not loaded, no resources consumed |
| `WARM` | Loaded and ready to serve |
| `ASLEEP` | Hibernated, minimal resources |
| `ACTIVE` | Currently serving requests |

## Backend Adapters

### A) Ollama Adapter (Simple Path -- "It Just Works")

The default for v0.1. Ollama already handles model management with built-in primitives.

**Adapter Behavior:**

| Operation | Implementation |
|-----------|---------------|
| **Preload (Wake)** | Send an empty request with `keep_alive: -1` (keep loaded) |
| **Keep Loaded** | `keep_alive: -1` |
| **Unload** | `keep_alive: 0` OR `ollama stop <model>` |

Default unload behavior is ~5 minutes; you can force unload with `ollama stop`.

**API call example:**

```bash
# Preload / keep warm
curl http://localhost:11434/api/chat -d '{
  "model": "qwen3:8b",
  "keep_alive": -1,
  "messages": [{"role":"user","content":"ping"}]
}'

# Force unload
curl http://localhost:11434/api/chat -d '{
  "model": "qwen3:8b",
  "keep_alive": 0,
  "messages": []
}'
```

### B) vLLM Sleep Mode Adapter (Fast-Switch Path)

Run each head as its own vLLM server (separate ports), using sleep/wake for near-instant switching.

**Sleep Levels:**

| Level | Behavior | Use When |
|-------|----------|----------|
| **Level 1** | Offload weights to CPU RAM; release GPU VRAM | You have enough CPU RAM to hold multiple models |
| **Level 2** | Discard weights entirely (more aggressive) | CPU RAM is also limited |

**Adapter Behavior:**

Before switching heads:

1. Mark current engine "draining"
2. Wait for `inflight == 0`
3. Call `POST /sleep?level=1` (or `level=2`)
4. Call `POST /wake_up` on next engine
5. (Optionally call `reload_weights` if needed)

**Key Engineering Consideration:** Waking can create transient memory pressure. There are real reports of OOM issues around wake-up and extra copies. v0.1 should enforce strict "one awake engine" on GPU and drain requests before sleeping.

**API calls:**

```bash
# Sleep model on port 8001
curl -X POST http://localhost:8001/sleep?level=1

# Wake model on port 8002
curl -X POST http://localhost:8002/wake_up
```

Wake times can be sub-second in some cases -- huge speedups vs cold starts.

### C) HuggingFace Transformers (Direct Loading)

For non-LLM models (SAM, OCR, custom vision models) that don't use Ollama or vLLM:

- Load model directly via `transformers` / custom Python
- Manage GPU memory manually
- Unload by deleting references and calling `torch.cuda.empty_cache()`

## VRAM Policies

### Model Profile Config

```yaml
profiles:
  core:
    provider: ollama
    model: qwen3:4b
    mode: cpu                    # Always available
  worker:
    provider: ollama
    model: qwen3:8b
    mode: gpu
    load_policy: per_stage       # Load/unload each step
  verifier:
    provider: ollama
    model: qwen3:1.8b
    mode: cpu                    # Cheap checks
```

### VRAM Policy Options

```yaml
core_mode: keep_loaded | cpu_fallback | unload_during_batch
worker_load_policy: per_stage | keep_warm
```

**`keep_loaded`**: Core stays in VRAM always (uses a small model).

**`cpu_fallback`**: Core runs on CPU when GPU is busy with a worker. Slower but always available.

**`unload_during_batch`**: Core is fully unloaded during batch WorkOrder execution. Wakes back up to summarize results.

### Resource Hints Per Step

Each step in a WorkOrder can declare resource requirements:

```yaml
requires_gpu: true
vram_mb: 9000
model_load_policy: "unload_after"   # or "keep_warm" in datacenter
max_concurrency: 1                  # on a consumer GPU
```

Locally, you unload between steps. In datacenter, you keep hot pools. Same spec.

## Switching Protocol

The full switching sequence:

```
1. Receive request to activate head_id=Y (currently head_id=X is active)
2. If X == Y: no-op (already loaded)
3. Mark X as "draining" -- stop accepting new requests
4. Wait for X inflight requests to complete
5. Sleep/unload X (based on backend adapter)
6. Verify VRAM is freed (poll if needed)
7. Wake/load Y (based on backend adapter)
8. Health check Y
9. Mark Y as ACTIVE
10. Proceed with step execution
```

## Head Manager API (Internal, Exposed for Debugging)

```
GET  /heads                     -> list heads + states (OFF/WARM/ASLEEP/ACTIVE)
POST /heads/{head_id}/wake
POST /heads/{head_id}/sleep?level=1|2
POST /heads/{head_id}/unload
```

