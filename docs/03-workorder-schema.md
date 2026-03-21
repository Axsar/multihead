# WorkOrder and StageResult Schemas

## Overview

The strict step contract is what makes "factory line" composition reliable. If every step speaks the same language, you can chain anything.

## WorkOrder Schema

A WorkOrder defines a complete job to execute.

```json
{
  "id": "wo_123",
  "goal": "Refactor repo module X and add tests",
  "inputs": [
    {
      "uri": "file://repo.zip",
      "sha256": "...",
      "mime": "application/zip"
    }
  ],
  "constraints": {
    "privacy": "local_only",
    "budget": {
      "max_seconds": 900,
      "max_cloud_usd": 2.00
    },
    "scopes": ["READ_REPO", "RUN_TESTS"]
  },
  "plan": [
    {"tool": "repo.unpack", "params": {}},
    {"tool": "code.analyze", "params": {"depth": "module"}},
    {"tool": "code.edit", "params": {"strategy": "small_diffs"}},
    {"tool": "tests.run", "params": {"cmd": "pytest -q"}},
    {"tool": "verify.diff", "params": {"max_changed_files": 20}}
  ]
}
```

### WorkOrder Fields

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | string | Unique run identifier |
| `goal` | string | Human-readable description of what the job does |
| `inputs` | array | Paths, URLs, inline text, artifact refs |
| `steps[]` | array | Ordered list of Step objects |
| `status` | enum | `queued` / `running` / `blocked` / `failed` / `done` / `cancelled` |
| `budgets` | object | `{tokens, time_s, cost_usd}` |
| `created_at` | string | ISO 8601 timestamp |

### Constraint Fields

| Field | Description |
|-------|-------------|
| `privacy` | `local_only` / `allow_cloud` |
| `budget.max_seconds` | Wall-clock time limit |
| `budget.max_cloud_usd` | Spending cap for remote API calls |
| `budget.max_tokens` | Token budget |
| `scopes` | Permission grants for the run |
| `network` | `off` / `on` |

### Scope Examples

```
READ_FILES:/projects/foo
WRITE_FILES:/projects/foo
RUN_SHELL
NETWORK:off          (default)
SECRETS:read:github_token   (optional)
```

## Step Schema

Each step within a WorkOrder plan.

### Step Fields

| Field | Type | Description |
|-------|------|-------------|
| `step_id` | string | Unique step identifier (auto-generated ULID if empty) |
| `name` | string | Human-readable step name |
| `head_id` | string | Which head/model to use (empty = auto-resolved by Router) |
| `prompt_template` | string | Jinja2-ish template or raw prompt |
| `input_refs` | array | Artifact refs from prior steps or `"user_input"` |
| `depends_on` | array | Explicit step_id dependencies for DAG ordering |
| `required_kind` | string? | `"llm"`, `"vlm"`, etc. — triggers Router auto-selection |
| `output_schema` | object | Expected output structure |
| `retry_policy` | object | `{max_attempts, backoff_ms}` |
| `checkpoint_mode` | enum | `sync` / `async` |
| `extra` | object | Arbitrary metadata (e.g., `use_tot`, `use_prm`) |
| `consensus` | object? | ConsensusConfig for multi-head voting on this step |
| `fallback` | array | Alternative head_ids if primary fails |
| `task_types` | array | Specific task types for capability routing (e.g., `["code_editing"]`) |
| `privacy` | enum? | `CONFIDENTIAL` / `INTERNAL` / `PUBLIC` — controls remote delegation |
| `budget` | object? | `{max_cost_usd, max_latency_ms}` — enables ACP remote delegation |
| `enable_reflection` | bool | Enable Actor-Evaluator-Reflect-Memory cycle (default: false) |
| `max_reflection_attempts` | int | Max refinement attempts via reflection (default: 3) |
| `deterministic_function` | string? | For deterministic solver: `"module.function"` |

### Gate Object

```json
{
  "gate": {
    "accept_if": [
      {"metric": "parse_success_rate", "op": ">=", "value": 0.995},
      {"metric": "chunk_coverage", "op": ">=", "value": 0.98}
    ],
    "retry": {
      "max_attempts": 2,
      "mutate_params": [
        {"chunk_chars": 1600, "overlap_chars": 160},
        {"chunk_chars": 1200, "overlap_chars": 120}
      ]
    },
    "fallback": {
      "tool": "alternate_tool",
      "params": {"mode": "strict"}
    },
    "on_fail": "abort | continue | skip | queue_review"
  }
}
```

### Step Commit Rule

A step is committed only when ALL of:

1. Output validates against schema
2. Artifacts are stored
3. Event is appended to the log

## StageResult Schema

What each step returns. This is the universal output contract.

```json
{
  "work_order_id": "wo_123",
  "stage": "tests.run",
  "status": "ok",
  "outputs": [
    {
      "uri": "file://artifacts/testlog.txt",
      "sha256": "...",
      "mime": "text/plain"
    }
  ],
  "metrics": {
    "pass_rate": 1.0,
    "runtime_ms": 48213
  },
  "trace": {
    "tool_version": "1.2.0",
    "model": "qwen3:8b",
    "params_hash": "..."
  },
  "warnings": [],
  "cost": {
    "cpu_ms": 0,
    "gpu_ms": 0,
    "cloud_usd": 0
  }
}
```

### StageResult Fields

| Field | Type | Description |
|-------|------|-------------|
| `work_order_id` | string | Parent WorkOrder ID |
| `stage` | string | Stage/tool name |
| `status` | enum | `ok` / `failed` / `skipped` |
| `outputs[]` | array | URIs + hashes of produced artifacts |
| `metrics` | object | Confidence, coverage, runtime, pass rates |
| `trace` | object | Provenance: tool version, model ID, params hash |
| `warnings` | array | What might be wrong |
| `cost` | object | CPU/GPU time, cloud USD spent |

## Tool Manifest

Tools are plugins, not hardcoded. A tool is declared via a manifest:

```yaml
tool: ocr.extract
version: 0.3.1
inputs_schema: schemas/ocr_extract_in.json
outputs_schema: schemas/ocr_extract_out.json
runner:
  type: docker
  image: ghcr.io/acp-tools/ocr:0.3.1
resources:
  gpu: optional
  vram_mb: 0
policy:
  scopes_required: [READ_FILES]
  sandbox: true
```

## Caching

Stage caching is by content hash:

```
cache_key = hash(inputs_hash + tool_version + params_hash)
```

If the cache key matches a prior run, skip the stage and reuse artifacts. This is what makes repeated runs efficient.

