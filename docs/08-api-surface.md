# API Surface

MultiHead exposes a REST API on `127.0.0.1:7337` (configurable via `MULTIHEAD_API_HOST` / `MULTIHEAD_API_PORT`). An MCP server (`multihead mcp`, stdio transport) is also available as an alternative interface for Claude Code integration.

---

## Health / Probes

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check. Returns `status`, `version`, `ready`, `heads_loaded`, `heads` list. |
| `GET` | `/healthz` | Liveness probe. Returns `{"status": "alive"}`. |
| `GET` | `/readyz` | Readiness probe. Checks `head_manager`, `event_store`, `orchestrator`. Returns 503 if any check fails. |

---

## Heads (`/heads`)

Manage model heads (load, unload, generate).

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/heads` | List all heads and their states (OFF/WARM/ASLEEP/ACTIVE). |
| `GET` | `/heads/{head_id}` | Get a single head's state. |
| `POST` | `/heads/{head_id}/wake` | Load/wake a head. |
| `POST` | `/heads/{head_id}/sleep` | Put a head to sleep. Query param: `level` (int, default 1). |
| `POST` | `/heads/{head_id}/unload` | Fully unload a head from memory. |
| `POST` | `/heads/{head_id}/generate` | Generate text through a head. |

**`POST /heads/{head_id}/generate` request body:**

```json
{
  "prompt": "string (required)",
  "temperature": 0.7,
  "max_tokens": 512,
  "images": ["base64-encoded string"]
}
```

---

## Runs (`/runs`)

Create and manage orchestrated pipeline runs.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/runs` | List all runs. |
| `POST` | `/runs` | Create and start a run from a recipe, goal, or inline work order. |
| `GET` | `/runs/{run_id}` | Get run status and progress. |
| `GET` | `/runs/{run_id}/results` | Detailed step results including consensus metrics. |
| `GET` | `/runs/{run_id}/events` | Get run events. Query param: `tail` (int, default 200). |
| `POST` | `/runs/{run_id}/cancel` | Cancel a running run. |
| `POST` | `/runs/{run_id}/replay` | Replay a run. Query param: `step_id` (optional, resume from step). |

**`POST /runs` request body:**

```json
{
  "recipe": "recipe-name",
  "goal": "high-level goal string",
  "inputs": {},
  "work_order": {}
}
```

Provide one of `recipe`, `goal`, or `work_order`. Returns `run_id`, `status`, `goal`, `total_steps`, `created_at`.

---

## Chat (`/chat`)

Agentic Core chat sessions.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Send a message. Creates a new session if `session_id` is omitted. |
| `GET` | `/chat/sessions` | List all chat sessions. |
| `GET` | `/chat/sessions/{session_id}` | Get session details and message history. |

**`POST /chat` request body:**

```json
{
  "message": "string (required, max 50000 chars)",
  "session_id": "optional existing session ID"
}
```

**Response:** `{ "session_id": "...", "response": "..." }`

---

## Knowledge (`/knowledge`)

Knowledge store: events, claims, records, and briefings.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/knowledge/events` | List knowledge events. Filters: `status`, `event_type`, `limit` (max 500). |
| `GET` | `/knowledge/events/{event_id}` | Get a specific knowledge event. |
| `POST` | `/knowledge/events` | Create a knowledge event. |
| `GET` | `/knowledge/claims` | List claims. Filters: `status`, `claim_type`, `scope_id`, `limit` (max 500). |
| `GET` | `/knowledge/claims/{claim_id}` | Get a specific claim. |
| `POST` | `/knowledge/claims` | Create a new claim. |
| `GET` | `/knowledge/records` | List ingested records. Filter: `limit` (max 500). |
| `GET` | `/knowledge/briefing` | Component briefing. Required: `component`. Optional: `scope_id`, `include_events`, `max_claims`, `max_events`. |

**`POST /knowledge/claims` request body:**

```json
{
  "claim_key": "h2v.vertical_layout.status",
  "statement": "string (required)",
  "subject_type": "component",
  "subject_id": "",
  "predicate": "has_state",
  "value": true,
  "value_type": "string",
  "claim_type": "fact",
  "claim_status": "accepted",
  "scope_type": "project",
  "scope_id": "h2v",
  "confidence": 0.9,
  "stability": "medium",
  "importance": 0.5,
  "rationale": "",
  "produced_by": "external",
  "tags": []
}
```

**`POST /knowledge/events` request body:**

```json
{
  "title": "string (required)",
  "summary": "",
  "event_type": "note",
  "event_status": "confirmed",
  "tags": [],
  "metrics": {},
  "produced_by": "external",
  "entities": [{"type": "component", "id": "balloon_layout"}]
}
```

---

## Decompose (`/decompose`)

LLM-powered task decomposition.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/decompose` | Decompose a goal into a hierarchical execution plan. |
| `POST` | `/decompose/refine` | Refine a single step into sub-steps. |

**`POST /decompose` request body:**

```json
{
  "goal": "string (required)",
  "context": "",
  "head_id": null,
  "max_depth": 4
}
```

**`POST /decompose/refine` request body:**

```json
{
  "node_id": "string (required)",
  "node_goal": "string (required)",
  "action_type": "",
  "target_files": [],
  "exploration_result": "",
  "head_id": null
}
```

---

## Solve (`/solve`)

Autonomous solve pipeline: decompose, route, execute.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/solve` | Run the full solve pipeline. |

**`POST /solve` request body:**

```json
{
  "task": "string (required)",
  "strategy": "first_to_ahead",
  "max_steps": 20,
  "max_depth": 3,
  "timeout": 240.0,
  "enable_marketplace": false,
  "enable_tests": false,
  "dry_run": false
}
```

**Response:** `run_id`, `status`, `output`, `confidence`, `steps_total`, `steps_succeeded`, `steps_failed`, `duration_seconds`, `plan_steps`, `parallel_steps`, `dry_run`.

---

## Consensus (`/consensus`)

Multi-head consensus execution.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/consensus/execute` | Execute a consensus query across multiple heads. |
| `GET` | `/consensus/strategies` | List available consensus strategies. |

**Strategies:** `majority`, `weighted`, `unanimous`, `threshold`, `first_to_ahead`.

**`POST /consensus/execute` request body:**

```json
{
  "prompt": "string (required)",
  "heads": [
    {
      "head_id": "qwen-llm",
      "prompt_template": "",
      "weight": 1.0,
      "required": true,
      "extract_fields": []
    }
  ],
  "strategy": "majority",
  "threshold": 0.5,
  "output_schema": {},
  "cross_modal": false,
  "fail_on_disagreement": false,
  "timeout_seconds": 30.0,
  "first_to_ahead": {
    "k_margin": 3,
    "max_samples": 25,
    "min_samples": 3,
    "stall_threshold": 9,
    "red_flag_max_tokens": 700,
    "red_flag_must_parse": false
  }
}
```

---

## Night Shift (`/nightshift`)

Background knowledge consolidation pipeline.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/nightshift/trigger` | Trigger a Night Shift run in the background. Query params: `debug` (bool), `concurrency` (int). |
| `GET` | `/nightshift/status` | Get Night Shift status and live progress. |
| `GET` | `/nightshift/report` | Get the last Night Shift report. |

---

## Artifacts (`/artifacts`)

Content-addressed artifact storage.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/artifacts` | Upload an artifact (multipart file upload). Query params: `name`, `media_type`. |
| `GET` | `/artifacts/{artifact_id}` | Download artifact bytes. |
| `GET` | `/artifacts/{artifact_id}/meta` | Get artifact metadata. |

---

## Packs (`/packs`)

Context packs for knowledge-grounded prompting.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/packs` | List all built context packs. |
| `POST` | `/packs/build` | Build a custom context pack. Body: `purpose`, `filters`, `budgets`. |
| `POST` | `/packs/build-standard` | Build all 5 standard Night Shift packs. |
| `GET` | `/packs/{pack_id}` | Get a specific pack's contents. |

---

## Config (`/config`)

Runtime configuration (mutable, persisted to JSON).

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/config` | Get current runtime config. |
| `POST` | `/config/set` | Set a config value. Body: `{ "key": "...", "value": "..." }`. |

---

## ACP (`/acp`)

Proxy to the BotVibes ACP task queue and cloud marketplace.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/acp/tasks` | Poll for available tasks. Query params: `capability` (default `com.claude.code`), `limit`. |
| `POST` | `/acp/tasks` | Create a new ACP task. |
| `GET` | `/acp/tasks/{task_id}` | Get task details. |
| `POST` | `/acp/tasks/{task_id}/claim` | Atomically reserve + dispatch a task. |
| `POST` | `/acp/tasks/{task_id}/result` | Submit results for a completed task. |
| `POST` | `/acp/marketplace/procure` | Procure a service via cloud marketplace RFQ workflow. |
| `GET` | `/acp/marketplace/search` | Search cloud marketplace for providers. Query params: `capability`, `limit`. |

**`POST /acp/tasks` request body:**

```json
{
  "required_capability": "com.claude.code",
  "payload_ref": "JSON string or artifact reference",
  "target_agent_id": null,
  "conversation_id": null,
  "input_schema": "application/json",
  "output_schema": "application/json",
  "priority": "normal"
}
```

**`POST /acp/marketplace/procure` request body:**

```json
{
  "capability": "string (required)",
  "payload": "string (required)",
  "max_price": null,
  "max_latency_ms": null,
  "min_quality": null,
  "quote_timeout": 30.0
}
```

---

## Mesh (`/v1`)

Local mesh protocol for multi-node communication, protected by bearer token when `MULTIHEAD_MESH_SECRET` is set.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/health` | Mesh-specific health check. |
| `GET` | `/v1/node` | Node info: `node_id`, version, capability count. |
| `GET` | `/v1/capabilities` | List this node's capabilities. Query param: `kind` (filter by llm/vlm/embed). |
| `POST` | `/v1/tasks` | Submit a task to this node for execution. |
| `GET` | `/v1/claims` | List shared claims for mesh replication. Query params: `since` (ISO timestamp), `scope_id`, `limit`. |
| `POST` | `/v1/claims/import` | Import claims from a peer node (verifies signatures, skips duplicates). |

**`POST /v1/tasks` request body:**

```json
{
  "task_id": "",
  "capability_kind": "llm",
  "model": "",
  "prompt": "",
  "params": {}
}
```

---

## Dashboard & Metrics

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/dashboard` | HTML monitoring dashboard. |
| `GET` | `/metrics` | Export metrics. Query param: `format` (`json` or `prometheus`). |

---

## WebSocket (`/ws`)

Real-time streaming over WebSocket.

| Endpoint | Description |
|----------|-------------|
| `ws://host:7337/ws/runs/{run_id}/events` | Stream run events in real-time. Auto-closes on run completion, idle timeout (5 min), or max duration (30 min). |
| `ws://host:7337/ws/chat/{session_id}` | Token-streaming chat. Send `{"message": "...", "stream": true}` for token-by-token streaming, or `{"message": "..."}` for full responses. |

---

## MCP Server (Alternative Interface)

The MCP server (`multihead mcp`) exposes the same functionality over stdio transport for Claude Code integration. It proxies to the REST API at `localhost:7337` via httpx. Tools include: `multihead_chat`, `multihead_generate`, `multihead_heads`, `multihead_swap_head`, `multihead_run_recipe`, `multihead_run_status`, `multihead_knowledge`, `multihead_config`, `multihead_briefing`, `multihead_delegate_claude`, `multihead_solve`, `multihead_decompose`, and others.
