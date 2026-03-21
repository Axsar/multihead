# MultiHead Architecture

## 4-Layer Architecture

MultiHead is **local-first** -- it runs on your infrastructure (laptop, LAN, your control). Each layer works standalone; no forced dependencies upward.

```
Layer 4: GitHub Issues (Optional)
         Bidirectional issue tracking, subtask creation, result posting
         ^
Layer 3: BotVibes Marketplace (Optional)
         External experts, privacy-preserving delegation via ACP
         ^
Layer 2: MultiHead Solve (Multi-Agent Consensus)
         Qwen + Claude Sessions, knowledge.db coordination, 5 voting strategies
         ^
Layer 1: Individual Solvers
         Qwen LLM/VLM, deterministic tools, Python scripts, YAML recipes
```

- **Layer 1** works standalone (zero cloud dependencies)
- **Layer 2** adds multi-session consensus (still local: shared knowledge.db)
- **Layer 3** adds marketplace delegation (optional, privacy-aware)
- **Layer 4** adds GitHub integration (optional, requires `gh` CLI)

## Core Components

| Component | What It Does |
|-----------|-------------|
| **Event Store** | Append-only JSONL per run + SQLite index -- kill -9 resilient, full audit trail |
| **Artifact Store** | SHA-256 content-addressed storage with 2-level directory sharding |
| **Head Manager** | GPU mutex, adapter lifecycle (load/unload/sleep), circuit-breaker resilience |
| **Weighted Router** | Multi-factor head selection (see below) |
| **DAG Executor** | Parallel execution respecting step dependencies |
| **Knowledge Store** | SQLite-backed semantic memory with confidence and evidence tracking |
| **Orchestrator** | Run state machine: plan steps, dispatch to heads, checkpoint, resume |

## The Three Roles

### 1. The Agentic Core (Always-On Brain)

A small local LLM (or Claude via Agent SDK) for communication, planning, and tool arbitration.

**What it does:**

- Holds the conversational thread and pinned context
- Turns user intent into WorkOrders
- Decides which tools to call next
- Monitors execution and reacts to failures (retry, switch head, ask user)
- Produces final explanations in plain language

**What it does NOT do:**

- Run expensive models directly
- Do huge-context reasoning (that is delegated to worker heads)
- Do high-precision computation (that is tools + checkers)

The core's output is validated JSON (action types), not free-form text.

### 2. The Executor (Factory Line Engine)

The deterministic runtime that runs steps reliably:

- Executes tools (Python, shell, model calls)
- Writes artifacts + metrics + trace
- Retries/fallbacks based on policy
- Caches by hash

The Executor can run with the core muted/offloaded once a WorkOrder is launched.

### 3. The Head Manager (VRAM-Aware)

Manages model lifecycle with GPU awareness:

- Enforces "only one GPU-heavy head active" (GPU lock)
- Implements sleep/wake/unload policies
- Health checks + circuit breakers + timeouts

**VRAM Policy Config:**

```yaml
core_mode: keep_loaded | cpu_fallback | unload_during_batch
worker_load_policy: per_stage | keep_warm
```

## Core Action Types

The Agentic Core responds with one of these validated action types:

| Action | Description |
|--------|-------------|
| `CALL_TOOL` | Single tool call with name and params |
| `CREATE_WORKORDER` | Spawn a multi-step pipeline |
| `PAUSE_AND_ASK` | Request a user decision before continuing |

Plain-text responses (conversational replies) are returned directly without an action wrapper. Any unrecognized action is rejected with a retry request.

## Weighted Router

The Router selects the best head for each step using a multi-factor weighted score:

| Factor | Weight | Description |
|--------|--------|-------------|
| Capability match | 40 | Head supports the required `kind` or `task_types` |
| Active bonus | 40 | Already loaded -- avoids GPU swap cost |
| Circuit breaker | 30 | Healthy heads preferred (closed > half_open >> open) |
| Accuracy score | 20 | Higher benchmark accuracy |
| VRAM fit | 15 | Head fits in available GPU memory |
| Error rate | 10 | Lower recent error rate |
| Cost | 10 | Lower cost per call |
| Latency | 5 | Lower recent latency |
| Learned preference | 5 | Meta-reasoning feedback from prior runs |

Steps can declare `required_kind` (e.g. "llm", "vlm") or `task_types` (e.g. "code_generation") for auto-routing. Hard filter: heads that do not match `required_kind` are excluded before scoring.

## DAG Executor

The `DAGExecutor` builds a dependency graph from step definitions and executes independent steps in parallel:

- Dependencies inferred from `depends_on` fields and `input_refs` artifact chains
- CPU-bound steps run concurrently (up to `max_parallel_cpu`, default 8)
- GPU steps are serialized through the Head Manager's GPU mutex
- Delegates each step to `orchestrator._execute_step()` so DAG runs emit identical events and persist artifacts like linear runs
- Auto-decomposition can infer DAG edges from file dependencies, action ordering, and artifact flow

## Knowledge Store

SQLite-backed semantic memory (`knowledge.db`) with four entity types:

| Entity | Purpose |
|--------|---------|
| **Claims** | Factual assertions with confidence scores, scoping, and lifecycle (proposed/accepted/superseded/retracted) |
| **Events** | Timestamped occurrences (task completions, errors, deployments) with metrics |
| **Records** | Structured data snapshots (config, benchmarks, state) |
| **Links** | Typed relationships between entities (supports/contradicts/depends_on) |

Features:
- FTS5 full-text search and embedding-based semantic search
- Evidence tracking: claims accumulate supporting/contradicting evidence over time
- Scoping: claims are namespaced by `scope_id` (e.g. "h2v", "multihead")
- Night Shift: 18-stage offline refinery that deduplicates, assesses confidence, and builds context packs
- Narrative Pipeline: auto-ingests from git commits, markdown docs, chat transcripts, and agent results
- External client: any Python process can read/write via REST API at localhost:7337

## K-Step Reasoning

Research-grade reasoning features, each independently toggleable per step via `step.extra`:

### Tree-of-Thoughts (ToT)

Systematic exploration of alternative reasoning paths. Three search strategies:

- **BFS** -- breadth-first, explores all branches at each depth
- **DFS** -- depth-first, follows most promising path
- **Beam** -- retains top-k candidates at each level

LLM generates candidate thoughts; LLM evaluates state quality. Auto-enabled for steps tagged as exploratory.

### Process Reward Models (PRM)

Step-level quality scoring (vs outcome-only validation). Three scorer types:

- **LLM** -- prompts a model to score each reasoning step
- **Rubric** -- domain-specific checklists (deterministic)
- **Composite** -- combines multiple scorers

Aggregation modes: `min` (weakest link), `avg`, `product` (probability chain). Auto-enabled for implementation steps.

### Reflection Loops

Actor-Evaluator-Reflect-Memory cycle for self-correction:

1. Actor produces an output
2. Evaluator scores it against criteria
3. Reflector analyzes failures and generates feedback
4. Memory tracks previous attempts to avoid repeated mistakes
5. Refined prompt is re-attempted (up to `max_reflection_attempts`)

Auto-enabled for verification steps. Configurable via `enable_reflection` and `max_reflection_attempts` on StepDef.

### Auto-Decomposition

LLM-driven task breakdown with four validators:

- **StepDependencyAnalyzer** -- infers DAG edges (file deps, action ordering, artifact flow)
- **AtomicityValidator** -- ensures each step has a single target (MAKER principle)
- **CompletenessValidator** -- checks goal keyword coverage and standard phases
- **ResearchFeatureIntegrator** -- auto-enables ToT/PRM/Reflection based on step type

## Consensus Strategies

Five voting strategies for multi-head consensus (`multihead solve`):

| Strategy | Behavior |
|----------|----------|
| `MAJORITY` | Simple majority wins |
| `WEIGHTED` | Votes weighted by head accuracy scores |
| `UNANIMOUS` | All heads must agree |
| `THRESHOLD` | Passes if agreement exceeds a configurable threshold |
| `FIRST_TO_AHEAD` | Dynamic sampling with red-flag pre-filtering and k-margin convergence |

## Two Operating Modes

### Interactive Mode (`multihead shell`)

- Core model stays loaded (or on CPU)
- Worker models load/unload per step
- User converses with the core while workers are idle
- Rich REPL with 19 slash commands, knowledge RAG, process management

### Batch Mode (`multihead serve` / `multihead solve`)

1. Core generates plan and launches WorkOrder
2. Core enters low-VRAM monitor mode (unload or swap to tiny CPU model)
3. Executor runs steps against the DAG
4. Core wakes back up to summarize results

## Adapters

Models are accessed through normalized adapters:

| Adapter | Backend |
|---------|---------|
| `transformers` | HuggingFace Transformers (4-bit quantization) |
| `ollama` | Ollama server |
| `vllm` | vLLM server |
| `openai` | OpenAI-compatible API |
| `claude` | Claude CLI (`claude -p`) |
| `claude_agent_sdk` | Claude via Agent SDK (brain mode) |
| `claude_session` | Multi-session consensus via shared knowledge.db |
| `botvibes` | BotVibes marketplace delegation |
| `embedding` | Embedding models (semantic search) |
| `deterministic` | Pure Python functions (no LLM) |
| `mesh` | Remote mesh peer |
| `mock` | Testing stubs |

## Data Layout

```
$MULTIHEAD_DATA_DIR/           # ~/.multihead by default
  knowledge.db                 # SQLite: claims, events, records, links
  runs/
    <run_id>/
      workorder.json           # Frozen work order
      events.jsonl             # Append-only event log
      artifacts/               # SHA-256 content-addressed outputs
  context_packs/               # Semantic context bundles (Night Shift output)
  multihead.log                # Runtime log
```

## CLI Reference

```bash
# Core
multihead serve              # Start API server (localhost:7337)
multihead shell              # Interactive agent terminal (Rich REPL)
multihead solve "<task>"     # Autonomous multi-agent task solving
multihead chat               # Interactive chat with Agentic Core
multihead daemon             # Run headless marketplace services

# Inspection
multihead heads              # List registered heads and states
multihead info               # System overview: config, hardware, knowledge
multihead status [run_id]    # Show run status or list all runs
multihead inspect <run_id>   # Inspect a run's events and artifacts
multihead doctor             # Run diagnostic checks

# Knowledge
multihead kb "query"         # Quick knowledge search (FTS5)
multihead kb "query" -s      # Semantic search (embedding-based)
multihead knowledge claims   # List claims
multihead knowledge events   # List events

# Pipelines
multihead run <recipe>       # Run a YAML pipeline
multihead export [run_id]    # Export run or project as zip

# Operations
multihead auth status        # Check JWT validity + ACP connectivity
multihead nightshift run     # Run Night Shift (15-stage memory refinery)
multihead packs build        # Build context packs
multihead narrative ingest   # Ingest git/chat/agent/markdown data
multihead consensus run      # Run multi-head consensus

# Setup
multihead init               # Initialize config
multihead init --auto        # Auto-detect hardware, generate config
multihead init --mesh        # Enable multi-session collaboration
multihead mcp                # Start MCP server (stdio, sse, or streamable-http)

# Mesh
multihead mesh discover      # Discover peers on the network
multihead mesh peers         # List known mesh peers

# Recipes
multihead recipes list       # List learned recipes
```

## Key Source Files

| File | Purpose |
|------|---------|
| `src/multihead/orchestrator.py` | Run state machine |
| `src/multihead/head_manager.py` | GPU mutex, head swapping |
| `src/multihead/router.py` | Weighted head selection |
| `src/multihead/dag_executor.py` | Parallel step execution (DAG) |
| `src/multihead/agentic_core.py` | Agentic Core (planning, tool dispatch) |
| `src/multihead/consensus.py` | Multi-head consensus engine |
| `src/multihead/knowledge_store.py` | SQLite knowledge store |
| `src/multihead/models.py` | StepDef, WorkOrder, RunState, enums |
| `src/multihead/shell.py` | Agent Terminal (Rich REPL) |
| `src/multihead/mcp_server.py` | MCP server (22 tools) |
| `src/multihead/auto_decomposition.py` | Auto-decomposition with DAG inference |
| `src/multihead/tree_of_thoughts.py` | Tree-of-Thoughts search |
| `src/multihead/process_reward_models.py` | Process Reward Models |
| `src/multihead/reflection.py` | Reflection loops |
| `src/multihead/adapters/` | All adapter implementations |
| `config/heads.yaml` | Head registry |
| `config/recipes/` | YAML pipeline definitions |
