# MultiHead Roadmap

Current version: **1.3.52**

---

## v0.1 -- Durable Head-Swap Orchestrator (DONE)

Single-machine orchestrator that hot-swaps GPU models between steps and survives kill -9.

- Event-sourced durable execution with append-only JSONL + replay on restart
- Head registry with manifests (id, type, adapter, VRAM requirements)
- GPU mutex enforcing one active heavy head at a time
- Content-addressed artifact store (SHA-256 sharded)
- Ollama and vLLM adapters
- WorkOrder + StepGraph schema with deterministic execution
- Full trace logging (prompt, inputs, head, timings, outputs, errors)
- CLI: `multihead run <recipe> --input ...`

---

## v0.2 -- Night Shift + Context Packs (DONE)

Offline memory refinery that builds institutional knowledge while you sleep.

- 15-stage Night Shift pipeline (cron-triggered)
- Claim/Event extraction and canon management
- Context Pack builder with token budgeting (Active Projects, Decisions, Constraints, Glossary, Open Loops)
- Daily/weekly briefing generation
- Cross-topic linking and index rebuild (embeddings + keyword)
- Knowledge Store: SQLite-backed claims, events, records, links with confidence and evidence tracking

---

## v0.3 -- Agentic Core (DONE)

Interactive chat-first mode with tools, web search, and slash commands.

- Always-on core LLM with structured action types (SAY, CALL_TOOL, CREATE_WORKORDER, etc.)
- Interactive mode (chat-first) and batch mode (factory-first)
- VRAM policy management (core_mode, worker_load_policy)
- Web fetch + web search tools for local LLMs
- Slash commands for runtime control

---

## v0.4 -- Research Features (DONE)

K-step reasoning capabilities for systematic problem solving. 178 new tests.

- **Auto-Decomposition**: LLM-driven task breakdown with DAG inference, atomicity validation, completeness checking
- **Tree-of-Thoughts**: BFS, DFS, and Beam search over alternative reasoning paths
- **Process Reward Models**: Step-level quality scoring (LLM, Rubric, Composite scorers)
- **Reflection Loops**: Actor-Evaluator-Reflect-Memory cycles for self-correction across attempts
- **Recipe Learning**: Query BotVibes experts, benchmark via Orchestrator, adopt superior recipes
- **Auto-Benchmarking**: Continuous performance monitoring bridging HeadManager and capability discovery

---

## v0.5 -- Portable for Friends (DONE)

One-command setup that works on any hardware profile.

- `bash scripts/install.sh` detects OS, GPU, and Python version
- `multihead init --auto` generates heads.yaml for detected hardware
- `multihead doctor` runs full diagnostics
- Safe defaults: mock heads enabled, real models opt-in
- Adapter validation on startup (graceful fallback when dependencies missing)
- Hardware templates for NVIDIA, Apple Silicon, CPU-only
- Zero-config: works out of the box with no `.env` edits

---

## v1.0 -- Full Local Mesh Protocol (DONE, 2022 tests)

Multi-agent collaboration with resilience and observability.

- 10-step autonomous solve workflow (intake, decompose, consensus, route, execute, learn)
- 5 consensus strategies: MAJORITY, WEIGHTED, UNANIMOUS, THRESHOLD, FIRST_TO_AHEAD
- ACP Bridge for BotVibes marketplace delegation (WebSocket push + HTTP fallback)
- Capability registry and dynamic discovery (22 tests)
- Multi-candidate routing with fallbacks and circuit breakers
- Knowledge feedback loop: router learns from past executions
- Presence discovery via mDNS (zeroconf)
- DAG execution with parallel steps (CPU concurrent, GPU serialized)
- Session poller (WebSocket + HTTP hybrid)
- Resilience: retry policies, circuit breakers, graceful degradation

---

## v1.1 -- Agent Terminal + Python SDK (DONE, 2160 tests)

Rich interactive shell and embeddable Python API.

- **Layer 2: `multihead shell`** -- Rich REPL with PLUR formatting
- 19 slash commands: /config, /tools, /heads, /wake, /sleep, /swap, /status, /knowledge, /session, /sessions, /mesh, /spawn, /ps, /output, /kill, /collab, /help
- Process manager: spawn/list/output/kill subprocesses (max 10)
- **Layer 1: `Engine` class** -- Python SDK for embedding MultiHead in any application
- Knowledge RAG integration in interactive chat

---

## v1.2 -- Claude SDK Integration (DONE, 112 tests)

Claude as a native MultiHead head with full brain-swap support.

- `claude-sdk` adapter using `claude-agent-sdk` Python package
- Claude sessions as first-class heads (load, unload, swap like any other)
- Brain swap: switch between local LLM and Claude mid-conversation
- 12 in-process MCP tools for SDK sessions (`sdk_mcp_tools.py`)
- Claude worker daemon for headless `claude -p` subprocess execution
- Multi-turn support via `--resume <session_id>`
- `multihead_delegate_claude` MCP tool for task delegation

---

## v1.3 -- Open-Source Polish (CURRENT, v1.3.52, 2905 tests)

Documentation, history tooling, and marketplace reliability.

- Comprehensive docs: QUICKSTART.md, CONTRIBUTING.md, architecture guides
- Session harvester: extract knowledge from Claude session history
- Narrative pipeline: auto-ingest from git, markdown, chat transcripts, agent results
- Claude enhancement: deep extraction via worker swarm (implicit claims, relationships, risks)
- Marketplace fixes: prevent self-bidding, batch-ack stale events, proactive JWT refresh
- Shell reliability: clean exit handling, shutdown timeouts
- 22 MCP tools bridging Claude Code, Claude Desktop, and Cowork
- GitHub Issues integration: bidirectional issue tracking in `solve`
- `multihead kb` quick search (FTS5 + semantic)
- Embedding-based semantic search

---

## Future -- v2.0 Ideas

Potential directions beyond v1.3:

- **Federated Marketplace**: Cross-network agent discovery and task routing beyond single BotVibes instance
- **Model Fine-Tuning from Knowledge**: Use knowledge.db claims and execution traces to fine-tune local models on your domain
- **Multi-Machine Scheduling**: Distributed GPU pool with job routing by VRAM/model availability
- **Streaming Execution**: WebSocket token/log streaming for real-time pipeline monitoring
- **Plugin System**: Third-party head adapters and tool packages as installable plugins
- **Evaluation Harness**: Golden test sets with per-stage metrics for regression detection
- **Visual Pipeline Editor**: Web UI for building and monitoring YAML recipes
- **Mobile/Edge Deployment**: Lightweight agent mode for constrained devices that delegates heavy work upstream
