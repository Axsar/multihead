# Changelog

All notable changes to MultiHead will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Resource monitoring with time-series history and sparklines (`resource_monitor.py`)
- Live-refresh dashboard mode (`/dashboard live [interval]`)
- Resource panels in dashboard (VRAM, RAM, disk with sparkline trends)

## [1.2.0] - 2026-03-02

### Added
- `multihead kb "query"` — one-line knowledge search from any terminal (FTS5)
- `multihead kb -s "query"` — embedding-based semantic search via sentence-transformers
- `multihead auth status` — JWT validity, expiry, ACP connectivity check
- Claim interaction tracking (`claim_interactions` table) — inbox dedup across sessions
- `--dry-run` flag for solve pipeline — decompose and show plan without executing
- TestGenerationHook wired into all entry points (CLI, SDK, API, MCP, shell)
- `solvers.yaml` auto-generation in `multihead init` with capability inference
- Interactive config editor (`/config interactive`) with Rich prompts
- SolvePipeline wired into all entry points (shell `/solve`, Engine SDK, API, MCP, CLI)
- Validator YAML hydration in `load_recipe()` — recipe validators now work
- SolveCoordinator test suite (22 tests for distributed consensus)
- Embedding search module with disk-cached vector index

### Changed
- Dashboard now shows resource panels with VRAM/RAM/disk sparklines
- Dashboard supports live auto-refresh mode (`/dashboard live`)
- Shell inbox uses interaction tracking (no more re-reading handled claims)
- README updated with new CLI commands and test count (2,800+)

### Fixed
- NULL signature claims causing Pydantic validation errors in auto-responder
- Validator configs in recipes remaining as raw dicts instead of Validator instances

## [1.1.0] - 2026-02-28

### Added
- Agent Terminal (`multihead shell`) — Rich REPL with 19 slash commands
- Python SDK (`from multihead import Engine`) — embeddable API
- Process manager for subprocess management (spawn/list/output/kill)
- Claude Agent SDK adapter with brain mode switching
- 12 in-process MCP tools for SDK sessions
- `/dashboard` observability command
- Responsive mode with proactive marketplace bidding
- Cloud marketplace with JWT refresh, listing registration, trust scoring
- `multihead daemon` headless service runner

### Changed
- Shell pipeline with PLUR (Plan, Learn, Use, Reflect) architecture
- Knowledge RAG integrated into every shell response

## [1.0.0] - 2026-02-21

Initial release of MultiHead.

### Features

#### Core
- Event-sourced durable execution (kill-9 resilient)
- Content-addressed artifact store (SHA-256 sharded)
- Hot-swappable GPU models (RTX 4090, 3060, Apple Silicon, CPU-only)
- Multi-head consensus voting (5 strategies including FIRST_TO_AHEAD)
- DAG and linear pipeline execution
- Privacy-aware routing with data sensitivity controls

#### Research Features
- Tree-of-Thoughts (BFS, DFS, Beam search strategies)
- Process Reward Models (step-level quality scoring)
- Reflection loops (Actor-Evaluator-Reflect-Memory cycle)
- Auto-Decomposition with DAG inference
- Recipe Learning (query experts, benchmark, adopt)
- Auto-Benchmarking (continuous performance monitoring)

#### Adapters
- Ollama (local models)
- vLLM (high-performance serving)
- HuggingFace Transformers (4-bit quantization)
- OpenAI API
- Claude Agent SDK
- Mock (testing)

#### Knowledge System
- Claim and event tracking (27,000+ claims)
- Component briefings
- Night Shift automated processing (15 stages)
- Narrative pipeline (markdown, git, chat extraction + Claude enhancer)
- Context packs for targeted knowledge retrieval

#### Integration
- BotVibes/ACP agent mesh (3-agent: Qwen, Claude worker, BotVibes coder)
- Claude Code MCP server (13 tools)
- Claude worker daemon with conversation threading
- REST API (localhost:7337)
- Python client
- RFQ marketplace delegation

#### CLI
- `multihead serve` — Start API server
- `multihead shell` — Interactive agent terminal
- `multihead solve` — Autonomous task solving
- `multihead init --auto` — Hardware detection and config generation
- `multihead doctor` — Diagnostic checks
- `multihead run <recipe>` — Execute pipeline recipes
- `multihead heads` — List model heads
- `multihead narrative` — Knowledge management
- `multihead mcp` — MCP server for Claude Code

#### Models Supported
- Qwen3-8B-Instruct (LLM, ~6GB VRAM)
- Qwen3-VL-7B (VLM, ~5GB VRAM)
- Qwen3-VL-32B-Thinking (VLM, ~18GB VRAM)
- GPT-4o Mini via OpenAI API
- Claude via Agent SDK

---

For older changes, see git history at https://github.com/Axsar/multihead
