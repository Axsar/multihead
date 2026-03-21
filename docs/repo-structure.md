# MultiHead Repository Structure

*Generated 2026-03-20 — 75K+ lines of Python across 200+ files*

---

## Top Level

```
multihead/
  src/multihead/           # Main package (75K lines)
  tests/                   # 166 test files, 1,446 tests
  config/                  # YAML configs (heads, solvers, recipes, templates)
  docs/                    # Documentation
  pyproject.toml           # Package definition (MIT license, Python 3.11+)
  .env                     # Runtime config (data dir, API keys)
```

---

## Source: `src/multihead/`

### Core Engine

```
orchestrator/              (1,445 lines, 6 files)
  _execution.py            — Step execution with reflection, ToT, PRM hooks
  _state.py                — Run state machine (PENDING → RUNNING → DONE)
  __init__.py              — Orchestrator class facade

head_manager.py            (410 lines) — VRAM-aware GPU mutex, circuit breakers
router/                    (1,260 lines, 7 files)
  _core.py                 — Weighted multi-factor scoring
  _scoring.py              — Multi-factor head ranking (availability, reliability, resource fit, latency, history)
  _filtering.py            — Capability matching, VRAM filtering
  _knowledge.py            — Knowledge-informed routing
  _marketplace.py          — BotVibes delegation fallback
  _discovery.py            — Dynamic capability lookup

dag_executor.py            (191 lines) — Parallel DAG execution, GPU serialization
event_store.py             (201 lines) — Append-only JSONL + SQLite index
artifact_store.py          (157 lines) — SHA-256 content-addressed storage
models.py                  (650 lines) — RunState, StepDef, WorkOrder, HeadManifest, enums
config.py                  (456 lines) — Settings, heads.yaml parsing
vram_policy.py             — VRAM management policies
resilience.py              — Circuit breakers, retry logic
resource_monitor.py        (250 lines) — VRAM/CPU/memory tracking
```

### Adapters (14 model types)

```
adapters/                  (3,559 lines, 19 files)
  base.py                  — HeadAdapter abstract base class
  transformers_adapter.py  — HuggingFace local LLMs (4-bit quantization)
  ollama.py                — Ollama API (local/remote)
  vllm.py                  — vLLM inference server
  openai_adapter.py        — OpenAI real-time + batch API (with stall recovery)
  anthropic_adapter.py     — Anthropic Messages API (real-time + batch)
  claude_adapter.py        — Claude Code CLI
  claude_agent_sdk.py      — Claude Agent SDK (brain mode)
  claude_session.py        — Multi-session consensus via knowledge.db
  embedding.py             — Sentence-transformers vector embeddings
  botvibes_adapter/        — BotVibes marketplace delegation (RFQ, escrow)
  mesh_adapter.py          — Peer-to-peer LAN mesh
  deterministic_adapter.py — Pure Python tools (no ML)
  mock.py                  — Testing mock
```

### Knowledge System

```
knowledge_store/           (2,319 lines, 12 files)
  _schema.py               — SQLite schema (claims, events, records, conflicts, FTS5)
  _store.py                — KnowledgeStore facade + migrations
  _claims.py               — Claim CRUD, lifecycle (proposed→corroborated→superseded)
  _events.py               — Knowledge events (decisions, milestones, tasks)
  _records.py              — Raw records (evidence sources)
  _inbox.py                — Unread messages directed at agents
  _claims_search.py        — Full-text + semantic search
  _claims_queries.py       — Advanced queries (presence, staleness)
  _links.py                — Claim-to-claim dependencies
  _helpers.py              — Normalization (timestamps, producer ID)
  _retry.py                — Transactional retry

knowledge_models.py        (373 lines) — Pydantic models: Claim, Provenance, Evidence
knowledge_hook.py          (491 lines) — Auto-inject knowledge into agent context
claim_fusion.py            (571 lines) — Cross-channel triangulation engine
claim_corroboration.py     — Independent verification logic
scope_inference.py         — Auto-detect scope from claim key/text
embedding_search.py        (215 lines) — Vector search over claims
```

### Night Shift Pipeline (26 stages)

```
night_shift/               (2,780 lines, 7 files)
  pipeline.py              — Orchestration, checkpointing, dependency-aware execution
  models.py                — Stage definitions with depends_on DAG
  stages_early.py          — PRODUCE: harvest, chunk, extract (stages 0-8)
  stages_advanced.py       — SCAN: behavioral, git, CI (stages 9-11)
  stages_late.py           — ANALYZE + OUTPUT: consistency, fusion, staleness, report (stages 12-25)
  checkpoint.py            — Resumable execution state

  Stages:
    0  session_harvest          — Harvest CLAUDE.md + conversation transcripts
    1  select_input_window      — Time-based record selection
    2  normalize_chunk          — Chunking + dedup
    3  hot_signals              — Word frequency (lightweight)
    4  entity_extraction        — Named entities via LLM (batch)
    5  topic_assignment         — Topic clustering via LLM (batch)
    6  event_extraction         — Events via LLM (batch)
    7  claim_extraction         — Claims via LLM (batch, v2.0 prompt)
    8  behavioral_code_analysis — LLM analyzes what code DOES (4 repos)
    9  narrative_fusion         — Git history → claims (4 repos)
    10 ci_results               — GitHub Actions → claims (4 repos)
    11 consistency_check        — Contradiction detection (topic-grouped)
    12 conflict_resolution      — LLM-judged resolution (full text + channel hierarchy)
    13 file_path_anchoring      — Regex extract file paths from conversation claims
    14 claim_fusion             — Cross-channel triangulation (convergence scoring)
    15 staleness_sweep          — Git SHA comparison across repos
    16 open_loops               — Unresolved questions from DB
    17 canon_pack_build         — Context packs for agents
    18 daily_brief              — Daily summary from live DB
    19 weekly_rollup            — Weekly trends
    20 cross_topic_links        — Cross-scope file + topic connections
    21 index_rebuild            — Keyword index from all claims
    22 publish_report           — Report with needs_human conflicts
    23 solver_discovery         — Capability scanning across codebases
    24 recipe_learning          — Expert recipe query + benchmarking
    25 backlog_sweep            — Process all claims through analysis
```

### Multi-Agent Coordination

```
consensus/                 (1,059 lines, 5 files)
  — 5 strategies: MAJORITY, WEIGHTED, UNANIMOUS, THRESHOLD, FIRST_TO_AHEAD

auto_decomposition/        (947 lines, 6 files)
  decomposer.py            — DAG inference from file I/O patterns
  dependency.py            — Write-after-read, test-after-edit detection
  validators.py            — Atomicity (m=1) + completeness checking
  _consensus.py            — Multi-model decomposition voting
  research.py              — Auto-enable ToT/PRM/Reflection per step

solve/                     (1,103 lines, 6 files)
  coordinator.py           — Multi-agent solve orchestration
  discovery.py             — Find capable solvers for a task
  prompts.py               — Solve prompt templates

solve_direct.py            (930 lines) — CLI-driven solve (mh-solve start/step/finalize)
```

### Research Features

```
tree_of_thoughts/          (695 lines, 6 files)
  engine.py                — ToT orchestrator
  searcher.py              — BFS, DFS, Beam search
  generators.py            — LLM-based thought generation
  evaluators.py            — State evaluation + scoring

process_reward_models/     (570 lines, 3 files)
  scorers.py               — LLM scorer, Rubric scorer, Composite
  models.py                — PathScore, StepScore, Quality enum

reflection.py              (432 lines) — Actor-Evaluator-Reflect-Memory cycles
tot_integration.py         — Wire ToT into orchestrator
```

### Extractors

```
extractors/                (1,907 lines, 10 files)
  base.py                  — BaseExtractor, BatchPending, BatchPartialError, map_generate
  entity_extractor.py      — Named entity extraction via LLM
  topic_assigner.py        — Topic assignment via LLM
  event_extractor.py       — Event extraction via LLM
  claim_extractor.py       — Claim extraction (v2.0 durable knowledge prompt)
  consistency_checker.py   — Topic-grouped contradiction detection
  code_reader.py           — AST scan + behavioral LLM analysis
  ci_extractor.py          — GitHub Actions CI results
  test_results_extractor.py — Local pytest/JUnit results
```

### Narrative Pipeline

```
narrative/                 (4,157 lines, 18 files)
  pipeline.py              — Git ingest → fuse → store → context generation
  fusion.py                — Claim merging + deduplication
  state_engine.py          — Extraction state tracking
  verification.py          — Evidence validation
  confidence.py            — Heuristic + data-driven confidence scoring
  context_gen.py           — Generate daemon context from claims + events
  source_extractors/
    markdown_extractor.py  — Heuristic markdown → claims
    git_extractor.py       — Git commits → claims (diff + message channels)
    chat_extractor.py      — Chat logs → dialog turns
    agent_extractor.py     — Agent logs → task decisions
  claude_enhancer/         — Claude-enhanced deep extraction via ACP worker swarm
```

### Interfaces

```
cli/                       (4,192 lines, 15 files)
  _knowledge.py            — multihead kb, briefing, deposit, nightshift
  _interactive.py          — multihead chat, shell
  _solve.py                — multihead solve
  _server.py               — multihead serve
  _init.py                 — multihead init (hardware detection)
  _mesh.py                 — multihead mesh
  _consensus.py            — multihead consensus
  _discover.py             — multihead discover
  _narrative.py            — multihead narrative
  _recipes.py              — multihead recipes
  _skills.py               — multihead skills
  _auth.py                 — multihead auth (BotVibes)
  _core.py                 — multihead status, help
  _helpers.py              — Shared utilities

api/                       (4,110 lines, 27 files)
  app.py                   — FastAPI app (localhost:7337)
  routes_runs.py           — POST /runs, GET /runs/{id}
  routes_solve.py          — POST /solve (consensus)
  routes_knowledge.py      — Claims/events CRUD + search
  routes_heads.py          — Head management
  routes_consensus.py      — Voting endpoints
  routes_ws.py             — WebSocket live streaming
  routes_acp/              — ACP/BotVibes integration (5 files)
  (+ 15 more route modules)

mcp_server/                (2,362 lines, 10 files)
  — 25+ MCP tools for Claude Code integration (FastMCP 2.14, stdio)
  — Tools: chat, generate, heads, swap, knowledge, deposit, briefing,
    file_briefing, solve, decompose, harvest, delegate, marketplace, etc.

shell/                     (1,910 lines, 10 files)
  core.py                  — Rich REPL with PLUR safety principles
  brain.py                 — Claude SDK brain mode
  tui.py                   — Terminal UI rendering
  events.py                — Event watcher (knowledge inbox, ACP, marketplace)
  context.py               — Session context management
  display.py               — Output formatting

slash_commands/            (1,895 lines, 9 files)
  — 19 commands: /config, /tools, /heads, /wake, /sleep, /swap,
    /status, /knowledge, /session, /sessions, /mesh, /spawn, /ps,
    /output, /kill, /collab, /help, /solve, /events
```

### Marketplace & Networking

```
cloud_marketplace/         (2,196 lines, 10 files)
  — RFQ negotiation, escrow contracts, bid scoring, delivery verification

acp_bridge/                (1,070 lines, 6 files)
  — Agent Communication Protocol integration (file-based fallback when offline)

mesh/                      (1,749 lines, 10 files)
  — P2P LAN discovery (mDNS/zeroconf), presence, knowledge sync, failover

rfq/                       (902 lines, 5 files)
  — Request-for-Quotation management, bid scoring, quality verification
```

### Supporting Modules

```
conversation_harvester.py  (546 lines) — Extract from Claude Code .jsonl sessions
session_harvester/         (746 lines) — MEMORY.md + CLAUDE.md → claims
codebase_scanner/          (1,140 lines) — AST analysis, capability discovery
context_packs.py           (414 lines) — Build knowledge packs for agents
skill_catalog.py           (404 lines) — Tool/skill registry
service_manager/           (588 lines) — Background service lifecycle
benchmarking/              (1,542 lines) — Performance monitoring
init_wizard/               (1,094 lines) — Hardware detection + config generation
test_generation.py         (494 lines) — Auto test creation
web_tools.py               — Web fetch + search for local LLM
runtime_config.py          — Mutable runtime JSON config
```

---

## Config: `config/`

```
config/
  heads.yaml               — Head registry (15+ models with VRAM, adapter, kind)
  solvers.yaml             — 15 solvers with capabilities, priority, cost
  discovery_sources.yaml   — External discovery sources
  prompts/
    base_agent.md          — Base agent prompt
    solve_executor.md      — Solve execution prompt
    solve_proposer.md      — Solve proposal prompt
    solve_reviewer.md      — Solve review prompt
    propose.md             — Generic proposal prompt
  recipes/
    architectural-decision.yaml
    consensus-verify.yaml
    img2report.yaml
    marketplace-delegation.yaml
    solver-selection.yaml
    text-analyze.yaml
    test-phase1-coordinate-transform.yaml
  templates/
    apple_silicon.yaml     — Mac M-series config
    cpu_only.yaml          — No GPU config
    rtx3060.yaml           — NVIDIA RTX 3060 config
    rtx4090.yaml           — NVIDIA RTX 4090 config
```

---

## Tests: `tests/`

```
tests/
  conftest.py              — Shared fixtures, factories
  0-nano/     (68 files)   — Core unit tests (fast, no GPU, no network)
  1-micro/    (64 files)   — Feature tests (mocked adapters)
  2-small/    (13 files)   — Integration tests (e2e, benchmarking)
  3-medium/   (5 files)    — Scenario tests
  4-large/    (6 files)    — Robustness tests
  5-huge/     (7 files)    — Scale tests
  6-extreme/  (3 files)    — Edge case tests

Total: 166 files, 1,446 tests, 52,972 lines of test code
```

---

## Lines of Code Summary

| Module | Lines | Files | Category |
|--------|-------|-------|----------|
| cli | 4,192 | 15 | Interface |
| narrative | 4,157 | 18 | Knowledge |
| api | 4,110 | 27 | Interface |
| adapters | 3,559 | 19 | Core |
| night_shift | 2,780 | 7 | Knowledge |
| mcp_server | 2,362 | 10 | Interface |
| knowledge_store | 2,319 | 12 | Knowledge |
| cloud_marketplace | 2,196 | 10 | Marketplace |
| discovery | 2,161 | 8 | Core |
| shell | 1,910 | 10 | Interface |
| extractors | 1,907 | 10 | Knowledge |
| slash_commands | 1,895 | 9 | Interface |
| mesh | 1,749 | 10 | Networking |
| benchmarking | 1,542 | 6 | Core |
| orchestrator | 1,445 | 6 | Core |
| router | 1,260 | 7 | Core |
| codebase_scanner | 1,140 | 5 | Knowledge |
| registry | 1,138 | 9 | Core |
| solve | 1,103 | 6 | Coordination |
| init_wizard | 1,094 | 7 | Setup |
| acp_bridge | 1,070 | 6 | Networking |
| consensus | 1,059 | 5 | Coordination |
| recipe_learning | 1,005 | 7 | Research |
| autonomous_executor | 956 | 5 | Core |
| auto_decomposition | 947 | 6 | Research |
| shell_pipeline | 946 | 7 | Interface |
| rfq | 902 | 5 | Marketplace |
| session_harvester | 746 | 5 | Knowledge |
| tree_of_thoughts | 695 | 6 | Research |
| process_reward_models | 570 | 3 | Research |
| Top-level .py files | ~8,000 | 30+ | Mixed |
| **Total** | **~75,000** | **200+** | |
