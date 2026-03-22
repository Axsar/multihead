# MultiHead

[![CI](https://github.com/Axsar/multihead/actions/workflows/test.yml/badge.svg)](https://github.com/Axsar/multihead/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Make your coding agent team smarter with tools.**

MultiHead is an orchestration and knowledge layer that sits alongside your existing tools — Claude Code, Codex, local models, or custom agents — and gives them persistent memory, verified knowledge, and multi-agent coordination.

You don't switch workflows. You enhance them.

```
Before: agent runs → output → forgets everything
After:  agent runs → output verified → knowledge stored → next run starts smarter
```

![MultiHead Architecture](docs/multihead-architecture.png)

<br>

> *"I built a team of coding agents, and became a human copy & paste machine. Then I built MultiHead."* — axsar

---

## What This Actually Does

Your coding agent (Claude Code, Codex, whatever) works normally. MultiHead adds three things:

**1. Knowledge that persists across sessions.** Before you edit a file, MultiHead briefs you on what's known — constraints, warnings, contradictions, history. After you work, it extracts what you learned and stores it for next time.

**2. Verification across independent sources.** Night Shift runs in the background, cross-checking what was said in conversations against what the code actually does, what git history shows, and what CI reports. Agreements become corroborated facts. Disagreements get flagged.

**3. Multi-agent coordination.** Multiple agents working in different repos can share knowledge through a common store. No copy-paste, no Slack messages — deposit a claim, other agents query it.

---

## Works With Your Existing Setup

MultiHead is not a replacement for your tools. It's a layer underneath them.

**Integration points:**
- **Claude Code** — MCP server provides 25+ tools. Knowledge hooks auto-inject briefings before file edits.
- **Any CLI agent** — call `multihead briefing <file>` or `multihead kb "topic"` from any script or agent.
- **Python** — `from multihead.client import MultiHeadClient` for programmatic access.
- **Background** — Night Shift runs nightly, extracting and verifying knowledge while you sleep.

**You can also use MultiHead directly** via the interactive shell (`multihead shell`) or CLI commands — but that's optional. The primary value is what it adds to your existing workflow.

---

## The Difference

Same input to two systems:

**Input:** `Analyze the authentication system`

**Typical AI:**
```
The authentication system uses JWT tokens for user authentication.
Tokens appear to expire after a set duration and are validated
during requests. There is also a refresh mechanism to issue new
tokens. The system seems standard and follows common patterns.
```

**Your agent + MultiHead:**
```
CONSTRAINTS (corroborated — don't violate):
  • JWT tokens use RS256 signing with 24h expiry
    Source: auth/token_manager.py:48 · Confidence: high

  • Token validation middleware runs on every /api/ route
    Source: auth/middleware.py:12 · Confidence: high

WARNINGS (stale — verify before assuming):
  • Refresh token logic does not handle expiration edge cases
    Source: auth/refresh.py:72 · Confidence: medium

CONTRADICTION:
  • Token expiry set to 24h in config, but 12h in validation logic
    Sources: config/auth.yaml:12 ↔ auth/validator.py:33

Recommendation: align validation logic with configuration
```

Typical AI describes what it sees. With MultiHead, your agent extracts facts, verifies them across sources, and surfaces conflicts — and remembers all of it for next time.

---

## How It Works

```
You work (code, conversations, decisions)
        ↓
Night Shift extracts knowledge from all channels
        ↓
Fusion cross-checks: code vs git vs conversation vs CI
        ↓
Knowledge store tracks verified truth
        ↓
Briefings feed it back when you (or your agent) edit files
```

Each cycle makes the system more accurate. Your agent doesn't start from scratch — it starts from verified institutional knowledge.

---

## Try It

```bash
pip install -e .
multihead init --auto
```

Then use it from your existing agent, or try it directly:

```bash
# Query what the system knows
multihead kb "authentication"

# Get a briefing before editing a file
multihead briefing src/auth/jwt_handler.py

# Deposit knowledge
multihead deposit "JWT tokens expire in 24h" -k auth.jwt.expiry

# Run the full knowledge pipeline
multihead nightshift run --head openai-gpt41-nano --batch

# Or use the interactive shell
multihead shell
```

---

## Examples

### Agent-to-Agent Knowledge Sharing

Two agents on the same machine, different repos. They share knowledge through the store — no copy-paste.

**Agent A** (temperature estimator) defines the API contract:

```bash
multihead deposit \
  "Temperature estimator exposes GET /api/v1/estimate?sensor_id=X — returns JSON {celsius: float, confidence: float, timestamp: iso8601}. Updated every 5s. Returns 404 if sensor_id unknown." \
  -k temperature_estimator.api.contract \
  -p agent-a
```

**Agent B** (pressure cooker controller) queries before writing integration code:

```bash
multihead kb "temperature estimator"
```

Agent B now knows the contract and builds to it. Later, Agent A asks for a briefing:

```bash
multihead briefing temperature-estimator
```
```
CONSTRAINTS:
  • API contract: GET /api/v1/estimate?sensor_id=X → {celsius, confidence, timestamp}
  • Pressure cooker depends on this endpoint — polls every 10s
  • Safety-critical: confidence < 0.5 triggers shutoff downstream
```

Agent A now knows: **don't change the response schema** — another system depends on it.

---

### Autonomous Task Solving

```bash
multihead solve "Fix the timeout bug in the webhook retry handler"
```

```
>> Step 1.1  explore   Read webhook handler code           OK (2.1s)
>> Step 1.2  explore   Check git history for timeout bugs  OK (1.8s)
>> Step 2.1  edit      Fix retry timeout from 5s to 30s   OK (3.4s)
>> Step 3.1  test      Run webhook test suite              OK (8.2s)  — 12/12 passing
>> Step 3.2  verify    Confirm no regression               OK (1.5s)

Solve complete: 5 steps, 0 failures, 17.0s
Knowledge deposited: 3 new claims
```

The task is decomposed into a DAG, routed to the right head, executed with verification, and the results are stored. Next time anyone touches this code, the briefing includes what was learned.

---

### Resurrect Old Code

Point MultiHead at a forgotten repo. It analyzes the code, discovers capabilities, and you can publish them to the marketplace.

```bash
export MULTIHEAD_PROJECT_ROOTS="~/repos/old-sentiment-pipeline"
multihead nightshift run --head openai-gpt41-nano --batch
```

```
>> solver_discovery      OK (3.2s)  — 2 capabilities found:
     • text_classification (sentiment_model.py — 94.2% accuracy on SST-2)
     • text_preprocessing (clean_pipeline.py — handles HTML, Unicode, emoji)
```

```bash
multihead marketplace publish --capability "text_classification" --price 0.01
```

Dead code becomes a running service on [BotVibes](https://botvibes.io).

---

## Why Not Just Use One Model?

Single models guess, hallucinate, don't verify themselves, and forget everything between sessions.

MultiHead adds:
- **Multiple passes** — cross-check outputs across heads
- **Verification** — consensus voting, reflection loops, process reward models
- **Persistent memory** — claims survive across sessions with lifecycle tracking
- **Coordination** — route work to the right specialist, not one model doing everything

It's the difference between asking one person to do everything and running a coordinated team.

---

## Advanced Modes and Ecosystem

Everything below extends the core system. The knowledge layer, briefings, and verification work without any of this.

---

### Heads and Adapters

A **head** is a specialist — anything that can do work. An **adapter** is how it connects to its backend:

```yaml
- head_id: devstral
  adapter: ollama
  model: "devstral-small-2:24b"

- head_id: claude-sonnet
  adapter: claude
  model: "claude-sonnet-4-6"
  gpu_required: false
```

| Adapter | Backend | Notes |
|---------|---------|-------|
| `ollama` | Ollama server | Easiest setup. Wide model library. |
| `transformers` | HuggingFace | Direct GPU. 4-bit/8-bit quantization. |
| `vllm` | vLLM server | High throughput. Sleep/wake. |
| `openai` | OpenAI API | Batch API (50% cheaper). |
| `anthropic` | Anthropic API | Claude models, batch API. |
| `claude` | Claude CLI | Claude Code as a head (uses your subscription). |
| `claude_agent_sdk` | Claude Agent SDK | Claude with native tool use. |
| `acp` | Agent Communication Protocol | Multi-agent coordination. |

### Distributed Solve

Multiple agents propose plans, consensus picks the winner, work is assigned by domain expertise. All coordination happens through the knowledge store.

### Night Shift

26-stage pipeline. Harvests from conversations, code, git, CI. Cross-checks independent sources. Resolves contradictions or flags them for human review. Runs nightly.

### Ecosystem: BotVibes Marketplace

[BotVibes](https://botvibes.io) is infrastructure for the agent economy. Your agents sell capabilities they have and buy capabilities they don't. MultiHead handles execution — [BotVibes](https://botvibes.io) handles discovery, bidding, escrow, and trust.

#### OpenClaw Integration

*Under development.* [OpenClaw](https://github.com/clawctl/openclaw) agents can participate as heads in MultiHead.

#### AutoResearch

*Under development.* [AutoResearch](https://github.com/Axsar/autoresearch) enables local agents that deeply understand your codebase.

---

## Customize It

| Customize | Where |
|-----------|-------|
| Extraction logic | `extractors/claim_extractor.py` |
| Fusion rules | `claim_fusion.py` |
| Pipeline stages | `night_shift/` |
| Router behavior | `router/_scoring.py` |
| Models/heads | `config/heads.yaml` |

---

## What This Is NOT

- Not a chatbot or CLI you switch to
- Not instant AI magic
- Not pre-trained on your domain
- Not a managed service

You bring your agent, your code, your workflow. MultiHead adds the knowledge layer.

---

## License

MIT

---

## Links

- [Hello World](docs/hello-world.md) — 5-minute quickstart
- [Customization Guide](docs/customization-guide.md) — make it yours
- [Architecture](docs/repo-structure.md) — full codebase map
- [All Examples](docs/examples.md) — consensus, shell, night shift details, and more
- [BotVibes](https://botvibes.io) — marketplace for capability trading
