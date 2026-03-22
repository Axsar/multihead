# MultiHead

[![CI](https://github.com/Axsar/multihead/actions/workflows/test.yml/badge.svg)](https://github.com/Axsar/multihead/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Build AI systems made of specialists that learn.**

Most AI tools forget everything between runs. MultiHead doesn't.

MultiHead organizes work into **persistent, domain-specific agents** that:
- own parts of your codebase
- verify what they learn
- store reusable knowledge
- improve through repeated use

![MultiHead Architecture](docs/multihead-architecture.png)

**Think of MultiHead as a team of evolving specialists coordinated into one system.**

---

## What This Actually Is

Most AI systems are stateless:

```
run → output → forget
```

MultiHead is stateful:

```
run → verify → store → improve
```

Instead of treating every task as brand new, MultiHead builds experience. Over time, specialists become more accurate, more consistent, and more useful in their own domains.

---

## Core Idea

You don't rely on one general-purpose model to do everything.

You create **specialists**:

- **Auth Agent** → understands authentication logic
- **Payments Agent** → understands billing flows
- **Infra Agent** → understands deployment and systems

Each specialist:
- lives next to the code it works on
- remembers what it has seen
- accumulates verified context
- improves over time

MultiHead coordinates them into a single working system.

---

## What Lives Inside a Specialist

A specialist is not just a prompt. It is a small working unit with memory.

```
agents/
  auth-agent/
    prompts.md
    memory.json
    constraints.json
    past_fixes/
    briefings/
```

Over time, each specialist builds **institutional knowledge** for its domain:
- known constraints
- prior fixes
- contradictions worth checking
- context that should not be rediscovered from scratch

---

## The Difference

Same input — different result.

**Input:** `Analyze the authentication system`

**Typical AI:**
```
The authentication system uses JWT tokens for user authentication.
Tokens appear to expire after a set duration and are validated
during requests. There is also a refresh mechanism to issue new
tokens. The system seems standard and follows common patterns.
```

**MultiHead:**
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

Typical AI describes what it sees. MultiHead extracts facts, verifies them across sources, and surfaces conflicts.

---

## A Real Solve Run

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

What matters is not just that the task was solved. What matters is that the result becomes part of the system's **reusable knowledge**. Next time any agent touches this code, the briefing includes what was learned.

---

## Start Here

```bash
pip install -e .
multihead init --auto
multihead shell
```

```
MultiHead Shell v1.2
System ready.

Knowledge Store: 42 claims
  Constraints: 8 corroborated
  Warnings: 3 stale
  Contested: 1 contradiction
  Domains: auth, payments, infra

Try:
  "what does the auth system do?"
  "show known constraints"
  "what contradictions exist?"

[qwen-llm] you>
```

---

## What You Can Do With It

- Turn a codebase into a **self-improving system**
- Build **domain-specific specialists** for parts of your stack
- Run **multi-step solve workflows** with verification
- Extract **corroborated constraints, warnings, and contradictions**
- Resurrect old repos by turning forgotten code into reusable capability

---

## Advanced Modes and Ecosystem

Everything below extends the core system. If you only want persistent specialists, verification, and reusable knowledge — you can stop here and start using MultiHead now.

---

### Agent-to-Agent Communication

Two agents on the same machine, different repos. They communicate through the knowledge store — no copy-paste.

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
```
temperature_estimator.api.contract (confidence: 0.90)
  GET /api/v1/estimate?sensor_id=X → {celsius, confidence, timestamp}
```

Agent B writes the integration, then records what it built:

```bash
multihead deposit \
  "Pressure cooker controller polls temperature_estimator every 10s at sensor_id=boiler-1. Triggers safety shutoff if celsius > 180 or confidence < 0.5." \
  -k pressure_cooker.integration.temperature \
  -p agent-b
```

Later, Agent A asks for a briefing before making changes:

```bash
multihead briefing temperature-estimator
```
```
CONSTRAINTS:
  • API contract: GET /api/v1/estimate?sensor_id=X → {celsius, confidence, timestamp}
  • Pressure cooker depends on this endpoint — polls every 10s
  • Safety-critical: confidence < 0.5 triggers shutoff downstream
```

Agent A now knows: **don't change the response schema** — another system depends on it, and it's safety-critical.

---

### Resurrect Old Code

You have a repo from two years ago — a sentiment analysis pipeline. It works, but nobody's using it.

```bash
# Point MultiHead at the old repo
export MULTIHEAD_PROJECT_ROOTS="~/repos/old-sentiment-pipeline"

# Night Shift analyzes everything
multihead nightshift run --head openai-gpt41-nano --batch
```

```
>> behavioral_analysis   OK (8.4s)  — 1 repo scanned
>> claim_extraction      OK (6.1s)  — 247 claims
>> solver_discovery      OK (3.2s)  — 2 capabilities found:
     • text_classification (sentiment_model.py — 94.2% accuracy on SST-2)
     • text_preprocessing (clean_pipeline.py — handles HTML, Unicode, emoji)
```

MultiHead read the code, found trained models and working pipelines, and registered them as capabilities. Publish to the marketplace:

```bash
multihead marketplace publish --capability "text_classification" --price 0.01
```

Your forgotten repo is now a running service on [BotVibes](https://botvibes.io). When someone needs sentiment analysis, your old code handles it — and you get paid.

---

### Distributed Solve (Multi-Agent)

A task that spans multiple repos. The coordinator posts it, specialist agents propose plans, consensus picks the winner, and work is assigned to whoever owns that domain.

```bash
multihead solve "Migrate user auth from session cookies to JWT" --strategy weighted
```

```
Discovering agents... found 3 active sessions:
  • auth-agent (~/repos/auth-service/)
  • api-agent (~/repos/api-gateway/)
  • frontend-agent (~/repos/web-client/)

Posting task to knowledge store... CLM_A1B2C3
Waiting for proposals...

Proposal from auth-agent (12s):
  "1. Add JWT signing endpoint  2. Migrate session store  3. Add token refresh"

Proposal from api-agent (18s):
  "1. Update middleware to validate JWT  2. Remove cookie parsing  3. Add token forwarding"

Proposal from frontend-agent (22s):
  "1. Replace cookie auth with Bearer header  2. Add refresh logic  3. Update login flow"

Running consensus (weighted by confidence)...
  auth-agent:      0.92 confidence  ← winner (owns the auth domain)
  api-agent:       0.88 confidence
  frontend-agent:  0.85 confidence

Assigning work:
  auth-agent     → JWT signing, session migration, token refresh
  api-agent      → middleware update, header forwarding
  frontend-agent → Bearer header, login flow, refresh UI
```

Each agent executes in their own repo. Results flow back through the knowledge store. If `api-agent` needs to know the JWT signing algorithm, it queries the knowledge base — `auth-agent` already deposited it.

---

### The Knowledge Loop

```
You work (code, conversations, decisions)
        ↓
Night Shift extracts knowledge
        ↓
Fusion verifies across independent sources
        ↓
Knowledge store tracks verified truth
        ↓
Briefing feeds it back when you edit files
```

This is how specialists compound value instead of starting over every run.

---

### Why Not Existing Frameworks?

| Framework | Good at | Missing |
|-----------|---------|---------|
| LangGraph | Orchestration + state | Persistent specialist knowledge |
| CrewAI | Team mental model | Verified knowledge accumulation |
| AutoGen | Multi-agent interaction | Durable memory + verification |
| **MultiHead** | **Persistent specialists + verified knowledge** | — |

---

### System Components

**Heads and Adapters** — A head is a specialist. An adapter is how it connects to its backend. Three lines of YAML per head, any backend:

```yaml
# Local model via Ollama
- head_id: devstral
  adapter: ollama
  model: "devstral-small-2:24b"
  kind: llm

# Cloud API
- head_id: gpt4o
  adapter: openai
  model: "gpt-4o"
  kind: llm
  gpu_required: false

# Claude as a native head
- head_id: claude-sonnet
  adapter: claude
  model: "claude-sonnet-4-6"
  kind: llm
  gpu_required: false
```

| Adapter | Backend | GPU | Notes |
|---------|---------|-----|-------|
| `ollama` | Ollama server | Optional | Easiest setup. Wide model library. |
| `transformers` | HuggingFace | Yes | Direct GPU control. 4-bit/8-bit quantization. |
| `vllm` | vLLM server | Yes | High throughput. Sleep/wake for fast swapping. |
| `openai` | OpenAI API | No | GPT-4o, GPT-4.1, batch API (50% cheaper). |
| `anthropic` | Anthropic API | No | Claude models, batch API. |
| `claude` | Claude CLI | No | Claude Code as a head. |
| `claude_agent_sdk` | Claude Agent SDK | No | Claude with native tool use and session resume. |
| `mock` | In-memory | No | Testing. No real inference. |
| `acp` | Agent Communication Protocol | No | Multi-agent coordination, task queuing, trust scoring. |

**Knowledge Store** — Verified, evolving facts with lifecycle: proposed → corroborated → stale → superseded. Every claim tracks its source, confidence, and evidence.

**Night Shift** — 26-stage pipeline that runs nightly. Harvests from conversations, code, git, and CI. Cross-checks independent sources. Resolves contradictions or flags them for review.

**Consensus** — Multiple heads vote on the same question. Used automatically during task decomposition — multiple heads propose plans, consensus picks the best one.

**Decomposition** — Complex goals become parallel DAGs of atomic steps. The decomposer infers dependencies from file access patterns and runs independent steps concurrently.

---

### Ecosystem: BotVibes Marketplace

[BotVibes](https://botvibes.io) is infrastructure for the agent economy. Agents don't call APIs — they trade results. MultiHead plugs directly into this marketplace.

**Sell capabilities you have:**

```bash
multihead marketplace publish \
  --capability "object_detection" \
  --model yolo-v8-custom \
  --price 0.02
```

**Buy capabilities you don't:**

```bash
multihead marketplace procure \
  --capability "text_translation" \
  --payload "Translate API docs to Japanese" \
  --max-price 0.50
```

[BotVibes](https://botvibes.io) handles discovery, bidding, escrow, and trust scoring. Your agents earn money doing what they're good at — and buy capabilities they don't have from agents who do.

#### OpenClaw Integration

*Under development.*

[OpenClaw](https://github.com/clawctl/openclaw) agents can participate as heads in MultiHead. A claw is a claw, a head is a head — together they form a multi-agent system where each specialist contributes what it's best at.

#### AutoResearch — Codebase-Aware Local Agents

*Under development.*

[AutoResearch](https://github.com/Axsar/autoresearch) enables local agents that deeply understand your codebase. Combined with MultiHead's knowledge store, this creates local agents that know your code AND remember what they've learned across sessions.

---

## Customize It

| Customize | Where |
|-----------|-------|
| Extraction logic | `extractors/claim_extractor.py` |
| Fusion rules | `claim_fusion.py` |
| Pipeline stages | `night_shift/` |
| Router behavior | `router/_scoring.py` |
| Models/heads | `config/heads.yaml` |

Start here → **docs/customization-guide.md**

---

## What This Is NOT

- Not a chatbot
- Not instant AI magic
- Not pre-trained on your domain by default
- Not a managed service

You bring the code, context, and workflow. MultiHead helps specialists learn that world over time.

---

## Architecture

<details>
<summary>Text version (for screen readers and bots)</summary>

```
Task Input
    ↓
MultiHead OS (Routing · Orchestration · Knowledge)
    ↓
┌───────────┬───────────────┬──────────────┐
│ Auth Agent│ Payments Agent│ Infra Agent  │
│ /auth     │ /payments     │ /infra       │
│ JWT,tokens│ billing,hooks │ deploy, CI/CD│
└─────┬─────┴───────┬───────┴──────┬───────┘
      │             │              │
      └─── Claims & Context ───────┘
                    ↓
      ┌─────────────┴─────────────┐
      │ Knowledge    │ Consensus  │
      │ Store        │ Engine     │
      │ (verified    │ (multi-    │
      │  claims +    │  agent     │
      │  evidence)   │  verify)   │
      └──────────────┴────────────┘
```
</details>

---

## License

MIT

---

## Links

- [Hello World](docs/hello-world.md) — 5-minute quickstart
- [Customization Guide](docs/customization-guide.md) — make it yours
- [Architecture](docs/repo-structure.md) — full codebase map
- [All Examples](docs/examples.md) — consensus, chat/shell, night shift details, and more
- [BotVibes](https://botvibes.io) — marketplace for capability trading
