# MultiHead

**Build AI systems made of specialists — not one model trying to do everything.**

MultiHead is a local-first system for creating **persistent, domain-specific AI agents** that:
- own parts of your codebase
- accumulate knowledge over time
- improve through repeated use

![MultiHead Architecture](docs/multihead-architecture.png)

---

## What This Actually Is

Most AI systems are stateless:
- run → output → forget

MultiHead is stateful:
- run → verify → store → improve

Instead of treating every task as new, your system **builds experience**.

Over time, agents become more accurate, more consistent, and more specialized.

---

## Core Idea

You don't use one general-purpose AI.

You create **specialists**:

- Auth Agent → understands authentication logic
- Payments Agent → understands billing flows
- Infra Agent → understands deployment and systems

Each agent:
- lives next to your code
- remembers what it has seen
- builds domain expertise over time

MultiHead coordinates them into a single system.

---

## How It Works

1. Break a task into steps
2. Route each step to the right specialist
3. Execute using the appropriate model or tool
4. Verify outputs
5. Store knowledge for future tasks

> Most systems execute tasks.
> MultiHead builds **agents that get better at executing them**.

---

## Start Here

```bash
# Install
pip install -e .

# Detect your hardware, generate config
multihead init --auto

# Deposit a fact into the knowledge store
multihead deposit "JWT tokens expire in 24h" -k auth.jwt.expiry

# Query it back
multihead kb "jwt"
```

That's it — you have a persistent knowledge store. Now point it at your codebase:

```bash
# Analyze your code and conversations — extract verified knowledge
multihead nightshift run --head openai-gpt41-nano --batch
```

```
>> claim_extraction     OK (17.2s) — 34,165 claims
>> consistency_check    OK (5.8s)  — 426 contradictions found
>> claim_fusion         OK (10.4s) — 639 independently verified facts
>> staleness_sweep      OK (5.7s)  — 97 outdated claims marked stale
```

Now ask what the system knows about any file:

```bash
multihead briefing src/auth/jwt_handler.py
```

```
CONSTRAINTS (corroborated — don't violate):
  • JWT tokens use RS256 signing with 24h expiry
  • Token validation middleware runs on every /api/ route

WARNINGS (stale — verify before assuming):
  • Previous implementation used HS256 — changed in commit abc123

HISTORY (superseded — don't repeat):
  • Tried storing tokens in localStorage — XSS vulnerability, reverted
```

Your codebase now has institutional memory.

---

## Examples

### 1. Agent-to-Agent Communication

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

### 2. Autonomous Solve

Give MultiHead a task. It decomposes, routes to the right model, executes, and deposits what it learned.

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

Steps 1.1 and 1.2 ran in parallel (no dependency). The decomposer inferred the DAG automatically. Research features were auto-enabled: Tree-of-Thoughts for exploration, Process Reward Models for code quality, Reflection loops for test verification.

Next time any agent touches this code, the briefing includes what was learned.

---

### 3. Distributed Solve (Multi-Agent)

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

### 4. Resurrect Old Code

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

Your forgotten repo is now a running service on BotVibes. When someone needs sentiment analysis, your old code handles it — and you get paid.

---

## Why Not Existing Frameworks?

| Framework | What it does | What's missing |
|-----------|-------------|----------------|
| LangGraph | Orchestration + state | Agents don't persist or specialize |
| CrewAI | Team mental model | No real knowledge accumulation |
| AutoGen | Multi-agent reasoning | Ephemeral, no verification |
| **MultiHead** | **Persistent specialists + verified knowledge** | — |

---

## What This Is NOT

* Not a chatbot
* Not "install and get AI magic"
* Not pre-tuned for your domain — you customize it
* Not a managed service (see [BotVibes](https://botvibes.io) for that)

---

## The Knowledge Loop

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

---

## Key Features

**Knowledge Store** — Not chat logs. Verified, evolving facts with lifecycle: proposed → corroborated → stale → superseded.

**Night Shift** — 26-stage pipeline. Harvests from conversations, code, git, CI. Cross-checks independent sources. Resolves contradictions. Runs nightly.

**Heads** — Pluggable intelligence. 14 adapters: Ollama, Transformers, vLLM, OpenAI, Anthropic, Claude SDK, BotVibes. Local GPU or cloud API — same interface.

**Consensus** — Multiple models vote on the answer. Strategies: MAJORITY, WEIGHTED, UNANIMOUS, THRESHOLD, FIRST_TO_AHEAD.

**Decomposition** — Complex goals → parallel DAG of atomic steps. Auto-routes each step to the best head. Infers dependencies from file access patterns.

**Research Features** — Tree-of-Thoughts, Process Reward Models, Reflection loops. Auto-enabled based on step type.

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

## Customize It

MultiHead is meant to be adapted.

| Customize        | Where                           |
| ---------------- | ------------------------------- |
| Extraction logic | `extractors/claim_extractor.py` |
| Fusion rules     | `claim_fusion.py`               |
| Pipeline stages  | `night_shift/`                  |
| Router behavior  | `router/_scoring.py`            |
| Models/heads     | `config/heads.yaml`             |

Start here → **docs/customization-guide.md**

---

## Numbers

| | |
|---|---|
| Python | 75K lines |
| Tests | 1,446 |
| Model adapters | 14 |
| Pipeline stages | 26 |
| MCP tools | 25 |
| CLI commands | 15 |
| Shell commands | 19 |

---

## Who This Is For

* Engineers building AI workflows or agent systems
* Teams that need memory + verification across sessions
* Anyone frustrated with stateless LLM setups

Not for:
* Casual chat use
* One-off prompts
* "Install and get AI magic" expectations

---

## License

MIT

---

## Links

* [Hello World](docs/hello-world.md) — 5-minute quickstart
* [Customization Guide](docs/customization-guide.md) — make it yours
* [Architecture](docs/repo-structure.md) — full codebase map
* [All Examples](docs/examples.md) — consensus, chat/shell, night shift details, and more
* [BotVibes](https://botvibes.io) — marketplace for capability trading

---

# Final note

MultiHead is not about generating better answers.

It's about building systems that:

* **know what they know**
* **know when they're wrong**
* **get better over time**
