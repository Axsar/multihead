# MultiHead

**MultiHead turns AI into a team of specialists that learn.**

Instead of one general model trying to do everything, MultiHead organizes work into persistent agents with domain expertise — code, auth, payments, infrastructure, vision, OCR, reasoning, and more. Each specialist improves over time by storing results, validating outputs, and reusing knowledge across tasks.

![MultiHead Architecture](docs/multihead-architecture.png)

---

## Core Idea

Most AI workflows look like this:

```
prompt → model → output → discard
```

MultiHead replaces that with:

```
task → specialists → cross-check → store → improve
```

Each run doesn't just produce an answer — it **builds capability**.

---

## What Makes MultiHead Different

**1. Persistent Specialists** — Agents live in folders, accumulate context, and develop domain expertise. Not stateless calls.

**2. Multi-Agent Execution** — Tasks are decomposed and routed across multiple specialists that collaborate and verify each other.

**3. Built-in Verification** — Outputs are cross-checked using consensus, reflection, or additional agents before being accepted.

**4. Memory That Compounds** — Results, decisions, and insights are stored and reused. Future runs get better.

**5. Local-First, Model-Agnostic** — Run on your own hardware (GPU/CPU), use local or API models, and swap them freely.

---

## What You Can Do With It

- Turn a codebase into a self-improving system
- Run multi-step reasoning workflows with verification
- Build domain-specific AI teams (e.g. auth + payments + infra)
- Resurrect and maintain legacy or complex systems
- Coordinate agents locally or across a marketplace ([BotVibes](https://botvibes.io))

---

## Quick Example

```bash
multihead solve "analyze this repo and fix failing tests"
```

What happens:
1. Task is decomposed into steps
2. Specialists (planner, coder, tester) are assigned
3. Outputs are cross-checked
4. Results are stored for reuse
5. System improves for next run

---

## The Difference

Same input — different output.

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

Typical AI describes what it sees. MultiHead extracts facts, verifies them across sources, and finds issues.

---

## Mental Model

Think:
- Not "one smart model"
- But **a team of evolving specialists**
- Coordinated, verified, and improving over time

---

## Start Here

```bash
pip install -e .
multihead init --auto
multihead shell
```

You'll enter an interactive system where agents collaborate and build knowledge over time.

```
MultiHead Shell v1.2
System ready.

Knowledge Store: 42 claims
  Constraints: 8 corroborated
  Warnings: 3 stale
  Contested: 1 contradiction
  Domains: auth, payments, infra

Try:
  Explore:  "what does the auth system do?"  ·  "explain payments flow"
  Inspect:  "show known constraints"  ·  "what contradictions exist?"
  Verify:   "verify: tokens expire in 24h"  ·  "what might be stale?"

[qwen-llm] you>
```

To populate the knowledge store, run the Night Shift pipeline:

```bash
multihead nightshift run --head openai-gpt41-nano --batch
```

This extracts knowledge from your code, conversations, and git history — then cross-checks independent sources to separate verified facts from stale assumptions.

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

Your forgotten repo is now a running service on [BotVibes](https://botvibes.io). When someone needs sentiment analysis, your old code handles it — and you get paid.

---

### 5. BotVibes Marketplace — Agents Buy and Sell Results

[BotVibes](https://botvibes.io) is infrastructure for the agent economy. Agents don't call APIs — they trade results. Summaries, datasets, transformations, code, labeled data. MultiHead plugs directly into this marketplace.

**Your agent has a capability. Someone else needs it.**

```bash
# Register your agent and list capabilities
multihead discover

# Publish to the marketplace
multihead marketplace publish \
  --capability "object_detection" \
  --model yolo-v8-custom \
  --price 0.02                    # $0.02 per inference
```

**The marketplace lifecycle:**

```
1. Register  — Your agent onboards with capabilities and pricing
2. Discover  — Buyers browse by capability, price, and trust score
3. Trade     — Buyer posts RFQ, your agent bids, contract accepted
4. Deliver   — MultiHead executes the work, result transferred via Vault
5. Review    — Buyer accepts → escrowed credits released to you
6. Reputation — Trust scores updated, better ranking for future work
```

**From the other side — you need a capability you don't have:**

```bash
# Procure work from the marketplace
multihead marketplace procure \
  --capability "text_translation" \
  --payload "Translate API docs to Japanese" \
  --max-price 0.50
```

```
Submitting RFQ... 3 providers quoted
  provider-a: $0.12 (trust: 94%)
  provider-b: $0.08 (trust: 87%)
  provider-c: $0.15 (trust: 98%)

Accepted: provider-c ($0.15, highest trust)
Result delivered. Credits deducted.
```

MultiHead handles execution on your side. [BotVibes](https://botvibes.io) handles discovery, bidding, escrow, and trust scoring. Your agents earn money doing what they're good at — and buy capabilities they don't have from agents who do.

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

### Heads and Adapters

A **head** is a specialist — anything that can do work. A local model, a cloud API, another team's agent, a trained pipeline. You don't care where intelligence comes from. You define what a head can do, and MultiHead routes work to it.

An **adapter** is how a head connects to its backend. Same interface, different plumbing:

```yaml
# Local model via Ollama
- head_id: devstral
  adapter: ollama
  model: "devstral-small-2:24b"
  kind: llm

# Local model via HuggingFace Transformers (4-bit quantized)
- head_id: qwen-llm
  adapter: transformers
  model: "Qwen/Qwen3-8B"
  kind: llm
  quantization: "4bit"

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

Three lines per head. The router scores each head on availability, health, VRAM fit, capability match, and error history — then picks the best one for each step.

**Available adapters:**

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
| `botvibes` | [BotVibes](https://botvibes.io) marketplace | No | Delegate to external providers. Pay per task. |
| `acp` | Agent Communication Protocol | No | Multi-agent coordination, task queuing, trust scoring. |

#### OpenClaw Integration

*Under development.*

[OpenClaw](https://github.com/clawctl/openclaw) agents can participate as heads in MultiHead. A claw is a claw, a head is a head — together they form a multi-agent system where each specialist contributes what it's best at. OpenClaw agents typically run on Ollama backends, and MultiHead coordinates them alongside local and cloud models through the shared knowledge store.

#### AutoResearch — Codebase-Aware Local Agents

*Under development.*

[AutoResearch](https://github.com/Axsar/autoresearch) enables local agents that deeply understand your codebase. Instead of generic LLM responses, an AutoResearch-powered head can navigate your code, understand architecture, and answer questions grounded in actual source files. Combined with MultiHead's knowledge store, this creates local agents that know your code AND remember what they've learned across sessions.

---

### Knowledge Store

Not chat logs — **verified, evolving facts** with lifecycle: proposed → corroborated → stale → superseded. Every claim tracks its source (conversation, code, git, CI), confidence, and evidence. Agents query it before acting and deposit what they learn after.

### Night Shift

26-stage pipeline that runs nightly. Harvests from conversations, code, git, and CI. Cross-checks independent sources via multi-channel fusion. Resolves contradictions automatically or flags them for human review.

### Consensus

Multiple heads vote on the same question. Strategies: majority, weighted, unanimous, threshold. Used automatically during task decomposition — multiple heads propose plans, consensus picks the best one.

### Decomposition

Complex goals become parallel DAGs of atomic steps. The decomposer infers dependencies from file access patterns (write-after-read, test-after-edit) and runs independent steps concurrently. Research features auto-enable: Tree-of-Thoughts for exploration, Process Reward Models for code quality, Reflection loops for verification.

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
