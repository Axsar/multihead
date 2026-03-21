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

You don’t use one general-purpose AI.

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

## What This Is NOT

* Not a chatbot
* Not “install and get AI magic”
* Not pre-tuned for your domain — you customize it
* Not a managed service (see [BotVibes](https://botvibes.io) for that)

---

## Why Not Existing Frameworks?

| Framework | What it does | What’s missing |
|-----------|-------------|----------------|
| LangGraph | Orchestration + state | Agents don’t persist or specialize |
| CrewAI | Team mental model | No real knowledge accumulation |
| AutoGen | Multi-agent reasoning | Ephemeral, no verification |
| **MultiHead** | **Persistent specialists + verified knowledge** | — |

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

The result: a system that **accumulates understanding**, not just responses.

---

## Quick Start (5 minutes)

```bash
# Install
pip install -e .

# Initialize
multihead init --auto

# Add a piece of knowledge
multihead deposit "JWT tokens expire in 24h" -k auth.jwt.expiry

# Query it
multihead kb "jwt"
```

That’s it—you now have a persistent knowledge store.

---

## Next Step: Run the Pipeline

Turn your conversations + codebase into structured knowledge:

```bash
multihead nightshift run --head openai-gpt41-nano --batch
```

This will:

* Extract facts from conversations
* Analyze your code
* Cross-reference sources
* Mark contradictions and stale knowledge

Now try:

```bash
multihead briefing src/auth/jwt_handler.py
```

You’ll see what the system *knows* about that file.

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

## Key Features

### 🧠 Knowledge Store (Institutional Memory)

Not chat logs—**verified, evolving facts**:

```bash
multihead briefing src/file.py
```

* Constraints (corroborated)
* Warnings (stale)
* Signals (contested)
* History (superseded)

Example output:
```
=== Knowledge: src/auth/jwt_handler.py ===
CONSTRAINTS (corroborated — don't violate):
  • JWT tokens use RS256 signing with 24h expiry
  • Token validation middleware runs on every /api/ route

WARNINGS (stale — verify before assuming):
  • Previous implementation used HS256 — changed in commit abc123

HISTORY (superseded — don't repeat):
  • Tried storing tokens in localStorage — XSS vulnerability, reverted
```

---

### 🔁 Night Shift (Automated Pipeline)

26-stage pipeline that:

* Extracts knowledge from conversations, code, git
* Cross-checks independent sources
* Resolves contradictions
* Tracks lifecycle over time

---

### 🤖 Heads (Pluggable Intelligence)

A head is a head — local GPU, cloud API, or marketplace peer. You focus on what needs to be done — the system handles where intelligence comes from.

```yaml
my-model:
  adapter: ollama  # or: transformers, openai, anthropic, vllm, mock
  model: "llama3:8b"
```

14 adapters: Local (Ollama, Transformers, vLLM) · Cloud (OpenAI, Anthropic, Claude SDK) · Marketplace (BotVibes)

---

### ⚖️ Consensus (Multi-Agent Verification)

When one model isn’t enough:

* MAJORITY
* WEIGHTED
* UNANIMOUS
* THRESHOLD
* (and more)

---

### 🔬 Research Features

* Tree-of-Thoughts
* Process Reward Models
* Reflection loops
* Auto-decomposition (task → DAG)

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

## What You Actually Build

MultiHead gives you the engine.

You build:

* Your own knowledge pipeline
* Your own domain extraction logic
* Your own verification rules
* Your own workflows

What we built with it is just one example.

---

## What a Real Run Looks Like

```
>> entity_extraction    OK (3.3s)  — 15,791 entities
>> claim_extraction     OK (17.2s) — 34,165 claims
>> consistency_check    OK (5.8s)  — 426 contradictions found
>> conflict_resolution  OK (26.1s) — 347 auto-resolved
>> claim_fusion         OK (10.4s) — 639 independently verified facts
>> staleness_sweep      OK (5.7s)  — 97 outdated claims marked stale
```

---

## Two Modes

### Development Mode

* Build and refine pipelines
* Use consensus, reflection, experimentation
* Optimize for quality

### Execution Mode

* Run optimized pipelines
* Single best head
* Batch processing
* Optimize for cost and speed

---

## Demo

The real value shows over time — days and weeks of accumulated knowledge, not a one-time trick.

👉 [See the system in action](https://multihead.dev/demo)

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

MIT for initial release. May move to BSL at a later milestone.

---

## Links

* [Hello World](docs/hello-world.md) — 5-minute quickstart
* [Customization Guide](docs/customization-guide.md) — make it yours
* [Architecture](docs/repo-structure.md) — full codebase map
* [BotVibes](https://botvibes.io) — marketplace for capability trading

---

# Final note

MultiHead is not about generating better answers.

It’s about building systems that:

* **know what they know**
* **know when they’re wrong**
* **get better over time**
