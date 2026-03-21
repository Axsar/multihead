# MultiHead

**Build AI systems that remember what's true, verify what's not, and improve over time.**

MultiHead is a **local-first framework for building AI task systems** with institutional memory, multi-agent consensus, and pluggable intelligence.

Think of it as an **operating system for a team of AI specialists**.

It’s not a chatbot. It’s not a hosted service.
It’s infrastructure for building systems that *get smarter the more you use them*.

![MultiHead Architecture](docs/multihead-architecture.png)

---

## What This Is (and Isn’t)

### ✅ What MultiHead is

* A **builder framework** (like Django for AI pipelines)
* A way to combine **models, tools, and workflows into systems**
* A system that builds **verified knowledge over time**
* A platform for **multi-step reasoning + execution**

### ❌ What it’s not

* Not a chatbot
* Not “install and get AI magic”
* Not pre-tuned for your domain
* Not a managed service (see BotVibes)

---

## Why MultiHead

Most AI setups today:

* Stateless (no real memory)
* Single-model (no specialization)
* No verification (trust whatever the model says)

MultiHead gives you:

* **Institutional memory** → not chat history, but *verified facts*
* **Multi-agent consensus** → cross-check critical decisions
* **Task decomposition** → break complex work into steps
* **Continuous learning loop** → improve over days/weeks

---

## The Core Loop

```
You work (code, conversations, decisions)
        ↓
Night Shift extracts knowledge
        ↓
Fusion verifies across sources
        ↓
Knowledge store tracks truth over time
        ↓
Briefing feeds it back into your workflow
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

## How This Is Actually Used

One effective way to use MultiHead: treat AI agents like a development team.

Instead of a single general-purpose assistant, create **persistent, domain-specific agents**:

* One agent lives in `auth/` and understands authentication
* One lives in `payments/` and owns billing logic
* Another specializes in infrastructure or deployment
* Another handles knowledge extraction

Each agent:
* Has local context (files, history, decisions)
* Becomes specialized over time
* Acts as the "owner" of that part of the system

MultiHead connects them:
* **Routes** tasks to the most relevant agent
* **Shares** verified knowledge between domains
* **Uses consensus** when decisions are uncertain
* **Tracks** what each part of the system actually knows

Instead of one assistant trying to do everything, you get a **system of specialists that collaborate and improve over time.**

```
           [ Task ]
              ↓
         [ Router ]
              ↓
      ┌───────┼────────┐
      ↓       ↓        ↓
  [Auth]  [Payments] [Infra]
   Agent    Agent     Agent
      \       |       /
       \      |      /
        → [Knowledge] ←
              ↓
        [Consensus]
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
