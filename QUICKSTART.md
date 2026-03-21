# MultiHead Quick Start

Get up and running in 5 minutes.

## 1. Install (2 minutes)

```bash
git clone https://github.com/Axsar/multihead.git
cd multihead
bash scripts/install.sh
```

The installer auto-detects your hardware and sets everything up.

## 2. Activate & Start (30 seconds)

```bash
source .venv/bin/activate
multihead serve
```

Leave this running in one terminal. Open a new terminal for the next steps.

## 3. Try It Out (2 minutes)

### Chat with the agent terminal

```bash
source .venv/bin/activate
multihead shell
```

Type your message and press Enter. On a fresh install this uses mock heads (instant replies for testing).
Run `multihead init --auto` to detect your hardware and enable real models (local GPU, Claude CLI, etc.).

### Or use it from Python

```python
from multihead.client import MultiHeadClient

mh = MultiHeadClient()
result = mh.chat("What is the capital of France?")
print(result["response"])
```

### Or use it from Claude Code

```bash
# In the MultiHead repo directory
claude

# Then use MCP tools:
# multihead_chat(message="Hello")
# multihead_heads()
# multihead_knowledge(query_type="claims")
```

## 4. Write to the Knowledge Store (1 minute)

Any process can deposit facts:

```python
from multihead.client import MultiHeadClient

mh = MultiHeadClient()

# Write a fact
mh.deposit_claim(
    claim_key="myapp.status",
    statement="Build passed at 2026-02-21 14:30",
    produced_by="ci_pipeline",
)

# Read it back
claims = mh.query_claims(claim_key="myapp.status")
print(claims[0]["statement"])
```

Or via curl:

```bash
curl -X POST http://localhost:7337/knowledge/claims \
  -H "Content-Type: application/json" \
  -d '{
    "claim_key": "myapp.status",
    "statement": "Build passed",
    "produced_by": "ci_pipeline"
  }'
```

## What's Running?

- **API server**: http://localhost:7337 (REST API)
- **Brain**: Mock heads by default, or local LLM / Claude after `multihead init --auto`
- **Knowledge store**: `~/.multihead/knowledge.db`

## 5. Join Multi-Session Collaboration (1 minute)

Have your Claude Code session participate in autonomous task solving alongside other agents.

### Start the auto-responder (plan-only — default)

```bash
source .venv/bin/activate
python scripts/auto_responder_poller.py \
  --session-id my-agent-name \
  --project-id multihead \
  --capabilities solve,decompose
```

That's it. Your session will now:
- Monitor `knowledge.db` for decomposition requests (every 30s)
- Auto-decompose tasks using the local LLM
- Submit proposals without human intervention
- Participate in consensus voting with other agents

### Autonomous execution mode

Want agents to actually **do the work**, not just plan? Add `--strategy execute`:

```bash
python scripts/auto_responder_poller.py \
  --session-id my-agent-name \
  --project-id multihead \
  --strategy execute \
  --max-budget 3.0 \
  --claude-model claude-sonnet-4-6
```

This spawns `claude -p` subprocesses per step with:
- **Role-specific tools** — explorers can only read, implementers can edit, reviewers can't write
- **Parallel DAG execution** — independent steps run concurrently
- **Quality-gated retries** — steps below threshold get reflection feedback and retry (up to 3x)
- **Context chaining** — each step's output feeds into dependent steps

The plan is always posted to knowledge.db first (so other agents see it), then execution runs.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--session-id` | required | Unique name for this session |
| `--project-id` | `multihead` | Project scope in knowledge.db |
| `--capabilities` | `solve,decompose` | What this agent can do |
| `--check-interval` | `30` | Check every N seconds |
| `--strategy` | `plan-only` | `plan-only` or `execute` |
| `--max-budget` | `1.0` | Max USD per step (execute mode only) |
| `--claude-model` | `claude-sonnet-4-6` | Model for execution (execute mode only) |

### How it works

```
You post a task ──→ knowledge.db ──→ All pollers detect it
                                      ├─ agent-1 decomposes & submits proposal
                                      ├─ agent-2 decomposes & submits proposal
                                      └─ agent-3 decomposes & submits proposal
                                           ↓
                                     Consensus voting picks the best plan
                                           ↓
                          (if --strategy execute) Autonomous execution
                            Layer 0: [explore, read]        ← parallel
                            Layer 1: [implement]            ← with context from L0
                            Layer 2: [review, test]         ← parallel
                            Layer 3: [verify]               ← with review+test results
                                           ↓
                                  Results posted to knowledge.db
```

Multiple Claude Code sessions, BotVibes agents, and local LLMs can all collaborate through the shared knowledge database — no manual wake-ups needed.

## 6. Multi-Session Mesh (Friends Mode)

Two or more developers can collaborate through a shared `knowledge.db` on a network drive,
NFS mount, or any path both machines can reach.

### Step 1: One person sets up the shared directory

On the first machine:

```bash
source .venv/bin/activate
multihead init --mesh
```

You'll be prompted:

```
Shared data directory? [~/.multihead]: /mnt/shared/MultiHead
✓ Mesh configured!
  Shared directory: /mnt/shared/MultiHead
  Config saved to: ~/.multihead/config.yaml

Share this path with collaborators.
They should run: multihead init --mesh
```

### Step 2: Collaborators join

On each additional machine, run the same command and enter the shared path:

```bash
multihead init --mesh
# Enter: /mnt/shared/MultiHead
```

The path is saved to `~/.multihead/config.yaml` as `MULTIHEAD_DATA_DIR`. All sessions
that point here share the same `knowledge.db`.

> **Windows/WSL tip**: Use the WSL mount of a network share, e.g. `/mnt/z/MultiHead`
> (mapped from `\\server\MultiHead`).

### Step 3: Start serving on each machine

```bash
multihead serve
```

Each session automatically emits a heartbeat presence claim to the shared `knowledge.db`
every 30 seconds, so every node knows who is online.

### Step 4: Verify peers

```bash
curl "http://localhost:7337/knowledge/claims?claim_key=mesh.presence"
```

Each online session shows up as a claim, e.g.:

```
Node 'alice-pc' on ALICE is online at port 7337.
Node 'bob-laptop' on BOB is online at port 7337.
```

A node is considered stale when its `last_seen` timestamp is older than 90 seconds.

### Step 5: Collaborate

Start the auto-responder on each machine (see §5 above), then post a task — all sessions
compete to solve it:

```bash
python scripts/auto_responder_poller.py \
  --session-id alice \
  --project-id myproject \
  --strategy execute
```

```
You post a task → shared knowledge.db → alice decomposes
                                       → bob decomposes
                                              ↓
                                   Consensus picks best plan
                                              ↓
                              Winning agent executes autonomously
```

## Next Steps

- **Explore MCP tools**: 18 tools for Claude Code integration
- **Create a pipeline**: See `config/recipes/` for examples
- **Add your data**: Use `multihead narrative ingest` to extract claims from git/docs
- **Run diagnostics**: `multihead doctor` to check your setup

## Need Help?

- **Documentation**: See [README.md](README.md) and [docs/12-getting-started.md](docs/12-getting-started.md)
- **Troubleshooting**: Check the Troubleshooting section in README.md
- **Issues**: https://github.com/Axsar/multihead/issues

---

**That's it!** You now have a local AI running with institutional memory. 🚀
