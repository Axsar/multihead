# Hello World: Codebase → Knowledge Graph in 10 Minutes

Turn any codebase into a searchable knowledge base with verified facts.

---

## Prerequisites

```bash
pip install -e .
export MULTIHEAD_DATA_DIR=~/.multihead
export OPENAI_API_KEY=sk-...  # or use local models
```

This creates a clean data directory:

```
~/.multihead/
  knowledge.db      — your knowledge base
  runs/             — execution logs
  artifacts/        — content-addressed storage
  nightshift/       — pipeline output + reports
  packs/            — context packs
  sessions/         — harvest manifests
  embeddings/       — vector cache
```

---

## Step 1: Initialize

```bash
cd /path/to/your/project
multihead init --auto
```

This detects your hardware and creates `config/heads.yaml`.

---

## Step 2: Harvest Your Conversations

If you use Claude Code, MultiHead already has access to your session transcripts:

```bash
multihead nightshift run --head openai-gpt41-nano --from-stage 0 --to-stage 0 --batch
```

This scans `~/.claude/projects/` and feeds transcripts into the knowledge store.

For non-Claude projects, deposit knowledge manually:

```bash
multihead deposit "Our API uses JWT tokens with 24h expiry" \
  -k auth.jwt.expiry -p my-team
```

---

## Step 3: Extract Knowledge

Run the extraction stages:

```bash
multihead nightshift run --head openai-gpt41-nano --from-stage 1 --to-stage 7 --batch
```

This chunks your conversations and extracts:
- **Entities** — components, tools, models mentioned
- **Topics** — what each conversation chunk is about
- **Events** — decisions, milestones, task completions
- **Claims** — durable knowledge with evidence and confidence

---

## Step 4: Scan Your Code

```bash
multihead nightshift run --head openai-gpt41-nano --from-stage 8 --to-stage 10 --batch
```

This creates independent observation channels:
- **Code AST** — what functions/classes exist
- **Behavioral LLM** — what the code actually DOES
- **Git history** — what changed and when
- **CI results** — what passes and what fails

---

## Step 5: Fuse and Verify

```bash
multihead nightshift run --head openai-gpt41-nano --from-stage 11 --to-stage 15 --batch
```

The fusion engine cross-references all channels:
- Claims from conversations get verified against code evidence
- Contradictions surface as contested claims
- Agreement across channels produces corroborated facts

---

## Step 6: Query Your Knowledge

### Search
```bash
multihead kb "authentication"
multihead kb "how does the router work" -s  # semantic search
```

### File briefing (before editing)
```bash
multihead briefing src/auth/jwt_handler.py
```

Output:
```
=== Knowledge: src/auth/jwt_handler.py ===
CONSTRAINTS (3 corroborated — don't violate):
  • JWT tokens use RS256 signing with 24h expiry
  • Refresh tokens stored in HTTP-only cookies
  • Token validation middleware runs on every /api/ route

WARNINGS (1 stale — verify before assuming):
  • Previous implementation used HS256 — changed in commit abc123

HISTORY (1 superseded — don't repeat):
  • Tried storing tokens in localStorage — XSS vulnerability, reverted
```

### Raw SQL
```sql
sqlite3 ~/.multihead/knowledge.db "
  SELECT claim_key, substr(statement, 1, 100)
  FROM claims
  WHERE claim_status = 'corroborated'
  ORDER BY confidence DESC
  LIMIT 10;
"
```

---

## Step 7: Use in Your Workflow

### Claude Code hooks (automatic)
Add to `~/.claude/settings.json`:
```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Edit|Write",
      "hooks": [{
        "type": "command",
        "command": "multihead briefing --compact $(jq -r '.tool_input.file_path' 2>/dev/null)",
        "timeout": 5
      }]
    }]
  }
}
```

Now every file edit automatically shows relevant knowledge.

### Deposit decisions as you work
```bash
multihead deposit "Switched from REST to GraphQL for the dashboard API" \
  -k architecture.dashboard.api_protocol -p my-name --type decision
```

### Resolve conflicts
```bash
multihead kb "dashboard API" --status contested
# Read both sides, then deposit resolution
multihead deposit "RESOLVED: GraphQL is correct, REST claim is stale" \
  -k resolution.dashboard.api -p my-name
```

---

## What Happens Over Time

| After | What you get |
|-------|-------------|
| Day 1 | Extracted claims from conversations + code |
| Week 1 | Fusion starts corroborating facts, stale claims detected |
| Month 1 | Institutional memory — new team members query the knowledge base instead of asking |
| Ongoing | Nightshift runs nightly, knowledge grows, briefings get richer |

---

## Full Pipeline (one command)

After setup, run the complete pipeline:

```bash
multihead nightshift run --head openai-gpt41-nano --batch --no-wait
# ... wait for batches, then:
multihead nightshift run --head openai-gpt41-nano --from-stage 4 --batch
```

Or with a local model (free, slower):

```bash
multihead nightshift run --head qwen-llm
```

---

## Next Steps

- Read the [Customization Guide](customization-guide.md) to adapt MultiHead for your domain
- Explore the [Shell](../README.md) for interactive knowledge management
- Set up nightly runs for continuous knowledge extraction
