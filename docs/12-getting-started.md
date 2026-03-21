# Getting Started with MultiHead

A guide for new Claude Code instances (or humans) joining the MultiHead network.

## Prerequisites

- MultiHead daemon running: `multihead serve` (localhost:7337)
- Python env activated: `source .venv/bin/activate`
- MCP configured (see connection options below)

## Connecting to MultiHead

### Option A: Claude Code CLI (stdio — already configured)

The repo has `.mcp.json` which auto-connects when you run Claude Code from the repo dir.

### Option B: Claude Desktop / Cowork (HTTP transport)

Cowork and Claude Desktop need an HTTP-based MCP server. Start it:

```bash
source .venv/bin/activate

# SSE transport (widely supported)
multihead mcp --transport sse --mcp-port 8338

# Or streamable-http (newer, supports session resumability)
multihead mcp --transport streamable-http --mcp-port 8338
```

Then add to your Claude Desktop config at
`C:\Users\<you>\AppData\Roaming\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "multihead": {
      "type": "url",
      "url": "http://localhost:8338/sse"
    }
  }
}
```

For Cowork: go to **Settings > Connectors > Add custom connector** and enter
`http://localhost:8338/sse` as the URL.

### Option C: Direct HTTP API (no MCP needed)

MultiHead's REST API is at `http://localhost:7337`. You can call it directly:

```bash
# Chat
curl -X POST http://localhost:7337/chat -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'

# List heads
curl http://localhost:7337/heads

# Query knowledge
curl "http://localhost:7337/knowledge/claims?status=accepted&limit=10"
```

### Option D: Python Client (for external processes)

Any Python process can talk to MultiHead using the built-in client:

```python
from multihead.client import MultiHeadClient

mh = MultiHeadClient()  # defaults to localhost:7337
print(mh.ping())        # True if server is up
```

---

## 1. Talking to the Local LLM

By default, MultiHead starts with a mock brain (no GPU required). Run `multihead init --auto` to detect your hardware and configure real models. You can chat through MCP:

```
# Chat with the local LLM (has tools, web search, knowledge access)
multihead_chat(message="What do you know about the project?")

# Continue a conversation thread
multihead_chat(message="Tell me more", session_id="ses_abc123")

# Raw inference on a specific model head (no tools, no context)
multihead_generate(head_id="qwen-llm", prompt="Summarize: ...")
```

## 2. Knowledge Store — Reading and Writing

The knowledge store holds **claims** (facts, decisions, constraints) and **events** (what happened). Any process can read from and write to it.

### Writing: Deposit claims

Claims are the building blocks of institutional memory. Deposit them when your process discovers or decides something:

**Via MCP:**
```
multihead_deposit_claim(
    claim_key="h2v.balloon_layout.balloon_count",
    statement="BubbleFill processed 12 balloons with 0 errors",
    produced_by="balloon_layout",
    claim_type="fact",
    confidence=1.0
)
```

**Via Python client:**
```python
from multihead.client import MultiHeadClient
mh = MultiHeadClient()

mh.deposit_claim(
    claim_key="h2v.balloon_layout.balloon_count",
    statement="BubbleFill processed 12 balloons with 0 errors",
    produced_by="balloon_layout",
    confidence=1.0,
)
```

**Via curl:**
```bash
curl -X POST http://localhost:7337/knowledge/claims \
  -H "Content-Type: application/json" \
  -d '{
    "claim_key": "h2v.balloon_layout.balloon_count",
    "statement": "BubbleFill processed 12 balloons with 0 errors",
    "produced_by": "balloon_layout",
    "confidence": 1.0
  }'
```

Claim fields:
- `claim_key` — dot-separated path (e.g. `h2v.balloon_layout.status`). Used for deduplication and lookup.
- `statement` — human-readable description of the claim
- `claim_type` — `fact`, `decision`, `constraint`, `preference`, `plan`, `assumption`, `risk`, `question`
- `claim_status` — `accepted` (default), `proposed`, `contested`, `superseded`, `rejected`
- `confidence` — 0.0 to 1.0
- `scope_id` — project scope (default `h2v`)
- `produced_by` — who/what created this claim

### Writing: Report events

Events track what happened — pipeline runs, completions, errors, decisions:

**Via MCP:**
```
multihead_report_event(
    title="Vertical layout export finished",
    summary="Exported page 5 with 8 panels, 12.5s duration",
    event_type="task_completed",
    produced_by="vertical_pipeline"
)
```

**Via Python client:**
```python
mh.report_event(
    title="Vertical layout export finished",
    summary="Exported page 5 with 8 panels",
    event_type="task_completed",
    produced_by="vertical_pipeline",
    metrics={"panels": 8, "duration_s": 12.5},
    tags=["h2v", "export"],
)
```

Event types: `note`, `task_completed`, `task_created`, `decision`, `commit`, `milestone`, `incident`, `tool_run`, `spec_change`, `question`, `answer`

### Reading: Get a component briefing

A component calls this at startup to learn what it needs to know before running. Returns direct claims (key matches), related claims (statement mentions), and recent events:

**Via MCP:**
```
multihead_briefing(component="balloon_layout")
```

**Via Python client:**
```python
briefing = mh.get_briefing("balloon_layout")
# {
#   "claims": [...],         # direct claims (key contains "balloon_layout")
#   "related_claims": [...], # claims mentioning balloon_layout in statement
#   "recent_events": [...],  # events tagged with or mentioning balloon_layout
#   "summary": "3 direct claims, 1 related, 2 events"
# }
```

**Via curl:**
```bash
curl "http://localhost:7337/knowledge/briefing?component=balloon_layout&scope_id=h2v"
```

### Reading: Query claims and events

**Via MCP:**
```
# Facts the system has learned
multihead_knowledge(query_type="claims", status="accepted", limit=20)

# Events that happened
multihead_knowledge(query_type="events", status="confirmed", limit=20)
```

**Via Python client:**
```python
claims = mh.query_claims(status="accepted", scope_id="h2v", limit=20)
events = mh.query_events(event_type="task_completed", limit=10)
```

## 3. Task Decomposition

Break complex goals into hierarchical execution plans:

**Via MCP:**
```
multihead_decompose(
    goal="Fix the text overflow in balloon layout",
    context="The freeze threshold is too high at 50%"
)
```

**Via Python client:**
```python
plan = mh.decompose(
    goal="Fix the text overflow in balloon layout",
    context="The freeze threshold is too high at 50%",
)
# Returns phases → steps → leaves, each with action_type, target_files, expected_output
```

Refine any step into sub-steps:
```
multihead_refine_step(
    node_id="2.1",
    node_goal="Modify freeze threshold",
    action_type="edit"
)
```

## 4. The Agent Mesh

Three agents communicate through BotVibes ACP (Agent Commons Protocol):

| Agent | Role | Capabilities |
|-------|------|-------------|
| `multihead-agent` | Qwen LLM on GPU | `com.multihead.llm`, `com.multihead.vlm` |
| `claude-session-agent` | Claude Code worker daemon | `com.claude.code` |
| `claude-vibebots-agent` | BotVibes coder | `com.botvibes.*` |

### Check your inbox

```
multihead_check_tasks(capability="com.claude.code")
```

### Claim and complete a task

```
# 1. See what's waiting
multihead_check_tasks()

# 2. Claim it (exclusive lock)
multihead_claim_task(task_id="<uuid>")

# 3. Do the work...

# 4. Submit the result
multihead_complete_task(task_id="<uuid>", output_ref="Fixed the bug in auth.py")
```

### Send tasks to other agents

```
# Ask the local Qwen LLM
multihead_create_task(
    capability="com.multihead.llm",
    payload_ref="Analyze this error log and suggest fixes",
    target_agent_id="multihead-agent"
)

# Delegate to the Claude worker daemon (spawns headless claude -p)
multihead_delegate_claude(
    prompt="Review and fix all TODO comments in src/",
    conversation_id="conv-123"  # optional, threads related tasks
)
```

### Conversation threading

Pass `conversation_id` to keep related tasks in a thread. The worker daemon uses `--resume` under the hood so Claude keeps context across turns.

## 5. GPU Model Heads

Only one GPU model at a time. Hot-swap as needed:

```
# See what's available and what's loaded
multihead_heads()

# Load a model
multihead_swap_head(head_id="qwen-vlm-8b", action="wake")

# Switch back
multihead_swap_head(head_id="qwen-llm", action="wake")

# Free GPU memory
multihead_swap_head(head_id="qwen-vlm-8b", action="unload")
```

| Head | Model | VRAM | Kind |
|------|-------|------|------|
| `qwen-llm` | Qwen3-8B | ~6 GB | text |
| `qwen-vlm-7b` | Qwen3-VL-7B | ~5 GB | vision |
| `qwen-vlm` | Qwen3-VL-32B-Thinking | ~18 GB | vision |
| `openai-gpt4o` | GPT-4o Mini | 0 (API) | text |

## 6. Runtime Config

```
# View all settings
multihead_config(action="show")

# Toggle features
multihead_config(action="set", key="web_tools_enabled", value="true")
multihead_config(action="set", key="strip_thinking", value="false")
```

## 7. Pipeline Recipes

YAML-defined multi-step pipelines that chain model calls:

```
# Run a recipe
multihead_run_recipe(recipe="my-recipe", inputs={"text": "hello"})

# Check status
multihead_run_status(run_id="<run_id>")
```

Recipes live in `config/recipes/`.

## 8. Narrative Pipeline (Knowledge Recording)

The narrative pipeline builds institutional memory by extracting claims and events from all activity.

### What's recorded automatically

- Chat exchanges in the Agentic Core (user / Qwen conversations)
- Completed daemon tasks (Claude worker results)
- These buffer as evidence. Fusion into claims/events runs separately.

### Manual ingestion (CLI)

```bash
# Ingest recent git commits
multihead narrative ingest --source git --path . --since 7d

# Ingest and immediately fuse into claims
multihead narrative ingest --source git --path . --since 7d --fuse

# Run fusion on all buffered evidence
multihead narrative fuse

# See what's in the knowledge store
multihead narrative status
```

## Key Paths

| What | Where |
|------|-------|
| Repo | `.` (wherever you cloned it) |
| Data | `$MULTIHEAD_DATA_DIR` or `~/.multihead` |
| Python env | `.venv/` (created by install.sh) |
| GitHub | `https://github.com/Axsar/multihead.git` |
| Head configs | `config/heads.yaml` |
| Recipes | `config/recipes/` |
| Knowledge DB | `<data_dir>/knowledge.db` |
| Logs | `<data_dir>/multihead.log` |

## Running Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

## Namespace Convention

Tasks are routed by capability prefix:

- `com.multihead.*` — targeting MultiHead (local LLM)
- `com.botvibes.*` — targeting BotVibes
- `com.claude.code` — shared fallback for Claude instances
- `com.multihead.interactive` — reserved for human-attended sessions (daemon skips these)
