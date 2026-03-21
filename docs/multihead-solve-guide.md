# MultiHead Solve: Multi-Agent Task Solving Guide

**Version**: v1.0 (2026-02-28)
**Status**: Production-ready

---

## Overview

**MultiHead Solve** enables structured task solving across three modes:
1. **Self-Solve** (MCP) — Any Claude Code session decomposes and executes tasks itself, with RAG context from knowledge.db and run tracking. No GPU, no serve, no Qwen needed.
2. **Simple** (CLI) — Local Qwen LLM decomposes and executes immediately.
3. **Advanced** (Multi-Session) — Multiple agents propose decompositions, vote via consensus, then execute the winner.

**Key insight**: The agent with direct codebase access produces the best decompositions. Self-solve leverages this — the MCP server provides knowledge context and tracking, while the calling session does the actual thinking and work.

---

## The 10-Step Autonomous Workflow

When you run `multihead solve`, the system executes a 10-step pipeline that handles everything from task intake to knowledge feedback. Each step is modular — the pipeline adapts based on what's available (local heads, ACP agents, knowledge history).

```
Step 1:  Task Intake          ── CLI, MCP, or chat
Step 2:  Auto-Decomposition   ── Break goal into atomic steps (DAG)
Step 3:  Consensus Vote       ── Multi-agent vote on decomposition plan
Step 4:  Rank Solvers         ── Router scores all heads per step (multi-candidate)
Step 5:  Assign Solvers       ── Primary + fallback heads assigned to each step
Step 6:  Execute Locally      ── GPU mutex, head swap, validators, consensus
Step 7:  Delegate Remotely    ── Steps with budget + task_types → ACP agents
Step 8:  VRAM Policy          ── Memory management between steps
Step 9:  Knowledge Extraction ── KnowledgeHook writes claims inline
Step 10: Feedback Loop        ── Router reads past success/failure claims
```

### How Steps Connect

**Steps 1-3** (planning): Your task is decomposed into atomic steps and validated by multiple agents. The winning decomposition becomes the execution plan.

**Steps 4-5** (routing): The Router scores every registered head for each step using weighted criteria — active state (40), circuit breaker (30), VRAM fit (15), error rate (10), latency (5), plus knowledge-based boosts from Step 10. Top candidate becomes primary, next 2 become fallbacks.

**Steps 6-7** (execution): Each step either runs locally (head swap + GPU mutex) or delegates to a remote ACP agent. The decision depends on:
- **Local** (Step 6): No budget constraint, or CONFIDENTIAL data sensitivity
- **Remote** (Step 7): Has `budget` + `task_types`, data sensitivity is PUBLIC or INTERNAL, and ACP bridge is connected

**Steps 9-10** (learning): As steps complete, KnowledgeHook writes success/failure claims to knowledge.db. On future runs, the Router reads these claims and adjusts head scores — heads that succeeded get boosted, heads that failed get penalized.

### ACP Remote Delegation (Step 7)

When a step is eligible for remote delegation:
1. Step's `task_types` map to an ACP capability (e.g., `text_generation` → `com.multihead.llm`)
2. An ACP task is created with the step's prompt, budget constraints, and privacy requirements
3. The orchestrator polls for completion (adaptive intervals: 2s → 5s → 10s → 30s)
4. On success, the remote result is stored as a local artifact
5. On failure or timeout, the step falls back to local execution

**Capability mapping**:

| task_type | ACP capability |
|-----------|---------------|
| `text_generation` | `com.multihead.llm` |
| `visual_reasoning` | `com.multihead.vlm` |
| `code_editing` | `com.claude.code` |
| `object_detection` | `com.multihead.vlm` |
| `coordinate_transform` | `com.multihead.deterministic` |

Steps without an ACP bridge fall back to RFQ-only mode (creates the request but doesn't submit it).

### Knowledge Feedback Loop (Steps 9-10)

The feedback loop makes the system learn from experience:

```
Run N:   step executes → KnowledgeHook writes "head.qwen-llm.success" claim
Run N+1: Router reads claim → boosts qwen-llm score by +5
Run N+2: qwen-llm selected more often → more success claims → stronger preference
```

Failure claims work the same way in reverse — heads that fail get penalized by -5, clamped to [-10, +10] total adjustment.

---

## Three Modes: Self-Solve, Simple, Advanced

### Self-Solve Mode (MCP, Recommended)

**Use when**: You're a Claude Code session (any project) and want structured task execution with run tracking.

**How it works**: The MCP server provides RAG context and run tracking. **You** do the actual decomposition and execution — you have the best context (loaded codebase, file access, conversation history).

**What the MCP server does**:
- Queries knowledge.db for claims relevant to your task (keyword matching)
- Returns a decomposition template + knowledge context for you to fill in
- Creates a run directory (`runs/<run_id>/`) with events.jsonl
- Tracks step completion and timing

**What the MCP server does NOT do**:
- It does NOT decompose the task — you do (you can read the actual code)
- It does NOT execute steps — you do (you have file access and tools)
- It does NOT call any LLM — the "decomposition" is a template returned to you

**The 3-call flow**:

```
1. multihead_solve(task="Fix text overflow in narrow balloons", mode="self")
   → Returns: run_id + decomposition template + knowledge claims
   → You: read relevant code, produce a step-by-step plan, execute each step

2. multihead_complete_step(run_id="run_...", step_id="1.1", output="Added clamp to line 42")
   → Call after each step — logs to events.jsonl, saves output as artifact

3. multihead_finalize_solve(run_id="run_...")
   → Call when all steps done — records duration and final metrics
```

**Prerequisites**: The `multihead-stdio` MCP server must be configured. If added at user scope (`claude mcp add --scope user multihead-stdio ...`), it's available to all Claude Code sessions on the machine.

**When to use**:
- Any task from any Claude Code session (any project directory)
- Solo development with structured tracking
- Tasks where you (the caller) have the best context
- When `multihead serve` is NOT running

---

### Simple Mode (Single Session, CLI)

**Use when**: You're working solo via the CLI and want fast, autonomous execution.

**How it works**:
1. You post a task
2. Your local Qwen LLM decomposes it immediately (no voting)
3. Task executes instantly
4. You get results

**Command**:
```bash
multihead solve "Add error handling to API endpoint" --auto-approve
```

**Result**: Zero-delay execution, single decomposition strategy.

**When to use**:
- Quick iterations via CLI
- Low-stakes tasks
- When `multihead serve` is running with a GPU head loaded

---

### Advanced Mode (Multi-Session)

**Use when**: You want multiple expert perspectives before committing to an approach.

**How it works**:
1. **You** (coordinator) post a task decomposition request to knowledge.db
2. **Other agents** (Qwen, Claude sessions) detect the request via `/collab` notifications
3. Each agent proposes a decomposition plan
4. **Consensus voting** selects the best proposal (MAJORITY, WEIGHTED, UNANIMOUS, etc.)
5. Winning proposal assigned back to proposing agent for execution
6. You monitor or review results

**Command**:
```bash
multihead solve "Refactor authentication system for OAuth2 support" --timeout 300
```

**Timeline**:
- 0s: Request posted to knowledge.db
- 0-90s: Agents detect and respond (human bottleneck)
- 60s after minimum: Collection window closes
- ~2-3 min total: Consensus vote + assignment

**Result**: Best-of-N decomposition, validated by multiple expert systems.

**When to use**:
- Architectural decisions
- Complex refactors
- High-stakes changes
- Learning from other perspectives

---

## The `/collab` Workflow

MultiHead uses a **non-interrupting notification** pattern to respect user flow:

### Step-by-Step

1. **Request Posted** (by another session)
   - Task decomposition request written to knowledge.db
   - Tagged with `claim_type=QUESTION` and scope

2. **Your Session Detects It** (on next message)
   - You send any message to your MultiHead chat
   - After response, you see:
     ```
     [collab] 2 requests pending — type /collab to review
     ```

3. **Review Requests** (when you're ready)
   ```
   > /collab

   [Cross-Session Collaboration] 2 pending request(s):

   1. FROM: claude-h2v
      Task: Add unit tests for balloon layout algorithm...
      ID: clm_ABC123...

   2. FROM: multihead-coordinator
      Task: Refactor authentication for OAuth2...
      ID: clm_XYZ789...

   To respond: /collab-respond <id>
   To ignore: /collab-ignore <id>
   ```

4. **Participate** (your choice)
   ```
   > /collab-respond clm_ABC123

   [System] Decomposing task...
   [System] Submitted proposal clm_DEF456 for request clm_ABC123
   ✓ Response submitted
   ```

5. **Coordinator Collects Responses**
   - Waits 60s after hitting minimum proposals
   - Runs consensus voting
   - Assigns work to winning proposer

6. **You Get Assignment** (if your proposal won)
   ```
   [collab] 1 request pending — type /collab to review

   > /collab

   [Work Assignment]
   FROM: multihead-coordinator
   Your proposal was selected. Execute: <decomposition>...
   ```

---

## Architecture: Three Layers

MultiHead's architecture clarifies what "local" means:

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: BotVibes Marketplace (Optional)                   │
│ - External expert solvers                                  │
│ - Privacy-preserving delegation (E2E encryption)           │
│ - Reputation-weighted voting                               │
│ - Recipe learning from community                           │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │ (Optional, privacy-constrained)
                            │
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: MultiHead Solve (Multi-Agent Consensus)           │
│ - Qwen LLM + Multiple Claude Sessions                      │
│ - Knowledge.db coordination (SQLite)                       │
│ - Consensus voting (MAJORITY, WEIGHTED, UNANIMOUS, etc.)   │
│ - /collab workflow                                         │
│ - Capability discovery                                     │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │ (Your infrastructure)
                            │
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Individual Solvers (Local-First)                  │
│ - Qwen3-8B LLM (transformers, 4bit)                        │
│ - Qwen3-VL-32B VLM (transformers, 4bit)                    │
│ - Deterministic solvers (coordinate transforms, heuristics)│
│ - Python tools (file I/O, git, shell)                      │
│ - Optional: Ollama, vLLM, Claude API, OpenAI API           │
└─────────────────────────────────────────────────────────────┘
```

**Key insight**: "Local" means **your infrastructure**, not "single process."

- **Layer 1** works standalone (zero dependencies)
- **Layer 2** adds multi-session consensus (still local: same machine or LAN)
- **Layer 3** adds external marketplace (optional, privacy-aware)

**Design principle**: Each layer works standalone, no forced dependencies upward.

---

## Real-World Example: Gap Analysis

**Scenario**: Strategic alignment analysis identified 2 integration gaps. Should we implement immediately or verify first?

### Step 1: Post to MultiHead Solve

```bash
multihead solve "Close integration gaps #1 and #2: Add knowledge_store to HeadManager, wire session poller to decomposer" --timeout 300
```

### Step 2: Agents Respond

**3 proposals received**:

1. **BotVibes**: Full implementation plan (4 steps, 42 lines, detailed edits)
2. **Claude Code Main**: "CRITICAL FINDING: BOTH GAPS ALREADY CLOSED" (audit report)
3. **Vertical Pipeline**: (timed out, didn't respond)

### Step 3: Consensus Vote

**Winner**: BotVibes (first to respond, detailed plan)

### Step 4: Verification (Before Execution)

Coordinator read Claude Code Main's audit claim:
> "Gap #1 is 100% complete (already wired in 3 CLI locations).
> Gap #2 detection works but response loop missing (~20 lines)."

Verified by reading actual code → **Claude Code Main was right!**

### Outcome

**Consensus value**: Saved 1-2 days of duplicating existing work. BotVibes planned full implementation, Claude Code Main caught that it's already done.

**Decision**: Mark gaps as closed, update docs instead of implementing.

---

## Test Run Evolution: Learning Through Iteration

Real-world progression of the git branch test (2026-02-22):

### Run #1: Timeout Too Short (60s)

**Command**:
```bash
multihead solve "Check if we're on a git branch, if not create one" --timeout 60
```

**Result**: ✗ 0 proposals received

**Timeline**:
- 0-60s: Collection window
- Result: Timeout before any agent responded

**Lesson**: Human bottleneck is real. Agents need time to see `/collab` notification, read request, decompose, and respond. 60s is too aggressive for human-in-loop systems.

### Run #2: Stability Enum Bug (60000s timeout)

**Command**:
```bash
multihead solve "Check if we're on a git branch, if not create one" --timeout 60000
```

**Result**: ✗ Crashed with ValueError

**Error**:
```
ValueError: 'high' is not a valid Stability
```

**Root cause**: Agents posting claims with `stability="high"` and `stability="temporary"`, but `Stability` enum only had `VOLATILE`, `MEDIUM`, `STABLE`.

**Fix**: Added `HIGH` and `TEMPORARY` to enum (knowledge_models.py:82-87).

**Lesson**: Multi-agent systems expose enum mismatches that single-agent code might never hit.

### Run #3: Early-Stop Bug (300s timeout)

**Command**:
```bash
multihead solve "Check if we're on a git branch, if not create one" --timeout 300
```

**Result**: ⚠️ Only 1/3 proposals collected

**Timeline**:
- 35s: First proposal (claude_code_main) → minimum met
- **Stopped immediately** (buggy code)
- Missed: Proposals from vertical_pipeline (45s) and BotVibes (75s)

**Bug**: Code stopped when `len(proposals) >= min_proposals` instead of waiting for collection window.

**Fix**: Added 60s collection window after hitting minimum.

**Lesson**: Agents don't respond simultaneously. Need patience to collect diverse perspectives.

### Run #4: Success! (300s timeout, fixed)

**Command**:
```bash
multihead solve "Check if we're on a git branch, if not create one" --timeout 300
```

**Result**: ✅ All 3 proposals collected

**Timeline**:
- 35s: First proposal (claude_code_main) → minimum met
- **Waited 60s** for additional proposals
- 45s: Second proposal (vertical_pipeline)
- 75s: Third proposal (BotVibes)
- 95s: Collection complete, 3 total proposals
- Consensus vote executed → BotVibes won
- Work assigned to claude-vibebots-agent

**Validation**: System stable, reproducible, consensus-based.

**Lessons learned**:
1. **Timeout tuning**: 300s (5 min) is sweet spot for 2-3 agents
2. **Collection window**: 60s after minimum catches late responders
3. **Human bottleneck**: 35-90s response times are normal (humans reading, thinking, typing)
4. **Stability**: All 3 agents responded consistently across multiple tests

---

## Architecture Consensus: 11/11 Unanimous Validation

Before implementing MultiHead Solve, we ran architecture consensus voting (2026-02-22):

**Request**: "Should we use pull-based (agents poll knowledge.db) or push-based (coordinator pushes to agents) architecture?"

**Participants**:
- claude-multihead-main
- claude-vibebots-agent
- vertical_pipeline_implementation

**Voting rounds**: 2 (initial + final)

**Consensus points achieved**: 11/11 unanimous

### Key Decisions Validated by Consensus

1. ✅ **Pull-based architecture** (agents poll knowledge.db, not push)
   - Simpler: No WebSocket/HTTP server per agent
   - More reliable: No connection failures, no port conflicts
   - Agent autonomy: Each agent polls at own pace

2. ✅ **Knowledge.db as message bus** (SQLite file, not REST API)
   - Local-first: File-based, no network required
   - Durable: Survives process crashes
   - Queryable: Full SQL access for debugging

3. ✅ **Claim-based protocol** (structured Claims, not free-form messages)
   - DECOMP_REQUEST → DECOMP_PROPOSAL → WORK_ASSIGNMENT
   - Evidence tracking: Provenance, timestamps, confidence
   - Versioned: Protocol can evolve

4. ✅ **--auto-approve flag default OFF** (manual approval by default)
   - Safety: User reviews assignment before execution
   - Flexibility: Can override with `--auto-approve` for CI/CD
   - Transparency: Explicit control over automation

5. ✅ **Coordinator model** (not peer-to-peer)
   - Simpler: One leader, N followers
   - Clearer: Coordinator posts request, assigns work
   - Evolvable: Can add P2P later (v1.0 LMP)

**Result**: Zero architectural conflicts after implementation. All 3 agents agreed on every major decision upfront.

**This documentation itself validates the consensus approach** - multiple agents proposed similar structures independently (95% alignment).

---

## Use Cases

### When to Use Self-Solve Mode (MCP)

✅ **Good for**:
- Any Claude Code session that wants structured task tracking
- Cross-project work (vp coder working on H2V can use multihead tools)
- Tasks where the caller has loaded codebase context
- When `multihead serve` is not running
- Solo development with run history (events.jsonl, artifacts)

❌ **Not ideal for**:
- Tasks that need GPU heads (VLM, local LLM inference)
- Multi-agent consensus (use Advanced mode)

### When to Use Simple Mode (CLI)

✅ **Good for**:
- Quick iterations and prototyping via CLI
- Low-stakes refactors
- Tasks with obvious solutions
- When Qwen is loaded and `multihead serve` is running

❌ **Not ideal for**:
- Architectural decisions
- High-stakes production changes
- Learning opportunities (multiple perspectives)

### When to Use Multi-Session Mode

✅ **Good for**:
- Complex architectural decisions
- High-stakes production changes
- Learning from other perspectives
- Catching edge cases early
- Validating assumptions

❌ **Not ideal for**:
- Time-critical quick fixes
- Solo work with no other sessions available
- Simple, obvious tasks

### When to Use Layer 3 (BotVibes)

✅ **Good for**:
- Tasks requiring specialized expertise you don't have locally
- Recipe learning (what's the best way to do X?)
- Performance benchmarking
- Access to commercial models (Claude Opus, GPT-4)

❌ **Not ideal for**:
- Sensitive proprietary code
- Tasks requiring local file access
- Offline development

---

## CLI Reference

### Command Syntax

```bash
multihead solve "<task>" [OPTIONS]
```

### Required Arguments

**`<task>`** (positional)
- Natural language task description
- Quoted string (use `"..."` for multi-word tasks)
- Examples:
  - `"Add unit tests for authentication"`
  - `"Refactor database queries for performance"`
  - `"Fix bug in user profile page"`

### Optional Flags

**`--timeout <seconds>`** (default: 300)
- How long to wait for agent proposals
- Minimum: 60s (shorter may miss human responses)
- Recommended: 300s (5 min) for 2-3 agents
- Extended: 43200s (12 hours) for async overnight collaboration
- Example: `--timeout 600` (10 minutes)

**`--auto-approve`** (default: OFF)
- Skip manual approval, execute winning proposal immediately
- Use for: CI/CD, scripting, low-stakes tasks
- Warning: No human review before execution
- Example: `multihead solve "Run tests" --auto-approve`

**`--project-id <id>`** (default: "multihead")
- Project scope for filtering requests/proposals
- All agents in same project see each other's requests
- Use different IDs for isolated workstreams
- Example: `--project-id h2v` (H2V comic processing project)

**`--min-proposals <n>`** (default: 1)
- Minimum proposals before consensus voting
- 1 = accept first response (fast)
- 2+ = wait for multiple perspectives
- Example: `--min-proposals 3` (require 3+ agents)

**`--max-proposals <n>`** (default: 10)
- Maximum proposals to collect
- Prevents unbounded waiting if many agents respond
- Example: `--max-proposals 5`

**`--strategy <strategy>`** (default: MAJORITY)
- Consensus voting algorithm
- Options: MAJORITY, WEIGHTED, UNANIMOUS, THRESHOLD, FIRST_TO_AHEAD
- See [Consensus Strategies](#consensus-strategies) section
- Example: `--strategy UNANIMOUS` (all agents must agree)

### Environment Variables (ACP Integration)

**`ACP_URL`** — ACP server URL (e.g., `http://localhost:8000/api/v1`)
- When set (along with `ACP_API_KEY`), the solve command connects to BotVibes/ACP
- Enables remote delegation for steps with `budget` + `task_types`
- Without this, all steps execute locally

**`ACP_API_KEY`** (or `ACP_SESSION_KEY`) — JWT token for ACP authentication
- Required alongside `ACP_URL` for remote delegation
- Obtain from BotVibes admin or token generation endpoint

**`ACP_AGENT_ID`** — Agent identity for ACP registration (default: `multihead-agent`)

**`ACP_TENANT_ID`** / **`ACP_PROJECT_ID`** — BotVibes tenant and project scope

### Examples

**Quick task (solo, instant)**:
```bash
multihead solve "Fix typo in README" --auto-approve
```

**Architectural decision (multi-agent, 5 min)**:
```bash
multihead solve "Should we use Redis or Memcached for caching?" --timeout 300 --min-proposals 2
```

**Overnight async (12 hour window)**:
```bash
multihead solve "Design API v2 with backward compatibility" --timeout 43200 --project-id api-redesign
```

**High-stakes (require unanimous agreement)**:
```bash
multihead solve "Migrate production database schema" --strategy UNANIMOUS --timeout 600
```

**With ACP remote delegation** (steps with budget delegate to remote agents):
```bash
ACP_URL=http://localhost:8000/api/v1 ACP_API_KEY=<token> \
  multihead solve "Detect objects in dataset" --auto-approve
```

**Local only** (no ACP, default behavior):
```bash
multihead solve "Refactor auth module" --auto-approve
```

---

## Configuration

### Single Session Setup

**Default**: Just works. No configuration needed.

```bash
multihead solve "Your task here" --auto-approve
```

### Multi-Session Setup

**Requirements**:
- Multiple Claude Code CLI sessions OR
- Qwen LLM + 1+ Claude sessions

**Step 1**: Each session needs a unique ID and project scope.

Edit `~/.claude/config/multihead.json` (or via environment):
```json
{
  "session_id": "claude-h2v",
  "project_id": "h2v"
}
```

**Step 2**: All sessions share the same knowledge.db.

Set `MULTIHEAD_DATA_DIR` to a shared location:
```bash
export MULTIHEAD_DATA_DIR=/mnt/shared/multihead
```

**Step 3**: Start all sessions.

```bash
# Terminal 1
multihead chat  # Session: claude-h2v

# Terminal 2
multihead chat  # Session: claude-botvibes

# Terminal 3 (coordinator)
multihead solve "Your task here" --timeout 300
```

**Optional**: Use tmux or screen for persistent sessions.

---

## Consensus Strategies

MultiHead supports 5 voting strategies:

### MAJORITY (Default)

**How it works**: Most votes wins (N/2 + 1 threshold).

**When to use**: Standard multi-agent voting, balanced confidence.

**Example**:
- 3 agents propose solutions A, A, B
- Solution A wins (2/3 votes)

### WEIGHTED

**How it works**: Votes weighted by agent expertise/track record.

**When to use**: Some agents are domain experts.

**Example**:
- Agent 1 (weight 2.0): Solution A
- Agent 2 (weight 1.0): Solution B
- Agent 3 (weight 1.0): Solution A
- Weighted score: A=4.0, B=1.0 → A wins

### UNANIMOUS

**How it works**: All agents must agree.

**When to use**: High-stakes decisions, safety-critical changes.

**Example**:
- 3 agents propose: A, A, A → A wins
- 3 agents propose: A, A, B → NO CONSENSUS (revote or abort)

### THRESHOLD

**How it works**: Configurable percentage (e.g., 75% agreement).

**When to use**: Flexible quorum-based decisions.

**Example** (threshold=0.75):
- 4 agents propose: A, A, A, B
- A has 75% → wins

### FIRST_TO_AHEAD

**How it works**: Dynamic sampling with k-margin convergence.

**When to use**: Expensive proposals, want to stop early when clear winner emerges.

**Example**:
- Agent 1: Solution A (t=30s)
- Agent 2: Solution A (t=45s) → A ahead by k=2
- Stop early, don't wait for Agent 3 (saves time)

**Advanced**: Includes red-flag pre-filtering (obvious bad proposals rejected immediately).

---

## Tips & Best Practices

### For Coordinators

1. **Use descriptive task names**: "Add OAuth2 support" not "fix auth"
2. **Set reasonable timeouts**: 300s (5 min) is standard, 60s for quick tasks
3. **Check knowledge.db periodically**: `multihead query claims --status=accepted`
4. **Monitor for assignments**: Other sessions may assign work back to you

### For Responders

1. **Check `/collab` regularly**: Notifications only appear after you send a message
2. **Respond thoughtfully**: Your proposal competes with others
3. **Use `/collab-ignore` liberally**: Don't feel obligated to respond to everything
4. **Learn from other proposals**: Even if you lose the vote, read the winner

### For Everyone

1. **Trust the consensus**: If 3 agents all vote differently, maybe the task needs clarification
2. **Document decisions**: Winning proposals are stored in knowledge.db
3. **Iterate**: First solve can be exploratory ("should we even do this?")
4. **Use claims**: Deposit findings back to knowledge.db for future tasks

---

## Troubleshooting

### "No proposals received within timeout"

**Cause**: No other sessions actively monitored during collection window.

**Solutions**:
1. Extend timeout: `--timeout 600` (10 minutes)
2. Check other sessions are running: `multihead chat` in separate terminals
3. Proceed solo: `--auto-approve` flag
4. Post as standing request, continue with other work (async)

### "Only 1 proposal when expecting 3"

**Cause**: Other agents didn't see notification before collection window closed.

**Solutions**:
1. Increase collection window (hardcoded 60s after minimum)
2. Check other sessions are actively used (send a message to trigger check)
3. Use longer timeout to give humans time to respond

### "All proposals are identical"

**Cause**: Task is too simple or obvious.

**Interpretation**: Consensus is strong! All agents agree on approach.

**Action**: Accept the unanimous proposal, proceed with confidence.

### "Proposals wildly different, no clear winner"

**Cause**: Task is ambiguous or under-specified.

**Solutions**:
1. Clarify requirements and re-run solve
2. Use UNANIMOUS strategy to force alignment
3. Ask agents to vote on which proposal to refine
4. Break task into smaller, clearer sub-tasks

---

## Advanced: Custom Consensus

You can implement custom voting logic by extending `ConsensusEngine`:

```python
from multihead.consensus import ConsensusEngine, ConsensusStrategy

class CustomVoting(ConsensusEngine):
    def vote(self, proposals, strategy=ConsensusStrategy.CUSTOM):
        # Your logic here
        # Example: Prefer proposals from agents with recent success
        scores = self._score_by_recent_success(proposals)
        return max(proposals, key=lambda p: scores[p.claim_id])
```

See `src/multihead/consensus.py` for examples.

---

## FAQ

**Q: Do I need BotVibes to use MultiHead Solve?**
A: No. MultiHead Solve works entirely via local knowledge.db (SQLite). BotVibes is Layer 3 (optional marketplace).

**Q: Can I use MultiHead Solve offline?**
A: Yes, if all sessions are local (same machine or LAN). No internet required.

**Q: How many agents do I need for consensus?**
A: Minimum 1 (solo mode), recommended 3+ for meaningful voting.

**Q: What if agents vote 1-1-1 (3-way tie)?**
A: First proposal wins (tiebreaker). Or use UNANIMOUS to force re-vote.

**Q: Can I see all proposals even if mine didn't win?**
A: Yes. Query knowledge.db: `multihead query claims --related-to <request_id>`

**Q: How do I know if my proposal was selected?**
A: You'll receive a work assignment via `/collab` notification.

**Q: Can I override the consensus vote?**
A: Yes (coordinator privilege). Manually assign work via knowledge.db claims.

**Q: What's the performance overhead of multi-session voting?**
A: ~2-3 minutes for proposal collection, ~1s for voting. Execution time unchanged.

---

## Protocol Specification

For deep technical details on the claim-based protocol:

**See**: [claude-session-consensus.md](claude-session-consensus.md)

Covers:
- DECOMP_REQUEST claim format
- DECOMP_PROPOSAL response format
- WORK_ASSIGNMENT protocol
- Provenance tracking
- Claim lifecycle (PROPOSED → ACCEPTED)

---

## Related Documentation

- [Architecture Overview](01-architecture.md) - Three-layer design
- [Knowledge Store](07-claim-event-schemas.md) - Claims and events
- [Consensus Strategies](../src/multihead/consensus.py) - Implementation details
- [Claude Session Consensus](claude-session-consensus.md) - Multi-session voting protocol (detailed)
- [Capability Discovery](capability-discovery-demo.md) - How agents advertise skills
- [BotVibes Integration](botvibes-integration-update.md) - Layer 3 marketplace

---

## Version History

- **v1.1** (2026-03-14): Self-Solve mode via MCP
  - `multihead_solve(mode="self")` — caller decomposes and executes, MCP provides RAG + tracking
  - `multihead_complete_step()` — per-step completion logging with artifact storage
  - `multihead_finalize_solve()` — run finalization with duration metrics
  - No dependency on `multihead serve` or GPU heads
  - User-scoped MCP server config (`claude mcp add --scope user`) for cross-project access
- **v1.0** (2026-02-28): Full 10-step autonomous workflow
  - ACP remote delegation (Steps 6-7): steps with budget + task_types delegate to BotVibes agents
  - Multi-candidate routing (Steps 4-5): Router returns ranked list with fallbacks
  - Knowledge feedback loop (Steps 9-10): Router reads success/failure claims to improve future routing
  - Capability mapping: 18 task_types mapped to ACP capability namespaces
  - CLI wiring: `solve` command creates ACP bridge when env vars are set
- **v0.75** (2026-02-22): Initial MultiHead Solve release
  - SolveCoordinator implementation
  - 5 consensus strategies
  - /collab workflow
  - All integration gaps closed

---

**Questions?** Open an issue: https://github.com/Axsar/multihead/issues
