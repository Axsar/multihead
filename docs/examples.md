# MultiHead Examples

These examples show what MultiHead can do. For the quickstart, see the [README](../README.md#start-here).

---

## 1. Agent-to-Agent Communication

See [README](../README.md#1-agent-to-agent-communication) — two agents sharing knowledge across repos via deposit/query/briefing.

---

## 2. Autonomous Solve

See [README](../README.md#2-autonomous-solve) — decompose, route, execute, and learn in one command.

---

## 3. Distributed Solve (Multi-Agent)

See [README](../README.md#3-distributed-solve-multi-agent) — specialist agents propose plans, consensus picks the winner.

---

## 4. Resurrect Old Code

See [README](../README.md#4-resurrect-old-code) — point MultiHead at a forgotten repo, discover capabilities, publish to marketplace.

---

## 5. Task Decomposition (Dry Run)

Preview what MultiHead will do before it runs. The decomposer breaks goals into steps, infers a DAG for parallel execution, and auto-enables research features.

```bash
multihead solve "Add rate limiting to the API gateway" --dry-run --show-plan
```

```
Decomposition Plan (3 heads voted, agreement: 87%)
Complexity: moderate | Steps: 6 | Parallel: 2

Phase 1 — Explore
  1.1 Read current middleware stack          [explore]  → qwen-llm
  1.2 Read existing rate limit configs       [explore]  → qwen-llm
       ↑ runs in parallel with 1.1 (no dependency)

Phase 2 — Implement
  2.1 Create rate limiter middleware         [create]   → qwen-llm
  2.2 Wire into gateway router              [edit]     → qwen-llm
       ↑ depends on 2.1

Phase 3 — Verify
  3.1 Write integration tests               [test]     → qwen-llm
  3.2 Load test with 1000 req/s             [verify]   → qwen-llm

Research features auto-enabled:
  • 1.1, 1.2: Tree-of-Thoughts (explore alternatives)
  • 2.1, 2.2: Process Reward Model (score code quality)
  • 3.1, 3.2: Reflection loop (self-correct test failures)
```

Steps 1.1 and 1.2 have no dependency on each other — the DAG executor runs them in parallel. The router picks the best available head for each step based on capability, VRAM, and health.

---

## 6. Consensus Voting

When one model isn't enough, ask multiple heads and vote on the answer.

```bash
multihead consensus test \
  -p "What's the correct retry strategy for this payment webhook?" \
  -s weighted \
  -h qwen-llm -h claude-sonnet -h openai-gpt4o
```

```
Individual Votes:
  qwen-llm       ✓  "Exponential backoff with jitter, max 5 retries..."    320ms
  claude-sonnet   ✓  "Exponential backoff with jitter, max 5 retries..."    890ms
  openai-gpt4o    ✓  "Fixed 30s interval, max 10 retries..."               450ms

Consensus Output: Exponential backoff with jitter, max 5 retries
  Agreement: 67% (2/3 heads)
  Strategy: weighted
  Red Flags: none
```

Two heads agreed on exponential backoff. The weighted strategy gave more influence to heads with higher confidence. Consensus is used automatically during decomposition — each plan is proposed by multiple heads and voted on before execution begins.

---

## 7. Interactive Chat and Shell

Talk to your local LLM with full knowledge store access.

```bash
multihead chat
```

```
You: What do we know about the payment webhook?
Assistant: Based on the knowledge store (3 claims):
  • Webhook retries use exponential backoff with jitter, max 5 attempts
  • Timeout was increased from 5s to 30s in commit abc123
  • Stripe is the only provider currently configured

You: Is the timeout change verified?
Assistant: Checking... the claim is marked "corroborated" — confirmed by both
  code_read (line 42 of webhook_handler.py) and git_diff (commit abc123).
```

The shell (`multihead shell`) adds process management, slash commands, and a richer REPL:

```bash
multihead shell
>> /heads                    # list available models
>> /swap qwen-vlm            # load the vision model
>> /knowledge "auth"         # search the knowledge base
>> /spawn "analyze image.png" # run a background task
>> /ps                       # check running processes
```

---

## 8. Night Shift Pipeline

While you sleep, the 26-stage pipeline harvests knowledge from everything that happened:

```bash
multihead nightshift run --head openai-gpt41-nano --batch
```

```
>> session_harvest       OK (4.1s)   — 12 sessions, 847 records
>> normalize_chunk       OK (2.3s)   — 3,291 chunks
>> entity_extraction     OK (3.3s)   — 15,791 entities
>> claim_extraction      OK (17.2s)  — 34,165 claims
>> behavioral_analysis   OK (12.8s)  — 4 repos scanned
>> ci_results            OK (1.1s)   — 89 test results ingested
>> consistency_check     OK (5.8s)   — 426 contradictions found
>> conflict_resolution   OK (26.1s)  — 347 auto-resolved, 79 need human review
>> claim_fusion          OK (10.4s)  — 639 independently verified facts
>> staleness_sweep       OK (5.7s)   — 97 outdated claims marked stale
>> publish_report        OK (0.8s)   — report written
```

**Fusion** is where independent channels (conversation, code, git, CI) are compared. If you *said* something was fixed but the code doesn't match, fusion marks it **contested**. If the code, git history, and tests all agree, it's **corroborated**. Night Shift runs nightly — your knowledge base gets more accurate over time without manual curation.

---

## 9. BotVibes Marketplace

Your agents have capabilities. Other teams need those capabilities. Post them to BotVibes and earn money.

```bash
# Register what your system can do
multihead discover              # scans your models and pipelines

# Your trained YOLO model becomes a sellable capability
multihead marketplace publish \
  --capability "object_detection" \
  --model yolo-v8-custom \
  --price 0.02                  # $0.02 per inference
```

When someone on the marketplace needs object detection, your head gets a task:

```
Incoming task from BotVibes:
  capability: object_detection
  payload: "Detect panels in comic page scan"
  budget: $0.50

Executing with yolo-v8-custom... done (1.2s)
Result posted. Revenue: $0.02
```

Your agents earn money by doing what they're already good at. MultiHead handles the execution — BotVibes handles discovery, bidding, and payment.
