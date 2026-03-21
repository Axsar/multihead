# Agentic Core: The Always-On LLM Brain

## Why It Exists

Without an agentic core LLM that's "always there," you don't have a sidekick -- you just have a job runner. The core is what makes MultiHead feel like a living assistant rather than a cron job.

## What the Core Does

- Holds the conversational thread + pinned context
- Turns "what you want" into WorkOrders
- Decides which tools to call next
- Monitors execution and reacts to failures ("retry with different settings", "switch tool", "summarize and ask user")
- Produces the final explanation in plain language

## What the Core Does NOT Do

- Run the expensive models (that's the workers)
- Do huge-context reasoning (that's RAG + context packs)
- Do high-precision computation (that's tools + checkers)

## Action Types

The core LLM must respond with one of these validated action types. If it returns anything else, the runtime rejects it and asks for a retry.

```
SAY                  -> Just talk to the user
CALL_TOOL            -> Single tool call
CREATE_WORKORDER     -> Spawn a pipeline
MONITOR_WORKORDER    -> Check status + decide intervention
PAUSE_AND_ASK        -> Needs a user decision
```

### SAY

Simple text response to the user. No side effects.

```json
{
  "action": "SAY",
  "content": "I've finished processing the images. Here's a summary of what I found..."
}
```

### CALL_TOOL

Invoke a single tool directly (not as part of a pipeline).

```json
{
  "action": "CALL_TOOL",
  "tool": "files.read",
  "params": {"path": "/projects/foo/README.md"}
}
```

### CREATE_WORKORDER

Spawn a multi-step pipeline. The core designs the plan, then the Executor takes over.

```json
{
  "action": "CREATE_WORKORDER",
  "workorder": {
    "goal": "Extract structured data from image folder",
    "steps": [
      {"tool": "llm.plan", "head_id": "qwen3-8b", "params": {}},
      {"tool": "vlm.extract", "head_id": "qwen3-vl-8b", "params": {}},
      {"tool": "llm.normalize", "head_id": "qwen3-8b", "params": {}}
    ]
  }
}
```

### MONITOR_WORKORDER

Check on a running pipeline and decide what to do.

```json
{
  "action": "MONITOR_WORKORDER",
  "run_id": "run_abc123",
  "decision": "continue | retry_step | cancel | escalate"
}
```

### PAUSE_AND_ASK

The core needs human input before proceeding.

```json
{
  "action": "PAUSE_AND_ASK",
  "question": "Step 2 produced low-confidence results (0.62). Should I retry with a larger model or accept and continue?",
  "options": ["retry_with_larger_model", "accept_and_continue", "cancel"]
}
```

## The Core Loop

```
gather context -> call core model -> validate action -> execute -> update memory -> reply
```

In detail:

1. **Gather context**: Load active context packs (from Night Shift), retrieve relevant chunks, include conversation history
2. **Call core model**: Send assembled prompt to the small local LLM
3. **Validate action**: Parse response, ensure it's a valid action type, reject if not
4. **Execute**: Dispatch the action (tool call, WorkOrder creation, etc.)
5. **Update memory**: Store the interaction as a record for Night Shift processing
6. **Reply**: Return result to user (or loop back if more work needed)

## How the Core Steps Aside During Batch

When a WorkOrder is launched in batch mode:

1. Core generates the complete plan (all steps)
2. Core hands off to the Executor
3. Core enters low-VRAM mode:
   - **Option A**: Swap to tiny CPU-quant model (stays responsive but slow)
   - **Option B**: Unload completely (frees all GPU VRAM)
4. Executor runs all steps sequentially, managing head swaps
5. When the WorkOrder completes (or hits an error requiring judgment):
   - Core wakes back up
   - Reviews results via `MONITOR_WORKORDER`
   - Summarizes for the user

## Model Profile for the Core

The core should ideally be a small, fast model:

```yaml
core:
  provider: ollama
  model: qwen3:4b          # Small enough for CPU
  mode: cpu                 # Always available, even during GPU batch work
  keep_alive: -1            # Never auto-unload
  max_context_tokens: 8192  # Modest context window
```

Recommendation: Run the core model fully on CPU by default so it's always available even when GPU is busy with worker models.

## What Makes It "Agentic" (Not Just a Chatbot)

1. **Structured output**: Every response is a validated action, not free-form text
2. **Tool access**: Can invoke tools, spawn pipelines, monitor jobs
3. **Memory**: Loads context packs built by Night Shift, remembers across sessions
4. **Reactive**: Monitors running WorkOrders and intervenes on failures
5. **Bounded**: Action types are finite and validated -- prevents runaway behavior

