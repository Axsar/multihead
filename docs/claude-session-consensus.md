# Multi-Session Claude Consensus

**Status**: ✅ Implemented (v0.4)
**Feature**: Distributed consensus voting using multiple Claude Code sessions via knowledge base

## Overview

MultiHead Solve can now use **other Claude Code sessions** as "heads" for consensus voting! This enables:

- **Hybrid consensus**: Combine external LLMs (GPT, Qwen) with local Claude sessions
- **Knowledge-aware voting**: Local Claude sessions have access to the knowledge base
- **Collaborative decomposition**: Multiple users/sessions can contribute to task planning
- **Async responses**: Sessions respond when available, even if late

## Architecture

```
Session 1 (Main)                Knowledge Base (.db)        Session 2 (Voter)
     ↓                                ↓                           ↓
  Post Request  ─────────►  ClaimType.QUESTION         ◄────  Poll for requests
  (DECOMP_REQUEST)              (unanswered)                      ↓
       ↓                                                    Decompose task
       │                                                          ↓
  Wait for responses  ◄───  ClaimType.PLAN           ◄─────  Submit proposal
  (5min timeout)         (related_claim_ids=[req])        (DECOMP_PROPOSAL)
       ↓
  Collect proposals
       ↓
  Run Consensus Vote
       ↓
  Execute winning plan
```

## Protocol

### Request Format (ClaimType.QUESTION)

```python
Claim(
    claim_type=ClaimType.QUESTION,
    claim_status=ClaimStatus.PROPOSED,  # Unanswered
    statement="DECOMP_REQUEST: Create a Python add function",
    canonical=ClaimCanonical(
        claim_key="decomp.request.req_123",
        subject=EntityRef(entity_type="decomposition_request", ...),
        predicate="needs_decomposition",
        object=ValueObject(value="Create...")
    ),
    related_claim_ids=[],
    provenance=Provenance(produced_by={"kind": "session", "id": "session-1"})
)
```

### Response Format (ClaimType.PLAN)

```python
Claim(
    claim_type=ClaimType.PLAN,
    claim_status=ClaimStatus.PROPOSED,
    statement="DECOMP_PROPOSAL: {\"phases\": [...], \"steps\": [...]}",
    related_claim_ids=["req_123"],  # Links to request!
    provenance=Provenance(produced_by={"kind": "session", "id": "session-2"})
)
```

## Configuration

### 1. Add Claude Session Heads to `config/heads.yaml`:

```yaml
# Claude Session Heads (for multi-session consensus)
claude-session-1:
  name: "Claude Session 1"
  adapter: "claude_session"
  model: "claude-session-voter"
  kind: "llm"
  config:
    session_id: "claude-session-1"
    timeout_seconds: 300  # 5 minutes (0 = no timeout)
    min_responses: 1
    poll_interval: 2.0

claude-session-2:
  name: "Claude Session 2"
  adapter: "claude_session"
  model: "claude-session-voter"
  kind: "llm"
  config:
    session_id: "claude-session-2"
    timeout_seconds: 300
    min_responses: 1
```

### 2. Run MultiHead Solve with Session Heads:

```bash
multihead solve "Create a calculator" \
  --heads qwen-llm \
  --heads claude-session-1 \
  --heads claude-session-2 \
  --strategy first_to_ahead
```

## Workflow

### Main Session (Posts Request):

1. Runs `multihead solve "task"`
2. ClaudeSessionAdapter posts DECOMP_REQUEST to knowledge base
3. Waits for responses (5min timeout, configurable)
4. Collects PLAN responses
5. Runs consensus vote
6. Executes winning plan

### Voter Sessions (Respond):

1. On every user message, poll knowledge base for requests
2. Find unanswered DECOMP_REQUEST claims
3. **Prompt user**: "Session X needs help decomposing: {task}. Participate? (y/n)"
4. If yes:
   - Decompose the task
   - Submit DECOMP_PROPOSAL claim with `related_claim_ids=[request_id]`
5. Main session picks up response and votes

## Polling Implementation

```python
# Called on every user message in Claude sessions
from multihead.session_poller import check_for_decomposition_requests

requests = check_for_decomposition_requests(
    knowledge_store=ks,
    project_id="multihead",
    session_id="my-session-id"  # Don't respond to own requests
)

if requests:
    # Prompt user to participate
    for request in requests:
        task = get_request_task(request)
        if user_confirms(f"Help decompose: {task}?"):
            # Decompose and submit
            plan = await decompose_task(task)
            submit_decomposition_proposal(ks, request.claim_id, plan)
```

## Benefits

✅ **Knowledge-aware**: Local sessions know what worked/failed before
✅ **Collaborative**: Multiple users can contribute
✅ **Async**: Responses don't block, even late ones are captured
✅ **Resilient**: System works with 0, 1, or N responses
✅ **Hybrid**: Mix external LLMs + local agents

## Timeout Behavior

**timeout_seconds = 300 (5 min, default)**:
- Waits up to 5 minutes for responses
- If min_responses met, stops early
- If no responses, proceeds without (system must work!)

**timeout_seconds = 0 (no timeout)**:
- Checks once and returns immediately
- Use when responses are optional

**Late responses**:
- Still captured in knowledge base
- Available for future adjustments
- Can inform post-execution reviews

## Future Improvements (Option B)

Currently using **prefix-based protocol** (`DECOMP_REQUEST:` in statement).

**Future**: Add `category` field to Claim model:
```python
Claim(
    category="decomposition_request",  # Explicit categorization
    ...
)
```

Benefits:
- More robust filtering
- No prefix parsing
- Better queries

## Testing

### Test End-to-End:

**Terminal 1 (Main Session)**:
```bash
multihead solve "Create add function" --heads claude-session-1
```

**Terminal 2 (Voter Session)**:
```bash
# Just type any message to trigger polling
# Will see prompt: "Help decompose: Create add function?"
# Respond with your proposal
```

Main session will collect response and vote!

## Implementation Files

- `src/multihead/adapters/claude_session.py` - ClaudeSessionAdapter
- `src/multihead/session_poller.py` - Polling and response helpers
- `docs/claude-session-consensus.md` - This documentation
