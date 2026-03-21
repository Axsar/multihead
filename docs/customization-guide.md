# How to Customize MultiHead

MultiHead is a framework, not a product. This guide shows how to make it yours.

---

## Quick Overview

MultiHead has two layers:
1. **Framework** — orchestration, knowledge store, adapters, CLI, shell (you keep these)
2. **Reference implementation** — our extraction prompts, fusion tuning, pipeline stages (you customize these)

You don't fork the framework. You customize the implementation layer for your domain.

---

## 1. Add Your Own Heads

Heads are pluggable intelligence sources. Add any LLM, tool, or service.

### Via config (no code)

Edit `config/heads.yaml`:

```yaml
my-local-llm:
  name: "My Fine-tuned Model"
  adapter: ollama
  model: "my-model:latest"
  kind: llm
  gpu_required: true
  vram_hint_mb: 8000

my-api-model:
  name: "GPT-4o via OpenAI"
  adapter: openai
  model_name: "gpt-4o"
  kind: llm
  gpu_required: false
  extra:
    api_key_env: "OPENAI_API_KEY"
```

Available adapters: `transformers`, `ollama`, `vllm`, `openai`, `anthropic`, `claude_agent_sdk`, `mock`, `deterministic`

### Via code (custom adapter)

```python
# src/multihead/adapters/my_adapter.py
from multihead.adapters.base import HeadAdapter

class MyAdapter(HeadAdapter):
    async def load(self):
        # Connect to your model/service
        pass

    async def generate(self, prompt: str) -> dict:
        # Call your model
        return {"text": response, "tokens_in": n, "tokens_out": m}

    async def unload(self):
        pass
```

Register in `heads.yaml`:
```yaml
my-head:
  adapter: my_adapter
  model: whatever
```

---

## 2. Write Your Own Extraction Prompts

The claim extractor prompt determines what knowledge gets extracted from conversations. Ours is tuned for software engineering — yours should match your domain.

### Where it lives

`src/multihead/extractors/claim_extractor.py` — the `PROMPT_TEMPLATE` variable.

### How to customize

Replace the prompt with your domain's knowledge types:

```python
PROMPT_TEMPLATE = """Extract knowledge from this text about [YOUR DOMAIN].

DO extract:
- [Your knowledge type 1]: "description"
- [Your knowledge type 2]: "description"
- [Your knowledge type 3]: "description"

DO NOT extract:
- [Noise type 1]
- [Noise type 2]

Each claim must be a complete, self-contained sentence of at least 50 characters.

Output ONLY a JSON array:
[{{"claim_type":"<TYPE>","claim_key":"<KEY>","statement":"<SENTENCE>","confidence":0.8}}]

Text:
{text}

JSON:"""
```

### Tips
- Be specific about what "durable knowledge" means in your domain
- List explicit DO NOT extract items — reduces noise dramatically
- Set confidence rules (direct observation > inference > hearsay)
- Test with 10-20 sample texts before running on your full corpus

---

## 3. Customize the Fusion Policy

Fusion determines how claims from different sources are compared and verified.

### Where it lives

`src/multihead/claim_fusion.py` — the `ClaimFusion` class.

### Key things to customize

**Channel reliability** — which observation methods are most trustworthy in YOUR domain:

```python
# In _score_convergence():
# Our defaults (software engineering):
#   code_read: 0.95, git_diff: 0.90, conversation: 0.50-0.80
#
# For medical domain you might use:
#   clinical_trial: 0.95, peer_review: 0.90, patient_report: 0.60
#
# For legal domain:
#   statute_text: 0.95, case_law: 0.85, legal_opinion: 0.70
```

**Convergence thresholds** — when to corroborate vs contest:

```python
# _fuse_topic():
if avg_convergence > 0.7:    # Adjust: higher = stricter corroboration
    action = "boost"
elif avg_convergence < -0.3:  # Adjust: lower = stricter contradiction
    action = "flag_contradiction"
```

**Sub-categories** — what types of disagreement matter in your domain:

```python
# _classify_divergence():
# We classify: TEMPORAL_DRIFT, CONFIG_DRIFT, IMPROVEMENT_UNVERIFIED
# You might add: JURISDICTION_CONFLICT, VERSION_MISMATCH, etc.
```

---

## 4. Customize the Night Shift Pipeline

The pipeline is a sequence of stages. Add, remove, or reorder them.

### Where stages are defined

`src/multihead/night_shift/models.py` — the `STAGES` list:

```python
STAGES = [
    StageDefinition(0, "session_harvest", ...),
    StageDefinition(1, "select_input_window", ...),
    # ... add your own stages here
    StageDefinition(N, "my_custom_stage", StageGate(on_fail="continue")),
]
```

### How to add a stage

1. Define the stage in `models.py`
2. Add a handler method in `stages_early.py`, `stages_advanced.py`, or `stages_late.py`:

```python
async def _stage_my_custom_stage(self, context: dict) -> dict:
    """My custom extraction/analysis stage."""
    # Query live DB
    with self.knowledge._connect() as conn:
        rows = conn.execute("SELECT ...").fetchall()

    # Process
    results = do_your_thing(rows)

    # Return metrics (shown in stage summary)
    return {"metrics": {"items_processed": len(results), "key_finding": "value"}}
```

3. The stage runner handles checkpointing, error recovery, and dependency management automatically.

### Stage dependencies

Use `depends_on` to control execution order:

```python
StageDefinition(10, "my_analysis", ..., depends_on=[7]),  # runs after claim_extraction
StageDefinition(11, "my_report", ..., depends_on=[10]),    # runs after my_analysis
```

Stages with no dependency on pending batch stages run in parallel.

---

## 5. Customize Router Scoring

The router decides which head handles each task.

### Where it lives

`src/multihead/router/_scoring.py`

### What to change

Adjust weights for your use case:

```python
# Default weights (software engineering focus):
#   active_state: 40  — prefer already-loaded head
#   circuit_breaker: 30  — avoid failing heads
#   vram_fit: 15  — must fit in available VRAM
#   error_rate: 10  — historical success
#   latency: 5  — speed preference

# For a cost-sensitive setup:
#   cost: 40, capability: 30, latency: 20, error_rate: 10

# For a quality-focused setup:
#   capability: 50, error_rate: 30, latency: 10, cost: 10
```

---

## 6. Add Your Own Observation Channels

MultiHead fuses knowledge from independent channels. Add channels for your domain.

### Built-in channels
- `code_read` — AST analysis of source files
- `code_behavior_llm` — LLM behavioral analysis
- `git_diff` — git commit diffs
- `git_history` — commit messages
- `ci_test` — CI/CD results
- `user_statement` / `assistant_statement` — conversation
- `agent_deposit` — explicit agent deposits

### Adding a custom channel

Write an extractor that produces claims with your `observation_method`:

```python
# src/multihead/extractors/my_channel.py
async def extract_from_my_source(source_path, adapter):
    claims = []
    for item in read_my_source(source_path):
        claims.append({
            "claim_key": f"my_domain.{item.topic}",
            "statement": item.text,
            "confidence": 0.85,
            "observation_method": "my_custom_channel",  # Your channel name
            "file_path": item.source_file,
        })
    return claims
```

Then add a nightshift stage that calls it. Fusion will automatically compare your channel's claims against other channels for the same files.

---

## 7. Customize the Knowledge Store Schema

The claims schema supports extension via JSON fields.

### Add custom metadata

Use `provenance_json` to store domain-specific metadata:

```python
provenance = Provenance(
    produced_by={"kind": "agent", "id": "my-agent"},
    observation_method="my_channel",
    source_anchor={
        "file_path": "path/to/file",
        "custom_field": "your domain data",
    },
    evidence=[{"type": "my_evidence", "text": "supporting quote"}],
)
```

### Query custom fields

```sql
SELECT claim_key, statement FROM claims
WHERE json_extract(provenance_json, '$.source_anchor.custom_field') = 'value';
```

---

## 8. Customize the Shell

Add slash commands for your domain:

```python
# src/multihead/slash_commands/_my_commands.py
from . import register_command

@register_command("/my-command")
async def my_command(args, context):
    """Description of what this does."""
    # Your logic here
    return "Output to display"
```

---

## Summary: What to Customize vs What to Keep

| Keep as-is (framework) | Customize (implementation) |
|------------------------|---------------------------|
| Orchestrator / DAG executor | Extraction prompts |
| Knowledge store schema | Fusion policy / thresholds |
| Consensus engine | Channel reliability hierarchy |
| Head manager / adapters | Router weights |
| CLI / Shell / API / MCP | Pipeline stages |
| Night Shift stage runner | Domain-specific channels |
| Batch recovery / checkpoint | Config (heads, solvers, recipes) |
