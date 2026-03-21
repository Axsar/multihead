# MultiHead Examples

This directory contains example scripts showing how to use MultiHead in your own projects.

## Prerequisites

1. **Start the MultiHead daemon**:
   ```bash
   source .venv/bin/activate
   multihead serve
   ```

   Leave this running in a terminal.

2. **Activate the virtual environment** (in a new terminal):
   ```bash
   source .venv/bin/activate
   ```

## Examples

### basic_usage.py

A simple script demonstrating core MultiHead features:
- Connecting to MultiHead
- Chatting with the local LLM
- Writing to the knowledge store
- Reading from the knowledge store
- Checking model head status

**Run it**:
```bash
python examples/basic_usage.py
```

**Expected output**:
```
🔌 Connecting to MultiHead...
✅ Connected!

============================================================
Example 1: Chat with the local LLM
============================================================
Question: What is the capital of France?
Answer: The capital of France is Paris.

============================================================
Example 2: Write to the knowledge store
============================================================
✅ Deposited claim: 'Successfully ran basic_usage.py example'
...
```

## Using MultiHead in Your Projects

### 1. Install MultiHead as a dependency

Add to your `requirements.txt` or `pyproject.toml`:
```
multihead @ git+https://github.com/Axsar/multihead.git
```

Or install locally:
```bash
pip install /path/to/multihead
```

### 2. Start the daemon

Your application needs the MultiHead daemon running:
```bash
multihead serve
```

You can check if it's running:
```python
from multihead.client import MultiHeadClient

mh = MultiHeadClient()
if not mh.ping():
    print("MultiHead daemon not running!")
```

### 3. Use the client

```python
from multihead.client import MultiHeadClient

# Connect
mh = MultiHeadClient()  # defaults to localhost:7337

# Chat
response = mh.chat("Hello!")

# Write knowledge
mh.deposit_claim(
    claim_key="myapp.status",
    statement="Application started successfully",
    produced_by="myapp",
)

# Read knowledge
claims = mh.query_claims(scope_id="myapp")
```

## Common Patterns

### Pattern 1: Application Startup Briefing

Get relevant knowledge before starting:
```python
from multihead.client import MultiHeadClient

mh = MultiHeadClient()
briefing = mh.get_briefing("myapp")

# briefing contains:
# - claims: direct matches (claim_key contains "myapp")
# - related_claims: claims mentioning "myapp" in statement
# - recent_events: events tagged with or mentioning "myapp"

print(f"Found {len(briefing['claims'])} relevant facts")
```

### Pattern 2: Recording Task Progress

Track what your application does:
```python
from multihead.client import MultiHeadClient

mh = MultiHeadClient()

# Before task
mh.report_event(
    title="Started data processing",
    event_type="task_created",
    produced_by="data_processor",
)

# After success
mh.deposit_claim(
    claim_key="data_processor.last_run",
    statement="Processed 1000 records successfully",
    produced_by="data_processor",
    confidence=1.0,
)

# After completion
mh.report_event(
    title="Data processing completed",
    event_type="task_completed",
    produced_by="data_processor",
    metrics={"records": 1000, "duration_s": 45.2},
)
```

### Pattern 3: Multi-Agent Coordination

Delegate work to other agents:
```python
from multihead.client import MultiHeadClient

mh = MultiHeadClient()

# Ask the local LLM for help
task_id = mh.create_task(
    capability="com.multihead.llm",
    payload_ref="Analyze this log file and suggest fixes",
    target_agent_id="multihead-agent",
)

# Or delegate to Claude worker daemon
task_id = mh.delegate_claude(
    prompt="Review the code in src/ and fix any bugs",
    conversation_id="my-review-session",  # thread related tasks
)
```

## API Documentation

See [docs/08-api-surface.md](../docs/08-api-surface.md) for the full REST API reference.

## More Examples

Coming soon:
- `pipeline_example.py` — Running multi-step YAML pipelines
- `knowledge_integration.py` — Deep knowledge store patterns
- `agent_mesh_example.py` — Multi-agent collaboration

## Need Help?

- **Main docs**: [README.md](../README.md)
- **Getting started**: [docs/12-getting-started.md](../docs/12-getting-started.md)
- **Issues**: https://github.com/Axsar/multihead/issues
