# Capability Discovery - Gap #3 Fix

**Status**: ✅ Implemented (2026-02-22)

## What This Fixes

Before: Router used static `config/solvers.yaml`, no dynamic "who can handle X?" queries.

After: Router can dynamically discover capabilities from:
1. Local registry (acp_state.json)
2. Knowledge base (model performance data)
3. Optional: BotVibes marketplace (future)

## Quick Start

```python
from multihead.capability_discovery import discover_capabilities

# Query by capability ID (prefix matching)
matches = discover_capabilities("com.multihead.llm")
# Returns: [CapabilityMatch(agent_id="multihead-agent", capability_id="com.multihead.llm.qwen-llm", ...)]

# Query by semantic search
matches = discover_capabilities("YOLO detection")
# Returns: [CapabilityMatch(model="YOLOv8m", performance={"mAP50": 0.9185}, ...)]

# Query Claude capabilities
matches = discover_capabilities("com.claude.code")
# Returns: [CapabilityMatch(agent_id="claude-session-agent", capability_id="com.claude.code.edit", ...)]
```

## Architecture

### CapabilityMatch

Results returned as `CapabilityMatch` objects:

```python
@dataclass
class CapabilityMatch:
    agent_id: str              # "multihead-agent" | "claude-session-agent"
    capability_id: str         # "com.multihead.llm.qwen-llm"
    source: str                # "local" | "knowledge" | "botvibes"

    kind: str                  # "llm" | "vlm" | "tool" | "model"
    model: str                 # "Qwen/Qwen3-8B" | "YOLOv8m"

    performance: dict | None   # {"mAP50": 0.9185, "iou": 0.9958}
    latency_p50_ms: int
    latency_p95_ms: int
    cost_per_call: float

    available: bool
    match_score: float         # 0.0-1.0 (confidence in match)
```

### CapabilityDiscovery

Main class for querying capabilities:

```python
from multihead.capability_discovery import CapabilityDiscovery

discovery = CapabilityDiscovery(data_dir="~/.multihead")

# Method 1: Query by capability ID or semantic search
matches = discovery.query("com.multihead.llm", exact_match=False, limit=20)

# Method 2: Query by kind
matches = discovery.query_by_kind("vlm", limit=20)

# Reload capability data (auto-reloads every 60s)
discovery.reload()
```

## Data Sources

### 1. Local Registry (acp_state.json)

Location: `$MULTIHEAD_DATA_DIR/acp_state.json`

Structure:
```json
{
  "agent_id": "multihead-agent",
  "capabilities": {
    "capabilities": [
      "com.multihead.llm.qwen-llm",
      "com.multihead.vlm.qwen-vlm"
    ],
    "latency_profile": {"p50_ms": 2000, "p95_ms": 10000},
    "cost_model": {"unit": "task", "price": 0.0}
  },
  "claude_capabilities": {
    "capabilities": [
      "com.claude.code.edit",
      "com.claude.code.test"
    ]
  }
}
```

### 2. Knowledge Base (knowledge.db)

Location: `$MULTIHEAD_DATA_DIR/knowledge.db`

Queries `claims` table for:
- Model performance data (YOLO, SAM2, UNet)
- Capability descriptions
- Semantic matches

Example claims:
```sql
SELECT * FROM claims
WHERE statement LIKE '%YOLO%'
  AND claim_status = 'accepted';
-- "YOLOv8m achieving 91.85% mAP50 for object detection"
```

### 3. BotVibes Marketplace (Future)

Will query BotVibes API for external capabilities:
- `GET /api/v1/marketplace/listings/search?capability_id=X`
- Returns: agent_id, pricing, SLA, quality_score

## Integration with Router

Router can now use capability discovery instead of static config:

```python
from multihead.router import Router
from multihead.capability_discovery import CapabilityDiscovery

class DynamicRouter(Router):
    def __init__(self, head_manager, metrics, resource_monitor, discovery):
        super().__init__(head_manager, metrics, resource_monitor)
        self.discovery = discovery

    def route(self, required_kind: str, **kwargs):
        # Discover available capabilities
        matches = self.discovery.query_by_kind(required_kind)

        # Score candidates using Router's existing logic
        scored = [(m.agent_id, self._score(m.agent_id)) for m in matches]
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[0][0] if scored else None
```

## Query Examples

### Example 1: Find LLM Capabilities

```python
from multihead.capability_discovery import discover_capabilities

matches = discover_capabilities("com.multihead.llm")
print(f"Found {len(matches)} LLM capabilities:")
for m in matches:
    print(f"  - {m.capability_id} | {m.model} | latency_p50={m.latency_p50_ms}ms")
```

Output:
```
Found 4 LLM capabilities:
  - com.multihead.llm.qwen-llm | Qwen/Qwen3-8B | latency_p50=2000ms
  - com.multihead.llm.claude-sonnet | sonnet | latency_p50=2000ms
  - com.multihead.llm.openai-gpt4o | gpt-4o-mini | latency_p50=2000ms
  - com.multihead.llm.mock-llm | mock-llm-v1 | latency_p50=2000ms
```

### Example 2: Find Detection Models

```python
matches = discover_capabilities("detection")
kb_matches = [m for m in matches if m.source == "knowledge"]

for m in kb_matches:
    print(f"{m.model}: {m.performance}")
```

Output:
```
YOLOv8m: {'mAP50': 0.9185}
```

### Example 3: Find Segmentation Models

```python
matches = discover_capabilities("segmentation")
kb_matches = [m for m in matches if m.source == "knowledge"]

for m in kb_matches:
    print(f"{m.model}: IoU={m.performance.get('iou', 'N/A')}, latency={m.latency_p50_ms}ms")
```

Output:
```
UNet: IoU=0.9958, latency=900ms
SAM2: IoU=N/A, latency=10500ms
```

### Example 4: Compare Capabilities

```python
from multihead.capability_discovery import CapabilityDiscovery

discovery = CapabilityDiscovery("~/.multihead")

# Find all segmentation models
matches = discovery.query("segmentation")
models = [m for m in matches if m.kind == "model"]

# Sort by latency (fastest first)
models.sort(key=lambda m: m.latency_p50_ms)

print("Segmentation models ranked by speed:")
for m in models:
    print(f"  {m.model}: {m.latency_p50_ms}ms")
```

Output:
```
Segmentation models ranked by speed:
  UNet: 900ms
  SAM2: 10500ms
```

## Testing

Run tests:
```bash
python -m pytest tests/test_capability_discovery.py -v
```

All 22 tests pass:
- Local registry queries (exact, prefix, substring)
- Claude capability queries
- Query by kind (llm, vlm)
- Knowledge base queries (YOLO, SAM2, UNet)
- Semantic search
- Match scoring and limits
- Graceful fallbacks

## Next Steps

1. ✅ **Gap #3: Capability Discovery** (DONE)
2. ⏳ **Gap #4: Router Dynamic Lookup** - Integrate discovery with Router
3. ⏳ **Gap #1: ClaudeSessionAdapter Wiring** - Add knowledge_store param to HeadManager
4. ⏳ **Gap #2: Session Poller Callsite** - Wire poller into decomposer

## Related Files

- `src/multihead/capability_discovery.py` - Main implementation
- `tests/test_capability_discovery.py` - 22 tests
- `$MULTIHEAD_DATA_DIR/acp_state.json` - Local capability registry
- `$MULTIHEAD_DATA_DIR/knowledge.db` - Model performance data
