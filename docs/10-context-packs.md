# Context Packs

## Overview

Context Packs are the mechanism for delivering high-signal, curated context to the core LLM. Instead of forcing the LLM to ingest raw history, the Night Shift builds focused packs that the core loads on startup.

## Pack Format

Each pack consists of two files:

### `<pack_name>.md`

A readable stitched summary with citations to artifacts (paths + event IDs). This is what the core LLM actually reads.

### `<pack_name>.pack.json`

Machine-readable metadata:

```json
{
  "pack_id": "pack_active_projects_2026-02-11",
  "purpose": "Current active projects with latest status",
  "budgets": {
    "max_tokens": 4000,
    "max_items": 50
  },
  "items": [
    {
      "type": "claim",
      "text": "The core LLM runs on CPU by default so it stays available while the GPU is busy.",
      "priority": 0.85,
      "evidence_refs": ["evt_01J...", "clm_01J..."],
      "token_estimate": 42,
      "why_included": "accepted decision, high importance, recent"
    },
    {
      "type": "event",
      "text": "Decided to use JSON Schema 2020-12 for all schema definitions.",
      "priority": 0.78,
      "evidence_refs": ["evt_02J..."],
      "token_estimate": 28,
      "why_included": "confirmed event, architecture decision"
    }
  ],
  "metrics": {
    "token_total": 3842,
    "item_count": 47,
    "dropped": [
      {
        "item": "clm_old...",
        "reason": "over_budget"
      },
      {
        "item": "clm_dup...",
        "reason": "duplicate"
      },
      {
        "item": "clm_sup...",
        "reason": "superseded"
      }
    ]
  }
}
```

### Item Fields

| Field | Description |
|-------|-------------|
| `type` | `claim` / `event` / `snippet` / `artifact` |
| `text` | The content string (or path for artifacts) |
| `priority` | Computed score for ranking |
| `evidence_refs[]` | Links to Event/Claim IDs with backing evidence |
| `token_estimate` | Approximate token count for budgeting |
| `why_included` | Human-readable reason for inclusion |

### Drop Reasons

| Reason | Description |
|--------|-------------|
| `over_budget` | Token budget exceeded |
| `low_score` | Priority score below threshold |
| `duplicate` | Near-duplicate already included |
| `superseded` | Claim/event has been superseded by newer version |

## Core Packs (Ship in v0.1)

### Default Pack

Pinned instructions + latest accepted claims + last 3 run summaries. Always loaded.

### Project Pack

Everything tied to a `project_id` (e.g., BotVibes, comics pipeline). Loaded when working on that project.

### Run Pack

Context needed to continue a specific run. Loaded when resuming an interrupted pipeline.

## Night Shift Built Packs

The Night Shift (Stage 9) builds these packs nightly:

| Pack | Content |
|------|---------|
| `packs/active_projects.md` | Current project status, recent activity |
| `packs/recent_decisions.md` | Decisions made in last 7 days |
| `packs/constraints.md` | Active constraints and rules |
| `packs/glossary.md` | Definitions and terminology |
| `packs/open_loops.md` | Unresolved questions, TODOs, pending decisions |

### Morning Load Order

Core loads on startup:

1. Active Projects Pack
2. Open Loops Pack
3. Recent Decisions Pack
4. Glossary Pack (stable, changes slowly)

And can fetch deeper evidence on demand via retrieval ("show me the record span").

## packs.build Tool Spec

### Input

```json
{
  "query": "continue the dataset pipeline",
  "filters": {
    "project_id": "acp-runtime",
    "time_range": "7d",
    "types": ["claim", "event"]
  },
  "budgets": {
    "max_tokens": 4000,
    "max_items": 50
  },
  "ranking": {
    "weights": {
      "recency": 0.3,
      "frequency": 0.1,
      "trust": 0.3,
      "relevance": 0.3
    }
  }
}
```

### Output

- Writes `.md` + `.pack.json` files
- Returns `pack_id` + metrics

### Ranking Formula (v0.1)

```
score = w_r * relevance
      + w_t * trust
      + w_c * recency
      + w_f * frequency
      - penalties(duplicate, superseded)
```

Where:

| Weight | Description |
|--------|-------------|
| `w_r` | Relevance to current query/context |
| `w_t` | Trust level (`accepted` > `proposed` > `contested`) |
| `w_c` | Recency (exponential decay from `updated_at`) |
| `w_f` | Frequency of reference in recent records |

### Token Budget Trimming

When a pack exceeds its token budget, trim by priority:

1. Keep `accepted` claims over `contested` over `proposed`
2. Keep higher `importance` scores
3. Keep more recent `updated_at`
4. Keep "open loops" always (they represent active work)
5. Drop lowest-priority items first

### Grounding Requirement

Pack entries must reference Claim/Event IDs, and those must have evidence pointers. No un-grounded statements in canon packs.

## Pack Lifecycle

1. **Night Shift builds packs** from the day's processed Claims/Events
2. **Core loads packs** on morning startup (or session start)
3. **Packs are rebuilt nightly** -- they are derived views, not primary data
4. **Deletion propagates**: If a Claim or Event is deleted/retracted, the next Night Shift run drops it from packs
5. **User can force rebuild**: `acp packs rebuild` triggers immediate pack generation

## Example: Active Projects Pack

```markdown
# Active Projects (2026-02-11)

## ACP Runtime (MultiHead)
- **Status**: v0.1 in active development
- **Last activity**: Today
- **Key decision**: Core LLM runs on CPU by default [clm_01J...]
- **Open**: Define exact WorkOrder schema fields [evt_05J...]
- **Open**: Choose between Ollama-only vs vLLM sleep for v0.1 [evt_06J...]

## BotVibes
- **Status**: Schema design phase
- **Last activity**: 2 days ago
- **Key decision**: LMP maps 1:1 to BotVibes concepts [clm_03J...]
- **Constraint**: Runtime never requires BotVibes [clm_04J...]
```

