# Night Shift: Off-Peak Context Building

## Why Night Shift Exists

Your always-on core LLM should not be forced to ingest your entire life/work history every time you talk. That's slow, expensive, and messy.

Instead:

- **Daytime core** = conversational + planner + tool router (fast, local, small model OK)
- **Night Shift** = offline "memory refinery" that turns raw logs into canon + packs + indexes

The win condition: tomorrow morning your core LLM starts with a high signal "starter brain" covering current projects, recent decisions + constraints, definitions/glossary, open loops, and top context connections -- and can retrieve deeper evidence on-demand.

## The Evidence + Claims Model

### Never Let Summaries Become the Source of Truth

Summaries drift. The fix:

- **Evidence** = immutable records (chat logs, files, commits, PDFs, outputs)
- **Claims** = extracted statements with pointers back to evidence
- **Summaries** = views over claims + evidence, never replacing them

### Four Data Primitives

1. **Record** (raw): chat transcript chunk, tool execution log, file snapshot/diff, web clipping, note you wrote
2. **Event** (what happened): "ran pipeline X and got artifact Y", "decided to use schema v1"
3. **Claim** (what's true, with evidence): statement + supports[] + confidence + status
4. **Link** (why two things relate): from entity -> to entity + reason + evidence

## Night Shift Pipeline Stages

The Night Shift runs as a WorkOrder with 18 stages:

### Stage 0 -- Session Harvest

- **Goal**: Harvest knowledge from all Claude Code session memory files (~/.claude/projects/). Track evolution between harvests.
- **Requires LLM**: No
- **on_fail**: continue

### Stage 1 -- Select Input Window

- **Goal**: Define what "today" means and collect new records since last run
- **Inputs**: `state.last_nightshift_at`, record store
- **Outputs**: `input_manifest.json` (record_ids, uris, hashes)
- **Gate**: if `new_record_count == 0`, SKIP everything except maintenance/index vacuum

### Stage 2 -- Normalize + Chunk Records

- **Goal**: Ensure every record is parseable, chunked, and span-addressable
- **Outputs**: `record_index.jsonl`, `chunks.jsonl`, `ingest_report.json`
- **Gate**: `parse_success_rate >= 0.995` else RETRY; `chunk_coverage >= 0.98` else RETRY; if still low -> ABORT

### Stage 3 -- Fast "Hot" Signals

- **Goal**: Compute recency/frequency signals cheaply (no LLM)
- **Outputs**: `hot_signals.json` (top entities/files/tools by frequency + recency)
- **Gate**: None (never blocks pipeline)

### Stage 4 -- Entity Extraction + Canonicalization

- **Goal**: Extract entities and map to stable IDs (dedupe "ACP runtime" vs "acp-runtime")
- **Outputs**: `entities.jsonl`, `entity_aliases.json`, `entity_link_map.json`
- **Gate**: `entity_yield_per_1k_tokens >= MIN_YIELD` else RETRY with larger model; `alias_conflict_rate <= 0.02` else FALLBACK to strict mode

### Stage 5 -- Topic Assignment (Clusters)

- **Goal**: Cluster chunks into topics + assign topic_ids
- **Outputs**: `topics.json`, `topic_assignments.jsonl`
- **Gate**: `unassigned_chunk_rate <= 0.03` else RETRY; `topic_coherence >= 0.35` else FALLBACK

### Stage 6 -- Event Extraction (Grounded)

- **Goal**: Create Event objects with evidence pointers
- **Outputs**: `events.jsonl`
- **Gate**: `event_extract_coverage >= 0.80` else RETRY; if still low -> CONTINUE with all events as `draft`

### Stage 7 -- Claim Extraction

- **Goal**: Create Claim objects keyed by `claim_key`, never auto-accept without evidence
- **Outputs**: `claims.jsonl`
- **Auto-accept rule**: Only auto-accept if ALL true:
  - `claim_type` in (definition, decision, constraint)
  - `confidence >= 0.85`
  - `supports >= 2` from distinct records OR one record + one tool trace

### Stage 8 -- Consistency + Contradiction Resolution

- **Goal**: Enforce "one accepted per (scope, claim_key)" and mark conflicts
- **Outputs**: `conflicts.jsonl`, updated claim statuses
- **Gate**: If two accepted exist for same key+scope -> REPAIR transactionally; if unresolvable -> set both `contested` and QUEUE_REVIEW

### Stage 9 -- Open Loops + Task Ledger

- **Goal**: Extract unresolved questions, TODOs, pending decisions
- **Outputs**: `open_loops.json`, `tasks.json`
- **Gate**: Items must include evidence pointers; otherwise stay "untrusted"

### Stage 10 -- Canon Pack Build

- **Goal**: Build stable "packs" the core LLM loads tomorrow
- **Outputs**: `packs/active_projects.md`, `packs/recent_decisions.md`, `packs/constraints.md`, `packs/glossary.md`, `packs/open_loops.md`
- **Gate**: Pack entries must reference Claim/Event IDs; token budget gate trims by priority

### Stage 11 -- Daily Brief (Grounded Narrative)

- **Goal**: Produce a daily narrative that cites Events/Claims
- **Outputs**: `briefs/daily_YYYY-MM-DD.md`
- **Gate**: No statement may appear unless it references an Event/Claim ID with evidence; if violations -> RETRY with "cite-or-drop"

### Stage 12 -- Weekly Rollup (Conditional)

- **Goal**: Weekly trend summary
- **Runs only if**: end of week or `days_since_last_weekly >= 7`
- **Outputs**: `briefs/weekly_YYYY-Www.md`
- **Gate**: Requires at least 4 daily briefs or sufficient Events; else SKIP

### Stage 13 -- Cross-Topic Links

- **Goal**: Create meaningful links (shared entities/constraints/repeated blockers)
- **Outputs**: `links.jsonl`
- **Gate**: Each link must have `reason_type`, `score`, and `evidence >= 1`; if `score < threshold` -> mark `link_status=draft`

### Stage 14 -- Index Rebuild

- **Goal**: Build/update keyword + embedding + graph indexes
- **Outputs**: Index files (derived)
- **Gate**: If embeddings fail (GPU busy), FALLBACK to keyword-only; mark `index_state=degraded`; never block the pipeline on embeddings

### Stage 15 -- Publish + Report

- **Goal**: Write a compact machine + human report of what changed
- **Outputs**: `nightshift_report.json`, `nightshift_report.md`
- **Gate**: Always runs unless Stage 2 aborted

### Stage 16 -- Narrative Fusion

- **Goal**: Fuse evidence from git/chat/agent extractors into canonical claim store.
- **on_fail**: continue

### Stage 17 -- Solver Discovery

- **Goal**: Identify and register available solver capabilities across the mesh.
- **on_fail**: continue

## Canon Ledger

The Night Shift produces a Canon Ledger for your life/projects:

| Category | Content |
|----------|---------|
| **Definitions** | What words mean in your ecosystem |
| **Decisions** | What was decided and why |
| **Architecture** | Current truth of system design |
| **Open Questions** | Unresolved, prioritized |
| **Timeline** | Changes over time with evidence |

Every canon item points back to raw records. That's what keeps it grounded.

## Cross-Topic Linking

Most systems do "semantic similarity" and call it a day. That's weak.

MultiHead links by **shared entities + shared claims + shared constraints**, then optional embedding similarity as a secondary signal.

High-value link examples:

- Same constraint appears in two projects ("local-first deletion guarantees")
- Same person/company referenced across threads
- Same failure pattern repeats in different pipelines
- Same tool interface gets reinvented twice

Each link includes: reason (type), evidence pointers, suggested action ("merge docs", "promote to glossary", "create reusable tool").

## Daily/Weekly Briefs

### Daily Brief Sections

- **Top Decisions**: Decisions made today with citations
- **Progress**: What moved forward
- **Open Loops**: Unresolved items
- **Next Actions**: Suggested priorities for tomorrow

### Weekly Rollup

- Trends across the week
- Repeated blockers
- Progress vs goals
- Cross-topic connections discovered

## Default "Don't Screw Yourself" Policies

1. **No canon without evidence**: Packs only include accepted claims / confirmed events
2. **Embeddings never block**: If GPU is busy, ship keyword index and keep going
3. **Conflicts never auto-hide**: Unresolved conflicts become `contested` + go into review queue

## Night Shift Gating Rule Format

Each stage follows a uniform format:

```json
{
  "accept_if": [{"metric": "...", "op": ">=", "value": 0.8}],
  "retry": {"max_attempts": 2, "mutate_params": [...]},
  "fallback": {"tool": "...", "params": {...}},
  "on_fail": "abort | continue | skip | queue_review"
}
```

## Night Shift Trigger Conditions

Only run when:

- GPU idle or Night Shift uses CPU-only heads
- No active user session (or after N minutes idle)

Budget caps:

- Max tokens/day
- Max pack size
- Max claims/day

