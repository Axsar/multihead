# Claim and Event Schemas (JSON Schema 2020-12)

## Evidence Pointer (Shared Building Block)

The evidence pointer is the fundamental reference type that links Claims and Events back to raw records.

```json
{
  "record_id": "rec_01J...",
  "uri": "file:///...records/2026-02-11/chat_0007.jsonl",
  "sha256": "...",
  "span": { "start": 1832, "end": 2451, "unit": "chars" },
  "locator": {
    "page": null,
    "line_start": null,
    "line_end": null,
    "json_path": "$.messages[14]"
  },
  "quote": "optional short excerpt <= 200 chars",
  "captured_at": "2026-02-11T08:12:03Z"
}
```

**Rules:**

- `quote` is optional and short (debugging aid only); truth comes from `record_id` + `span`
- `span` is required for text-like records; for binary records rely on `sha256`

## Event Schema

### Event Statuses

| Status | Description |
|--------|-------------|
| `draft` | Created by extractor, not verified by checks |
| `confirmed` | Passed validation or user confirmed it |
| `corrected` | Updated because earlier extraction was wrong/incomplete |
| `retracted` | Should not be used (bad parse, wrong association) |

### Event Types

`decision`, `task_created`, `task_completed`, `tool_run`, `meeting`, `commit`, `deployment`, `payment`, `note`, `milestone`, `question`, `answer`, `incident`, `spec_change`

### Event Object (Example)

```json
{
  "event_id": "evt_01J...",
  "event_status": "draft",
  "event_type": "decision",
  "title": "Short human title",
  "summary": "1-3 sentences, optional",

  "time": {
    "happened_at": "2026-02-11T07:58:10Z",
    "ended_at": null,
    "timezone": "America/Los_Angeles",
    "time_precision": "second"
  },

  "actors": [
    { "actor_type": "user", "actor_id": "alice", "display": "Alice" }
  ],

  "entities": [
    { "entity_type": "project", "entity_id": "acp-runtime", "label": "ACP Runtime" }
  ],

  "tags": ["memory", "architecture", "night-shift"],
  "topic_ids": ["tpc_01J..."],

  "evidence": {
    "supports": [ /* EvidencePointer[] */ ],
    "related": [ /* EvidencePointer[] */ ]
  },

  "metrics": {
    "confidence": 0.82,
    "coverage": 0.6,
    "importance": 0.7
  },

  "relationships": {
    "caused_by_event_ids": ["evt_..."],
    "supersedes_event_ids": [],
    "duplicates_event_ids": []
  },

  "provenance": {
    "produced_by": { "kind": "extractor", "id": "nightshift_v1" },
    "toolchain": [{ "tool": "llm.core", "model": "qwen3:8b", "params_hash": "..." }],
    "created_at": "2026-02-11T08:15:00Z",
    "updated_at": "2026-02-11T08:15:00Z"
  }
}
```

### Event Invariants

1. `confirmed` events must have `evidence.supports.length >= 1`
2. `corrected` events must reference at least one `supersedes_event_ids`
3. `retracted` events remain stored (audit) but must be excluded from context packs by default

## event.schema.json (JSON Schema 2020-12)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://acp.local/schemas/event.schema.json",
  "title": "Event",
  "type": "object",
  "additionalProperties": false,
  "required": ["event_id", "event_status", "event_type", "title", "time", "evidence", "provenance"],
  "$defs": {
    "id_evt": { "type": "string", "pattern": "^evt_[A-Za-z0-9]{6,}$" },
    "id_rec": { "type": "string", "pattern": "^rec_[A-Za-z0-9]{6,}$" },
    "id_tpc": { "type": "string", "pattern": "^tpc_[A-Za-z0-9]{6,}$" },

    "actorRef": {
      "type": "object",
      "additionalProperties": false,
      "required": ["actor_type", "actor_id"],
      "properties": {
        "actor_type": { "type": "string", "enum": ["user", "person", "agent", "service"] },
        "actor_id": { "type": "string", "minLength": 1 },
        "display": { "type": "string" }
      }
    },

    "entityRef": {
      "type": "object",
      "additionalProperties": false,
      "required": ["entity_type", "entity_id"],
      "properties": {
        "entity_type": {
          "type": "string",
          "enum": ["project", "repo", "file", "model", "tool", "company", "property", "contract", "concept", "component"]
        },
        "entity_id": { "type": "string", "minLength": 1 },
        "label": { "type": "string" }
      }
    },

    "evidencePointer": {
      "type": "object",
      "additionalProperties": false,
      "required": ["record_id", "captured_at"],
      "properties": {
        "record_id": { "$ref": "#/$defs/id_rec" },
        "uri": { "type": "string", "minLength": 1 },
        "sha256": { "type": "string", "pattern": "^[a-fA-F0-9]{64}$" },
        "span": {
          "type": "object",
          "additionalProperties": false,
          "required": ["start", "end", "unit"],
          "properties": {
            "start": { "type": "integer", "minimum": 0 },
            "end": { "type": "integer", "minimum": 0 },
            "unit": { "type": "string", "enum": ["chars", "bytes", "tokens"] }
          }
        },
        "locator": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "page": { "type": ["integer", "null"], "minimum": 0 },
            "line_start": { "type": ["integer", "null"], "minimum": 0 },
            "line_end": { "type": ["integer", "null"], "minimum": 0 },
            "json_path": { "type": ["string", "null"] }
          }
        },
        "quote": { "type": "string", "maxLength": 200 },
        "captured_at": { "type": "string", "format": "date-time" }
      },
      "anyOf": [
        { "required": ["uri"] },
        { "required": ["sha256"] }
      ]
    },

    "timeBlock": {
      "type": "object",
      "additionalProperties": false,
      "required": ["happened_at", "timezone", "time_precision"],
      "properties": {
        "happened_at": { "type": "string", "format": "date-time" },
        "ended_at": { "type": ["string", "null"], "format": "date-time" },
        "timezone": { "type": "string", "minLength": 1 },
        "time_precision": { "type": "string", "enum": ["second", "minute", "hour", "day", "unknown"] }
      }
    },

    "metrics": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "confidence": { "type": "number", "minimum": 0, "maximum": 1 },
        "coverage": { "type": "number", "minimum": 0, "maximum": 1 },
        "importance": { "type": "number", "minimum": 0, "maximum": 1 }
      }
    },

    "relationships": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "caused_by_event_ids": { "type": "array", "items": { "$ref": "#/$defs/id_evt" }, "default": [] },
        "supersedes_event_ids": { "type": "array", "items": { "$ref": "#/$defs/id_evt" }, "default": [] },
        "duplicates_event_ids": { "type": "array", "items": { "$ref": "#/$defs/id_evt" }, "default": [] }
      }
    },

    "provenance": {
      "type": "object",
      "additionalProperties": false,
      "required": ["produced_by", "created_at", "updated_at"],
      "properties": {
        "produced_by": {
          "type": "object",
          "additionalProperties": false,
          "required": ["kind", "id"],
          "properties": {
            "kind": { "type": "string", "enum": ["extractor", "user", "agent"] },
            "id": { "type": "string", "minLength": 1 }
          }
        },
        "toolchain": {
          "type": "array",
          "items": {
            "type": "object",
            "additionalProperties": false,
            "required": ["tool"],
            "properties": {
              "tool": { "type": "string" },
              "model": { "type": "string" },
              "params_hash": { "type": "string" }
            }
          },
          "default": []
        },
        "created_at": { "type": "string", "format": "date-time" },
        "updated_at": { "type": "string", "format": "date-time" }
      }
    }
  },

  "properties": {
    "event_id": { "$ref": "#/$defs/id_evt" },
    "event_status": { "type": "string", "enum": ["draft", "confirmed", "corrected", "retracted"] },
    "event_type": {
      "type": "string",
      "enum": [
        "decision", "task_created", "task_completed", "tool_run", "meeting", "commit",
        "deployment", "payment", "note", "milestone", "question", "answer", "incident", "spec_change"
      ]
    },
    "title": { "type": "string", "minLength": 1, "maxLength": 160 },
    "summary": { "type": "string", "maxLength": 2000 },
    "time": { "$ref": "#/$defs/timeBlock" },
    "actors": { "type": "array", "items": { "$ref": "#/$defs/actorRef" }, "default": [] },
    "entities": { "type": "array", "items": { "$ref": "#/$defs/entityRef" }, "default": [] },
    "tags": { "type": "array", "items": { "type": "string", "minLength": 1 }, "default": [] },
    "topic_ids": { "type": "array", "items": { "$ref": "#/$defs/id_tpc" }, "default": [] },
    "evidence": {
      "type": "object",
      "additionalProperties": false,
      "required": ["supports", "related"],
      "properties": {
        "supports": { "type": "array", "items": { "$ref": "#/$defs/evidencePointer" }, "default": [] },
        "related": { "type": "array", "items": { "$ref": "#/$defs/evidencePointer" }, "default": [] }
      }
    },
    "metrics": { "$ref": "#/$defs/metrics" },
    "relationships": { "$ref": "#/$defs/relationships" },
    "provenance": { "$ref": "#/$defs/provenance" }
  },

  "allOf": [
    {
      "if": { "properties": { "event_status": { "const": "confirmed" } } },
      "then": { "properties": { "evidence": { "properties": { "supports": { "minItems": 1 } } } } }
    },
    {
      "if": { "properties": { "event_status": { "const": "corrected" } } },
      "then": { "properties": { "relationships": { "properties": { "supersedes_event_ids": { "minItems": 1 } } } } }
    }
  ]
}
```

## Claim Schema

A Claim is "something asserted as true" with explicit supporting evidence and a lifecycle.

### Claim Statuses

| Status | Description |
|--------|-------------|
| `proposed` | Extracted or suggested, not yet trusted |
| `accepted` | Considered true "for now" within scope |
| `contested` | Conflicting accepted/proposed claims exist |
| `superseded` | Replaced by a newer accepted claim (kept for history) |
| `rejected` | Evaluated and determined false/unsupported |
| `deprecated` | True but no longer relevant/used (helps pruning) |
| `resolved` | A question/request that has been answered or fulfilled |

### Claim Types

`definition` (glossary), `decision` (architectural/business), `fact` (stable-ish statement), `constraint` (must/should rule), `preference` (user preference), `plan` (intended future), `assumption` (explicit assumption), `risk` (risk statement), `question` (decomposition request or inquiry)

### Claim Object (Example)

```json
{
  "claim_id": "clm_01J...",
  "claim_status": "proposed",
  "claim_type": "decision",

  "scope": {
    "scope_type": "project",
    "scope_id": "acp-runtime",
    "visibility": "private",
    "valid_from": "2026-02-11T00:00:00Z",
    "valid_to": null
  },

  "canonical": {
    "claim_key": "acp.runtime.core_model.default_mode",
    "subject": { "entity_type": "component", "entity_id": "core_llm" },
    "predicate": "runs_on",
    "object": { "value_type": "enum", "value": "cpu_by_default" }
  },

  "statement": "The core LLM runs on CPU by default so it stays available while the GPU is busy.",
  "rationale": "Keeps agentic core responsive during batch pipelines; avoids VRAM contention.",

  "evidence": {
    "supports": [ /* EvidencePointer[] */ ],
    "opposes": [ /* EvidencePointer[] */ ]
  },

  "quality": {
    "confidence": 0.78,
    "stability": "medium",   // volatile | temporary | medium | stable | high
    "importance": 0.85
  },

  "links": {
    "derived_from_event_ids": ["evt_01J..."],
    "related_claim_ids": ["clm_..."],
    "conflicts_with_claim_ids": ["clm_..."]
  },

  "resolution": {
    "superseded_by_claim_id": null,
    "rejection_reason": null,
    "contested_reason": null
  },

  "provenance": {
    "produced_by": { "kind": "extractor", "id": "nightshift_v1" },
    "toolchain": [{ "tool": "llm.core", "model": "qwen3:8b", "params_hash": "..." }],
    "created_at": "2026-02-11T08:20:00Z",
    "updated_at": "2026-02-11T08:20:00Z"
  }
}
```

### Claim Invariants

1. `accepted` claims must have `evidence.supports.length >= 1` (no evidence = can't be "canon")
2. `superseded` claims must have `resolution.superseded_by_claim_id` set
3. A `claim_key` should have at most one `accepted` claim per scope at a time
4. If a new claim with the same `claim_key` is accepted, older accepted becomes `superseded`
5. `contested` claims must have `links.conflicts_with_claim_ids.length >= 1`
6. `rejected` claims keep evidence (audit) but are excluded from packs by default
7. `assumption` claims should never be auto-upgraded to `accepted` without user confirmation or very high confidence + repeated support

### Recommended claim_key Conventions

Use dot paths that stay stable:

```
acp.runtime.core_model.default_mode
acp.permissions.default_network
project.botvibes.discovery.capability_schema.v1
user.pref.response_style.honest_checked_logic
```

## claim.schema.json (JSON Schema 2020-12)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://acp.local/schemas/claim.schema.json",
  "title": "Claim",
  "type": "object",
  "additionalProperties": false,
  "required": ["claim_id", "claim_status", "claim_type", "scope", "canonical", "statement", "evidence", "provenance"],
  "$defs": {
    "id_clm": { "type": "string", "pattern": "^clm_[A-Za-z0-9]{6,}$" },
    "id_rec": { "type": "string", "pattern": "^rec_[A-Za-z0-9]{6,}$" },
    "id_evt": { "type": "string", "pattern": "^evt_[A-Za-z0-9]{6,}$" },

    "evidencePointer": {
      "type": "object",
      "additionalProperties": false,
      "required": ["record_id", "captured_at"],
      "properties": {
        "record_id": { "$ref": "#/$defs/id_rec" },
        "uri": { "type": "string", "minLength": 1 },
        "sha256": { "type": "string", "pattern": "^[a-fA-F0-9]{64}$" },
        "span": {
          "type": "object",
          "additionalProperties": false,
          "required": ["start", "end", "unit"],
          "properties": {
            "start": { "type": "integer", "minimum": 0 },
            "end": { "type": "integer", "minimum": 0 },
            "unit": { "type": "string", "enum": ["chars", "bytes", "tokens"] }
          }
        },
        "locator": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "page": { "type": ["integer", "null"], "minimum": 0 },
            "line_start": { "type": ["integer", "null"], "minimum": 0 },
            "line_end": { "type": ["integer", "null"], "minimum": 0 },
            "json_path": { "type": ["string", "null"] }
          }
        },
        "quote": { "type": "string", "maxLength": 200 },
        "captured_at": { "type": "string", "format": "date-time" }
      },
      "anyOf": [
        { "required": ["uri"] },
        { "required": ["sha256"] }
      ]
    },

    "scope": {
      "type": "object",
      "additionalProperties": false,
      "required": ["scope_type", "scope_id", "visibility", "valid_from", "valid_to"],
      "properties": {
        "scope_type": { "type": "string", "enum": ["global", "project", "repo", "person", "session"] },
        "scope_id": { "type": "string", "minLength": 1 },
        "visibility": { "type": "string", "enum": ["private", "shared"] },
        "valid_from": { "type": "string", "format": "date-time" },
        "valid_to": { "type": ["string", "null"], "format": "date-time" }
      }
    },

    "entityRef": {
      "type": "object",
      "additionalProperties": false,
      "required": ["entity_type", "entity_id"],
      "properties": {
        "entity_type": {
          "type": "string",
          "enum": ["project", "repo", "file", "model", "tool", "company", "property", "contract", "concept", "component"]
        },
        "entity_id": { "type": "string", "minLength": 1 },
        "label": { "type": "string" }
      }
    },

    "valueObject": {
      "type": "object",
      "additionalProperties": false,
      "required": ["value_type", "value"],
      "properties": {
        "value_type": { "type": "string", "enum": ["enum", "string", "number", "boolean", "json"] },
        "value": {}
      },
      "allOf": [
        {
          "if": { "properties": { "value_type": { "const": "number" } }, "required": ["value_type"] },
          "then": { "properties": { "value": { "type": "number" } } }
        },
        {
          "if": { "properties": { "value_type": { "const": "boolean" } }, "required": ["value_type"] },
          "then": { "properties": { "value": { "type": "boolean" } } }
        },
        {
          "if": { "properties": { "value_type": { "enum": ["enum", "string"] } }, "required": ["value_type"] },
          "then": { "properties": { "value": { "type": "string" } } }
        }
      ]
    },

    "canonical": {
      "type": "object",
      "additionalProperties": false,
      "required": ["claim_key", "subject", "predicate", "object"],
      "properties": {
        "claim_key": { "type": "string", "pattern": "^[a-z0-9]+(\\.[a-z0-9_]+)+$" },
        "subject": { "$ref": "#/$defs/entityRef" },
        "predicate": { "type": "string", "minLength": 1, "maxLength": 80 },
        "object": { "$ref": "#/$defs/valueObject" }
      }
    },

    "quality": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "confidence": { "type": "number", "minimum": 0, "maximum": 1 },
        "stability": { "type": "string", "enum": ["volatile", "medium", "stable", "high", "temporary"] },
        "importance": { "type": "number", "minimum": 0, "maximum": 1 }
      }
    },

    "links": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "derived_from_event_ids": { "type": "array", "items": { "$ref": "#/$defs/id_evt" }, "default": [] },
        "related_claim_ids": { "type": "array", "items": { "$ref": "#/$defs/id_clm" }, "default": [] },
        "conflicts_with_claim_ids": { "type": "array", "items": { "$ref": "#/$defs/id_clm" }, "default": [] }
      }
    },

    "resolution": {
      "type": "object",
      "additionalProperties": false,
      "required": ["superseded_by_claim_id", "rejection_reason", "contested_reason"],
      "properties": {
        "superseded_by_claim_id": { "type": ["string", "null"], "pattern": "^clm_[A-Za-z0-9]{6,}$" },
        "rejection_reason": { "type": ["string", "null"], "maxLength": 1000 },
        "contested_reason": { "type": ["string", "null"], "maxLength": 1000 }
      }
    },

    "provenance": {
      "type": "object",
      "additionalProperties": false,
      "required": ["produced_by", "created_at", "updated_at"],
      "properties": {
        "produced_by": {
          "type": "object",
          "additionalProperties": false,
          "required": ["kind", "id"],
          "properties": {
            "kind": { "type": "string", "enum": ["extractor", "user", "agent"] },
            "id": { "type": "string", "minLength": 1 }
          }
        },
        "toolchain": {
          "type": "array",
          "items": {
            "type": "object",
            "additionalProperties": false,
            "required": ["tool"],
            "properties": {
              "tool": { "type": "string" },
              "model": { "type": "string" },
              "params_hash": { "type": "string" }
            }
          },
          "default": []
        },
        "created_at": { "type": "string", "format": "date-time" },
        "updated_at": { "type": "string", "format": "date-time" }
      }
    }
  },

  "properties": {
    "claim_id": { "$ref": "#/$defs/id_clm" },
    "claim_status": {
      "type": "string",
      "enum": ["proposed", "accepted", "contested", "superseded", "rejected", "deprecated", "resolved"]
    },
    "claim_type": {
      "type": "string",
      "enum": ["definition", "decision", "fact", "constraint", "preference", "plan", "assumption", "risk", "question"]
    },
    "scope": { "$ref": "#/$defs/scope" },
    "canonical": { "$ref": "#/$defs/canonical" },
    "statement": { "type": "string", "minLength": 1, "maxLength": 4000 },
    "rationale": { "type": "string", "maxLength": 4000 },
    "evidence": {
      "type": "object",
      "additionalProperties": false,
      "required": ["supports", "opposes"],
      "properties": {
        "supports": { "type": "array", "items": { "$ref": "#/$defs/evidencePointer" }, "default": [] },
        "opposes": { "type": "array", "items": { "$ref": "#/$defs/evidencePointer" }, "default": [] }
      }
    },
    "quality": { "$ref": "#/$defs/quality" },
    "links": { "$ref": "#/$defs/links" },
    "resolution": { "$ref": "#/$defs/resolution" },
    "provenance": { "$ref": "#/$defs/provenance" }
  },

  "allOf": [
    {
      "if": { "properties": { "claim_status": { "const": "accepted" } } },
      "then": { "properties": { "evidence": { "properties": { "supports": { "minItems": 1 } } } } }
    },
    {
      "if": { "properties": { "claim_status": { "const": "superseded" } } },
      "then": { "properties": { "resolution": { "properties": { "superseded_by_claim_id": { "type": "string" } }, "required": ["superseded_by_claim_id"] } } }
    },
    {
      "if": { "properties": { "claim_status": { "const": "contested" } } },
      "then": { "properties": { "links": { "properties": { "conflicts_with_claim_ids": { "minItems": 1 } } } } }
    }
  ]
}
```

## SQLite Layout

### Tables

```sql
PRAGMA foreign_keys = ON;

-- 1) Raw records (immutable evidence sources)
CREATE TABLE records (
  record_id     TEXT PRIMARY KEY,
  uri           TEXT NOT NULL,
  sha256        TEXT,
  mime          TEXT,
  captured_at   TEXT NOT NULL,
  created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  CHECK (record_id GLOB 'rec_*'),
  CHECK (sha256 IS NULL OR length(sha256)=64)
);

CREATE INDEX idx_records_captured_at ON records(captured_at);
CREATE INDEX idx_records_sha256 ON records(sha256);

-- 2) Evidence pointers
CREATE TABLE evidence_pointers (
  evidence_id   TEXT PRIMARY KEY,
  record_id     TEXT NOT NULL REFERENCES records(record_id) ON DELETE CASCADE,
  uri           TEXT,
  sha256        TEXT,
  span_start    INTEGER,
  span_end      INTEGER,
  span_unit     TEXT,
  page          INTEGER,
  line_start    INTEGER,
  line_end      INTEGER,
  json_path     TEXT,
  quote         TEXT,
  captured_at   TEXT NOT NULL,
  created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  CHECK (sha256 IS NULL OR length(sha256)=64),
  CHECK (span_unit IS NULL OR span_unit IN ('chars','bytes','tokens')),
  CHECK (quote IS NULL OR length(quote) <= 200)
);

CREATE INDEX idx_evp_record ON evidence_pointers(record_id);

-- 3) Events
CREATE TABLE events (
  event_id      TEXT PRIMARY KEY,
  event_status  TEXT NOT NULL CHECK (event_status IN ('draft','confirmed','corrected','retracted')),
  event_type    TEXT NOT NULL,
  title         TEXT NOT NULL,
  summary       TEXT,
  happened_at   TEXT NOT NULL,
  ended_at      TEXT,
  timezone      TEXT NOT NULL,
  time_precision TEXT NOT NULL CHECK (time_precision IN ('second','minute','hour','day','unknown')),
  actors_json   TEXT NOT NULL DEFAULT '[]',
  entities_json TEXT NOT NULL DEFAULT '[]',
  tags_json     TEXT NOT NULL DEFAULT '[]',
  topic_ids_json TEXT NOT NULL DEFAULT '[]',
  metrics_json  TEXT,
  provenance_json TEXT NOT NULL,
  created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  updated_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  CHECK (event_id GLOB 'evt_*')
);

CREATE INDEX idx_events_time ON events(happened_at);
CREATE INDEX idx_events_status ON events(event_status);
CREATE INDEX idx_events_type ON events(event_type);

-- Event evidence links
CREATE TABLE event_evidence (
  event_id     TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
  evidence_id  TEXT NOT NULL REFERENCES evidence_pointers(evidence_id) ON DELETE CASCADE,
  role         TEXT NOT NULL CHECK (role IN ('supports','related')),
  PRIMARY KEY (event_id, evidence_id, role)
);

-- Event relationship links
CREATE TABLE event_links (
  from_event_id TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
  to_event_id   TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
  link_type     TEXT NOT NULL CHECK (link_type IN ('caused_by','supersedes','duplicates')),
  PRIMARY KEY (from_event_id, to_event_id, link_type)
);

-- 4) Claims
CREATE TABLE claims (
  claim_id      TEXT PRIMARY KEY,
  claim_status  TEXT NOT NULL CHECK (claim_status IN ('proposed','accepted','contested','superseded','rejected','deprecated')),
  claim_type    TEXT NOT NULL CHECK (claim_type IN ('definition','decision','fact','constraint','preference','plan','assumption','risk')),
  scope_type    TEXT NOT NULL CHECK (scope_type IN ('global','project','repo','person','session')),
  scope_id      TEXT NOT NULL,
  visibility    TEXT NOT NULL CHECK (visibility IN ('private','shared')),
  valid_from    TEXT NOT NULL,
  valid_to      TEXT,
  claim_key     TEXT NOT NULL,
  predicate     TEXT NOT NULL,
  subject_json  TEXT NOT NULL,
  object_json   TEXT NOT NULL,
  statement     TEXT NOT NULL,
  rationale     TEXT,
  confidence    REAL,
  stability     TEXT CHECK (stability IN ('volatile','medium','stable')),
  importance    REAL,
  superseded_by_claim_id TEXT REFERENCES claims(claim_id),
  rejection_reason TEXT,
  contested_reason TEXT,
  links_json    TEXT NOT NULL DEFAULT '{}',
  provenance_json TEXT NOT NULL,
  created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  updated_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  CHECK (claim_id GLOB 'clm_*')
);

CREATE INDEX idx_claims_key_scope ON claims(scope_type, scope_id, claim_key);
CREATE INDEX idx_claims_status ON claims(claim_status);
CREATE INDEX idx_claims_updated ON claims(updated_at);

-- Enforce: at most ONE accepted claim per (scope_type, scope_id, claim_key)
CREATE UNIQUE INDEX ux_claims_one_accepted_per_key_scope
  ON claims(scope_type, scope_id, claim_key)
  WHERE claim_status = 'accepted';

-- Claim evidence links
CREATE TABLE claim_evidence (
  claim_id     TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
  evidence_id  TEXT NOT NULL REFERENCES evidence_pointers(evidence_id) ON DELETE CASCADE,
  stance       TEXT NOT NULL CHECK (stance IN ('supports','opposes')),
  PRIMARY KEY (claim_id, evidence_id, stance)
);

-- Claim conflicts
CREATE TABLE claim_conflicts (
  claim_id_a   TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
  claim_id_b   TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
  reason       TEXT,
  PRIMARY KEY (claim_id_a, claim_id_b)
);
```

### Enforcement Triggers

```sql
-- Accepted claim must have >= 1 supporting evidence
CREATE TRIGGER trg_claim_accept_requires_support
BEFORE UPDATE OF claim_status ON claims
WHEN NEW.claim_status = 'accepted'
BEGIN
  SELECT CASE
    WHEN (SELECT COUNT(*) FROM claim_evidence WHERE claim_id = NEW.claim_id AND stance = 'supports') < 1
    THEN RAISE(ABORT, 'accepted claims require at least one supporting evidence pointer')
  END;
END;

-- Superseded claim must point to superseding claim
CREATE TRIGGER trg_claim_superseded_requires_target
BEFORE UPDATE OF claim_status ON claims
WHEN NEW.claim_status = 'superseded'
BEGIN
  SELECT CASE
    WHEN NEW.superseded_by_claim_id IS NULL
    THEN RAISE(ABORT, 'superseded claims require superseded_by_claim_id')
  END;
END;

-- Confirmed event must have >= 1 supporting evidence
CREATE TRIGGER trg_event_confirm_requires_support
BEFORE UPDATE OF event_status ON events
WHEN NEW.event_status = 'confirmed'
BEGIN
  SELECT CASE
    WHEN (SELECT COUNT(*) FROM event_evidence WHERE event_id = NEW.event_id AND role = 'supports') < 1
    THEN RAISE(ABORT, 'confirmed events require at least one supporting evidence pointer')
  END;
END;

-- Corrected event must supersede at least one prior event
CREATE TRIGGER trg_event_corrected_requires_supersedes
BEFORE UPDATE OF event_status ON events
WHEN NEW.event_status = 'corrected'
BEGIN
  SELECT CASE
    WHEN (SELECT COUNT(*) FROM event_links WHERE from_event_id = NEW.event_id AND link_type = 'supersedes') < 1
    THEN RAISE(ABORT, 'corrected events must supersede at least one prior event')
  END;
END;
```

### Canon Update Pattern

When a new claim becomes accepted and shares the same (scope_type, scope_id, claim_key):

1. Set old accepted claim -> `superseded` with `superseded_by_claim_id = NEW.claim_id`
2. Then set NEW claim -> `accepted`

That order avoids violating the unique index.

