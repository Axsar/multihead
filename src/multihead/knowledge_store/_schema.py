"""SQL schema and triggers for the knowledge store database."""

_SCHEMA = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

-- 1) Raw records (immutable evidence sources)
CREATE TABLE IF NOT EXISTS records (
    record_id     TEXT PRIMARY KEY,
    uri           TEXT NOT NULL,
    sha256        TEXT,
    mime          TEXT,
    captured_at   TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    CHECK (record_id GLOB 'rec_*')
);

CREATE INDEX IF NOT EXISTS idx_records_captured ON records(captured_at);

-- 2) Evidence pointers
CREATE TABLE IF NOT EXISTS evidence_pointers (
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
    CHECK (span_unit IS NULL OR span_unit IN ('chars','bytes','tokens')),
    CHECK (quote IS NULL OR length(quote) <= 200)
);

CREATE INDEX IF NOT EXISTS idx_evp_record ON evidence_pointers(record_id);

-- 3) Knowledge events (semantic, not run events)
CREATE TABLE IF NOT EXISTS knowledge_events (
    event_id      TEXT PRIMARY KEY,
    event_status  TEXT NOT NULL CHECK (event_status IN ('draft','confirmed','corrected','retracted')),
    event_type    TEXT NOT NULL,
    title         TEXT NOT NULL,
    summary       TEXT,
    happened_at   TEXT NOT NULL,
    ended_at      TEXT,
    tz            TEXT NOT NULL DEFAULT 'UTC',
    time_precision TEXT NOT NULL DEFAULT 'unknown',
    actors_json   TEXT NOT NULL DEFAULT '[]',
    entities_json TEXT NOT NULL DEFAULT '[]',
    tags_json     TEXT NOT NULL DEFAULT '[]',
    topic_ids_json TEXT NOT NULL DEFAULT '[]',
    metrics_json  TEXT NOT NULL DEFAULT '{}',
    caused_by_json TEXT NOT NULL DEFAULT '[]',
    supersedes_json TEXT NOT NULL DEFAULT '[]',
    duplicates_json TEXT NOT NULL DEFAULT '[]',
    provenance_json TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    CHECK (event_id GLOB 'evt_*')
);

CREATE INDEX IF NOT EXISTS idx_kevt_time ON knowledge_events(happened_at);
CREATE INDEX IF NOT EXISTS idx_kevt_status ON knowledge_events(event_status);
CREATE INDEX IF NOT EXISTS idx_kevt_type ON knowledge_events(event_type);

-- Event evidence links
CREATE TABLE IF NOT EXISTS event_evidence (
    event_id     TEXT NOT NULL REFERENCES knowledge_events(event_id) ON DELETE CASCADE,
    evidence_id  TEXT NOT NULL REFERENCES evidence_pointers(evidence_id) ON DELETE CASCADE,
    role         TEXT NOT NULL CHECK (role IN ('supports','related')),
    PRIMARY KEY (event_id, evidence_id, role)
);

-- Event relationship links
CREATE TABLE IF NOT EXISTS event_links (
    from_event_id TEXT NOT NULL REFERENCES knowledge_events(event_id) ON DELETE CASCADE,
    to_event_id   TEXT NOT NULL REFERENCES knowledge_events(event_id) ON DELETE CASCADE,
    link_type     TEXT NOT NULL CHECK (link_type IN ('caused_by','supersedes','duplicates')),
    PRIMARY KEY (from_event_id, to_event_id, link_type)
);

-- 4) Claims
CREATE TABLE IF NOT EXISTS claims (
    claim_id      TEXT PRIMARY KEY,
    claim_status  TEXT NOT NULL CHECK (claim_status IN ('proposed','accepted','corroborated','contested','superseded','rejected','deprecated','resolved','stale')),
    claim_type    TEXT NOT NULL,
    scope_type    TEXT NOT NULL,
    scope_id      TEXT NOT NULL,
    visibility    TEXT NOT NULL DEFAULT 'private',
    valid_from    TEXT NOT NULL,
    valid_to      TEXT,
    claim_key     TEXT NOT NULL,
    predicate     TEXT NOT NULL,
    subject_json  TEXT NOT NULL,
    object_json   TEXT NOT NULL,
    statement     TEXT NOT NULL,
    rationale     TEXT,
    confidence    REAL,
    stability     TEXT,
    importance    REAL,
    superseded_by_claim_id TEXT,
    rejection_reason TEXT,
    contested_reason TEXT,
    derived_from_json TEXT NOT NULL DEFAULT '[]',
    related_json  TEXT NOT NULL DEFAULT '[]',
    conflicts_json TEXT NOT NULL DEFAULT '[]',
    provenance_json TEXT NOT NULL,
    signature     TEXT DEFAULT '',
    signed_by     TEXT DEFAULT '',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    observation_method TEXT DEFAULT '',
    producer TEXT DEFAULT '',
    CHECK (claim_id GLOB 'clm_*')
);

CREATE INDEX IF NOT EXISTS idx_claims_key_scope ON claims(scope_type, scope_id, claim_key);
CREATE INDEX IF NOT EXISTS idx_claims_status ON claims(claim_status);
CREATE INDEX IF NOT EXISTS idx_claims_type ON claims(claim_type);
CREATE INDEX IF NOT EXISTS idx_claims_type_status ON claims(claim_type, claim_status, created_at);
CREATE INDEX IF NOT EXISTS idx_claims_updated ON claims(updated_at);
-- idx_claims_producer created via migration (column may not exist on old DBs)

-- At most ONE accepted claim per (scope_type, scope_id, claim_key)
CREATE UNIQUE INDEX IF NOT EXISTS ux_claims_one_accepted
    ON claims(scope_type, scope_id, claim_key)
    WHERE claim_status = 'accepted';

-- Claim evidence links
CREATE TABLE IF NOT EXISTS claim_evidence (
    claim_id     TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    evidence_id  TEXT NOT NULL REFERENCES evidence_pointers(evidence_id) ON DELETE CASCADE,
    stance       TEXT NOT NULL CHECK (stance IN ('supports','opposes')),
    PRIMARY KEY (claim_id, evidence_id, stance)
);

-- Claim conflicts
CREATE TABLE IF NOT EXISTS claim_conflicts (
    claim_id_a   TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    claim_id_b   TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    reason       TEXT,
    PRIMARY KEY (claim_id_a, claim_id_b)
);

-- 5) Cross-topic links
CREATE TABLE IF NOT EXISTS cross_links (
    link_id       TEXT PRIMARY KEY,
    from_type     TEXT NOT NULL,
    from_id       TEXT NOT NULL,
    from_label    TEXT,
    to_type       TEXT NOT NULL,
    to_id         TEXT NOT NULL,
    to_label      TEXT,
    reason_type   TEXT NOT NULL,
    reason        TEXT,
    score         REAL DEFAULT 0.0,
    link_status   TEXT DEFAULT 'draft',
    provenance_json TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    CHECK (link_id GLOB 'lnk_*')
);

-- Link evidence
CREATE TABLE IF NOT EXISTS link_evidence (
    link_id      TEXT NOT NULL REFERENCES cross_links(link_id) ON DELETE CASCADE,
    evidence_id  TEXT NOT NULL REFERENCES evidence_pointers(evidence_id) ON DELETE CASCADE,
    PRIMARY KEY (link_id, evidence_id)
);

-- 6) Claim interactions (agent read-receipts / action tracking)
CREATE TABLE IF NOT EXISTS claim_interactions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id      TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    agent_id      TEXT NOT NULL,
    action        TEXT NOT NULL CHECK (action IN ('read','responded','dismissed','acknowledged','resolved')),
    response_claim_id TEXT,
    context       TEXT,
    created_at    TEXT NOT NULL,
    UNIQUE(claim_id, agent_id, action)
);

CREATE INDEX IF NOT EXISTS idx_ci_claim ON claim_interactions(claim_id);
CREATE INDEX IF NOT EXISTS idx_ci_agent ON claim_interactions(agent_id);
CREATE INDEX IF NOT EXISTS idx_ci_agent_action ON claim_interactions(agent_id, action);

-- 7) Participant registry (stable agent identity across restarts)
CREATE TABLE IF NOT EXISTS participants (
    participant_id   TEXT PRIMARY KEY,
    name             TEXT NOT NULL UNIQUE,
    agent_type       TEXT NOT NULL DEFAULT 'shell',
    context_hash     TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    last_seen_at     TEXT NOT NULL,
    metadata_json    TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_participants_context ON participants(context_hash);
"""

_TRIGGERS = """
-- Accepted claim must have >= 1 supporting evidence
CREATE TRIGGER IF NOT EXISTS trg_claim_accept_requires_support
BEFORE UPDATE OF claim_status ON claims
WHEN NEW.claim_status = 'accepted'
BEGIN
    SELECT CASE
        WHEN (SELECT COUNT(*) FROM claim_evidence
              WHERE claim_id = NEW.claim_id AND stance = 'supports') < 1
        THEN RAISE(ABORT, 'accepted claims require at least one supporting evidence pointer')
    END;
END;

-- Superseded claim must point to superseding claim
CREATE TRIGGER IF NOT EXISTS trg_claim_superseded_requires_target
BEFORE UPDATE OF claim_status ON claims
WHEN NEW.claim_status = 'superseded'
BEGIN
    SELECT CASE
        WHEN NEW.superseded_by_claim_id IS NULL
        THEN RAISE(ABORT, 'superseded claims require superseded_by_claim_id')
    END;
END;

-- Confirmed event must have >= 1 supporting evidence
CREATE TRIGGER IF NOT EXISTS trg_event_confirm_requires_support
BEFORE UPDATE OF event_status ON knowledge_events
WHEN NEW.event_status = 'confirmed'
BEGIN
    SELECT CASE
        WHEN (SELECT COUNT(*) FROM event_evidence
              WHERE event_id = NEW.event_id AND role = 'supports') < 1
        THEN RAISE(ABORT, 'confirmed events require at least one supporting evidence pointer')
    END;
END;

-- Corrected event must supersede at least one prior event
CREATE TRIGGER IF NOT EXISTS trg_event_corrected_requires_supersedes
BEFORE UPDATE OF event_status ON knowledge_events
WHEN NEW.event_status = 'corrected'
BEGIN
    SELECT CASE
        WHEN (SELECT COUNT(*) FROM event_links
              WHERE from_event_id = NEW.event_id AND link_type = 'supersedes') < 1
        THEN RAISE(ABORT, 'corrected events must supersede at least one prior event')
    END;
END;
"""
