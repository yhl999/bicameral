-- Canonical candidates DB schema (v1)
-- Single source of truth to prevent drift between:
-- - tools/graphiti/truth/candidates.py
-- - tools/graphiti/ui/db.py

CREATE TABLE IF NOT EXISTS candidates (
  candidate_id TEXT PRIMARY KEY,       -- ULID
  created_at TEXT NOT NULL,            -- ISO-8601
  candidate_fingerprint TEXT NOT NULL, -- sha256(subject + LF + predicate + LF + scope + LF + value_json_c14n)

  -- Proposed fact
  subject TEXT NOT NULL,
  predicate TEXT NOT NULL,
  scope TEXT NOT NULL,                 -- v1: 'private'
  assertion_type TEXT NOT NULL,        -- decision|preference|factual_assertion|episode|question|hypothetical|quote|plan
  value_json TEXT NOT NULL,            -- canonical JSON value

  -- Provenance
  evidence_refs_json TEXT NOT NULL,    -- canonical JSON array
  evidence_quote TEXT,                 -- optional, <=200 chars

  -- Trust / extraction
  speaker_id TEXT,
  confidence REAL,
  source_trust TEXT,                   -- T1/T2/T3/T4 (optional)

  -- Policy state
  risk_level TEXT NOT NULL DEFAULT 'low',
  status TEXT NOT NULL DEFAULT 'pending',
  policy_version TEXT,

  -- Explainability (UI "why" box)
  policy_trace_json TEXT NOT NULL DEFAULT '{}',
  evidence_stats_json TEXT NOT NULL DEFAULT '{}',

  -- Conflict info
  conflict_with_fact_id TEXT,

  -- Decision (human or policy)
  decided_at TEXT,
  actor_id TEXT,
  ledger_event_id TEXT,
  decision TEXT,                       -- approved|denied|auto_promoted|expired
  decision_reason TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_candidates_fingerprint ON candidates(candidate_fingerprint);
CREATE INDEX IF NOT EXISTS idx_candidates_status ON candidates(status);
CREATE INDEX IF NOT EXISTS idx_candidates_predicate ON candidates(predicate);
CREATE INDEX IF NOT EXISTS idx_candidates_subject ON candidates(subject);
CREATE INDEX IF NOT EXISTS idx_candidates_conflict ON candidates(conflict_with_fact_id);

CREATE TABLE IF NOT EXISTS candidate_verifications (
  candidate_id TEXT NOT NULL,
  verification_status TEXT NOT NULL,
  evidence_source_ids TEXT NOT NULL,
  verifier_version TEXT NOT NULL,
  verified_at TEXT NOT NULL,
  PRIMARY KEY (candidate_id, verifier_version, verified_at)
);

CREATE INDEX IF NOT EXISTS idx_candidate_verifications_status_time
  ON candidate_verifications(verification_status, verified_at);

CREATE TABLE IF NOT EXISTS om_dead_letter_queue (
  message_id TEXT PRIMARY KEY,
  source_session_id TEXT NOT NULL,
  attempts INTEGER NOT NULL,
  last_error TEXT NOT NULL,
  first_failed_at TEXT NOT NULL,
  last_failed_at TEXT NOT NULL,
  last_chunk_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_om_dead_letter_queue_source_session
  ON om_dead_letter_queue(source_session_id);

CREATE INDEX IF NOT EXISTS idx_om_dead_letter_queue_last_failed
  ON om_dead_letter_queue(last_failed_at);
