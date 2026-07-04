-- PR Merge Predictor -- D1 schema
-- This table is the entire "verification" mechanism: every scored PR gets
-- a row here at prediction time, and a cron worker fills in the actual
-- outcome once the PR closes. The public accuracy dashboard is just a
-- query over this table.

CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  owner TEXT NOT NULL,
  repo TEXT NOT NULL,
  pr_number INTEGER NOT NULL,
  pr_url TEXT NOT NULL,
  predicted_score REAL NOT NULL,       -- 0.0 to 1.0, probability of merge
  features_json TEXT NOT NULL,         -- snapshot of features used, for auditability
  created_at TEXT NOT NULL,            -- when we scored it
  resolved INTEGER NOT NULL DEFAULT 0, -- 0 = still open, 1 = resolved
  actual_merged INTEGER,               -- NULL until resolved, then 0 or 1
  resolved_at TEXT,
  UNIQUE(owner, repo, pr_number)
);

CREATE INDEX IF NOT EXISTS idx_unresolved ON predictions (resolved);
CREATE INDEX IF NOT EXISTS idx_created_at ON predictions (created_at);
