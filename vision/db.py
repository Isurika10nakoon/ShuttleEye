# db.py — ShuttleEye PostgreSQL connection + schema
# ═══════════════════════════════════════════════════════════════════════
#  Everything the app persists — accounts, matches, sets, rallies — goes
#  through this module. Connection settings come from a local .env file
#  (see .env.example), loaded once at import time.
# ═══════════════════════════════════════════════════════════════════════

import os
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": os.environ.get("DB_PORT", "5432"),
    "dbname": os.environ.get("DB_NAME", "shuttleeye"),
    "user": os.environ.get("DB_USER", "shuttleeye_app"),
    "password": os.environ.get("DB_PASSWORD", ""),
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    username      TEXT UNIQUE NOT NULL,
    salt          TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('admin', 'umpire')),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS matches (
    id            SERIAL PRIMARY KEY,
    umpire_id     INTEGER REFERENCES users(id) ON DELETE SET NULL,
    umpire_name   TEXT NOT NULL,
    court_name    TEXT NOT NULL DEFAULT 'Court 1',
    name_a        TEXT NOT NULL,
    name_b        TEXT NOT NULL,
    winning_score INTEGER NOT NULL,
    score_a       INTEGER NOT NULL DEFAULT 0,
    score_b       INTEGER NOT NULL DEFAULT 0,
    set_num       INTEGER NOT NULL DEFAULT 1,
    sets_a        INTEGER NOT NULL DEFAULT 0,
    sets_b        INTEGER NOT NULL DEFAULT 0,
    winner_name   TEXT,
    status        TEXT NOT NULL DEFAULT 'in_progress'
                      CHECK (status IN ('in_progress', 'completed', 'abandoned')),
    started_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at      TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS sets (
    id          SERIAL PRIMARY KEY,
    match_id    INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    set_num     INTEGER NOT NULL,
    score_a     INTEGER NOT NULL,
    score_b     INTEGER NOT NULL,
    winner_side TEXT CHECK (winner_side IN ('A', 'B')),
    ended_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (match_id, set_num)
);

CREATE TABLE IF NOT EXISTS rallies (
    id          SERIAL PRIMARY KEY,
    match_id    INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    set_num     INTEGER NOT NULL,
    rally_num   INTEGER NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    side        TEXT NOT NULL CHECK (side IN ('A', 'B')),
    decision    TEXT NOT NULL,
    line_offset TEXT,
    score_a     INTEGER NOT NULL,
    score_b     INTEGER NOT NULL,
    note        TEXT
);

CREATE INDEX IF NOT EXISTS idx_sets_match    ON sets(match_id);
CREATE INDEX IF NOT EXISTS idx_rallies_match ON rallies(match_id);
"""

# Columns added after the original schema shipped — CREATE TABLE IF NOT
# EXISTS above won't retrofit these onto a database that already has the
# tables, so they're applied separately and are safe to re-run.
MIGRATIONS = """
ALTER TABLE matches ADD COLUMN IF NOT EXISTS court_name TEXT NOT NULL DEFAULT 'Court 1';
ALTER TABLE matches ADD COLUMN IF NOT EXISTS score_a    INTEGER NOT NULL DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS score_b    INTEGER NOT NULL DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS set_num    INTEGER NOT NULL DEFAULT 1;
CREATE INDEX IF NOT EXISTS idx_matches_status ON matches(status);
"""


@contextmanager
def get_conn():
    conn = psycopg.connect(row_factory=dict_row, **DB_CONFIG)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create tables if they don't already exist, and apply any schema
    migrations. Safe to call on every app startup."""
    with get_conn() as conn:
        conn.execute(SCHEMA)
        conn.execute(MIGRATIONS)


# ── Matches / sets / rallies ─────────────────────────────────────────

def create_match(umpire_id, umpire_name, court_name, name_a, name_b, winning_score):
    with get_conn() as conn:
        row = conn.execute(
            "INSERT INTO matches (umpire_id, umpire_name, court_name, name_a, name_b, winning_score) "
            "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
            (umpire_id, umpire_name, court_name, name_a, name_b, winning_score),
        ).fetchone()
    return row["id"]


def update_match_names(match_id, name_a, name_b):
    with get_conn() as conn:
        conn.execute(
            "UPDATE matches SET name_a = %s, name_b = %s WHERE id = %s",
            (name_a, name_b, match_id),
        )


def update_match_live_score(match_id, score_a, score_b, set_num):
    """Called on every point/undo so admins watching all courts see a
    near-live in-set score, not just completed-set totals."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE matches SET score_a = %s, score_b = %s, set_num = %s WHERE id = %s",
            (score_a, score_b, set_num, match_id),
        )


def update_match_progress(match_id, sets_a, sets_b):
    with get_conn() as conn:
        conn.execute(
            "UPDATE matches SET sets_a = %s, sets_b = %s WHERE id = %s",
            (sets_a, sets_b, match_id),
        )


def finish_match(match_id, sets_a, sets_b, winner_name):
    with get_conn() as conn:
        conn.execute(
            "UPDATE matches SET sets_a = %s, sets_b = %s, winner_name = %s, "
            "status = 'completed', ended_at = now() WHERE id = %s",
            (sets_a, sets_b, winner_name, match_id),
        )


def abandon_match(match_id, sets_a, sets_b):
    with get_conn() as conn:
        conn.execute(
            "UPDATE matches SET sets_a = %s, sets_b = %s, status = 'abandoned', "
            "ended_at = now() WHERE id = %s AND status = 'in_progress'",
            (sets_a, sets_b, match_id),
        )


def record_set(match_id, set_num, score_a, score_b, winner_side):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sets (match_id, set_num, score_a, score_b, winner_side) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (match_id, set_num) DO UPDATE SET "
            "score_a = EXCLUDED.score_a, score_b = EXCLUDED.score_b, "
            "winner_side = EXCLUDED.winner_side, ended_at = now()",
            (match_id, set_num, score_a, score_b, winner_side),
        )


def record_rally(match_id, set_num, rally_num, side, decision, line_offset,
                  score_a, score_b, note):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO rallies (match_id, set_num, rally_num, side, decision, "
            "line_offset, score_a, score_b, note) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (match_id, set_num, rally_num, side, decision, line_offset,
             score_a, score_b, note),
        )


# ── Tournament-wide views (admin, multi-court) ────────────────────────
#  These read straight from the `matches` table, which every court's
#  UmpireDashboard keeps live-updated — so they work across *any* number
#  of courts as long as each court's ShuttleEye instance points at this
#  same database (see .env / DB_HOST).

def list_active_matches():
    """One row per match currently in progress, across all courts."""
    with get_conn() as conn:
        return conn.execute(
            "SELECT id, court_name, umpire_name, name_a, name_b, "
            "score_a, score_b, set_num, sets_a, sets_b, winning_score, "
            "status, started_at "
            "FROM matches WHERE status = 'in_progress' "
            "ORDER BY court_name"
        ).fetchall()


def list_recent_matches(limit=10):
    """Most recently finished/abandoned matches, for context under the
    live board."""
    with get_conn() as conn:
        return conn.execute(
            "SELECT id, court_name, umpire_name, name_a, name_b, "
            "score_a, score_b, sets_a, sets_b, winning_score, "
            "status, winner_name, started_at, ended_at "
            "FROM matches WHERE status != 'in_progress' "
            "ORDER BY ended_at DESC LIMIT %s", (limit,)
        ).fetchall()
