"""SQLite schema: WAL journal, FTS5 lexical index, binary/Matryoshka vector blobs, causal tables."""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 2

DDL = [
    """CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)""",
    # Source files (content addressed by file_vec_id)
    """CREATE TABLE IF NOT EXISTS files (
        file_vec_id TEXT PRIMARY KEY,
        path        TEXT NOT NULL,
        language    TEXT NOT NULL,
        source      TEXT NOT NULL,
        tokens      INTEGER NOT NULL,
        mipmap      BLOB,            -- zlib(pickle(FileMipmap)) fast path for re-sightings / restarts
        created_at  REAL NOT NULL DEFAULT (unixepoch('subsec'))
    )""",
    """CREATE INDEX IF NOT EXISTS files_path ON files(path)""",
    # Leaf units with all four resolutions + vectors
    """CREATE TABLE IF NOT EXISTS units (
        id          INTEGER PRIMARY KEY,
        vec_id      TEXT NOT NULL UNIQUE,
        file_vec_id TEXT NOT NULL,
        path        TEXT NOT NULL,
        language    TEXT NOT NULL,
        qualname    TEXT NOT NULL,
        kind        TEXT NOT NULL,
        start_line  INTEGER NOT NULL,
        end_line    INTEGER NOT NULL,
        l0 TEXT NOT NULL, l1 TEXT NOT NULL, l2 TEXT NOT NULL, l3 TEXT NOT NULL,
        tok_l0 INTEGER NOT NULL, tok_l1 INTEGER NOT NULL, tok_l2 INTEGER NOT NULL, tok_l3 INTEGER NOT NULL,
        role        TEXT NOT NULL,
        idents      TEXT NOT NULL DEFAULT '',
        calls       TEXT NOT NULL DEFAULT '',
        emb         BLOB NOT NULL,   -- float32[dim] (full Matryoshka vector)
        mrl64       BLOB NOT NULL,   -- float32[64]  (re-normalised prefix)
        bq          BLOB NOT NULL,   -- uint8[dim/8] (1-bit sign quantisation)
        colbert     BLOB NOT NULL,   -- float16[t, token_dim]
        colbert_n   INTEGER NOT NULL,
        engine      TEXT NOT NULL,
        created_at  REAL NOT NULL DEFAULT (unixepoch('subsec'))
    )""",
    """CREATE INDEX IF NOT EXISTS units_file ON units(file_vec_id)""",
    """CREATE VIRTUAL TABLE IF NOT EXISTS units_fts USING fts5(
        path, qualname, body, tokenize = 'porter unicode61'
    )""",
    # Static reference graph (caller -> callee)
    """CREATE TABLE IF NOT EXISTS edges (
        src TEXT NOT NULL, dst TEXT NOT NULL, kind TEXT NOT NULL DEFAULT 'ref',
        PRIMARY KEY (src, dst)
    ) WITHOUT ROWID""",
    # Negative RAG: causal anti-pattern memory
    """CREATE TABLE IF NOT EXISTS negative_memory (
        id          INTEGER PRIMARY KEY,
        fingerprint TEXT NOT NULL UNIQUE,
        kind        TEXT NOT NULL,          -- exception | rollback | diff | review | note
        title       TEXT NOT NULL,
        task        TEXT NOT NULL,
        code        TEXT NOT NULL,
        error       TEXT NOT NULL,
        why         TEXT NOT NULL,
        remedy      TEXT NOT NULL,
        emb         BLOB NOT NULL,
        hits        INTEGER NOT NULL DEFAULT 0,
        created_at  REAL NOT NULL DEFAULT (unixepoch('subsec'))
    )""",
    # KV paging sessions
    """CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        state      TEXT NOT NULL,
        updated_at REAL NOT NULL DEFAULT (unixepoch('subsec'))
    )""",
    # Harvested tools + shadow contracts
    """CREATE TABLE IF NOT EXISTS contracts (
        tool_name  TEXT PRIMARY KEY,
        spec       TEXT NOT NULL,
        shadow     TEXT NOT NULL,
        code       TEXT NOT NULL,
        version    INTEGER NOT NULL DEFAULT 1,
        created_at REAL NOT NULL DEFAULT (unixepoch('subsec')),
        updated_at REAL NOT NULL DEFAULT (unixepoch('subsec'))
    )""",
    """CREATE TABLE IF NOT EXISTS drift_events (
        id         INTEGER PRIMARY KEY,
        tool_name  TEXT NOT NULL,
        report     TEXT NOT NULL,
        created_at REAL NOT NULL DEFAULT (unixepoch('subsec'))
    )""",
    # Telemetry for /v1/aegis/stats and the dashboard
    """CREATE TABLE IF NOT EXISTS telemetry (
        id                INTEGER PRIMARY KEY,
        ts                REAL NOT NULL DEFAULT (unixepoch('subsec')),
        kind              TEXT NOT NULL,
        session_id        TEXT,
        tokens_original   INTEGER,
        tokens_compressed INTEGER,
        kv_hit_rate       REAL,
        latency_ms        REAL,
        payload           TEXT
    )""",
    """CREATE INDEX IF NOT EXISTS telemetry_kind_ts ON telemetry(kind, ts)""",
]


def connect(path: str, autocheckpoint: bool = True) -> sqlite3.Connection:
    """``autocheckpoint=False`` for the serving connection: WAL checkpoints are run by the
    background writer instead of stalling a request-path COMMIT every ~1000 pages."""
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    if path != ":memory:" and not path.startswith("file::memory:"):
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        if not autocheckpoint:
            conn.execute("PRAGMA wal_autocheckpoint=0")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    with conn:
        for stmt in DDL:
            conn.execute(stmt)
        # v1 -> v2 additive columns
        for table, col, decl in (("files", "mipmap", "BLOB"), ("units", "idents", "TEXT NOT NULL DEFAULT ''"),
                                 ("units", "calls", "TEXT NOT NULL DEFAULT ''")):
            cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
            if col not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
        conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
