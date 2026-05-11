"""
history_db.py
Persist every user-submitted query in a local SQLite file so the Model
Performance page can compute regression metrics (R², MAE) over the
growing history.

Schema
------
query_history:
    id, timestamp, database, sql_text,
    runtime_s, row_count,
    predicted_runtime_s,       ← per-db regression model output
    predicted_label,           ← 'fast' | 'slow' derived from predicted_runtime_s
    actual_label,              ← 'fast' | 'slow' derived from measured runtime_s
    had_error
"""

import sqlite3
from datetime import datetime
from pathlib import Path

_HISTORY_PATH = Path(__file__).resolve().parent.parent / "query_history.db"

# Threshold used to derive actual_label from measured runtime
ACTUAL_SLOW_THRESHOLD_S = 0.050   # 50 ms


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_HISTORY_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _existing_columns(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("PRAGMA table_info(query_history)").fetchall()
    return {r["name"] for r in rows}


def init_db() -> None:
    """
    Create the history table if it does not already exist, and migrate
    any older schema by adding missing columns via ALTER TABLE.
    """
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_history (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp            TEXT    NOT NULL,
                database             TEXT    NOT NULL,
                sql_text             TEXT    NOT NULL,
                runtime_s            REAL,
                row_count            INTEGER,
                predicted_runtime_s  REAL,
                predicted_label      TEXT,
                actual_label         TEXT,
                had_error            INTEGER DEFAULT 0
            )
        """)
        conn.commit()

        # ── migrate older DBs that are missing the predicted_runtime_s column ──
        existing = _existing_columns(conn)
        if "predicted_runtime_s" not in existing:
            conn.execute(
                "ALTER TABLE query_history ADD COLUMN predicted_runtime_s REAL"
            )
            conn.commit()

        # Back-compat: older schema had probability_slow instead of predicted_label
        if "predicted_label" not in existing:
            conn.execute(
                "ALTER TABLE query_history ADD COLUMN predicted_label TEXT"
            )
            conn.commit()

        # Remove the old probability_slow column if it exists (SQLite doesn't
        # support DROP COLUMN in older versions — we just leave it and ignore it).


def record_query(
    *,
    database: str,
    sql_text: str,
    runtime_s: float | None,
    row_count: int | None,
    predicted_runtime_s: float | None,
    predicted_label: str,
    had_error: bool = False,
) -> int:
    """
    Insert one row and return the new row id.

    actual_label is derived from measured runtime_s vs ACTUAL_SLOW_THRESHOLD_S.
    """
    actual_label: str | None = None
    if runtime_s is not None and not had_error:
        actual_label = "slow" if runtime_s >= ACTUAL_SLOW_THRESHOLD_S else "fast"

    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO query_history
                (timestamp, database, sql_text, runtime_s, row_count,
                 predicted_runtime_s, predicted_label, actual_label, had_error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(timespec="seconds"),
                database,
                sql_text,
                runtime_s,
                row_count,
                predicted_runtime_s,
                predicted_label,
                actual_label,
                int(had_error),
            ),
        )
        conn.commit()
        return cur.lastrowid


def fetch_all() -> list[dict]:
    """Return all history rows as a list of dicts, newest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM query_history ORDER BY id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def fetch_for_metrics() -> list[dict]:
    """
    Return rows suitable for regression metric computation:
    both runtime_s (actual) and predicted_runtime_s must be present.
    """
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT runtime_s, predicted_runtime_s, actual_label, predicted_label
            FROM   query_history
            WHERE  had_error = 0
              AND  runtime_s IS NOT NULL
              AND  predicted_runtime_s IS NOT NULL
            ORDER BY id
            """
        ).fetchall()
    return [dict(r) for r in rows]


def clear_history() -> None:
    """Delete all rows."""
    with _connect() as conn:
        conn.execute("DELETE FROM query_history")
        conn.commit()
