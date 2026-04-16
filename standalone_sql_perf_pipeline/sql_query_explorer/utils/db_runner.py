"""
db_runner.py
Execute a SQL query against a SQLite database file and return the
results together with the wall-clock runtime in seconds.

Database paths are resolved via config.BIRD_DB_DIR so that training,
re-timing, and the Streamlit app all hit the same files.
"""

import sqlite3
import sys
import time
from pathlib import Path

# ── path wiring ───────────────────────────────────────────────────────────────
_HERE          = Path(__file__).resolve().parent.parent   # sql_query_explorer/
_PIPELINE_ROOT = _HERE.parent                              # standalone_sql_perf_pipeline/
_PROJECT_ROOT  = _PIPELINE_ROOT.parent                     # repo root

# standalone pipeline must be first so its src/ shadows the repo root's src/
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from config import BIRD_DB_DIR  # noqa: E402

# ── supported databases ───────────────────────────────────────────────────────
DB_NAMES = ["formula_1", "financial"]

DB_PATHS = {
    db: Path(BIRD_DB_DIR) / db / f"{db}.sqlite"
    for db in DB_NAMES
}


def get_db_path(db_name: str) -> Path:
    """Return the resolved Path for a supported database name."""
    if db_name not in DB_PATHS:
        raise ValueError(
            f"Unknown database '{db_name}'. Choose from: {list(DB_PATHS)}"
        )
    path = DB_PATHS[db_name]
    if not path.exists():
        raise FileNotFoundError(f"Database file not found: {path}")
    return path


def run_query(sql: str, db_name: str) -> dict:
    """
    Execute *sql* on the named SQLite database.

    Returns
    -------
    dict with keys:
        rows        – list of tuples (the result set)
        columns     – list of column name strings
        runtime_s   – wall-clock execution time in seconds
        row_count   – number of rows returned
        error       – None on success, else the exception message string
    """
    path = get_db_path(db_name)
    result = {
        "rows": [],
        "columns": [],
        "runtime_s": 0.0,
        "row_count": 0,
        "error": None,
    }

    try:
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        t0 = time.perf_counter()
        cursor.execute(sql)
        rows = cursor.fetchall()
        result["runtime_s"] = time.perf_counter() - t0

        if rows:
            result["columns"] = list(rows[0].keys())
            result["rows"] = [tuple(r) for r in rows]
        elif cursor.description:
            result["columns"] = [d[0] for d in cursor.description]

        result["row_count"] = len(result["rows"])
        conn.close()

    except Exception as exc:
        result["error"] = str(exc)

    return result
