"""
src/schema_stats.py
-------------------
Single source of truth for extracting schema-level statistics from a
SQLite database file.

Used by:
  • run_schema_stats_model.py  (training / evaluation)
  • sql_query_explorer/utils/predictor.py  (Streamlit inference)
  • sql_query_explorer/utils/db_runner.py  (path resolution)

The database path is always resolved via config.BIRD_DB_DIR so that
training and inference hit the same files.
"""

import sqlite3
import sys
from pathlib import Path
import numpy as np

# ── make config importable when this module is imported from the app ──────────
_SRC_DIR      = Path(__file__).resolve().parent          # standalone_sql_perf_pipeline/src/
_PIPELINE_ROOT = _SRC_DIR.parent                          # standalone_sql_perf_pipeline/
_PROJECT_ROOT  = _PIPELINE_ROOT.parent                    # repo root

for _p in [str(_PIPELINE_ROOT), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import BIRD_DB_DIR  # noqa: E402


def db_path(db_id: str) -> Path:
    """Return the resolved Path to <db_id>/<db_id>.sqlite inside BIRD_DB_DIR."""
    return Path(BIRD_DB_DIR) / db_id / f"{db_id}.sqlite"


def schema_stats(db_id: str) -> dict:
    """
    Compute the 6 schema-level statistics for *db_id*.

    Returns an empty dict if the database file cannot be found.

    Keys returned
    -------------
    schema_n_tables, schema_total_rows, schema_max_table_rows,
    schema_total_indexes, schema_index_coverage, schema_log_total_rows
    """
    path = db_path(db_id)
    if not path.exists():
        return {}

    conn = sqlite3.connect(str(path))
    cur  = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]

    total_rows = max_rows = total_idx = with_idx = 0
    for t in tables:
        try:
            cur.execute(f'SELECT COUNT(*) FROM "{t}"')
            rc = cur.fetchone()[0]
            total_rows += rc
            max_rows    = max(max_rows, rc)
            cur.execute(f'PRAGMA index_list("{t}")')
            idxs = cur.fetchall()
            total_idx += len(idxs)
            if idxs:
                with_idx += 1
        except Exception:
            pass
    conn.close()

    n = len(tables)
    return {
        "schema_n_tables":        n,
        "schema_total_rows":      total_rows,
        "schema_max_table_rows":  max_rows,
        "schema_total_indexes":   total_idx,
        "schema_index_coverage":  (with_idx / n) if n else 0.0,
        "schema_log_total_rows":  float(np.log1p(total_rows)),
    }
