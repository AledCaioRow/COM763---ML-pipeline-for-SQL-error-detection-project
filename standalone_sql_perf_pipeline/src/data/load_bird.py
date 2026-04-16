"""
Stage 1 (BIRD) - Load and time Mini-Dev queries against SQLite databases.
"""

import os
import re
import sqlite3
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from config import BIRD_MYSQL_JSON, BIRD_SQLITE_JSON


def convert_mysql_to_sqlite(sql):
    """
    Best-effort MySQL to SQLite syntax conversion.
    Returns converted SQL string. Not guaranteed to work for all queries.
    """
    if not isinstance(sql, str):
        return ""

    out = sql
    out = out.replace("`", "")

    # LIMIT x, y -> LIMIT y OFFSET x
    out = re.sub(
        r"LIMIT\s+(\d+)\s*,\s*(\d+)",
        lambda m: f"LIMIT {m.group(2)} OFFSET {m.group(1)}",
        out,
        flags=re.IGNORECASE,
    )

    out = re.sub(r"\bNOW\(\)", "datetime('now')", out, flags=re.IGNORECASE)

    # GROUP_CONCAT(expr SEPARATOR ',') -> GROUP_CONCAT(expr)
    out = re.sub(
        r"GROUP_CONCAT\s*\((.*?)\s+SEPARATOR\s+.*?\)",
        r"GROUP_CONCAT(\1)",
        out,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Remove collate clauses like COLLATE utf8mb4_general_ci
    out = re.sub(r"\s+COLLATE\s+[A-Za-z0-9_]+", "", out, flags=re.IGNORECASE)

    out = out.strip().rstrip(";").strip()
    return out


def _categorize_error(exc):
    msg = str(exc).lower()
    if "timeout" in msg or "interrupted" in msg:
        return "timeout"
    if "no such table" in msg or "no such column" in msg:
        return "missing object"
    if "syntax error" in msg or "near " in msg:
        return "syntax error"
    return "other"


def _read_queries(json_path):
    df = pd.read_json(json_path)
    required = {"db_id", "SQL"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required fields in {json_path}: {sorted(missing)}")

    if "question_id" not in df.columns:
        df["question_id"] = df.index
    if "difficulty" not in df.columns:
        df["difficulty"] = "unknown"

    return df[["question_id", "db_id", "SQL", "difficulty"]].copy()


def load_and_time_bird_queries(json_path, db_dir, timing_runs=3, timeout_s=30):
    """
    Load BIRD Mini-Dev queries, time each against its SQLite database.
    Returns DataFrame: question_id, db_id, sql, difficulty, runtime_s
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Query JSON not found: {json_path}")
    if not os.path.exists(db_dir):
        raise FileNotFoundError(f"Database directory not found: {db_dir}")

    use_mysql_fallback = os.path.abspath(json_path) == os.path.abspath(BIRD_MYSQL_JSON)
    source_name = "MySQL JSON (with conversion)" if use_mysql_fallback else "SQLite JSON"
    print(f"[STAGE 1] Loading BIRD queries from {source_name}: {json_path}")

    query_df = _read_queries(json_path)

    rows = []
    failures = Counter()
    failed_examples = []
    per_db_total = defaultdict(int)
    per_db_success = defaultdict(int)

    for idx, rec in query_df.iterrows():
        question_id = rec["question_id"]
        db_id = str(rec["db_id"])
        difficulty = rec["difficulty"]
        raw_sql = rec["SQL"]
        sql = convert_mysql_to_sqlite(raw_sql) if use_mysql_fallback else str(raw_sql).strip().rstrip(";")

        per_db_total[db_id] += 1
        db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            failures["missing database file"] += 1
            failed_examples.append((db_id, question_id, "missing database file"))
            continue

        timings = []
        query_failed = False
        failure_reason = None

        for _ in range(max(1, int(timing_runs))):
            conn = None
            try:
                conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
                conn.execute("PRAGMA query_only=ON;")
                try:
                    conn.execute("PRAGMA journal_mode=WAL;")
                except Exception:
                    # Safe to ignore in read-only mode on some SQLite builds.
                    pass

                deadline = time.perf_counter() + timeout_s

                def _timeout_handler():
                    if time.perf_counter() > deadline:
                        raise TimeoutError(f"query timeout > {timeout_s}s")
                    return 0

                conn.set_progress_handler(_timeout_handler, 1_000_000)
                cursor = conn.cursor()

                t0 = time.perf_counter()
                cursor.execute(sql)
                cursor.fetchall()
                t1 = time.perf_counter()
                timings.append(t1 - t0)
            except Exception as exc:
                query_failed = True
                failure_reason = f"{_categorize_error(exc)}: {str(exc)}"
                failures[_categorize_error(exc)] += 1
                failed_examples.append((db_id, question_id, failure_reason))
                break
            finally:
                if conn is not None:
                    conn.close()

        if query_failed or not timings:
            continue

        rows.append(
            {
                "question_id": question_id,
                "db_id": db_id,
                "sql": sql,
                "difficulty": difficulty,
                "runtime_s": float(np.median(timings)),
            }
        )
        per_db_success[db_id] += 1

        if (idx + 1) % 50 == 0:
            print(f"  Timed {idx + 1}/{len(query_df)} queries...")

    out_df = pd.DataFrame(
        rows,
        columns=["question_id", "db_id", "sql", "difficulty", "runtime_s"],
    )
    print(f"[STAGE 1] Timed {len(out_df)}/{len(query_df)} queries successfully")
    print(f"  Failed: {len(query_df) - len(out_df)}")
    if failures:
        print("  Failure breakdown:")
        for reason, count in failures.most_common():
            print(f"    {reason}: {count}")

    print("\n  Per-database counts:")
    for db_id in sorted(per_db_total):
        print(f"    {db_id:24s} {per_db_success[db_id]:3d}/{per_db_total[db_id]:3d}")

    if failed_examples:
        print("\n  Sample failures (up to 10):")
        for db_id, question_id, err in failed_examples[:10]:
            print(f"    db={db_id}, question_id={question_id}, error={err}")

    return out_df


def pick_bird_json_path():
    """Pick SQLite JSON if present, else fallback to MySQL JSON."""
    if os.path.exists(BIRD_SQLITE_JSON):
        return BIRD_SQLITE_JSON
    if os.path.exists(BIRD_MYSQL_JSON):
        return BIRD_MYSQL_JSON
    raise FileNotFoundError(
        "Neither mini_dev_sqlite.json nor mini_dev_mysql.json was found. "
        "Run python setup_bird.py and download missing files."
    )
