"""
Schema inspection, path resolution, and MySQL→SQLite conversion for BIRD queries.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ColumnInfo:
    name: str
    ctype: str
    notnull: int
    pk: int


@dataclass
class TableSchema:
    name: str
    columns: list[ColumnInfo] = field(default_factory=list)
    row_count: int = 0
    indexes: list[tuple[str, str]] = field(default_factory=list)  # (index_name, sql)


@dataclass
class DatabaseSchema:
    db_path: str
    db_name: str
    tables: dict[str, TableSchema] = field(default_factory=dict)
    table_order: list[str] = field(default_factory=list)


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> dict[str, Any]:
    if path is None:
        path = package_root() / "configs" / "default.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_bird_root(cfg: dict[str, Any]) -> Path:
    env = os.environ.get("BIRD_ROOT") or os.environ.get("SRP_BIRD_ROOT")
    if env:
        p = Path(env).resolve()
        if p.is_dir():
            return p
    base = package_root()
    for rel in cfg.get("bird_root_candidates", []):
        cand = (base / rel).resolve()
        if cand.is_dir():
            return cand
    return (base / "../Mini Dev/MINIDEV").resolve()


def bird_paths(cfg: dict[str, Any]) -> tuple[Path, Path, Path]:
    root = resolve_bird_root(cfg)
    db_dir = root / "dev_databases"
    sqlite_json = root / "mini_dev_sqlite.json"
    mysql_json = root / "mini_dev_mysql.json"
    return root, db_dir, sqlite_json if sqlite_json.exists() else mysql_json


def convert_mysql_to_sqlite(sql: str) -> str:
    """Best-effort MySQL dialect → SQLite (matches course project helper)."""
    if not isinstance(sql, str):
        return ""
    out = sql.replace("`", "")
    out = re.sub(
        r"LIMIT\s+(\d+)\s*,\s*(\d+)",
        lambda m: f"LIMIT {m.group(2)} OFFSET {m.group(1)}",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(r"\bNOW\(\)", "datetime('now')", out, flags=re.IGNORECASE)
    out = re.sub(
        r"GROUP_CONCAT\s*\((.*?)\s+SEPARATOR\s+.*?\)",
        r"GROUP_CONCAT(\1)",
        out,
        flags=re.IGNORECASE | re.DOTALL,
    )
    out = re.sub(r"\s+COLLATE\s+[A-Za-z0-9_]+", "", out, flags=re.IGNORECASE)
    return out.strip().rstrip(";").strip()


def list_sqlite_databases(db_dir: Path) -> list[str]:
    if not db_dir.is_dir():
        return []
    names = []
    for child in sorted(db_dir.iterdir()):
        if child.is_dir():
            sqlite_file = child / f"{child.name}.sqlite"
            if sqlite_file.is_file():
                names.append(child.name)
    return names


def inspect_schema(db_path: str | Path) -> DatabaseSchema:
    import sqlite3

    db_path = str(db_path)
    name = Path(db_path).parent.name
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        table_rows = cur.fetchall()
        tables: dict[str, TableSchema] = {}
        order: list[str] = []
        for row in table_rows:
            tname = row["name"]
            order.append(tname)
            cur.execute(f'PRAGMA table_info("{tname}")')
            cols = [
                ColumnInfo(
                    name=r["name"],
                    ctype=r["type"] or "TEXT",
                    notnull=r["notnull"],
                    pk=r["pk"],
                )
                for r in cur.fetchall()
            ]
            cur.execute(f'SELECT COUNT(*) AS c FROM "{tname}"')
            rc = int(cur.fetchone()["c"])
            cur.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name=? AND sql IS NOT NULL",
                (tname,),
            )
            idx = [(r["name"], r["sql"] or "") for r in cur.fetchall()]
            tables[tname] = TableSchema(
                name=tname, columns=cols, row_count=rc, indexes=idx
            )
        return DatabaseSchema(db_path=db_path, db_name=name, tables=tables, table_order=order)
    finally:
        conn.close()


def column_value_samples(
    conn, table: str, column: str, ctype: str, k: int = 8
) -> list[Any]:
    """Sample distinct non-null values for predicate generation."""
    import sqlite3

    q = f'SELECT DISTINCT "{column}" AS v FROM "{table}" WHERE "{column}" IS NOT NULL LIMIT ?'
    try:
        cur = conn.cursor()
        cur.execute(q, (k * 5,))
        rows = [r[0] for r in cur.fetchall() if r[0] is not None]
    except sqlite3.Error:
        return []
    if not rows:
        return []
    step = max(1, len(rows) // k)
    return rows[::step][:k]


def numeric_quantiles(conn, table: str, column: str) -> list[float | None]:
    import sqlite3

    try:
        cur = conn.cursor()
        cur.execute(
            f'SELECT MIN("{column}") AS mn, MAX("{column}") AS mx FROM "{table}"'
        )
        mn, mx = cur.fetchone()
        if mn is None or mx is None:
            return [None] * 5
        if mn == mx:
            return [float(mn)] * 5
        cur.execute(
            f'SELECT "{column}" AS v FROM "{table}" WHERE "{column}" IS NOT NULL ORDER BY 1'
        )
        vals = [float(r[0]) for r in cur.fetchall()]
        if not vals:
            return [None] * 5
        n = len(vals)
        qs = [0.0, 0.25, 0.5, 0.75, 1.0]
        out = []
        for q in qs:
            idx = min(n - 1, int(q * (n - 1)))
            out.append(vals[idx])
        return out
    except (sqlite3.Error, TypeError, ValueError):
        return [None] * 5


def save_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _sorted_floats(values: list[float]) -> list[float]:
    return sorted(float(v) for v in values if v is not None)


def _validate_cutoff_inputs(
    quantiles: list[float], labels: list[str]
) -> tuple[list[float], list[str]]:
    q = [float(x) for x in quantiles]
    if not q:
        raise ValueError("cutoff_quantiles must not be empty")
    if any(x <= 0.0 or x >= 1.0 for x in q):
        raise ValueError("cutoff_quantiles must be in (0, 1)")
    if any(q[i] >= q[i + 1] for i in range(len(q) - 1)):
        raise ValueError("cutoff_quantiles must be strictly increasing")
    labs = [str(x) for x in labels]
    if len(labs) != len(q) + 1:
        raise ValueError("cutoff_labels must have len(cutoff_quantiles) + 1")
    return q, labs


def _quantile(values: list[float], q: float) -> float:
    vals = _sorted_floats(values)
    if not vals:
        raise ValueError("No values to compute quantiles.")
    if len(vals) == 1:
        return vals[0]
    pos = q * (len(vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def calibrate_runtime_thresholds(
    runtimes_seconds: list[float],
    quantiles: list[float],
) -> list[float]:
    if not runtimes_seconds:
        raise ValueError("runtimes_seconds must not be empty")
    return [_quantile(runtimes_seconds, q) for q in quantiles]


def calibrate_runtime_cutoffs(
    runtimes_seconds: list[float],
    quantiles: list[float],
    labels: list[str],
) -> dict[str, Any]:
    qs, labs = _validate_cutoff_inputs(quantiles, labels)
    th = calibrate_runtime_thresholds(runtimes_seconds, qs)
    return {
        "quantiles": qs,
        "labels": labs,
        "thresholds_seconds": [float(x) for x in th],
    }


def build_runtime_cutoff_artifact(
    policy: str,
    runtimes_seconds: list[float],
    db_ids: list[str],
    quantiles: list[float],
    labels: list[str],
    min_samples_per_db: int = 20,
) -> dict[str, Any]:
    qs, labs = _validate_cutoff_inputs(quantiles, labels)
    if len(runtimes_seconds) != len(db_ids):
        raise ValueError("runtimes_seconds and db_ids must have the same length")
    global_cutoffs = calibrate_runtime_cutoffs(runtimes_seconds, qs, labs)

    thresholds_by_db: dict[str, list[float]] = {}
    if policy == "per_db_quantile":
        by_db: dict[str, list[float]] = {}
        for rt, db in zip(runtimes_seconds, db_ids):
            by_db.setdefault(str(db), []).append(float(rt))
        for db, vals in by_db.items():
            if len(vals) < int(min_samples_per_db):
                continue
            thresholds_by_db[db] = calibrate_runtime_thresholds(vals, qs)

    return {
        "version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": policy,
        "labels": labs,
        "quantiles": qs,
        "global_thresholds_seconds": global_cutoffs["thresholds_seconds"],
        "thresholds_by_db_seconds": thresholds_by_db,
        "min_samples_per_db": int(min_samples_per_db),
    }


def runtime_tier_label(
    runtime_seconds: float,
    cutoff_artifact: dict[str, Any],
    db_id: str | None = None,
) -> dict[str, Any]:
    labels = [str(x) for x in cutoff_artifact.get("labels", [])]
    global_th = [
        float(x) for x in cutoff_artifact.get("global_thresholds_seconds", [])
    ]
    if len(labels) != len(global_th) + 1:
        raise ValueError("Invalid cutoff artifact: label/threshold length mismatch.")
    thresholds = global_th
    source = "global"

    policy = str(cutoff_artifact.get("policy", "global_quantile"))
    if policy == "per_db_quantile" and db_id:
        db_map = cutoff_artifact.get("thresholds_by_db_seconds", {}) or {}
        if str(db_id) in db_map:
            thresholds = [float(x) for x in db_map[str(db_id)]]
            if len(thresholds) == len(global_th):
                source = "per_db"
            else:
                thresholds = global_th

    v = float(runtime_seconds)
    idx = 0
    while idx < len(thresholds) and v > thresholds[idx]:
        idx += 1
    return {
        "label": labels[idx],
        "label_index": idx,
        "threshold_source": source,
        "thresholds_seconds": thresholds,
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_data_dirs(root: Path, cfg: dict[str, Any]) -> dict[str, Path]:
    d = root / cfg["data_dir"]
    sub = {
        "root": d,
        "synthetic": d / cfg["synthetic_subdir"],
        "runtimes": d / cfg["runtimes_subdir"],
        "features": d / cfg["features_subdir"],
        "artifacts": root / cfg.get("artifacts_subdir", "artifacts"),
    }
    for p in sub.values():
        p.mkdir(parents=True, exist_ok=True)
    return sub
