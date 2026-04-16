"""
Phase 3 — EXPLAIN QUERY PLAN → tree + per-node and global features (JSON for training).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from src.utils import (  # noqa: E402
    DatabaseSchema,
    ensure_data_dirs,
    inspect_schema,
    load_config,
    resolve_bird_root,
    load_jsonl,
    save_jsonl,
    convert_mysql_to_sqlite,
)

# Fixed vocabularies for one-hot blocks (order matters)
OPERATOR_TYPES = [
    "ROOT",
    "SCAN",
    "SEARCH",
    "SORT",
    "USE_TEMP",
    "COMPOUND",
    "SUBQUERY",
    "COROUTINE",
    "LIST",
    "OTHER",
]
ACCESS_TYPES = [
    "NONE",
    "INDEX",
    "COVERING_INDEX",
    "INTEGER_PK",
    "SCAN_TABLE",
]

NODE_FEATURE_DIM = len(OPERATOR_TYPES) + len(ACCESS_TYPES) + 8
# +8: table_norm, log_rows_norm, col_count_norm, cols_ref_norm, has_idx_pred,
#     selectivity, pred_count_norm, bias_1

GLOBAL_FEATURE_DIM = 16


def _one_hot(idx: int, n: int) -> list[float]:
    return [1.0 if i == idx else 0.0 for i in range(n)]


def _classify_operator(detail: str) -> int:
    u = detail.upper()
    if "COMPOUND" in u:
        return OPERATOR_TYPES.index("COMPOUND")
    if "COROUTINE" in u or "CO-ROUTINE" in u:
        return OPERATOR_TYPES.index("COROUTINE")
    if "SUBQUERY" in u:
        return OPERATOR_TYPES.index("SUBQUERY")
    if "SCAN" in u and "SEARCH" not in u:
        return OPERATOR_TYPES.index("SCAN")
    if "SEARCH" in u:
        return OPERATOR_TYPES.index("SEARCH")
    if "TEMP" in u or "SORT" in u or "ORDER" in u:
        return OPERATOR_TYPES.index("SORT" if "ORDER" in u else "USE_TEMP")
    if "LIST" in u:
        return OPERATOR_TYPES.index("LIST")
    return OPERATOR_TYPES.index("OTHER")


def _classify_access(detail: str) -> int:
    u = detail.upper()
    if "COVERING INDEX" in u or "USING COVERING INDEX" in u:
        return ACCESS_TYPES.index("COVERING_INDEX")
    if "USING INDEX" in u or "USING INTEGER PRIMARY KEY" in u:
        if "INTEGER PRIMARY KEY" in u:
            return ACCESS_TYPES.index("INTEGER_PK")
        return ACCESS_TYPES.index("INDEX")
    if "SCAN" in u:
        return ACCESS_TYPES.index("SCAN_TABLE")
    return ACCESS_TYPES.index("NONE")


def _table_from_detail(detail: str, schema: DatabaseSchema) -> str | None:
    # "SCAN t" / "SEARCH table USING ..."
    for t in schema.table_order:
        if re.search(rf"\b{re.escape(t)}\b", detail, re.IGNORECASE):
            return t
    return None


def _index_covers_column(index_sql: str, col: str) -> bool:
    if not index_sql:
        return False
    return bool(re.search(rf"\b{re.escape(col)}\b", index_sql, re.IGNORECASE))


def _predicate_columns_simple(sql: str, table: str) -> tuple[int, int]:
    """Rough count of predicates and columns from table in WHERE (heuristic)."""
    u = sql.upper()
    if "WHERE" not in u:
        return 0, 0
    where_part = u.split("WHERE", 1)[1]
    for stop in ["GROUP BY", "ORDER BY", "LIMIT", "HAVING", "EXCEPT", "UNION"]:
        if stop in where_part:
            where_part = where_part.split(stop)[0]
    tn = table.upper()
    preds = max(1, where_part.count(" AND ") + where_part.count(" OR "))
    if tn not in where_part and f'"{table.upper()}"' not in where_part:
        # try unqualified
        cols_mentioned = min(5, preds)
        return preds, cols_mentioned
    return preds, min(preds, 8)


def _cols_referenced_in_query(sql: str, table: str, schema: DatabaseSchema) -> int:
    count = 0
    tinfo = schema.tables.get(table)
    if not tinfo:
        return 0
    low = sql.lower()
    for c in tinfo.columns:
        pat = rf'\b{re.escape(table)}\s*\.\s*{re.escape(c.name)}\b'
        if re.search(pat, sql, re.IGNORECASE):
            count += 1
        elif re.search(rf'[,\s\(]{re.escape(c.name)}\b', low):
            count += 1
    return min(count, 20)


def _root_feature_vector_global(max_tables: int) -> list[float]:
    z = _one_hot(OPERATOR_TYPES.index("ROOT"), len(OPERATOR_TYPES))
    ac = _one_hot(ACCESS_TYPES.index("NONE"), len(ACCESS_TYPES))
    rest = [0.0, math.log1p(max_tables * 100) / 25.0, 0.0, 0.0, 0.0, 0.1, 0.0, 1.0]
    return z + ac + rest


def _has_index_on_predicates(
    schema: DatabaseSchema, table: str, sql: str
) -> float:
    tinfo = schema.tables.get(table)
    if not tinfo:
        return 0.0
    u = sql.upper()
    if "WHERE" not in u:
        return 0.0
    where_part = u.split("WHERE", 1)[1]
    for stop in ["GROUP BY", "ORDER BY", "LIMIT", "HAVING"]:
        if stop in where_part:
            where_part = where_part.split(stop)[0]
    for col in tinfo.columns:
        if col.name.upper() not in where_part:
            continue
        for _, idx_sql in tinfo.indexes:
            if _index_covers_column(idx_sql, col.name):
                return 1.0
    return 0.0


def _build_plan_tree(
    rows: list[tuple[Any, ...]],
    schema: DatabaseSchema,
    query: str,
    max_tables: int,
) -> dict[str, Any]:
    """Parse SQLite EXPLAIN QUERY PLAN rows into nested dict."""
    if not rows:
        return {
            "node_type": "ROOT",
            "detail": "",
            "features": _root_feature_vector_global(max_tables),
            "children": [],
        }

    # Columns are id, parent, notused, detail (detail is always last).
    parsed = []
    for r in rows:
        nid = int(r[0])
        pid = r[1]
        detail = str(r[-1])
        parsed.append(
            {"id": nid, "parent": None if pid is None else int(pid), "detail": detail}
        )

    by_id = {p["id"]: p for p in parsed}
    ids_set = set(by_id.keys())
    children_map: dict[int, list[int]] = {}
    for p in parsed:
        par = p["parent"]
        if par is None:
            parent_key = -1
        elif par not in ids_set and par == 0:
            parent_key = -1
        else:
            parent_key = par
        children_map.setdefault(parent_key, []).append(p["id"])

    sql_roots = sorted(children_map.get(-1, []))
    if not sql_roots:
        sql_roots = [min(by_id.keys())]

    def node_features(detail: str, table: str | None) -> list[float]:
        opi = _classify_operator(detail)
        aci = _classify_access(detail)
        op = _one_hot(opi, len(OPERATOR_TYPES))
        ac = _one_hot(aci, len(ACCESS_TYPES))
        tnorm = 0.0
        log_rows = 0.0
        ncol = 0.0
        cref = 0.0
        hidx = 0.0
        sel = 0.1
        pcount = 0.0
        if table and table in schema.tables:
            ti = schema.tables[table]
            order = schema.table_order
            tnorm = (order.index(table) + 1) / max(len(order), 1)
            log_rows = math.log1p(max(ti.row_count, 0)) / 25.0
            ncol = min(len(ti.columns), 100) / 100.0
            cref = _cols_referenced_in_query(query, table, schema) / 20.0
            hidx = _has_index_on_predicates(schema, table, query)
            np, _ = _predicate_columns_simple(query, table)
            pcount = min(np, 20) / 20.0
            sel = min(1.0, 10.0 / max(ti.row_count, 1))
        rest = [tnorm, log_rows, ncol, cref, hidx, sel, pcount, 1.0]
        return op + ac + rest

    def build(nid: int) -> dict[str, Any]:
        p = by_id[nid]
        detail = p["detail"]
        table = _table_from_detail(detail, schema)
        feats = node_features(detail, table)
        child_ids = sorted(children_map.get(nid, []))
        ch_nodes = [build(cid) for cid in child_ids]
        return {
            "node_type": OPERATOR_TYPES[_classify_operator(detail)],
            "detail": detail,
            "table": table,
            "features": feats,
            "children": ch_nodes,
        }

    top_trees = [build(r) for r in sql_roots]
    return {
        "node_type": "ROOT",
        "detail": "",
        "features": _root_feature_vector_global(max_tables),
        "children": top_trees,
    }


def _fix_tree_features(tree: dict, max_tables: int) -> None:
    if tree.get("node_type") == "ROOT" and not tree.get("detail"):
        tree["features"] = _root_feature_vector_global(max_tables)


def global_features(sql: str, schema: DatabaseSchema) -> list[float]:
    u = sql.upper()
    joins = u.count(" JOIN ")
    from_commas = u.count(" FROM ")
    total_tables = min(10, max(1, joins + 1 if joins else from_commas))
    pred = 0
    if "WHERE" in u:
        wp = u.split("WHERE", 1)[1]
        for stop in ["GROUP BY", "ORDER BY", "LIMIT", "HAVING"]:
            if stop in wp:
                wp = wp.split(stop)[0]
        pred = 1 + wp.count(" AND ") + wp.count(" OR ")
    pred = min(pred, 20)

    has_gb = 1.0 if "GROUP BY" in u else 0.0
    has_ob = 1.0 if "ORDER BY" in u else 0.0
    has_dist = 1.0 if "DISTINCT" in u else 0.0
    has_lim = 1.0 if "LIMIT" in u else 0.0
    has_sub = 1.0 if u.count("SELECT") > 1 else 0.0

    agg_kind = [0.0] * 6  # none, count, sum, avg, min, max
    has_agg = 0.0
    if "COUNT(" in u:
        agg_kind[1] = 1.0
        has_agg = 1.0
    elif "SUM(" in u:
        agg_kind[2] = 1.0
        has_agg = 1.0
    elif "AVG(" in u:
        agg_kind[3] = 1.0
        has_agg = 1.0
    elif "MIN(" in u:
        agg_kind[4] = 1.0
        has_agg = 1.0
    elif "MAX(" in u:
        agg_kind[5] = 1.0
        has_agg = 1.0
    else:
        agg_kind[0] = 1.0

    ref_tables = re.findall(
        r"FROM\s+([A-Za-z_][A-Za-z0-9_]*)", sql, re.IGNORECASE
    )
    ref_tables += re.findall(
        r"JOIN\s+([A-Za-z_][A-Za-z0-9_]*)", sql, re.IGNORECASE
    )
    est_rows = 0
    for t in set(ref_tables):
        if t in schema.tables:
            est_rows += schema.tables[t].row_count

    vec = [
        total_tables / 10.0,
        pred / 20.0,
        has_gb,
        has_ob,
        has_dist,
        has_lim,
        has_sub,
        has_agg,
        len(sql) / 5000.0,
        math.log1p(est_rows) / 30.0,
    ] + agg_kind
    assert len(vec) == GLOBAL_FEATURE_DIM, len(vec)
    return vec


def explain_rows(conn: sqlite3.Connection, query: str) -> list[tuple]:
    cur = conn.cursor()
    cur.execute(f"EXPLAIN QUERY PLAN {query}")
    return cur.fetchall()


def extract_one(
    db_path: Path,
    query: str,
    query_id: str,
    target_runtime: float | None,
    use_mysql_convert: bool = False,
) -> dict[str, Any] | None:
    if use_mysql_convert:
        query = convert_mysql_to_sqlite(query)
    schema = inspect_schema(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        try:
            rows = explain_rows(conn, query)
        except sqlite3.Error:
            return None
        max_t = len(schema.table_order)
        tree = _build_plan_tree(rows, schema, query, max_t)
        _fix_tree_features(tree, max_t)
        g = global_features(query, schema)
        return {
            "query_id": query_id,
            "plan_tree": tree,
            "global_features": g,
            "target_runtime": target_runtime,
            "node_feature_dim": NODE_FEATURE_DIM,
            "global_feature_dim": GLOBAL_FEATURE_DIM,
        }
    finally:
        conn.close()


def process_runtime_jsonl(path: Path, db_dir: Path) -> list[dict]:
    out = []
    for rec in load_jsonl(path):
        if rec.get("median_runtime_seconds") is None:
            continue
        db_name = rec["database"]
        db_path = db_dir / db_name / f"{db_name}.sqlite"
        if not db_path.is_file():
            continue
        feat = extract_one(
            db_path,
            rec["query_text"],
            rec["query_id"],
            float(rec["median_runtime_seconds"]),
        )
        if feat:
            out.append(feat)
    return out


def process_bird_json(json_path: Path, db_dir: Path) -> list[dict]:
    import pandas as pd

    df = pd.read_json(json_path)
    use_conv = "mysql" in json_path.name.lower()
    out = []
    for _, row in df.iterrows():
        db_name = str(row["db_id"])
        db_path = db_dir / db_name / f"{db_name}.sqlite"
        if not db_path.is_file():
            continue
        qid = str(row.get("question_id", row.name))
        sql = str(row["SQL"])
        feat = extract_one(
            db_path,
            sql,
            f"bird_{qid}",
            None,
            use_mysql_convert=use_conv,
        )
        if feat:
            out.append(feat)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract plan trees + features.")
    parser.add_argument(
        "--bird-only",
        action="store_true",
        help="Only process BIRD dev JSON (for test features)",
    )
    args = parser.parse_args()

    cfg = load_config()
    dirs = ensure_data_dirs(_PKG, cfg)
    bird_root = resolve_bird_root(cfg)
    db_dir = bird_root / "dev_databases"
    synth_run_dir = dirs["runtimes"]
    feat_dir = dirs["features"]

    if args.bird_only:
        jq = bird_root / "mini_dev_sqlite.json"
        if not jq.is_file():
            jq = bird_root / "mini_dev_mysql.json"
        if not jq.is_file():
            print("No mini_dev_sqlite.json or mini_dev_mysql.json")
            sys.exit(1)
        feats = process_bird_json(jq, db_dir)
        outp = feat_dir / "bird_dev_features.jsonl"
        save_jsonl(outp, feats)
        print(f"Wrote {len(feats)} rows -> {outp}")
        return

    files = sorted(synth_run_dir.glob("*.jsonl"))
    if not files:
        print(f"No timed jsonl in {synth_run_dir}. Run collect_runtimes first.")
        sys.exit(1)

    all_train: list[dict] = []
    for fp in files:
        part = process_runtime_jsonl(fp, db_dir)
        all_train.extend(part)
        save_jsonl(feat_dir / f"train_{fp.stem}.jsonl", part)
        print(f"{fp.name}: {len(part)} feature rows")

    save_jsonl(feat_dir / "train_all.jsonl", all_train)
    print(f"Total training features: {len(all_train)}")

    jq = bird_root / "mini_dev_sqlite.json"
    if not jq.is_file():
        jq = bird_root / "mini_dev_mysql.json"
    if jq.is_file():
        bird_feats = process_bird_json(jq, db_dir)
        save_jsonl(feat_dir / "bird_dev_features.jsonl", bird_feats)
        print(f"BIRD dev features: {len(bird_feats)}")


if __name__ == "__main__":
    main()
