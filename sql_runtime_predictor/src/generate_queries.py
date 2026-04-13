"""
Phase 1 — Synthetic SQL generator: schema-driven, diverse templates, validated on SQLite.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

# Allow `python sql_runtime_predictor/src/generate_queries.py` from repo root
_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from src.utils import (  # noqa: E402
    DatabaseSchema,
    column_value_samples,
    ensure_data_dirs,
    inspect_schema,
    list_sqlite_databases,
    load_config,
    resolve_bird_root,
    save_jsonl,
)


def _fk_pairs(conn, table: str) -> list[tuple[str, str, str]]:
    """(referenced_table, from_col, to_col) from PRAGMA foreign_key_list."""
    cur = conn.cursor()
    try:
        cur.execute(f'PRAGMA foreign_key_list("{table}")')
        rows = cur.fetchall()
    except sqlite3.Error:
        return []
    # id, seq, table, from, to, on_update, on_delete, match
    return [(r[2], r[3], r[4]) for r in rows]


def _guess_joins(conn, tables: list[str]) -> list[tuple[str, str, str, str]]:
    """
    Return list of (left_table, right_table, left_col, right_col) for chaining joins.
    Prefer foreign keys; else same-named columns (excluding rowid).
    """
    if len(tables) < 2:
        return []
    pairs = []
    for i in range(len(tables) - 1):
        a, b = tables[i], tables[i + 1]
        fks = _fk_pairs(conn, a) + _fk_pairs(conn, b)
        found = None
        for ref, fc, tc in fks:
            if ref == b and a in tables:
                found = (a, b, fc, tc)
                break
            if ref == a and b in tables:
                found = (b, a, tc, fc)
                break
        if found:
            pairs.append(found)
            continue
        # PRAGMA table_info rows are tuples; index 1 is column name.
        cols_a = {c[1] for c in conn.execute(f'PRAGMA table_info("{a}")').fetchall()}
        cols_b = {c[1] for c in conn.execute(f'PRAGMA table_info("{b}")').fetchall()}
        common = cols_a & cols_b - {"rowid"}
        if common:
            c = sorted(common, key=lambda x: (0 if x.lower().endswith("id") else 1, x))[0]
            pairs.append((a, b, c, c))
        else:
            return []
    return pairs


def _is_numeric_type(ctype: str) -> bool:
    t = (ctype or "").upper()
    return any(x in t for x in ("INT", "REAL", "FLOAT", "DOUBLE", "NUM", "DEC"))


def _quote_id(name: str) -> str:
    return f'"{name.replace(chr(34), chr(34)+chr(34))}"'


def _validate_and_run(conn: sqlite3.Connection, sql: str) -> bool:
    try:
        conn.execute(f"EXPLAIN {sql}")
    except sqlite3.Error:
        return False
    try:
        conn.execute(sql).fetchmany(500)
    except sqlite3.Error:
        return False
    return True


def _build_where_fragment(
    rng: random.Random,
    schema: DatabaseSchema,
    conn: sqlite3.Connection,
    tables: list[str],
    n_pred: int,
) -> tuple[str, dict[str, Any]]:
    parts = []
    meta = {"predicate_types": [], "n_predicates": 0}
    if n_pred <= 0:
        return "", meta

    attempts = 0
    while len(parts) < n_pred and attempts < n_pred * 8:
        attempts += 1
        t = rng.choice(tables)
        cols = [c for c in schema.tables[t].columns if c.name != "rowid"]
        if not cols:
            continue
        col = rng.choice(cols)
        cn = _quote_id(col.name)
        tn = _quote_id(t)
        ctype = col.ctype

        if _is_numeric_type(ctype):
            q = numeric_quantiles_local(conn, t, col.name)
            if q[0] is None:
                continue
            op_kind = rng.choice(["eq", "gt", "lt", "between", "in"])
            meta["predicate_types"].append(op_kind)
            if op_kind == "eq":
                parts.append(f"{tn}.{cn} = {q[2]}")
            elif op_kind == "gt":
                parts.append(f"{tn}.{cn} > {q[1]}")
            elif op_kind == "lt":
                parts.append(f"{tn}.{cn} < {q[3]}")
            elif op_kind == "between":
                parts.append(f"{tn}.{cn} BETWEEN {q[1]} AND {q[3]}")
            else:
                pool = [x for x in q if x is not None]
                if len(pool) >= 2:
                    a, b = pool[0], pool[-1]
                    parts.append(f"{tn}.{cn} IN ({a}, {b})")
        else:
            samples = column_value_samples(conn, t, col.name, ctype, k=6)
            if not samples:
                continue
            op_kind = rng.choice(["eq", "like", "in", "is_null"])
            meta["predicate_types"].append(op_kind)
            if op_kind == "eq":
                v = samples[rng.randrange(len(samples))]
                if isinstance(v, str):
                    esc = v.replace("'", "''")[:80]
                    parts.append(f"{tn}.{cn} = '{esc}'")
                else:
                    parts.append(f"{tn}.{cn} = {v}")
            elif op_kind == "like":
                v = str(samples[0])[:20].replace("'", "''")
                pref = v[: max(1, len(v) // 2)]
                parts.append(f"{tn}.{cn} LIKE '{pref}%'")
            elif op_kind == "in":
                vs = samples[:3]
                lit = ", ".join(
                    f"'{str(x).replace(chr(39), chr(39)+chr(39))[:40]}'"
                    if isinstance(x, str)
                    else str(x)
                    for x in vs
                )
                parts.append(f"{tn}.{cn} IN ({lit})")
            else:
                parts.append(f"{tn}.{cn} IS NULL")

    if not parts:
        return "", meta
    comb = " AND " if rng.random() < 0.85 else " OR "
    meta["n_predicates"] = len(parts)
    return " WHERE " + comb.join(parts), meta


def numeric_quantiles_local(conn, table: str, column: str) -> list[Any]:
    try:
        cur = conn.cursor()
        cur.execute(
            f'SELECT MIN("{column}") AS mn, MAX("{column}") AS mx FROM "{table}"'
        )
        row = cur.fetchone()
        if not row or row[0] is None:
            return [None] * 5
        mn, mx = row
        if mn == mx:
            v = float(mn) if isinstance(mn, (int, float)) else mn
            return [v, v, v, v, v]
        cur.execute(
            f'SELECT "{column}" AS v FROM "{table}" WHERE "{column}" IS NOT NULL ORDER BY 1'
        )
        vals = [r[0] for r in cur.fetchall()]
        if not vals:
            return [None] * 5
        n = len(vals)
        out = []
        for q in (0.0, 0.25, 0.5, 0.75, 1.0):
            idx = min(n - 1, int(q * (n - 1)))
            out.append(vals[idx])
        return out
    except sqlite3.Error:
        return [None] * 5


def _select_clause(
    rng: random.Random, schema: DatabaseSchema, tables: list[str], mode: str
) -> tuple[str, dict[str, Any]]:
    meta: dict[str, Any] = {"select_mode": mode}
    if mode == "star":
        return "*", meta
    if mode == "one":
        t = rng.choice(tables)
        c = rng.choice(schema.tables[t].columns).name
        return f'{_quote_id(t)}.{_quote_id(c)}', meta
    cols_out = []
    for _ in range(rng.choice([3, 5])):
        t = rng.choice(tables)
        c = rng.choice(schema.tables[t].columns).name
        cols_out.append(f'{_quote_id(t)}.{_quote_id(c)}')
    meta["n_select_cols"] = len(cols_out)
    return ", ".join(cols_out), meta


def _aggregation_clause(rng: random.Random, tables: list[str], schema: DatabaseSchema):
    """Return (select_inner, group_by_sql, having_sql, meta)."""
    meta: dict[str, Any] = {
        "aggregation": "none",
        "group_by_cols": 0,
        "has_having": False,
    }
    if rng.random() > 0.45:
        return None, "", "", meta
    t = rng.choice(tables)
    cols = schema.tables[t].columns
    num_cols = [c for c in cols if _is_numeric_type(c.ctype)]
    any_cols = cols[:]
    agg = rng.choice(["COUNT", "SUM", "AVG", "MIN", "MAX"])
    meta["aggregation"] = agg.lower()
    if agg == "COUNT" and rng.random() < 0.3:
        c = rng.choice(any_cols).name
        sel = f"COUNT(DISTINCT {_quote_id(t)}.{_quote_id(c)})"
    elif agg in ("SUM", "AVG") and num_cols:
        c = rng.choice(num_cols).name
        sel = f"{agg}({_quote_id(t)}.{_quote_id(c)})"
    else:
        c = rng.choice(any_cols).name
        sel = f"{agg}({_quote_id(t)}.{_quote_id(c)})"

    gb = ""
    meta["group_by_cols"] = 0
    if rng.random() < 0.7:
        gc = rng.choice(any_cols).name
        gb = f" GROUP BY {_quote_id(t)}.{_quote_id(gc)}"
        meta["group_by_cols"] = 1
        if rng.random() < 0.25 and num_cols:
            sel = f"{sel}, {_quote_id(t)}.{_quote_id(rng.choice(num_cols).name)}"

    having = ""
    if rng.random() < 0.2 and num_cols:
        hc = rng.choice(num_cols).name
        having = f" HAVING {_quote_id(t)}.{_quote_id(hc)} > 0"
        meta["has_having"] = True

    return sel, gb, having, meta


def generate_one_query(
    rng: random.Random,
    schema: DatabaseSchema,
    conn: sqlite3.Connection,
) -> tuple[str | None, dict[str, Any]]:
    n_tables = rng.choice([1, 2, 3, 4])
    n_tables = min(n_tables, len(schema.table_order))
    tables = rng.sample(schema.table_order, n_tables)

    join_meta: dict[str, Any] = {
        "num_tables": len(tables),
        "tables": tables,
        "join_style": "single",
    }

    from_sql = ""
    if len(tables) == 1:
        from_sql = f'FROM {_quote_id(tables[0])} '
        join_meta["join_style"] = "single"
    else:
        jt = rng.choice(["inner", "left", "cross_implicit"])
        jp = _guess_joins(conn, tables)
        if not jp and jt != "cross_implicit":
            return None, {}
        if jt == "cross_implicit":
            join_meta["join_style"] = "cross"
            from_sql = "FROM " + ", ".join(_quote_id(t) for t in tables) + " "
        elif jt == "inner":
            join_meta["join_style"] = "inner"
            if jp:
                lt, rt, lc, rc = jp[0]
                from_sql = f'FROM {_quote_id(lt)} INNER JOIN {_quote_id(rt)} ON {_quote_id(lt)}.{_quote_id(lc)} = {_quote_id(rt)}.{_quote_id(rc)} '
                rest = tables[2:]
                cur = rt
                for tnext in rest:
                    sub = _guess_joins(conn, [cur, tnext])
                    if not sub:
                        return None, {}
                    lt, rt, lc, rc = sub[0]
                    from_sql += f'INNER JOIN {_quote_id(rt)} ON {_quote_id(lt)}.{_quote_id(lc)} = {_quote_id(rt)}.{_quote_id(rc)} '
                    cur = rt
            else:
                return None, {}
        else:
            join_meta["join_style"] = "left"
            if not jp:
                return None, {}
            lt, rt, lc, rc = jp[0]
            from_sql = f'FROM {_quote_id(lt)} LEFT JOIN {_quote_id(rt)} ON {_quote_id(lt)}.{_quote_id(lc)} = {_quote_id(rt)}.{_quote_id(rc)} '
            cur = rt
            for tnext in tables[2:]:
                sub = _guess_joins(conn, [cur, tnext])
                if not sub:
                    return None, {}
                lt, rt, lc, rc = sub[0]
                from_sql += f'LEFT JOIN {_quote_id(rt)} ON {_quote_id(lt)}.{_quote_id(lc)} = {_quote_id(rt)}.{_quote_id(rc)} '
                cur = rt

    n_pred = rng.choice([0, 1, 2, 3, 4])
    where_sql, pred_meta = _build_where_fragment(rng, schema, conn, tables, n_pred)

    agg_sel, gb, having, agg_meta = _aggregation_clause(rng, tables, schema)

    sel_mode = rng.choice(["star", "one", "multi"])
    distinct = rng.random() < 0.15
    meta = {**join_meta, **pred_meta, **agg_meta}
    meta["distinct"] = distinct
    meta["select_mode"] = sel_mode

    if agg_sel:
        select_core = agg_sel
    else:
        sel_list, sm = _select_clause(rng, schema, tables, sel_mode)
        meta.update(sm)
        select_core = sel_list

    prefix = "SELECT DISTINCT " if distinct else "SELECT "
    sql = prefix + select_core + " " + from_sql + where_sql + gb + having

    ob = rng.choice([None, "one", "multi"])
    meta["order_by"] = ob
    if ob and not agg_sel:
        t = rng.choice(tables)
        c = rng.choice(schema.tables[t].columns).name
        dire = rng.choice(["ASC", "DESC"])
        sql += f' ORDER BY {_quote_id(t)}.{_quote_id(c)} {dire}'
        if ob == "multi":
            t2 = rng.choice(tables)
            c2 = rng.choice(schema.tables[t2].columns).name
            sql += f', {_quote_id(t2)}.{_quote_id(c2)} ASC'

    lim = rng.choice([None, 10, 100, 1000])
    meta["limit"] = lim
    if lim is not None:
        sql += f" LIMIT {lim}"

    # Subquery wrapper (scalar in WHERE)
    if rng.random() < 0.12 and len(tables) >= 1:
        t = tables[0]
        num_cols = [c for c in schema.tables[t].columns if _is_numeric_type(c.ctype)]
        if num_cols:
            c = rng.choice(num_cols).name
            sub = f'(SELECT MAX({_quote_id(c)}) FROM {_quote_id(t)})'
            if "WHERE" in sql:
                sql += f' AND {_quote_id(t)}.{_quote_id(c)} < {sub}'
            else:
                sql += f' WHERE {_quote_id(t)}.{_quote_id(c)} < {sub}'
            meta["has_subquery"] = True
        else:
            meta["has_subquery"] = False
    else:
        meta["has_subquery"] = False

    # Set ops: duplicate FROM body simplified
    if rng.random() < 0.06 and len(tables) == 1:
        op = rng.choice(["UNION", "INTERSECT", "EXCEPT"])
        sql = f"{sql} {op} {sql}"
        meta["set_op"] = op.lower()
    else:
        meta["set_op"] = None

    sql = re.sub(r"\s+", " ", sql).strip()
    if _validate_and_run(conn, sql):
        return sql, meta
    return None, {}


def generate_for_database(
    db_path: Path,
    target_count: int,
    seed: int,
    max_attempts: int,
) -> list[dict[str, Any]]:
    schema = inspect_schema(db_path)
    rng = random.Random(seed)
    conn = sqlite3.connect(str(db_path))
    try:
        out: list[dict[str, Any]] = []
        attempts = 0
        while len(out) < target_count and attempts < max_attempts:
            attempts += 1
            sql, meta = generate_one_query(rng, schema, conn)
            if sql:
                out.append(
                    {
                        "query_id": f"synth_{schema.db_name}_{len(out):05d}",
                        "query_text": sql,
                        "database": schema.db_name,
                        "template_metadata": meta,
                    }
                )
        return out
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic SQL for BIRD SQLite DBs.")
    parser.add_argument("--per-db", type=int, default=None, help="Queries per database")
    parser.add_argument("--db", type=str, default=None, help="Single database id")
    args = parser.parse_args()

    cfg = load_config()
    dirs = ensure_data_dirs(_PKG, cfg)
    bird_root = resolve_bird_root(cfg)
    db_dir = bird_root / "dev_databases"
    dbs = list_sqlite_databases(db_dir)
    if not dbs:
        print(f"No SQLite databases under {db_dir}. Download BIRD Mini-Dev .sqlite files.")
        sys.exit(1)
    if args.db:
        dbs = [args.db] if args.db in dbs else []
        if not dbs:
            print(f"Unknown db {args.db}")
            sys.exit(1)

    per = args.per_db or min(
        cfg["queries_per_db_max"],
        max(cfg["queries_per_db_min"], 500),
    )
    max_att = cfg.get("max_generation_attempts_per_db", per * 3)
    seed = int(cfg.get("random_seed", 42))

    for i, db_name in enumerate(dbs):
        db_path = db_dir / db_name / f"{db_name}.sqlite"
        print(f"Generating ~{per} queries for {db_name} ...")
        records = generate_for_database(
            db_path, per, seed + i * 10_000, max_att
        )
        out_path = dirs["synthetic"] / f"{db_name}.jsonl"
        save_jsonl(out_path, records)
        print(f"  wrote {len(records)} -> {out_path}")


if __name__ == "__main__":
    main()
