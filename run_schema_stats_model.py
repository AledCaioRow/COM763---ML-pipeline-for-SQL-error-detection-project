# -*- coding: utf-8 -*-
"""
Within-database regression with schema statistics.

For each database that has enough queries, trains a regression model on 80%
of that database's queries and tests on the remaining 20%.

Schema statistics (table row counts, index coverage) are added alongside the
25 SQL structural features. Because we train and test within the same database,
schema stats are constant across all rows -- but they act as an anchor that
tells the model the scale of the environment it is working in.

Models compared:
  Ridge (alpha=1), Ridge (alpha=10), Lasso, Random Forest, GBM
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sqlite3
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = r"C:\Users\aled_\Downloads\COM763---ML-pipeline-for-SQL-error-detection-project"
DATA_CSV     = os.path.join(PROJECT_ROOT, "data", "query_dataset_features.csv")
DB_BASE      = os.path.join(PROJECT_ROOT, "Mini Dev", "MINIDEV", "dev_databases")
SEED         = 42
MIN_QUERIES  = 15   # skip databases too small to split meaningfully

SQL_FEATURES = [
    "n_tokens", "query_length", "n_joins", "n_tables_approx",
    "n_where_predicates", "has_group_by", "has_order_by", "has_having",
    "has_distinct", "has_limit", "has_union", "n_subqueries", "has_subquery",
    "max_nesting_depth", "n_count", "n_sum", "n_avg", "n_max", "n_min",
    "n_aggregations", "has_between", "has_in_clause", "has_like",
    "has_exists", "has_correlated_subquery",
]

SCHEMA_FEATURES = [
    "schema_n_tables", "schema_total_rows", "schema_max_table_rows",
    "schema_total_indexes", "schema_index_coverage", "schema_log_total_rows",
]

FULL_FEATURES = SQL_FEATURES + SCHEMA_FEATURES

# ---------------------------------------------------------------------------
# Extract schema statistics from each SQLite database
# ---------------------------------------------------------------------------
def schema_stats(db_id):
    path = os.path.join(DB_BASE, db_id, f"{db_id}.sqlite")
    if not os.path.exists(path):
        return {}
    conn = sqlite3.connect(path)
    cur  = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    total_rows, max_rows, total_idx, with_idx = 0, 0, 0, 0
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
        "schema_n_tables":       n,
        "schema_total_rows":     total_rows,
        "schema_max_table_rows": max_rows,
        "schema_total_indexes":  total_idx,
        "schema_index_coverage": (with_idx / n) if n else 0.0,
        "schema_log_total_rows": np.log1p(total_rows),
    }

# ---------------------------------------------------------------------------
# Load data and attach schema stats
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_CSV)
df = df[df["runtime_s"] > 0].copy()
df["log_runtime"] = np.log(df["runtime_s"])

schema_rows = []
for db_id in df["db_id"].unique():
    s = schema_stats(db_id)
    if s:
        s["db_id"] = db_id
        schema_rows.append(s)
schema_df = pd.DataFrame(schema_rows)
df = df.merge(schema_df, on="db_id", how="left")

# ---------------------------------------------------------------------------
# Models to test
# ---------------------------------------------------------------------------
def make_models():
    return {
        "Ridge(a=1)":  Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        "Ridge(a=10)": Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=10.0))]),
        "Lasso":       Pipeline([("sc", StandardScaler()), ("m", Lasso(alpha=0.01, max_iter=5000))]),
        "RF":          Pipeline([("sc", StandardScaler()), ("m", RandomForestRegressor(n_estimators=100, random_state=SEED))]),
        "GBM":         Pipeline([("sc", StandardScaler()), ("m", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=SEED))]),
    }

# ---------------------------------------------------------------------------
# Per-database within-split evaluation
# ---------------------------------------------------------------------------
results = []

print("=" * 70)
print("Within-database regression  (80% train / 20% test per database)")
print("Features: 25 SQL structural + 6 schema statistics")
print("=" * 70)

for db_id, grp in df.groupby("db_id"):
    grp = grp.copy()
    n   = len(grp)
    if n < MIN_QUERIES:
        print(f"\n  {db_id} (n={n}) -- skipped, fewer than {MIN_QUERIES} queries")
        continue

    X = grp[FULL_FEATURES]
    y = grp["log_runtime"]
    y_s = np.exp(y.values)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    y_te_s = np.exp(y_te.values)

    n_tr  = len(X_tr)
    n_te  = len(X_te)
    r_min = grp["runtime_s"].min()
    r_max = grp["runtime_s"].max()
    r_mean= grp["runtime_s"].mean()

    st = schema_stats(db_id)
    slow_pct = int(100 * grp["label_binary"].mean())

    print(f"\n  {db_id}")
    print(f"    n={n}  train={n_tr}  test={n_te}  slow%={slow_pct}%")
    print(f"    runtime: {r_min:.6f}s - {r_max:.6f}s  mean={r_mean:.6f}s")
    print(f"    schema:  total_rows={st['schema_total_rows']:,}  "
          f"max_table={st['schema_max_table_rows']:,}  "
          f"idx_cov={st['schema_index_coverage']:.2f}")
    print(f"    {'Model':<12}  {'MAE(log)':>9}  {'R2(log)':>8}  {'MAE(s)':>10}  {'R2(s)':>8}")
    print(f"    {'-'*12}  {'-'*9}  {'-'*8}  {'-'*10}  {'-'*8}")

    best_r2 = -999
    best_name = ""

    for name, m in make_models().items():
        m.fit(X_tr, y_tr)
        p_log = m.predict(X_te)
        p_s   = np.exp(p_log)

        mae_log  = mean_absolute_error(y_te.values, p_log)
        r2_log   = r2_score(y_te.values, p_log)
        mae_s    = mean_absolute_error(y_te_s, p_s)
        r2_s     = r2_score(y_te_s, p_s)

        marker = ""
        if r2_log > best_r2:
            best_r2   = r2_log
            best_name = name

        print(f"    {name:<12}  {mae_log:>9.4f}  {r2_log:>8.4f}  {mae_s:>10.6f}  {r2_s:>8.4f}")

        results.append({
            "db_id":     db_id,
            "n":         n,
            "n_train":   n_tr,
            "n_test":    n_te,
            "slow_pct":  slow_pct,
            "total_rows": st["schema_total_rows"],
            "max_table_rows": st["schema_max_table_rows"],
            "index_coverage": st["schema_index_coverage"],
            "model":     name,
            "mae_log":   round(mae_log, 4),
            "r2_log":    round(r2_log,  4),
            "mae_s":     round(mae_s,   6),
            "r2_s":      round(r2_s,    4),
        })

    print(f"    --> best: {best_name}  R2(log)={best_r2:.4f}")

# ---------------------------------------------------------------------------
# Summary table -- best model per database
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("SUMMARY -- best model per database (by R2 log)")
print("=" * 70)
res_df = pd.DataFrame(results)
best_per_db = (
    res_df.sort_values("r2_log", ascending=False)
          .groupby("db_id", sort=False)
          .first()
          .reset_index()
)
best_per_db = best_per_db.sort_values("r2_log", ascending=False)

print(f"\n  {'Database':<25} {'n':>4} {'slow%':>6} {'total_rows':>12} {'idx_cov':>8} "
      f"{'Best model':<12} {'R2(log)':>8} {'MAE(log)':>9} {'MAE(s)':>10}")
print(f"  {'-'*25} {'-'*4} {'-'*6} {'-'*12} {'-'*8} {'-'*12} {'-'*8} {'-'*9} {'-'*10}")
for _, row in best_per_db.iterrows():
    print(f"  {row['db_id']:<25} {int(row['n']):>4} {int(row['slow_pct']):>5}% "
          f"{int(row['total_rows']):>12,} {row['index_coverage']:>8.2f} "
          f"{row['model']:<12} {row['r2_log']:>8.4f} {row['mae_log']:>9.4f} {row['mae_s']:>10.6f}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = os.path.join(PROJECT_ROOT, "reports", "within_db_schema_metrics.csv")
res_df.to_csv(out_path, index=False)
print(f"\nFull results saved to reports/within_db_schema_metrics.csv")
print("DONE")
