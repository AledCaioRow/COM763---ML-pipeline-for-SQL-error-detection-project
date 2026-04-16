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

Artifact output
---------------
After evaluating, the script refits the *best* pipeline for each database on
ALL available rows (not just the 80% train mask) and writes:

  artifacts/regression_by_db/<db_id>.joblib   – fitted sklearn Pipeline
  artifacts/regression_by_db/manifest.json    – feature list, target, metadata
"""
import json
import warnings
warnings.filterwarnings("ignore")

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── path wiring ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
# Insert standalone root first so its src/ takes priority over the repo root's src/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# Append repo root last so config.py is still reachable (it lives in standalone root anyway)
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.append(str(PROJECT_ROOT.parent))

from src.schema_stats import schema_stats  # noqa: E402 — shared module
from config import BIRD_DB_DIR             # noqa: E402

# ── constants ────────────────────────────────────────────────────────────────
DATA_CSV    = str(PROJECT_ROOT / "data" / "query_dataset_features.csv")
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "regression_by_db"
SEED         = 42
MIN_QUERIES  = 15

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


# ── load data ────────────────────────────────────────────────────────────────
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


# ── model factory ────────────────────────────────────────────────────────────
def make_models():
    return {
        "Ridge(a=1)":  Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        "Ridge(a=10)": Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=10.0))]),
        "Lasso":       Pipeline([("sc", StandardScaler()), ("m", Lasso(alpha=0.01, max_iter=5000))]),
        "RF":          Pipeline([("sc", StandardScaler()), ("m", RandomForestRegressor(n_estimators=100, random_state=SEED))]),
        "GBM":         Pipeline([("sc", StandardScaler()), ("m", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=SEED))]),
    }


# ── per-database evaluation loop ─────────────────────────────────────────────
results       = []
best_by_db    = {}   # db_id -> {"name": str, "r2_log": float}

print("=" * 70)
print("Within-database regression  (80% train / 20% test per database)")
print(f"Features: {len(SQL_FEATURES)} SQL structural + {len(SCHEMA_FEATURES)} schema statistics")
print(f"Database directory: {BIRD_DB_DIR}")
print("=" * 70)

for db_id, grp in df.groupby("db_id"):
    grp = grp.copy()
    n   = len(grp)
    if n < MIN_QUERIES:
        print(f"\n  {db_id} (n={n}) -- skipped, fewer than {MIN_QUERIES} queries")
        continue

    X = grp[FULL_FEATURES]
    y = grp["log_runtime"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    y_te_s = np.exp(y_te.values)

    st_stats = schema_stats(db_id)
    slow_pct  = int(100 * grp["label_binary"].mean())

    print(f"\n  {db_id}")
    print(f"    n={n}  train={len(X_tr)}  test={len(X_te)}  slow%={slow_pct}%")
    print(f"    runtime: {grp['runtime_s'].min():.6f}s - {grp['runtime_s'].max():.6f}s")
    print(f"    schema:  total_rows={st_stats.get('schema_total_rows',0):,}  "
          f"idx_cov={st_stats.get('schema_index_coverage',0):.2f}")
    print(f"    {'Model':<12}  {'MAE(log)':>9}  {'R2(log)':>8}  {'MAE(s)':>10}  {'R2(s)':>8}")
    print(f"    {'-'*12}  {'-'*9}  {'-'*8}  {'-'*10}  {'-'*8}")

    best_r2   = -999
    best_name = ""

    for name, m in make_models().items():
        m.fit(X_tr, y_tr)
        p_log = m.predict(X_te)
        p_s   = np.exp(p_log)

        mae_log = mean_absolute_error(y_te.values, p_log)
        r2_log  = r2_score(y_te.values, p_log)
        mae_s   = mean_absolute_error(y_te_s, p_s)
        r2_s    = r2_score(y_te_s, p_s)

        if r2_log > best_r2:
            best_r2   = r2_log
            best_name = name

        print(f"    {name:<12}  {mae_log:>9.4f}  {r2_log:>8.4f}  {mae_s:>10.6f}  {r2_s:>8.4f}")

        results.append({
            "db_id":          db_id,
            "n":              n,
            "n_train":        len(X_tr),
            "n_test":         len(X_te),
            "slow_pct":       slow_pct,
            "total_rows":     st_stats.get("schema_total_rows", 0),
            "max_table_rows": st_stats.get("schema_max_table_rows", 0),
            "index_coverage": st_stats.get("schema_index_coverage", 0),
            "model":          name,
            "mae_log":        round(mae_log, 4),
            "r2_log":         round(r2_log,  4),
            "mae_s":          round(mae_s,   6),
            "r2_s":           round(r2_s,    4),
        })

    print(f"    --> best: {best_name}  R2(log)={best_r2:.4f}")
    best_by_db[db_id] = {"name": best_name, "r2_log": best_r2}


# ── summary table ─────────────────────────────────────────────────────────────
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
          .sort_values("r2_log", ascending=False)
)

print(f"\n  {'Database':<25} {'n':>4} {'slow%':>6} {'total_rows':>12} {'idx_cov':>8} "
      f"{'Best model':<12} {'R2(log)':>8} {'MAE(s)':>10}")
print(f"  {'-'*25} {'-'*4} {'-'*6} {'-'*12} {'-'*8} {'-'*12} {'-'*8} {'-'*10}")
for _, row in best_per_db.iterrows():
    print(f"  {row['db_id']:<25} {int(row['n']):>4} {int(row['slow_pct']):>5}% "
          f"{int(row['total_rows']):>12,} {row['index_coverage']:>8.2f} "
          f"{row['model']:<12} {row['r2_log']:>8.4f} {row['mae_s']:>10.6f}")


# ── save evaluation CSV ───────────────────────────────────────────────────────
reports_dir = PROJECT_ROOT / "reports"
reports_dir.mkdir(exist_ok=True)
out_path = str(reports_dir / "within_db_schema_metrics.csv")
res_df.to_csv(out_path, index=False)
print(f"\nFull results saved to reports/within_db_schema_metrics.csv")


# ── export deployment artifacts ───────────────────────────────────────────────
print()
print("=" * 70)
print("Exporting per-database regression artifacts (refit on ALL rows)")
print("=" * 70)

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
manifest = {
    "feature_order":  FULL_FEATURES,
    "sql_features":   SQL_FEATURES,
    "schema_features": SCHEMA_FEATURES,
    "target":         "log_runtime",
    "notes":          "Predict log_runtime; convert to seconds via exp(pred).",
    "databases":      {},
}

for db_id, grp in df.groupby("db_id"):
    grp = grp.copy()
    n   = len(grp)
    if n < MIN_QUERIES:
        continue
    if db_id not in best_by_db:
        continue

    best_name = best_by_db[db_id]["name"]

    # Refit the winning pipeline on ALL rows for this database
    X_all = grp[FULL_FEATURES]
    y_all = grp["log_runtime"]
    final_model = make_models()[best_name]
    final_model.fit(X_all, y_all)

    out_file = ARTIFACT_DIR / f"{db_id}.joblib"
    joblib.dump(final_model, str(out_file))
    print(f"  {db_id:<25} best={best_name:<12}  n={n}  -> {out_file.name}")

    manifest["databases"][db_id] = {
        "best_model": best_name,
        "r2_log":     best_by_db[db_id]["r2_log"],
        "n_train":    n,
    }

manifest_path = ARTIFACT_DIR / "manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2))
print(f"\nManifest written to {manifest_path}")
print("DONE")
