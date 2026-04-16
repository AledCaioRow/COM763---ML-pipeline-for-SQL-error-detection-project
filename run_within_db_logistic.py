# -*- coding: utf-8 -*-
"""
Iteration 3 — per-database SQL-feature classification with LogisticRegression.

Mirrors the per-DB 80/20 loop in run_schema_stats_model.py but predicts label_binary
from SQL structural features only (no schema stats). Writes:
  reports/within_db_logistic_metrics.csv

Rows with degenerate train/test labels are recorded with status != ok.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_CSV = PROJECT_ROOT / "data" / "query_dataset_features.csv"
OUT_CSV = PROJECT_ROOT / "reports" / "within_db_logistic_metrics.csv"
SEED = 42
MIN_QUERIES = 15

SQL_FEATURES = [
    "n_tokens", "query_length", "n_joins", "n_tables_approx",
    "n_where_predicates", "has_group_by", "has_order_by", "has_having",
    "has_distinct", "has_limit", "has_union", "n_subqueries", "has_subquery",
    "max_nesting_depth", "n_count", "n_sum", "n_avg", "n_max", "n_min",
    "n_aggregations", "has_between", "has_in_clause", "has_like",
    "has_exists", "has_correlated_subquery",
]


def _make_model() -> Pipeline:
    return Pipeline(
        [
            ("sc", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=5000, random_state=SEED),
            ),
        ]
    )


def main() -> None:
    df = pd.read_csv(DATA_CSV)
    df = df[df["runtime_s"] > 0].copy()
    results: list[dict] = []

    print("=" * 70)
    print("Within-database classification — LogisticRegression, SQL features only")
    print("=" * 70)

    for db_id, grp in df.groupby("db_id"):
        grp = grp.copy()
        n = len(grp)
        if n < MIN_QUERIES:
            results.append({
                "db_id": db_id,
                "n": n,
                "n_train": np.nan,
                "n_test": np.nan,
                "slow_pct": int(round(100 * grp["label_binary"].mean())) if n else np.nan,
                "f1_slow": np.nan,
                "roc_auc": np.nan,
                "accuracy": np.nan,
                "status": "skipped_small",
            })
            print(f"\n  {db_id} (n={n}) -- skipped (< {MIN_QUERIES} queries)")
            continue

        feats = [c for c in SQL_FEATURES if c in grp.columns]
        if len(feats) < len(SQL_FEATURES):
            missing = set(SQL_FEATURES) - set(feats)
            print(f"\n  {db_id} -- missing columns {missing}")
        X = grp[feats].fillna(0)
        y = grp["label_binary"].astype(int)
        strat = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=SEED, stratify=strat
            )
        except ValueError:
            results.append({
                "db_id": db_id,
                "n": n,
                "n_train": np.nan,
                "n_test": np.nan,
                "slow_pct": int(round(100 * y.mean())),
                "f1_slow": np.nan,
                "roc_auc": np.nan,
                "accuracy": np.nan,
                "status": "split_failed",
            })
            print(f"\n  {db_id} -- train_test_split failed (degenerate stratify)")
            continue

        if y_tr.nunique() < 2:
            results.append({
                "db_id": db_id,
                "n": n,
                "n_train": len(X_tr),
                "n_test": len(X_te),
                "slow_pct": int(round(100 * y.mean())),
                "f1_slow": np.nan,
                "roc_auc": np.nan,
                "accuracy": np.nan,
                "status": "degenerate_train",
            })
            print(f"\n  {db_id} -- degenerate training labels")
            continue
        if y_te.nunique() < 2:
            results.append({
                "db_id": db_id,
                "n": n,
                "n_train": len(X_tr),
                "n_test": len(X_te),
                "slow_pct": int(round(100 * y.mean())),
                "f1_slow": np.nan,
                "roc_auc": np.nan,
                "accuracy": np.nan,
                "status": "degenerate_test",
            })
            print(f"\n  {db_id} -- degenerate test labels")
            continue

        m = _make_model()
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        proba = m.predict_proba(X_te)[:, 1]
        f1 = float(f1_score(y_te, pred, pos_label=1, zero_division=0))
        acc = float(accuracy_score(y_te, pred))
        try:
            roc = float(roc_auc_score(y_te, proba))
        except ValueError:
            roc = float("nan")

        results.append({
            "db_id": db_id,
            "n": n,
            "n_train": len(X_tr),
            "n_test": len(X_te),
            "slow_pct": int(round(100 * y.mean())),
            "f1_slow": round(f1, 6),
            "roc_auc": round(roc, 6) if pd.notna(roc) else np.nan,
            "accuracy": round(acc, 6),
            "status": "ok",
        })
        print(f"\n  {db_id}  n={n}  train={len(X_tr)} test={len(X_te)}  "
              f"slow%={int(round(100 * y.mean()))}  F1(slow)={f1:.4f}  ROC-AUC={roc:.4f}  acc={acc:.4f}")

    out_df = pd.DataFrame(results)
    os.makedirs(PROJECT_ROOT / "reports", exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")
    print("DONE")


if __name__ == "__main__":
    main()
