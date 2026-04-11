"""
Stage 7 — Evaluation on the held-out test set + reporting.

Produces:
  - Classification report (precision / recall / F1)
  - Confusion matrix
  - ROC-AUC
  - Feature importance ranking
  - Text report saved to reports/model_results.txt
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import FEATURE_COLS, REPORTS_DIR, SPLIT_METHOD, HOLDOUT_DATABASES


# ============================================================
# TEST-SET EVALUATION
# ============================================================

def evaluate_on_test(model, X_test, y_test, test_meta):
    """Score the fitted model on the held-out test set.

    Returns a dict with report, confusion matrix, F1, ROC-AUC, and
    raw predictions / probabilities.
    """
    y_pred = model.predict(X_test)

    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, target_names=["fast", "slow"])
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    per_database_rows = []
    per_difficulty_rows = []
    tmp_df = pd.DataFrame(
        {
            "db_id": test_meta["db_id"].values,
            "difficulty": test_meta["difficulty"].values,
            "y_true": y_test.values,
            "y_pred": y_pred,
        }
    )

    for db_id, grp in tmp_df.groupby("db_id"):
        n = len(grp)
        n_classes = grp["y_true"].nunique()
        if n < 2 or n_classes < 2:
            per_database_rows.append(
                {
                    "db_id": db_id,
                    "n_queries": int(n),
                    "accuracy": "N/A",
                    "f1": "N/A",
                    "precision": "N/A",
                    "recall": "N/A",
                }
            )
            continue

        per_database_rows.append(
            {
                "db_id": db_id,
                "n_queries": int(n),
                "accuracy": accuracy_score(grp["y_true"], grp["y_pred"]),
                "f1": f1_score(grp["y_true"], grp["y_pred"], zero_division=0),
                "precision": precision_score(
                    grp["y_true"], grp["y_pred"], zero_division=0
                ),
                "recall": recall_score(grp["y_true"], grp["y_pred"], zero_division=0),
            }
        )

    for difficulty, grp in tmp_df.groupby("difficulty"):
        n = len(grp)
        n_classes = grp["y_true"].nunique()
        if n < 2 or n_classes < 2:
            per_difficulty_rows.append(
                {
                    "difficulty": difficulty,
                    "n_queries": int(n),
                    "accuracy": "N/A",
                    "f1": "N/A",
                    "precision": "N/A",
                    "recall": "N/A",
                }
            )
            continue

        per_difficulty_rows.append(
            {
                "difficulty": difficulty,
                "n_queries": int(n),
                "accuracy": accuracy_score(grp["y_true"], grp["y_pred"]),
                "f1": f1_score(grp["y_true"], grp["y_pred"], zero_division=0),
                "precision": precision_score(
                    grp["y_true"], grp["y_pred"], zero_division=0
                ),
                "recall": recall_score(grp["y_true"], grp["y_pred"], zero_division=0),
            }
        )

    per_database_df = pd.DataFrame(per_database_rows).sort_values("db_id").reset_index(
        drop=True
    )
    per_difficulty_df = pd.DataFrame(per_difficulty_rows).sort_values(
        "difficulty"
    ).reset_index(drop=True)

    per_database_path = os.path.join(REPORTS_DIR, "per_database_results.csv")
    per_difficulty_path = os.path.join(REPORTS_DIR, "per_difficulty_results.csv")
    per_database_df.to_csv(per_database_path, index=False)
    per_difficulty_df.to_csv(per_difficulty_path, index=False)

    print(f"\n[STAGE 7] Test-set evaluation  ({len(y_test)} samples)")
    print(f"  F1 Score : {f1:.4f}")
    if auc is not None:
        print(f"  ROC-AUC  : {auc:.4f}")
    print(f"\n{report}")

    return {
        "report": report,
        "confusion_matrix": cm,
        "f1": f1,
        "roc_auc": auc,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "per_database": per_database_df,
        "per_difficulty": per_difficulty_df,
        "per_database_csv": per_database_path,
        "per_difficulty_csv": per_difficulty_path,
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(model, feature_cols=None):
    """Extract feature importances from a tree-based or linear model."""
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif (hasattr(model, "named_steps")
          and hasattr(model.named_steps.get("clf"), "coef_")):
        importances = np.abs(model.named_steps["clf"].coef_[0])
    else:
        return None

    return (
        pd.Series(importances[: len(feature_cols)],
                   index=feature_cols[: len(importances)])
        .sort_values(ascending=False)
    )


# ============================================================
# REPORT WRITER
# ============================================================

def save_report(cv_results, test_metrics, best_name, feat_imp,
                train_size, test_size):
    """Write a human-readable summary to reports/model_results.txt."""
    path = os.path.join(REPORTS_DIR, "model_results.txt")

    with open(path, "w") as f:
        f.write("COM 763 — SQL Query Performance Predictor\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Train set : {train_size} queries\n")
        f.write(f"Test  set : {test_size} queries\n\n")

        f.write("Cross-Validation Results (on training set)\n")
        f.write("-" * 50 + "\n")
        for name, res in sorted(cv_results.items(),
                                key=lambda x: -x[1]["f1_mean"]):
            f.write(f"  {name:25s}  F1 = {res['f1_mean']:.4f} "
                    f"± {res['f1_std']:.4f}\n")

        f.write(f"\nBest model: {best_name}\n\n")

        f.write("Held-Out Test Set Evaluation\n")
        f.write("-" * 50 + "\n")
        f.write(f"  F1 Score : {test_metrics['f1']:.4f}\n")
        if test_metrics["roc_auc"] is not None:
            f.write(f"  ROC-AUC  : {test_metrics['roc_auc']:.4f}\n")
        f.write(f"\n{test_metrics['report']}\n")

        if feat_imp is not None:
            f.write("Top 10 Features\n")
            f.write("-" * 50 + "\n")
            for feat, imp in feat_imp.head(10).items():
                f.write(f"  {feat:30s} {imp:.4f}\n")

        def _fmt(v):
            return "N/A" if isinstance(v, str) else f"{v:.2f}"

        f.write("\n=== Per-Database Breakdown ===\n")
        f.write(f"{'Database':24s}{'N':>6s}{'Acc':>7s}{'F1':>7s}{'Prec':>8s}{'Recall':>8s}\n")
        for row in test_metrics["per_database"].itertuples(index=False):
            f.write(
                f"{row.db_id:24s}"
                f"{row.n_queries:6d}"
                f"{_fmt(row.accuracy):>7s}"
                f"{_fmt(row.f1):>7s}"
                f"{_fmt(row.precision):>8s}"
                f"{_fmt(row.recall):>8s}\n"
            )

        f.write("\n=== Per-Difficulty Breakdown ===\n")
        f.write(f"{'Difficulty':24s}{'N':>6s}{'Acc':>7s}{'F1':>7s}\n")
        for row in test_metrics["per_difficulty"].itertuples(index=False):
            f.write(
                f"{str(row.difficulty):24s}"
                f"{row.n_queries:6d}"
                f"{_fmt(row.accuracy):>7s}"
                f"{_fmt(row.f1):>7s}\n"
            )

        f.write("\n")
        if SPLIT_METHOD == "database_aware":
            f.write(
                "Note: database-aware split was used. "
                "The following databases were held out and never seen during training: "
                f"{', '.join(HOLDOUT_DATABASES)}.\n"
            )
        else:
            f.write(
                "Note: random stratified split was used. "
                "Per-database metrics may be optimistic because the same databases "
                "can appear in both train and test sets.\n"
            )

    print(f"  Report saved -> {path}")
    print(f"  Per-database CSV saved -> {test_metrics['per_database_csv']}")
    print(f"  Per-difficulty CSV saved -> {test_metrics['per_difficulty_csv']}")
    return path
