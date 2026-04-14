"""
Quick, deadline-safe improvement experiments for System A.

Produces:
  - reports/quick_experiments_summary.csv
  - reports/quick_experiments_summary.md
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DATA_DIR, RANDOM_SEED, REPORTS_DIR
from src.models.train import split_data


@dataclass
class ExperimentResult:
    name: str
    threshold: float
    f1_slow: float
    precision_slow: float
    recall_slow: float
    roc_auc: float

    def to_dict(self) -> dict:
        return {
            "Experiment": self.name,
            "Threshold": round(self.threshold, 3),
            "F1_slow": round(self.f1_slow, 4),
            "Precision_slow": round(self.precision_slow, 4),
            "Recall_slow": round(self.recall_slow, 4),
            "ROC_AUC": round(self.roc_auc, 4),
        }


def _weighted_logistic() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_SEED,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def _evaluate_from_proba(
    name: str, y_true: pd.Series, y_proba: np.ndarray, threshold: float
) -> ExperimentResult:
    y_pred = (y_proba >= threshold).astype(int)
    return ExperimentResult(
        name=name,
        threshold=threshold,
        f1_slow=float(f1_score(y_true, y_pred, zero_division=0)),
        precision_slow=float(precision_score(y_true, y_pred, zero_division=0)),
        recall_slow=float(recall_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_proba)),
    )


def _best_threshold_by_cv(X_train: pd.DataFrame, y_train: pd.Series) -> float:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    model = _weighted_logistic()
    y_oof_proba = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    thresholds = np.arange(0.05, 0.96, 0.01)
    best_t = 0.50
    best_f1 = -1.0
    for t in thresholds:
        y_oof_pred = (y_oof_proba >= t).astype(int)
        score = f1_score(y_train, y_oof_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return best_t


def _write_markdown_table(df: pd.DataFrame, path: str) -> None:
    lines = [
        "# Quick Experiments (System A)",
        "",
        "| Experiment | Threshold | F1 slow | Precision slow | Recall slow | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"| {row.Experiment} | {row.Threshold:.3f} | {row.F1_slow:.4f} | "
            f"{row.Precision_slow:.4f} | {row.Recall_slow:.4f} | {row.ROC_AUC:.4f} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)

    features_path = os.path.join(DATA_DIR, "query_dataset_features.csv")
    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"Missing feature dataset: {features_path}. Run `python -u main.py` first."
        )

    df = pd.read_csv(features_path)
    X_train, X_test, y_train, y_test, _, _ = split_data(df)

    # Baseline: current best model artifact.
    baseline_path = os.path.join("artifacts", "best_model.joblib")
    baseline_model = joblib.load(baseline_path)
    baseline_proba = baseline_model.predict_proba(X_test)[:, 1]
    baseline_result = _evaluate_from_proba(
        name="Baseline (best_model.joblib, threshold=0.5)",
        y_true=y_test,
        y_proba=baseline_proba,
        threshold=0.50,
    )

    # Exp1: weighted logistic regression with default threshold 0.5.
    weighted_model = _weighted_logistic()
    weighted_model.fit(X_train, y_train)
    weighted_proba = weighted_model.predict_proba(X_test)[:, 1]
    exp1_result = _evaluate_from_proba(
        name="Exp1 Weighted Logistic Regression (class_weight=balanced)",
        y_true=y_test,
        y_proba=weighted_proba,
        threshold=0.50,
    )

    # Exp2: weighted logistic regression + threshold tuned on CV OOF probs.
    tuned_threshold = _best_threshold_by_cv(X_train=X_train, y_train=y_train)
    exp2_result = _evaluate_from_proba(
        name=f"Exp2 Weighted Logistic + tuned threshold ({tuned_threshold:.2f})",
        y_true=y_test,
        y_proba=weighted_proba,
        threshold=tuned_threshold,
    )

    summary = pd.DataFrame(
        [
            baseline_result.to_dict(),
            exp1_result.to_dict(),
            exp2_result.to_dict(),
        ]
    )
    summary_path = os.path.join(REPORTS_DIR, "quick_experiments_summary.csv")
    summary.to_csv(summary_path, index=False)

    summary_md_path = os.path.join(REPORTS_DIR, "quick_experiments_summary.md")
    _write_markdown_table(summary, summary_md_path)

    print(f"Saved: {summary_path}")
    print(f"Saved: {summary_md_path}")


if __name__ == "__main__":
    main()
