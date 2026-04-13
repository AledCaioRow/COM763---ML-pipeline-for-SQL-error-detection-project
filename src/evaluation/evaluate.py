"""Stage 7 — rich held-out evaluation + reporting artefacts."""

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, learning_curve

from config import FEATURE_COLS, HOLDOUT_DATABASES, REPORTS_DIR, SPLIT_METHOD

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None


# ============================================================
# TEST-SET EVALUATION
# ============================================================

def _safe_binary_metric(metric_fn, y_true, y_pred):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(metric_fn(y_true, y_pred))


def get_feature_importance(model, feature_cols=None):
    """Extract feature importances from a tree-based or linear model."""
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("clf"), "coef_"):
        importances = np.abs(model.named_steps["clf"].coef_[0])
    else:
        return None

    return pd.Series(
        importances[: len(feature_cols)],
        index=feature_cols[: len(importances)],
    ).sort_values(ascending=False)


def _plot_confusion_matrices(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=["fast_true", "slow_true"], columns=["fast_pred", "slow_pred"])
    cm_df.to_csv(os.path.join(REPORTS_DIR, "confusion_matrix.csv"), index=True)

    fig1, ax1 = plt.subplots(figsize=(6, 5), dpi=180)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["fast", "slow"]).plot(
        ax=ax1, cmap="Blues", colorbar=False
    )
    ax1.set_title("Confusion Matrix")
    fig1.tight_layout()
    fig1.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"))
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(6, 5), dpi=180)
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, normalize="true", display_labels=["fast", "slow"], ax=ax2, cmap="Blues", colorbar=False
    )
    ax2.set_title("Confusion Matrix (Normalised)")
    fig2.tight_layout()
    fig2.savefig(os.path.join(REPORTS_DIR, "confusion_matrix_normalised.png"))
    plt.close(fig2)


def _plot_roc_pr_curves(y_test, best_name, model_outputs):
    plt.figure(figsize=(7, 5), dpi=180)
    for model_name, pred in model_outputs.items():
        if pred["y_proba"] is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, pred["y_proba"])
        model_auc = roc_auc_score(y_test, pred["y_proba"])
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={model_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test Set)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "roc_curve.png"))
    plt.close()

    best_pred = model_outputs[best_name]
    if best_pred["y_proba"] is not None:
        precision, recall, _ = precision_recall_curve(y_test, best_pred["y_proba"])
        ap = average_precision_score(y_test, best_pred["y_proba"])
        plt.figure(figsize=(7, 5), dpi=180)
        plt.plot(recall, precision, label=f"{best_name} (AP={ap:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve (Test Set)")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, "pr_curve.png"))
        plt.close()

        frac_pos, mean_pred = calibration_curve(y_test, best_pred["y_proba"], n_bins=10)
        plt.figure(figsize=(7, 5), dpi=180)
        plt.plot(mean_pred, frac_pos, marker="o", label="Model")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title("Calibration Curve")
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, "calibration_curve.png"))
        plt.close()


def _save_cv_fold_scores(cv_results: Dict[str, Dict]):
    rows = []
    for model_name, values in cv_results.items():
        raw = values["cv_raw"]
        for fold_idx in range(len(raw["test_f1"])):
            rows.append(
                {
                    "model": model_name,
                    "fold": fold_idx + 1,
                    "train_f1": float(raw["train_f1"][fold_idx]),
                    "test_f1": float(raw["test_f1"][fold_idx]),
                    "test_accuracy": float(raw["test_accuracy"][fold_idx]),
                    "test_precision": float(raw["test_precision"][fold_idx]),
                    "test_recall": float(raw["test_recall"][fold_idx]),
                    "test_roc_auc": float(raw["test_roc_auc"][fold_idx]),
                    "fit_time_s": float(raw["fit_time"][fold_idx]),
                    "score_time_s": float(raw["score_time"][fold_idx]),
                }
            )

    fold_df = pd.DataFrame(rows).sort_values(["model", "fold"]).reset_index(drop=True)
    fold_df.to_csv(os.path.join(REPORTS_DIR, "cv_fold_scores.csv"), index=False)

    plt.figure(figsize=(8, 5), dpi=180)
    ordered_models = fold_df["model"].drop_duplicates().tolist()
    box_data = [fold_df.loc[fold_df["model"] == m, "test_f1"].values for m in ordered_models]
    plt.boxplot(box_data, labels=ordered_models, vert=True)
    plt.ylabel("F1 (CV fold)")
    plt.title("Cross-Validation Fold F1 Distribution")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "cv_boxplot.png"))
    plt.close()


def _save_feature_importance(best_model):
    feat_imp = get_feature_importance(best_model)
    if feat_imp is None:
        return None

    feat_df = feat_imp.reset_index()
    feat_df.columns = ["feature", "importance"]
    feat_df.to_csv(os.path.join(REPORTS_DIR, "feature_importance.csv"), index=False)

    plt.figure(figsize=(8, 7), dpi=180)
    top = feat_df.sort_values("importance", ascending=True)
    plt.barh(top["feature"], top["importance"])
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "feature_importance.png"))
    plt.close()
    return feat_imp


def _group_breakdown(y_test, y_pred, test_meta):
    tmp_df = pd.DataFrame(
        {
            "db_id": test_meta["db_id"].values,
            "difficulty": test_meta["difficulty"].values,
            "y_true": np.asarray(y_test),
            "y_pred": np.asarray(y_pred),
        }
    )

    per_database_rows = []
    for db_id, grp in tmp_df.groupby("db_id"):
        per_database_rows.append(
            {
                "db_id": db_id,
                "support": int(len(grp)),
                "accuracy": _safe_binary_metric(accuracy_score, grp["y_true"], grp["y_pred"]),
                "f1": _safe_binary_metric(f1_score, grp["y_true"], grp["y_pred"]),
                "precision": _safe_binary_metric(precision_score, grp["y_true"], grp["y_pred"]),
                "recall": _safe_binary_metric(recall_score, grp["y_true"], grp["y_pred"]),
            }
        )

    per_difficulty_rows = []
    for difficulty, grp in tmp_df.groupby("difficulty"):
        per_difficulty_rows.append(
            {
                "difficulty": difficulty,
                "support": int(len(grp)),
                "accuracy": _safe_binary_metric(accuracy_score, grp["y_true"], grp["y_pred"]),
                "f1": _safe_binary_metric(f1_score, grp["y_true"], grp["y_pred"]),
                "precision": _safe_binary_metric(precision_score, grp["y_true"], grp["y_pred"]),
                "recall": _safe_binary_metric(recall_score, grp["y_true"], grp["y_pred"]),
            }
        )

    per_database_df = pd.DataFrame(per_database_rows).sort_values("db_id").reset_index(drop=True)
    per_difficulty_df = pd.DataFrame(per_difficulty_rows).sort_values("difficulty").reset_index(drop=True)
    per_database_df.to_csv(os.path.join(REPORTS_DIR, "per_database_results.csv"), index=False)
    per_difficulty_df.to_csv(os.path.join(REPORTS_DIR, "per_difficulty_results.csv"), index=False)
    return per_database_df, per_difficulty_df, tmp_df


def _save_dataset_descriptives(raw_df, labeled_df, train_meta, test_meta, train_y, test_y):
    label_counts = raw_df["runtime_s"].copy()
    p50 = float(label_counts.quantile(0.50))
    p75 = float(label_counts.quantile(0.75))
    class_df = pd.DataFrame(
        [
            {"label": "fast", "count": int((raw_df["runtime_s"] <= p50).sum())},
            {"label": "mid", "count": int(((raw_df["runtime_s"] > p50) & (raw_df["runtime_s"] < p75)).sum())},
            {"label": "slow", "count": int((raw_df["runtime_s"] >= p75).sum())},
        ]
    )
    class_df["percentage"] = (class_df["count"] / class_df["count"].sum() * 100.0).round(2)
    class_df.to_csv(os.path.join(REPORTS_DIR, "class_distribution.csv"), index=False)

    plt.figure(figsize=(6, 4), dpi=180)
    plt.bar(class_df["label"], class_df["count"])
    plt.title("Class Distribution (Quantile Labels)")
    plt.ylabel("Queries")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "class_distribution.png"))
    plt.close()

    plt.figure(figsize=(8, 4), dpi=180)
    plt.hist(raw_df["runtime_s"].values, bins=30, alpha=0.85, edgecolor="black")
    plt.axvline(p50, color="orange", linestyle="--", label=f"p50={p50:.3f}s")
    plt.axvline(p75, color="red", linestyle="--", label=f"p75={p75:.3f}s")
    plt.xlabel("runtime_s")
    plt.ylabel("Count")
    plt.title("Runtime Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "runtime_distribution.png"))
    plt.close()

    train_counts = pd.Series(train_y).value_counts().to_dict()
    test_counts = pd.Series(test_y).value_counts().to_dict()
    split_rows = [
        {
            "split": "Train (CV)",
            "n_queries": int(len(train_y)),
            "fast": int(train_counts.get(0, 0)),
            "slow": int(train_counts.get(1, 0)),
            "databases": ", ".join(sorted(train_meta["db_id"].astype(str).unique().tolist())),
        },
        {
            "split": "Test",
            "n_queries": int(len(test_y)),
            "fast": int(test_counts.get(0, 0)),
            "slow": int(test_counts.get(1, 0)),
            "databases": ", ".join(sorted(test_meta["db_id"].astype(str).unique().tolist())),
        },
    ]
    split_df = pd.DataFrame(split_rows)
    split_df.to_csv(os.path.join(REPORTS_DIR, "split_summary.csv"), index=False)


def evaluate_models_on_test(
    fitted_models,
    best_name,
    cv_results,
    X_train,
    y_train,
    X_test,
    y_test,
    train_meta,
    test_meta,
    raw_df,
    labeled_df,
):
    """Evaluate all models on test, save required artefacts, and return summary dict."""
    model_outputs = {}
    all_rows = []
    for model_name, model in fitted_models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        model_outputs[model_name] = {"y_pred": y_pred, "y_proba": y_proba}

        cv_mean = cv_results[model_name]["f1_mean"]
        cv_std = cv_results[model_name]["f1_std"]
        row = {
            "Model": model_name,
            "CV F1 (mean±std)": f"{cv_mean:.4f}±{cv_std:.4f}",
            "Test Accuracy": float(accuracy_score(y_test, y_pred)),
            "Test Precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "Test Recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "Test F1": float(f1_score(y_test, y_pred, zero_division=0)),
            "Test ROC-AUC": float(roc_auc_score(y_test, y_proba)) if y_proba is not None else np.nan,
            "Fit Time (s)": float(np.mean(cv_results[model_name]["cv_raw"]["fit_time"])),
        }
        all_rows.append(row)

    comparison_df = pd.DataFrame(all_rows).sort_values("Test F1", ascending=False).reset_index(drop=True)
    comparison_df.to_csv(os.path.join(REPORTS_DIR, "all_models_test_comparison.csv"), index=False)

    best_pred = model_outputs[best_name]["y_pred"]
    best_proba = model_outputs[best_name]["y_proba"]
    report_dict = classification_report(y_test, best_pred, target_names=["fast", "slow"], output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index": "class"})
    report_df.to_csv(os.path.join(REPORTS_DIR, "classification_report.csv"), index=False)

    _save_cv_fold_scores(cv_results)
    _plot_confusion_matrices(y_test, best_pred)
    _plot_roc_pr_curves(y_test, best_name, model_outputs)
    feat_imp = _save_feature_importance(fitted_models[best_name])
    per_database_df, per_difficulty_df, error_df = _group_breakdown(y_test, best_pred, test_meta)
    _save_dataset_descriptives(raw_df, labeled_df, train_meta, test_meta, y_train, y_test)

    misclassified = error_df.loc[error_df["y_true"] != error_df["y_pred"]].copy()
    if not misclassified.empty:
        summary = (
            misclassified.groupby(["db_id", "difficulty"])
            .size()
            .reset_index(name="n_errors")
            .sort_values("n_errors", ascending=False)
        )
        summary.to_csv(os.path.join(REPORTS_DIR, "error_analysis.csv"), index=False)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes, train_scores, val_scores = learning_curve(
        fitted_models[best_name],
        X_train,
        y_train,
        cv=cv,
        scoring="f1",
        train_sizes=np.linspace(0.1, 1.0, 10),
    )
    plt.figure(figsize=(7, 5), dpi=180)
    plt.plot(train_sizes, train_scores.mean(axis=1), marker="o", label="Train F1")
    plt.plot(train_sizes, val_scores.mean(axis=1), marker="o", label="Validation F1")
    plt.xlabel("Training samples")
    plt.ylabel("F1")
    plt.title("Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "learning_curve.png"))
    plt.close()

    significance_note = "Wilcoxon test unavailable."
    if wilcoxon and len(cv_results) >= 2:
        ordered = sorted(cv_results.items(), key=lambda x: x[1]["f1_mean"], reverse=True)[:2]
        top_a, top_b = ordered[0][0], ordered[1][0]
        stat, p_value = wilcoxon(
            cv_results[top_a]["cv_raw"]["test_f1"],
            cv_results[top_b]["cv_raw"]["test_f1"],
        )
        significance_note = (
            f"Wilcoxon {top_a} vs {top_b}: stat={float(stat):.4f}, p={float(p_value):.4f}"
        )

    print(f"\n[STAGE 7] Test-set evaluation ({len(y_test)} samples)")
    print(f"  Best model: {best_name}")
    print(f"  F1 Score : {f1_score(y_test, best_pred):.4f}")
    if best_proba is not None:
        print(f"  ROC-AUC  : {roc_auc_score(y_test, best_proba):.4f}")
    print(f"  {significance_note}")

    return {
        "report": classification_report(y_test, best_pred, target_names=["fast", "slow"]),
        "f1": float(f1_score(y_test, best_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, best_proba)) if best_proba is not None else None,
        "per_database": per_database_df,
        "per_difficulty": per_difficulty_df,
        "per_database_csv": os.path.join(REPORTS_DIR, "per_database_results.csv"),
        "per_difficulty_csv": os.path.join(REPORTS_DIR, "per_difficulty_results.csv"),
        "classification_report_csv": os.path.join(REPORTS_DIR, "classification_report.csv"),
        "comparison_csv": os.path.join(REPORTS_DIR, "all_models_test_comparison.csv"),
        "best_model": best_name,
        "significance_note": significance_note,
        "feature_importance": feat_imp,
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def save_report(cv_results, test_metrics, best_name, feat_imp, train_size, test_size):
    """Write a human-readable summary to reports/model_results.txt."""
    path = os.path.join(REPORTS_DIR, "model_results.txt")

    with open(path, "w") as f:
        f.write("COM 763 — SQL Query Performance Predictor\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Train set : {train_size} queries\n")
        f.write(f"Test  set : {test_size} queries\n\n")

        f.write("Cross-Validation Results (on training set)\n")
        f.write("-" * 50 + "\n")
        for name, res in sorted(cv_results.items(), key=lambda x: -x[1]["f1_mean"]):
            f.write(
                f"  {name:25s}  F1 = {res['f1_mean']:.4f} "
                f"± {res['f1_std']:.4f}\n"
            )

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
            if pd.isna(v):
                return "N/A"
            if isinstance(v, str):
                return v
            return f"{v:.2f}"

        f.write("\n=== Per-Database Breakdown ===\n")
        f.write(f"{'Database':24s}{'N':>6s}{'Acc':>7s}{'F1':>7s}{'Prec':>8s}{'Recall':>8s}\n")
        for row in test_metrics["per_database"].itertuples(index=False):
            f.write(
                f"{row.db_id:24s}"
                f"{int(row.support):6d}"
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
                f"{int(row.support):6d}"
                f"{_fmt(row.accuracy):>7s}"
                f"{_fmt(row.f1):>7s}\n"
            )

        f.write("\n")
        if test_metrics.get("significance_note"):
            f.write(test_metrics["significance_note"] + "\n")
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
