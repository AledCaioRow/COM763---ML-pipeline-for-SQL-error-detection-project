#!/usr/bin/env python3
"""
RERUN_FULL_STATS — single-file drop-in for the root legacy pipeline.

Run from repo root:
    python rerun_full_stats.py

Implements every stat, plot, and CSV from RERUN_FULL_STATS.md Steps 1-5.
Does NOT touch sql_runtime_predictor/.
"""

import os, sys, json, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from pathlib import Path

warnings.filterwarnings("ignore")

# ── Resolve project root ─────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# ── Import project config ────────────────────────────────────
try:
    import config
    FEATURE_COLS = config.FEATURE_COLS
    SPLIT_METHOD = getattr(config, "SPLIT_METHOD", "database_aware")
    HOLDOUT_DATABASES = getattr(config, "HOLDOUT_DATABASES", ["formula_1", "financial"])
    LABEL_METHOD = getattr(config, "LABEL_METHOD", "quantile")
    TIMING_RUNS = getattr(config, "TIMING_RUNS", 3)
    QUERY_TIMEOUT_S = getattr(config, "QUERY_TIMEOUT_S", 30)
except ImportError:
    print("FATAL: config.py not found in project root.")
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────
RAW_CSV     = PROJECT_ROOT / "data" / "query_dataset_raw.csv"
FEAT_CSV    = PROJECT_ROOT / "data" / "query_dataset_features.csv"
REPORTS     = PROJECT_ROOT / "reports"
ARTIFACTS   = PROJECT_ROOT / "artifacts"

REPORTS.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)

# ── Plotting defaults ─────────────────────────────────────────
DPI = 150
plt.rcParams.update({
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "figure.figsize": (8, 5),
    "font.size": 11,
})

# ══════════════════════════════════════════════════════════════
#  STEP 0: Purge old artifacts
# ══════════════════════════════════════════════════════════════
def purge():
    targets = [
        ARTIFACTS / "best_model.joblib",
        RAW_CSV, FEAT_CSV,
        REPORTS / "model_results.txt",
        REPORTS / "per_database_results.csv",
        REPORTS / "per_difficulty_results.csv",
    ]
    for f in targets:
        if f.exists():
            f.unlink()
            print(f"  purged {f.relative_to(PROJECT_ROOT)}")
    # purge old images
    for ext in ("*.png", "*.jpg", "*.svg"):
        for f in REPORTS.glob(ext):
            f.unlink()
            print(f"  purged {f.relative_to(PROJECT_ROOT)}")

# ══════════════════════════════════════════════════════════════
#  STEP 0b: Run main.py if raw data missing
# ══════════════════════════════════════════════════════════════
def ensure_data():
    """Run main.py to generate raw + feature CSVs if they don't exist."""
    if RAW_CSV.exists() and FEAT_CSV.exists():
        print("  data CSVs already exist, skipping main.py")
        return
    print("  running main.py to generate data...")
    ret = os.system(f"{sys.executable} -u main.py")
    if ret != 0:
        print("WARNING: main.py exited with non-zero code. Checking if CSVs exist anyway...")
    if not FEAT_CSV.exists():
        print("FATAL: query_dataset_features.csv not generated. Cannot continue.")
        sys.exit(1)

# ══════════════════════════════════════════════════════════════
#  DATA LOADING + SPLITTING
# ══════════════════════════════════════════════════════════════
def load_and_split():
    df = pd.read_csv(FEAT_CSV)

    # ensure label column exists
    label_col = "label"
    if label_col not in df.columns:
        # try to find it
        for c in ["target", "class", "slow"]:
            if c in df.columns:
                label_col = c
                break
        else:
            print("FATAL: no label column found in features CSV")
            sys.exit(1)

    # drop NaN labels (middle band in quantile mode)
    df = df.dropna(subset=[label_col]).copy()
    df[label_col] = df[label_col].astype(int)

    # identify feature columns that actually exist
    available_feats = [c for c in FEATURE_COLS if c in df.columns]
    if len(available_feats) < len(FEATURE_COLS):
        missing = set(FEATURE_COLS) - set(available_feats)
        print(f"  WARNING: {len(missing)} feature cols missing: {missing}")
    print(f"  using {len(available_feats)} features, {len(df)} labelled queries")

    # split
    if SPLIT_METHOD == "database_aware" and "db_id" in df.columns:
        test_mask = df["db_id"].isin(HOLDOUT_DATABASES)
        train_df = df[~test_mask].copy()
        test_df  = df[test_mask].copy()
        print(f"  database-aware split: train={len(train_df)}, test={len(test_df)} (holdout: {HOLDOUT_DATABASES})")
    else:
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[label_col])
        print(f"  random split: train={len(train_df)}, test={len(test_df)}")

    X_train = train_df[available_feats].values
    y_train = train_df[label_col].values
    X_test  = test_df[available_feats].values
    y_test  = test_df[label_col].values

    return df, train_df, test_df, X_train, y_train, X_test, y_test, available_feats, label_col

# ══════════════════════════════════════════════════════════════
#  STEP 1: Cross-validation with full stats
# ══════════════════════════════════════════════════════════════
def build_models():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, random_state=42),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
    except ImportError:
        print("  WARNING: xgboost not installed, skipping XGBoost")
    return models

def run_cross_validation(X_train, y_train, models):
    from sklearn.model_selection import StratifiedKFold, cross_validate

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "f1": "f1", "accuracy": "accuracy",
        "precision": "precision", "recall": "recall",
        "roc_auc": "roc_auc",
    }

    all_cv = {}
    for name, model in models.items():
        t0 = time.time()
        cv_res = cross_validate(
            model, X_train, y_train, cv=skf,
            scoring=scoring,
            return_train_score=True,
            return_estimator=True,
            n_jobs=-1,
        )
        elapsed = time.time() - t0
        cv_res["wall_time"] = elapsed
        all_cv[name] = cv_res
        mean_f1 = cv_res["test_f1"].mean()
        std_f1  = cv_res["test_f1"].std()
        print(f"  {name:25s}  CV F1 = {mean_f1:.4f} ± {std_f1:.4f}  ({elapsed:.1f}s)")

    return all_cv, skf

def save_cv_fold_scores(all_cv):
    rows = []
    for name, cv in all_cv.items():
        for fold_i in range(len(cv["test_f1"])):
            row = {"model": name, "fold": fold_i + 1}
            for metric in ["f1", "accuracy", "precision", "recall", "roc_auc"]:
                row[f"train_{metric}"] = cv[f"train_{metric}"][fold_i]
                row[f"test_{metric}"]  = cv[f"test_{metric}"][fold_i]
            row["fit_time"]   = cv["fit_time"][fold_i]
            row["score_time"] = cv["score_time"][fold_i]
            rows.append(row)
    pd.DataFrame(rows).to_csv(REPORTS / "cv_fold_scores.csv", index=False)

# ══════════════════════════════════════════════════════════════
#  STEP 1.3 & 2: Train all models on full train, evaluate on test
# ══════════════════════════════════════════════════════════════
def train_and_evaluate_all(X_train, y_train, X_test, y_test, models, all_cv, feature_names):
    from sklearn.metrics import (
        classification_report, confusion_matrix, ConfusionMatrixDisplay,
        roc_curve, auc, precision_recall_curve, average_precision_score,
        f1_score, accuracy_score, precision_score, recall_score, roc_auc_score,
    )
    import joblib

    fitted = {}
    test_results = {}
    best_name = max(all_cv, key=lambda n: all_cv[n]["test_f1"].mean())

    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        joblib.dump(model, ARTIFACTS / f"{name.lower().replace(' ', '_')}_model.joblib")

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        test_results[name] = {
            "y_pred": y_pred, "y_proba": y_proba,
            "f1":        f1_score(y_test, y_pred),
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall":    recall_score(y_test, y_pred, zero_division=0),
            "roc_auc":   roc_auc_score(y_test, y_proba) if y_proba is not None else float("nan"),
        }

    # copy best as best_model.joblib for Streamlit compat
    import shutil
    best_path = ARTIFACTS / f"{best_name.lower().replace(' ', '_')}_model.joblib"
    shutil.copy2(best_path, ARTIFACTS / "best_model.joblib")
    print(f"  best model: {best_name}")

    # ── 2.2 Classification report (best model) ───────────────
    best_pred = test_results[best_name]["y_pred"]
    report = classification_report(y_test, best_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(REPORTS / "classification_report.csv")

    # ── 2.3 Confusion matrices ────────────────────────────────
    cm = confusion_matrix(y_test, best_pred)
    pd.DataFrame(cm, index=["actual_fast", "actual_slow"], columns=["pred_fast", "pred_slow"]).to_csv(REPORTS / "confusion_matrix.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay(cm, display_labels=["fast", "slow"]).plot(ax=axes[0], cmap="Blues")
    axes[0].set_title(f"{best_name} — Confusion Matrix")
    cm_norm = confusion_matrix(y_test, best_pred, normalize="true")
    ConfusionMatrixDisplay(cm_norm, display_labels=["fast", "slow"]).plot(ax=axes[1], cmap="Oranges", values_format=".2f")
    axes[1].set_title(f"{best_name} — Normalised")
    plt.tight_layout()
    plt.savefig(REPORTS / "confusion_matrix.png")
    plt.savefig(REPORTS / "confusion_matrix_normalised.png")
    plt.close()

    # ── 2.4 ROC curves (all models overlaid) ─────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in test_results.items():
        if res["y_proba"] is not None:
            fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
            ax.plot(fpr, tpr, label=f'{name} (AUC={res["roc_auc"]:.3f})')
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(REPORTS / "roc_curve.png")
    plt.close()

    # ── 2.5 Precision-Recall curves ──────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in test_results.items():
        if res["y_proba"] is not None:
            prec, rec, _ = precision_recall_curve(y_test, res["y_proba"])
            ap = average_precision_score(y_test, res["y_proba"])
            ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Models")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(REPORTS / "pr_curve.png")
    plt.close()

    # ── 2.6 Multi-model comparison table ─────────────────────
    rows = []
    for name, res in test_results.items():
        cv = all_cv[name]
        rows.append({
            "Model": name,
            "CV F1 (mean±std)": f'{cv["test_f1"].mean():.4f} ± {cv["test_f1"].std():.4f}',
            "Test Accuracy":  round(res["accuracy"], 4),
            "Test Precision": round(res["precision"], 4),
            "Test Recall":    round(res["recall"], 4),
            "Test F1":        round(res["f1"], 4),
            "Test ROC-AUC":   round(res["roc_auc"], 4),
            "Fit Time (s)":   round(cv["fit_time"].mean(), 2),
        })
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(REPORTS / "all_models_test_comparison.csv", index=False)
    print("\n  Model comparison:")
    print(comp_df.to_string(index=False))

    # ── 2.7 Feature importance (best model) ──────────────────
    best_model = fitted[best_name]
    if hasattr(best_model, "feature_importances_"):
        imp = best_model.feature_importances_
        fi_df = pd.DataFrame({"feature": feature_names, "importance": imp})
        fi_df = fi_df.sort_values("importance", ascending=True)
        fi_df.to_csv(REPORTS / "feature_importance.csv", index=False)

        fig, ax = plt.subplots(figsize=(8, max(6, len(feature_names) * 0.3)))
        ax.barh(fi_df["feature"], fi_df["importance"], color="#4C72B0")
        ax.set_xlabel("Importance")
        ax.set_title(f"Feature Importance — {best_name}")
        plt.tight_layout()
        plt.savefig(REPORTS / "feature_importance.png")
        plt.close()

    return fitted, test_results, best_name

# ══════════════════════════════════════════════════════════════
#  STEP 3: Dataset-level statistics
# ══════════════════════════════════════════════════════════════
def dataset_stats(df, train_df, test_df, label_col):

    # ── 3.1 Class distribution ────────────────────────────────
    counts = df[label_col].value_counts().sort_index()
    label_map = {0: "fast", 1: "slow"}
    counts.index = counts.index.map(lambda x: label_map.get(x, str(x)))

    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot.bar(ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_title("Class Distribution After Labelling")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts):
        ax.text(i, v + 1, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORTS / "class_distribution.png")
    plt.close()

    # ── 3.2 Runtime distribution ──────────────────────────────
    if RAW_CSV.exists():
        raw = pd.read_csv(RAW_CSV)
        if "runtime_s" in raw.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            runtimes = raw["runtime_s"].dropna()
            ax.hist(runtimes, bins=50, color="#4C72B0", edgecolor="white", alpha=0.8)
            for q, color, label in [(0.5, "green", "p50"), (0.75, "red", "p75")]:
                val = runtimes.quantile(q)
                ax.axvline(val, color=color, linestyle="--", linewidth=2, label=f"{label} = {val:.4f}s")
            ax.set_xlabel("Runtime (seconds)")
            ax.set_ylabel("Count")
            ax.set_title("Runtime Distribution with Quantile Thresholds")
            ax.legend()
            plt.tight_layout()
            plt.savefig(REPORTS / "runtime_distribution.png")
            plt.close()

    # ── 3.3 Split summary ────────────────────────────────────
    rows = []
    for name, split_df in [("Train (CV)", train_df), ("Test", test_df)]:
        fast = (split_df[label_col] == 0).sum()
        slow = (split_df[label_col] == 1).sum()
        rows.append({"Split": name, "N queries": len(split_df), "fast": fast, "slow": slow})
    split_df_out = pd.DataFrame(rows)
    split_df_out.to_csv(REPORTS / "split_summary.csv", index=False)

    # ── 3.4 Per-database and per-difficulty ───────────────────
    if "db_id" in df.columns:
        db_counts = df.groupby("db_id").agg(
            n_queries=(label_col, "count"),
            n_fast=(label_col, lambda x: (x == 0).sum()),
            n_slow=(label_col, lambda x: (x == 1).sum()),
        ).reset_index()
        db_counts.to_csv(REPORTS / "per_database_results.csv", index=False)

    if "difficulty" in df.columns:
        diff_counts = df.groupby("difficulty").agg(
            n_queries=(label_col, "count"),
            n_fast=(label_col, lambda x: (x == 0).sum()),
            n_slow=(label_col, lambda x: (x == 1).sum()),
        ).reset_index()
        diff_counts.to_csv(REPORTS / "per_difficulty_results.csv", index=False)

# ══════════════════════════════════════════════════════════════
#  STEP 4: Advanced stats
# ══════════════════════════════════════════════════════════════
def advanced_stats(all_cv, X_train, y_train, X_test, y_test, test_results, best_name, test_df, label_col, feature_names):
    from sklearn.model_selection import StratifiedKFold, learning_curve

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── 4.1 Error analysis ────────────────────────────────────
    best_pred = test_results[best_name]["y_pred"]
    errors = test_df.copy()
    errors["predicted"] = best_pred
    errors["correct"]   = (errors[label_col] == best_pred).astype(int)
    misclassified = errors[errors["correct"] == 0]

    error_rows = []
    if "db_id" in errors.columns:
        for db, grp in misclassified.groupby("db_id"):
            error_rows.append({"group_type": "db_id", "group": db, "n_errors": len(grp)})
    if "difficulty" in errors.columns:
        for diff, grp in misclassified.groupby("difficulty"):
            error_rows.append({"group_type": "difficulty", "group": diff, "n_errors": len(grp)})
    if error_rows:
        pd.DataFrame(error_rows).to_csv(REPORTS / "error_analysis.csv", index=False)

    # ── 4.2 Learning curve ────────────────────────────────────
    try:
        best_model_fresh = build_models()[best_name]
        sizes = np.linspace(0.1, 1.0, 8)
        train_sizes, train_scores, val_scores = learning_curve(
            best_model_fresh, X_train, y_train,
            cv=skf, scoring="f1",
            train_sizes=sizes, n_jobs=-1,
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Train F1")
        ax.fill_between(train_sizes,
                        train_scores.mean(axis=1) - train_scores.std(axis=1),
                        train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.15)
        ax.plot(train_sizes, val_scores.mean(axis=1), "o-", label="Validation F1")
        ax.fill_between(train_sizes,
                        val_scores.mean(axis=1) - val_scores.std(axis=1),
                        val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.15)
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("F1 Score")
        ax.set_title(f"Learning Curve — {best_name}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(REPORTS / "learning_curve.png")
        plt.close()
    except Exception as e:
        print(f"  learning curve skipped: {e}")

    # ── 4.3 Wilcoxon significance test ────────────────────────
    model_names = list(all_cv.keys())
    if len(model_names) >= 2:
        from scipy.stats import wilcoxon
        sorted_models = sorted(model_names, key=lambda n: all_cv[n]["test_f1"].mean(), reverse=True)
        top2 = sorted_models[:2]
        try:
            stat, p = wilcoxon(all_cv[top2[0]]["test_f1"], all_cv[top2[1]]["test_f1"])
            print(f"\n  Wilcoxon {top2[0]} vs {top2[1]}: stat={stat:.4f}, p={p:.4f}")
        except Exception as e:
            print(f"  Wilcoxon test skipped: {e}")

    # ── 4.4 CV box plots ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [all_cv[n]["test_f1"] for n in model_names]
    bp = ax.boxplot(data, labels=model_names, patch_artist=True)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for patch, color in zip(bp["boxes"], colors[:len(model_names)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("F1 Score (per fold)")
    ax.set_title("Cross-Validation F1 Distribution — All Models")
    plt.tight_layout()
    plt.savefig(REPORTS / "cv_boxplot.png")
    plt.close()

    # ── 4.5 Calibration curve ────────────────────────────────
    try:
        from sklearn.calibration import calibration_curve
        best_proba = test_results[best_name]["y_proba"]
        if best_proba is not None:
            frac_pos, mean_pred = calibration_curve(y_test, best_proba, n_bins=10)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(mean_pred, frac_pos, "o-", label=best_name)
            ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfectly calibrated")
            ax.set_xlabel("Mean Predicted Probability")
            ax.set_ylabel("Fraction of Positives")
            ax.set_title(f"Calibration Curve — {best_name}")
            ax.legend()
            plt.tight_layout()
            plt.savefig(REPORTS / "calibration_curve.png")
            plt.close()
    except Exception as e:
        print(f"  calibration curve skipped: {e}")

# ══════════════════════════════════════════════════════════════
#  STEP 5: Rich text report
# ══════════════════════════════════════════════════════════════
def write_text_report(all_cv, test_results, best_name):
    lines = []
    lines.append("=" * 60)
    lines.append("MODEL RESULTS REPORT")
    lines.append("=" * 60)
    lines.append("")

    lines.append("Cross-Validation Results (5-fold Stratified):")
    lines.append("-" * 50)
    for name, cv in all_cv.items():
        lines.append(f"  {name}:")
        for metric in ["f1", "accuracy", "precision", "recall", "roc_auc"]:
            vals = cv[f"test_{metric}"]
            lines.append(f"    {metric:12s}: {vals.mean():.4f} +/- {vals.std():.4f}")
        lines.append(f"    fit_time    : {cv['fit_time'].mean():.2f}s")
        lines.append("")

    lines.append(f"Best model (by CV F1): {best_name}")
    lines.append("")

    lines.append("Held-Out Test Results:")
    lines.append("-" * 50)
    for name, res in test_results.items():
        marker = " <-- BEST" if name == best_name else ""
        lines.append(f"  {name}{marker}:")
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            lines.append(f"    {metric:12s}: {res[metric]:.4f}")
        lines.append("")

    report_text = "\n".join(lines)
    (REPORTS / "model_results.txt").write_text(report_text, encoding="utf-8")
    print(f"\n{report_text}")

# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  RERUN FULL STATS — Root Legacy Pipeline")
    print("=" * 60)

    print("\n[Step 0] Purging old artifacts...")
    purge()

    print("\n[Step 0b] Ensuring data exists...")
    ensure_data()

    print("\n[Loading] Data + split...")
    df, train_df, test_df, X_train, y_train, X_test, y_test, feat_names, label_col = load_and_split()

    print("\n[Step 1] Cross-validation (all models)...")
    models = build_models()
    all_cv, skf = run_cross_validation(X_train, y_train, models)
    save_cv_fold_scores(all_cv)

    print("\n[Step 2] Train all + evaluate on test set...")
    fitted, test_results, best_name = train_and_evaluate_all(
        X_train, y_train, X_test, y_test, models, all_cv, feat_names
    )

    print("\n[Step 3] Dataset statistics...")
    dataset_stats(df, train_df, test_df, label_col)

    print("\n[Step 4] Advanced stats...")
    advanced_stats(all_cv, X_train, y_train, X_test, y_test, test_results, best_name, test_df, label_col, feat_names)

    print("\n[Step 5] Writing text report...")
    write_text_report(all_cv, test_results, best_name)

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DONE — all artifacts written")
    print("=" * 60)
    print("\nGenerated files:")
    for f in sorted(REPORTS.glob("*")):
        print(f"  reports/{f.name}  ({f.stat().st_size / 1024:.1f} KB)")
    for f in sorted(ARTIFACTS.glob("*.joblib")):
        print(f"  artifacts/{f.name}  ({f.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
