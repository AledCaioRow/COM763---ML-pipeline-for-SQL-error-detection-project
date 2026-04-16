"""
COM 763 — SQL Query Performance Predictor
==========================================
Single entry point for all report / evidence “iterations”.

Usage:
    python -u main.py
    python -u main.py --iteration 2

Select the default iteration by uncommenting exactly one ACTIVE_ITERATION line below
(or pass --iteration N to override without editing this file).

Iteration map (see markdowns / evidence bundle plan):
  1 — Global classifier, database-aware holdout (full BIRD pipeline)
  2 — Seen / unseen commit comparison (commit_rerun_metrics.csv, etc.)
  3 — Placeholder: per-DB logistic grid not shipped; extend when you add a script
  4 — Within-DB regression + schema stats (within_db_schema_metrics.csv)
  5 — Cross-schema regression holdout (stdout; run_full_regression.py)
  6 — Figures 13–14 (needs iteration 4 output first)
  7 — Narrative EDA figures (run_report_graphs.py)
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    RANDOM_SEED,
    DATA_DIR,
    ARTIFACTS_DIR,
    REPORTS_DIR,
    BIRD_DB_DIR,
    TIMING_RUNS,
    QUERY_TIMEOUT_S,
)

from src.data.load_bird import load_and_time_bird_queries, pick_bird_json_path
from src.features.extract_features import add_parsed_features, add_labels
from src.models.train import split_data, build_models, train_and_select
from src.evaluation.evaluate import (
    evaluate_models_on_test,
    get_feature_importance,
    save_report,
)

warnings.filterwarnings("ignore")

# =============================================================================
# Default iteration — uncomment exactly ONE line (or use: python main.py -n 4)
# =============================================================================
ACTIVE_ITERATION = 1
# ACTIVE_ITERATION = 2
# ACTIVE_ITERATION = 3
# ACTIVE_ITERATION = 4
# ACTIVE_ITERATION = 5
# ACTIVE_ITERATION = 6
# ACTIVE_ITERATION = 7
# =============================================================================

_ROOT = Path(__file__).resolve().parent


def _ensure_dirs():
    for d in [DATA_DIR, ARTIFACTS_DIR, REPORTS_DIR]:
        os.makedirs(d, exist_ok=True)


def _run_standalone_script(relative_name: str) -> None:
    """Run a repo-root script in its own interpreter (avoids import-time side effects)."""
    script = _ROOT / relative_name
    if not script.is_file():
        raise FileNotFoundError(f"Missing script: {script}")
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(_ROOT),
        check=False,
    )
    if proc.returncode != 0:
        sys.exit(proc.returncode)


def run_iteration_1() -> None:
    """Iteration 1 — held-out databases, global classifier (database_aware split)."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    _ensure_dirs()

    print("=" * 60)
    print("Iteration 1 — SQL Query Performance Predictor (full pipeline)")
    print("=" * 60)

    try:
        json_path = pick_bird_json_path()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        print("Run `python setup_bird.py` to verify required BIRD files.")
        return
    df = load_and_time_bird_queries(
        json_path=json_path,
        db_dir=BIRD_DB_DIR,
        timing_runs=TIMING_RUNS,
        timeout_s=QUERY_TIMEOUT_S,
    )
    if df.empty:
        print("ERROR: No queries were timed successfully.")
        print("Check that .sqlite files exist under:")
        print(f"  {BIRD_DB_DIR}")
        print("Run `python setup_bird.py` and ensure each db folder has <db_id>.sqlite.")
        return

    raw_path = os.path.join(DATA_DIR, "query_dataset_raw.csv")
    df.to_csv(raw_path, index=False)
    print(f"  Saved raw dataset -> {raw_path}")

    df = add_parsed_features(df)
    df = add_labels(df)

    X_train, X_test, y_train, y_test, meta_train, meta_test = split_data(df)
    models = build_models()
    train_outputs = train_and_select(
        X_train, y_train, models,
    )
    cv_results = train_outputs["cv_results"]
    best_name = train_outputs["best_name"]
    fitted_models = train_outputs["fitted_models"]

    test_metrics = evaluate_models_on_test(
        fitted_models=fitted_models,
        best_name=best_name,
        cv_results=cv_results,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_meta=meta_train,
        test_meta=meta_test,
        raw_df=pd.read_csv(raw_path),
        labeled_df=df,
    )
    feat_imp = get_feature_importance(train_outputs["best_model"])

    save_report(
        cv_results, test_metrics, best_name, feat_imp,
        train_size=len(X_train), test_size=len(X_test),
    )

    print("\n" + "=" * 60)
    print("DONE.  Generated artefacts:")
    for f in [
        os.path.join(DATA_DIR, "query_dataset_raw.csv"),
        os.path.join(DATA_DIR, "query_dataset_features.csv"),
        os.path.join(ARTIFACTS_DIR, "best_model.joblib"),
        os.path.join(REPORTS_DIR, "model_results.txt"),
        os.path.join(REPORTS_DIR, "per_database_results.csv"),
        os.path.join(REPORTS_DIR, "per_difficulty_results.csv"),
        os.path.join(REPORTS_DIR, "all_models_test_comparison.csv"),
        os.path.join(REPORTS_DIR, "classification_report.csv"),
        os.path.join(REPORTS_DIR, "cv_fold_scores.csv"),
        os.path.join(REPORTS_DIR, "confusion_matrix.csv"),
        os.path.join(REPORTS_DIR, "confusion_matrix.png"),
        os.path.join(REPORTS_DIR, "confusion_matrix_normalised.png"),
        os.path.join(REPORTS_DIR, "roc_curve.png"),
        os.path.join(REPORTS_DIR, "pr_curve.png"),
        os.path.join(REPORTS_DIR, "feature_importance.csv"),
        os.path.join(REPORTS_DIR, "feature_importance.png"),
        os.path.join(REPORTS_DIR, "class_distribution.png"),
        os.path.join(REPORTS_DIR, "runtime_distribution.png"),
        os.path.join(REPORTS_DIR, "cv_boxplot.png"),
        os.path.join(REPORTS_DIR, "learning_curve.png"),
        os.path.join(REPORTS_DIR, "split_summary.csv"),
    ]:
        tag = "OK" if os.path.exists(f) else "MISSING"
        print(f"  {tag}  {f}")
    print("=" * 60)


def run_iteration_2() -> None:
    """Iteration 2 — pooled seen split / commit comparison (rerun_commit_comparison)."""
    from rerun_commit_comparison import main as rerun_main

    rerun_main()


def run_iteration_3() -> None:
    """Iteration 3 — per-DB SQL-only LogisticRegression (within_db_logistic_metrics.csv)."""
    _run_standalone_script("run_within_db_logistic.py")


def run_iteration_4() -> None:
    """Iteration 4 — within-DB regression with schema statistics."""
    _run_standalone_script("run_schema_stats_model.py")


def run_iteration_5() -> None:
    """Cross-schema regression bridge (financial + formula_1 holdout)."""
    _run_standalone_script("run_full_regression.py")


def run_iteration_6() -> None:
    """Figures 13–14 (expects reports/within_db_schema_metrics.csv from iteration 4)."""
    _run_standalone_script("run_phase6_figures.py")


def run_iteration_7() -> None:
    """Narrative EDA / comparison figures under reports/narrative_figures/."""
    _run_standalone_script("run_report_graphs.py")


_ITER_DISPATCH = {
    1: run_iteration_1,
    2: run_iteration_2,
    3: run_iteration_3,
    4: run_iteration_4,
    5: run_iteration_5,
    6: run_iteration_6,
    7: run_iteration_7,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="COM763 unified pipeline / report iterations.")
    parser.add_argument(
        "-n",
        "--iteration",
        type=int,
        choices=sorted(_ITER_DISPATCH.keys()),
        default=None,
        help="Run this iteration (overrides ACTIVE_ITERATION in main.py).",
    )
    args = parser.parse_args()
    it = args.iteration if args.iteration is not None else ACTIVE_ITERATION

    if it not in _ITER_DISPATCH:
        print(f"Unknown iteration {it}; choose 1–7.")
        sys.exit(2)

    print(f"--- Running iteration {it} ---")
    _ITER_DISPATCH[it]()


if __name__ == "__main__":
    main()
