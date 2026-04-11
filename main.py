"""
COM 763 — SQL Query Performance Predictor
==========================================
Full pipeline: BIRD load/timing -> features -> ML

Usage:
    python -u main.py

Outputs (all paths relative to project root):
    data/query_dataset_raw.csv       queries + runtimes
    data/query_dataset_features.csv  queries + features + labels
    artifacts/best_model.joblib      persisted best classifier
    reports/model_results.txt        evaluation summary
"""

import os
import random
import warnings

import numpy as np

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
    evaluate_on_test,
    get_feature_importance,
    save_report,
)

warnings.filterwarnings("ignore")

def _ensure_dirs():
    for d in [DATA_DIR, ARTIFACTS_DIR, REPORTS_DIR]:
        os.makedirs(d, exist_ok=True)


def main():
    # ---- reproducibility: seed everything once at the top ----
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    _ensure_dirs()

    print("=" * 60)
    print("SQL Query Performance Predictor — Full Pipeline")
    print("=" * 60)

    # Stage 1 — load and time BIRD queries
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

    # Stage 2 — extract parsed features
    df = add_parsed_features(df)

    # Stage 3 — add slow / fast labels
    df = add_labels(df)

    # Stage 4 — train / test split + model selection via CV
    X_train, X_test, y_train, y_test, meta_test = split_data(df)
    models = build_models()
    cv_results, best_model, best_name = train_and_select(
        X_train, y_train, models,
    )

    # Stage 5 — evaluate on held-out test set
    test_metrics = evaluate_on_test(best_model, X_test, y_test, meta_test)
    feat_imp = get_feature_importance(best_model)

    # Stage 6 — save report
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
    ]:
        tag = "OK" if os.path.exists(f) else "MISSING"
        print(f"  {tag}  {f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
