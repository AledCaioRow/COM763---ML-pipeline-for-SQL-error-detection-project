"""Live Compare: run SQL, predict runtime tier, and compare against measured runtime."""

from __future__ import annotations

import os
import sqlite3
import statistics
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from utils.paths import ensure_project_root_on_path
from utils.predictor import (
    load_cutoff_artifact,
    map_runtime_to_tier,
    predict_runtime_seconds,
)


def _sqlite_readonly_uri(db_path: str) -> str:
    """Read-only file URI for SQLite (avoids write locks; works on Windows)."""
    return Path(db_path).resolve().as_uri() + "?mode=ro"


def resolve_db_path(db_id: str, project_root: str) -> str | None:
    ensure_project_root_on_path(project_root)
    from config import BIRD_DB_DIR

    primary = os.path.join(BIRD_DB_DIR, db_id, f"{db_id}.sqlite")
    if os.path.isfile(primary):
        return primary
    candidates = [
        os.path.join(project_root, "Mini Dev", "MINIDEV", "dev_databases", db_id, f"{db_id}.sqlite"),
        os.path.join(project_root, "data", "bird", "mini_dev_data", "dev_databases", db_id, f"{db_id}.sqlite"),
    ]
    bird_root = os.environ.get("BIRD_ROOT", "").strip()
    if bird_root:
        candidates.insert(
            0,
            os.path.join(bird_root, "dev_databases", db_id, f"{db_id}.sqlite"),
        )
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


DATABASES = [
    "california_schools",
    "card_games",
    "codebase_community",
    "debit_card_specializing",
    "european_football_2",
    "formula_1",
    "student_club",
    "superhero",
    "thrombosis_prediction",
    "toxicology",
    "financial",
]


def time_query(
    db_path: str,
    sql: str,
    runs: int,
    timeout: float,
) -> dict[str, Any]:
    timings: list[float] = []
    error: str | None = None
    rows_returned = 0

    uri = _sqlite_readonly_uri(db_path)
    for i in range(runs):
        try:
            with sqlite3.connect(uri, uri=True, timeout=10.0) as conn:
                cursor = conn.cursor()
                start = time.perf_counter()
                cursor.execute(sql)
                results = cursor.fetchall()
                elapsed = time.perf_counter() - start
            if elapsed > timeout:
                return {
                    "error": f"Timeout ({timeout}s) exceeded on run {i + 1}",
                    "timings": timings,
                }
            timings.append(elapsed)
            rows_returned = len(results)
        except Exception as e:
            error = str(e)
            break

    if not timings:
        return {"error": error or "No timings collected", "timings": []}

    return {
        "timings": timings,
        "median_s": statistics.median(timings),
        "min_s": min(timings),
        "max_s": max(timings),
        "rows_returned": rows_returned,
        "error": None,
    }


def _fallback_cutoffs_from_raw(project_root: str) -> dict[str, Any] | None:
    """Build percentile tiers from raw runtimes when cutoff artifact is missing."""
    raw_path = os.path.join(project_root, "data", "query_dataset_raw.csv")
    if not os.path.isfile(raw_path):
        return None
    try:
        df = pd.read_csv(raw_path)
        runtimes = pd.to_numeric(df["runtime_s"], errors="coerce").dropna()
    except Exception:
        return None
    if runtimes.empty:
        return None

    thresholds = [
        float(runtimes.quantile(0.20)),
        float(runtimes.quantile(0.40)),
        float(runtimes.quantile(0.70)),
        float(runtimes.quantile(0.90)),
    ]
    return {
        "policy": "global_percentiles_fallback",
        "percentiles": [20, 40, 70, 90],
        "labels": ["very_fast", "fast", "moderate", "slow", "very_slow"],
        "global_thresholds_seconds": thresholds,
    }


def _predict_classifier_from_sql(sql: str, model, feature_cols: list[str], project_root: str) -> dict[str, Any]:
    """Fallback prediction path when runtime model checkpoint is unavailable."""
    ensure_project_root_on_path(project_root)
    try:
        from src.features.extract_features import extract_features
    except ModuleNotFoundError as e:
        return {"error": f"Could not import feature extractor: {e}"}

    try:
        feat = extract_features(sql) or {}
        row = {c: float(feat.get(c, 0)) for c in feature_cols}
        X = pd.DataFrame([row]).reindex(columns=feature_cols, fill_value=0)
        pred = int(model.predict(X)[0])
        out: dict[str, Any] = {"label": "slow" if pred == 1 else "fast"}
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            out["probability_slow"] = float(proba[1] if len(proba) > 1 else proba[0])
        return out
    except Exception as e:
        return {"error": str(e)}


def _tier_from_prob_slow(prob_slow: float, labels: list[str]) -> dict[str, Any]:
    """Map slow-class probability into ordered tier buckets as a proxy."""
    p = max(0.0, min(1.0, float(prob_slow)))
    idx = min(int(p * len(labels)), len(labels) - 1)
    return {
        "label": labels[idx],
        "label_index": idx,
        "proxy_source": "classifier_probability",
    }


def render_live_compare(project_root: str) -> None:
    ensure_project_root_on_path(project_root)

    import config

    from utils.model_loader import load_model
    from utils.paths import project_path

    TIMING_RUNS = getattr(config, "TIMING_RUNS", 3)
    QUERY_TIMEOUT_S = getattr(config, "QUERY_TIMEOUT_S", 30)
    feature_cols = list(getattr(config, "FEATURE_COLS", []))

    st.header("Live Compare — Actual vs Predicted")
    st.caption(
        "Run SQL on SQLite and compare measured runtime tiers to runtime-model predictions."
    )

    runtime_checkpoint = project_path(
        "sql_runtime_predictor", "artifacts", "runtime_predictor.pt"
    )
    has_runtime_predictor = os.path.isfile(runtime_checkpoint)
    runtime_cutoff_path = project_path(
        "sql_runtime_predictor", "artifacts", "runtime_tier_cutoffs.json"
    )
    runtime_cutoffs = load_cutoff_artifact(runtime_cutoff_path)
    using_fallback = False
    if runtime_cutoffs is None:
        runtime_cutoffs = _fallback_cutoffs_from_raw(project_root)
        using_fallback = runtime_cutoffs is not None

    classifier_model = None
    if not has_runtime_predictor:
        model_path = project_path("artifacts", "best_model.joblib")
        classifier_model = load_model(model_path)
        if classifier_model is None:
            st.error(
                "Runtime checkpoint is missing and classifier fallback model is also unavailable. "
                "Train one of them first."
            )
            return
        if not feature_cols:
            st.error("FEATURE_COLS missing from config.py, cannot use classifier fallback.")
            return
        st.warning(
            "Runtime predictor checkpoint not found; using classifier-probability tier proxy. "
            "Train runtime model for true runtime-tier predictions."
        )

    col1, col2 = st.columns([3, 1])
    with col2:
        db_id = st.selectbox("Database", DATABASES)
    with col1:
        sql = st.text_area("SQL Query", height=120, placeholder="SELECT * FROM ...")

    run_button = st.button("Run Comparison", type="primary", use_container_width=True)

    if not run_button or not sql.strip():
        return

    db_path = resolve_db_path(db_id, project_root)
    if not db_path:
        st.error(
            f"SQLite file not found for `{db_id}`. "
            "Check BIRD data under config.BIRD_DB_DIR or set BIRD_ROOT."
        )
        return

    with st.spinner("Executing query on SQLite..."):
        actual = time_query(db_path, sql.strip(), TIMING_RUNS, QUERY_TIMEOUT_S)

    prediction_source = "runtime_model"
    with st.spinner("Running prediction..."):
        if has_runtime_predictor:
            prediction = predict_runtime_seconds(
                project_root=project_root,
                sql=sql.strip(),
                db_id=db_id,
                checkpoint_path=runtime_checkpoint,
            )
        else:
            prediction_source = "classifier_proxy"
            prediction = _predict_classifier_from_sql(
                sql=sql.strip(),
                model=classifier_model,
                feature_cols=feature_cols,
                project_root=project_root,
            )

    st.divider()

    if actual["error"]:
        st.error(f"**SQL execution error:** {actual['error']}")
        return

    if "error" in prediction:
        st.error(f"**Prediction error:** {prediction['error']}")
        return

    actual_s = float(actual["median_s"])
    predicted_s = None
    abs_err_ms = None
    if prediction_source == "runtime_model":
        predicted_s = float(prediction["runtime_seconds"])
        abs_err_ms = abs(actual_s - predicted_s) * 1000.0

    pred_tier = None
    if prediction_source == "runtime_model" and runtime_cutoffs and predicted_s is not None:
        pred_tier = map_runtime_to_tier(predicted_s, runtime_cutoffs, db_id=db_id)
    elif prediction_source == "classifier_proxy" and runtime_cutoffs:
        labels = [str(x) for x in runtime_cutoffs.get("labels", [])]
        if labels:
            prob_slow = float(prediction.get("probability_slow", 0.5))
            pred_tier = _tier_from_prob_slow(prob_slow=prob_slow, labels=labels)
    actual_tier = map_runtime_to_tier(actual_s, runtime_cutoffs, db_id=db_id) if runtime_cutoffs else None

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Actual runtime", f"{actual_s * 1000:.2f} ms")
    if predicted_s is not None:
        c2.metric("Predicted runtime", f"{predicted_s * 1000:.2f} ms")
        c3.metric("Absolute error", f"{abs_err_ms:.2f} ms")
    else:
        c2.metric("Predicted runtime", "n/a")
        c3.metric("Absolute error", "n/a")
    c4.metric("Actual tier", actual_tier["label"] if actual_tier else "n/a")
    c5.metric("Predicted tier", pred_tier["label"] if pred_tier else "n/a")

    st.divider()
    st.subheader("Timing detail")

    tcol1, tcol2 = st.columns(2)
    with tcol1:
        st.write("**Per-run timings:**")
        for i, t in enumerate(actual["timings"]):
            st.text(f"  Run {i + 1}: {t * 1000:.3f} ms")
        st.text(f"  Median: {actual['median_s'] * 1000:.3f} ms")
        st.text(f"  Range:  {actual['min_s'] * 1000:.3f} — {actual['max_s'] * 1000:.3f} ms")

    with tcol2:
        st.write("**Prediction detail:**")
        st.text(f"  Source: {prediction_source}")
        if predicted_s is not None and abs_err_ms is not None:
            st.text(f"  Predicted seconds: {predicted_s:.6f}")
            st.text(f"  Absolute error ms: {abs_err_ms:.3f}")
        if "label" in prediction:
            st.text(f"  Classifier label:  {prediction['label']}")
        if "probability_slow" in prediction:
            st.text(f"  P(slow):           {float(prediction['probability_slow']):.3f}")
        if pred_tier:
            st.text(f"  Predicted tier idx: {pred_tier['label_index']}")
        if actual_tier:
            st.text(f"  Actual tier idx:    {actual_tier['label_index']}")

    st.divider()
    if pred_tier and actual_tier:
        idx_gap = abs(int(pred_tier["label_index"]) - int(actual_tier["label_index"]))
        if idx_gap == 0:
            st.success(
                f"Exact tier match: both predicted and measured runtime are `{actual_tier['label']}`."
            )
        elif idx_gap == 1:
            st.info(
                "Near match: predicted tier is one percentile-band away from measured tier."
            )
        else:
            st.warning(
                "Tier mismatch: predicted and measured runtimes fall in distant percentile bands."
            )
    else:
        st.info(
            "Tier cutoffs unavailable; showing runtime error only. "
            "Generate runtime tier cutoffs for percentile-based correctness."
        )

    if runtime_cutoffs:
        with st.expander("Percentile tier thresholds"):
            payload = {
                "labels": runtime_cutoffs.get("labels", []),
                "thresholds_seconds": runtime_cutoffs.get("global_thresholds_seconds", []),
                "policy": runtime_cutoffs.get("policy", "unknown"),
            }
            if "percentiles" in runtime_cutoffs:
                payload["percentiles"] = runtime_cutoffs.get("percentiles")
            st.json(payload)
            if using_fallback:
                st.caption(
                    "Using fallback cutoffs computed from `data/query_dataset_raw.csv` "
                    "(p20, p40, p70, p90)."
                )
