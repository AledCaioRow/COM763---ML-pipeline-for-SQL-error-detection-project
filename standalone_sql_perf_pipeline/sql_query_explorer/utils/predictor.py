"""
predictor.py
------------
Load a per-database regression artifact and predict the *runtime in seconds*
for a raw SQL string.

Public API
----------
    result = predict_runtime(sql, db_id)
    # result["predicted_runtime_s"]  -> float, seconds
    # result["predicted_ms"]         -> float, milliseconds (convenience)
    # result["label"]                -> "slow" | "fast"  (50 ms threshold)
    # result["model_path"]           -> str

Falls back gracefully if the per-db artifact does not exist yet.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── path wiring ───────────────────────────────────────────────────────────────
_HERE          = Path(__file__).resolve().parent.parent   # sql_query_explorer/
_PIPELINE_ROOT = _HERE.parent                              # standalone_sql_perf_pipeline/
_PROJECT_ROOT  = _PIPELINE_ROOT.parent                     # repo root

if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from src.features.extract_features import extract_features  # noqa: E402
from src.schema_stats import schema_stats                    # noqa: E402

# ── artifact locations ────────────────────────────────────────────────────────
_ARTIFACT_DIR = _PIPELINE_ROOT / "artifacts" / "regression_by_db"
_MANIFEST_PATH = _ARTIFACT_DIR / "manifest.json"

# ── slow threshold (seconds) for deriving fast/slow label from predicted time ─
SLOW_THRESHOLD_S = 0.050   # 50 ms


def _load_manifest() -> dict:
    if _MANIFEST_PATH.exists():
        return json.loads(_MANIFEST_PATH.read_text())
    # Default feature order if manifest is missing
    return {
        "feature_order": [
            "n_tokens", "query_length", "n_joins", "n_tables_approx",
            "n_where_predicates", "has_group_by", "has_order_by", "has_having",
            "has_distinct", "has_limit", "has_union", "n_subqueries",
            "has_subquery", "max_nesting_depth", "n_aggregations",
            "n_count", "n_sum", "n_avg", "n_max", "n_min",
            "has_between", "has_in_clause", "has_like", "has_exists",
            "has_correlated_subquery",
            "schema_n_tables", "schema_total_rows", "schema_max_table_rows",
            "schema_total_indexes", "schema_index_coverage",
            "schema_log_total_rows",
        ]
    }


# Module-level caches so models are only loaded once per process
_MANIFEST: dict | None = None
_MODEL_CACHE: dict[str, object] = {}
_SCHEMA_CACHE: dict[str, dict] = {}


def _manifest() -> dict:
    global _MANIFEST
    if _MANIFEST is None:
        _MANIFEST = _load_manifest()
    return _MANIFEST


def _get_model(db_id: str):
    if db_id not in _MODEL_CACHE:
        path = _ARTIFACT_DIR / f"{db_id}.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"No regression artifact for '{db_id}'.\n"
                f"Run:  python standalone_sql_perf_pipeline/run_schema_stats_model.py"
            )
        _MODEL_CACHE[db_id] = joblib.load(str(path))
    return _MODEL_CACHE[db_id]


def _get_schema_stats(db_id: str) -> dict:
    if db_id not in _SCHEMA_CACHE:
        _SCHEMA_CACHE[db_id] = schema_stats(db_id)
    return _SCHEMA_CACHE[db_id]


def clear_cache() -> None:
    """Evict all loaded models so the next prediction reloads from disk."""
    _MODEL_CACHE.clear()
    _SCHEMA_CACHE.clear()
    global _MANIFEST
    _MANIFEST = None


def predict_runtime(sql: str, db_id: str) -> dict:
    """
    Predict the runtime (in seconds) for *sql* on *db_id*.

    Returns
    -------
    dict with keys:
        predicted_runtime_s  – predicted execution time in seconds
        predicted_ms         – same value in milliseconds
        label                – 'slow' if predicted >= SLOW_THRESHOLD_S else 'fast'
        model_path           – path the model was loaded from
        error                – None on success, error message string on failure
    """
    result = {
        "predicted_runtime_s": None,
        "predicted_ms":        None,
        "label":               "unknown",
        "model_path":          None,
        "error":               None,
    }

    try:
        model = _get_model(db_id)
        result["model_path"] = str(_ARTIFACT_DIR / f"{db_id}.joblib")

        # Build 31-column feature vector
        sql_feats    = extract_features(sql)
        schema_feats = _get_schema_stats(db_id)
        all_feats    = {**sql_feats, **schema_feats}

        feature_order = _manifest()["feature_order"]
        X = pd.DataFrame([all_feats]).reindex(columns=feature_order, fill_value=0)

        log_pred = float(model.predict(X)[0])
        runtime_s = float(np.exp(log_pred))

        result["predicted_runtime_s"] = runtime_s
        result["predicted_ms"]        = runtime_s * 1000
        result["label"]               = "slow" if runtime_s >= SLOW_THRESHOLD_S else "fast"

    except Exception as exc:
        result["error"] = str(exc)

    return result
