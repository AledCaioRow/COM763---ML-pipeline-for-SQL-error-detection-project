"""
retrain.py
----------
Merge new query-history observations into the pipeline's training CSVs
then re-run run_schema_stats_model.py to produce fresh per-database joblibs.

Called from the "Retrain model" button on Page 2.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE          = Path(__file__).resolve().parent.parent   # sql_query_explorer/
_PIPELINE_ROOT = _HERE.parent                              # standalone_sql_perf_pipeline/
_PROJECT_ROOT  = _PIPELINE_ROOT.parent                     # repo root

if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from src.features.extract_features import extract_features  # noqa: E402
from src.schema_stats import schema_stats                    # noqa: E402
from utils.history_db import fetch_all                       # noqa: E402
from utils.predictor import clear_cache                      # noqa: E402

_RAW_CSV      = _PIPELINE_ROOT / "data" / "query_dataset_raw.csv"
_FEATURES_CSV = _PIPELINE_ROOT / "data" / "query_dataset_features.csv"
_RETRAIN_SCRIPT = _PIPELINE_ROOT / "run_schema_stats_model.py"

# Label method mirrors config.LABEL_METHOD = "quantile"
_SLOW_QUANTILE = 0.75
_FAST_QUANTILE = 0.50


def _label_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply quantile labelling and drop the mid tier."""
    p50 = df["runtime_s"].quantile(_FAST_QUANTILE)
    p75 = df["runtime_s"].quantile(_SLOW_QUANTILE)
    df["label"] = np.where(
        df["runtime_s"] >= p75, "slow",
        np.where(df["runtime_s"] <= p50, "fast", "mid"),
    )
    df = df[df["label"] != "mid"].copy()
    df["label_binary"] = (df["label"] == "slow").astype(int)
    return df


def merge_history_into_training() -> int:
    """
    Pull valid rows from query_history.db, extract SQL features + schema stats,
    and merge them into query_dataset_features.csv.

    Returns the number of new rows added (0 if nothing to add).
    """
    all_rows = fetch_all()
    new_rows = [
        r for r in all_rows
        if not r["had_error"]
        and r.get("runtime_s") is not None
        and r["runtime_s"] > 0
    ]
    if not new_rows:
        return 0

    # Build feature rows
    records = []
    for r in new_rows:
        sql   = r["sql_text"]
        db_id = r["database"]
        feats = extract_features(sql)
        s_feats = schema_stats(db_id)
        row = {
            "question_id": f"history_{r['id']}",
            "db_id":       db_id,
            "sql":         sql,
            "difficulty":  "unknown",
            "runtime_s":   r["runtime_s"],
            **feats,
            **s_feats,
        }
        records.append(row)

    new_df = pd.DataFrame(records)

    # Merge with existing features CSV
    if _FEATURES_CSV.exists():
        existing = pd.read_csv(_FEATURES_CSV)
        # Avoid duplicating rows that were already merged in a previous retrain
        existing_ids = set(existing.get("question_id", pd.Series(dtype=str)))
        new_df = new_df[~new_df["question_id"].isin(existing_ids)]
        if new_df.empty:
            return 0
        merged = pd.concat([existing, new_df], ignore_index=True)
    else:
        merged = new_df

    # Re-derive labels on the full merged set
    merged = merged[merged["runtime_s"] > 0].copy()
    merged["log_runtime"] = np.log(merged["runtime_s"])
    merged = _label_df(merged)

    merged.to_csv(_FEATURES_CSV, index=False)
    return len(new_df)


def run_schema_stats_script() -> tuple[int, str]:
    """
    Run run_schema_stats_model.py in a subprocess.
    Returns (returncode, combined stdout+stderr text).
    """
    result = subprocess.run(
        [sys.executable, str(_RETRAIN_SCRIPT)],
        capture_output=True,
        text=True,
        cwd=str(_PIPELINE_ROOT),
    )
    output = result.stdout + ("\n" + result.stderr if result.stderr.strip() else "")
    return result.returncode, output.strip()


def merge_and_retrain() -> dict:
    """
    Full retraining pipeline:
      1. Merge history queries into training CSV
      2. Run run_schema_stats_model.py (retrains + dumps new joblibs)
      3. Clear the in-process model cache so the app loads fresh artifacts

    Returns a dict with keys:
        n_added     – new rows merged
        returncode  – 0 = success
        output      – script stdout / stderr
        error       – None or error message string
    """
    result = {"n_added": 0, "returncode": -1, "output": "", "error": None}

    try:
        result["n_added"] = merge_history_into_training()
    except Exception as exc:
        result["error"] = f"Merge failed: {exc}"
        return result

    try:
        rc, out = run_schema_stats_script()
        result["returncode"] = rc
        result["output"]     = out
        if rc != 0:
            result["error"] = f"Training script exited with code {rc}"
    except Exception as exc:
        result["error"] = f"Training script failed: {exc}"
        return result

    # Evict cached models so next prediction loads the new joblibs
    clear_cache()
    return result
