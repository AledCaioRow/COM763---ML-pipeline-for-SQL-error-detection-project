"""
retime_subset.py  (optional utility)
=====================================
Re-time the BIRD Mini-Dev queries for a subset of databases and merge the
fresh timing rows back into the pipeline's raw and features CSV files.

Useful when you want a larger or more up-to-date training set for the
per-database regression models without re-running the full pipeline.

Usage
-----
    python retime_subset.py                         # default: formula_1, financial
    python retime_subset.py --dbs formula_1         # single database
    python retime_subset.py --dbs formula_1 financial card_games

After running, re-train the per-database models with:
    python run_schema_stats_model.py
"""

import argparse
import sys
from pathlib import Path

# ── path wiring ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.append(str(PROJECT_ROOT.parent))

import numpy as np
import pandas as pd

from config import (
    BIRD_SQLITE_JSON, BIRD_DB_DIR,
    TIMING_RUNS, QUERY_TIMEOUT_S, RANDOM_SEED,
)
from src.data.load_bird import load_and_time_bird_queries
from src.features.extract_features import add_parsed_features, add_labels

RAW_CSV      = PROJECT_ROOT / "data" / "query_dataset_raw.csv"
FEATURES_CSV = PROJECT_ROOT / "data" / "query_dataset_features.csv"

DEFAULT_DBS = ["formula_1", "financial"]


# ── argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Re-time a subset of BIRD databases.")
    p.add_argument(
        "--dbs", nargs="+", default=DEFAULT_DBS,
        help="Database IDs to re-time (default: formula_1 financial)",
    )
    p.add_argument(
        "--replace", action="store_true", default=True,
        help="Replace existing rows for these db_ids (default: True)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    target_dbs = set(args.dbs)
    print(f"Re-timing databases: {sorted(target_dbs)}")
    print(f"BIRD JSON:  {BIRD_SQLITE_JSON}")
    print(f"DB dir:     {BIRD_DB_DIR}")

    # ── time the subset ───────────────────────────────────────────────────────
    full_df = load_and_time_bird_queries(
        json_path=BIRD_SQLITE_JSON,
        db_dir=BIRD_DB_DIR,
        timing_runs=TIMING_RUNS,
        timeout_s=QUERY_TIMEOUT_S,
    )
    new_rows = full_df[full_df["db_id"].isin(target_dbs)].copy()
    print(f"\nTimed {len(new_rows)} queries for {sorted(target_dbs)}")

    # ── merge into existing raw CSV ───────────────────────────────────────────
    if RAW_CSV.exists():
        old_raw = pd.read_csv(RAW_CSV)
        if args.replace:
            old_raw = old_raw[~old_raw["db_id"].isin(target_dbs)]
            print(f"Dropped old rows for {sorted(target_dbs)} from raw CSV")
        merged_raw = pd.concat([old_raw, new_rows], ignore_index=True)
    else:
        merged_raw = new_rows

    merged_raw.to_csv(RAW_CSV, index=False)
    print(f"Saved raw CSV: {RAW_CSV}  ({len(merged_raw)} total rows)")

    # ── rebuild features CSV ──────────────────────────────────────────────────
    print("\nRebuilding features CSV...")
    feats_df = add_parsed_features(merged_raw)
    feats_df = add_labels(feats_df)
    print(f"Saved features CSV: {FEATURES_CSV}  ({len(feats_df)} labelled rows)")

    print("\nDone. Re-train models with:")
    print("    python run_schema_stats_model.py")


if __name__ == "__main__":
    main()
