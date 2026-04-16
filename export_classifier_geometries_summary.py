"""
One-row export: unseen (iter 1 geometry) vs seen (iter 2 geometry) on the current worktree.

Writes reports/classifier_geometries_summary.csv for easy copy into the report / evidence bundle.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from rerun_commit_comparison import evaluate_commit_classifier

ROOT = Path(__file__).resolve().parent


def main() -> None:
    REPORTS = ROOT / "reports"
    REPORTS.mkdir(parents=True, exist_ok=True)
    row = evaluate_commit_classifier(ROOT, "current_worktree")
    out = REPORTS / "classifier_geometries_summary.csv"
    pd.DataFrame([row]).to_csv(out, index=False)
    print(f"Wrote {out}")
    print(pd.DataFrame([row]).to_string(index=False))


if __name__ == "__main__":
    main()
