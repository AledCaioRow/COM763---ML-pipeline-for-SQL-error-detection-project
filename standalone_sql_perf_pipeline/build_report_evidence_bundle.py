# -*- coding: utf-8 -*-
"""
Assemble reports/report_evidence_bundle.md and reports/report_metrics_long.csv
from pipeline outputs (Part B — one-place evidence bundle).
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
DATA = ROOT / "data"


def _src(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _add_row(
    rows: list[dict],
    bundle_key: str,
    iteration: str,
    value,
    source_file: str,
    source_field: str = "",
    notes: str = "",
) -> None:
    rows.append(
        {
            "bundle_key": bundle_key,
            "iteration": iteration,
            "value": value if value is None or isinstance(value, str) else json.dumps(value) if isinstance(value, (list, dict)) else value,
            "source_file": source_file,
            "source_field": source_field,
            "notes": notes,
        }
    )


def _refresh_cross_schema_log() -> Path:
    log_path = REPORTS / "cross_schema_regression_log.txt"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "run_full_regression.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log_path.write_text(
        proc.stdout + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else ""),
        encoding="utf-8",
    )
    return log_path


def collect_long_metrics() -> list[dict]:
    rows: list[dict] = []
    generated = _utc_now()
    _add_row(rows, "bundle_generated_at", "meta", generated, _src(ROOT / "build_report_evidence_bundle.py"), "", "")

    split_path = REPORTS / "split_summary.csv"
    if split_path.exists():
        sp = pd.read_csv(split_path)
        for _, r in sp.iterrows():
            split = str(r["split"])
            if "Train" in split:
                _add_row(rows, "iter1_split_train_n", "1", int(r["n_queries"]), _src(split_path), "n_queries", split)
                _add_row(rows, "iter1_split_train_fast", "1", int(r["fast"]), _src(split_path), "fast", split)
                _add_row(rows, "iter1_split_train_slow", "1", int(r["slow"]), _src(split_path), "slow", split)
            if "Test" in split and "Train" not in split:
                _add_row(rows, "iter1_split_test_n", "1", int(r["n_queries"]), _src(split_path), "n_queries", split)
                _add_row(rows, "iter1_split_test_fast", "1", int(r["fast"]), _src(split_path), "fast", split)
                _add_row(rows, "iter1_split_test_slow", "1", int(r["slow"]), _src(split_path), "slow", split)
                dbs = str(r["databases"])
                _add_row(rows, "iter1_holdout_dbs", "1", dbs, _src(split_path), "databases", split)

    cmp_path = REPORTS / "all_models_test_comparison.csv"
    if cmp_path.exists():
        mc = pd.read_csv(cmp_path)
        if not mc.empty and "Test F1" in mc.columns:
            best_idx = mc["Test F1"].astype(float).idxmax()
            best = mc.loc[best_idx]
            _add_row(rows, "iter1_best_model", "1", str(best["Model"]), _src(cmp_path), "Model", "by max Test F1")
            _add_row(rows, "iter1_best_test_f1", "1", float(best["Test F1"]), _src(cmp_path), "Test F1", "")
            if "Test ROC-AUC" in best:
                _add_row(rows, "iter1_best_test_roc_auc", "1", float(best["Test ROC-AUC"]), _src(cmp_path), "Test ROC-AUC", "")
            if "Test Accuracy" in best:
                _add_row(rows, "iter1_best_test_acc", "1", float(best["Test Accuracy"]), _src(cmp_path), "Test Accuracy", "")

    perdb = REPORTS / "per_database_results.csv"
    if perdb.exists():
        pdb = pd.read_csv(perdb)
        for _, r in pdb.iterrows():
            db = str(r["db_id"])
            _add_row(rows, f"iter1_perdb_{db}_f1", "1", float(r["f1"]), _src(perdb), "f1", db)
            _add_row(rows, f"iter1_perdb_{db}_support", "1", int(r["support"]), _src(perdb), "support", db)

    geom = REPORTS / "classifier_geometries_summary.csv"
    if geom.exists():
        g = pd.read_csv(geom).iloc[0]
        for col in g.index:
            if col == "commit":
                continue
            _add_row(rows, f"geometry_{col}", "1_and_2", g[col], _src(geom), col, "current worktree; unseen≈iter1 geometry, seen≈iter2")

    commit_path = REPORTS / "commit_rerun_metrics.csv"
    if commit_path.exists():
        cr = pd.read_csv(commit_path)
        for _, r in cr.iterrows():
            cmt = str(r["commit"])
            for col in cr.columns:
                if col == "commit":
                    continue
                _add_row(rows, f"commit_{cmt}_{col}", "2_multi_commit", r[col], _src(commit_path), col, cmt)

    log_path = REPORTS / "within_db_logistic_metrics.csv"
    if log_path.exists():
        lg = pd.read_csv(log_path)
        for _, r in lg.iterrows():
            db = str(r["db_id"])
            for col in ("n", "n_train", "n_test", "slow_pct", "f1_slow", "roc_auc", "accuracy", "status"):
                if col in r:
                    _add_row(rows, f"iter3_db_{db}_{col}", "3", r[col], _src(log_path), col, db)

    schema_path = REPORTS / "within_db_schema_metrics.csv"
    if schema_path.exists():
        sm = pd.read_csv(schema_path)
        for db_id, grp in sm.groupby("db_id"):
            best = grp.loc[grp["r2_log"].astype(float).idxmax()]
            _add_row(rows, f"iter4_db_{db_id}_best_model", "4", str(best["model"]), _src(schema_path), "model", "max r2_log")
            _add_row(rows, f"iter4_db_{db_id}_r2_log", "4", float(best["r2_log"]), _src(schema_path), "r2_log", "")
            _add_row(rows, f"iter4_db_{db_id}_mae_s", "4", float(best["mae_s"]), _src(schema_path), "mae_s", "")
            _add_row(rows, f"iter4_db_{db_id}_index_coverage", "4", float(best["index_coverage"]), _src(schema_path), "index_coverage", "")

    bridge_log = REPORTS / "cross_schema_regression_log.txt"
    if bridge_log.exists():
        txt = bridge_log.read_text(encoding="utf-8", errors="replace")
        _add_row(rows, "bridge_cross_schema_log_excerpt", "bridge", txt[:8000], _src(bridge_log), "", "truncate 8k; full file in reports/")

    dist_path = REPORTS / "class_distribution.csv"
    if dist_path.exists():
        dd = pd.read_csv(dist_path)
        for _, r in dd.iterrows():
            _add_row(rows, f"iter2_eda_class_{r['label']}_count", "2_eda", int(r["count"]), _src(dist_path), "count", str(r.get("label", "")))

    return rows


def _md_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df.empty:
        return "_empty_\n"
    df = df.head(max_rows).copy()
    for c in df.columns:
        df[c] = df[c].apply(lambda x: str(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) else "")
    headers = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = "\n".join("| " + " | ".join(str(row[c]) for c in df.columns) + " |" for _, row in df.iterrows())
    return "\n".join([headers, sep, body]) + "\n"


def degenerate_summary() -> str:
    feat = DATA / "query_dataset_features.csv"
    if not feat.exists():
        return "_No features CSV for degenerate-label summary._\n"
    df = pd.read_csv(feat)
    if "label_binary" not in df.columns:
        return "_No label_binary column._\n"
    lines = ["| db_id | n | slow_rate | degenerate |", "| --- | ---: | ---: | --- |"]
    for db_id, grp in df.groupby("db_id"):
        n = len(grp)
        rate = float(grp["label_binary"].mean())
        deg = rate <= 0.0 or rate >= 1.0
        lines.append(f"| {db_id} | {n} | {rate:.4f} | {deg} |")
    return "\n".join(lines) + "\n"


def write_markdown(long_df: pd.DataFrame) -> None:
    out = REPORTS / "report_evidence_bundle.md"
    lines = [
        "# Report evidence bundle (Part B)",
        "",
        f"Generated: `{_utc_now()}` (UTC). Paths relative to project root.",
        "",
        "This file consolidates **where each number lives** and how it maps to report iterations. ",
        "Machine-readable key/value rows: `reports/report_metrics_long.csv`.",
        "",
        "## Producer commands (refresh)",
        "",
        "| Iteration | Command | Primary outputs |",
        "| --- | --- | --- |",
        "| 1 | `python main.py` or `python main.py -n 1` | `reports/model_results.txt`, `all_models_test_comparison.csv`, `per_database_results.csv`, `split_summary.csv`, … |",
        "| 2 | `python main.py -n 2` or `rerun_commit_comparison.py` | `commit_rerun_metrics.csv` |",
        "| 2 (one row, current tree) | `python export_classifier_geometries_summary.py` | `classifier_geometries_summary.csv` |",
        "| 3 | `python main.py -n 3` or `run_within_db_logistic.py` | `within_db_logistic_metrics.csv` |",
        "| 4 | `python main.py -n 4` or `run_schema_stats_model.py` | `within_db_schema_metrics.csv` |",
        "| Bridge | `python main.py -n 5` or `run_full_regression.py` | stdout → `cross_schema_regression_log.txt` (captured by bundle builder) |",
        "| Figures | `python main.py -n 6` / `-n 7` | `reports/narrative_figures/` |",
        "",
        "## Iteration 1 — Global classifier (held-out DBs)",
        "",
        "| Metric | Source file | Fields |",
        "| --- | --- | --- |",
        "| Train/test counts, DB lists | `reports/split_summary.csv` | `n_queries`, `fast`, `slow`, `databases` |",
        "| Model comparison | `reports/all_models_test_comparison.csv` | `Model`, `Test F1`, `Test ROC-AUC`, … |",
        "| Narrative | `reports/model_results.txt` | summary + per-DB breakdown |",
        "| Per-DB test | `reports/per_database_results.csv` | `db_id`, `f1`, `support`, … |",
        "",
        "### Snapshot (from bundle CSV)",
        "",
    ]
    s1 = long_df[long_df["bundle_key"].str.startswith("iter1_") | long_df["bundle_key"].str.startswith("geometry_")]
    if not s1.empty:
        lines.append(_md_table(s1[["bundle_key", "value", "source_file"]], 40))
        if len(s1) > 40:
            lines.append(f"\n_… {len(s1) - 40} more rows in `report_metrics_long.csv`._\n")
    else:
        lines.append("_No iteration-1 keys found; run `main.py`._\n")

    lines.extend(
        [
            "",
            "## Iteration 2 — Seen split (pooled) + multi-commit table",
            "",
            "| Metric | Source | Fields |",
            "| --- | --- | --- |",
            "| Seen / unseen metrics | `reports/commit_rerun_metrics.csv` | `seen_*`, `unseen_*`, row counts |",
            "| **Single source of truth (current tree)** | `reports/classifier_geometries_summary.csv` | same columns, one row |",
            "",
            "### Degenerate-label check (features CSV)",
            "",
            degenerate_summary(),
            "",
            "## Iteration 3 — Per-DB logistic (SQL features)",
            "",
            "Artefact: `reports/within_db_logistic_metrics.csv`. Use rows with `status=ok`; ",
            "degenerate / skipped databases are explicit for the “ill-posed class” argument.",
            "",
        ]
    )

    s3 = long_df[long_df["bundle_key"].str.startswith("iter3_")]
    if not s3.empty:
        lines.append(_md_table(s3[["bundle_key", "value", "source_file"]], 30))
    else:
        lines.append("_Run `python run_within_db_logistic.py`._\n")

    lines.extend(
        [
            "",
            "## Iteration 4 — Per-DB regression + schema stats",
            "",
            "| Metric | Source | Fields |",
            "| --- | --- | --- |",
            "| Grid | `reports/within_db_schema_metrics.csv` | `db_id`, `model`, `r2_log`, `mae_s`, … |",
            "| Best per DB | same | row with max `r2_log` per `db_id` |",
            "| Figures | `reports/narrative_figures/fig13_*.png`, `fig14_*.png` | after `run_phase6_figures.py` |",
            "",
        ]
    )
    s4 = long_df[long_df["bundle_key"].str.startswith("iter4_")]
    if not s4.empty:
        lines.append(_md_table(s4[["bundle_key", "value", "source_file"]], 35))
    else:
        lines.append("_Run `run_schema_stats_model.py`._\n")

    lines.extend(
        [
            "",
            "## Cross-schema regression (bridge)",
            "",
            "Captured log: `reports/cross_schema_regression_log.txt` (from `run_full_regression.py`).",
            "",
            "```text",
        ]
    )
    blog = REPORTS / "cross_schema_regression_log.txt"
    if blog.exists():
        bl = blog.read_text(encoding="utf-8", errors="replace")
        lines.append(bl[:4000])
        if len(bl) > 4000:
            lines.append("\n... [truncated; see full file] ...\n")
    else:
        lines.append("(log not found — run bundle builder; it refreshes this log by default.)")
    lines.extend(["```", "", "## Side-by-side unseen vs seen (geometry summary)", "", "Use `classifier_geometries_summary.csv`: `unseen_*` aligns with held-out-DB training geometry; `seen_*` with per-DB 80/20 pooled geometry.", ""])

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


def main() -> None:
    print("Refreshing cross-schema regression log...")
    _refresh_cross_schema_log()
    rows = collect_long_metrics()
    long_df = pd.DataFrame(rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    long_path = REPORTS / "report_metrics_long.csv"
    long_df.to_csv(long_path, index=False)
    print(f"Wrote {long_path} ({len(long_df)} rows)")
    write_markdown(long_df)


if __name__ == "__main__":
    main()
