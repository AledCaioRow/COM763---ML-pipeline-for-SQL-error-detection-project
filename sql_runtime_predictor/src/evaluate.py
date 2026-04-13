"""
Phase 5 — Evaluate on BIRD Mini-Dev: q-error, Spearman, MAE in log-space, baselines.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from src.collect_runtimes import measure_runtime  # noqa: E402
from src.extract_features import NODE_FEATURE_DIM, extract_one  # noqa: E402
from src.model import RuntimePredictor  # noqa: E402
from src.utils import (  # noqa: E402
    load_config,
    load_jsonl,
    resolve_bird_root,
    convert_mysql_to_sqlite,
    build_runtime_cutoff_artifact,
    save_json,
)

try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
except ImportError:
    TfidfVectorizer = None
    Ridge = None


def q_error(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    eps = 1e-9
    a = np.maximum(actual, eps)
    p = np.maximum(pred, eps)
    return np.maximum(p / a, a / p)


def explain_opcode_count(conn: sqlite3.Connection, query: str) -> int:
    try:
        cur = conn.cursor()
        cur.execute(f"EXPLAIN {query}")
        return len(cur.fetchall())
    except sqlite3.Error:
        return -1


def flatten_plan_features(tree: dict) -> np.ndarray:
    vecs: list[np.ndarray] = []

    def walk(n: dict):
        vecs.append(np.array(n["features"], dtype=np.float64))
        for c in n.get("children") or []:
            walk(c)

    walk(tree)
    if not vecs:
        return np.zeros(NODE_FEATURE_DIM)
    return np.stack(vecs).mean(axis=0)


def load_bird_rows(json_path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    df = pd.read_json(json_path)
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "question_id": r.get("question_id", r.name),
                "db_id": str(r["db_id"]),
                "SQL": str(r["SQL"]),
            }
        )
    return rows


def train_baselines(
    train_features_path: Path,
    runtimes_dir: Path,
    db_dir: Path,
) -> tuple[Any | None, Any | None, Any | None, Any | None]:
    """Returns (flat_ridge, tfidf_ridge, tfidf_vectorizer, opcode_ridge) or Nones."""
    flat_X: list[np.ndarray] = []
    flat_y: list[float] = []
    if train_features_path.is_file():
        for rec in load_jsonl(train_features_path):
            if rec.get("target_runtime") is None:
                continue
            flat_X.append(
                np.concatenate(
                    [
                        flatten_plan_features(rec["plan_tree"]),
                        np.array(rec["global_features"], dtype=np.float64),
                    ]
                )
            )
            flat_y.append(math.log1p(float(rec["target_runtime"])))

    flat_model = None
    if flat_X and len(flat_X) > 50 and Ridge is not None:
        flat_model = Ridge(alpha=1.0)
        flat_model.fit(np.stack(flat_X), np.array(flat_y))

    tfidf_texts: list[str] = []
    tfidf_y: list[float] = []
    op_X: list[list[float]] = []
    op_y: list[float] = []

    if runtimes_dir.is_dir():
        for fp in sorted(runtimes_dir.glob("*.jsonl")):
            for rec in load_jsonl(fp):
                sql = rec.get("query_text")
                med = rec.get("median_runtime_seconds")
                if not sql or med is None:
                    continue
                db_name = rec["database"]
                db_path = db_dir / db_name / f"{db_name}.sqlite"
                if not db_path.is_file():
                    continue
                tfidf_texts.append(sql)
                ly = math.log1p(float(med))
                tfidf_y.append(ly)
                conn = sqlite3.connect(str(db_path))
                try:
                    oc = explain_opcode_count(conn, sql)
                finally:
                    conn.close()
                if oc >= 0:
                    op_X.append([float(oc)])
                    op_y.append(ly)

    tfidf_model = None
    vectorizer = None
    if (
        len(tfidf_texts) > 50
        and TfidfVectorizer is not None
        and Ridge is not None
    ):
        vectorizer = TfidfVectorizer(max_features=256, ngram_range=(1, 2))
        Xm = vectorizer.fit_transform(tfidf_texts)
        tfidf_model = Ridge(alpha=1.0)
        tfidf_model.fit(Xm, np.array(tfidf_y))

    opcode_model = None
    if len(op_X) > 50 and Ridge is not None:
        opcode_model = Ridge(alpha=10.0)
        opcode_model.fit(np.array(op_X), np.array(op_y))

    return flat_model, tfidf_model, vectorizer, opcode_model


def report_metrics(name: str, yp: np.ndarray, yt: np.ndarray) -> dict[str, Any]:
    mae_log = float(np.mean(np.abs(np.log1p(yp) - np.log1p(yt))))
    qe = q_error(yp, yt)
    med_q = float(np.median(qe))
    p90_q = float(np.percentile(qe, 90))
    sp = None
    if spearmanr is not None:
        sp, _ = spearmanr(yp, yt)
    med_t = float(np.median(yt))
    bin_acc = float(np.mean(((yp >= med_t) == (yt >= med_t)).astype(np.float64)))
    return {
        "name": name,
        "median_q_error": med_q,
        "p90_q_error": p90_q,
        "mae_log_runtime": mae_log,
        "spearman": None if sp is None else float(sp),
        "binary_accuracy_median_split": bin_acc,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to runtime_predictor.pt",
    )
    args = parser.parse_args()

    cfg = load_config()
    bird_root = resolve_bird_root(cfg)
    db_dir = bird_root / "dev_databases"
    jq = bird_root / "mini_dev_sqlite.json"
    use_conv = False
    if not jq.is_file():
        jq = bird_root / "mini_dev_mysql.json"
        use_conv = True
    if not jq.is_file():
        print("BIRD JSON not found.")
        sys.exit(1)

    art = Path(args.checkpoint) if args.checkpoint else _PKG / "artifacts" / "runtime_predictor.pt"
    if not art.is_file():
        print(f"No checkpoint at {art}. Train first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(art, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(art, map_location=device)
    model = RuntimePredictor(
        node_feature_dim=int(ckpt.get("node_feature_dim", NODE_FEATURE_DIM)),
        global_feature_dim=int(ckpt.get("global_feature_dim", GLOBAL_FEATURE_DIM)),
        hidden_dim=int(ckpt.get("hidden_dim", 128)),
        dropout=float(ckpt.get("dropout", 0.15)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    n_runs = int(cfg["timing_runs"])
    cache_sz = int(cfg["cache_size_pages"])
    timeout_s = float(cfg["timeout_seconds"])

    train_feat = _PKG / "data" / "features" / "train_all.jsonl"
    run_dir = _PKG / "data" / "collected_runtimes"
    flat_m, tfidf_m, vec, op_m = train_baselines(train_feat, run_dir, db_dir)

    bird_rows = load_bird_rows(jq)
    y_true: list[float] = []
    y_tree: list[float] = []
    y_flat: list[float] = []
    y_tfidf: list[float] = []
    y_op: list[float] = []
    db_ids_eval: list[str] = []

    for row in bird_rows:
        db_name = row["db_id"]
        db_path = db_dir / db_name / f"{db_name}.sqlite"
        if not db_path.is_file():
            continue
        sql = row["SQL"]
        if use_conv:
            sql = convert_mysql_to_sqlite(sql)

        med, _, _, err = measure_runtime(
            db_path, sql, n_runs, cache_sz, timeout_s
        )
        if err or med is None:
            continue

        feat = extract_one(
            db_path,
            sql,
            f"bird_{row['question_id']}",
            float(med),
            use_mysql_convert=False,
        )
        if not feat:
            continue

        g = torch.tensor(feat["global_features"], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred_log = model(feat["plan_tree"], g).cpu().numpy().reshape(-1)[0]

        y_true.append(float(med))
        y_tree.append(float(math.expm1(pred_log)))
        db_ids_eval.append(db_name)

        fx = np.concatenate(
            [
                flatten_plan_features(feat["plan_tree"]),
                np.array(feat["global_features"], dtype=np.float64),
            ]
        ).reshape(1, -1)
        if flat_m is not None:
            y_flat.append(float(math.expm1(flat_m.predict(fx)[0])))
        if tfidf_m is not None and vec is not None:
            Xt = vec.transform([sql])
            y_tfidf.append(float(math.expm1(tfidf_m.predict(Xt)[0])))
        if op_m is not None:
            conn = sqlite3.connect(str(db_path))
            try:
                oc = explain_opcode_count(conn, sql)
            finally:
                conn.close()
            if oc >= 0:
                y_op.append(float(math.expm1(op_m.predict(np.array([[float(oc)]]))[0])))
            else:
                y_op.append(float(med))

    if not y_true:
        print("No successful BIRD evaluations.")
        sys.exit(1)

    yt = np.array(y_true)
    out: dict[str, Any] = {
        "n": len(y_true),
        "tree_model": report_metrics("tree_qppnet", np.array(y_tree), yt),
    }
    if len(y_flat) == len(y_true) and flat_m is not None:
        out["flat_ridge"] = report_metrics("flat_ridge", np.array(y_flat), yt)
    if len(y_tfidf) == len(y_true) and tfidf_m is not None:
        out["tfidf_ridge"] = report_metrics("tfidf_ridge", np.array(y_tfidf), yt)
    if len(y_op) == len(y_true) and op_m is not None:
        out["explain_opcode_ridge"] = report_metrics(
            "explain_opcode_ridge", np.array(y_op), yt
        )

    cutoff_policy = str(cfg.get("cutoff_policy", "global_quantile"))
    cutoff_quantiles = list(cfg.get("cutoff_quantiles", [0.2, 0.4, 0.6, 0.8]))
    cutoff_labels = list(
        cfg.get("cutoff_labels", ["very_fast", "fast", "medium", "slow", "very_slow"])
    )
    min_samples_per_db = int(cfg.get("cutoff_min_samples_per_db", 20))
    cutoff_artifact = build_runtime_cutoff_artifact(
        policy=cutoff_policy,
        runtimes_seconds=y_true,
        db_ids=db_ids_eval,
        quantiles=cutoff_quantiles,
        labels=cutoff_labels,
        min_samples_per_db=min_samples_per_db,
    )
    cutoff_path = _PKG / "artifacts" / "runtime_tier_cutoffs.json"
    save_json(cutoff_path, cutoff_artifact)
    out["runtime_tier_cutoffs"] = {
        "path": str(cutoff_path),
        "policy": cutoff_artifact["policy"],
        "labels": cutoff_artifact["labels"],
        "quantiles": cutoff_artifact["quantiles"],
        "global_thresholds_seconds": cutoff_artifact["global_thresholds_seconds"],
        "thresholds_by_db_count": len(cutoff_artifact["thresholds_by_db_seconds"]),
    }

    rep_path = _PKG / "artifacts" / "eval_report.json"
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
