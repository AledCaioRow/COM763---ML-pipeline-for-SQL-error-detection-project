"""
Cross-commit rerun utility for seen/unseen evaluations.

This script evaluates the three narrative commits listed in
`markdowns/RESULTS_ACROSS_ATTEMPTS.md` using a single, consistent protocol.
"""

from __future__ import annotations

import json
import os
import re
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
WORKTREE_ROOT = ROOT / ".tmp_commit_eval"
REPORTS_DIR = ROOT / "reports"
TARGET_COMMITS = ["1a537e0", "061f0ff", "018ac87"]
HOLDOUT_DATABASES = {"financial", "formula_1"}
SEED = 42


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc.stdout.strip()


def _run_streaming(cmd: list[str], cwd: Path | None = None) -> None:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else str(ROOT),
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def _ensure_worktree(commit: str) -> Path:
    path = WORKTREE_ROOT / commit
    if path.exists():
        return path
    WORKTREE_ROOT.mkdir(exist_ok=True)
    _run(["git", "worktree", "add", "--detach", str(path), commit], cwd=ROOT)
    return path


def _best_model_by_cv(X: pd.DataFrame, y: pd.Series):
    models: dict[str, Any] = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=SEED, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, random_state=SEED
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            random_state=SEED,
            eval_metric="logloss",
            n_jobs=1,
        )
    except Exception:
        pass

    class_counts = y.value_counts()
    min_class = int(class_counts.min()) if not class_counts.empty else 0
    if min_class < 2:
        fallback = next(iter(models.items()))
        return fallback[0], fallback[1], np.nan

    n_splits = min(5, min_class)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    best_name = None
    best_model = None
    best_score = -1.0
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=None)
        score = float(np.mean(scores))
        if score > best_score:
            best_name = name
            best_model = model
            best_score = score

    assert best_name is not None and best_model is not None
    return best_name, best_model, best_score


def _safe_roc(y_true: pd.Series, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _evaluate_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    best_name, model, cv_f1 = _best_model_by_cv(X_train, y_train)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred.astype(float)
    return {
        "best_model": best_name,
        "cv_f1": float(cv_f1) if cv_f1 == cv_f1 else np.nan,
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": _safe_roc(y_test, y_prob),
        "accuracy": float(accuracy_score(y_test, y_pred)),
    }


def _split_seen(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    test_parts = []
    for _, grp in df.groupby("db_id"):
        grp = grp.sample(frac=1.0, random_state=SEED)
        y = grp["label_binary"]
        class_counts = y.value_counts()
        strat = y if (y.nunique() > 1 and int(class_counts.min()) >= 2) else None
        if len(grp) < 5:
            cutoff = max(1, int(round(len(grp) * 0.8)))
            train_parts.append(grp.iloc[:cutoff])
            test_parts.append(grp.iloc[cutoff:])
            continue
        g_train, g_test = train_test_split(
            grp, test_size=0.2, random_state=SEED, stratify=strat
        )
        train_parts.append(g_train)
        test_parts.append(g_test)
    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, test_df


def evaluate_commit_classifier(worktree: Path, commit: str) -> dict[str, Any]:
    feat_path = worktree / "data" / "query_dataset_features.csv"
    config_path = worktree / "config.py"
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing features CSV in {commit}: {feat_path}")

    df = pd.read_csv(feat_path)
    if "label_binary" not in df.columns and "label" in df.columns:
        df["label_binary"] = (df["label"].astype(str).str.lower() == "slow").astype(int)
    if "label_binary" not in df.columns:
        raise ValueError(f"No label_binary/label column in {feat_path}")
    if "db_id" not in df.columns:
        raise ValueError(f"No db_id column in {feat_path}")

    cfg_text = config_path.read_text(encoding="utf-8", errors="ignore")
    feature_cols_match = re.search(r"FEATURE_COLS\s*=\s*\[(.*?)\]", cfg_text, re.S)
    feature_cols = []
    if feature_cols_match:
        feature_cols = re.findall(r"\"([^\"]+)\"", feature_cols_match.group(1))
    if not feature_cols:
        feature_cols = [c for c in df.columns if c not in {"label", "label_binary", "db_id", "difficulty", "sql", "question_id", "runtime_s"}]

    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].fillna(0)
    y = df["label_binary"].astype(int)

    unseen_mask = df["db_id"].isin(HOLDOUT_DATABASES)
    X_train_u, X_test_u = X.loc[~unseen_mask], X.loc[unseen_mask]
    y_train_u, y_test_u = y.loc[~unseen_mask], y.loc[unseen_mask]
    unseen = _evaluate_split(X_train_u, X_test_u, y_train_u, y_test_u)

    seen_train_df, seen_test_df = _split_seen(df)
    X_train_s = seen_train_df[feature_cols].fillna(0)
    X_test_s = seen_test_df[feature_cols].fillna(0)
    y_train_s = seen_train_df["label_binary"].astype(int)
    y_test_s = seen_test_df["label_binary"].astype(int)
    seen = _evaluate_split(X_train_s, X_test_s, y_train_s, y_test_s)

    return {
        "commit": commit,
        "raw_rows": int(len(pd.read_csv(worktree / "data" / "query_dataset_raw.csv")))
        if (worktree / "data" / "query_dataset_raw.csv").exists()
        else np.nan,
        "labelled_rows": int(len(df)),
        "fast_rows": int((y == 0).sum()),
        "slow_rows": int((y == 1).sum()),
        "unseen_best_model": unseen["best_model"],
        "unseen_f1": unseen["f1"],
        "unseen_roc_auc": unseen["roc_auc"],
        "unseen_accuracy": unseen["accuracy"],
        "seen_best_model": seen["best_model"],
        "seen_f1": seen["f1"],
        "seen_roc_auc": seen["roc_auc"],
        "seen_accuracy": seen["accuracy"],
    }


def _flatten_tree_mean(tree: dict[str, Any]) -> np.ndarray:
    vectors: list[np.ndarray] = []

    def walk(node: dict[str, Any]) -> None:
        vectors.append(np.asarray(node["features"], dtype=float))
        for child in node.get("children", []) or []:
            walk(child)

    walk(tree)
    if not vectors:
        return np.zeros(1, dtype=float)
    return np.vstack(vectors).mean(axis=0)


def _db_from_query_id(query_id: str) -> str:
    if query_id.startswith("synth_"):
        parts = query_id.split("_")
        if len(parts) >= 3:
            return parts[1]
    return "unknown"


def evaluate_tree_ablation(worktree: Path, commit: str) -> dict[str, Any]:
    feat_csv = worktree / "data" / "query_dataset_features.csv"
    if not feat_csv.exists():
        return {"commit": commit, "status": "unavailable", "reason": "missing query_dataset_features.csv"}

    # Reuse plan-tree extractor from current working tree for every commit.
    srp_src = ROOT / "sql_runtime_predictor" / "src"
    if str(srp_src) not in sys.path:
        sys.path.insert(0, str(srp_src))
    try:
        from extract_features import extract_one
    except Exception as exc:
        return {"commit": commit, "status": "unavailable", "reason": f"extractor import failed: {exc}"}

    df = pd.read_csv(feat_csv)
    required = {"db_id", "sql"}
    if not required.issubset(df.columns):
        return {"commit": commit, "status": "unavailable", "reason": "missing db_id/sql columns"}
    if "label_binary" not in df.columns:
        if "label" in df.columns:
            df["label_binary"] = (df["label"].astype(str).str.lower() == "slow").astype(int)
        else:
            return {"commit": commit, "status": "unavailable", "reason": "missing labels"}

    db_root_candidates = [
        worktree / "Mini Dev" / "MINIDEV" / "dev_databases",
        worktree / "data" / "bird" / "mini_dev_data" / "dev_databases",
        worktree / "Mini Dev" / "MINIDEV" / "mini_dev_data" / "dev_databases",
    ]
    db_root = next((p for p in db_root_candidates if p.exists()), None)
    if db_root is None:
        return {"commit": commit, "status": "unavailable", "reason": "dev_databases not found"}

    rows = []
    for i, row in df.iterrows():
        db = str(row["db_id"])
        db_path = db_root / db / f"{db}.sqlite"
        if not db_path.exists():
            continue
        sql_text = str(row["sql"])
        target_runtime = float(row["runtime_s"]) if "runtime_s" in df.columns and pd.notna(row.get("runtime_s")) else None
        feat = extract_one(
            db_path=db_path,
            query=sql_text,
            query_id=f"{commit}_{i}",
            target_runtime=target_runtime,
            use_mysql_convert=False,
        )
        if not feat:
            continue
        tree_vec = _flatten_tree_mean(feat["plan_tree"])
        global_vec = np.asarray(feat["global_features"], dtype=float)
        label = int(row["label_binary"])
        rows.append((db, tree_vec, global_vec, label))

    if len(rows) < 20:
        return {"commit": commit, "status": "unavailable", "reason": f"insufficient extracted rows ({len(rows)})"}

    y = np.array([r[3] for r in rows], dtype=int)
    db_ids = np.array([r[0] for r in rows], dtype=object)
    tree_X = np.vstack([r[1] for r in rows])
    global_X = np.vstack([r[2] for r in rows])
    both_X = np.hstack([tree_X, global_X])

    def run_one(feature_matrix: np.ndarray) -> dict[str, float]:
        unseen_mask = np.isin(db_ids, list(HOLDOUT_DATABASES))
        if unseen_mask.sum() == 0 or (~unseen_mask).sum() == 0:
            return {"seen_f1": np.nan, "seen_roc_auc": np.nan, "unseen_f1": np.nan, "unseen_roc_auc": np.nan}

        def fit_eval(X_train, X_test, y_train, y_test):
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
                ]
            )
            model.fit(X_train, y_train)
            yp = model.predict(X_test)
            ypr = model.predict_proba(X_test)[:, 1]
            return float(f1_score(y_test, yp, zero_division=0)), _safe_roc(pd.Series(y_test), ypr)

        unseen_f1, unseen_roc = fit_eval(
            feature_matrix[~unseen_mask],
            feature_matrix[unseen_mask],
            y[~unseen_mask],
            y[unseen_mask],
        )

        train_idx = []
        test_idx = []
        for db in np.unique(db_ids):
            idx = np.where(db_ids == db)[0]
            if len(idx) < 4:
                continue
            y_db = y[idx]
            unique, counts = np.unique(y_db, return_counts=True)
            min_count = int(np.min(counts)) if len(counts) else 0
            strat = y_db if (len(unique) > 1 and min_count >= 2) else None
            i_train, i_test = train_test_split(
                idx, test_size=0.2, random_state=SEED, stratify=strat
            )
            train_idx.extend(i_train.tolist())
            test_idx.extend(i_test.tolist())
        if not train_idx or not test_idx:
            return {"seen_f1": np.nan, "seen_roc_auc": np.nan, "unseen_f1": unseen_f1, "unseen_roc_auc": unseen_roc}

        seen_f1, seen_roc = fit_eval(
            feature_matrix[np.array(train_idx)],
            feature_matrix[np.array(test_idx)],
            y[np.array(train_idx)],
            y[np.array(test_idx)],
        )
        return {"seen_f1": seen_f1, "seen_roc_auc": seen_roc, "unseen_f1": unseen_f1, "unseen_roc_auc": unseen_roc}

    global_only = run_one(global_X)
    both = run_one(both_X)

    return {
        "commit": commit,
        "status": "ok",
        "rows": int(len(rows)),
        "global_seen_f1": global_only["seen_f1"],
        "global_unseen_f1": global_only["unseen_f1"],
        "both_seen_f1": both["seen_f1"],
        "both_unseen_f1": both["unseen_f1"],
        "global_seen_roc_auc": global_only["seen_roc_auc"],
        "global_unseen_roc_auc": global_only["unseen_roc_auc"],
        "both_seen_roc_auc": both["seen_roc_auc"],
        "both_unseen_roc_auc": both["unseen_roc_auc"],
    }


def regenerate_timings_for_commit(worktree: Path, commit: str) -> None:
    print(f"[retime] {commit}: running main.py for fresh SQLite timings...")
    _run_streaming([sys.executable, "-u", "main.py"], cwd=worktree)
    print(f"[retime] {commit}: timing regeneration complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-commit rerun utility.")
    parser.add_argument(
        "--retime",
        action="store_true",
        help="Regenerate raw SQLite timings per commit by running main.py first.",
    )
    args = parser.parse_args()

    REPORTS_DIR.mkdir(exist_ok=True)
    commit_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []

    for commit in TARGET_COMMITS:
        wt = _ensure_worktree(commit)
        if args.retime:
            regenerate_timings_for_commit(wt, commit)
        commit_rows.append(evaluate_commit_classifier(wt, commit))
        ablation_rows.append(evaluate_tree_ablation(wt, commit))

    commit_df = pd.DataFrame(commit_rows)
    commit_path = REPORTS_DIR / "commit_rerun_metrics.csv"
    commit_df.to_csv(commit_path, index=False)

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_path = REPORTS_DIR / "tree_ablation_commit_metrics.csv"
    ablation_df.to_csv(ablation_path, index=False)

    print(f"Wrote {commit_path}")
    print(f"Wrote {ablation_path}")
    print(commit_df.to_string(index=False))
    print(ablation_df.to_string(index=False))


if __name__ == "__main__":
    main()
