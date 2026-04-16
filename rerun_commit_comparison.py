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
from sklearn.base import clone
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
QUERY_KEY_COL = "_query_key"


def _build_logistic_regression_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
        ]
    )


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
        "Logistic Regression": _build_logistic_regression_pipeline(),
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
    fixed_model_name: str | None = None,
    fixed_model: Any | None = None,
) -> dict[str, Any]:
    if fixed_model is not None:
        best_name = fixed_model_name or fixed_model.__class__.__name__
        model = clone(fixed_model)
        cv_f1 = np.nan
    else:
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


def _normalize_sql(sql_text: str) -> str:
    return re.sub(r"\s+", " ", sql_text).strip()


def _build_query_key(row: pd.Series) -> str:
    db_id = str(row.get("db_id", ""))
    question_id = row.get("question_id")
    if pd.notna(question_id) and str(question_id).strip():
        return f"{db_id}::qid::{question_id}"
    return f"{db_id}::sql::{_normalize_sql(str(row.get('sql', '')))}"


def _load_labelled_features_df(feat_path: Path) -> pd.DataFrame:
    df = pd.read_csv(feat_path)
    if "label_binary" not in df.columns and "label" in df.columns:
        df["label_binary"] = (df["label"].astype(str).str.lower() == "slow").astype(int)
    if "label_binary" not in df.columns:
        raise ValueError(f"No label_binary/label column in {feat_path}")
    if "db_id" not in df.columns:
        raise ValueError(f"No db_id column in {feat_path}")
    df = df.copy()
    df[QUERY_KEY_COL] = df.apply(_build_query_key, axis=1)
    return df


def _get_feature_cols(df: pd.DataFrame, config_path: Path) -> list[str]:
    cfg_text = config_path.read_text(encoding="utf-8", errors="ignore")
    feature_cols_match = re.search(r"FEATURE_COLS\s*=\s*\[(.*?)\]", cfg_text, re.S)
    feature_cols = []
    if feature_cols_match:
        feature_cols = re.findall(r"\"([^\"]+)\"", feature_cols_match.group(1))
    if not feature_cols:
        feature_cols = [c for c in df.columns if c not in {"label", "label_binary", "db_id", "difficulty", "sql", "question_id", "runtime_s", QUERY_KEY_COL}]

    return [c for c in feature_cols if c in df.columns]


def _evaluate_classifier_df(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    X = df[feature_cols].fillna(0)
    y = df["label_binary"].astype(int)

    unseen_mask = df["db_id"].isin(HOLDOUT_DATABASES)
    unseen_train_n = int((~unseen_mask).sum())
    unseen_test_n = int(unseen_mask.sum())
    if unseen_train_n == 0 or unseen_test_n == 0:
        unseen = {"best_model": None, "cv_f1": np.nan, "f1": np.nan, "roc_auc": np.nan, "accuracy": np.nan}
    else:
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
        "rows": int(len(df)),
        "fast_rows": int((y == 0).sum()),
        "slow_rows": int((y == 1).sum()),
        "unseen_train_rows": unseen_train_n,
        "unseen_test_rows": unseen_test_n,
        "seen_train_rows": int(len(seen_train_df)),
        "seen_test_rows": int(len(seen_test_df)),
        "unseen_best_model": unseen["best_model"],
        "unseen_f1": unseen["f1"],
        "unseen_roc_auc": unseen["roc_auc"],
        "unseen_accuracy": unseen["accuracy"],
        "seen_best_model": seen["best_model"],
        "seen_f1": seen["f1"],
        "seen_roc_auc": seen["roc_auc"],
        "seen_accuracy": seen["accuracy"],
    }


def _evaluate_classifier_df_fixed_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    fixed_model_name: str,
    fixed_model: Any,
) -> dict[str, Any]:
    X = df[feature_cols].fillna(0)
    y = df["label_binary"].astype(int)

    unseen_mask = df["db_id"].isin(HOLDOUT_DATABASES)
    unseen_train_n = int((~unseen_mask).sum())
    unseen_test_n = int(unseen_mask.sum())
    if unseen_train_n == 0 or unseen_test_n == 0:
        unseen = {
            "best_model": fixed_model_name,
            "cv_f1": np.nan,
            "f1": np.nan,
            "roc_auc": np.nan,
            "accuracy": np.nan,
        }
    else:
        X_train_u, X_test_u = X.loc[~unseen_mask], X.loc[unseen_mask]
        y_train_u, y_test_u = y.loc[~unseen_mask], y.loc[unseen_mask]
        unseen = _evaluate_split(
            X_train_u,
            X_test_u,
            y_train_u,
            y_test_u,
            fixed_model_name=fixed_model_name,
            fixed_model=fixed_model,
        )

    seen_train_df, seen_test_df = _split_seen(df)
    X_train_s = seen_train_df[feature_cols].fillna(0)
    X_test_s = seen_test_df[feature_cols].fillna(0)
    y_train_s = seen_train_df["label_binary"].astype(int)
    y_test_s = seen_test_df["label_binary"].astype(int)
    seen = _evaluate_split(
        X_train_s,
        X_test_s,
        y_train_s,
        y_test_s,
        fixed_model_name=fixed_model_name,
        fixed_model=fixed_model,
    )

    return {
        "rows": int(len(df)),
        "fast_rows": int((y == 0).sum()),
        "slow_rows": int((y == 1).sum()),
        "unseen_train_rows": unseen_train_n,
        "unseen_test_rows": unseen_test_n,
        "seen_train_rows": int(len(seen_train_df)),
        "seen_test_rows": int(len(seen_test_df)),
        "unseen_best_model": unseen["best_model"],
        "unseen_f1": unseen["f1"],
        "unseen_roc_auc": unseen["roc_auc"],
        "unseen_accuracy": unseen["accuracy"],
        "seen_best_model": seen["best_model"],
        "seen_f1": seen["f1"],
        "seen_roc_auc": seen["roc_auc"],
        "seen_accuracy": seen["accuracy"],
    }


def _downsample_df_to_size(df: pd.DataFrame, target_rows: int) -> pd.DataFrame:
    if target_rows >= len(df):
        return df.copy().reset_index(drop=True)
    strata = (
        np.where(df["db_id"].isin(HOLDOUT_DATABASES), "unseen", "seen")
        + "|"
        + df["label_binary"].astype(int).astype(str)
    )
    stratify = None
    if target_rows >= strata.nunique():
        counts = pd.Series(strata).value_counts()
        if not counts.empty and int(counts.min()) >= 2:
            stratify = strata
    sampled, _ = train_test_split(
        df,
        train_size=target_rows,
        random_state=SEED,
        stratify=stratify,
    )
    return sampled.reset_index(drop=True)


def evaluate_commit_classifier(
    worktree: Path,
    commit: str,
    subset_query_keys: set[str] | None = None,
) -> dict[str, Any]:
    feat_path = worktree / "data" / "query_dataset_features.csv"
    config_path = worktree / "config.py"
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing features CSV in {commit}: {feat_path}")

    df = _load_labelled_features_df(feat_path)
    if subset_query_keys is not None:
        df = df[df[QUERY_KEY_COL].isin(subset_query_keys)].copy()
    feature_cols = _get_feature_cols(df, config_path)
    metrics = _evaluate_classifier_df(df, feature_cols)

    return {
        "commit": commit,
        "raw_rows": int(len(pd.read_csv(worktree / "data" / "query_dataset_raw.csv")))
        if (worktree / "data" / "query_dataset_raw.csv").exists()
        else np.nan,
        "labelled_rows": int(len(df)),
        "fast_rows": metrics["fast_rows"],
        "slow_rows": metrics["slow_rows"],
        "unseen_train_rows": metrics["unseen_train_rows"],
        "unseen_test_rows": metrics["unseen_test_rows"],
        "seen_train_rows": metrics["seen_train_rows"],
        "seen_test_rows": metrics["seen_test_rows"],
        "unseen_best_model": metrics["unseen_best_model"],
        "unseen_f1": metrics["unseen_f1"],
        "unseen_roc_auc": metrics["unseen_roc_auc"],
        "unseen_accuracy": metrics["unseen_accuracy"],
        "seen_best_model": metrics["seen_best_model"],
        "seen_f1": metrics["seen_f1"],
        "seen_roc_auc": metrics["seen_roc_auc"],
        "seen_accuracy": metrics["seen_accuracy"],
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


def _collect_tree_feature_rows(worktree: Path, commit: str) -> dict[str, Any]:
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

    df = _load_labelled_features_df(feat_csv)
    required = {"db_id", "sql"}
    if not required.issubset(df.columns):
        return {"commit": commit, "status": "unavailable", "reason": "missing db_id/sql columns"}

    db_root_candidates = [
        worktree / "Mini Dev" / "MINIDEV" / "dev_databases",
        worktree / "data" / "bird" / "mini_dev_data" / "dev_databases",
        worktree / "Mini Dev" / "MINIDEV" / "mini_dev_data" / "dev_databases",
    ]
    db_root = next((p for p in db_root_candidates if p.exists()), None)
    if db_root is None:
        return {"commit": commit, "status": "unavailable", "reason": "dev_databases not found"}

    rows: list[dict[str, Any]] = []
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
        rows.append(
            {
                "query_key": str(row[QUERY_KEY_COL]),
                "question_id": row.get("question_id"),
                "db_id": db,
                "sql": sql_text,
                "label_binary": label,
                "tree_vec": tree_vec,
                "global_vec": global_vec,
            }
        )

    if len(rows) < 20:
        return {"commit": commit, "status": "unavailable", "reason": f"insufficient extracted rows ({len(rows)})"}

    return {"commit": commit, "status": "ok", "rows": rows}


def _evaluate_tree_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y = np.array([r["label_binary"] for r in rows], dtype=int)
    db_ids = np.array([r["db_id"] for r in rows], dtype=object)
    tree_X = np.vstack([np.asarray(r["tree_vec"], dtype=float) for r in rows])
    global_X = np.vstack([np.asarray(r["global_vec"], dtype=float) for r in rows])
    both_X = np.hstack([tree_X, global_X])

    def run_one(feature_matrix: np.ndarray) -> dict[str, float]:
        unseen_mask = np.isin(db_ids, list(HOLDOUT_DATABASES))
        unseen_train_rows = int((~unseen_mask).sum())
        unseen_test_rows = int(unseen_mask.sum())
        if unseen_mask.sum() == 0 or (~unseen_mask).sum() == 0:
            return {
                "seen_f1": np.nan,
                "seen_roc_auc": np.nan,
                "unseen_f1": np.nan,
                "unseen_roc_auc": np.nan,
                "unseen_train_rows": unseen_train_rows,
                "unseen_test_rows": unseen_test_rows,
                "seen_train_rows": 0,
                "seen_test_rows": 0,
            }

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
            return {
                "seen_f1": np.nan,
                "seen_roc_auc": np.nan,
                "unseen_f1": unseen_f1,
                "unseen_roc_auc": unseen_roc,
                "unseen_train_rows": unseen_train_rows,
                "unseen_test_rows": unseen_test_rows,
                "seen_train_rows": 0,
                "seen_test_rows": 0,
            }

        seen_f1, seen_roc = fit_eval(
            feature_matrix[np.array(train_idx)],
            feature_matrix[np.array(test_idx)],
            y[np.array(train_idx)],
            y[np.array(test_idx)],
        )
        return {
            "seen_f1": seen_f1,
            "seen_roc_auc": seen_roc,
            "unseen_f1": unseen_f1,
            "unseen_roc_auc": unseen_roc,
            "unseen_train_rows": unseen_train_rows,
            "unseen_test_rows": unseen_test_rows,
            "seen_train_rows": int(len(train_idx)),
            "seen_test_rows": int(len(test_idx)),
        }

    global_only = run_one(global_X)
    both = run_one(both_X)
    return {
        "rows": int(len(rows)),
        "fast_rows": int((y == 0).sum()),
        "slow_rows": int((y == 1).sum()),
        "seen_train_rows": global_only["seen_train_rows"],
        "seen_test_rows": global_only["seen_test_rows"],
        "unseen_train_rows": global_only["unseen_train_rows"],
        "unseen_test_rows": global_only["unseen_test_rows"],
        "global_seen_f1": global_only["seen_f1"],
        "global_unseen_f1": global_only["unseen_f1"],
        "both_seen_f1": both["seen_f1"],
        "both_unseen_f1": both["unseen_f1"],
        "global_seen_roc_auc": global_only["seen_roc_auc"],
        "global_unseen_roc_auc": global_only["unseen_roc_auc"],
        "both_seen_roc_auc": both["seen_roc_auc"],
        "both_unseen_roc_auc": both["unseen_roc_auc"],
    }


def evaluate_tree_ablation(
    worktree: Path,
    commit: str,
    tree_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tree_payload = tree_payload or _collect_tree_feature_rows(worktree, commit)
    if tree_payload.get("status") != "ok":
        return tree_payload
    metrics = _evaluate_tree_rows(tree_payload["rows"])

    return {
        "commit": commit,
        "status": "ok",
        **metrics,
    }


def evaluate_tree_fairness_control(
    worktree: Path,
    commit: str,
    tree_payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tree_payload = tree_payload or _collect_tree_feature_rows(worktree, commit)
    if tree_payload.get("status") != "ok":
        return tree_payload, []

    rows = tree_payload["rows"]
    matched_query_keys = {str(r["query_key"]) for r in rows}
    matched_manifest = [
        {
            "commit": commit,
            "query_key": str(r["query_key"]),
            "question_id": r.get("question_id"),
            "db_id": r["db_id"],
            "label_binary": r["label_binary"],
            "sql": r["sql"],
        }
        for r in rows
    ]

    feat_path = worktree / "data" / "query_dataset_features.csv"
    config_path = worktree / "config.py"
    df = _load_labelled_features_df(feat_path)
    feature_cols = _get_feature_cols(df, config_path)
    matched_df = df[df[QUERY_KEY_COL].isin(matched_query_keys)].copy()
    matched_global = _evaluate_classifier_df(matched_df, feature_cols)
    tree_metrics = _evaluate_tree_rows(rows)

    return (
        {
            "commit": commit,
            "status": "ok",
            "full_labelled_rows": int(len(df)),
            "matched_rows": int(len(matched_df)),
            "matched_fast_rows": int((matched_df["label_binary"].astype(int) == 0).sum()),
            "matched_slow_rows": int((matched_df["label_binary"].astype(int) == 1).sum()),
            "matched_seen_train_rows": matched_global["seen_train_rows"],
            "matched_seen_test_rows": matched_global["seen_test_rows"],
            "matched_unseen_train_rows": matched_global["unseen_train_rows"],
            "matched_unseen_test_rows": matched_global["unseen_test_rows"],
            "matched_global_seen_best_model": matched_global["seen_best_model"],
            "matched_global_seen_f1": matched_global["seen_f1"],
            "matched_global_seen_roc_auc": matched_global["seen_roc_auc"],
            "matched_global_seen_accuracy": matched_global["seen_accuracy"],
            "matched_global_unseen_best_model": matched_global["unseen_best_model"],
            "matched_global_unseen_f1": matched_global["unseen_f1"],
            "matched_global_unseen_roc_auc": matched_global["unseen_roc_auc"],
            "matched_global_unseen_accuracy": matched_global["unseen_accuracy"],
            "tree_rows": tree_metrics["rows"],
            "tree_seen_train_rows": tree_metrics["seen_train_rows"],
            "tree_seen_test_rows": tree_metrics["seen_test_rows"],
            "tree_unseen_train_rows": tree_metrics["unseen_train_rows"],
            "tree_unseen_test_rows": tree_metrics["unseen_test_rows"],
            "tree_global_seen_f1": tree_metrics["both_seen_f1"],
            "tree_global_seen_roc_auc": tree_metrics["both_seen_roc_auc"],
            "tree_global_unseen_f1": tree_metrics["both_unseen_f1"],
            "tree_global_unseen_roc_auc": tree_metrics["both_unseen_roc_auc"],
            "sql_global_seen_f1": tree_metrics["global_seen_f1"],
            "sql_global_seen_roc_auc": tree_metrics["global_seen_roc_auc"],
            "sql_global_unseen_f1": tree_metrics["global_unseen_f1"],
            "sql_global_unseen_roc_auc": tree_metrics["global_unseen_roc_auc"],
        },
        matched_manifest,
    )


def evaluate_tree_fairness_fixed_lr(
    worktree: Path,
    commit: str,
    tree_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tree_payload = tree_payload or _collect_tree_feature_rows(worktree, commit)
    if tree_payload.get("status") != "ok":
        return tree_payload

    rows = tree_payload["rows"]
    matched_query_keys = {str(r["query_key"]) for r in rows}
    feat_path = worktree / "data" / "query_dataset_features.csv"
    config_path = worktree / "config.py"
    df = _load_labelled_features_df(feat_path)
    feature_cols = _get_feature_cols(df, config_path)
    matched_df = df[df[QUERY_KEY_COL].isin(matched_query_keys)].copy()
    matched_global = _evaluate_classifier_df_fixed_model(
        matched_df,
        feature_cols,
        fixed_model_name="Logistic Regression",
        fixed_model=_build_logistic_regression_pipeline(),
    )
    tree_metrics = _evaluate_tree_rows(rows)

    return {
        "commit": commit,
        "status": "ok",
        "fixed_model": "Logistic Regression",
        "full_labelled_rows": int(len(df)),
        "matched_rows": int(len(matched_df)),
        "matched_fast_rows": int((matched_df["label_binary"].astype(int) == 0).sum()),
        "matched_slow_rows": int((matched_df["label_binary"].astype(int) == 1).sum()),
        "matched_seen_train_rows": matched_global["seen_train_rows"],
        "matched_seen_test_rows": matched_global["seen_test_rows"],
        "matched_unseen_train_rows": matched_global["unseen_train_rows"],
        "matched_unseen_test_rows": matched_global["unseen_test_rows"],
        "matched_global_seen_f1": matched_global["seen_f1"],
        "matched_global_seen_roc_auc": matched_global["seen_roc_auc"],
        "matched_global_seen_accuracy": matched_global["seen_accuracy"],
        "matched_global_unseen_f1": matched_global["unseen_f1"],
        "matched_global_unseen_roc_auc": matched_global["unseen_roc_auc"],
        "matched_global_unseen_accuracy": matched_global["unseen_accuracy"],
        "tree_seen_train_rows": tree_metrics["seen_train_rows"],
        "tree_seen_test_rows": tree_metrics["seen_test_rows"],
        "tree_unseen_train_rows": tree_metrics["unseen_train_rows"],
        "tree_unseen_test_rows": tree_metrics["unseen_test_rows"],
        "tree_global_seen_f1": tree_metrics["both_seen_f1"],
        "tree_global_seen_roc_auc": tree_metrics["both_seen_roc_auc"],
        "tree_global_unseen_f1": tree_metrics["both_unseen_f1"],
        "tree_global_unseen_roc_auc": tree_metrics["both_unseen_roc_auc"],
    }


def evaluate_global_size_match_control(
    worktree: Path,
    commit: str,
    tree_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tree_payload = tree_payload or _collect_tree_feature_rows(worktree, commit)
    if tree_payload.get("status") != "ok":
        return tree_payload

    rows = tree_payload["rows"]
    target_rows = int(len(rows))
    feat_path = worktree / "data" / "query_dataset_features.csv"
    config_path = worktree / "config.py"
    df = _load_labelled_features_df(feat_path)
    feature_cols = _get_feature_cols(df, config_path)

    full_metrics = _evaluate_classifier_df(df, feature_cols)
    downsampled_df = _downsample_df_to_size(df, target_rows)
    downsampled_metrics = _evaluate_classifier_df(downsampled_df, feature_cols)
    tree_metrics = _evaluate_tree_rows(rows)

    return {
        "commit": commit,
        "status": "ok",
        "full_labelled_rows": int(len(df)),
        "target_rows": target_rows,
        "downsampled_rows": int(len(downsampled_df)),
        "downsampled_fast_rows": int((downsampled_df["label_binary"].astype(int) == 0).sum()),
        "downsampled_slow_rows": int((downsampled_df["label_binary"].astype(int) == 1).sum()),
        "full_seen_best_model": full_metrics["seen_best_model"],
        "full_seen_f1": full_metrics["seen_f1"],
        "full_seen_roc_auc": full_metrics["seen_roc_auc"],
        "full_seen_accuracy": full_metrics["seen_accuracy"],
        "full_unseen_best_model": full_metrics["unseen_best_model"],
        "full_unseen_f1": full_metrics["unseen_f1"],
        "full_unseen_roc_auc": full_metrics["unseen_roc_auc"],
        "full_unseen_accuracy": full_metrics["unseen_accuracy"],
        "downsampled_seen_best_model": downsampled_metrics["seen_best_model"],
        "downsampled_seen_f1": downsampled_metrics["seen_f1"],
        "downsampled_seen_roc_auc": downsampled_metrics["seen_roc_auc"],
        "downsampled_seen_accuracy": downsampled_metrics["seen_accuracy"],
        "downsampled_unseen_best_model": downsampled_metrics["unseen_best_model"],
        "downsampled_unseen_f1": downsampled_metrics["unseen_f1"],
        "downsampled_unseen_roc_auc": downsampled_metrics["unseen_roc_auc"],
        "downsampled_unseen_accuracy": downsampled_metrics["unseen_accuracy"],
        "downsampled_seen_train_rows": downsampled_metrics["seen_train_rows"],
        "downsampled_seen_test_rows": downsampled_metrics["seen_test_rows"],
        "downsampled_unseen_train_rows": downsampled_metrics["unseen_train_rows"],
        "downsampled_unseen_test_rows": downsampled_metrics["unseen_test_rows"],
        "tree_rows": tree_metrics["rows"],
        "tree_seen_f1": tree_metrics["both_seen_f1"],
        "tree_seen_roc_auc": tree_metrics["both_seen_roc_auc"],
        "tree_unseen_f1": tree_metrics["both_unseen_f1"],
        "tree_unseen_roc_auc": tree_metrics["both_unseen_roc_auc"],
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
    fairness_rows: list[dict[str, Any]] = []
    fairness_manifest_rows: list[dict[str, Any]] = []
    fairness_fixed_lr_rows: list[dict[str, Any]] = []
    size_match_rows: list[dict[str, Any]] = []

    for commit in TARGET_COMMITS:
        wt = _ensure_worktree(commit)
        if args.retime:
            regenerate_timings_for_commit(wt, commit)
        tree_payload = _collect_tree_feature_rows(wt, commit)
        commit_rows.append(evaluate_commit_classifier(wt, commit))
        ablation_rows.append(evaluate_tree_ablation(wt, commit, tree_payload=tree_payload))
        fairness_row, fairness_manifest = evaluate_tree_fairness_control(wt, commit, tree_payload=tree_payload)
        fairness_rows.append(fairness_row)
        fairness_manifest_rows.extend(fairness_manifest)
        fairness_fixed_lr_rows.append(
            evaluate_tree_fairness_fixed_lr(wt, commit, tree_payload=tree_payload)
        )
        size_match_rows.append(
            evaluate_global_size_match_control(wt, commit, tree_payload=tree_payload)
        )

    commit_df = pd.DataFrame(commit_rows)
    commit_path = REPORTS_DIR / "commit_rerun_metrics.csv"
    commit_df.to_csv(commit_path, index=False)

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_path = REPORTS_DIR / "tree_ablation_commit_metrics.csv"
    ablation_df.to_csv(ablation_path, index=False)

    fairness_df = pd.DataFrame(fairness_rows)
    fairness_path = REPORTS_DIR / "tree_fairness_control_metrics.csv"
    fairness_df.to_csv(fairness_path, index=False)

    fairness_manifest_df = pd.DataFrame(fairness_manifest_rows)
    fairness_manifest_path = REPORTS_DIR / "tree_fairness_query_manifest.csv"
    fairness_manifest_df.to_csv(fairness_manifest_path, index=False)

    fairness_fixed_lr_df = pd.DataFrame(fairness_fixed_lr_rows)
    fairness_fixed_lr_path = REPORTS_DIR / "tree_fairness_fixed_lr_metrics.csv"
    fairness_fixed_lr_df.to_csv(fairness_fixed_lr_path, index=False)

    size_match_df = pd.DataFrame(size_match_rows)
    size_match_path = REPORTS_DIR / "global_downsampled_to_tree_size_metrics.csv"
    size_match_df.to_csv(size_match_path, index=False)

    print(f"Wrote {commit_path}")
    print(f"Wrote {ablation_path}")
    print(f"Wrote {fairness_path}")
    print(f"Wrote {fairness_manifest_path}")
    print(f"Wrote {fairness_fixed_lr_path}")
    print(f"Wrote {size_match_path}")
    print(commit_df.to_string(index=False))
    print(ablation_df.to_string(index=False))
    print(fairness_df.to_string(index=False))
    print(fairness_fixed_lr_df.to_string(index=False))
    print(size_match_df.to_string(index=False))


if __name__ == "__main__":
    main()
