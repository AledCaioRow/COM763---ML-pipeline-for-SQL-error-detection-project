"""
Stage 6 — Model training with proper train / test separation.

Workflow
--------
1. split_data()        → stratified 80 / 20 train-test split
2. build_models()      → dictionary of candidate classifiers
3. train_and_select()  → 5-fold stratified CV on *train only*,
                         refit best model, save artifact to disk
"""

import os

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin

from config import (
    RANDOM_SEED,
    FEATURE_COLS,
    TEST_SIZE,
    ARTIFACTS_DIR,
    SPLIT_METHOD,
    HOLDOUT_DATABASES,
)


# ============================================================
# DATA SPLIT
# ============================================================

def split_data(df):
    """Return train/test splits and aligned metadata DataFrame for test rows."""
    required_meta = {"db_id", "difficulty", "question_id", "sql"}
    missing = required_meta - set(df.columns)
    if missing:
        raise ValueError(
            "Missing required metadata columns in features DataFrame: "
            f"{sorted(missing)}"
        )

    X = df[FEATURE_COLS].fillna(0)
    y = df["label_binary"]
    meta = df[["question_id", "db_id", "difficulty", "sql"]].copy()

    def _label_counts(series):
        counts = series.value_counts().to_dict()
        return {"fast": int(counts.get(0, 0)), "slow": int(counts.get(1, 0))}

    if SPLIT_METHOD == "random":
        X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
            X,
            y,
            meta,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=y,
        )
        print(
            f"\n[STAGE 6a] Split method: random (stratified)"
            f"\n  train: {len(X_train)}, test: {len(X_test)} (test = {TEST_SIZE:.0%})"
        )
    elif SPLIT_METHOD == "database_aware":
        holdout_set = set(HOLDOUT_DATABASES)
        test_mask = meta["db_id"].isin(holdout_set)
        train_mask = ~test_mask

        X_train, X_test = X.loc[train_mask], X.loc[test_mask]
        y_train, y_test = y.loc[train_mask], y.loc[test_mask]
        meta_train, meta_test = meta.loc[train_mask], meta.loc[test_mask]

        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError(
                "Database-aware split produced an empty train or test set. "
                "Check HOLDOUT_DATABASES and dataset db_id coverage."
            )

        print(
            "\n[STAGE 6a] Split method: database_aware (leave-databases-out)"
            f"\n  holdout databases: {sorted(holdout_set)}"
            f"\n  train: {len(X_train)}, test: {len(X_test)}"
        )
    else:
        raise ValueError(
            f"Unsupported SPLIT_METHOD '{SPLIT_METHOD}'. "
            "Use 'random' or 'database_aware'."
        )

    train_dbs = sorted(meta_train["db_id"].unique().tolist())
    test_dbs = sorted(meta_test["db_id"].unique().tolist())
    print(f"  databases in train ({len(train_dbs)}): {train_dbs}")
    print(f"  databases in test  ({len(test_dbs)}): {test_dbs}")
    print(f"  train label balance: {_label_counts(y_train)}")
    print(f"  test label balance : {_label_counts(y_test)}")

    return X_train, X_test, y_train, y_test, meta_test.reset_index(drop=True)


# ============================================================
# MODEL REGISTRY
# ============================================================

class XGBWrapper(BaseEstimator, ClassifierMixin):
    """Thin wrapper so XGBoost plays nicely with pandas DataFrames."""

    def __init__(self, **kwargs):
        from xgboost import XGBClassifier

        self.model = XGBClassifier(**kwargs)

    def fit(self, X, y, **kw):
        self.model.fit(np.asarray(X), np.asarray(y), **kw)
        self.feature_importances_ = self.model.feature_importances_
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(np.asarray(X))

    def predict_proba(self, X):
        return self.model.predict_proba(np.asarray(X))


def build_models():
    """Return an ordered dict of name → estimator."""
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, random_state=RANDOM_SEED,
        ),
    }

    try:
        models["XGBoost"] = XGBWrapper(
            n_estimators=200, random_state=RANDOM_SEED,
            eval_metric="logloss",
        )
    except ImportError:
        print("  (XGBoost not installed — skipping)")

    return models


# ============================================================
# TRAINING + MODEL SELECTION
# ============================================================

def train_and_select(X_train, y_train, models):
    """Run stratified 5-fold CV on the *training* set, pick the best model,
    refit on the full training set, and persist to disk.

    Returns (cv_results, best_model, best_name).
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_results = {}

    print(f"[STAGE 6b] Cross-validation on {len(X_train)} training samples, "
          f"{len(FEATURE_COLS)} features")
    print(f"  Class balance: slow={int(y_train.sum())}, "
          f"fast={int(len(y_train) - y_train.sum())}")

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
        cv_results[name] = {
            "f1_mean": scores.mean(),
            "f1_std": scores.std(),
            "scores": scores,
        }
        print(f"  {name:25s}  F1 = {scores.mean():.4f} ± {scores.std():.4f}")

    best_name = max(cv_results, key=lambda k: cv_results[k]["f1_mean"])
    print(f"\n  Best model (by CV F1): {best_name}")

    best_model = models[best_name]
    best_model.fit(X_train, y_train)

    model_path = os.path.join(ARTIFACTS_DIR, "best_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"  Model saved -> {model_path}")

    return cv_results, best_model, best_name
