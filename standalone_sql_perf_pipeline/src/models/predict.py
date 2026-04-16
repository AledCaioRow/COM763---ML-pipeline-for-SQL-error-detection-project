"""
Inference — load a saved model and predict on new SQL queries.

Usage example
-------------
    from src.models.predict import predict_query

    result = predict_query("SELECT COUNT(*) FROM orders WHERE status = 'shipped';")
    print(result)
    # {'label': 'fast', 'label_binary': 0, 'probability_slow': 0.12}
"""

import os

import pandas as pd
import joblib

from config import ARTIFACTS_DIR, FEATURE_COLS
from src.features.extract_features import extract_features


def load_model(path=None):
    """Load the persisted model from disk."""
    if path is None:
        path = os.path.join(ARTIFACTS_DIR, "best_model.joblib")
    return joblib.load(path)


def predict_query(sql_text, model=None):
    """Predict whether a single SQL query will be slow or fast.

    Returns a dict with label, binary prediction, and probability.
    """
    if model is None:
        model = load_model()

    feats = extract_features(sql_text)
    X = pd.DataFrame([feats]).reindex(columns=FEATURE_COLS, fill_value=0)

    pred = int(model.predict(X)[0])
    result = {
        "label": "slow" if pred == 1 else "fast",
        "label_binary": pred,
    }

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        result["probability_slow"] = float(proba[1])

    return result


def predict_batch(sql_list, model=None):
    """Predict labels for a list of SQL strings. Returns a DataFrame."""
    if model is None:
        model = load_model()

    feat_rows = [extract_features(sql) for sql in sql_list]
    X = pd.DataFrame(feat_rows).reindex(columns=FEATURE_COLS, fill_value=0)

    preds = model.predict(X)
    out = pd.DataFrame({"sql": sql_list, "label_binary": preds})
    out["label"] = out["label_binary"].map({0: "fast", 1: "slow"})

    if hasattr(model, "predict_proba"):
        out["probability_slow"] = model.predict_proba(X)[:, 1]

    return out
