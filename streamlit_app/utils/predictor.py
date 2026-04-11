"""Run inference with the saved model and feature vector."""

from typing import Any, Dict, Optional

import pandas as pd


def predict_from_features(
    model,
    feature_dict: Dict[str, float],
    feature_cols: list,
) -> Optional[Dict[str, Any]]:
    if model is None:
        return None
    row = {c: float(feature_dict.get(c, 0)) for c in feature_cols}
    X = pd.DataFrame([row]).reindex(columns=feature_cols, fill_value=0)

    try:
        pred = int(model.predict(X)[0])
    except Exception:
        return None

    out: Dict[str, Any] = {
        "label": "slow" if pred == 1 else "fast",
        "label_binary": pred,
    }
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)[0]
            out["probability_slow"] = float(proba[1] if len(proba) > 1 else proba[0])
        except Exception:
            pass
    return out
