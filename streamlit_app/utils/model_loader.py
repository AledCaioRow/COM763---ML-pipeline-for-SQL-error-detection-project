"""Load persisted model and feature importances."""

from typing import List, Optional, Tuple

import joblib
import numpy as np


def load_model(path: str):
    if not path:
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def get_feature_importances(
    model,
    feature_cols: List[str],
) -> Optional[Tuple[np.ndarray, List[str]]]:
    """
    Return (importances aligned to feature_cols, feature_cols).

    Handles: tree models, sklearn Pipeline with coef_, XGBWrapper-like objects.
    """
    if model is None:
        return None

    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
        n = min(len(imp), len(feature_cols))
        return imp[:n], feature_cols[:n]

    if hasattr(model, "named_steps") and "clf" in model.named_steps:
        clf = model.named_steps["clf"]
        if hasattr(clf, "coef_"):
            imp = np.abs(np.asarray(clf.coef_).ravel())
            if imp.sum() > 0:
                imp = imp / imp.sum()
            n = min(len(imp), len(feature_cols))
            return imp[:n], feature_cols[:n]

    return None
