"""Inference helpers for legacy classifier and hybrid runtime display tiers."""

from __future__ import annotations

import json
import subprocess
import sys
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


def load_cutoff_artifact(path: str) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def map_runtime_to_tier(
    runtime_seconds: float,
    cutoff_artifact: Dict[str, Any],
    db_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not cutoff_artifact:
        return None
    labels = [str(x) for x in cutoff_artifact.get("labels", [])]
    global_thresholds = cutoff_artifact.get("global_thresholds_seconds", []) or []
    try:
        thresholds = [float(x) for x in global_thresholds]
    except Exception:
        return None
    if len(labels) != len(thresholds) + 1:
        return None

    source = "global"
    if (
        str(cutoff_artifact.get("policy", "")) == "per_db_quantile"
        and db_id
        and isinstance(cutoff_artifact.get("thresholds_by_db_seconds"), dict)
    ):
        db_thresholds_raw = cutoff_artifact["thresholds_by_db_seconds"].get(str(db_id))
        if isinstance(db_thresholds_raw, list):
            try:
                db_thresholds = [float(x) for x in db_thresholds_raw]
            except Exception:
                db_thresholds = []
            if len(db_thresholds) == len(thresholds):
                thresholds = db_thresholds
                source = "per_db"

    value = float(runtime_seconds)
    idx = 0
    while idx < len(thresholds) and value > thresholds[idx]:
        idx += 1
    return {
        "label": labels[idx],
        "label_index": idx,
        "threshold_source": source,
        "thresholds_seconds": thresholds,
        "labels": labels,
    }


def predict_runtime_seconds(
    project_root: str,
    sql: str,
    db_id: str,
    checkpoint_path: str,
    timeout_seconds: int = 90,
) -> Dict[str, Any]:
    payload = {
        "project_root": project_root,
        "sql": sql,
        "db_id": db_id,
        "checkpoint_path": checkpoint_path,
    }

    helper = r"""
import json
import math
import sys
from pathlib import Path

def main():
    req = json.loads(sys.stdin.read())
    project_root = Path(req["project_root"]).resolve()
    runtime_root = project_root / "sql_runtime_predictor"
    checkpoint_path = Path(req["checkpoint_path"]).resolve()
    sys.path.insert(0, str(runtime_root))

    try:
        import torch
        from src.extract_features import extract_one
        from src.model import RuntimePredictor
        from src.utils import load_config, resolve_bird_root
    except Exception as e:
        print(json.dumps({"error": f"Runtime predictor imports failed: {e}"}))
        return

    if not checkpoint_path.is_file():
        print(json.dumps({"error": f"Missing checkpoint: {checkpoint_path}"}))
        return

    cfg = load_config()
    bird_root = resolve_bird_root(cfg)
    db_id = str(req["db_id"])
    db_path = bird_root / "dev_databases" / db_id / f"{db_id}.sqlite"
    if not db_path.is_file():
        print(json.dumps({"error": f"SQLite DB not found for db_id={db_id}: {db_path}"}))
        return

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location=device)

        model = RuntimePredictor(
            node_feature_dim=int(ckpt.get("node_feature_dim", 23)),
            global_feature_dim=int(ckpt.get("global_feature_dim", 16)),
            hidden_dim=int(ckpt.get("hidden_dim", 128)),
            dropout=float(ckpt.get("dropout", 0.15)),
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        feat = extract_one(
            db_path=db_path,
            query=str(req["sql"]),
            query_id="streamlit_runtime_pred",
            target_runtime=None,
            use_mysql_convert=True,
        )
        if not feat:
            print(json.dumps({"error": "Feature extraction failed for provided SQL."}))
            return

        g = torch.tensor(feat["global_features"], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred_log = model(feat["plan_tree"], g).cpu().numpy().reshape(-1)[0]
        pred_s = float(math.expm1(float(pred_log)))
        print(json.dumps({"runtime_seconds": max(0.0, pred_s)}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()
"""

    try:
        proc = subprocess.run(
            [sys.executable, "-c", helper],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=max(10, int(timeout_seconds)),
        )
    except subprocess.TimeoutExpired:
        return {"error": "Runtime prediction process timed out."}
    except Exception as e:
        return {"error": f"Failed to start runtime predictor process: {e}"}

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        return {"error": f"Runtime predictor process failed ({proc.returncode}): {detail}"}

    raw = (proc.stdout or "").strip()
    if not raw:
        return {"error": "Runtime predictor returned empty output."}
    try:
        out = json.loads(raw)
    except Exception:
        return {"error": f"Runtime predictor returned invalid JSON: {raw}"}
    if not isinstance(out, dict):
        return {"error": "Runtime predictor returned unexpected payload."}
    return out
