"""Load CSVs, parse model_results.txt, load pipeline config without name clashes."""

import importlib.util
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd


def load_pipeline_config(project_root: str) -> Any:
    """Load ../config.py as a distinct module (avoids `config` name clashes)."""
    path = os.path.join(project_root, "config.py")
    if not os.path.isfile(path):
        return None
    spec = importlib.util.spec_from_file_location("sqpp_pipeline_config", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def read_csv_safe(path: str) -> Optional[pd.DataFrame]:
    if not path or not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def parse_model_results(filepath: str) -> Dict[str, Any]:
    """
    Parse reports/model_results.txt into a dict.

    Tolerates UTF-8 mojibake where ± becomes in CV lines.
    """
    out: Dict[str, Any] = {
        "train_size": None,
        "test_size": None,
        "cv_models": [],  # list of {name, f1_mean, f1_std}
        "best_model": None,
        "test_f1": None,
        "test_roc_auc": None,
        "test_accuracy": None,
        "class_metrics": {},  # fast/slow -> {precision, recall, f1}
        "top_features": [],  # list of (name, score) from report file if present
        "raw_text": "",
    }
    if not filepath or not os.path.isfile(filepath):
        return out

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    out["raw_text"] = text

    m_train = re.search(r"Train set\s*:\s*(\d+)", text)
    m_test = re.search(r"Test\s+set\s*:\s*(\d+)", text)
    if m_train:
        out["train_size"] = int(m_train.group(1))
    if m_test:
        out["test_size"] = int(m_test.group(1))

    # CV block: lines like "  XGBoost                    F1 = 0.5362 ± 0.0515"
    cv_section = re.search(
        r"Cross-Validation Results.*?\n-{5,}\n(.*?)\n\s*\nBest model:",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if cv_section:
        block = cv_section.group(1)
        for line in block.splitlines():
            line = line.rstrip()
            if "F1" not in line or "=" not in line:
                continue
            parts = line.split("F1 =", 1)
            if len(parts) != 2:
                continue
            name = parts[0].strip()
            nums = re.findall(r"[\d.]+", parts[1])
            if len(nums) >= 2 and name:
                out["cv_models"].append(
                    {
                        "name": name,
                        "f1_mean": float(nums[0]),
                        "f1_std": float(nums[1]),
                    }
                )

    m_best = re.search(r"Best model:\s*(.+)", text)
    if m_best:
        out["best_model"] = m_best.group(1).strip()

    m_f1 = re.search(r"F1 Score\s*:\s*([\d.]+)", text)
    if m_f1:
        out["test_f1"] = float(m_f1.group(1))

    m_auc = re.search(r"ROC-AUC\s*:\s*([\d.]+)", text)
    if m_auc:
        out["test_roc_auc"] = float(m_auc.group(1))

    m_acc = re.search(r"^\s*accuracy\s+([\d.]+)", text, re.MULTILINE | re.IGNORECASE)
    if m_acc:
        out["test_accuracy"] = float(m_acc.group(1))

    for cls in ("fast", "slow"):
        pat = rf"^\s*{cls}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$"
        m_cls = re.search(pat, text, re.MULTILINE | re.IGNORECASE)
        if m_cls:
            out["class_metrics"][cls.lower()] = {
                "precision": float(m_cls.group(1)),
                "recall": float(m_cls.group(2)),
                "f1": float(m_cls.group(3)),
                "support": int(m_cls.group(4)),
            }

    top_section = re.search(
        r"Top 10 Features\s*\n-{5,}\n(.*?)(?:\n\s*\n===|\n\s*\nNote:|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if top_section:
        for line in top_section.group(1).splitlines():
            mtf = re.match(r"^\s{2,}(\S+)\s+([\d.]+)\s*$", line)
            if mtf:
                out["top_features"].append((mtf.group(1), float(mtf.group(2))))

    return out


def check_artifacts(root: str) -> Dict[str, bool]:
    """Which expected files exist under project root."""
    paths = {
        "raw_csv": os.path.join(root, "data", "query_dataset_raw.csv"),
        "features_csv": os.path.join(root, "data", "query_dataset_features.csv"),
        "model_results": os.path.join(root, "reports", "model_results.txt"),
        "per_db": os.path.join(root, "reports", "per_database_results.csv"),
        "per_diff": os.path.join(root, "reports", "per_difficulty_results.csv"),
        "model_joblib": os.path.join(root, "artifacts", "best_model.joblib"),
        "config_py": os.path.join(root, "config.py"),
    }
    return {k: os.path.isfile(p) for k, p in paths.items()}


def artifact_paths(root: str) -> Dict[str, str]:
    return {
        "raw_csv": os.path.join(root, "data", "query_dataset_raw.csv"),
        "features_csv": os.path.join(root, "data", "query_dataset_features.csv"),
        "model_results": os.path.join(root, "reports", "model_results.txt"),
        "per_db": os.path.join(root, "reports", "per_database_results.csv"),
        "per_diff": os.path.join(root, "reports", "per_difficulty_results.csv"),
        "model_joblib": os.path.join(root, "artifacts", "best_model.joblib"),
    }
