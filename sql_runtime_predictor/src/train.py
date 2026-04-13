"""
Train RuntimePredictor on JSONL feature files (synthetic, timed).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from src.extract_features import GLOBAL_FEATURE_DIM, NODE_FEATURE_DIM  # noqa: E402
from src.model import RuntimePredictor  # noqa: E402
from src.utils import ensure_data_dirs, load_config, load_jsonl  # noqa: E402


def _db_from_record(rec: dict) -> str:
    qid = rec.get("query_id", "")
    if qid.startswith("synth_"):
        parts = qid.split("_")
        if len(parts) >= 3:
            return parts[1]
    return "unknown"


def stratified_split(
    records: list[dict], val_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    by_db: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_db[_db_from_record(r)].append(r)
    rng = random.Random(seed)
    train, val = [], []
    for rows in by_db.values():
        rng.shuffle(rows)
        if len(rows) == 1:
            train.extend(rows)
            continue
        n_val = max(1, int(len(rows) * val_fraction))
        n_val = min(n_val, len(rows) - 1)
        val.extend(rows[:n_val])
        train.extend(rows[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def train_epoch(
    model: RuntimePredictor,
    opt: torch.optim.Optimizer,
    records: list[dict],
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for rec in records:
        target = rec.get("target_runtime")
        if target is None:
            continue
        y = torch.tensor([math.log1p(float(target))], device=device)
        g = torch.tensor(rec["global_features"], dtype=torch.float32, device=device)
        pred = model(rec["plan_tree"], g)
        loss = nn.functional.mse_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(
    model: RuntimePredictor, records: list[dict], device: torch.device
) -> float:
    model.eval()
    total = 0.0
    n = 0
    for rec in records:
        target = rec.get("target_runtime")
        if target is None:
            continue
        y = torch.tensor([math.log1p(float(target))], device=device)
        g = torch.tensor(rec["global_features"], dtype=torch.float32, device=device)
        pred = model(rec["plan_tree"], g)
        loss = nn.functional.mse_loss(pred, y)
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Path to train_all.jsonl (default: data/features/train_all.jsonl)",
    )
    args = parser.parse_args()

    cfg = load_config()
    dirs = ensure_data_dirs(_PKG, cfg)
    feat_path = (
        Path(args.features)
        if args.features
        else dirs["features"] / "train_all.jsonl"
    )
    if not feat_path.is_file():
        print(f"Missing {feat_path}. Run extract_features after collect_runtimes.")
        sys.exit(1)

    records = load_jsonl(feat_path)
    records = [r for r in records if r.get("target_runtime") is not None]
    if len(records) < 10:
        print("Not enough training rows with target_runtime.")
        sys.exit(1)

    seed = int(cfg.get("random_seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    train_frac = float(cfg.get("train_fraction", 0.8))
    train_r, val_r = stratified_split(records, 1.0 - train_frac, seed)
    if not val_r and len(train_r) > 5:
        n = max(1, len(train_r) // 10)
        val_r = train_r[:n]
        train_r = train_r[n:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RuntimePredictor(
        node_feature_dim=NODE_FEATURE_DIM,
        global_feature_dim=GLOBAL_FEATURE_DIM,
        hidden_dim=int(cfg["hidden_dim"]),
        dropout=float(cfg["dropout"]),
    ).to(device)

    opt = AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    epochs = int(cfg["epochs"])
    sched = CosineAnnealingLR(opt, T_max=max(epochs, 1))

    best_val = float("inf")
    patience = int(cfg.get("early_stopping_patience", 12))
    bad = 0
    art_dir = dirs["artifacts"]
    best_path = art_dir / "runtime_predictor.pt"

    for ep in range(epochs):
        random.shuffle(train_r)
        tr_loss = train_epoch(model, opt, train_r, device)
        va_loss = eval_epoch(model, val_r, device)
        sched.step()
        print(f"epoch {ep+1}/{epochs}  train_mse_log {tr_loss:.6f}  val {va_loss:.6f}")
        if va_loss < best_val:
            best_val = va_loss
            bad = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "node_feature_dim": NODE_FEATURE_DIM,
                    "global_feature_dim": GLOBAL_FEATURE_DIM,
                    "hidden_dim": int(cfg["hidden_dim"]),
                    "dropout": float(cfg["dropout"]),
                },
                best_path,
            )
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    meta = {"best_val_mse_log": best_val, "checkpoint": str(best_path)}
    (art_dir / "train_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"Saved checkpoint to {best_path}")


if __name__ == "__main__":
    main()
