"""
Phase 4 — QPPNet-style recursive encoder over plan trees + global features → log1p(runtime).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from src.extract_features import GLOBAL_FEATURE_DIM, NODE_FEATURE_DIM


@dataclass
class PlanTreeBatch:
    """Single-example batch wrapper (trees processed sequentially in training loop)."""

    roots: list[dict[str, Any]]
    global_features: torch.Tensor  # (B, G)


class PlanNodeModel(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float = 0.15):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.leaf_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.internal_mlp = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward_recursive(self, node: dict[str, Any]) -> torch.Tensor:
        feat = torch.as_tensor(
            node["features"], dtype=torch.float32, device=next(self.parameters()).device
        )
        children = node.get("children") or []
        if not children:
            return self.leaf_mlp(feat)
        child_h = torch.stack(
            [self.forward_recursive(ch) for ch in children], dim=0
        )
        agg = child_h.mean(dim=0)
        x = torch.cat([feat, agg], dim=0)
        return self.internal_mlp(x)

    def forward(self, root: dict[str, Any]) -> torch.Tensor:
        return self.forward_recursive(root)


class RuntimePredictor(nn.Module):
    def __init__(
        self,
        node_feature_dim: int = NODE_FEATURE_DIM,
        global_feature_dim: int = GLOBAL_FEATURE_DIM,
        hidden_dim: int = 128,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.tree_model = PlanNodeModel(node_feature_dim, hidden_dim, dropout)
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim + global_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, plan_root: dict[str, Any], global_features: torch.Tensor) -> torch.Tensor:
        tree_repr = self.tree_model(plan_root)
        if global_features.dim() == 1:
            global_features = global_features.unsqueeze(0)
        combined = torch.cat([tree_repr.unsqueeze(0), global_features], dim=1)
        return self.output_mlp(combined).squeeze(-1)
