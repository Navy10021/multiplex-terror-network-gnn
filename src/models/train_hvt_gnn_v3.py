"""train_hvt_gnn_v3.py

High Value Target (HVT) node classification for multiplex PyG graphs built by
`build_pyg_dataset_v3.py`.

v3 upgrades (vs train_hvt_gnn.py):
  - Robust torch.load for PyTorch 2.6+ (weights_only default).
  - Optional node-feature augmentation from edge_attr WITHOUT double log1p.
  - Optional filtering of false/copied edges if those flags exist on Data.
  - Early stopping on validation ROC-AUC.

Expected Data fields:
  x, edge_index, edge_type, (optional) edge_attr,
  y_hvt, train_mask, val_mask, test_mask,
  layer_type_mapping (dict).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv


# -----------------------------
# Repro / IO
# -----------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pyg_data(path: str, map_location: str = "cpu") -> Data:
    """Compatibility loader for torch>=2.6."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _auto_edge_attr_transform(edge_attr: torch.Tensor) -> str:
    """Heuristic: if edge_attr values look raw/heavy-tailed, log1p; else none."""
    if edge_attr is None:
        return "none"
    ea = edge_attr.detach().view(-1)
    if ea.numel() == 0:
        return "none"
    # If max is extremely large, likely raw amounts / counts.
    # v3 builder generally outputs log-scaled weights; those tend to be < ~25.
    mx = float(ea.max().item())
    return "log1p" if mx > 30.0 else "none"


def apply_edge_attr_transform(edge_attr: torch.Tensor, mode: str) -> torch.Tensor:
    if edge_attr is None:
        return edge_attr
    if edge_attr.dim() == 2 and edge_attr.size(1) == 1:
        ea = edge_attr[:, 0]
    else:
        ea = edge_attr.view(-1)

    if mode == "none":
        out = ea.float()
    elif mode == "log1p":
        out = torch.log1p(ea.float().clamp(min=0.0))
    elif mode == "auto":
        return apply_edge_attr_transform(edge_attr, _auto_edge_attr_transform(edge_attr))
    else:
        raise ValueError(f"Unknown edge_attr_transform: {mode}")

    return out


def filter_edges(data: Data, *, drop_false: bool, drop_copied: bool) -> Data:
    """Optionally drop edges flagged as false/copied (if present)."""
    keep = torch.ones(data.edge_index.size(1), dtype=torch.bool)

    if drop_false and hasattr(data, "edge_is_false") and data.edge_is_false is not None:
        keep &= (data.edge_is_false.view(-1).to(torch.bool) == 0)

    if drop_copied and hasattr(data, "edge_is_copied") and data.edge_is_copied is not None:
        keep &= (data.edge_is_copied.view(-1).to(torch.bool) == 0)

    if keep.all():
        return data

    data.edge_index = data.edge_index[:, keep]
    data.edge_type = data.edge_type[keep]
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        data.edge_attr = data.edge_attr[keep]
    if hasattr(data, "edge_is_false") and data.edge_is_false is not None:
        data.edge_is_false = data.edge_is_false[keep]
    if hasattr(data, "edge_is_copied") and data.edge_is_copied is not None:
        data.edge_is_copied = data.edge_is_copied[keep]

    return data


def augment_with_edge_attr(
    data: Data,
    *,
    edge_attr_transform: str = "none",
    include_rel_sum: bool = True,
    include_edge_flags: bool = False,
) -> Data:
    """Append relation-wise mean edge weight features to each node.

    Notes:
      - This is a cheap way to expose edge weight information to an RGCN.
      - For v3 datasets, edge_attr is usually already log-scaled. Default transform='none'.
      - If include_edge_flags=True, also aggregates edge_is_false/edge_is_copied per-relation.
    """
    if not hasattr(data, "edge_attr") or data.edge_attr is None:
        print("[augment] edge_attr missing -> skip")
        return data

    edge_attr = data.edge_attr
    if edge_attr.dim() == 1:
        edge_attr = edge_attr.view(-1, 1)

    edge_index = data.edge_index
    edge_type = data.edge_type
    num_nodes = int(data.x.size(0))
    num_rel = int(edge_type.max().item()) + 1

    w = apply_edge_attr_transform(edge_attr, edge_attr_transform).to(torch.float32)  # [E]
    w = w.clamp(min=0.0)

    one_hot_rel = F.one_hot(edge_type, num_classes=num_rel).float()  # [E, R]
    weighted_rel = one_hot_rel * w.unsqueeze(-1)  # [E, R]

    agg = torch.zeros((num_nodes, num_rel), dtype=torch.float32)
    src, dst = edge_index
    agg.index_add_(0, src, weighted_rel)
    agg.index_add_(0, dst, weighted_rel)

    # mean by degree
    deg = torch.zeros((num_nodes, 1), dtype=torch.float32)
    ones = torch.ones((w.numel(), 1), dtype=torch.float32)
    deg.index_add_(0, src, ones)
    deg.index_add_(0, dst, ones)
    deg = deg.clamp(min=1.0)
    agg_mean = agg / deg

    feats = [agg_mean]

    if include_edge_flags:
        # Aggregate edge flags per relation (counts and mean ratios).
        # If flags are missing, treat as zeros.
        ef = getattr(data, 'edge_is_false', None)
        ec = getattr(data, 'edge_is_copied', None)
        if ef is None:
            ef = torch.zeros((edge_type.size(0),), dtype=torch.float32)
        else:
            ef = ef.view(-1).to(torch.float32)
        if ec is None:
            ec = torch.zeros((edge_type.size(0),), dtype=torch.float32)
        else:
            ec = ec.view(-1).to(torch.float32)

        rel_false = one_hot_rel * ef.unsqueeze(-1)
        rel_copied = one_hot_rel * ec.unsqueeze(-1)

        agg_false = torch.zeros((num_nodes, num_rel), dtype=torch.float32)
        agg_copied = torch.zeros((num_nodes, num_rel), dtype=torch.float32)
        agg_false.index_add_(0, src, rel_false)
        agg_false.index_add_(0, dst, rel_false)
        agg_copied.index_add_(0, src, rel_copied)
        agg_copied.index_add_(0, dst, rel_copied)

        feats.append(agg_false / deg)
        feats.append(agg_copied / deg)
        if include_rel_sum:
            feats.append(agg_false)
            feats.append(agg_copied)
    if include_rel_sum:
        feats.append(agg)  # also expose sums

    old_dim = int(data.x.size(1))
    data.x = torch.cat([data.x, *feats], dim=1)
    print(f"[*] augment_with_edge_attr: x_dim {old_dim} -> {int(data.x.size(1))}")
    return data


# -----------------------------
# Model
# -----------------------------


class HvtRGCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_relations: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        self.dropout = float(dropout)
        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations=num_relations))
        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations=num_relations))
        self.out = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_type):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index, edge_type)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.out(h).view(-1)


# -----------------------------
# Train / eval
# -----------------------------


@torch.no_grad()
def evaluate(model: nn.Module, data: Data, device: torch.device, threshold: float = 0.5) -> Dict[str, Dict[str, float]]:
    model.eval()

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_type = data.edge_type.to(device)
    y = data.y_hvt.to(device).long()

    logits = model(x, edge_index, edge_type)
    probs = torch.sigmoid(logits)

    out: Dict[str, Dict[str, float]] = {}
    for split, mask in {
        "train": data.train_mask,
        "val": data.val_mask,
        "test": data.test_mask,
    }.items():
        mask = mask.to(device)
        y_true = y[mask]
        y_prob = probs[mask]
        if y_true.numel() == 0:
            continue

        y_pred = (y_prob >= threshold).long()
        yt = y_true.detach().cpu().numpy()
        yp = y_pred.detach().cpu().numpy()
        ypr = y_prob.detach().cpu().numpy()

        acc = accuracy_score(yt, yp)
        f1 = f1_score(yt, yp, zero_division=0)
        try:
            auc = roc_auc_score(yt, ypr)
        except ValueError:
            auc = float("nan")

        out[split] = {"acc": float(acc), "f1": float(f1), "auc": float(auc)}

    return out


def train_one_epoch(model: nn.Module, data: Data, optimizer, criterion, device: torch.device) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_type = data.edge_type.to(device)
    y = data.y_hvt.to(device).float()
    mask = data.train_mask.to(device)

    logits = model(x, edge_index, edge_type)
    loss = criterion(logits[mask], y[mask])
    loss.backward()
    optimizer.step()
    return float(loss.item())


@dataclass
class EarlyStop:
    best: float = -1.0
    patience: int = 30
    min_delta: float = 1e-4
    bad_epochs: int = 0

    def step(self, value: float) -> bool:
        """Return True if should stop."""
        if value > self.best + self.min_delta:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def main() -> None:
    p = argparse.ArgumentParser(description="Train HVT node classifier (v3).")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--pos_weight", type=float, default=None)
    p.add_argument("--pos_weight_cap", type=float, default=20.0)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--min_delta", type=float, default=1e-4)
    p.add_argument("--edge_attr_agg", action="store_true", help="Augment x with relation-wise mean/sum edge_attr features.")
    p.add_argument("--include_edge_flags", action="store_true", help="When using --edge_attr_agg, also aggregate edge_is_false/edge_is_copied as node features (no filtering).")
    p.add_argument("--edge_attr_transform", type=str, default="none", choices=["none", "log1p", "auto"],
                   help="Transform used inside edge_attr aggregation (default: none for v3 datasets).")
    p.add_argument("--drop_false_edges", action="store_true", help="Drop edges with edge_is_false==1 if present.")
    p.add_argument("--drop_copied_edges", action="store_true", help="Drop edges with edge_is_copied==1 if present.")
    p.add_argument("--out_dir", type=str, default=None)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] device={device}")

    data = load_pyg_data(args.data_path, map_location="cpu")
    data = filter_edges(data, drop_false=args.drop_false_edges, drop_copied=args.drop_copied_edges)

    if args.include_edge_flags and (not hasattr(data, "edge_is_false")):
        print("[!] include_edge_flags enabled, but dataset has no edge_is_false/edge_is_copied attributes. Flags will be zeros.")

    if args.edge_attr_agg:
        data = augment_with_edge_attr(data, edge_attr_transform=args.edge_attr_transform, include_rel_sum=True, include_edge_flags=bool(args.include_edge_flags))

    num_rel = int(data.edge_type.max().item()) + 1
    model = HvtRGCN(
        in_channels=int(data.x.size(1)),
        hidden_channels=int(args.hidden_dim),
        num_relations=num_rel,
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
    ).to(device)

    # pos_weight
    y_train = data.y_hvt[data.train_mask].numpy().astype(np.int32)
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    pos = max(pos, 1)
    auto_pos_w = neg / pos
    pos_w = float(args.pos_weight) if args.pos_weight is not None else float(auto_pos_w)
    if args.pos_weight_cap and args.pos_weight_cap > 0:
        pos_w = float(min(pos_w, float(args.pos_weight_cap)))
    print(f"[*] pos_weight={pos_w:.3f} (auto={auto_pos_w:.3f})")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))
    optim = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    es = EarlyStop(best=-1.0, patience=int(args.patience), min_delta=float(args.min_delta))
    best_state = None
    best_metrics = None

    for epoch in range(1, int(args.epochs) + 1):
        loss = train_one_epoch(model, data, optim, criterion, device)
        metrics = evaluate(model, data, device)
        val_auc = metrics.get("val", {}).get("auc", float("nan"))

        if not np.isnan(val_auc) and val_auc >= es.best + es.min_delta:
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_metrics = metrics

        if epoch == 1 or epoch % 10 == 0 or epoch == int(args.epochs):
            tr = metrics.get("train", {})
            va = metrics.get("val", {})
            te = metrics.get("test", {})
            print(
                f"[Epoch {epoch:03d}] loss={loss:.4f} | "
                f"val_auc={va.get('auc', float('nan')):.3f} val_f1={va.get('f1', float('nan')):.3f} | "
                f"test_auc={te.get('auc', float('nan')):.3f}"
            )

        if not np.isnan(val_auc) and es.step(float(val_auc)):
            print(f"[*] Early stop at epoch {epoch} (best val_auc={es.best:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_metrics = evaluate(model, data, device)

    print("\n[*] Final metrics:")
    print(json.dumps(final_metrics, indent=2))

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.data_path))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "hvt_gnn_v3_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"best": best_metrics, "final": final_metrics, "args": vars(args)}, f, indent=2)
    print(f"[*] saved metrics: {out_path}")


if __name__ == "__main__":
    main()
