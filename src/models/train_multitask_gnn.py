"""
experiments/train_multitask_gnn.py

Multitask GNN:
  - Task 1: role classification (y_role, multi-class)
  - Task 2: HVT classification (y_hvt, binary)
  - Task 3: importance_score regression (data.importance_score)

Inputs:
  - x           : [N, F]
  - edge_index  : [2, E]
  - edge_type   : [E]
  - train/val/test_mask
"""

from __future__ import annotations

import argparse
import os
import random
import json 
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    r2_score,
)
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv


# -----------------------------
# Utility helpers
# -----------------------------

def augment_with_edge_attr(data: Data) -> Data:
    """
    Aggregate edge_attr (per-layer weights) to the node level and
    append [relation-wise aggregated weight] features to x.

    - data.edge_attr : [E, 1] or [E]
    - data.edge_type : [E]
    - data.edge_index: [2, E]
    """
    if not hasattr(data, "edge_attr") or data.edge_attr is None:
        print("[augment] data.edge_attr missing → skip")
        return data

    edge_attr = data.edge_attr
    if edge_attr.dim() == 1:
        edge_attr = edge_attr.unsqueeze(-1)
    edge_attr = edge_attr.float()          # [E, attr_dim] (assume attr_dim=1 for now)

    edge_index = data.edge_index
    edge_type = data.edge_type
    num_nodes = data.x.size(0)
    num_relations = int(edge_type.max().item()) + 1

    w = torch.log1p(edge_attr[:, 0].clamp(min=0))  # avoid negatives

    one_hot_rel = F.one_hot(edge_type, num_classes=num_relations).float()
    weighted_rel = one_hot_rel * w.unsqueeze(-1)  # [E, R]

    agg = torch.zeros(num_nodes, num_relations, dtype=torch.float32)
    src, dst = edge_index

    agg.index_add_(0, src, weighted_rel)
    agg.index_add_(0, dst, weighted_rel)

    # 2) Normalize by degree to take the mean
    deg = torch.zeros(num_nodes, 1, dtype=torch.float32)
    deg.index_add_(0, src, torch.ones_like(w).unsqueeze(-1))
    deg.index_add_(0, dst, torch.ones_like(w).unsqueeze(-1))
    deg = deg.clamp(min=1.0)

    agg = agg / deg  # degree normalization

    old_dim = data.x.size(1)
    data.x = torch.cat([data.x, agg], dim=1)

    print(
        f"[*] augment_with_edge_attr: x_dim {old_dim} → {data.x.size(1)} "
        f"(+{num_relations} from edge_attr aggregation)"
    )
    return data


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Multi-task R-GCN definition
# -----------------------------


class MultiTaskRGCN(nn.Module):
    """
    Shared R-GCN encoder + three heads:
      - role_head: role classification (num_roles-way softmax)
      - hvt_head : HVT binary classification (sigmoid)
      - imp_head : importance_score regression (real value)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_relations: int,
        num_roles: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        assert num_layers >= 2, "num_layers must be at least 2."

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout

        # First layer
        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Final layer
        self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        ### Multitask heads ###
        self.role_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_channels, num_roles),
        )
        self.hvt_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_channels, 1),
        )
        self.imp_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_channels, 1),
        )


    def encode(self, x, edge_index, edge_type):
        h = x
        prev = None
        for conv, bn in zip(self.convs, self.bns):
            out = conv(h, edge_index, edge_type)
            out = bn(out)
            out = F.relu(out)

            # Residual connection (only when shapes match)
            if prev is not None and prev.shape == out.shape:
                out = out + prev

            out = F.dropout(out, p=self.dropout, training=self.training)
            prev = out
            h = out
        return h


    def forward(self, x, edge_index, edge_type):
        h = self.encode(x, edge_index, edge_type)
        role_logits = self.role_head(h)      # [N, num_roles]
        hvt_logits = self.hvt_head(h).view(-1)  # [N]
        imp_pred = self.imp_head(h).view(-1)    # [N]
        return {
            "role_logits": role_logits,
            "hvt_logits": hvt_logits,
            "imp_pred": imp_pred,
        }


# -----------------------------
# Training / evaluation loop
# -----------------------------


def train_one_epoch(
    model: nn.Module,
    data: Data,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha_role: float,
    alpha_hvt: float,
    alpha_imp: float,
    pos_weight: float,
    imp_mean: float,
    imp_std: float,
):
    model.train()
    optimizer.zero_grad()

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_type = data.edge_type.to(device)
    y_role = data.y_role.to(device).long()
    y_hvt = data.y_hvt.to(device).float()
    imp = data.importance_score.to(device).float()   # <- important

    train_mask = data.train_mask.to(device)

    out = model(x, edge_index, edge_type)
    role_logits = out["role_logits"]
    hvt_logits = out["hvt_logits"]
    imp_pred_norm = out["imp_pred"]                 # <- interpreted as "normalized importance" prediction

    # 1) Role classification loss
    role_loss = F.cross_entropy(role_logits[train_mask], y_role[train_mask])

    # 2) HVT loss (BCE + pos_weight)
    pos_w = torch.tensor([pos_weight], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    hvt_loss = bce(hvt_logits[train_mask], y_hvt[train_mask])

    # 3) Importance regression loss (only targets are normalized)
    imp_norm = (imp - imp_mean) / (imp_std + 1e-8)
    reg_loss = F.mse_loss(imp_pred_norm[train_mask], imp_norm[train_mask])

    total_loss = alpha_role * role_loss + alpha_hvt * hvt_loss + alpha_imp * reg_loss
    total_loss.backward()
    optimizer.step()

    return float(total_loss.item()), float(role_loss.item()), float(hvt_loss.item()), float(
        reg_loss.item()
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: Data,
    device: torch.device,
    threshold: float,
    imp_mean: float,
    imp_std: float,
):
    model.eval()

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_type = data.edge_type.to(device)
    y_role = data.y_role.to(device).long()
    y_hvt = data.y_hvt.to(device).float()
    imp = data.importance_score.to(device).float()

    out = model(x, edge_index, edge_type)
    role_logits = out["role_logits"]
    hvt_logits = out["hvt_logits"]
    imp_pred_norm = out["imp_pred"]              # <- normalized prediction

    hvt_probs = torch.sigmoid(hvt_logits)

    metrics = {}

    for split_name, mask_name in [
        ("train", "train_mask"),
        ("val", "val_mask"),
        ("test", "test_mask"),
    ]:
        mask = getattr(data, mask_name).to(device)
        if mask.sum() == 0:
            continue

        # --- Role classification ---
        y_role_true = y_role[mask].cpu().numpy()
        role_log = role_logits[mask]
        role_pred = role_log.argmax(dim=-1).cpu().numpy()

        role_acc = accuracy_score(y_role_true, role_pred)
        role_f1_macro = f1_score(
            y_role_true, role_pred, average="macro", zero_division=0
        )

        # --- HVT classification ---
        y_hvt_true = y_hvt[mask].cpu().numpy()
        y_hvt_prob = hvt_probs[mask].cpu().numpy()
        y_hvt_pred = (y_hvt_prob > threshold).astype(float)

        hvt_acc = accuracy_score(y_hvt_true, y_hvt_pred)
        hvt_f1 = f1_score(y_hvt_true, y_hvt_pred, zero_division=0)
        try:
            hvt_auc = roc_auc_score(y_hvt_true, y_hvt_prob)
        except ValueError:
            hvt_auc = float("nan")

        # --- importance regression ---
        imp_true = imp[mask].cpu().numpy()

        # Denormalize predictions back to the original scale
        imp_pred_norm_split = imp_pred_norm[mask].cpu().numpy()
        imp_pred_denorm = imp_pred_norm_split * (imp_std + 1e-8) + imp_mean

        mse = mean_squared_error(imp_true, imp_pred_denorm)
        rmse = mse ** 0.5
        try:
            r2 = r2_score(imp_true, imp_pred_denorm)
        except ValueError:
            r2 = float("nan")

        metrics[split_name] = {
            "role_acc": role_acc,
            "role_f1_macro": role_f1_macro,
            "hvt_acc": hvt_acc,
            "hvt_f1": hvt_f1,
            "hvt_auc": hvt_auc,
            "imp_mse": mse,
            "imp_rmse": rmse,
            "imp_r2": r2,
        }

    return metrics


@torch.no_grad()
def collect_hvt_probs(
    model: nn.Module,
    data: Data,
    device: torch.device,
):
    """
    Collect (y_true, y_prob) pairs for HVT threshold sweeps.
    """
    model.eval()

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_type = data.edge_type.to(device)
    y_hvt = data.y_hvt.to(device).float()

    out = model(x, edge_index, edge_type)
    hvt_logits = out["hvt_logits"]
    probs = torch.sigmoid(hvt_logits)

    out_dict = {}
    for split_name, mask_name in [
        ("train", "train_mask"),
        ("val", "val_mask"),
        ("test", "test_mask"),
    ]:
        mask = getattr(data, mask_name).to(device)
        if mask.sum() == 0:
            continue

        y_true = y_hvt[mask].cpu().numpy()
        y_prob = probs[mask].cpu().numpy()

        out_dict[split_name] = (y_true, y_prob)

    return out_dict


def compute_hvt_metrics_at_threshold(y_true, y_prob, threshold: float):
    y_pred = (y_prob > threshold).astype(float)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return {"acc": acc, "f1": f1, "auc": auc}


def find_best_threshold_for_f1(y_true, y_prob, num_steps: int = 100):
    """
    Search for the F1-maximizing threshold over the [0.01, 0.99] range.
    """
    best_thr = 0.5
    best_f1 = -1.0
    best_metrics = None

    thresholds = np.linspace(0.01, 0.99, num_steps)
    for thr in thresholds:
        m = compute_hvt_metrics_at_threshold(y_true, y_prob, thr)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thr = float(thr)
            best_metrics = m

    return best_thr, best_metrics

def plot_training_curves(history, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    epochs = history["epoch"]

    # 1) Loss curves
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["total_loss"], label="total_loss")
    plt.plot(epochs, history["role_loss"], label="role_loss")
    plt.plot(epochs, history["hvt_loss"], label="hvt_loss")
    plt.plot(epochs, history["imp_loss"], label="imp_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Multitask Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "multitask_loss.png"))
    plt.close()

    # 2) Validation metrics (role / HVT)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["val_role_acc"], label="val_role_acc")
    plt.plot(epochs, history["val_role_f1_macro"], label="val_role_f1_macro")
    plt.plot(epochs, history["val_hvt_acc"], label="val_hvt_acc")
    plt.plot(epochs, history["val_hvt_f1"], label="val_hvt_f1")
    plt.plot(epochs, history["val_hvt_auc"], label="val_hvt_auc")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Validation Metrics (Role / HVT)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "multitask_val_role_hvt.png"))
    plt.close()

    # 3) Validation importance RMSE
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_imp_rmse"], label="val_imp_rmse")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Validation importance RMSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "multitask_val_imp_rmse.png"))
    plt.close()

# -----------------------------
# Main
# -----------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Multi-task R-GCN: role classification + HVT classification + importance regression."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to pyg_data.pt (output of build_pyg_dataset.py)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Hidden dimension size (default: 64)",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of RGCN layers (default: 2)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout probability (default: 0.5)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 regularization, default: 1e-4)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,  
        help="Number of training epochs (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=5.0,
        help="Positive class weight for HVT BCE loss (default: 5.0)",
    )
    parser.add_argument(
        "--alpha_role",
        type=float,
        default=1.0,
        help="Loss weight for role classification (default: 1.0)",
    )
    parser.add_argument(
        "--alpha_hvt",
        type=float,
        default=1.0,
        help="Loss weight for HVT classification (default: 1.0)",
    )
    parser.add_argument(
        "--alpha_imp",
        type=float,
        default=0.5,
        help="Loss weight for importance regression (default: 0.5)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience based on val HVT AUC (default: 50 epochs)",
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=1e-3,
        help="Minimum improvement in val HVT AUC to reset patience (default: 1e-3)",
    )

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"[*] Using device: {device}")

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data not found at: {args.data_path}")

    print("[*] Loading PyG Data from:", args.data_path)

    from torch_geometric.data import Data as PyGData

    torch.serialization.add_safe_globals([PyGData])
    data: Data = torch.load(args.data_path, weights_only=False)

    # Node feature augmentation using edge_attr
    data = augment_with_edge_attr(data)

    print(data)

    in_channels = data.x.size(1)
    num_relations = int(data.edge_type.max().item()) + 1

    # Determine number of roles
    num_roles = int(data.y_role.max().item()) + 1

    model = MultiTaskRGCN(
        in_channels=in_channels,
        hidden_channels=args.hidden_dim,
        num_relations=num_relations,
        num_roles=num_roles,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    # importance_score statistics (train split) - prefer build_pyg_dataset metadata
    if hasattr(data, "imp_mean") and hasattr(data, "imp_std"):
        imp_mean = float(data.imp_mean)
        imp_std = float(data.imp_std)
        print(f"[*] importance_score (from Data meta) mean={imp_mean:.3f}, std={imp_std:.3f}")
    else:
        imp = data.importance_score.float()
        train_mask = data.train_mask
        imp_mean = imp[train_mask].mean().item()
        imp_std = imp[train_mask].std().item()
        print(f"[*] importance_score (computed in train_multitask_gnn) mean={imp_mean:.3f}, std={imp_std:.3f}")


    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # === Record training history ===
    history = {
        "epoch": [],
        "total_loss": [],
        "role_loss": [],
        "hvt_loss": [],
        "imp_loss": [],
        "val_role_acc": [],
        "val_role_f1_macro": [],
        "val_hvt_acc": [],
        "val_hvt_f1": [],
        "val_hvt_auc": [],
        "val_imp_rmse": [],
    }

    best_val_hvt_auc = -1.0
    best_val_hvt_f1_at_05 = None
    best_test_metrics_at_05 = None
    epochs_no_improve = 0  # <-- counter for early stopping


    best_val_hvt_auc = -1.0
    best_val_hvt_f1_at_05 = None
    best_test_metrics_at_05 = None

    # -------------------------
    # Monitor training with default threshold=0.5
    # -------------------------
    for epoch in range(1, args.epochs + 1):
        total_loss, role_loss, hvt_loss, reg_loss = train_one_epoch(
            model=model,
            data=data,
            optimizer=optimizer,
            device=device,
            alpha_role=args.alpha_role,
            alpha_hvt=args.alpha_hvt,
            alpha_imp=args.alpha_imp,
            pos_weight=args.pos_weight,
            imp_mean=imp_mean,
            imp_std=imp_std,
        )

        metrics = evaluate(
            model=model,
            data=data,
            device=device,
            threshold=0.5,
            imp_mean=imp_mean,
            imp_std=imp_std,
        )

        train_m = metrics.get("train", {})
        val_m = metrics.get("val", {})
        test_m = metrics.get("test", {})

        val_hvt_auc = val_m.get("hvt_auc", float("nan"))

        # === History logging ===
        history["epoch"].append(epoch)
        history["total_loss"].append(total_loss)
        history["role_loss"].append(role_loss)
        history["hvt_loss"].append(hvt_loss)
        history["imp_loss"].append(reg_loss)
        history["val_role_acc"].append(val_m.get("role_acc", float("nan")))
        history["val_role_f1_macro"].append(val_m.get("role_f1_macro", float("nan")))
        history["val_hvt_acc"].append(val_m.get("hvt_acc", float("nan")))
        history["val_hvt_f1"].append(val_m.get("hvt_f1", float("nan")))
        history["val_hvt_auc"].append(val_m.get("hvt_auc", float("nan")))
        history["val_imp_rmse"].append(val_m.get("imp_rmse", float("nan")))

        # === Update best val HVT AUC (with early stopping) ===
        if not np.isnan(val_hvt_auc):
            # Check improvement margin against min_delta
            if val_hvt_auc > best_val_hvt_auc + args.min_delta:
                best_val_hvt_auc = val_hvt_auc
                best_val_hvt_f1_at_05 = val_m.get("hvt_f1", float("nan"))
                best_test_metrics_at_05 = test_m
                epochs_no_improve = 0  # reset counter on improvement
            else:
                epochs_no_improve += 1

        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"[Epoch {epoch:03d}] "
                f"loss={total_loss:.4f} "
                f"(role={role_loss:.4f}, hvt={hvt_loss:.4f}, imp={reg_loss:.4f}) | "
                f"val_role_acc={val_m.get('role_acc', float('nan')):.3f}, "
                f"val_hvt_acc={val_m.get('hvt_acc', float('nan')):.3f}, "
                f"val_hvt_auc={val_m.get('hvt_auc', float('nan')):.3f}, "
                f"val_imp_rmse={val_m.get('imp_rmse', float('nan')):.3f}"
            )

        # === Check early stopping condition ===
        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(
                f"\n[*] Early stopping triggered at epoch {epoch} "
                f"(no val HVT AUC improvement for {args.patience} epochs)."
            )
            break


    print("\n[*] Training finished.")
    print(f"[0.5 threshold] Best val HVT AUC: {best_val_hvt_auc:.3f}")
    if best_test_metrics_at_05 is not None:
        print("[0.5 threshold] Corresponding test metrics at best val HVT AUC:")
        print(
            "  [Role] "
            f"acc={best_test_metrics_at_05.get('role_acc', float('nan')):.3f}, "
            f"f1_macro={best_test_metrics_at_05.get('role_f1_macro', float('nan')):.3f}"
        )
        print(
            "  [HVT ] "
            f"acc={best_test_metrics_at_05.get('hvt_acc', float('nan')):.3f}, "
            f"f1={best_test_metrics_at_05.get('hvt_f1', float('nan')):.3f}, "
            f"auc={best_test_metrics_at_05.get('hvt_auc', float('nan')):.3f}"
        )
        print(
            "  [Imp ] "
            f"rmse={best_test_metrics_at_05.get('imp_rmse', float('nan')):.3f}, "
            f"r2={best_test_metrics_at_05.get('imp_r2', float('nan')):.3f}"
        )

    # -------------------------
    # Validation-set threshold sweep for HVT only (maximize F1)
    # -------------------------
    print("\n[*] Sweeping threshold on validation set to maximize HVT F1...")

    probs_dict = collect_hvt_probs(model, data, device)
    val_y, val_p = probs_dict["val"]

    best_thr, best_val_hvt_metrics = find_best_threshold_for_f1(val_y, val_p, num_steps=100)
    print(
        f"[*] Best threshold (val HVT F1): {best_thr:.3f} | "
        f"val_acc={best_val_hvt_metrics['acc']:.3f}, "
        f"val_f1={best_val_hvt_metrics['f1']:.3f}, "
        f"val_auc={best_val_hvt_metrics['auc']:.3f}"
    )

    # Recompute train/test HVT performance with that threshold
    train_y, train_p = probs_dict["train"]
    test_y, test_p = probs_dict["test"]

    train_hvt_best = compute_hvt_metrics_at_threshold(train_y, train_p, best_thr)
    test_hvt_best = compute_hvt_metrics_at_threshold(test_y, test_p, best_thr)

    print("\n[*] HVT metrics with best validation F1 threshold:")
    print(
        f"  [Train] acc={train_hvt_best['acc']:.3f}, "
        f"f1={train_hvt_best['f1']:.3f}, "
        f"auc={train_hvt_best['auc']:.3f}"
    )
    print(
        f"  [Test]  acc={test_hvt_best['acc']:.3f}, "
        f"f1={test_hvt_best['f1']:.3f}, "
        f"auc={test_hvt_best['auc']:.3f}"
    )

    # -------------------------
    # Plot training curves
    # -------------------------
    # Use data_path to define the default output directory
    default_plot_dir = os.path.join(
        os.path.dirname(args.data_path),
        "multitask_plots"
    )
    print(f"\n[*] Saving training curves to: {default_plot_dir}")
    plot_training_curves(history, default_plot_dir)

    out_dir = os.path.dirname(args.data_path)
    metrics_path = os.path.join(out_dir, "multitask_metrics.json")

    def _to_float_dict(d):
        return {k: (float(v) if v is not None else None) for k, v in d.items()}

    mt_result = {
        "task": "multitask",
        "data_path": args.data_path,
        "seed": args.seed,
        "generator_meta": getattr(data, "generator_meta", None),
        "generator_config": getattr(data, "generator_config", None),
        "generator_config_hash": getattr(data, "generator_config_hash", None),
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "pos_weight": args.pos_weight,
        "alpha_role": args.alpha_role,
        "alpha_hvt": args.alpha_hvt,
        "alpha_imp": args.alpha_imp,
        "best_val_hvt_auc": float(best_val_hvt_auc),
        # Test metrics for the multitask model at best val HVT AUC with threshold=0.5
        "fixed_threshold": {
            "threshold": 0.5,
            "val_hvt_auc": float(best_val_hvt_auc),
            "val_hvt_f1": float(best_val_hvt_f1_at_05)
            if best_val_hvt_f1_at_05 is not None
            else None,
            "test": _to_float_dict(best_test_metrics_at_05)
            if best_test_metrics_at_05 is not None
            else None,
        },
        # Threshold-swept results for HVT only
        "hvt_threshold_tuned": {
            "best_thr": float(best_thr),
            "val": _to_float_dict(best_val_hvt_metrics),
            "train": _to_float_dict(train_hvt_best),
            "test": _to_float_dict(test_hvt_best),
        },
    }

    with open(metrics_path, "w") as f:
        json.dump(mt_result, f, indent=2)

    print(f"[*] Saved multitask metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
