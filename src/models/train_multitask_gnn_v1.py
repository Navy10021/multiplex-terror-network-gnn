"""
src/models/train_multitask_gnn_v2.py

Multitask GNN trainer (v2) with a more robust, proven encoder:

  - Default encoder: Graph Transformer (PyG TransformerConv) with edge features:
      edge_feat = [log1p(edge_attr), rel_type_embedding(edge_type)]
  - Optional encoder: R-GCN (fallback / sanity baseline)

Targets (same as v1):
  1) Role classification (multi-class)
  2) HVT classification (binary, highly imbalanced)
  3) Importance regression

Key improvements vs the initial v2 draft:
  - Uses a stable Transformer-based encoder instead of a custom relational attention layer.
  - Default multi-task weighting = fixed (uncertainty weighting kept as an option, but not default).
  - Optional focal loss for HVT (often helps when pos_weight is large).
  - Edge attribute normalization (log1p + global z-score).
  - Early-stopping & LR scheduler monitor a composite score: val_hvt_auc + k * val_role_f1_macro
  - Prints generator_config if present (helps catch easy/baseline config mismatches).

Outputs:
  - multitask_metrics.json (same keys as v1 so summary scripts remain compatible)
  - multitask_gnn_v2_best.pt
  - multitask_plots_v2/ (curves)

Example:
  python src/models/train_multitask_gnn_v2.py \
    --data_path data/multiplex_baseline/pyg_data.pt \
    --seed 2025 \
    --encoder transformer \
    --hidden_dim 128 --num_layers 3 --heads 4 \
    --mtl_weighting fixed --alpha_role 1.0 --alpha_hvt 1.0 --alpha_imp 0.5 \
    --use_focal_hvt --focal_gamma 2.0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv, TransformerConv

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, mean_squared_error, r2_score, roc_curve, brier_score_loss

import matplotlib.pyplot as plt


# -----------------------------
# Reproducibility
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pyg_data(path: str, map_location: str = "cpu") -> Data:
    """
    Compatibility loader for PyTorch 2.6+ where torch.load defaults to weights_only=True.

    We save PyG Data objects via torch.save(...). To load them reliably across torch versions,
    we explicitly set weights_only=False when the argument exists.

    Security note:
      - weights_only=False enables full pickle loading. Use only for files you trust.
        (In this project, pyg_data.pt is generated locally by our own scripts.)
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older torch without weights_only argument
        return torch.load(path, map_location=map_location)


# -----------------------------
# Losses
# -----------------------------

def sigmoid_focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[float] = None,
    pos_weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Numerically-stable focal loss for binary classification with logits.
    Optionally uses:
      - alpha: class balancing factor (scalar)
      - pos_weight: like BCEWithLogitsLoss(pos_weight=...), multiplies positive term.
    """
    targets = targets.float()
    # BCE terms (stable)
    ce_loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    focal = (1 - p_t).clamp(min=1e-6).pow(gamma)

    loss = focal * ce_loss
    if alpha is not None:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


# -----------------------------
# Model definitions
# -----------------------------

class MultiTaskTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_relations: int,
        num_roles: int,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3,
        rel_emb_dim: int = 8,
        use_edge_attr: bool = True,
        edge_attr_mean: float = 0.0,
        edge_attr_std: float = 1.0,
    ):
        super().__init__()
        assert hidden_channels % heads == 0, "hidden_channels must be divisible by heads"
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.num_relations = num_relations
        self.use_edge_attr = use_edge_attr

        # Edge-type embedding (relation ID)
        self.rel_emb = nn.Embedding(num_relations, rel_emb_dim)

        # Edge feature dimension: [log1p(edge_attr), rel_emb]
        self.edge_dim = (1 if use_edge_attr else 0) + rel_emb_dim

        self.in_proj = nn.Linear(in_channels, hidden_channels)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            conv = TransformerConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels // heads,
                heads=heads,
                concat=True,
                dropout=dropout,
                edge_dim=self.edge_dim,
                beta=True,
            )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.dropout = nn.Dropout(dropout)

        # Buffers for edge_attr normalization
        self.register_buffer("edge_attr_mean", torch.tensor(float(edge_attr_mean)))
        self.register_buffer("edge_attr_std", torch.tensor(float(edge_attr_std)))

        # Heads
        self.role_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_roles),
        )
        self.hvt_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )
        self.imp_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def _build_edge_feat(self, edge_type: torch.Tensor, edge_attr: Optional[torch.Tensor]) -> torch.Tensor:
        rel = self.rel_emb(edge_type)  # [E, rel_emb_dim]
        feats = [rel]

        if self.use_edge_attr:
            if edge_attr is None:
                ea = torch.zeros((edge_type.size(0), 1), device=edge_type.device, dtype=torch.float32)
            else:
                ea = edge_attr.float()
                if ea.dim() == 1:
                    ea = ea.view(-1, 1)
                # log1p + global z-score
                ea = torch.log1p(ea.clamp(min=0.0))
                ea = (ea - self.edge_attr_mean) / (self.edge_attr_std + 1e-8)
            feats = [ea] + feats

        return torch.cat(feats, dim=-1)

    def encode(self, x, edge_index, edge_type, edge_attr=None):
        h = self.in_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            edge_feat = self._build_edge_feat(edge_type, edge_attr)
            h_new = conv(h, edge_index, edge_attr=edge_feat)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)
            h = h + h_new  # residual
        return h

    def forward(self, x, edge_index, edge_type, edge_attr=None):
        h = self.encode(x, edge_index, edge_type, edge_attr=edge_attr)
        role_logits = self.role_head(h)
        hvt_logits = self.hvt_head(h).view(-1)
        imp_pred = self.imp_head(h).view(-1)
        return {"role_logits": role_logits, "hvt_logits": hvt_logits, "imp_pred": imp_pred}


class MultiTaskRGCN(nn.Module):
    """Fallback encoder (sanity baseline)."""
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_relations: int,
        num_roles: int,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations=num_relations))
        self.norms.append(nn.LayerNorm(hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations=num_relations))
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.dropout = nn.Dropout(dropout)

        self.role_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_roles),
        )
        self.hvt_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )
        self.imp_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def encode(self, x, edge_index, edge_type, edge_attr=None):
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, edge_type)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)
            h = h_new
        return h

    def forward(self, x, edge_index, edge_type, edge_attr=None):
        h = self.encode(x, edge_index, edge_type, edge_attr=edge_attr)
        return {
            "role_logits": self.role_head(h),
            "hvt_logits": self.hvt_head(h).view(-1),
            "imp_pred": self.imp_head(h).view(-1),
        }


class UncertaintyWeighting(nn.Module):
    """
    Kendall et al. uncertainty weighting:
      L = Σ exp(-s_i) * L_i + s_i
    """
    def __init__(self, init_log_vars=(0.0, 0.0, 0.0)):
        super().__init__()
        self.log_vars = nn.Parameter(torch.tensor(init_log_vars, dtype=torch.float32))

    def forward(self, role_loss, hvt_loss, imp_loss):
        s0, s1, s2 = self.log_vars
        loss = torch.exp(-s0) * role_loss + s0
        loss = loss + torch.exp(-s1) * hvt_loss + s1
        loss = loss + torch.exp(-s2) * imp_loss + s2
        return loss


# -----------------------------
# Metrics
# -----------------------------

@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    data: Data,
    device: torch.device,
    mask: torch.Tensor,
    hvt_threshold: float,
    imp_mean: float,
    imp_std: float,
    hvt_temperature: float | None = None,
) -> Dict[str, float]:
    model.eval()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_type = data.edge_type.to(device)
    edge_attr = data.edge_attr.to(device) if hasattr(data, "edge_attr") and data.edge_attr is not None else None

    y_role = data.y_role.to(device).long()
    y_hvt = data.y_hvt.to(device).float()
    imp = data.importance_score.to(device).float()

    out = model(x, edge_index, edge_type, edge_attr=edge_attr)
    role_logits = out["role_logits"]
    hvt_logits = out["hvt_logits"]
    imp_pred = out["imp_pred"]

    mask = mask.to(device)
    if mask.sum() == 0:
        return {}

    # Role
    y_role_true = y_role[mask].cpu().numpy()
    y_role_pred = role_logits[mask].argmax(dim=-1).cpu().numpy()
    role_acc = accuracy_score(y_role_true, y_role_pred)
    role_f1_macro = f1_score(y_role_true, y_role_pred, average="macro", zero_division=0)

    # HVT
    y_hvt_true = y_hvt[mask].cpu().numpy()
    if hvt_temperature is not None:
        y_hvt_prob = apply_temperature_to_logits(hvt_logits[mask], float(hvt_temperature)).cpu().numpy()
    else:
        y_hvt_prob = torch.sigmoid(hvt_logits[mask]).cpu().numpy()
    y_hvt_pred = (y_hvt_prob > float(hvt_threshold)).astype(np.int32)
    hvt_acc = accuracy_score(y_hvt_true, y_hvt_pred)
    hvt_f1 = f1_score(y_hvt_true, y_hvt_pred, zero_division=0)
    try:
        hvt_auc = roc_auc_score(y_hvt_true, y_hvt_prob)
        hvt_ap = average_precision_score(y_hvt_true, y_hvt_prob)
    except Exception:
        hvt_auc = float("nan")
        hvt_ap = float("nan")

    # Importance (normalized -> real scale)
    imp_true = imp[mask].cpu().numpy()
    imp_pred_real = (imp_pred[mask] * imp_std + imp_mean).detach().cpu().numpy()
    mse = mean_squared_error(imp_true, imp_pred_real)
    rmse = float(math.sqrt(mse))
    try:
        imp_r2 = float(r2_score(imp_true, imp_pred_real))
    except Exception:
        imp_r2 = float("nan")

    return {
        "role_acc": float(role_acc),
        "role_f1_macro": float(role_f1_macro),
        "hvt_acc": float(hvt_acc),
        "f1": float(hvt_f1),
        "auc": float(hvt_auc),
        "ap": float(hvt_ap),
        "imp_rmse": float(rmse),
        "imp_r2": float(imp_r2),
    }


def sweep_threshold_for_best_f1(y_true: np.ndarray, y_prob: np.ndarray, steps: int = 100) -> Tuple[float, Dict[str, float]]:
    best_thr = 0.5
    best = {"f1": -1.0, "auc": float("nan"), "acc": float("nan")}
    for thr in np.linspace(0.01, 0.99, steps):
        y_pred = (y_prob > thr).astype(np.int32)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = float("nan")
        if f1 > best["f1"]:
            best_thr = float(thr)
            best = {"acc": float(acc), "f1": float(f1), "auc": float(auc)}
    return best_thr, best



def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
    adaptive: bool = False,
    min_bin_size: int = 20,
) -> float:
    """Expected Calibration Error (ECE) for binary probabilities.

    Notes:
      - `adaptive=False`: equal-width bins in [0,1] (classic ECE).
      - `adaptive=True`: approximately equal-mass bins via quantiles, with a small guard
        (`min_bin_size`) to reduce high-variance bins when val set is small / imbalanced.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_prob = np.clip(y_prob, 1e-8, 1.0 - 1e-8)

    n = len(y_true)
    if n == 0:
        return 0.0

    if adaptive:
        # Choose bin edges by quantiles of probabilities (equal-mass bins).
        # Guard: if dataset is small, reduce #bins to keep a minimum bin size.
        max_bins = max(1, min(n_bins, int(max(1, n // max(1, min_bin_size)))))
        if max_bins <= 1:
            bins = np.array([0.0, 1.0], dtype=np.float64)
        else:
            qs = np.linspace(0.0, 1.0, max_bins + 1)
            bins = np.quantile(y_prob, qs)
            # Ensure strict monotonicity & include endpoints.
            bins[0] = 0.0
            bins[-1] = 1.0
            bins = np.unique(bins)
            if len(bins) < 2:
                bins = np.array([0.0, 1.0], dtype=np.float64)
    else:
        bins = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)

    # Assign bins
    bin_ids = np.digitize(y_prob, bins, right=False) - 1
    n_bins_eff = len(bins) - 1

    ece = 0.0
    for b in range(n_bins_eff):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)

def fit_temperature_scaling(
    val_logits: torch.Tensor,
    val_labels: torch.Tensor,
    device: torch.device,
    max_iter: int = 200,
    init_temperature: float = 1.0,
    reg_lambda: float = 0.0,
) -> float:
    """Fit a single temperature `T` on a validation set by minimizing NLL.

    Regularization:
      - If `reg_lambda>0`, adds `reg_lambda * (log(T))^2` to discourage extreme temperatures
        and reduce overfitting / instability (useful for small / imbalanced val sets).
    """
    val_logits = val_logits.detach().to(device)
    val_labels = val_labels.detach().float().to(device)

    log_t = torch.tensor([math.log(max(init_temperature, 1e-6))], device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    bce = torch.nn.BCEWithLogitsLoss(reduction="mean")

    def closure():
        optimizer.zero_grad()
        t = torch.exp(log_t).clamp(min=1e-6, max=100.0)
        loss = bce(val_logits / t, val_labels)
        if reg_lambda and reg_lambda > 0.0:
            loss = loss + float(reg_lambda) * (log_t ** 2).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    t_final = float(torch.exp(log_t).clamp(min=1e-6, max=100.0).item())
    return t_final

def apply_temperature_to_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    t = float(max(temperature, 1e-6))
    return torch.sigmoid(logits / t)




def crossfit_temperature_scaling(
    val_logits: torch.Tensor,
    val_labels: torch.Tensor,
    device: torch.device,
    n_folds: int = 5,
    seed: int = 0,
    max_iter: int = 200,
    init_temperature: float = 1.0,
    reg_lambda: float = 0.0,
) -> Tuple[float, torch.Tensor, Dict[str, Any]]:
    """Cross-fit temperature scaling to reduce calibration overfit on a single val split.

    Returns:
      T_final: median temperature across folds
      oof_probs: out-of-fold calibrated probabilities for val set
      info: dict with fold temperatures and summary stats
    """
    from sklearn.model_selection import StratifiedKFold

    val_logits = val_logits.detach().to(device)
    val_labels = val_labels.detach().long().to(device)

    n = int(val_labels.numel())
    if n_folds <= 1 or n < 10:
        # fallback: single fit, no OOF
        T = fit_temperature_scaling(
            val_logits=val_logits,
            val_labels=val_labels,
            device=device,
            max_iter=max_iter,
            init_temperature=init_temperature,
            reg_lambda=reg_lambda,
        )
        probs = torch.sigmoid(val_logits / T).detach()
        info = {"fold_temperatures": [T], "T_median": T, "T_iqr": (T, T)}
        return T, probs, info

    y_np = val_labels.detach().cpu().numpy()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    oof_probs = torch.empty((n,), device=device, dtype=torch.float32)
    fold_ts: List[float] = []

    for train_idx, hold_idx in skf.split(np.zeros(n), y_np):
        train_idx_t = torch.tensor(train_idx, device=device, dtype=torch.long)
        hold_idx_t = torch.tensor(hold_idx, device=device, dtype=torch.long)

        T_k = fit_temperature_scaling(
            val_logits=val_logits[train_idx_t],
            val_labels=val_labels[train_idx_t],
            device=device,
            max_iter=max_iter,
            init_temperature=init_temperature,
            reg_lambda=reg_lambda,
        )
        fold_ts.append(float(T_k))
        oof_probs[hold_idx_t] = torch.sigmoid(val_logits[hold_idx_t] / float(T_k))

    # Robust aggregate
    T_sorted = sorted(fold_ts)
    T_median = float(np.median(T_sorted))
    q25 = float(np.quantile(T_sorted, 0.25))
    q75 = float(np.quantile(T_sorted, 0.75))

    info = {"fold_temperatures": fold_ts, "T_median": T_median, "T_iqr": (q25, q75)}
    return T_median, oof_probs.detach(), info

def select_threshold_by_prevalence(y_prob: np.ndarray, prevalence: float) -> float:
    """Choose threshold so that roughly prevalence fraction are predicted positive (top-k)."""
    y_prob = np.asarray(y_prob).astype(np.float64)
    prevalence = float(np.clip(prevalence, 1e-6, 1 - 1e-6))
    n = len(y_prob)
    k = int(round(prevalence * n))
    k = max(1, min(n, k))
    # threshold is the k-th largest value
    thr = float(np.partition(y_prob, n - k)[n - k])
    return thr


def select_threshold_fixed_fpr(y_true: np.ndarray, y_prob: np.ndarray, target_fpr: float = 0.01) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(np.float64)
    target_fpr = float(np.clip(target_fpr, 0.0, 1.0))
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    # roc_curve returns thresholds in descending order (first is inf)
    idx = int(np.argmin(np.abs(fpr - target_fpr)))
    chosen = thr[idx]
    if not np.isfinite(chosen):
        # fall back to max finite threshold
        finite = thr[np.isfinite(thr)]
        chosen = float(np.max(finite)) if finite.size else 0.5
    return float(chosen)


def bootstrap_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    strategy: str,
    steps: int,
    prevalence: float | None,
    target_fpr: float,
    n_boot: int,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap threshold selection to stabilize the operating point.

    Returns: (median, p25, p75)
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    idx = np.arange(n)

    thrs = []
    for _ in range(int(n_boot)):
        sample = rng.choice(idx, size=n, replace=True)
        yt = y_true[sample]
        yp = y_prob[sample]

        if strategy == "f1":
            thr, _ = sweep_threshold_for_best_f1(yt, yp, steps=steps)
        elif strategy in ("prevalence", "topk"):
            thr = select_threshold_by_prevalence(yp, prevalence if prevalence is not None else float(np.mean(yt)))
        elif strategy == "fixed_fpr":
            thr = select_threshold_fixed_fpr(yt, yp, target_fpr=target_fpr)
        else:
            thr, _ = sweep_threshold_for_best_f1(yt, yp, steps=steps)

        thrs.append(float(thr))

    thrs = np.asarray(thrs, dtype=np.float64)
    return float(np.median(thrs)), float(np.percentile(thrs, 25)), float(np.percentile(thrs, 75))

# -----------------------------
# Plotting
# -----------------------------

def plot_training_curves(history: Dict[str, list], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Loss
    plt.figure()
    plt.plot(history["epoch"], history["loss_total"], label="total")
    plt.plot(history["epoch"], history["loss_role"], label="role")
    plt.plot(history["epoch"], history["loss_hvt"], label="hvt")
    plt.plot(history["epoch"], history["loss_imp"], label="imp")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss (v2)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curves.png"))
    plt.close()

    # Val metrics
    plt.figure()
    plt.plot(history["epoch"], history["val_hvt_auc"], label="val_hvt_auc")
    plt.plot(history["epoch"], history["val_hvt_f1_at_05"], label="val_hvt_f1@0.5")
    plt.plot(history["epoch"], history["val_role_f1"], label="val_role_f1_macro")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Validation Metrics (v2)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "val_metrics.png"))
    plt.close()


# -----------------------------
# Helpers
# -----------------------------

def compute_role_class_weights(y: torch.Tensor, mask: torch.Tensor, num_classes: int, clip_max: float = 5.0) -> torch.Tensor:
    """
    Inverse-frequency weights, normalized to mean=1, with clipping to avoid extreme instability.
    """
    y = y[mask].cpu().numpy()
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    w = inv / inv.mean()
    w = np.minimum(w, clip_max)
    return torch.tensor(w, dtype=torch.float32)


def compute_hvt_pos_weight(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    y = y[mask].cpu().numpy().astype(np.int32)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    pos = max(pos, 1)
    return torch.tensor([neg / pos], dtype=torch.float32)


def summarize_edges(data: Data, num_relations: int) -> Dict[str, Any]:
    edge_type = data.edge_type.cpu().numpy().astype(np.int64)
    counts = np.bincount(edge_type, minlength=num_relations).tolist()
    return {"num_edges": int(data.edge_index.size(1)), "edges_per_relation": counts}


def compute_edge_attr_stats(data: Data) -> Tuple[float, float]:
    if not hasattr(data, "edge_attr") or data.edge_attr is None:
        return 0.0, 1.0
    ea = data.edge_attr.view(-1).cpu().numpy().astype(np.float32)
    ea = np.log1p(np.clip(ea, 0.0, None))
    mean = float(ea.mean())
    std = float(ea.std() + 1e-8)
    return mean, std


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Train multitask GNN (v2 transformer encoder).")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=2025)

    # model
    parser.add_argument("--encoder", type=str, default="transformer", choices=["transformer", "rgcn"])
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--rel_emb_dim", type=int, default=8)
    parser.add_argument("--no_edge_attr", action="store_true", help="Disable edge_attr in encoder edge features (default: edge_attr enabled).")

    # edge filtering / balancing (helps with extreme relation imbalance)
    parser.add_argument("--drop_relations", type=str, default="", help="Comma-separated relation IDs to drop (applied to train & eval). Example: '4'")
    parser.add_argument("--edge_balance", action="store_true", help="Downsample edges per relation during TRAIN to reduce relation imbalance (eval uses full graph).")
    parser.add_argument("--edge_balance_mult", type=float, default=4.0, help="Target edges per relation = (2nd-largest relation edge count) * mult (default: 4.0).")
    parser.add_argument("--edge_balance_cap", type=int, default=50000, help="Upper cap for target edges per relation when edge balancing (default: 50000).")

    # optimization
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # multi-task weighting
    parser.add_argument("--mtl_weighting", type=str, default="fixed", choices=["fixed", "uncertainty"])
    parser.add_argument("--alpha_role", type=float, default=1.0)
    parser.add_argument("--alpha_hvt", type=float, default=1.0)
    parser.add_argument("--alpha_imp", type=float, default=0.5)
    parser.add_argument("--uncertainty_init", type=float, default=0.0)

    # losses
    parser.add_argument("--role_label_smoothing", type=float, default=0.0)
    parser.add_argument("--use_focal_hvt", action="store_true")

    parser.add_argument("--pos_weight", type=float, default=None, help="Override HVT pos_weight (if omitted: auto=neg/pos from train split).")
    parser.add_argument("--pos_weight_cap", type=float, default=20.0, help="Cap pos_weight after auto/override (default: 20.0; set <=0 to disable).")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha", type=float, default=None)

    # early stop / scheduler
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--early_stop_metric", type=str, default="score", choices=["score", "hvt_auc"], help="Early stopping / LR scheduler monitor: combined val_score or HVT AUC.")
    parser.add_argument("--score_role_weight", type=float, default=0.2, help="Composite score = val_hvt_auc + k*val_role_f1")
    parser.add_argument("--lr_patience", type=int, default=10)

    # outputs
    parser.add_argument("--save_best", action="store_true", help="Save best checkpoint")
    parser.add_argument("--tag", type=str, default="v2", help="Filename tag")

    # Threshold sweep controls (used for selecting the best HVT threshold by validation F1)
    parser.add_argument(
        "--thr_steps",
        type=int,
        default=200,
        help="Number of thresholds to sweep on the validation set to maximize HVT F1 (default: 200).",
    )


    # HVT calibration / thresholding (recommended for stable operating points)
    parser.add_argument("--hvt_calibrate", action="store_true",
                        help="Fit post-hoc probability calibration for the HVT head on the validation split (default: off).")
    parser.add_argument("--hvt_calib_method", type=str, default="temperature", choices=["temperature"],
                        help="Calibration method for HVT probabilities (default: temperature).")
    parser.add_argument("--hvt_ece_bins", type=int, default=15,
                        help="Number of bins for calibration error (ECE) computation (default: 15).")

    parser.add_argument("--hvt_ece_adaptive", action="store_true",
                        help="Use adaptive (equal-mass) binning for ECE to reduce high-variance estimates on small/imbalanced val sets.")
    parser.add_argument("--hvt_calib_cv_folds", type=int, default=5,
                        help="Number of folds for cross-fit temperature scaling on validation set (default: 5). Use 1 to disable cross-fitting.")
    parser.add_argument("--hvt_calib_reg_lambda", type=float, default=0.0,
                        help="Regularization strength for temperature scaling: adds reg_lambda*(logT)^2 to NLL (default: 0.0).")


    parser.add_argument("--thr_strategy", type=str, default="f1",
                        choices=["f1", "prevalence", "topk", "fixed_fpr"],
                        help="How to select the HVT decision threshold from validation predictions.")
    parser.add_argument("--thr_prevalence", type=float, default=None,
                        help="Target positive rate used by thr_strategy=prevalence/topk. If None, uses generator_config['hvt_ratio'] if available, else empirical validation prevalence.")
    parser.add_argument("--thr_fpr", type=float, default=0.01,
                        help="Target false-positive-rate used by thr_strategy=fixed_fpr (default: 0.01).")
    parser.add_argument("--thr_bootstrap", type=int, default=0,
                        help="Bootstrap resamples on validation set to stabilize the chosen threshold (0=off). Median threshold is used.")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    print(f"[*] Loading PyG Data from: {args.data_path}")
    data: Data = load_pyg_data(args.data_path, map_location="cpu")

    # Optionally drop selected relations (applies to both train & eval).
    # This is useful when one relation is extremely dense and tends to drown out others.
    if args.drop_relations:
        try:
            drop_rel_ids = [int(s) for s in args.drop_relations.split(",") if s.strip() != ""]
        except Exception as e:
            raise ValueError(f"--drop_relations must be comma-separated ints (got: {args.drop_relations})") from e

        if drop_rel_ids:
            et_cpu = data.edge_type.cpu()
            keep = torch.ones(et_cpu.size(0), dtype=torch.bool)
            for rid in drop_rel_ids:
                keep &= (et_cpu != rid)
            before = int(et_cpu.numel())
            after = int(keep.sum().item())

            data.edge_index = data.edge_index[:, keep]
            data.edge_type = data.edge_type[keep]
            if hasattr(data, "edge_attr") and data.edge_attr is not None:
                data.edge_attr = data.edge_attr[keep]

            print(f"[*] drop_relations: dropped={drop_rel_ids} | edges {before} -> {after}")

    # Print generator config if present (helps detect easy/baseline mismatches)
    gen_cfg = getattr(data, "generator_config", None)
    if isinstance(gen_cfg, dict) and gen_cfg:
        print("[*] generator_config:", {k: gen_cfg.get(k) for k in ["seed","hvt_ratio","finance_structure_strength","comm_structure_strength","comm_randomness"]})
    else:
        print("[*] generator_config: (not available)")

    num_roles = int(data.y_role.max().item() + 1)
    num_relations = int(data.edge_type.max().item() + 1)

    edge_summary = summarize_edges(data, num_relations)
    print(f"[*] edges: {edge_summary}")

    # importance stats (from Data meta if available)
    imp_mean = float(getattr(data, "imp_mean", float(data.importance_score.mean().item())))
    imp_std = float(getattr(data, "imp_std", float(data.importance_score.std().item() + 1e-8)))
    print(f"[*] importance_score (from Data meta) mean={imp_mean:.3f}, std={imp_std:.3f}")

    # edge_attr stats (for transformer edge features)
    ea_mean, ea_std = compute_edge_attr_stats(data)
    print(f"[*] edge_attr log1p stats mean={ea_mean:.3f}, std={ea_std:.3f}")

    # weights
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    role_w = compute_role_class_weights(data.y_role, train_mask, num_roles, clip_max=5.0)
    role_w = role_w.to(device)
    print("[*] role class weights (mean≈1, clipped):", role_w.detach().cpu().numpy().tolist())

    auto_pos_w = compute_hvt_pos_weight(data.y_hvt, train_mask).to(device)
    if args.pos_weight is not None:
        hvt_pos_w = torch.tensor(float(args.pos_weight), device=device)
    else:
        hvt_pos_w = auto_pos_w
    if args.pos_weight_cap is not None and float(args.pos_weight_cap) > 0:
        hvt_pos_w = torch.clamp(hvt_pos_w, max=float(args.pos_weight_cap))
    print(f"[*] HVT pos_weight: {float(hvt_pos_w.item()):.3f} (auto={float(auto_pos_w.item()):.3f})")

    # loss functions
    role_crit = nn.CrossEntropyLoss(weight=role_w, label_smoothing=float(args.role_label_smoothing))
    imp_crit = nn.SmoothL1Loss()

    # build model
    # By default, we USE edge_attr (normalized). Disable with --no_edge_attr.
    use_edge_attr = (not args.no_edge_attr)

    if args.encoder == "transformer":
        model = MultiTaskTransformer(
            in_channels=int(data.x.size(1)),
            hidden_channels=args.hidden_dim,
            num_relations=num_relations,
            num_roles=num_roles,
            num_layers=args.num_layers,
            heads=args.heads,
            dropout=args.dropout,
            rel_emb_dim=args.rel_emb_dim,
            use_edge_attr=use_edge_attr,
            edge_attr_mean=ea_mean,
            edge_attr_std=ea_std,
        )
    else:
        model = MultiTaskRGCN(
            in_channels=int(data.x.size(1)),
            hidden_channels=args.hidden_dim,
            num_relations=num_relations,
            num_roles=num_roles,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )

    model = model.to(device)

    uw = None
    if args.mtl_weighting == "uncertainty":
        uw = UncertaintyWeighting(init_log_vars=(args.uncertainty_init, args.uncertainty_init, args.uncertainty_init)).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + (list(uw.parameters()) if uw is not None else []),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # scheduler with backward compatibility
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=args.lr_patience, verbose=True
        )
    except TypeError:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=args.lr_patience
        )

    # training history
    history = {
        "epoch": [],
        "loss_total": [],
        "loss_role": [],
        "loss_hvt": [],
        "loss_imp": [],
        "val_hvt_auc": [],
        "val_hvt_ap": [],
        "val_hvt_f1_at_05": [],
        "val_role_f1": [],
        "val_score": [],
    }

    # tensors to device once
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_type = data.edge_type.to(device)
    edge_attr = data.edge_attr.to(device) if hasattr(data, "edge_attr") and data.edge_attr is not None else None

    # Precompute per-relation edge indices for optional relation-balanced TRAIN sampling.
    # (Evaluation always uses the full graph.)
    rel_edge_indices = None
    edge_balance_target = None
    if args.edge_balance:
        rel_edge_indices = []
        rel_counts = []
        for r in range(num_relations):
            idx_r = (edge_type == r).nonzero(as_tuple=False).view(-1)
            rel_edge_indices.append(idx_r)
            rel_counts.append(int(idx_r.numel()))
        if len(rel_counts) >= 2:
            second_largest = sorted(rel_counts)[-2]
        else:
            second_largest = rel_counts[0] if rel_counts else 0
        edge_balance_target = int(max(1, second_largest * float(args.edge_balance_mult)))
        if args.edge_balance_cap is not None and int(args.edge_balance_cap) > 0:
            edge_balance_target = min(edge_balance_target, int(args.edge_balance_cap))
        print(
            f"[*] edge_balance: enabled | counts={rel_counts} | "
            f"target_per_relation={edge_balance_target} (mult={args.edge_balance_mult}, cap={args.edge_balance_cap})"
        )


    y_role = data.y_role.to(device).long()
    y_hvt = data.y_hvt.to(device).float()
    imp = data.importance_score.to(device).float()

    best_score = -1e9
    best_val_auc = -1e9
    best_epoch = -1
    best_state = None
    # Track best checkpoint by *HVT AUC* (for v1-style reporting).
    best_auc_epoch = -1
    best_auc_state = None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        if uw is not None:
            uw.train()

        optimizer.zero_grad(set_to_none=True)

        ei, et, ea = edge_index, edge_type, edge_attr
        if rel_edge_indices is not None and edge_balance_target is not None and model.training:
            sampled = []
            for r, idx_r in enumerate(rel_edge_indices):
                if idx_r.numel() <= edge_balance_target:
                    sampled.append(idx_r)
                else:
                    perm = torch.randperm(idx_r.numel(), device=idx_r.device)[:edge_balance_target]
                    sampled.append(idx_r[perm])
            sel = torch.cat(sampled, dim=0)
            sel = sel[torch.randperm(sel.numel(), device=sel.device)]
            ei = edge_index[:, sel]
            et = edge_type[sel]
            ea = edge_attr[sel] if edge_attr is not None else None

        out = model(x, ei, et, edge_attr=ea)
        role_logits = out["role_logits"]
        hvt_logits = out["hvt_logits"]
        imp_pred = out["imp_pred"]

        # losses (train_mask)
        r_loss = role_crit(role_logits[train_mask], y_role[train_mask])

        if args.use_focal_hvt:
            h_loss = sigmoid_focal_loss_with_logits(
                hvt_logits[train_mask],
                y_hvt[train_mask],
                gamma=float(args.focal_gamma),
                alpha=(float(args.focal_alpha) if args.focal_alpha is not None else None),
                pos_weight=hvt_pos_w,
            )
        else:
            h_loss = F.binary_cross_entropy_with_logits(
                hvt_logits[train_mask], y_hvt[train_mask], pos_weight=hvt_pos_w
            )

        # importance regression on normalized target
        imp_norm = (imp - imp_mean) / (imp_std + 1e-8)
        i_loss = imp_crit(imp_pred[train_mask], imp_norm[train_mask])

        if uw is not None:
            total = uw(r_loss, h_loss, i_loss)
        else:
            total = args.alpha_role * r_loss + args.alpha_hvt * h_loss + args.alpha_imp * i_loss

        total.backward()
        if args.grad_clip and args.grad_clip > 0:
            nn.utils.clip_grad_norm_(list(model.parameters()) + (list(uw.parameters()) if uw is not None else []), args.grad_clip)
        optimizer.step()

        # validation
        val_m = evaluate_split(model, data, device, val_mask, 0.5, imp_mean, imp_std)
        val_auc = float(val_m.get("auc", float("nan")))
        val_ap  = float(val_m.get("ap",  float("nan")))
        val_f1_05 = float(val_m.get("f1", float("nan")))
        val_role_f1 = float(val_m.get("role_f1_macro", float("nan")))
        val_score = val_auc + float(args.score_role_weight) * val_role_f1

        monitor = val_score if args.early_stop_metric == "score" else val_auc
        scheduler.step(monitor)

        # logging
        history["epoch"].append(epoch)
        history["loss_total"].append(float(total.item()))
        history["loss_role"].append(float(r_loss.item()))
        history["loss_hvt"].append(float(h_loss.item()))
        history["loss_imp"].append(float(i_loss.item()))
        history["val_hvt_auc"].append(val_auc)
        history["val_hvt_ap"].append(val_ap)
        history["val_hvt_f1_at_05"].append(val_f1_05)
        history["val_role_f1"].append(val_role_f1)
        history["val_score"].append(val_score)

        if epoch == 1 or epoch % 10 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"[Epoch {epoch:03d}] lr={lr_now:.2e} loss={float(total.item()):.4f} "
                f"(role={float(r_loss.item()):.4f}, hvt={float(h_loss.item()):.4f}, imp={float(i_loss.item()):.4f}) | "
                f"val_score={val_score:.3f} val_hvt_auc={val_auc:.3f} val_hvt_ap={val_ap:.3f} val_role_f1={val_role_f1:.3f}"
            )

        
        # Update best-by-HVT-AUC checkpoint (independent of composite score).
        if not np.isnan(val_auc) and (val_auc > best_val_auc + 1e-8):
            best_val_auc = float(val_auc)
            best_auc_epoch = epoch
            best_auc_state = {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "uw": ({k: v.detach().cpu().clone() for k, v in uw.state_dict().items()} if uw is not None else None),
            }
# save best by composite score (but keep best_val_auc for reporting)
        improved = monitor > best_score + 1e-8
        if improved:
            best_score = float(monitor)
            best_epoch = epoch
            best_state = {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "uw": {k: v.detach().cpu().clone() for k, v in uw.state_dict().items()} if uw is not None else None,
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"\n[*] Early stopping at epoch {epoch} (no {args.early_stop_metric} improvement for {args.patience} epochs).")
            break

    print("\n[*] Training finished.")

    # Prefer best-by-HVT-AUC checkpoint for v1-style reporting.
    # If not available (e.g., degenerate AUC), fall back to best-by-monitor checkpoint.
    if best_auc_state is not None:
        model.load_state_dict(best_auc_state["model"])
        if uw is not None and best_auc_state["uw"] is not None:
            uw.load_state_dict(best_auc_state["uw"])
    elif best_state is not None:
        model.load_state_dict(best_state["model"])
        if uw is not None and best_state["uw"] is not None:
            uw.load_state_dict(best_state["uw"])

    # ------------------------------------------------------------
    # V1-style summary @ threshold=0.5
    # ------------------------------------------------------------
    val_fixed = evaluate_split(model, data, device, val_mask, 0.5, imp_mean, imp_std)
    test_fixed = evaluate_split(model, data, device, test_mask, 0.5, imp_mean, imp_std)

    print(f"[0.5 threshold] Best val HVT AUC: {val_fixed.get('auc', float('nan')):.3f}")
    print("[0.5 threshold] Corresponding test metrics at best val HVT AUC:")
    print(f"  [Role] acc={test_fixed.get('role_acc', float('nan')):.3f}, f1_macro={test_fixed.get('role_f1_macro', float('nan')):.3f}")
    print(f"  [HVT ] acc={test_fixed.get('hvt_acc', float('nan')):.3f}, f1={test_fixed.get('f1', float('nan')):.3f}, auc={test_fixed.get('auc', float('nan')):.3f}")
    print(f"  [Imp ] rmse={test_fixed.get('imp_rmse', float('nan')):.3f}, r2={test_fixed.get('imp_r2', float('nan')):.3f}")

    # ------------------------------------------------------------
    # Threshold sweep on validation set to maximize HVT F1
    # ------------------------------------------------------------
    print("\n[*] Sweeping threshold on validation set to maximize HVT F1...")

    model.eval()
    if uw is not None:
        uw.eval()
    with torch.no_grad():
        ei, et, ea = edge_index, edge_type, edge_attr
        out = model(x, ei, et, edge_attr=ea)
        logits_hvt = out["hvt_logits"].view(-1)
        hvt_prob_uncal = torch.sigmoid(logits_hvt).detach().cpu().numpy()

    y_val_true = y_hvt[val_mask].detach().cpu().numpy().astype(int)
    # Validation arrays (uncalibrated vs calibrated)
    val_logits = logits_hvt[val_mask].detach()
    y_val_prob_uncal = hvt_prob_uncal[val_mask.cpu().numpy()]

    temperature = 1.0
    y_val_prob_oof = None

    if args.hvt_calibrate:
        # Cross-fit temperature scaling on validation to reduce overfitting of calibration.
        if args.hvt_calib_cv_folds and args.hvt_calib_cv_folds > 1:
            temperature, oof_probs_t, calib_info = crossfit_temperature_scaling(
                val_logits=val_logits,
                val_labels=torch.as_tensor(y_val_true, device=val_logits.device),
                device=val_logits.device,
                n_folds=args.hvt_calib_cv_folds,
                seed=args.seed,
                max_iter=200,
                init_temperature=1.0,
                reg_lambda=args.hvt_calib_reg_lambda,
            )
            y_val_prob = apply_temperature_to_logits(val_logits, temperature).detach().cpu().numpy()
            y_val_prob_oof = oof_probs_t.detach().cpu().numpy()
        else:
            temperature = fit_temperature_scaling(
                val_logits=val_logits,
                val_labels=torch.as_tensor(y_val_true, device=val_logits.device),
                device=val_logits.device,
                max_iter=200,
                init_temperature=1.0,
                reg_lambda=args.hvt_calib_reg_lambda,
            )
            y_val_prob = apply_temperature_to_logits(val_logits, temperature).detach().cpu().numpy()
    else:
        y_val_prob = y_val_prob_uncal

    # Calibration diagnostics (validation)
    try:
        ece_before = compute_ece(y_val_true, y_val_prob_uncal, n_bins=args.hvt_ece_bins, adaptive=args.hvt_ece_adaptive)
        brier_before = float(brier_score_loss(y_val_true, y_val_prob_uncal))

        _prob_for_calib_metrics = y_val_prob_oof if (y_val_prob_oof is not None) else y_val_prob
        ece_after = compute_ece(y_val_true, _prob_for_calib_metrics, n_bins=args.hvt_ece_bins, adaptive=args.hvt_ece_adaptive)
        brier_after = float(brier_score_loss(y_val_true, _prob_for_calib_metrics))
    except Exception:
        ece_before = float('nan')
        brier_before = float('nan')
        ece_after = float('nan')
        brier_after = float('nan')
    thr_steps = int(getattr(args, "thr_steps", 200))

    # Threshold selection: prefer stable, policy-driven operating points for imbalanced HVT
    prevalence = args.thr_prevalence
    if prevalence is None:
        generator_cfg = getattr(data, "generator_config", {})
        if isinstance(generator_cfg, dict) and ("hvt_ratio" in generator_cfg):
            prevalence = float(generator_cfg["hvt_ratio"])
        else:
            prevalence = float(np.mean(y_val_true))

    if args.thr_strategy == "f1":
        best_thr, best_val_hvt = sweep_threshold_for_best_f1(y_val_true, y_val_prob, steps=thr_steps)
    elif args.thr_strategy in ("prevalence", "topk"):
        best_thr = select_threshold_by_prevalence(y_val_prob, prevalence)
        yhat = (y_val_prob >= best_thr).astype(int)
        best_val_hvt = {
            "acc": float(accuracy_score(y_val_true, yhat)),
            "f1": float(f1_score(y_val_true, yhat, zero_division=0)),
            "auc": float(roc_auc_score(y_val_true, y_val_prob)),
            "ap": float(average_precision_score(y_val_true, y_val_prob)),
        }
    elif args.thr_strategy == "fixed_fpr":
        best_thr = select_threshold_fixed_fpr(y_val_true, y_val_prob, target_fpr=args.thr_fpr)
        yhat = (y_val_prob >= best_thr).astype(int)
        best_val_hvt = {
            "acc": float(accuracy_score(y_val_true, yhat)),
            "f1": float(f1_score(y_val_true, yhat, zero_division=0)),
            "auc": float(roc_auc_score(y_val_true, y_val_prob)),
            "ap": float(average_precision_score(y_val_true, y_val_prob)),
        }
    else:
        best_thr, best_val_hvt = sweep_threshold_for_best_f1(y_val_true, y_val_prob, steps=thr_steps)

    # Optional bootstrap to stabilize the chosen operating threshold
    thr_p25 = None
    thr_p75 = None
    if getattr(args, "thr_bootstrap", 0) and int(args.thr_bootstrap) > 0:
        thr_med, thr_p25, thr_p75 = bootstrap_threshold(
            y_val_true,
            y_val_prob,
            strategy=args.thr_strategy,
            steps=thr_steps,
            prevalence=prevalence,
            target_fpr=args.thr_fpr,
            n_boot=int(args.thr_bootstrap),
            seed=int(getattr(args, "seed", 0)),
        )
        best_thr = thr_med
        yhat = (y_val_prob >= best_thr).astype(int)
        best_val_hvt = {
            "acc": float(accuracy_score(y_val_true, yhat)),
            "f1": float(f1_score(y_val_true, yhat, zero_division=0)),
            "auc": float(roc_auc_score(y_val_true, y_val_prob)),
            "ap": float(average_precision_score(y_val_true, y_val_prob)),
        }
        print(f"[*] Threshold bootstrap: median={best_thr:.3f}, IQR=[{float(thr_p25):.3f}, {float(thr_p75):.3f}] (n={int(args.thr_bootstrap)})")

    calib_msg = ""
    if args.hvt_calibrate:
        calib_msg = (
            f", calib=temperature(T={temperature:.3f}), "
            f"ECE {ece_before:.3f}->{ece_after:.3f}, "
            f"Brier {brier_before:.3f}->{brier_after:.3f}"
        )

    print(
        f"[*] Selected HVT threshold: {best_thr:.3f} "
        f"(strategy={args.thr_strategy}, prevalence={prevalence:.3f}{calib_msg}) | "
        f"val_acc={best_val_hvt['acc']:.3f}, val_f1={best_val_hvt['f1']:.3f}, "
        f"val_auc={best_val_hvt['auc']:.3f}, val_ap={best_val_hvt.get('ap', float('nan')):.3f}"
    )

    # Evaluate HVT metrics with best validation-F1 threshold
    hvt_temp = temperature if args.hvt_calibrate else None
    train_hvt = evaluate_split(model, data, device, train_mask, best_thr, imp_mean, imp_std, hvt_temp)
    test_hvt  = evaluate_split(model, data, device, test_mask,  best_thr, imp_mean, imp_std, hvt_temp)

    # Convert train/test HVT metrics to the same key schema as val (acc/f1/auc)
    train_at_best = {
        "acc": float(train_hvt.get("hvt_acc", train_hvt.get("acc", 0.0))),
        "f1": float(train_hvt.get("f1", 0.0)),
        "auc": float(train_hvt.get("auc", 0.0)),
    }
    test_at_best = {
        "acc": float(test_hvt.get("hvt_acc", test_hvt.get("acc", 0.0))),
        "f1": float(test_hvt.get("f1", 0.0)),
        "auc": float(test_hvt.get("auc", 0.0)),
    }

    print("\n[*] HVT metrics with best validation F1 threshold:")
    print(f"  [Train] acc={train_hvt.get('hvt_acc', float('nan')):.3f}, f1={train_hvt.get('f1', float('nan')):.3f}, auc={train_hvt.get('auc', float('nan')):.3f}")
    print(f"  [Test]  acc={test_hvt.get('hvt_acc', float('nan')):.3f}, f1={test_hvt.get('f1', float('nan')):.3f}, auc={test_hvt.get('auc', float('nan')):.3f}")

# output dirs
    out_dir = os.path.dirname(args.data_path)
    plot_dir = os.path.join(out_dir, "multitask_plots")
    print(f"[*] Saving training curves to: {plot_dir}")
    plot_training_curves(history, plot_dir)

    # save checkpoint
    ckpt_path = os.path.join(out_dir, f"multitask_gnn_{args.tag}_best.pt")
    if args.save_best:
        torch.save({"model": model.state_dict(), "uw": (uw.state_dict() if uw is not None else None), "args": vars(args)}, ckpt_path)
        print(f"[*] Saved best checkpoint to: {ckpt_path}")

    # save metrics (v1 compatible)
    metrics_path = os.path.join(out_dir, "multitask_metrics.json")

    def _to_float_dict(d: Dict[str, Any]):
        return {k: (float(v) if v is not None else None) for k, v in d.items()}

    mt_result = {
        "task": "multitask_v2",
        "data_path": args.data_path,
        "seed": args.seed,
        "encoder": args.encoder,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "heads": args.heads,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "mtl_weighting": args.mtl_weighting,
        "alpha_role": args.alpha_role,
        "alpha_hvt": args.alpha_hvt,
        "alpha_imp": args.alpha_imp,
        f"best_{args.early_stop_metric}": float(best_score),
        "fixed_threshold": {
            "threshold": 0.5,
            "val": _to_float_dict(val_fixed),
            "test": _to_float_dict(test_fixed),
        },
        "hvt_threshold_tuned": {
            "best_thr": float(best_thr),
            "val": _to_float_dict(best_val_hvt),
            "train": _to_float_dict(train_at_best),
            "test": _to_float_dict(test_at_best),
        },
    }

    # attach generator meta if available
    if hasattr(data, "generator_meta"):
        mt_result["generator_meta"] = getattr(data, "generator_meta")
    if hasattr(data, "generator_config"):
        mt_result["generator_config"] = getattr(data, "generator_config")

    # calibration & operating-point info (HVT)
    mt_result["hvt_calibration"] = {
        "enabled": bool(getattr(args, "hvt_calibrate", False)),
        "method": str(getattr(args, "hvt_calib_method", "temperature")),
        "temperature": float(temperature) if "temperature" in locals() else 1.0,
        "ece_val_before": float(ece_before) if "ece_before" in locals() else None,
        "ece_val_after": float(ece_after) if "ece_after" in locals() else None,
        "brier_val_before": float(brier_before) if "brier_before" in locals() else None,
        "brier_val_after": float(brier_after) if "brier_after" in locals() else None,
        "thr_strategy": str(getattr(args, "thr_strategy", "f1")),
        "thr_prevalence": float(prevalence) if "prevalence" in locals() else None,
        "thr_fpr": float(getattr(args, "thr_fpr", 0.01)),
        "thr_bootstrap": int(getattr(args, "thr_bootstrap", 0)),
        "thr_iqr": {"p25": float(thr_p25), "p75": float(thr_p75)} if ("thr_p25" in locals() and thr_p25 is not None) else None,
    }

    with open(metrics_path, "w") as f:
        json.dump(mt_result, f, indent=2)

    print(f"[*] Saved multitask metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
