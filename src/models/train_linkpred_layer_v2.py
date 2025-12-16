# src/models/train_linkpred_layer_v3.py

import argparse
from typing import Tuple, Dict, List
import json 
import os 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data

from sklearn.metrics import roc_auc_score, average_precision_score


# -----------------------------
# Safe PyG loader (PyTorch >=2.6 friendly)
# -----------------------------
def load_pyg_data(path: str, map_location: str | torch.device = "cpu") -> Data:
    try:
        import torch.serialization
        torch.serialization.add_safe_globals([Data])
    except Exception:
        pass
    return torch.load(path, map_location=map_location, weights_only=False)


# -----------------------------
# Edge-attr transform utilities
# -----------------------------
def resolve_edge_attr_transform_np(
    ea: np.ndarray,
    mode: str = "auto",
    q: float = 0.99,
    thresh: float = 20.0,
) -> str:
    mode = (mode or "auto").lower()
    if mode in ("none", "raw"):
        return "none"
    if mode in ("log1p", "log"):
        return "log1p"
    ea = np.asarray(ea, dtype=np.float32).reshape(-1)
    ea = np.clip(ea, 0.0, None)
    if ea.size == 0:
        return "none"
    try:
        qq = float(np.quantile(ea, q))
    except Exception:
        qq = float(ea.max())
    mx = float(ea.max())
    return "log1p" if (qq > float(thresh) or mx > float(thresh)) else "none"


def transform_edge_attr_torch(w: torch.Tensor, transform: str) -> torch.Tensor:
    w = w.clamp(min=0.0)
    if transform == "log1p":
        return torch.log1p(w)
    return w


def is_layer_directed(data: Data, layer_name: str, override: bool | None = None) -> bool:
    if override is not None:
        return bool(override)
    cfg = getattr(data, "generator_config", None)
    if isinstance(cfg, dict) and "directed_layers" in cfg:
        try:
            return layer_name in set(cfg["directed_layers"])
        except Exception:
            pass
    # fallback heuristics
    return layer_name in ("finance", "digital")


def canonicalize_undirected(edge_index: torch.Tensor) -> torch.Tensor:
    """Return unique undirected pairs u<v."""
    src, dst = edge_index
    lo = torch.minimum(src, dst)
    hi = torch.maximum(src, dst)
    pairs = torch.stack([lo, hi], dim=1)  # [E,2]
    uniq = torch.unique(pairs, dim=0)     # [Euniq,2]
    return uniq.t().contiguous()


class RGCNEncoder(nn.Module):
    """
    Simple multi-relation R-GCN encoder.
    - Input: x, edge_index, edge_type
    - Output: node embedding h
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_relations: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        assert num_layers >= 2, "num_layers must be >= 2."

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

    def forward(self, x, edge_index, edge_type):
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index, edge_type)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class DotProductLinkPredictor(nn.Module):
    """
    Predict link existence probability via dot product over (u, v) node embeddings h.
    """

    def forward(self, h, edge_index):
        src, dst = edge_index
        score = (h[src] * h[dst]).sum(dim=-1)  # [E]
        return score  


class MLPLinkPredictor(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h, edge_index):
        src, dst = edge_index
        z = torch.cat([h[src], h[dst]], dim=-1)  # [E, 2H]
        return self.mlp(z).view(-1)


def augment_with_edge_attr(data: Data, edge_attr_transform: str = "none") -> Data:
    """
    Aggregate edge_attr (per-layer weights) to the node level and
    append [relation-wise mean weight] features to x.

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
    edge_attr = edge_attr.float()  # [E, attr_dim] (assume attr_dim=1 for now)

    edge_index = data.edge_index
    edge_type = data.edge_type
    num_nodes = data.x.size(0)
    num_relations = int(edge_type.max().item()) + 1

    # 1) Optional transform + clamp
    w = transform_edge_attr_torch(edge_attr[:, 0], edge_attr_transform)  # [E]

    one_hot_rel = F.one_hot(edge_type, num_classes=num_relations).float()  # [E, R]
    weighted_rel = one_hot_rel * w.unsqueeze(-1)  # [E, R]

    agg = torch.zeros(num_nodes, num_relations, dtype=torch.float32)
    src, dst = edge_index

    # Add weights to both endpoints (approximate undirected)
    agg.index_add_(0, src, weighted_rel)
    agg.index_add_(0, dst, weighted_rel)

    # 2) Normalize by degree to obtain mean weights
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




def split_edges(
    edge_index: torch.Tensor,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split positive (existing) edges into train/val/test.
    """
    num_edges = edge_index.size(1)
    perm = torch.randperm(num_edges, generator=torch.Generator().manual_seed(seed))

    train_end = int(num_edges * train_ratio)
    val_end = int(num_edges * (train_ratio + val_ratio))

    train_edges = edge_index[:, perm[:train_end]]
    val_edges = edge_index[:, perm[train_end:val_end]]
    test_edges = edge_index[:, perm[val_end:]]
    return train_edges, val_edges, test_edges


def sample_negative_edges(
    num_nodes: int,
    num_samples: int,
    existing_edges: set,
    seed: int = 42,
) -> torch.Tensor:
    """
    Sample non-existent (u, v) pairs as negative edges.
    existing_edges is assumed to be a set of (min(u,v), max(u,v)) tuples.
    """
    rng = np.random.RandomState(seed)
    neg_edges = []
    while len(neg_edges) < num_samples:
        u = rng.randint(0, num_nodes)
        v = rng.randint(0, num_nodes)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in existing_edges:
            continue
        neg_edges.append((u, v))
    neg_edges = torch.tensor(neg_edges, dtype=torch.long).t().contiguous()  # [2, num_samples]
    return neg_edges


def sample_negative_edges_hard_region(
    data,
    num_samples: int,
    existing_edges: set,
    seed: int = 42,
) -> torch.Tensor:
    """
    Hard-negative mode sampling negative edges only within the same region.
    - Assumes the first part of x holds region one-hot features of length len(data.region_mapping).
    """
    num_nodes = data.x.size(0)
    num_regions = len(data.region_mapping)
    x = data.x

    # region one-hot → region index [N]
    region_oh = x[:, :num_regions]                   # [N, R]
    region_idx = region_oh.argmax(dim=1).cpu().numpy()

    # Nodes grouped by region
    region_to_nodes: Dict[int, List[int]] = {}
    for i, r in enumerate(region_idx):
        region_to_nodes.setdefault(int(r), []).append(i)

    rng = np.random.default_rng(seed)
    neg_edges = []

    while len(neg_edges) < num_samples:
        # Select a region
        r = int(rng.integers(len(region_to_nodes)))
        nodes_in_r = region_to_nodes.get(r, [])
        if len(nodes_in_r) < 2:
            continue

        u, v = rng.choice(nodes_in_r, size=2, replace=False)
        u, v = int(u), int(v)
        if u == v:
            continue

        a, b = (u, v) if u < v else (v, u)
        if (a, b) in existing_edges:
            continue

        neg_edges.append((u, v))

    neg_edges = torch.tensor(neg_edges, dtype=torch.long).t().contiguous()
    return neg_edges


def build_edge_sets_for_layer(
    data,
    layer_name: str,
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42,
    neg_mode="uniform",
):
    edge_index = data.edge_index
    edge_type = data.edge_type
    layer_type_mapping = data.layer_type_mapping

    if isinstance(layer_type_mapping, dict):
        target_rel = layer_type_mapping[layer_name]
    else:
        target_rel = int(layer_type_mapping[layer_name])

    mask = edge_type == target_rel
    pos_edge_index = edge_index[:, mask]
    num_pos = pos_edge_index.size(1)
    print(f"[Layer={layer_name}] #positive edges = {num_pos}")

    directed = is_layer_directed(data, layer_name, override=None)

    if not directed:
        pos_edge_index = canonicalize_undirected(pos_edge_index)

    train_pos, val_pos, test_pos = split_edges(
        pos_edge_index, train_ratio, val_ratio, seed
    )

    # --------------------
    # build existing edge set
    # --------------------
    existing = set()

    if directed:
        for u, v in zip(pos_edge_index[0].tolist(), pos_edge_index[1].tolist()):
            if u != v:
                existing.add((int(u), int(v)))
    else:
        for u, v in zip(pos_edge_index[0].tolist(), pos_edge_index[1].tolist()):
            if u != v:
                a, b = (u, v) if u < v else (v, u)
                existing.add((int(a), int(b)))

    num_nodes = data.x.size(0)

    # --------------------
    # negative sampling
    # --------------------
    if neg_mode == "uniform":
        train_neg = sample_negative_edges(
            num_nodes, train_pos.size(1), existing, seed=seed + 1
        )
        val_neg = sample_negative_edges(
            num_nodes, val_pos.size(1), existing, seed=seed + 2
        )
        test_neg = sample_negative_edges(
            num_nodes, test_pos.size(1), existing, seed=seed + 3
        )

    elif neg_mode == "hard_region":
        train_neg = sample_negative_edges_hard_region(
            data, train_pos.size(1), existing, seed=seed + 1
        )
        val_neg = sample_negative_edges_hard_region(
            data, val_pos.size(1), existing, seed=seed + 2
        )
        test_neg = sample_negative_edges_hard_region(
            data, test_pos.size(1), existing, seed=seed + 3
        )
    else:
        raise ValueError(f"Unknown neg_mode: {neg_mode}")

    return (train_pos, val_pos, test_pos), (train_neg, val_neg, test_neg)

    


def compute_link_metrics(logits, labels):
    """
    logits: [E]
    labels: [E] (0/1)
    """
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    try:
        auc = roc_auc_score(labels_np, probs)
    except ValueError:
        # e.g., edge_type contains only 0s or only 1s
        auc = float("nan")
    try:
        ap = average_precision_score(labels_np, probs)
    except ValueError:
        ap = float("nan")
    return {"auc": float(auc), "ap": float(ap)}


def train_one_epoch_linkpred(
    encoder,
    decoder,
    data,
    pos_edges,
    neg_edges,
    optimizer,
    device,
):
    encoder.train()
    decoder.train()

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_type = data.edge_type.to(device)

    h = encoder(x, edge_index, edge_type)

    # Positive predictions
    pos_logits = decoder(h, pos_edges.to(device))
    pos_labels = torch.ones(pos_logits.size(0), device=device)

    # Negative predictions
    neg_logits = decoder(h, neg_edges.to(device))
    neg_labels = torch.zeros(neg_logits.size(0), device=device)

    logits = torch.cat([pos_logits, neg_logits], dim=0)
    labels = torch.cat([pos_labels, neg_labels], dim=0)

    loss = F.binary_cross_entropy_with_logits(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        metrics = compute_link_metrics(logits, labels)
        metrics["loss"] = float(loss.item())
    return loss.item(), metrics


@torch.no_grad()
def eval_linkpred(
    encoder,
    decoder,
    data,
    pos_edges,
    neg_edges,
    device,
):
    encoder.eval()
    decoder.eval()

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_type = data.edge_type.to(device)

    h = encoder(x, edge_index, edge_type)

    pos_logits = decoder(h, pos_edges.to(device))
    pos_labels = torch.ones(pos_logits.size(0), device=device)

    neg_logits = decoder(h, neg_edges.to(device))
    neg_labels = torch.zeros(neg_logits.size(0), device=device)

    logits = torch.cat([pos_logits, neg_logits], dim=0)
    labels = torch.cat([pos_labels, neg_labels], dim=0)

    metrics = compute_link_metrics(logits, labels)
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="R-GCN-based link prediction experiment script for a specific layer"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to PyG Data .pt file",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="finance",
        help="Layer name for link prediction (e.g., finance, communication)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience based on val AUC",
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=1e-3,
        help="Minimum improvement in val AUC to reset patience",
    )
    parser.add_argument(
        "--neg_mode",
        type=str,
        default="uniform",
        choices=["uniform", "hard_region"],
        help="Negative sampling mode (uniform or hard_region)",
    )


    parser.add_argument(
        "--edge_attr_transform",
        type=str,
        default="auto",
        choices=["auto", "none", "log1p"],
        help="Transform for edge_attr aggregation features. Use 'auto' to log1p only if heavy-tailed (recommended).",
    )
    parser.add_argument("--edge_attr_auto_q", type=float, default=0.99)
    parser.add_argument("--edge_attr_auto_thresh", type=float, default=20.0)
    parser.add_argument(
        "--layer_directed",
        type=int,
        default=-1,
        help="Override directedness for the selected layer: 1=directed, 0=undirected, -1=auto(from generator_config).",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    # -----------------------------
    # Load data
    # -----------------------------
    print(f"[*] Loading PyG Data from: {args.data_path}")
    data = load_pyg_data(args.data_path, map_location="cpu")

    # Resolve a fixed edge_attr transform once for consistency (v3 datasets may already be log-scaled)
    edge_attr_transform = "none"
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        ea_np = data.edge_attr.view(-1).cpu().numpy()
        edge_attr_transform = resolve_edge_attr_transform_np(
            ea_np, mode=args.edge_attr_transform, q=args.edge_attr_auto_q, thresh=args.edge_attr_auto_thresh
        )
    print(f"[*] edge_attr_transform resolved: {edge_attr_transform}")

    # -----------------------------
    # Optional feature augmentation
    # -----------------------------
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        data = augment_with_edge_attr(
            data,
            edge_attr_transform=edge_attr_transform
        )

    # -----------------------------
    # Always define model dimensions
    # -----------------------------
    num_nodes = data.x.size(0)
    num_features = data.x.size(1)
    num_relations = len(data.layer_type_mapping)

    # -----------------------------
    # Optional directedness override
    # -----------------------------
    if args.layer_directed >= 0:
        data._layer_directed_override = {
            args.layer: (args.layer_directed == 1)
        }

    # -----------------------------
    # Build edge sets per layer
    # -----------------------------
    (train_pos, val_pos, test_pos), (train_neg, val_neg, test_neg) = build_edge_sets_for_layer(
        data,
        layer_name=args.layer,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=args.seed,
        neg_mode=args.neg_mode,
    )


    # -----------------------------
    # Prepare model
    # -----------------------------
    encoder = RGCNEncoder(
        in_channels=num_features,
        hidden_channels=args.hidden_dim,
        num_relations=num_relations,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    #decoder = DotProductLinkPredictor().to(device)
    decoder = MLPLinkPredictor(hidden_dim=args.hidden_dim).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_auc = -1.0
    best_test_metrics = None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        loss, train_metrics = train_one_epoch_linkpred(
            encoder,
            decoder,
            data,
            train_pos,
            train_neg,
            optimizer,
            device,
        )

        val_metrics = eval_linkpred(encoder, decoder, data, val_pos, val_neg, device)
        test_metrics = eval_linkpred(encoder, decoder, data, test_pos, test_neg, device)

        val_auc = val_metrics["auc"]

        if val_auc > best_val_auc + args.min_delta:
            best_val_auc = val_auc
            best_test_metrics = test_metrics
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"[Epoch {epoch:03d}] "
                f"loss={loss:.4f} | "
                f"train_auc={train_metrics['auc']:.3f}, train_ap={train_metrics['ap']:.3f} | "
                f"val_auc={val_metrics['auc']:.3f}, val_ap={val_metrics['ap']:.3f} | "
                f"test_auc={test_metrics['auc']:.3f}, test_ap={test_metrics['ap']:.3f}"
            )

        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(
                f"[*] Early stopping at epoch {epoch} "
                f"(no val AUC improvement for {args.patience} epochs)."
            )
            break

    print("\n[*] Training finished.")
    print(f"Best val AUC: {best_val_auc:.3f}")
    if best_test_metrics is not None:
        print(
            f"[*] Corresponding test metrics at best val AUC: "
            f"test_auc={best_test_metrics['auc']:.3f}, test_ap={best_test_metrics['ap']:.3f}"
        )


    out_dir = os.path.dirname(args.data_path)
    metrics_path = os.path.join(
        out_dir, f"linkpred_{args.layer}_{args.neg_mode}.json"
    )

    link_result = {
        "task": "linkpred",
        "layer": args.layer,
        "neg_mode": args.neg_mode,
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
        "best_val_auc": float(best_val_auc),
        "test_at_best_val_auc": {
            "auc": float(best_test_metrics["auc"])
            if best_test_metrics is not None
            else None,
            "ap": float(best_test_metrics["ap"])
            if best_test_metrics is not None
            else None,
        },
    }

    with open(metrics_path, "w") as f:
        json.dump(link_result, f, indent=2)

    print(f"[*] Saved link prediction metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
