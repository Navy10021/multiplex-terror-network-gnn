#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_linkpred_layer_v3.py

Layer-wise link prediction for multiplex PyG datasets produced by:
- multiplex_generator_v3.py
- build_pyg_dataset_v3.py

Key v3 considerations implemented here:
- PyTorch >=2.6 safe loading (torch.load(weights_only=False))
- Optional edge_attr aggregation into node features (leakage-safe: uses TRAIN graph only)
- Optional edge flag usage (edge_is_false / edge_is_copied):
    - aggregated into node features via --edge_attr_agg --include_edge_flags
    - (optional) expanded-relation encoding via --edge_flags_as_relations
- Leakage-safe encoder graph: removes VAL/TEST positives of the target layer from the message-passing graph.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv

# ---------------------------------------------------------------------
# Repro / utils
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_serializable(obj: Any) -> Any:
    """Convert common non-JSON types (torch/numpy) into JSON-serializable objects."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return float(obj.detach().cpu().item())
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    # fallback: stringify
    return str(obj)


# ---------------------------------------------------------------------
# Safe PyG loader (PyTorch >= 2.6 friendly)
# ---------------------------------------------------------------------

def load_pyg_data(path: str, map_location: str | torch.device = "cpu") -> Data:
    """
    PyTorch >= 2.6 defaults torch.load(weights_only=True), which cannot load PyG Data objects.

    This repository generates the dataset files itself, so we load with weights_only=False.
    Only do this for files you trust.
    """
    # Best-effort allowlisting for environments that still route through safe globals
    try:
        import torch.serialization as ts  # type: ignore

        allow = [Data]
        try:
            from torch_geometric.data.data import DataEdgeAttr  # type: ignore
            allow.append(DataEdgeAttr)
        except Exception:
            pass

        try:
            ts.add_safe_globals(allow)
        except Exception:
            pass
    except Exception:
        pass

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # older torch without weights_only kwarg
        return torch.load(path, map_location=map_location)


# ---------------------------------------------------------------------
# Edge-attr transform / aggregation
# ---------------------------------------------------------------------

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
        qq = float(np.quantile(ea, float(q)))
    except Exception:
        qq = float(ea.max())
    mx = float(ea.max())
    return "log1p" if (qq > float(thresh) or mx > float(thresh)) else "none"


def transform_edge_attr_torch(w: torch.Tensor, transform: str) -> torch.Tensor:
    w = w.clamp(min=0.0)
    if transform == "log1p":
        return torch.log1p(w)
    return w


def augment_with_edge_attr(
    data: Data,
    *,
    edge_attr_transform: str = "none",
    include_edge_flags: bool = False,
) -> Data:
    """
    Aggregate edge_attr per relation into node features.
    - Adds R columns: mean(edge_attr) per relation over incident edges.
    - Optionally also adds 2R columns: mean(edge_is_false), mean(edge_is_copied) per relation.

    IMPORTANT: Call this on the TRAIN graph to avoid leakage.
    """
    if not hasattr(data, "edge_attr") or data.edge_attr is None:
        return data
    if not hasattr(data, "edge_type") or data.edge_type is None:
        return data

    edge_index = data.edge_index
    edge_type = data.edge_type
    edge_attr = data.edge_attr
    if edge_attr.dim() == 2 and edge_attr.size(1) == 1:
        w = edge_attr[:, 0]
    else:
        # if multi-dim, reduce to first component
        w = edge_attr.view(edge_attr.size(0), -1)[:, 0]

    num_nodes = int(data.x.size(0))
    num_relations = int(edge_type.max().item()) + 1

    w = transform_edge_attr_torch(w.float(), edge_attr_transform)
    one_hot_rel = F.one_hot(edge_type, num_classes=num_relations).float()  # [E, R]
    weighted_rel = one_hot_rel * w.unsqueeze(-1)  # [E, R]

    # sum over incident edges (both endpoints)
    agg_sum = torch.zeros((num_nodes, num_relations), dtype=torch.float32)
    deg = torch.zeros((num_nodes, 1), dtype=torch.float32)

    src, dst = edge_index
    ones = torch.ones((edge_index.size(1), 1), dtype=torch.float32)

    agg_sum.index_add_(0, src, weighted_rel)
    agg_sum.index_add_(0, dst, weighted_rel)
    deg.index_add_(0, src, ones)
    deg.index_add_(0, dst, ones)
    deg = deg.clamp(min=1.0)

    mean_w = agg_sum / deg  # [N, R]

    feats: List[torch.Tensor] = [mean_w]

    if include_edge_flags:
        # default to zeros if not present
        is_false = getattr(data, "edge_is_false", None)
        is_copied = getattr(data, "edge_is_copied", None)
        if is_false is None:
            is_false = torch.zeros((edge_index.size(1),), dtype=torch.float32)
        else:
            is_false = is_false.float().view(-1)
        if is_copied is None:
            is_copied = torch.zeros((edge_index.size(1),), dtype=torch.float32)
        else:
            is_copied = is_copied.float().view(-1)

        for flag in (is_false, is_copied):
            weighted_flag = one_hot_rel * flag.unsqueeze(-1)
            flag_sum = torch.zeros((num_nodes, num_relations), dtype=torch.float32)
            flag_sum.index_add_(0, src, weighted_flag)
            flag_sum.index_add_(0, dst, weighted_flag)
            feats.append(flag_sum / deg)

    old_dim = int(data.x.size(1))
    data.x = torch.cat([data.x, *feats], dim=1)
    print(f"[*] augment_with_edge_attr: x_dim {old_dim} -> {int(data.x.size(1))}")
    return data


# ---------------------------------------------------------------------
# Edge helpers (unique, split, negatives, leakage-safe train graph)
# ---------------------------------------------------------------------

def is_directed_layer_default(layer_name: str) -> bool:
    # v3 generator: hierarchy/finance are directed (others are undirected)
    return layer_name in {"hierarchy", "finance"}


def unique_edges(edge_index: torch.Tensor, *, directed: bool) -> torch.Tensor:
    """Deduplicate edges; for undirected, canonicalize to (min, max) pairs."""
    src = edge_index[0].cpu().numpy().astype(np.int64)
    dst = edge_index[1].cpu().numpy().astype(np.int64)

    if directed:
        pairs = np.stack([src, dst], axis=1)
    else:
        a = np.minimum(src, dst)
        b = np.maximum(src, dst)
        pairs = np.stack([a, b], axis=1)

    pairs = pairs[pairs[:, 0] != pairs[:, 1]]  # remove self-loops
    pairs_u = np.unique(pairs, axis=0)
    return torch.tensor(pairs_u, dtype=torch.long).t().contiguous()


def unique_edges_with_optional_time(
    edge_index: torch.Tensor,
    *,
    directed: bool,
    edge_time: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Deduplicate edges with optional timestamp consolidation (earliest time per unique edge)."""
    src = edge_index[0].cpu().numpy().astype(np.int64)
    dst = edge_index[1].cpu().numpy().astype(np.int64)
    ts = None if edge_time is None else edge_time.view(-1).cpu().numpy().astype(np.float64)

    key_to_time: Dict[Tuple[int, int], float] = {}
    keys: Set[Tuple[int, int]] = set()
    for i, (u0, v0) in enumerate(zip(src, dst)):
        u = int(u0)
        v = int(v0)
        if u == v:
            continue
        key = (u, v) if directed else ((u, v) if u < v else (v, u))
        keys.add(key)
        if ts is not None:
            t = float(ts[i])
            prev = key_to_time.get(key)
            if prev is None or t < prev:
                key_to_time[key] = t

    pairs_u = np.array(sorted(keys), dtype=np.int64)
    if pairs_u.size == 0:
        pairs_u = np.zeros((0, 2), dtype=np.int64)

    edges = torch.tensor(pairs_u, dtype=torch.long).t().contiguous()
    if ts is None:
        return edges, None

    out_t = np.array([key_to_time.get((int(u), int(v)), 0.0) for u, v in pairs_u], dtype=np.float32)
    return edges, torch.tensor(out_t, dtype=torch.float32)


def split_edges(
    edge_index: torch.Tensor,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    *,
    split_mode: str = "random",
    edge_time: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_edges = int(edge_index.size(1))
    if num_edges <= 0:
        z = torch.zeros((2, 0), dtype=torch.long)
        return z, z, z

    n_train = int(num_edges * float(train_ratio))
    n_val = int(num_edges * float(val_ratio))

    mode = str(split_mode or "random").lower()
    if mode == "temporal":
        if edge_time is None or int(edge_time.numel()) != num_edges:
            raise ValueError("temporal split requires edge_time with one timestamp per edge")
        t = edge_time.view(-1).cpu().numpy().astype(np.float64)
        # stable order: time asc, then deterministic random jitter from seed for ties
        rng = np.random.default_rng(int(seed))
        jitter = rng.random(num_edges) * 1e-6
        order = np.argsort(t + jitter, kind="mergesort")
        perm = torch.tensor(order, dtype=torch.long)
    else:
        g = torch.Generator().manual_seed(int(seed))
        perm = torch.randperm(num_edges, generator=g)

    train = edge_index[:, perm[:n_train]]
    val = edge_index[:, perm[n_train:n_train + n_val]]
    test = edge_index[:, perm[n_train + n_val:]]
    return train, val, test


def edge_index_to_set(edge_index: torch.Tensor, *, directed: bool) -> Set[Tuple[int, int]]:
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    out: Set[Tuple[int, int]] = set()
    if directed:
        for u, v in zip(src, dst):
            if u != v:
                out.add((int(u), int(v)))
    else:
        for u, v in zip(src, dst):
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            out.add((int(a), int(b)))
    return out


def sample_negative_edges_uniform(
    num_nodes: int,
    num_samples: int,
    existing: Set[Tuple[int, int]],
    *,
    directed: bool,
    seed: int,
) -> torch.Tensor:
    rng = np.random.default_rng(int(seed))
    neg: List[Tuple[int, int]] = []
    while len(neg) < int(num_samples):
        u = int(rng.integers(num_nodes))
        v = int(rng.integers(num_nodes))
        if u == v:
            continue
        key = (u, v) if directed else (min(u, v), max(u, v))
        if key in existing:
            continue
        neg.append((u, v))
    return torch.tensor(neg, dtype=torch.long).t().contiguous()


def _infer_node_regions(data: Data) -> Optional[np.ndarray]:
    """
    Infer node region id per node from x if region one-hot exists.
    build_pyg_dataset_v3 uses region one-hot at the front, and stores data.region_mapping.
    """
    region_mapping = getattr(data, "region_mapping", None)
    if not isinstance(region_mapping, dict) or len(region_mapping) == 0:
        return None
    region_dim = int(len(region_mapping))
    if data.x.size(1) < region_dim:
        return None
    x0 = data.x[:, :region_dim]
    regions = torch.argmax(x0, dim=1).cpu().numpy().astype(np.int64)
    return regions


def sample_negative_edges_degree(
    num_nodes: int,
    num_samples: int,
    existing: Set[Tuple[int, int]],
    *,
    directed: bool,
    degree_weights: np.ndarray,
    seed: int,
) -> torch.Tensor:
    """Sample negatives with degree-biased node proposals (harder than uniform)."""
    rng = np.random.default_rng(int(seed))
    w = np.asarray(degree_weights, dtype=np.float64).reshape(-1)
    if w.size != int(num_nodes):
        raise ValueError(f"degree_weights size mismatch: expected {num_nodes}, got {w.size}")
    w = np.clip(w, 0.0, None)
    if float(w.sum()) <= 0.0:
        w = np.ones(int(num_nodes), dtype=np.float64)
    p = w / float(w.sum())

    neg: List[Tuple[int, int]] = []
    while len(neg) < int(num_samples):
        u = int(rng.choice(num_nodes, p=p))
        v = int(rng.choice(num_nodes, p=p))
        if u == v:
            continue
        key = (u, v) if directed else (min(u, v), max(u, v))
        if key in existing:
            continue
        neg.append((u, v))
    return torch.tensor(neg, dtype=torch.long).t().contiguous()


def sample_negative_edges_hard_region(
    num_nodes: int,
    num_samples: int,
    existing: Set[Tuple[int, int]],
    *,
    directed: bool,
    regions: np.ndarray,
    seed: int,
) -> torch.Tensor:
    """
    Sample negatives within the same region to create "harder" negatives.

    Requires regions array of shape [N] with integer region ids.
    """
    rng = np.random.default_rng(int(seed))
    neg: List[Tuple[int, int]] = []

    # build region -> nodes list
    region_ids = np.unique(regions)
    buckets: Dict[int, np.ndarray] = {int(r): np.where(regions == r)[0] for r in region_ids}

    region_list = list(buckets.keys())
    while len(neg) < int(num_samples):
        r = int(rng.choice(region_list))
        nodes = buckets[r]
        if nodes.size < 2:
            continue
        u = int(rng.choice(nodes))
        v = int(rng.choice(nodes))
        if u == v:
            continue
        key = (u, v) if directed else (min(u, v), max(u, v))
        if key in existing:
            continue
        neg.append((u, v))

    return torch.tensor(neg, dtype=torch.long).t().contiguous()


def build_train_graph_without_leakage(
    data: Data,
    *,
    target_rel: int,
    heldout_pos: torch.Tensor,
    directed: bool,
) -> Data:
    """
    Construct the message-passing graph for the encoder:
    - Keep all edges of non-target relations.
    - Keep only TRAIN positives of the target relation by removing held-out (val/test) positives.

    This avoids leakage where the encoder would "see" val/test edges during message passing.
    """
    edge_index = data.edge_index
    edge_type = data.edge_type
    E = int(edge_index.size(1))

    # heldout keys
    held = edge_index_to_set(heldout_pos, directed=directed)

    src_all = edge_index[0].tolist()
    dst_all = edge_index[1].tolist()
    et_all = edge_type.tolist()

    keep_mask = np.ones(E, dtype=bool)
    for i in range(E):
        if int(et_all[i]) != int(target_rel):
            continue
        u = int(src_all[i])
        v = int(dst_all[i])
        key = (u, v) if directed else (min(u, v), max(u, v))
        if key in held:
            keep_mask[i] = False

    keep_idx = torch.tensor(np.where(keep_mask)[0], dtype=torch.long)

    train_data = Data(
        x=data.x,
        edge_index=data.edge_index[:, keep_idx],
        edge_type=data.edge_type[keep_idx],
        edge_attr=(data.edge_attr[keep_idx] if getattr(data, "edge_attr", None) is not None else None),
    )

    # carry over optional edge flags, if present
    if hasattr(data, "edge_is_false"):
        train_data.edge_is_false = data.edge_is_false[keep_idx]
    if hasattr(data, "edge_is_copied"):
        train_data.edge_is_copied = data.edge_is_copied[keep_idx]

    # carry over meta fields commonly used elsewhere
    for attr in ("layer_type_mapping", "region_mapping", "role_mapping", "generator_meta", "generator_config", "generator_config_hash"):
        if hasattr(data, attr):
            setattr(train_data, attr, getattr(data, attr))

    return train_data


def assert_no_target_edge_leakage(
    train_graph: Data,
    *,
    target_rel: int,
    heldout_pos: torch.Tensor,
    directed: bool,
) -> None:
    """Assert target-layer heldout positives are removed from encoder graph."""
    mask_target = train_graph.edge_type == int(target_rel)
    train_target = train_graph.edge_index[:, mask_target]
    train_set = edge_index_to_set(train_target, directed=directed)
    held_set = edge_index_to_set(heldout_pos, directed=directed)
    leaked = sorted(list(train_set.intersection(held_set)))
    if leaked:
        raise AssertionError(
            "Leakage detected: heldout target-layer edges are present in encoder graph. "
            f"examples={leaked[:5]}"
        )


def assert_hard_region_negatives(neg_edge_index: torch.Tensor, regions: np.ndarray) -> None:
    """Assert each sampled negative edge stays within the same region bucket."""
    src = neg_edge_index[0].tolist()
    dst = neg_edge_index[1].tolist()
    bad = []
    for u, v in zip(src, dst):
        if int(regions[int(u)]) != int(regions[int(v)]):
            bad.append((int(u), int(v)))
            if len(bad) >= 5:
                break
    if bad:
        raise AssertionError(f"hard_region negatives violate region constraint: examples={bad}")


def expand_relations_with_edge_flags(data: Data) -> Tuple[Data, int]:
    """
    Optional: encode (is_false, is_copied) as part of relation type for message passing.

    new_rel = base_rel * 4 + (is_false*2 + is_copied)

    Returns:
        (data, num_relations)
    """
    if not hasattr(data, "edge_type"):
        raise RuntimeError("edge_type missing")
    base = data.edge_type.long().view(-1)

    is_false = getattr(data, "edge_is_false", None)
    is_copied = getattr(data, "edge_is_copied", None)
    if is_false is None:
        is_false = torch.zeros_like(base, dtype=torch.long)
    else:
        is_false = is_false.long().view(-1)
    if is_copied is None:
        is_copied = torch.zeros_like(base, dtype=torch.long)
    else:
        is_copied = is_copied.long().view(-1)

    flags = (is_false * 2 + is_copied).clamp(min=0, max=3)
    new_edge_type = base * 4 + flags
    data.edge_type = new_edge_type

    num_relations = int(new_edge_type.max().item()) + 1
    return data, num_relations


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class RGCNEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_relations: int,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        self.dropout = float(dropout)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations=num_relations))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations=num_relations))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index, edge_type)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class MLPLinkPredictor(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        z = torch.cat([h[src], h[dst]], dim=-1)
        return self.mlp(z).view(-1)


# ---------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------

@torch.no_grad()
def compute_metrics_from_logits(labels: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    # handle degenerate cases safely
    if labels.ndim != 1:
        labels = labels.reshape(-1)
    if logits.ndim != 1:
        logits = logits.reshape(-1)

    # If labels contain only one class, roc_auc_score will error.
    auc = None
    try:
        if np.unique(labels).size >= 2:
            auc = float(roc_auc_score(labels, logits))
    except Exception:
        auc = None

    ap = None
    try:
        ap = float(average_precision_score(labels, logits))
    except Exception:
        ap = None

    return {"auc": auc, "ap": ap}


def train_one_epoch(
    encoder: nn.Module,
    decoder: nn.Module,
    data: Data,
    pos_edges: torch.Tensor,
    neg_edges: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    encoder.train()
    decoder.train()

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_type = data.edge_type.to(device)

    h = encoder(x, edge_index, edge_type)

    pos_logits = decoder(h, pos_edges.to(device))
    neg_logits = decoder(h, neg_edges.to(device))

    logits = torch.cat([pos_logits, neg_logits], dim=0)
    labels = torch.cat(
        [torch.ones_like(pos_logits), torch.zeros_like(neg_logits)],
        dim=0,
    )

    loss = F.binary_cross_entropy_with_logits(logits, labels)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        y = labels.detach().cpu().numpy().astype(np.int32)
        s = logits.detach().cpu().numpy().astype(np.float32)
        metrics = compute_metrics_from_logits(y, s)

    return float(loss.item()), metrics


@torch.no_grad()
def eval_linkpred(
    encoder: nn.Module,
    decoder: nn.Module,
    data: Data,
    pos_edges: torch.Tensor,
    neg_edges: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    encoder.eval()
    decoder.eval()

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_type = data.edge_type.to(device)

    h = encoder(x, edge_index, edge_type)

    pos_logits = decoder(h, pos_edges.to(device))
    neg_logits = decoder(h, neg_edges.to(device))

    logits = torch.cat([pos_logits, neg_logits], dim=0).detach().cpu().numpy().astype(np.float32)
    labels = np.concatenate(
        [np.ones(pos_logits.numel(), dtype=np.int32), np.zeros(neg_logits.numel(), dtype=np.int32)],
        axis=0,
    )

    return compute_metrics_from_logits(labels, logits)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train layer-wise link prediction (R-GCN) on v3 PyG dataset.")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--layer", type=str, default="finance", choices=["hierarchy", "finance", "communication", "operation", "ideology"])

    # model / optim
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    # training
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--split_mode", type=str, default="random", choices=["random", "temporal"])
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--min_delta", type=float, default=1e-3)

    # negatives
    p.add_argument("--neg_mode", type=str, default="uniform", choices=["uniform", "hard_region", "degree", "hybrid"])
    p.add_argument("--neg_hybrid_hard_ratio", type=float, default=0.5, help="Ratio of hard_region negatives in hybrid mode [0,1].")

    # feature augmentation
    p.add_argument("--edge_attr_agg", action="store_true", help="Aggregate edge_attr into node features (leakage-safe).")
    p.add_argument("--include_edge_flags", action="store_true", help="Also aggregate edge_is_false/edge_is_copied into node features (requires --edge_attr_agg).")
    p.add_argument("--edge_flags_as_relations", action="store_true", help="Encode edge flags by expanding relation ids for message passing (optional).")

    p.add_argument("--edge_attr_transform", type=str, default="auto", choices=["auto", "none", "log1p"])
    p.add_argument("--edge_attr_auto_q", type=float, default=0.99)
    p.add_argument("--edge_attr_auto_thresh", type=float, default=20.0)

    # directedness override (useful if configs differ)
    p.add_argument("--layer_directed", type=int, default=-1, help="Override directedness for target layer: 1=directed, 0=undirected, -1=default.")

    # output
    p.add_argument("--out_dir", type=str, default=None)
    return p


def main() -> None:
    args = build_argparser().parse_args()

    # enforce flag dependency inside main
    if args.include_edge_flags and not args.edge_attr_agg:
        print("[!] --include_edge_flags implies --edge_attr_agg. Enabling --edge_attr_agg.")
        args.edge_attr_agg = True

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] device={device}")

    print(f"[*] Loading PyG Data from: {args.data_path}")
    data = load_pyg_data(args.data_path, map_location="cpu")

    if not hasattr(data, "layer_type_mapping") or not isinstance(data.layer_type_mapping, dict):
        raise RuntimeError("Data.layer_type_mapping missing. Did you build the dataset with build_pyg_dataset_v3.py?")
    rel_map: Dict[str, int] = {str(k): int(v) for k, v in data.layer_type_mapping.items()}
    if args.layer not in rel_map:
        raise RuntimeError(f"Layer '{args.layer}' not in layer_type_mapping keys={list(rel_map.keys())}")

    target_rel = int(rel_map[args.layer])
    directed = is_directed_layer_default(args.layer) if args.layer_directed < 0 else bool(args.layer_directed == 1)

    # -----------------------------------------------------------------
    # positives for the selected layer
    # -----------------------------------------------------------------
    mask_rel = (data.edge_type == target_rel)
    pos_raw = data.edge_index[:, mask_rel]
    pos_time_raw = None
    if args.split_mode == "temporal":
        for t_attr in ("edge_time", "edge_timestamp", "edge_ts"):
            tv = getattr(data, t_attr, None)
            if tv is not None:
                pos_time_raw = tv[mask_rel]
                break
    pos, pos_time = unique_edges_with_optional_time(pos_raw, directed=directed, edge_time=pos_time_raw)

    print(f"[Layer={args.layer}] directed={directed} | raw_pos={int(pos_raw.size(1))} | unique_pos={int(pos.size(1))}")
    if pos.size(1) < 10:
        raise RuntimeError("Too few positive edges for link prediction.")

    if args.split_mode == "temporal" and pos_time is None:
        print("[!] temporal split requested but edge timestamps not found; falling back to random split.")
        args.split_mode = "random"

    train_pos, val_pos, test_pos = split_edges(
        pos,
        args.train_ratio,
        args.val_ratio,
        args.seed,
        split_mode=args.split_mode,
        edge_time=pos_time,
    )

    # existing edges set (avoid sampling true edges as negatives)
    existing = edge_index_to_set(pos, directed=directed)

    num_nodes = int(data.x.size(0))
    degrees = np.bincount(pos[0].cpu().numpy(), minlength=num_nodes) + np.bincount(pos[1].cpu().numpy(), minlength=num_nodes)

    if args.neg_mode == "uniform":
        train_neg = sample_negative_edges_uniform(num_nodes, int(train_pos.size(1)), existing, directed=directed, seed=args.seed + 1)
        val_neg = sample_negative_edges_uniform(num_nodes, int(val_pos.size(1)), existing, directed=directed, seed=args.seed + 2)
        test_neg = sample_negative_edges_uniform(num_nodes, int(test_pos.size(1)), existing, directed=directed, seed=args.seed + 3)
    elif args.neg_mode == "degree":
        train_neg = sample_negative_edges_degree(num_nodes, int(train_pos.size(1)), existing, directed=directed, degree_weights=degrees, seed=args.seed + 1)
        val_neg = sample_negative_edges_degree(num_nodes, int(val_pos.size(1)), existing, directed=directed, degree_weights=degrees, seed=args.seed + 2)
        test_neg = sample_negative_edges_degree(num_nodes, int(test_pos.size(1)), existing, directed=directed, degree_weights=degrees, seed=args.seed + 3)
    elif args.neg_mode == "hard_region":
        regions = _infer_node_regions(data)
        if regions is None:
            print("[!] hard_region requested but region info not found; falling back to uniform negatives.")
            train_neg = sample_negative_edges_uniform(num_nodes, int(train_pos.size(1)), existing, directed=directed, seed=args.seed + 1)
            val_neg = sample_negative_edges_uniform(num_nodes, int(val_pos.size(1)), existing, directed=directed, seed=args.seed + 2)
            test_neg = sample_negative_edges_uniform(num_nodes, int(test_pos.size(1)), existing, directed=directed, seed=args.seed + 3)
        else:
            train_neg = sample_negative_edges_hard_region(num_nodes, int(train_pos.size(1)), existing, directed=directed, regions=regions, seed=args.seed + 1)
            val_neg = sample_negative_edges_hard_region(num_nodes, int(val_pos.size(1)), existing, directed=directed, regions=regions, seed=args.seed + 2)
            test_neg = sample_negative_edges_hard_region(num_nodes, int(test_pos.size(1)), existing, directed=directed, regions=regions, seed=args.seed + 3)
            assert_hard_region_negatives(train_neg, regions)
            assert_hard_region_negatives(val_neg, regions)
            assert_hard_region_negatives(test_neg, regions)
    else:
        regions = _infer_node_regions(data)
        hard_ratio = float(np.clip(args.neg_hybrid_hard_ratio, 0.0, 1.0))

        def _hybrid(num_s: int, seed_off: int) -> torch.Tensor:
            n_hard = int(round(int(num_s) * hard_ratio))
            n_deg = int(num_s) - n_hard
            chunks: List[torch.Tensor] = []
            if n_hard > 0 and regions is not None:
                chunks.append(sample_negative_edges_hard_region(num_nodes, n_hard, existing, directed=directed, regions=regions, seed=args.seed + seed_off))
            if n_deg > 0:
                chunks.append(sample_negative_edges_degree(num_nodes, n_deg, existing, directed=directed, degree_weights=degrees, seed=args.seed + seed_off + 1000))
            if not chunks:
                return torch.zeros((2, 0), dtype=torch.long)
            out = torch.cat(chunks, dim=1)
            if regions is not None and n_hard > 0:
                assert_hard_region_negatives(out[:, :n_hard], regions)
            return out

        train_neg = _hybrid(int(train_pos.size(1)), 1)
        val_neg = _hybrid(int(val_pos.size(1)), 2)
        test_neg = _hybrid(int(test_pos.size(1)), 3)

    # -----------------------------------------------------------------
    # Leakage-safe training graph (encoder sees only TRAIN positives for target layer)
    # -----------------------------------------------------------------
    heldout_pos = torch.cat([val_pos, test_pos], dim=1) if (val_pos.numel() and test_pos.numel()) else (val_pos if val_pos.numel() else test_pos)
    train_graph = build_train_graph_without_leakage(
        data,
        target_rel=target_rel,
        heldout_pos=heldout_pos,
        directed=directed,
    )
    assert_no_target_edge_leakage(
        train_graph,
        target_rel=target_rel,
        heldout_pos=heldout_pos,
        directed=directed,
    )
    print(f"[*] train_graph edges: {int(data.edge_index.size(1))} -> {int(train_graph.edge_index.size(1))}")

    # -----------------------------------------------------------------
    # Resolve edge_attr transform (once) + optional feature augmentation
    # -----------------------------------------------------------------
    edge_attr_transform = "none"
    if getattr(train_graph, "edge_attr", None) is not None:
        ea_np = train_graph.edge_attr.view(-1).cpu().numpy()
        edge_attr_transform = resolve_edge_attr_transform_np(
            ea_np,
            mode=args.edge_attr_transform,
            q=args.edge_attr_auto_q,
            thresh=args.edge_attr_auto_thresh,
        )
    print(f"[*] edge_attr_transform resolved: {edge_attr_transform}")

    if args.edge_attr_agg:
        train_graph = augment_with_edge_attr(
            train_graph,
            edge_attr_transform=edge_attr_transform,
            include_edge_flags=bool(args.include_edge_flags),
        )

    # -----------------------------------------------------------------
    # Optional: treat edge flags as relations for message passing
    # -----------------------------------------------------------------
    num_relations = int(train_graph.edge_type.max().item()) + 1
    if args.edge_flags_as_relations:
        train_graph, num_relations = expand_relations_with_edge_flags(train_graph)
        print(f"[*] edge_flags_as_relations enabled -> num_relations={num_relations}")

    # -----------------------------------------------------------------
    # Model / train loop
    # -----------------------------------------------------------------
    in_dim = int(train_graph.x.size(1))
    encoder = RGCNEncoder(
        in_channels=in_dim,
        hidden_channels=args.hidden_dim,
        num_relations=num_relations,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    decoder = MLPLinkPredictor(hidden_dim=args.hidden_dim).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_auc = -1.0
    best_epoch = 0
    best_test = {"auc": None, "ap": None}
    best_val = {"auc": None, "ap": None}
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        loss, train_metrics = train_one_epoch(
            encoder, decoder, train_graph, train_pos, train_neg, optimizer, device
        )
        val_metrics = eval_linkpred(encoder, decoder, train_graph, val_pos, val_neg, device)
        test_metrics = eval_linkpred(encoder, decoder, train_graph, test_pos, test_neg, device)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[Epoch {epoch:03d}] loss={loss:.4f} "
                f"train_auc={train_metrics['auc'] if train_metrics['auc'] is not None else None} | "
                f"val_auc={val_metrics['auc'] if val_metrics['auc'] is not None else None} "
                f"val_ap={val_metrics['ap'] if val_metrics['ap'] is not None else None} | "
                f"test_auc={test_metrics['auc'] if test_metrics['auc'] is not None else None}"
            )

        val_auc = val_metrics["auc"]
        improved = (val_auc is not None) and (val_auc > best_val_auc + float(args.min_delta))
        if improved:
            best_val_auc = float(val_auc)
            best_epoch = int(epoch)
            best_val = val_metrics
            best_test = test_metrics
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if args.patience > 0 and epochs_no_improve >= int(args.patience):
            print(f"[*] Early stop at epoch {epoch} (best val_auc={best_val_auc:.4f} at epoch {best_epoch})")
            break

    print("\n[*] Final (best-on-val) metrics:")
    print(json.dumps({"val": best_val, "test": best_test, "best_epoch": best_epoch, "best_val_auc": best_val_auc}, indent=2))

    # -----------------------------------------------------------------
    # Save metrics
    # -----------------------------------------------------------------
    out_dir = args.out_dir or os.path.dirname(args.data_path)
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, f"linkpred_{args.layer}_{args.neg_mode}_v3.json")

    result: Dict[str, Any] = {
        "task": "linkpred_layer_v3",
        "data_path": args.data_path,
        "layer": args.layer,
        "target_rel": int(target_rel),
        "directed": bool(directed),
        "neg_mode": args.neg_mode,
        "split_mode": args.split_mode,
        "seed": int(args.seed),
        "hyperparams": {
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "dropout": float(args.dropout),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "min_delta": float(args.min_delta),
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
        },
        "features": {
            "edge_attr_agg": bool(args.edge_attr_agg),
            "include_edge_flags": bool(args.include_edge_flags),
            "edge_flags_as_relations": bool(args.edge_flags_as_relations),
            "edge_attr_transform": str(edge_attr_transform),
        },
        "splits": {
            "num_pos_unique": int(pos.size(1)),
            "num_train_pos": int(train_pos.size(1)),
            "num_val_pos": int(val_pos.size(1)),
            "num_test_pos": int(test_pos.size(1)),
            "num_train_neg": int(train_neg.size(1)),
            "num_val_neg": int(val_neg.size(1)),
            "num_test_neg": int(test_neg.size(1)),
        },
        "best": {
            "epoch": int(best_epoch),
            "val": best_val,
            "test": best_test,
        },
    }

    # attach generator meta if available
    if hasattr(data, "generator_meta"):
        result["generator_meta"] = getattr(data, "generator_meta")
    if hasattr(data, "generator_config"):
        result["generator_config"] = getattr(data, "generator_config")
    if hasattr(data, "generator_config_hash"):
        result["generator_config_hash"] = getattr(data, "generator_config_hash")

    with open(metrics_path, "w") as f:
        json.dump(to_serializable(result), f, indent=2)

    print(f"[*] Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
