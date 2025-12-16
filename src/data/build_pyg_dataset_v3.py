"""
build_pyg_dataset_v3.py

Convert synthetic multiplex terror network manifests (v1/v2/v3) into a PyTorch Geometric `Data`.

Key upgrades vs v2:
  - Uses manifest layer "directed" flag to symmetrize undirected layers (comm/operation/ideology) by default.
  - Builds richer edge_attr from aggregated event statistics when available (or infers from events if missing).
  - Adds ideology as a continuous node feature (if present).

v3+ compatibility:
  - Preserves edge observation flags as tensors when present on edges:
      * edge_is_false  (a.k.a. is_false)
      * edge_is_copied (a.k.a. is_copied / copied_from*)
    These are stored on the returned Data object and can optionally be consumed by
    downstream training scripts as edge features (rather than filtering).

Outputs (single homogeneous graph):
  - x           : [N, F] node features (region one-hot + group one-hot + continuous)
  - edge_index  : [2, E] edges (concatenated from all layers)
  - edge_type   : [E] relation type index (0=hier, 1=finance, 2=comm, 3=ops, 4=ideo)
  - edge_attr   : [E, 1] scalar edge weight per edge (log-scaled for heavy-tailed layers)
  - edge_is_false  : [E] 0/1 observation-noise flag (if available; else zeros)
  - edge_is_copied : [E] 0/1 cross-layer copy flag (if available; else zeros)
  - y_role      : [N] role class index
  - y_hvt       : [N] high_value_target (0/1)
  - train_mask / val_mask / test_mask : [N] boolean masks (node-level split)

Metadata attached to `Data`:
  node_id, role_mapping, region_mapping, group_mapping, layer_type_mapping,
  importance_score, imp_mean, imp_std, generator_meta, generator_config
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


# -----------------------------
# Utility
# -----------------------------


def validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float, tol: float = 1e-6) -> None:
    ratios = [train_ratio, val_ratio, test_ratio]
    if any(r < 0 for r in ratios):
        raise ValueError("train/val/test ratios must be non-negative.")
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > tol:
        raise ValueError("train/val/test ratios must sum to 1.0.")


def encode_categorical(series: pd.Series) -> Tuple[np.ndarray, Dict[str, int]]:
    uniques = sorted(series.unique().tolist())
    mapping = {v: i for i, v in enumerate(uniques)}
    idx = series.map(mapping).astype(int).to_numpy()
    return idx, mapping


def one_hot(indices: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((len(indices), num_classes), dtype=np.float32)
    out[np.arange(len(indices)), indices] = 1.0
    return out


def _default_split_seed_from_manifest(mani: dict) -> int:
    meta = mani.get("meta", {}) or {}
    if isinstance(meta, dict):
        if meta.get("seed") is not None:
            try:
                return int(meta["seed"])
            except Exception:
                pass
        cfg = meta.get("config", {}) or {}
        if isinstance(cfg, dict) and cfg.get("seed") is not None:
            try:
                return int(cfg["seed"])
            except Exception:
                pass
    return 42


# -----------------------------
# Load multiplex manifest (v1/v2/v3)
# -----------------------------


def load_multiplex(manifest_path: str):
    """
    Supports:
      - v1: CSV-pointer manifest
      - v2: inline nodes + layers with {directed, edges}
      - v3: same as v2, but edges may include aggregated statistics and flags
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        mani = json.load(f)

    # nodes
    nodes_raw = mani.get("nodes")
    if isinstance(nodes_raw, str):
        # v1
        nodes = pd.read_csv(nodes_raw)
        labels_path = mani.get("labels")
        if labels_path is None:
            raise ValueError("v1 format requires 'labels' to provide a CSV path.")
        labels = pd.read_csv(labels_path)

    elif isinstance(nodes_raw, list):
        # v2/v3
        df_nodes = pd.DataFrame(nodes_raw)
        if "node_id" in df_nodes.columns:
            pass
        elif "id" in df_nodes.columns:
            df_nodes = df_nodes.rename(columns={"id": "node_id"})
        else:
            raise ValueError("'nodes' must include an 'id' or 'node_id' column.")

        # required categorical columns
        for col in ["node_id", "role", "region", "group"]:
            if col not in df_nodes.columns:
                raise ValueError(f"Required column '{col}' missing in nodes.")

        # optional continuous columns (fill defaults)
        for col, default in [
            ("skill_level", 0.0),
            ("radicalization", 0.0),
            ("past_incidents", 0.0),
            ("ideology", 0.0),
            ("activity_rate", 1.0),
            ("observability", 1.0),
            ("importance_score", 0.0),
            ("high_value_target", 0),
        ]:
            if col not in df_nodes.columns:
                df_nodes[col] = default

        # mimic v1 split
        nodes = df_nodes[["node_id", "role", "region", "group"]].copy()
        labels = df_nodes[[
            "node_id",
            "role",
            "region",
            "group",
            "skill_level",
            "radicalization",
            "past_incidents",
            "ideology",
            "activity_rate",
            "observability",
            "importance_score",
            "high_value_target",
        ]].copy()

    else:
        raise ValueError(f"Unrecognized type for 'nodes' field: {type(nodes_raw)}")

    # layers
    layers: Dict[str, pd.DataFrame] = {}
    raw_layers = mani.get("layers", {}) or {}

    for layer_name, layer_obj in raw_layers.items():
        if isinstance(layer_obj, str):
            layers[layer_name] = pd.read_csv(layer_obj)
        elif isinstance(layer_obj, dict) and "edges" in layer_obj:
            df_layer = pd.DataFrame(layer_obj["edges"])
            if df_layer.empty:
                layers[layer_name] = df_layer
                continue
            if "source" not in df_layer.columns or "target" not in df_layer.columns:
                raise ValueError(f"Layer '{layer_name}' requires 'source' and 'target' columns.")
            layers[layer_name] = df_layer
        else:
            raise ValueError(f"Unrecognized layer format for '{layer_name}': {type(layer_obj)}")

    return mani, nodes, labels, layers


# -----------------------------
# Edge-attribute helpers
# -----------------------------


def _aggregate_from_events(events: List[Dict[str, Any]]) -> Tuple[
    Dict[Tuple[int, int], Dict[str, float]],
    Dict[Tuple[int, int], Dict[str, float]],
    Dict[Tuple[int, int], Dict[str, float]],
]:
    """Return (txn_stats, comm_stats, op_stats)."""
    txn: Dict[Tuple[int, int], Dict[str, float]] = {}
    comm: Dict[Tuple[int, int], Dict[str, float]] = {}
    op: Dict[Tuple[int, int], Dict[str, float]] = {}

    for ev in events:
        et = ev.get("event_type")
        u = int(ev.get("u"))
        v = int(ev.get("v"))
        meta = ev.get("meta", {}) or {}

        if et == "txn":
            k = (u, v)  # directed
            d = txn.setdefault(k, {"txn_count": 0.0, "txn_amount_sum": 0.0})
            amt = float(meta.get("amount", 0.0)) if isinstance(meta, dict) else 0.0
            d["txn_count"] += 1.0
            d["txn_amount_sum"] += amt
        elif et == "comm":
            a, b = (u, v) if u < v else (v, u)
            k = (a, b)
            d = comm.setdefault(k, {"comm_count": 0.0, "comm_duration_sum": 0.0})
            dur = float(meta.get("duration", 0.0)) if isinstance(meta, dict) else 0.0
            d["comm_count"] += 1.0
            d["comm_duration_sum"] += dur
        elif et == "op":
            a, b = (u, v) if u < v else (v, u)
            k = (a, b)
            d = op.setdefault(k, {"op_count": 0.0})
            d["op_count"] += 1.0

    return txn, comm, op


def _safe_log1p(x: float) -> float:
    try:
        return float(np.log1p(max(0.0, float(x))))
    except Exception:
        return 0.0


def _layer_directed_flag(mani: dict, layer_name: str) -> bool:
    """Best-effort directedness: prefer manifest, fallback to conventional defaults."""
    layer_obj = (mani.get("layers", {}) or {}).get(layer_name)
    if isinstance(layer_obj, dict) and "directed" in layer_obj:
        return bool(layer_obj.get("directed"))
    return layer_name in {"hierarchy", "finance"}


def _get_int_flag(row: Optional[pd.Series], keys: List[str], default: int = 0) -> int:
    """Extract a 0/1 int flag from a pandas row, supporting multiple possible key names."""
    if row is None:
        return int(default)
    for k in keys:
        if k in row.index:
            v = row.get(k)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            try:
                return int(v)
            except Exception:
                # handle strings like "true"/"false"
                s = str(v).strip().lower()
                if s in {"1", "true", "t", "yes", "y"}:
                    return 1
                if s in {"0", "false", "f", "no", "n"}:
                    return 0
    return int(default)


def _infer_copied_flag(row: Optional[pd.Series]) -> int:
    """Heuristic for copy flags: looks for is_copied/edge_is_copied or any copied_from* field."""
    if row is None:
        return 0
    # direct flags
    v = _get_int_flag(row, ["edge_is_copied", "is_copied"], default=0)
    if v in (0, 1):
        return int(v)
    # copied provenance
    for k in row.index:
        lk = str(k).lower()
        if lk.startswith("copied_from") or lk.endswith("copied_from") or "copied_from" in lk:
            val = row.get(k)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            # if provenance exists, treat as copied
            return 1
    return 0


# -----------------------------
# Main conversion
# -----------------------------


def build_pyg_data(
    manifest_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    split_seed: Optional[int] = None,
    symmetrize_undirected: bool = True,
) -> Data:
    validate_split_ratios(train_ratio, val_ratio, test_ratio)

    mani, nodes, labels, layers = load_multiplex(manifest_path)

    # node index mapping
    node_ids = nodes["node_id"].tolist()
    num_nodes = len(node_ids)
    node2idx = {nid: i for i, nid in enumerate(node_ids)}

    # consolidate
    df = nodes.merge(labels, on=["node_id", "role", "region", "group"], how="left")

    # 1) node features
    role_idx, role_map = encode_categorical(df["role"])
    region_idx, region_map = encode_categorical(df["region"])
    group_idx, group_map = encode_categorical(df["group"])

    region_oh = one_hot(region_idx, len(region_map))
    group_oh = one_hot(group_idx, len(group_map))

    cont_cols = ["skill_level", "radicalization", "past_incidents", "ideology"]
    for col in cont_cols:
        if col not in df.columns:
            df[col] = 0.0

    cont = df[cont_cols].to_numpy(dtype=np.float32)
    cont_mean = cont.mean(axis=0, keepdims=True)
    cont_std = cont.std(axis=0, keepdims=True) + 1e-8
    cont = (cont - cont_mean) / cont_std

    x_np = np.concatenate([region_oh, group_oh, cont], axis=1)
    x = torch.from_numpy(x_np)

    # 2) labels
    y_role = torch.from_numpy(role_idx.astype(np.int64))
    y_hvt = torch.from_numpy(df["high_value_target"].astype(np.int64).to_numpy())

    # 3) edge aggregation (prefer edge columns, fallback to events)
    events_raw = mani.get("events") or []
    txn_stats, comm_stats, op_stats = ({}, {}, {})
    if isinstance(events_raw, list) and events_raw:
        txn_stats, comm_stats, op_stats = _aggregate_from_events(events_raw)

    layer_type_map = {
        "hierarchy": 0,
        "finance": 1,
        "communication": 2,
        "operation": 3,
        "ideology": 4,
    }

    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_type_list: List[int] = []
    edge_attr_vals: List[float] = []
    edge_false_list: List[int] = []
    edge_copied_list: List[int] = []

    def _edge_weight(layer_name: str, u: int, v: int, row: Optional[pd.Series], directed: bool) -> float:
        """Return a single scalar edge weight."""
        if layer_name == "hierarchy":
            return 1.0

        if layer_name == "finance":
            # prefer aggregated sums if present
            if row is not None:
                for key in ["txn_amount_sum", "amount_sum", "amount"]:
                    if key in row and row.get(key) is not None and not (isinstance(row.get(key), float) and np.isnan(row.get(key))):
                        return _safe_log1p(float(row.get(key)))
            st = txn_stats.get((u, v))
            if st:
                return _safe_log1p(float(st.get("txn_amount_sum", 0.0)))
            return 0.0

        if layer_name == "communication":
            if row is not None:
                for key in ["comm_count", "num_events"]:
                    if key in row and row.get(key) is not None and not (isinstance(row.get(key), float) and np.isnan(row.get(key))):
                        return _safe_log1p(float(row.get(key)))
                if "comm_duration_sum" in row and row.get("comm_duration_sum") is not None and not (isinstance(row.get("comm_duration_sum"), float) and np.isnan(row.get("comm_duration_sum"))):
                    return _safe_log1p(float(row.get("comm_duration_sum")))
            a, b = (u, v) if u < v else (v, u)
            st = comm_stats.get((a, b))
            if st:
                # count is often more stable than duration
                return _safe_log1p(float(st.get("comm_count", 0.0)))
            return 0.0

        if layer_name == "operation":
            if row is not None:
                for key in ["op_count", "joint_ops"]:
                    if key in row and row.get(key) is not None and not (isinstance(row.get(key), float) and np.isnan(row.get(key))):
                        return _safe_log1p(float(row.get(key)))
            a, b = (u, v) if u < v else (v, u)
            st = op_stats.get((a, b))
            if st:
                return _safe_log1p(float(st.get("op_count", 0.0)))
            return 0.0

        if layer_name == "ideology":
            if row is not None and "similarity" in row and row.get("similarity") is not None and not (isinstance(row.get("similarity"), float) and np.isnan(row.get("similarity"))):
                try:
                    return float(row.get("similarity"))
                except Exception:
                    return 0.0
            return 0.0

        return 0.0

    def _add_edge(u_id: int, v_id: int, ltype: int, w: float, is_false: int, is_copied: int):
        if u_id not in node2idx or v_id not in node2idx:
            return
        ui = node2idx[u_id]
        vi = node2idx[v_id]
        edge_src.append(ui)
        edge_dst.append(vi)
        edge_type_list.append(ltype)
        edge_attr_vals.append(float(w))
        edge_false_list.append(int(is_false))
        edge_copied_list.append(int(is_copied))

    # deterministic layer order
    layer_order = ["hierarchy", "finance", "communication", "operation", "ideology"]

    for lname in layer_order:
        if lname not in layers or lname not in layer_type_map:
            continue

        df_layer = layers[lname]
        if df_layer is None or df_layer.empty:
            continue

        directed = _layer_directed_flag(mani, lname)
        ltype = layer_type_map[lname]

        for _, row in df_layer.iterrows():
            u = int(row["source"])
            v = int(row["target"])
            if u == v:
                continue

            # normalize keys for undirected layers
            if not directed:
                a, b = (u, v) if u < v else (v, u)
                u, v = a, b

            w = _edge_weight(lname, u, v, row=row, directed=directed)

            # flags (robust to different key spellings)
            is_false = _get_int_flag(row, ["edge_is_false", "is_false"], default=0)
            is_copied = _infer_copied_flag(row)

            _add_edge(u, v, ltype, w, is_false=is_false, is_copied=is_copied)

            if symmetrize_undirected and not directed:
                _add_edge(v, u, ltype, w, is_false=is_false, is_copied=is_copied)

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_type = torch.tensor(edge_type_list, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_vals, dtype=torch.float32).view(-1, 1)
    edge_is_false = torch.tensor(edge_false_list, dtype=torch.float32)
    edge_is_copied = torch.tensor(edge_copied_list, dtype=torch.float32)

    # 4) deterministic node split
    if split_seed is None:
        split_seed = _default_split_seed_from_manifest(mani)

    gen = torch.Generator().manual_seed(int(split_seed))
    perm = torch.randperm(num_nodes, generator=gen)

    n_train = int(train_ratio * num_nodes)
    n_val = int(val_ratio * num_nodes)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # importance_score train stats
    imp_np = df["importance_score"].to_numpy(dtype=np.float32)
    train_idx_np = train_idx.cpu().numpy() if hasattr(train_idx, "cpu") else train_idx.numpy()
    imp_train = imp_np[train_idx_np]
    if imp_train.size > 0:
        imp_mean = float(imp_train.mean())
        imp_std = float(imp_train.std())
    else:
        imp_mean = float(imp_np.mean())
        imp_std = float(imp_np.std())

    # quick flag rates (for sanity)
    e = edge_is_false.numel() if hasattr(edge_is_false, 'numel') else len(edge_false_list)
    if e > 0:
        false_rate = float(edge_is_false.mean().item())
        copied_rate = float(edge_is_copied.mean().item())
    else:
        false_rate = 0.0
        copied_rate = 0.0

    print(f"[*] split_seed={split_seed} | importance_score train mean={imp_mean:.3f}, std={imp_std:.3f}")
    print(f"[*] edges={int(edge_index.size(1))} | edge_is_false mean={false_rate:.4f} | edge_is_copied mean={copied_rate:.4f}")

    # 5) assemble Data
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_attr=edge_attr,
        y_role=y_role,
        y_hvt=y_hvt,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    # attach edge flag tensors
    data.edge_is_false = edge_is_false
    data.edge_is_copied = edge_is_copied

    data.node_id = node_ids
    data.role_mapping = role_map
    data.region_mapping = region_map
    data.group_mapping = group_map
    data.layer_type_mapping = layer_type_map

    data.importance_score = torch.from_numpy(imp_np)
    data.imp_mean = float(imp_mean)
    data.imp_std = float(imp_std)

    data.generator_meta = mani.get("meta", {}) or {}
    meta = data.generator_meta if isinstance(data.generator_meta, dict) else {}
    data.generator_config = meta.get("config", {}) if isinstance(meta, dict) else {}

    return data


def main():
    parser = argparse.ArgumentParser(description="Convert multiplex synthetic dataset to a PyG Data object (v3).")
    parser.add_argument("--manifest", type=str, required=True, help="Path to multiplex.json")
    parser.add_argument("--out_path", type=str, required=True, help="Output .pt file path")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument(
        "--no_symmetrize_undirected",
        action="store_true",
        help="Disable adding reverse edges for undirected layers (comm/operation/ideology).",
    )

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    print("[*] Building PyG Data from:", args.manifest)
    data = build_pyg_data(
        manifest_path=args.manifest,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
        symmetrize_undirected=(not args.no_symmetrize_undirected),
    )

    torch.save(data, args.out_path)
    print("[*] Saved PyG Data to:", os.path.abspath(args.out_path))
    print("    #nodes   :", data.num_nodes)
    print("    #edges   :", data.edge_index.size(1))
    print("    x.shape  :", tuple(data.x.shape))
    print("    edge_attr.shape:", tuple(data.edge_attr.shape))
    print("    edge_is_false.shape:", tuple(data.edge_is_false.shape))
    print("    edge_is_copied.shape:", tuple(data.edge_is_copied.shape))


if __name__ == "__main__":
    main()
